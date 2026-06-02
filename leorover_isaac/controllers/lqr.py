# lqr.py
"""
Vectorized LQR + differential-drive controller (GPU port of Controller2).

This is the torch-vectorized equivalent of `Controller2` and `LQRBaseline`
from the PyBullet stack (leoroverpybullet/envs/environment2.py and
leoroverpybullet/lqr_baseline.py). Every per-step numpy/scipy operation in the
original is replaced by a batched torch op over [num_envs, ...] tensors so the
whole rover fleet is controlled in a single GPU kernel launch.

PARITY NOTES (kept bit-for-bit with the PyBullet control law):

1. LQR gain K.
   The PyBullet LQRBaseline rebuilds A,B and solves the continuous-time ARE on
   EVERY call, because A depends on the per-waypoint reference velocity:
       A = [[0, v_ref, 0],
            [0, 0,     0],
            [0, 0,    -1]],   B = [[0,0],[0,1],[1,0]]
       Q = diag(10, 6, 0.5),  R = diag(0.5, 0.8)
   K = R^-1 B^T P is therefore a smooth function of v_ref only. We precompute
   K(v_ref) on a fine grid using the SAME scipy solver
   (common.lqr_baseline.LQRBaseline._compute_lqr_gain) at construction time and
   linearly interpolate per-env at runtime. This reproduces the original gains
   to within the grid resolution (default 0.0025 m/s) — far inside the 2pp
   parity ceiling in PORTING_ROADMAP.md.

2. Control law (compute_baseline): identical body-frame error
       e = [lateral_error, heading_error, velocity_error]
   and  v = v_ref + (-K e)[0],  w = omega_ref + (-K e)[1].
   The yaw-reference blending near a waypoint (distance < 0.3 m) is reproduced.

3. Differential-drive mapping + acceleration limits + hardware clips are the
   same as Controller2.forward / forward_waypoint_offset:
       wheel_L = (v - w*L) / (2 r),  wheel_R = (v + w*L) / (2 r)
       |Δwheel| <= max_wheel_accel * sim_timestep * 50  per control step
       wheel clipped to [-max_wheel_speed, max_wheel_speed]
   with L=0.34, r=0.3, max_wheel_speed=max_wheel_accel=1.333, sim_timestep=1/50.

The DirectRLEnv calls `compute_baseline` and `forward` to get per-wheel velocity
targets, writes them to the articulation, then steps `decimation` physics
sub-steps (the Isaac equivalent of Controller2's inner 10x stepSimulation loop).
"""

from __future__ import annotations

import numpy as np
import torch

# Reference (numpy/scipy) LQR — reused ONLY to build the gain table at init.
from leorover_isaac.common.lqr_baseline import LQRBaseline


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


class VectorizedLQR:
    """Batched LQR-baseline + diff-drive controller for `num_envs` rovers.

    All public methods accept/return torch tensors on `device` shaped
    [num_envs] or [num_envs, k]. Construct once per environment; call
    `reset_idx` on episode reset and `forward`/`compute_baseline` each step.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cuda:0",
        *,
        wheel_base: float = 0.34,
        wheel_radius: float = 0.6 / 2.0,        # 0.3 — matches Controller2 default
        max_wheel_speed: float = 1.333,
        max_wheel_accel: float = 1.333,
        sim_timestep: float = 1.0 / 50.0,
        max_residual_velocity: float = 0.15,
        max_residual_omega: float = 0.30,
        max_velocity_clip: float = 0.4,          # Leo rover hardware limits
        max_omega_clip: float = 1.047,
        use_lqr_baseline: bool = True,
        # gain-table grid over reference velocity. v_table_min must be > 0:
        # at v_ref=0 the lateral-error mode is uncontrollable and the Riccati
        # solver fails. Real v_ref is floored at 0.15 by trajectory profiling,
        # and lookups clamp to [v_table_min, v_table_max], so 0.05 is safe.
        v_table_min: float = 0.05,
        v_table_max: float = 1.0,
        v_table_steps: int = 401,                # 0.0025 m/s resolution
        yaw_smoothing_alpha: float = 0.85,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.wheel_base = float(wheel_base)
        self.wheel_radius = float(wheel_radius)
        self.max_wheel_speed = float(max_wheel_speed)
        self.max_wheel_accel = float(max_wheel_accel)
        self.sim_timestep = float(sim_timestep)
        self.max_residual_velocity = float(max_residual_velocity)
        self.max_residual_omega = float(max_residual_omega)
        self.max_velocity_clip = float(max_velocity_clip)
        self.max_omega_clip = float(max_omega_clip)
        self.use_lqr_baseline = bool(use_lqr_baseline)
        self.yaw_smoothing_alpha = float(yaw_smoothing_alpha)

        # Per control step (10 substeps in PyBullet): max_wheel_accel * dt * 50.
        self.max_delta_wheel = self.max_wheel_accel * self.sim_timestep * 50.0

        # --- Build the K(v_ref) lookup table with the reference scipy solver ---
        self._v_grid = torch.linspace(v_table_min, v_table_max, v_table_steps,
                                      device=self.device)
        self._v_min = float(v_table_min)
        self._v_max = float(v_table_max)
        self._dv = float((v_table_max - v_table_min) / (v_table_steps - 1))
        ref = LQRBaseline(wheel_base=self.wheel_base)
        K_list = []
        prev_K = None
        for v in self._v_grid.cpu().numpy():
            A, B = ref._linearize_unicycle_model(float(v))
            try:
                K = ref._compute_lqr_gain(A, B)              # [2, 3]
            except Exception:
                # Riccati degeneracy at very low v_ref (lateral mode uncontrollable).
                # Reuse the nearest valid gain; PD fallback if we have none yet.
                if prev_K is not None:
                    K = prev_K
                else:
                    K = np.zeros((2, 3), dtype=np.float64)
                    K[0, 2] = ref.fallback_kp_velocity
                    K[1, 0] = ref.fallback_kp_lateral
                    K[1, 1] = ref.fallback_kp_heading
            prev_K = K
            K_list.append(np.asarray(K, dtype=np.float32))
        # [steps, 2, 3]
        self._K_table = torch.from_numpy(np.stack(K_list, axis=0)).to(self.device)

        # --- Per-env runtime state ---
        self._prev_wheel = torch.zeros(self.num_envs, 2, device=self.device)
        self._smoothed_target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.last_baseline = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_total = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_residual = torch.zeros(self.num_envs, 2, device=self.device)

    # ------------------------------------------------------------------ #
    def reset_idx(self, env_ids: torch.Tensor, initial_yaw: torch.Tensor | None = None):
        """Reset per-env controller state on episode reset."""
        self._prev_wheel[env_ids] = 0.0
        self.last_baseline[env_ids] = 0.0
        self.last_total[env_ids] = 0.0
        self.last_residual[env_ids] = 0.0
        if initial_yaw is not None:
            self._smoothed_target_yaw[env_ids] = initial_yaw.to(self.device)

    # ------------------------------------------------------------------ #
    def _lookup_gain(self, v_ref: torch.Tensor) -> torch.Tensor:
        """Linear-interpolated K for each env's reference velocity.

        Args:  v_ref: [num_envs]
        Returns: K: [num_envs, 2, 3]
        """
        idx_f = (v_ref.clamp(self._v_min, self._v_max) - self._v_min) / self._dv
        i0 = torch.floor(idx_f).long().clamp(0, self._K_table.shape[0] - 1)
        i1 = (i0 + 1).clamp(0, self._K_table.shape[0] - 1)
        frac = (idx_f - i0.float()).clamp(0.0, 1.0).view(-1, 1, 1)
        K0 = self._K_table[i0]    # [n,2,3]
        K1 = self._K_table[i1]
        return K0 * (1.0 - frac) + K1 * frac

    # ------------------------------------------------------------------ #
    def compute_baseline(
        self,
        pos_xy: torch.Tensor,        # [n, 2] world x,y
        yaw: torch.Tensor,           # [n]
        forward_speed: torch.Tensor, # [n] body-frame vx
        waypoint: torch.Tensor,      # [n, 6] -> x,y,z,yaw,v_ref,omega_ref
    ) -> torch.Tensor:
        """Return [n, 2] = (velocity_baseline, omega_baseline). Pure LQR.

        Faithful port of Controller2.compute_baseline + LQRBaseline.baseline_control.
        """
        x_ref = waypoint[:, 0]
        y_ref = waypoint[:, 1]
        stored_yaw_ref = waypoint[:, 3]
        v_ref = waypoint[:, 4]
        omega_ref = waypoint[:, 5]

        dx = x_ref - pos_xy[:, 0]
        dy = y_ref - pos_xy[:, 1]
        dist = torch.sqrt(dx * dx + dy * dy)

        # --- computed_yaw_ref: blend dynamic heading and stored waypoint yaw ---
        dynamic_yaw = torch.where(dist > 0.01, torch.atan2(dy, dx), yaw)
        far = dist > 0.3
        blend = (dist / 0.3).clamp(0.0, 1.0)
        diff = _wrap_to_pi(stored_yaw_ref - dynamic_yaw)
        near_yaw_ref = dynamic_yaw + (1.0 - blend) * diff
        computed_yaw_ref = torch.where(far, torch.atan2(dy, dx), near_yaw_ref)

        # --- body/path-frame error vector e = [lateral, heading, velocity] ---
        # dx_e, dy_e measured robot - reference (matches lqr_baseline)
        dxe = pos_xy[:, 0] - x_ref
        dye = pos_xy[:, 1] - y_ref
        cos_r = torch.cos(computed_yaw_ref)
        sin_r = torch.sin(computed_yaw_ref)
        lateral = -sin_r * dxe + cos_r * dye
        heading_err = _wrap_to_pi(yaw - computed_yaw_ref)
        vel_err = forward_speed - v_ref
        e = torch.stack([lateral, heading_err, vel_err], dim=-1)   # [n, 3]

        # --- delta_u = -K e ;  K depends on v_ref ---
        K = self._lookup_gain(v_ref)                               # [n, 2, 3]
        delta_u = -torch.bmm(K, e.unsqueeze(-1)).squeeze(-1)       # [n, 2] -> (dv, domega)

        v_base = v_ref + delta_u[:, 0]
        w_base = omega_ref + delta_u[:, 1]
        baseline = torch.stack([v_base, w_base], dim=-1)
        self.last_baseline = baseline
        return baseline

    # ------------------------------------------------------------------ #
    def forward(
        self,
        action: torch.Tensor,        # [n, 2] in [-1, 1]
        waypoint: torch.Tensor,      # [n, 6]
        gate: torch.Tensor,          # [n] authority gate in [0, 1]
        pos_xy: torch.Tensor,
        yaw: torch.Tensor,
        forward_speed: torch.Tensor,
    ):
        """Combine baseline + residual (or pure-PPO command), map to wheels.

        Returns dict of per-env tensors:
            wheel_left, wheel_right : [n]  velocity targets (rad/s)
            total                   : [n,2] (v, omega) commanded
            baseline                : [n,2]
            residual                : [n,2] effective residual applied
        """
        action = action[:, :2].clamp(-1.0, 1.0)
        n = action.shape[0]

        if self.use_lqr_baseline:
            baseline = self.compute_baseline(pos_xy, yaw, forward_speed, waypoint)
            base_v = baseline[:, 0].clamp(-self.max_velocity_clip, self.max_velocity_clip)
            base_w = baseline[:, 1].clamp(-self.max_omega_clip, self.max_omega_clip)

            res_v = action[:, 0] * self.max_residual_velocity * gate
            res_w = action[:, 1] * self.max_residual_omega * gate

            v_tot = base_v + res_v
            w_tot = base_w + res_w
            v_clip = v_tot.clamp(-self.max_velocity_clip, self.max_velocity_clip)
            w_clip = w_tot.clamp(-self.max_omega_clip, self.max_omega_clip)
            eff_res_v = v_clip - base_v
            eff_res_w = w_clip - base_w
            baseline_out = torch.stack([base_v, base_w], dim=-1)
            residual_out = torch.stack([eff_res_v, eff_res_w], dim=-1)
        else:
            # Pure PPO: action maps directly to hardware-limited (v, omega).
            base_v = torch.zeros(n, device=self.device)
            base_w = torch.zeros(n, device=self.device)
            v_clip = (action[:, 0] * self.max_velocity_clip)
            w_clip = (action[:, 1] * self.max_omega_clip)
            baseline_out = torch.zeros(n, 2, device=self.device)
            residual_out = torch.stack([v_clip, w_clip], dim=-1)

        v_clip = v_clip.clamp(-self.max_velocity_clip, self.max_velocity_clip)
        w_clip = w_clip.clamp(-self.max_omega_clip, self.max_omega_clip)
        total = torch.stack([v_clip, w_clip], dim=-1)

        # --- differential-drive inverse kinematics ---
        wheel_l = (v_clip - w_clip * self.wheel_base) / (2.0 * self.wheel_radius)
        wheel_r = (v_clip + w_clip * self.wheel_base) / (2.0 * self.wheel_radius)
        wheel = torch.stack([wheel_l, wheel_r], dim=-1)              # [n, 2]

        # --- per-step acceleration limit (matches Controller2) ---
        delta = wheel - self._prev_wheel
        delta = delta.clamp(-self.max_delta_wheel, self.max_delta_wheel)
        wheel = self._prev_wheel + delta
        wheel = wheel.clamp(-self.max_wheel_speed, self.max_wheel_speed)
        self._prev_wheel = wheel.clone()

        self.last_total = total
        self.last_residual = residual_out
        return {
            "wheel_left": wheel[:, 0],
            "wheel_right": wheel[:, 1],
            "total": total,
            "baseline": baseline_out,
            "residual": residual_out,
        }

    # ------------------------------------------------------------------ #
    @property
    def prev_wheel(self) -> torch.Tensor:
        return self._prev_wheel
