# recorder.py
"""
EpisodeMetricsRecorder — writes episode_metrics.csv in the PyBullet schema.

This is the Isaac-Lab equivalent of the PyBullet `MetricsCallback` in
run_experiment.py. It accumulates per-env, per-episode statistics during a
vectorized rollout and appends one CSV row per finished episode, using the
IDENTICAL column order so that:

  * evaluate_training.py (the Tk + matplotlib GUI, carried over verbatim), and
  * analyze_training.py / analyze_latest.py

read Isaac Lab runs exactly as they read PyBullet runs — including the v33.9
residual-analysis plots (mean_residual_v_norm / mean_residual_w_norm columns).

Column schema (must match run_experiment.py MetricsCallback):
    episode,mean_cte,max_cte,total_reward,mean_reward_per_step,mean_slip,steps,
    success,terrain_intensity,friction_intensity,terrain_max_slope_deg,
    terrain_avg_slope_deg,mean_local_slope_deg,path_progress,roll_max,pitch_max,
    mean_residual_v_norm,mean_residual_w_norm

Usage (in the training loop / a wrapper, once per env.step):
    rec = EpisodeMetricsRecorder(log_dir, env)
    ...
    rec.record_step(reward_tensor, done_tensor)

Slip and terrain slope columns: PyBullet logged ~0 slip and per-episode terrain
slope stats. On the Isaac mesh terrain we expose terrain_intensity and the
rover's instantaneous local slope (from body-frame gravity); terrain_max/avg
slope are filled from the per-env terrain difficulty when available, else 0.
These columns are not used by the primary evaluate_training plots.
"""

from __future__ import annotations

import os
import math
import numpy as np
import torch

_HEADER = (
    "episode,mean_cte,max_cte,total_reward,mean_reward_per_step,mean_slip,steps,success,"
    "terrain_intensity,friction_intensity,"
    "terrain_max_slope_deg,terrain_avg_slope_deg,mean_local_slope_deg,"
    "path_progress,roll_max,pitch_max,"
    "mean_residual_v_norm,mean_residual_w_norm,"
    "max_slip,scenario_id,path_type\n"
)


class EpisodeMetricsRecorder:
    def __init__(self, log_dir: str, env, max_residual_v: float = None, max_residual_w: float = None,
                 filename: str = "episode_metrics.csv"):
        self.env = env
        self.n = env.num_envs
        self.device = env.device
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, filename)
        with open(self.csv_path, "w") as f:
            f.write(_HEADER)
        self.episode_count = 0

        import config as cfg_mod
        self._mv = float(max_residual_v if max_residual_v is not None else cfg_mod.MAX_RESIDUAL_VELOCITY)
        self._mw = float(max_residual_w if max_residual_w is not None else cfg_mod.MAX_RESIDUAL_OMEGA)

        self._reset_accum(slice(None))

    def _reset_accum(self, idx):
        z = lambda: torch.zeros(self.n, device=self.device)
        if not hasattr(self, "_cte_sum"):
            self._cte_sum = z(); self._cte_max = z(); self._rew_sum = z()
            self._steps = torch.zeros(self.n, device=self.device)
            self._resv_sum = z(); self._resw_sum = z()
            self._roll_max = z(); self._pitch_max = z()
            self._slope_sum = z(); self._slope_max = z()
            self._slip_sum = z(); self._slip_max = z()
        else:
            for buf in (self._cte_sum, self._cte_max, self._rew_sum, self._steps,
                        self._resv_sum, self._resw_sum, self._roll_max,
                        self._pitch_max, self._slope_sum, self._slope_max,
                        self._slip_sum, self._slip_max):
                buf[idx] = 0.0

    @torch.no_grad()
    def record_step(self, reward: torch.Tensor, done: torch.Tensor):
        env = self.env
        # Skip while the deterministic ADR eval is running, so those noise-free
        # eval episodes never land in the training episode_metrics.csv.
        if getattr(env, "_skip_record", False):
            return
        cte, _ = env._true_cte_and_along()
        cte = cte.abs()
        # residual norms (hybrid); for pure PPO the residual IS the command -> 0 contribution
        resv = (env._last_residual[:, 0].abs() / self._mv) if env.cfg.use_lqr_baseline else torch.zeros_like(cte)
        resw = (env._last_residual[:, 1].abs() / self._mw) if env.cfg.use_lqr_baseline else torch.zeros_like(cte)
        # roll/pitch from quaternion
        try:
            try:
                from isaaclab.utils.math import euler_xyz_from_quat
            except Exception:
                from omni.isaac.lab.utils.math import euler_xyz_from_quat
            roll, pitch, _ = euler_xyz_from_quat(env.robot.data.root_quat_w)
            # Isaac Lab returns Euler angles wrapped to [0, 2*pi): a level rover with roll
            # -0.02 rad comes back as 6.26, so max(|roll|) saturated at ~2*pi in every CSV
            # (the 6.28 roll_max/pitch_max columns). Re-wrap to [-pi, pi] before |.|.
            roll = torch.remainder(roll + math.pi, 2.0 * math.pi) - math.pi
            pitch = torch.remainder(pitch + math.pi, 2.0 * math.pi) - math.pi
        except Exception:
            roll = torch.zeros_like(cte); pitch = torch.zeros_like(cte)
        # local slope (deg) from body-frame gravity tilt: slope ~ atan(|g_xy|/|g_z|).
        # On a rigid mesh the chassis rests on the local ground plane, so this IS the
        # terrain slope under the rover (the paper describes the projected-gravity tilt
        # as "local terrain tilt under the chassis"). Fills the terrain_*_slope_deg cols.
        grav = env.robot.data.projected_gravity_b
        tilt = torch.atan2(torch.norm(grav[:, :2], dim=-1), grav[:, 2].abs().clamp(min=1e-6))
        slope_deg = torch.rad2deg(tilt)

        # longitudinal wheel slip: the fraction of COMMANDED forward speed not realized on
        # the ground. Velocity-controlled wheels turn at the commanded v, so (cmd-actual)/cmd
        # is the slip ratio. Gated to |cmd|>0.05 m/s (undefined near standstill) and clamped
        # to [0,1] (actual>cmd downhill is not slip). This is the paper's slip metric.
        try:
            v_cmd = env._last_total_cmd[:, 0]
            v_act = env.robot.data.root_lin_vel_b[:, 0]
            denom = v_cmd.abs().clamp(min=1e-6)
            slip = ((v_cmd - v_act) / denom).clamp(0.0, 1.0)
            slip = torch.where(v_cmd.abs() > 0.05, slip, torch.zeros_like(slip))
        except Exception:
            slip = torch.zeros_like(cte)

        # Reward + step count belong to the episode that just ENDED for done envs (the
        # terminal reward is paid on the done step), so accumulate them BEFORE flushing.
        self._rew_sum += reward
        self._steps += 1.0

        done_idx = torch.nonzero(done, as_tuple=False).flatten()
        if len(done_idx) > 0:
            self._flush(done_idx)
            self._reset_accum(done_idx)

        # State-based metrics on the done step describe the RESPAWNED robot (Isaac
        # auto-resets inside step, before this hook runs), so exclude just-reset envs:
        # the finished episode keeps only its real frames and the new episode starts
        # accumulating from the next call.
        live = (~done).float()
        self._cte_sum += cte * live
        self._cte_max = torch.maximum(self._cte_max, cte * live)
        self._resv_sum += resv * live
        self._resw_sum += resw * live
        self._roll_max = torch.maximum(self._roll_max, roll.abs() * live)
        self._pitch_max = torch.maximum(self._pitch_max, pitch.abs() * live)
        self._slope_sum += slope_deg * live
        self._slope_max = torch.maximum(self._slope_max, slope_deg * live)
        self._slip_sum += slip * live
        self._slip_max = torch.maximum(self._slip_max, slip * live)

    def _flush(self, idx):
        env = self.env
        # Use the terminal snapshots captured in _get_dones BEFORE Isaac's auto-reset.
        # Reading _path_progress()/_is_goal_reached() here would return the respawn
        # state (path_progress~3.8%, goal=False) for the just-finished episodes.
        progress = getattr(env, "_log_progress", None)
        if progress is None:
            progress = env._path_progress()
        _goal = getattr(env, "_log_goal", None)
        success = _goal.float() if _goal is not None else env._is_goal_reached().float()
        # Identity columns MUST come from the pre-reset snapshots (_get_dones): by flush
        # time _reset_idx has already redrawn terrain level / scenario / path for the NEXT
        # episode, and logging those decorrelated every per-level table (the flat-sweep
        # bug). Fall back to live values only if the env predates the snapshots.
        terr_int = getattr(env, "_log_terr_int", None)
        if terr_int is None:
            terr_int = getattr(env, "_terrain_intensity", torch.zeros(self.n, device=self.device))
        fric_int = getattr(env, "_log_fric_int", None)
        if fric_int is None:
            fric_int = getattr(env, "_friction_intensity", torch.zeros(self.n, device=self.device))

        idx_l = idx.tolist()
        rows = []
        steps = self._steps.clamp(min=1.0)
        # scenario id + path-type code: pre-reset snapshots for the same reason (the live
        # buffers already hold the NEXT episode's scenario -> a paired join would be
        # off-by-one). Default -1 / "random" when not in scenario/discrete mode.
        scen = getattr(env, "_log_scen_final", None)
        if scen is None:
            scen = getattr(env, "_log_scenario_id", None)
        ptype = getattr(env, "_log_ptype_final", None)
        if ptype is None:
            ptype = getattr(env, "_log_path_type", None)
        _PT = {0: "random", 1: "zigzag", 2: "curved", 3: "polygon"}
        for e in idx_l:
            self.episode_count += 1
            mean_cte = float(self._cte_sum[e] / steps[e])
            mean_rps = float(self._rew_sum[e] / steps[e])
            mean_slope = float(self._slope_sum[e] / steps[e])
            max_slope = float(self._slope_max[e])
            mean_slip = float(self._slip_sum[e] / steps[e])
            max_slip = float(self._slip_max[e])
            mean_rv = float(self._resv_sum[e] / steps[e])
            mean_rw = float(self._resw_sum[e] / steps[e])
            sid = int(scen[e]) if scen is not None else -1
            pt = _PT.get(int(ptype[e]) if ptype is not None else 0, "random")
            rows.append(
                f"{self.episode_count},{mean_cte:.4f},{float(self._cte_max[e]):.4f},"
                f"{float(self._rew_sum[e]):.2f},{mean_rps:.4f},{mean_slip:.4f},"
                f"{int(steps[e])},{int(success[e])},"
                f"{float(terr_int[e]):.1f},{float(fric_int[e]):.1f},"
                f"{max_slope:.2f},{mean_slope:.2f},{mean_slope:.2f},"
                f"{float(progress[e]):.1f},"
                f"{float(self._roll_max[e]):.4f},{float(self._pitch_max[e]):.4f},"
                f"{mean_rv:.4f},{mean_rw:.4f},"
                f"{max_slip:.4f},{sid},{pt}\n"
            )
        with open(self.csv_path, "a") as f:
            f.writelines(rows)
