# leo_rover_base_env.py
"""
LeoRoverBaseEnv — the shared, torch-vectorized DirectRLEnv that all three task
environments (flat / mars / mars-hybrid) inherit from.

This is the GPU port of the PyBullet `MyEnv2` step/reset loop. Every per-env
quantity is a [num_envs, ...] tensor on the sim device, so the whole rover fleet
advances in one kernel per physics substep instead of a Python `for env in envs`
loop. The semantics — observation layout, the v33.9 pure-PPO reward, the hybrid
residual + L2-effort term, every termination condition, waypoint skipping,
stagnation detection, the trajectory-profiled LQR baseline, the ADR terrain
curriculum, and the forward camera-lookahead slope features — are reproduced
faithfully from leoroverpybullet/envs/environment2.py.

WHAT MAPS TO WHAT (PyBullet MyEnv2 -> here):
  reset()                      -> _reset_idx(env_ids)         (per-env, batched)
  Controller2.forward()        -> _pre_physics_step + _apply_action (+ VectorizedLQR)
  10x p.stepSimulation()       -> decimation physics substeps
  _build_observation()         -> _get_observations()
  _compute_pure_ppo_reward()   -> _get_rewards()              (vectorized)
  _is_done()/_is_*()           -> _get_dones()                (vectorized)
  ADRCurriculum + reset terrain -> _adr + per-episode terrain-patch resampling
  _compute_terrain_lookahead() -> _compute_lookahead() via a forward RayCaster
  info dict                    -> per-env buffers the CSV Recorder reads

TERRAIN VARIETY (answers "same hills every episode?"): NO. Isaac Lab bakes a bank
of distinct terrain patches at startup (num_rows difficulty levels x num_cols
variations). On EVERY reset we reassign each env to a random patch drawn from
[0, current ADR ceiling] — so a rover sees a fresh hill layout each episode and
still revisits easy terrain (the original ADR's "sample from [0, current_max]"
design). The ADR ceiling rises as the rolling success rate clears the threshold.

Config knobs come from the repo-root `config.py` (the SAME file the PyBullet
stack uses) so reward weights / residual bounds / terrain ranges never diverge.
"""

from __future__ import annotations

import math
import numpy as np
import torch

# Repo-root config.py — single source of truth for all hyperparameters.
import config as cfg_mod

from leorover_isaac.common import path_templates, terrain_stats
from leorover_isaac.common.random_path_generator import generate_random_curved_path
from leorover_isaac.common.trajectory_profile import compute_trajectory_profile
from leorover_isaac.common.mars_terrain_numpy import friction_from_intensity
from leorover_isaac.controllers.lqr import VectorizedLQR, _wrap_to_pi

# ADR curriculum (engine-agnostic, carried over from adr_curriculum.py at root).
try:
    from adr_curriculum import ADRCurriculum, ADRConfig
    _HAS_ADR = True
except Exception:  # pragma: no cover
    _HAS_ADR = False

# Isaac Lab imports are guarded so the module can be imported for inspection /
# unit tests outside the Isaac python environment.
try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    import isaaclab.envs.mdp as mdp
    from isaaclab.utils import configclass
    from isaaclab.utils.math import euler_xyz_from_quat
    _ISAAC = True
except Exception:  # pragma: no cover
    _ISAAC = False
    def configclass(c):  # type: ignore
        return c
    class DirectRLEnv:  # type: ignore
        pass
    class DirectRLEnvCfg:  # type: ignore
        pass


MAX_WAYPOINTS = 80   # padding cap; longest template/random path is < 70 waypoints

# Friction range mapped from the config friction-intensity sweep (0.3 -> 2.0).
_FRIC_LO = friction_from_intensity(cfg_mod.TRAINING_FRICTION_MIN)
_FRIC_HI = friction_from_intensity(cfg_mod.TRAINING_FRICTION_MAX)


# ============================================================================ #
# CONFIG
# ============================================================================ #
if _ISAAC:
    @configclass
    class EventCfg:
        """Per-reset domain randomization. Randomizing the WHEEL contact material
        over the config friction sweep reproduces PyBullet's per-episode terrain
        friction randomization (contact friction = combine(wheel, terrain))."""
        wheel_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel.*"),
                "static_friction_range": (_FRIC_LO, _FRIC_HI),
                "dynamic_friction_range": (_FRIC_LO, _FRIC_HI),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )

    @configclass
    class LeoRoverBaseEnvCfg(DirectRLEnvCfg):
        # --- timing: 10 substeps @ 1/50 s = 0.2 s policy step (matches PyBullet) ---
        decimation: int = 10
        sim: SimulationCfg = SimulationCfg(
            dt=1.0 / 50.0,
            render_interval=10,
            gravity=(0.0, 0.0, -3.71),   # Mars gravity, as in environment2.reset()
        )
        # 2000 policy steps @ 0.2 s — matches the MyEnv2 effective agent-step cap.
        episode_length_s: float = 400.0

        # spaces (filled by subclasses; obs dim depends on lqr/camera flags)
        action_space: int = 2
        observation_space: int = 9
        state_space: int = 0

        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=4096, env_spacing=30.0, replicate_physics=True
        )

        # robot + terrain cfgs are set by subclasses in __post_init__
        robot = None
        terrain: TerrainImporterCfg = None

        # per-reset friction randomization
        events: EventCfg = EventCfg()

        # --- behaviour flags (subclasses set these) ---
        use_lqr_baseline: bool = False
        use_camera_lookahead: bool = False
        use_mars_terrain: bool = True
        use_adr: bool = True

        # --- path generation (mirror config.TRAINING_*) ---
        use_random_paths: bool = True
        min_curvature_angle: float = cfg_mod.TRAINING_MIN_CURVATURE_ANGLE
        max_curvature_angle: float = cfg_mod.TRAINING_MAX_CURVATURE_ANGLE
        total_path_distance: float = cfg_mod.TRAINING_TOTAL_PATH_DISTANCE
        num_random_paths: int = cfg_mod.TRAINING_NUM_RANDOM_PATHS

        # --- terrain intensity range (ADR ramps the ceiling at runtime) ---
        terrain_intensity_min: float = cfg_mod.TRAINING_TERRAIN_MIN
        terrain_intensity_max: float = cfg_mod.ADR_TERRAIN_MAX_LIMIT
        friction_intensity_min: float = cfg_mod.TRAINING_FRICTION_MIN
        friction_intensity_max: float = cfg_mod.TRAINING_FRICTION_MAX

        # --- camera lookahead (XVisio-style forward stereo, from config) ---
        camera_forward_offset: float = cfg_mod.CAMERA_FORWARD_OFFSET
        camera_height_offset: float = cfg_mod.CAMERA_HEIGHT_OFFSET
        camera_max_range: float = cfg_mod.CAMERA_MAX_RANGE
        camera_zone_near: float = cfg_mod.CAMERA_ZONE_NEAR
        camera_zone_mid: float = cfg_mod.CAMERA_ZONE_MID
else:
    class LeoRoverBaseEnvCfg:  # type: ignore
        pass


# ============================================================================ #
# ENV
# ============================================================================ #
class LeoRoverBaseEnv(DirectRLEnv):
    """Shared logic for all Leo Rover Isaac Lab tasks. See module docstring."""

    cfg: "LeoRoverBaseEnvCfg"

    # ---------------------------------------------------------------- init
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        n = self.num_envs

        # Reward / behaviour config from config.py (single source of truth).
        self._ppo = cfg_mod.get_pure_ppo_reward_config()
        self._res = cfg_mod.get_residual_reward_config()
        self._goal_tol = 0.2
        self._waypoint_tol = 0.2
        self._flip_threshold_gz = 0.5
        self._max_cte_term = float(self._ppo['ppo_max_cte_termination'])

        # --- waypoint buffers (env-local frame) ---
        self._wps = torch.zeros(n, MAX_WAYPOINTS, 6, device=dev)       # x,y,z,yaw,v_ref,omega_ref
        self._num_wp = torch.ones(n, dtype=torch.long, device=dev)
        self._cur_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self._prev_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self._cum_len = torch.zeros(n, MAX_WAYPOINTS, device=dev)      # cumulative arc length
        self._total_len = torch.ones(n, device=dev)
        self._goal_xy = torch.zeros(n, 2, device=dev)

        # --- runtime state ---
        self._actions = torch.zeros(n, 2, device=dev)
        self._prev_total_cmd = torch.zeros(n, 2, device=dev)
        self._last_total_cmd = torch.zeros(n, 2, device=dev)
        self._last_baseline = torch.zeros(n, 2, device=dev)
        self._last_residual = torch.zeros(n, 2, device=dev)
        self._last_lookahead = torch.full((n, 6), -1.0, device=dev)
        self._prev_progress = torch.zeros(n, device=dev)
        self._stagnation = torch.zeros(n, dtype=torch.long, device=dev)
        self._recovery_sustain = torch.zeros(n, dtype=torch.long, device=dev)
        self._sim_time = torch.zeros(n, device=dev)

        # Per-env terrain/friction intensity (for logging + ADR).
        self._terrain_intensity = torch.zeros(n, device=dev)
        self._friction_intensity = torch.full((n,), 0.5 * (cfg_mod.TRAINING_FRICTION_MIN + cfg_mod.TRAINING_FRICTION_MAX), device=dev)

        # --- ADR curriculum (global rolling-window ceiling) ---
        self._ep_cte_sum = torch.zeros(n, device=dev)
        self._ep_steps = torch.zeros(n, device=dev)
        self._ep_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self._adr = None
        if self.cfg.use_adr and _HAS_ADR:
            adr_cfg = ADRConfig(
                terrain_intensity_min=cfg_mod.TRAINING_TERRAIN_MIN,
                terrain_intensity_max_start=cfg_mod.ADR_TERRAIN_MAX_START,
                terrain_intensity_max_limit=cfg_mod.ADR_TERRAIN_MAX_LIMIT,
                friction_intensity_min=cfg_mod.TRAINING_FRICTION_MIN,
                friction_intensity_max_start=cfg_mod.TRAINING_FRICTION_MAX,
                friction_intensity_max_limit=100.0,
                success_rate_threshold=cfg_mod.ADR_SUCCESS_THRESHOLD,
                mean_cte_threshold=cfg_mod.ADR_CTE_THRESHOLD,
                regression_success_threshold=cfg_mod.ADR_REGRESSION_SUCCESS_THRESHOLD,
                regression_cte_threshold=cfg_mod.ADR_REGRESSION_CTE_THRESHOLD,
                eval_window_size=cfg_mod.ADR_EVAL_WINDOW,
                intensity_step_up=cfg_mod.ADR_STEP_UP,
                intensity_step_down=cfg_mod.ADR_STEP_DOWN,
                min_episodes_per_level=cfg_mod.ADR_MIN_EPISODES_PER_LEVEL,
                cooldown_episodes=cfg_mod.ADR_COOLDOWN_EPISODES,
            )
            self._adr = ADRCurriculum(adr_cfg)

        # --- terrain patch grid (set if the importer is a generator) ---
        self._terrain_origins = None     # [rows, cols, 3]
        self._t_rows = self._t_cols = 0
        ti = getattr(self, "_terrain", None)
        if ti is not None and getattr(ti, "terrain_origins", None) is not None:
            self._terrain_origins = ti.terrain_origins
            self._t_rows, self._t_cols = self._terrain_origins.shape[0], self._terrain_origins.shape[1]

        # --- vectorized controller ---
        self._controller = VectorizedLQR(
            n, device=dev,
            max_residual_velocity=cfg_mod.MAX_RESIDUAL_VELOCITY,
            max_residual_omega=cfg_mod.MAX_RESIDUAL_OMEGA,
            max_velocity_clip=cfg_mod.MAX_VELOCITY_CLIP,
            max_omega_clip=cfg_mod.MAX_OMEGA_CLIP,
            use_lqr_baseline=self.cfg.use_lqr_baseline,
        )

        # Precompute the fixed/random path bank on CPU once (waypoints + profile).
        self._build_path_bank()

        # wheel joint indices (left = FL,RL ; right = FR,RR)
        from leorover_isaac.assets.leo_rover import LEFT_WHEELS, RIGHT_WHEELS
        self._left_ids, _ = self.robot.find_joints(LEFT_WHEELS)
        self._right_ids, _ = self.robot.find_joints(RIGHT_WHEELS)

    # ---------------------------------------------------------------- scene
    def _setup_scene(self):
        from leorover_isaac.assets.leo_rover import LEO_ROVER_CFG
        self.robot = Articulation(self.cfg.robot if self.cfg.robot is not None else LEO_ROVER_CFG)
        # terrain
        if self.cfg.terrain is not None:
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone & add
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        # --- camera lookahead: a forward, yaw-aligned downward height scanner ---
        # Reproduces _compute_terrain_lookahead's slope sensing. A grid of
        # downward rays ahead of the rover samples terrain height; per-zone
        # finite differences give (slope_mag, cross_slope) for near/mid/far.
        if self.cfg.use_camera_lookahead:
            fwd = self.cfg.camera_forward_offset
            rng = self.cfg.camera_max_range
            # Explicit env-regex path (the sensor is created here, not via the
            # scene cfg, so the {ENV_REGEX_NS} placeholder wouldn't get substituted).
            scanner_cfg = RayCasterCfg(
                prim_path="/World/envs/env_.*/Robot/base_link",
                update_period=0.0,
                offset=RayCasterCfg.OffsetCfg(pos=(fwd + rng * 0.5, 0.0, 20.0)),
                attach_yaw_only=True,
                pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(rng, 1.2)),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
            )
            self._scanner = RayCaster(scanner_cfg)
            self.scene.sensors["scanner"] = self._scanner
        else:
            self._scanner = None

        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.85, 0.8))
        light.func("/World/Light", light)

    # ------------------------------------------------------ path bank (CPU)
    def _build_path_bank(self):
        """Generate the random/template path waypoints + trajectory profile once.

        Mirrors MyEnv2.__init__'s random-path-template generation (base_seed=42)
        and reset()'s trajectory profiling. Stored as numpy arrays; copied into
        per-env GPU buffers on reset.
        """
        v_max = cfg_mod.PATH_V_MAX
        self._bank = []   # list of dict(wps[K,6], total_len, cum[K], goal_xy)
        if self.cfg.use_random_paths:
            base_seed = 42
            for i in range(self.cfg.num_random_paths):
                gen = generate_random_curved_path(
                    min_curvature_angle=self.cfg.min_curvature_angle,
                    max_curvature_angle=self.cfg.max_curvature_angle,
                    total_distance=self.cfg.total_path_distance,
                    seed=base_seed + i,
                )
                self._bank.append(self._profile_path(gen.get_waypoints(), v_max))
        else:
            for pt in path_templates.ALL_PATHS:
                self._bank.append(self._profile_path(pt.get_waypoints(), v_max))
        self._bank_size = len(self._bank)

    def _profile_path(self, wpts, v_max):
        x = [w[0] for w in wpts]
        y = [w[1] for w in wpts]
        yaw = [w[2] for w in wpts]
        vel, omega = compute_trajectory_profile(
            x, y, yaw, v_max=v_max,
            wheel_base=self._controller.wheel_base,
            wheel_radius=self._controller.wheel_radius,
            max_wheel_speed=self._controller.max_wheel_speed,
        )
        K = len(x)
        arr = np.zeros((K, 6), dtype=np.float32)
        arr[:, 0] = x
        arr[:, 1] = y
        arr[:, 3] = yaw
        arr[:, 4] = vel
        arr[:, 5] = omega
        seg = np.sqrt(np.diff(arr[:, 0]) ** 2 + np.diff(arr[:, 1]) ** 2)
        cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
        total = float(cum[-1]) if cum[-1] > 1e-6 else 1.0
        return {"wps": arr, "cum": cum, "total": total,
                "goal": np.array([x[-1], y[-1]], dtype=np.float32), "K": K}

    # ---------------------------------------------------- per-step control
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clamp(-1.0, 1.0).to(self.device)
        self._prev_total_cmd = self._last_total_cmd.clone()

        pos_local, yaw, fwd_vel, _ = self._kin()
        wp = self._gather_current_wp()
        gate = torch.ones(self.num_envs, device=self.device)   # active config: gating disabled
        out = self._controller.forward(self._actions, wp, gate, pos_local[:, :2], yaw, fwd_vel)
        self._last_total_cmd = out["total"]
        self._last_baseline = out["baseline"]
        self._last_residual = out["residual"]
        # left command -> FL,RL ; right command -> FR,RR
        self._wheel_l = out["wheel_left"]
        self._wheel_r = out["wheel_right"]

    def _apply_action(self):
        # Write the same velocity target every substep (PyBullet holds wheel vel
        # constant across its 10 inner steps).
        targets = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        for j in self._left_ids:
            targets[:, j] = self._wheel_l
        for j in self._right_ids:
            targets[:, j] = self._wheel_r
        self.robot.set_joint_velocity_target(targets)

    # ---------------------------------------------------------- kinematics
    def _origins(self):
        """Per-env world origin (terrain patch origin if terrain present)."""
        if self._terrain_origins is not None and getattr(self, "_terrain", None) is not None:
            return self._terrain.env_origins
        return self.scene.env_origins

    def _kin(self):
        """Return (pos_local[n,3], yaw[n], forward_vel[n], lateral_vel[n])."""
        pos_w = self.robot.data.root_pos_w
        pos_local = pos_w - self._origins()
        lin_b = self.robot.data.root_lin_vel_b   # body-frame velocity (Isaac provides it)
        fwd_vel = lin_b[:, 0]
        lat_vel = lin_b[:, 1]
        roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        yaw = _wrap_to_pi(yaw)
        return pos_local, yaw, fwd_vel, lat_vel

    def _gather_current_wp(self):
        idx = self._cur_idx.clamp(0, MAX_WAYPOINTS - 1)
        return torch.gather(self._wps, 1, idx.view(-1, 1, 1).expand(-1, 1, 6)).squeeze(1)

    def _gather_wp(self, idx):
        idx = idx.clamp(0, MAX_WAYPOINTS - 1)
        return torch.gather(self._wps, 1, idx.view(-1, 1, 1).expand(-1, 1, 6)).squeeze(1)

    # ------------------------------------------------- path-tracking metrics
    def _true_cte_and_along(self):
        """Perpendicular distance to segment prev->cur waypoint + along-track dist."""
        pos_local, _, _, _ = self._kin()
        rover = pos_local[:, :2]
        cur = self._gather_wp(self._cur_idx)[:, :2]
        prev = self._gather_wp((self._cur_idx - 1).clamp(min=0))[:, :2]
        ab = cur - prev
        seg_len = torch.norm(ab, dim=-1)
        ap = rover - prev
        denom = (seg_len ** 2).clamp(min=1e-12)
        t = ((ap * ab).sum(-1) / denom).clamp(0.0, 1.0)
        closest = prev + t.unsqueeze(-1) * ab
        cte = torch.norm(rover - closest, dim=-1)
        along = t * seg_len
        degen = seg_len < 1e-6
        cte = torch.where(degen, torch.norm(rover - prev, dim=-1), cte)
        along = torch.where(degen, torch.zeros_like(along), along)
        return cte, along

    def _heading_error(self, yaw):
        target_yaw = self._gather_current_wp()[:, 3]
        return _wrap_to_pi(target_yaw - yaw)

    def _path_progress(self):
        completed = torch.gather(self._cum_len, 1, (self._cur_idx - 1).clamp(min=0).view(-1, 1)).squeeze(1)
        _, along = self._true_cte_and_along()
        completed = completed + along
        return (completed / self._total_len * 100.0).clamp(0.0, 100.0)

    # ------------------------------------------------- camera lookahead
    def _compute_lookahead(self):
        """6 terrain-slope features [slope_near,cross_near, slope_mid,cross_mid,
        slope_far,cross_far] from the forward height scanner.

        Robust to ray ordering: each hit is transformed into the rover body frame
        (forward, lateral), binned into the near/mid/far distance zones from
        config, and the longitudinal + cross-path slope is estimated by a
        weighted finite difference within the zone. Sentinel -1.0 where a zone
        has too few hits (matches MyEnv2's camera_min_hits_per_bin behavior).
        """
        n = self.num_envs
        out = torch.full((n, 6), -1.0, device=self.device)
        if self._scanner is None:
            return out
        try:
            hits = self._scanner.data.ray_hits_w           # [n, R, 3]
            sensor_pos = self._scanner.data.pos_w          # [n, 3]
        except Exception:
            return out

        rover_xy = self.robot.data.root_pos_w[:, :2]
        _, yaw, _, _ = self._kin()
        cy = torch.cos(yaw).unsqueeze(1)
        sy = torch.sin(yaw).unsqueeze(1)
        rel = hits[:, :, :2] - rover_xy.unsqueeze(1)       # [n,R,2]
        fwd = cy * rel[:, :, 0] + sy * rel[:, :, 1]        # forward dist [n,R]
        lat = -sy * rel[:, :, 0] + cy * rel[:, :, 1]       # lateral [n,R]
        z = hits[:, :, 2]                                  # world height [n,R]
        valid = torch.isfinite(z) & torch.isfinite(fwd) & (fwd > 0.0)

        bounds = [(0.0, self.cfg.camera_zone_near),
                  (self.cfg.camera_zone_near, self.cfg.camera_zone_mid),
                  (self.cfg.camera_zone_mid, self.cfg.camera_max_range)]
        min_hits = 3
        for zi, (lo, hi) in enumerate(bounds):
            m = valid & (fwd >= lo) & (fwd < hi)
            cnt = m.sum(dim=1)
            ok = cnt >= min_hits
            mf = m.float()
            wsum = mf.sum(dim=1).clamp(min=1.0)
            # zero-mean coords within the zone (masked)
            f_mean = (fwd * mf).sum(1) / wsum
            l_mean = (lat * mf).sum(1) / wsum
            z_mean = (z * mf).sum(1) / wsum
            df = (fwd - f_mean.unsqueeze(1)) * mf
            dl = (lat - l_mean.unsqueeze(1)) * mf
            dz = (z - z_mean.unsqueeze(1)) * mf
            # least-squares slope along forward and lateral (independent 1-D fits)
            slope_long = (df * dz).sum(1) / (df * df).sum(1).clamp(min=1e-6)
            slope_cross = (dl * dz).sum(1) / (dl * dl).sum(1).clamp(min=1e-6)
            slope_mag = torch.sqrt(slope_long ** 2 + slope_cross ** 2)
            out[:, 2 * zi] = torch.where(ok, slope_mag, torch.full_like(slope_mag, -1.0))
            out[:, 2 * zi + 1] = torch.where(ok, slope_cross, torch.full_like(slope_cross, -1.0))
        return torch.nan_to_num(out, nan=-1.0, posinf=1.5, neginf=-1.5)

    # ----------------------------------------------------- observations
    def _get_observations(self):
        pos_local, yaw, fwd_vel, lat_vel = self._kin()
        cte, _ = self._true_cte_and_along()
        head = self._heading_error(yaw)
        wp = self._gather_current_wp()
        wp_dx = wp[:, 0] - pos_local[:, 0]
        wp_dy = wp[:, 1] - pos_local[:, 1]
        cy = torch.cos(-yaw); sy = torch.sin(-yaw)
        wp_dx_b = cy * wp_dx - sy * wp_dy
        wp_dy_b = sy * wp_dx + cy * wp_dy
        grav = self.robot.data.projected_gravity_b   # [n,3] unit gravity in body frame

        cols = [cte, head, fwd_vel, lat_vel, wp_dx_b, wp_dy_b]
        if self.cfg.use_lqr_baseline:
            cols += [self._last_baseline[:, 0], self._last_baseline[:, 1]]
        cols += [grav[:, 0], grav[:, 1], grav[:, 2]]
        obs = torch.stack(cols, dim=-1)
        if self.cfg.use_camera_lookahead:
            self._last_lookahead = self._compute_lookahead()
            obs = torch.cat([obs, self._last_lookahead], dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return {"policy": obs}

    # ---------------------------------------------------------- rewards
    def _get_rewards(self):
        pos_local, yaw, fwd_vel, lat_vel = self._kin()
        cte, _ = self._true_cte_and_along()
        head = self._heading_error(yaw)

        w = self._ppo
        r_cte = -w['ppo_w_cte'] * cte ** 2
        r_head = -w['ppo_w_heading'] * head ** 2
        vel_scale = (1.0 - cte.abs() / w['ppo_cte_ok_threshold']).clamp(min=0.0)
        r_vel = w['ppo_w_velocity'] * vel_scale * fwd_vel.clamp(min=0.0)
        dvw = (self._last_total_cmd - self._prev_total_cmd).abs()
        r_smooth = -w['ppo_w_smoothness'] * (dvw[:, 0] + dvw[:, 1])
        r_alive = -w['ppo_w_alive'] * torch.ones(self.num_envs, device=self.device)

        progress = self._path_progress()
        prog_delta = (progress - self._prev_progress) / 100.0
        self._prev_progress = progress
        r_prog = w['ppo_w_progress'] * prog_delta

        if self.cfg.use_lqr_baseline and w['ppo_w_effort'] > 0.0:
            mv = max(cfg_mod.MAX_RESIDUAL_VELOCITY, 1e-6)
            mw = max(cfg_mod.MAX_RESIDUAL_OMEGA, 1e-6)
            rvn = self._last_residual[:, 0] / mv
            rwn = self._last_residual[:, 1] / mw
            r_eff = -w['ppo_w_effort'] * (rvn ** 2 + rwn ** 2)
        else:
            r_eff = torch.zeros(self.num_envs, device=self.device)

        reward = r_cte + r_head + r_vel + r_smooth + r_alive + r_prog + r_eff

        goal = self._is_goal_reached()
        fail = (self._is_cte_too_large() | self._is_oob() | self._is_stagnation_timeout()
                | self._is_out_of_time() | self._is_flipped())
        reward = reward + goal.float() * w['ppo_success_bonus']
        reward = reward - (fail & ~goal).float() * w['ppo_failure_penalty']
        return torch.nan_to_num(reward, nan=-1.0)

    # ----------------------------------------------------------- dones
    def _get_dones(self):
        self._update_waypoint_skip()
        self._update_stagnation()
        self._sim_time += self.cfg.sim.dt * self.cfg.decimation

        # accumulate per-episode CTE + success for ADR (read at reset)
        cte, _ = self._true_cte_and_along()
        self._ep_cte_sum += cte.abs()
        self._ep_steps += 1.0
        goal = self._is_goal_reached()
        self._ep_success = self._ep_success | goal

        terminated = (goal | self._is_flipped() | self._is_oob()
                      | self._is_cte_too_large() | self._is_stagnation_timeout())
        truncated = self._is_out_of_time()
        return terminated, truncated

    # termination predicates (vectorized)
    def _is_goal_reached(self):
        pos_local, _, _, _ = self._kin()
        d = torch.norm(pos_local[:, :2] - self._goal_xy, dim=-1)
        return d < self._goal_tol

    def _is_flipped(self):
        return self.robot.data.projected_gravity_b[:, 2] > self._flip_threshold_gz

    def _is_oob(self):
        pos_local, _, _, _ = self._kin()
        gx = self._goal_xy[:, 0].abs() + 5.0
        gy = self._goal_xy[:, 1].abs() + 5.0
        x, y = pos_local[:, 0], pos_local[:, 1]
        return ~((x.abs() <= gx) & (y.abs() <= gy))

    def _is_cte_too_large(self):
        if not self._ppo['use_pure_ppo_reward']:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        cte, _ = self._true_cte_and_along()
        return cte.abs() > self._max_cte_term

    def _is_stagnation_timeout(self):
        return self._stagnation > self._res['stagnation_termination_steps']

    def _is_out_of_time(self):
        return self._sim_time > self.cfg.episode_length_s

    # ------------------------------------------------- per-step bookkeeping
    def _update_waypoint_skip(self):
        last = self._num_wp - 1
        can = self._cur_idx < last
        pos_local, _, _, _ = self._kin()
        rover = pos_local[:, :2]
        cur = self._gather_wp(self._cur_idx)[:, :2]
        nxt = self._gather_wp((self._cur_idx + 1).clamp(max=MAX_WAYPOINTS - 1))[:, :2]
        d_cur = torch.norm(rover - cur, dim=-1)
        d_nxt = torch.norm(rover - nxt, dim=-1)
        advance = can & (d_nxt < d_cur)
        self._prev_idx = torch.where(advance, self._cur_idx, self._prev_idx)
        self._cur_idx = torch.where(advance, self._cur_idx + 1, self._cur_idx)

    def _update_stagnation(self):
        _, _, fwd_vel, _ = self._kin()
        stuck = fwd_vel.abs() < self._res['stagnation_velocity_threshold']
        sustain_steps = self._res['residual_recovery_sustain_steps']
        self._stagnation = torch.where(stuck, self._stagnation + 1, self._stagnation)
        self._recovery_sustain = torch.where(stuck, torch.zeros_like(self._recovery_sustain),
                                             self._recovery_sustain + 1)
        recovered = (~stuck) & (self._recovery_sustain >= sustain_steps)
        decaying = (~stuck) & (self._recovery_sustain < sustain_steps)
        self._stagnation = torch.where(recovered, torch.zeros_like(self._stagnation), self._stagnation)
        self._stagnation = torch.where(decaying, (self._stagnation - 1).clamp(min=0), self._stagnation)

    # ----------------------------------------------- ADR + terrain resample
    def _adr_max_level(self) -> int:
        """Map the ADR terrain ceiling (0-100%) to a difficulty row index."""
        if self._adr is None or self._t_rows <= 1:
            return max(self._t_rows - 1, 0)
        frac = self._adr.terrain_max / max(self._adr.config.terrain_intensity_max_limit, 1e-6)
        return int(min(self._t_rows - 1, max(0, round(frac * (self._t_rows - 1)))))

    def _report_adr_and_resample(self, env_ids):
        """Report finished episodes to ADR, advance the curriculum, and reassign
        each resetting env to a fresh random terrain patch in [0, ADR ceiling]."""
        # 1. report each finished episode (sequential, like the SB3 ADRCallback)
        if self._adr is not None:
            steps = self._ep_steps[env_ids].clamp(min=1.0)
            mean_cte = (self._ep_cte_sum[env_ids] / steps).detach().cpu().numpy()
            succ = self._ep_success[env_ids].detach().cpu().numpy()
            for s, c in zip(succ.tolist(), mean_cte.tolist()):
                self._adr.report_episode(success=bool(s), mean_cte=float(c))
                if self._adr.should_update_difficulty():
                    self._adr.update_difficulty()

        # 2. reassign terrain patches (random row up to ceiling, random column)
        if self._terrain_origins is not None and self._t_rows > 0:
            k = len(env_ids)
            max_level = self._adr_max_level()
            levels = torch.randint(0, max_level + 1, (k,), device=self.device)
            cols = torch.randint(0, self._t_cols, (k,), device=self.device)
            new_origins = self._terrain_origins[levels, cols]            # [k,3]
            self._terrain.env_origins[env_ids] = new_origins
            # intensity for logging: difficulty fraction * configured ceiling
            denom = max(self._t_rows - 1, 1)
            self._terrain_intensity[env_ids] = (levels.float() / denom) * cfg_mod.ADR_TERRAIN_MAX_LIMIT

    # ----------------------------------------------------------- reset
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        dev = self.device

        # ADR bookkeeping + terrain-patch reassignment for the finishing envs.
        self._report_adr_and_resample(env_ids)

        # --- choose a path from the bank for each resetting env ---
        n_reset = len(env_ids)
        choices = np.random.randint(0, self._bank_size, size=n_reset)
        for k, e in enumerate(env_ids.tolist()):
            entry = self._bank[choices[k]]
            K = entry["K"]
            self._wps[e].zero_()
            self._wps[e, :K] = torch.from_numpy(entry["wps"]).to(dev)
            self._cum_len[e].zero_()
            self._cum_len[e, :K] = torch.from_numpy(entry["cum"]).to(dev)
            self._num_wp[e] = K
            self._total_len[e] = entry["total"]
            self._goal_xy[e] = torch.from_numpy(entry["goal"]).to(dev)
            wp2 = self._wps[e, :K, :2]
            dist0 = torch.norm(wp2, dim=-1)
            found = torch.nonzero(dist0 >= 0.5, as_tuple=False)
            self._cur_idx[e] = int(found[0]) if len(found) else (1 if K > 1 else 0)
            self._prev_idx[e] = 0

        # --- reset robot state at the (possibly new) env origin ---
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._origins()[env_ids]
        root_state[:, 2] += 0.3   # spawn slightly above terrain; PhysX settles it
        self.robot.write_root_state_to_sim(root_state, env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # --- reset per-env runtime buffers ---
        self._prev_total_cmd[env_ids] = 0.0
        self._last_total_cmd[env_ids] = 0.0
        self._last_baseline[env_ids] = 0.0
        self._last_residual[env_ids] = 0.0
        self._last_lookahead[env_ids] = -1.0
        self._prev_progress[env_ids] = 0.0
        self._stagnation[env_ids] = 0
        self._recovery_sustain[env_ids] = 0
        self._sim_time[env_ids] = 0.0
        self._ep_cte_sum[env_ids] = 0.0
        self._ep_steps[env_ids] = 0.0
        self._ep_success[env_ids] = False
        init_yaw = self._gather_wp(self._cur_idx)[env_ids, 3]
        self._controller.reset_idx(env_ids, init_yaw)

    # ----------------------------------------------------- ADR diagnostics
    @property
    def adr_stats(self):
        return self._adr.get_stats() if self._adr is not None else {}
