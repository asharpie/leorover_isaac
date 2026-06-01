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
    "mean_residual_v_norm,mean_residual_w_norm\n"
)


class EpisodeMetricsRecorder:
    def __init__(self, log_dir: str, env, max_residual_v: float = None, max_residual_w: float = None):
        self.env = env
        self.n = env.num_envs
        self.device = env.device
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "episode_metrics.csv")
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
            self._slope_sum = z()
        else:
            for buf in (self._cte_sum, self._cte_max, self._rew_sum, self._steps,
                        self._resv_sum, self._resw_sum, self._roll_max,
                        self._pitch_max, self._slope_sum):
                buf[idx] = 0.0

    @torch.no_grad()
    def record_step(self, reward: torch.Tensor, done: torch.Tensor):
        env = self.env
        cte, _ = env._true_cte_and_along()
        cte = cte.abs()
        # residual norms (hybrid); for pure PPO the residual IS the command -> 0 contribution
        resv = (env._last_residual[:, 0].abs() / self._mv) if env.cfg.use_lqr_baseline else torch.zeros_like(cte)
        resw = (env._last_residual[:, 1].abs() / self._mw) if env.cfg.use_lqr_baseline else torch.zeros_like(cte)
        # roll/pitch from quaternion
        try:
            from isaaclab.utils.math import euler_xyz_from_quat
            roll, pitch, _ = euler_xyz_from_quat(env.robot.data.root_quat_w)
        except Exception:
            roll = torch.zeros_like(cte); pitch = torch.zeros_like(cte)
        # local slope (deg) from body-frame gravity tilt: slope ~ atan(|g_xy|/|g_z|)
        grav = env.robot.data.projected_gravity_b
        tilt = torch.atan2(torch.norm(grav[:, :2], dim=-1), grav[:, 2].abs().clamp(min=1e-6))
        slope_deg = torch.rad2deg(tilt)

        self._cte_sum += cte
        self._cte_max = torch.maximum(self._cte_max, cte)
        self._rew_sum += reward
        self._steps += 1.0
        self._resv_sum += resv
        self._resw_sum += resw
        self._roll_max = torch.maximum(self._roll_max, roll.abs())
        self._pitch_max = torch.maximum(self._pitch_max, pitch.abs())
        self._slope_sum += slope_deg

        done_idx = torch.nonzero(done, as_tuple=False).flatten()
        if len(done_idx) == 0:
            return
        self._flush(done_idx)
        self._reset_accum(done_idx)

    def _flush(self, idx):
        env = self.env
        progress = env._path_progress()
        success = env._is_goal_reached().float()
        terr_int = getattr(env, "_terrain_intensity", torch.zeros(self.n, device=self.device))
        fric_int = getattr(env, "_friction_intensity", torch.zeros(self.n, device=self.device))

        idx_l = idx.tolist()
        rows = []
        steps = self._steps.clamp(min=1.0)
        for e in idx_l:
            self.episode_count += 1
            mean_cte = float(self._cte_sum[e] / steps[e])
            mean_rps = float(self._rew_sum[e] / steps[e])
            mean_slope = float(self._slope_sum[e] / steps[e])
            mean_rv = float(self._resv_sum[e] / steps[e])
            mean_rw = float(self._resw_sum[e] / steps[e])
            rows.append(
                f"{self.episode_count},{mean_cte:.4f},{float(self._cte_max[e]):.4f},"
                f"{float(self._rew_sum[e]):.2f},{mean_rps:.4f},0.0000,"
                f"{int(steps[e])},{int(success[e])},"
                f"{float(terr_int[e]):.1f},{float(fric_int[e]):.1f},"
                f"0.00,0.00,{mean_slope:.2f},"
                f"{float(progress[e]):.1f},"
                f"{float(self._roll_max[e]):.4f},{float(self._pitch_max[e]):.4f},"
                f"{mean_rv:.4f},{mean_rw:.4f}\n"
            )
        with open(self.csv_path, "a") as f:
            f.writelines(rows)
