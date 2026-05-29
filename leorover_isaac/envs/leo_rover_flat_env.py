"""leo_rover_flat_env.py — Phase 2 stub.

Minimal Isaac Lab env: rover on a flat ground plane, no terrain, no LQR
baseline. The purpose of this env is to validate the data path end-to-end
before introducing complexity. If training PPO on this can't learn to
follow a straight line, none of the later phases will work either.

PyBullet equivalent: MyEnv2(terrain_intensity=0.0, use_lqr_baseline=False,
use_pure_ppo_reward=False, use_camera_lookahead=False).

TODO (Phase 2):
  - Implement _setup_scene with flat GroundPlane and one rover articulation
  - Implement _pre_physics_step / _apply_action mapping [v, omega] action
    to per-wheel velocities via the unicycle inverse kinematics
  - Implement _get_observations returning [cte, heading_err, vx, vy, omega_z,
    distance_to_goal] as a [num_envs, 6] tensor
  - Implement _get_rewards returning -cte (placeholder, swap for real reward
    later)
  - Implement _get_dones with timeout (1024 steps) and goal-reached
    (distance < 0.5 m) conditions
  - Implement _reset_idx randomizing rover spawn pose

This file is intentionally a stub so the package imports cleanly even
before the port begins.
"""

from __future__ import annotations

# Real implementation will import:
#   from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
#   from isaaclab.scene import InteractiveSceneCfg
#   from isaaclab.assets import ArticulationCfg
#   from isaaclab.sim import SimulationCfg
#   import torch

__all__ = ["LeoRoverFlatEnv"]


class LeoRoverFlatEnv:
    """Stub for Phase 2 — see module docstring."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LeoRoverFlatEnv is a Phase 2 stub. See PORTING_ROADMAP.md."
        )
