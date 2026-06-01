# leo_rover_mars_env.py
"""
LeoRoverMarsEnv — Phase 3: pure PPO on randomized Martian heightfield terrain.

PyBullet equivalent: the `train_ppo` configuration —
MyEnv2(use_lqr_baseline=False, use_pure_ppo_reward=True, use_camera_lookahead=True),
ADR terrain curriculum, terrain 10-50%+ (ramped), friction 50-90%.

Observation: 9D pure state + 6 camera-lookahead slope features = 15D
(matches config.USE_CAMERA_LOOKAHEAD=True).

Reward/termination logic is inherited from LeoRoverBaseEnv (the v33.9 pure-PPO
reward). Terrain comes from leorover_isaac.terrain.make_mars_terrain_cfg(), whose
difficulty rows are the ADR curriculum axis.
"""

from __future__ import annotations

import config as cfg_mod
from leorover_isaac.envs.leo_rover_base_env import LeoRoverBaseEnv, LeoRoverBaseEnvCfg, _ISAAC

__all__ = ["LeoRoverMarsEnv", "LeoRoverMarsEnvCfg"]

if _ISAAC:
    from isaaclab.utils import configclass
    from leorover_isaac.assets.leo_rover import LEO_ROVER_CFG
    from leorover_isaac.terrain.mars_heightfield import make_mars_terrain_cfg

    @configclass
    class LeoRoverMarsEnvCfg(LeoRoverBaseEnvCfg):
        observation_space: int = 15        # 9 base + 6 camera lookahead
        episode_length_s: float = 400.0
        use_lqr_baseline: bool = False
        use_camera_lookahead: bool = bool(cfg_mod.USE_CAMERA_LOOKAHEAD)
        use_mars_terrain: bool = True

        def __post_init__(self):
            if hasattr(super(), "__post_init__"):
                super().__post_init__()
            self.robot = LEO_ROVER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
            self.terrain = make_mars_terrain_cfg(
                intensity_min=cfg_mod.TRAINING_TERRAIN_MIN,
                intensity_max=cfg_mod.ADR_TERRAIN_MAX_LIMIT,
                curriculum=True,
            )
            # If camera lookahead disabled in config, drop the 6 extra dims.
            if not self.use_camera_lookahead:
                self.observation_space = 9


class LeoRoverMarsEnv(LeoRoverBaseEnv):
    """Pure-PPO Mars-terrain task (see module docstring)."""
    pass
