# leo_rover_mars_hybrid_env.py
"""
LeoRoverMarsHybridEnv — Phase 4: hybrid residual RL (LQR baseline + bounded PPO
residual) on Martian terrain. The v33.9 hybrid configuration ported.

PyBullet equivalent: the `train_hybrid_ppo` configuration —
MyEnv2(use_lqr_baseline=True, use_pure_ppo_reward=True, use_camera_lookahead=True)
with MAX_RESIDUAL_VELOCITY=0.15, MAX_RESIDUAL_OMEGA=0.30, PPO_W_EFFORT=0.5.

Observation: 11D base (adds the two LQR baseline commands) + 6 camera lookahead
= 17D. The LQR baseline is computed every step by the GPU-vectorized
VectorizedLQR controller; PPO outputs a residual in [-1,1] that is scaled by the
residual bounds and added on top, then the L2 residual-effort penalty
(PPO_W_EFFORT) pulls it toward zero unless it actively helps — exactly the v33.9
design. All of this lives in LeoRoverBaseEnv, switched on by use_lqr_baseline.
"""

from __future__ import annotations

import config as cfg_mod
from leorover_isaac.envs.leo_rover_base_env import LeoRoverBaseEnv, LeoRoverBaseEnvCfg, _ISAAC

__all__ = ["LeoRoverMarsHybridEnv", "LeoRoverMarsHybridEnvCfg"]

if _ISAAC:
    from isaaclab.utils import configclass
    from leorover_isaac.assets.leo_rover import LEO_ROVER_CFG
    from leorover_isaac.terrain.mars_heightfield import make_mars_terrain_cfg

    @configclass
    class LeoRoverMarsHybridEnvCfg(LeoRoverBaseEnvCfg):
        observation_space: int = 17        # 11 base (incl. LQR cmds) + 6 camera
        episode_length_s: float = 400.0
        use_lqr_baseline: bool = True
        use_camera_lookahead: bool = bool(cfg_mod.USE_CAMERA_LOOKAHEAD)
        use_mars_terrain: bool = True

        def __post_init__(self):
            if hasattr(super(), "__post_init__"):
                super().__post_init__()
            self.robot = LEO_ROVER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
            # Hybrid mode applies the ADR mode-conditional thresholds in config.py
            # (ADR_TERRAIN_MAX_START=30, etc. — set there when agent_mode=="Hybrid").
            self.terrain = make_mars_terrain_cfg(
                intensity_min=cfg_mod.TRAINING_TERRAIN_MIN,
                intensity_max=cfg_mod.ADR_TERRAIN_MAX_LIMIT,
                curriculum=True,
            )
            if not self.use_camera_lookahead:
                self.observation_space = 11


class LeoRoverMarsHybridEnv(LeoRoverBaseEnv):
    """Hybrid LQR+residual Mars-terrain task (see module docstring)."""
    pass
