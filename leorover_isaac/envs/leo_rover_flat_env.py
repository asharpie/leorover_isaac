# leo_rover_flat_env.py
"""
LeoRoverFlatEnv — Phase 2: rover on a flat ground plane, pure PPO, no camera.

PyBullet equivalent: MyEnv2(terrain_intensity=0.0, use_lqr_baseline=False,
use_pure_ppo_reward=True, use_camera_lookahead=False).

This is the simplest task — used to validate the data path (action -> wheels ->
motion -> obs -> reward) before introducing terrain. Observation is the 9D pure
state vector [cte, heading_err, fwd_vel, lat_vel, wp_dx_b, wp_dy_b, gx, gy, gz].
All reward/termination logic lives in LeoRoverBaseEnv.
"""

from __future__ import annotations

from leorover_isaac.envs.leo_rover_base_env import LeoRoverBaseEnv, LeoRoverBaseEnvCfg, _ISAAC

__all__ = ["LeoRoverFlatEnv", "LeoRoverFlatEnvCfg"]

if _ISAAC:
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
    from leorover_isaac.assets.leo_rover import LEO_ROVER_CFG

    @configclass
    class LeoRoverFlatEnvCfg(LeoRoverBaseEnvCfg):
        observation_space: int = 9
        episode_length_s: float = 400.0   # 2000 policy steps @ 0.2 s (matches MyEnv2 cap)
        use_lqr_baseline: bool = False
        use_camera_lookahead: bool = False
        use_mars_terrain: bool = False

        def __post_init__(self):
            if hasattr(super(), "__post_init__"):
                super().__post_init__()
            self.robot = LEO_ROVER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
            self.terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                debug_vis=False,
            )


class LeoRoverFlatEnv(LeoRoverBaseEnv):
    """Flat-ground pure-PPO task (see module docstring)."""
    pass
