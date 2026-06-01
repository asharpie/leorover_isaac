"""Task configs + gymnasium registrations for the Leo Rover Isaac Lab port.

Registers three gym tasks, mirroring the three PyBullet agent modes:

    Isaac-LeoRover-Flat-v0          flat ground, pure PPO          (smoke test)
    Isaac-LeoRover-Mars-v0          Mars terrain, pure PPO         (train_ppo)
    Isaac-LeoRover-Mars-Hybrid-v0   Mars terrain, LQR + residual   (train_hybrid_ppo)

Each registration wires both the env cfg and the rsl_rl PPO runner cfg, the
Isaac Lab convention so `scripts/train.py --task <id>` resolves everything.
"""

from __future__ import annotations


def register_tasks():
    """Idempotently register the Leo Rover gym tasks. No-op if gymnasium/Isaac
    Lab is unavailable (e.g. running unit tests outside the conda env)."""
    try:
        import gymnasium as gym
    except Exception:
        return

    specs = {
        "Isaac-LeoRover-Flat-v0": (
            "leorover_isaac.envs.leo_rover_flat_env:LeoRoverFlatEnv",
            "leorover_isaac.envs.leo_rover_flat_env:LeoRoverFlatEnvCfg",
            "leorover_isaac.tasks.leo_rover_agents:LeoRoverFlatPPORunnerCfg",
        ),
        "Isaac-LeoRover-Mars-v0": (
            "leorover_isaac.envs.leo_rover_mars_env:LeoRoverMarsEnv",
            "leorover_isaac.envs.leo_rover_mars_env:LeoRoverMarsEnvCfg",
            "leorover_isaac.tasks.leo_rover_agents:LeoRoverMarsPPORunnerCfg",
        ),
        "Isaac-LeoRover-Mars-Hybrid-v0": (
            "leorover_isaac.envs.leo_rover_mars_hybrid_env:LeoRoverMarsHybridEnv",
            "leorover_isaac.envs.leo_rover_mars_hybrid_env:LeoRoverMarsHybridEnvCfg",
            "leorover_isaac.tasks.leo_rover_agents:LeoRoverMarsHybridPPORunnerCfg",
        ),
    }
    existing = set(gym.registry.keys())
    for task_id, (entry, env_cfg, rsl_cfg) in specs.items():
        if task_id in existing:
            continue
        gym.register(
            id=task_id,
            entry_point=entry,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": env_cfg,
                "rsl_rl_cfg_entry_point": rsl_cfg,
            },
        )
