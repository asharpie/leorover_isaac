#!/usr/bin/env python
# scripts/train.py
"""
Train a Leo Rover policy in Isaac Lab with rsl_rl PPO.

This mirrors `run_experiment.py`'s train_ppo / train_hybrid_ppo, but on the
GPU-vectorized Isaac Lab stack. It:
  1. launches the SimulationApp,
  2. builds the requested task env (Flat / Mars / Mars-Hybrid),
  3. attaches an EpisodeMetricsRecorder that writes episode_metrics.csv in the
     PyBullet schema (so evaluate_training.py analyses Isaac runs unchanged),
  4. runs rsl_rl's OnPolicyRunner with the v33.9-mirrored PPO config,
  5. saves checkpoints under logs/<experiment>/.

Examples:
    # smoke test (flat, few envs, GUI)
    isaaclab -p scripts/train.py --task Isaac-LeoRover-Flat-v0 --num_envs 64

    # pure-PPO Mars training, headless, 4096 envs (the fast path)
    isaaclab -p scripts/train.py --task Isaac-LeoRover-Mars-v0 --num_envs 4096 --headless

    # hybrid residual training
    isaaclab -p scripts/train.py --task Isaac-LeoRover-Mars-Hybrid-v0 --num_envs 4096 --headless
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable so `import config` / `import leorover_isaac`
# resolve no matter how this is launched (CLI, GUI, with or without PYTHONPATH).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- 1. CLI + AppLauncher (must come before importing isaac sim modules) ----
parser = argparse.ArgumentParser(description="Train Leo Rover (Isaac Lab / rsl_rl PPO).")
parser.add_argument("--task", type=str, default="Isaac-LeoRover-Mars-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")

try:
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        f"[train] Could not launch Isaac Sim ({exc}).\n"
        f"Run this inside the Isaac Lab python env, e.g.:\n"
        f"    isaaclab -p scripts/train.py --task Isaac-LeoRover-Mars-v0 --headless"
    )

# ---- 2. Imports that require the running app ----
import gymnasium as gym
import torch

import leorover_isaac  # registers the gym tasks
from leorover_isaac.tasks.leo_rover_agents import (
    LeoRoverFlatPPORunnerCfg, LeoRoverMarsPPORunnerCfg, LeoRoverMarsHybridPPORunnerCfg,
)
from leorover_isaac.envs.leo_rover_flat_env import LeoRoverFlatEnv, LeoRoverFlatEnvCfg
from leorover_isaac.envs.leo_rover_mars_env import LeoRoverMarsEnv, LeoRoverMarsEnvCfg
from leorover_isaac.envs.leo_rover_mars_hybrid_env import (
    LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg,
)
from leorover_isaac.utils.recorder import EpisodeMetricsRecorder

from rsl_rl.runners import OnPolicyRunner
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
except Exception:  # older namespace
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper  # type: ignore


_TASKS = {
    "Isaac-LeoRover-Flat-v0":        (LeoRoverFlatEnv, LeoRoverFlatEnvCfg, LeoRoverFlatPPORunnerCfg),
    "Isaac-LeoRover-Mars-v0":        (LeoRoverMarsEnv, LeoRoverMarsEnvCfg, LeoRoverMarsPPORunnerCfg),
    "Isaac-LeoRover-Mars-Hybrid-v0": (LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg, LeoRoverMarsHybridPPORunnerCfg),
}


def main():
    env_cls, env_cfg_cls, runner_cfg_cls = _TASKS[args.task]

    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    agent_cfg = runner_cfg_cls()
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    log_root = os.path.join(args.log_dir, agent_cfg.experiment_name)
    os.makedirs(log_root, exist_ok=True)
    from datetime import datetime
    run_dir = os.path.join(log_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Build env
    env = env_cls(cfg=env_cfg, render_mode="rgb_array" if not args.headless else None)

    # --- attach the PyBullet-schema CSV recorder (non-invasive step hook) ---
    recorder = EpisodeMetricsRecorder(os.path.join(run_dir, "csv"), env)
    _orig_step = env.step

    def _step_with_record(action):
        obs, rew, terminated, truncated, extras = _orig_step(action)
        try:
            recorder.record_step(rew, terminated | truncated)
        except Exception as e:
            if os.environ.get("LEOROVER_DEBUG"):
                print(f"[recorder] {e}")
        return obs, rew, terminated, truncated, extras

    env.step = _step_with_record

    # Wrap for rsl_rl and run PPO
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=run_dir, device=str(env.unwrapped.device))
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", "leorover_isaac")

    print(f"[train] task={args.task} num_envs={args.num_envs} -> logging to {run_dir}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    runner.save(os.path.join(run_dir, "model_final.pt"))
    print(f"[train] done. checkpoints + episode_metrics.csv in {run_dir}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
