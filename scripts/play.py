#!/usr/bin/env python
# scripts/play.py
"""
Load a trained Leo Rover checkpoint and roll it out (visual or headless),
writing an episode_metrics.csv so the run is analysable in evaluate_training.py.

Equivalent to running run_experiment.py in a deployment/eval pass.

    isaaclab -p scripts/play.py --task Isaac-LeoRover-Mars-v0 \
        --checkpoint logs/leo_rover_mars/<run>/model_final.pt --num_envs 16
"""

from __future__ import annotations
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-LeoRover-Mars-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=200)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
import leorover_isaac  # registers tasks
from scripts.train import _TASKS  # reuse the task table
from leorover_isaac.utils.recorder import EpisodeMetricsRecorder
from rsl_rl.runners import OnPolicyRunner
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
except Exception:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper  # type: ignore


def main():
    env_cls, env_cfg_cls, runner_cfg_cls = _TASKS[args.task]
    env_cfg = env_cfg_cls(); env_cfg.scene.num_envs = args.num_envs
    agent_cfg = runner_cfg_cls()

    env = env_cls(cfg=env_cfg, render_mode="rgb_array")
    run_dir = os.path.join(os.path.dirname(args.checkpoint) or ".", "eval")
    recorder = EpisodeMetricsRecorder(os.path.join(run_dir, "csv"), env)
    _orig = env.step
    def _rec(a):
        o, r, te, tr, ex = _orig(a); recorder.record_step(r, te | tr); return o, r, te, tr, ex
    env.step = _rec

    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(env.unwrapped.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations() if hasattr(env, "get_observations") else (env.reset()[0], None)
    done_count = 0
    target = args.episodes
    while done_count < target and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        done_count += int(dones.sum().item()) if hasattr(dones, "sum") else 0

    print(f"[play] wrote {recorder.episode_count} episodes -> {recorder.csv_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
