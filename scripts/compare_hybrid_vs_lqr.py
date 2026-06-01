#!/usr/bin/env python
# scripts/compare_hybrid_vs_lqr.py
"""
Head-to-head Hybrid (LQR + trained residual) vs pure LQR on identical terrain
seeds — the Isaac Lab equivalent of run_experiment.py's `run_comparison` mode
and the PORTING_ROADMAP.md Phase 7 / "port is done" parity test.

Runs the hybrid env twice over the SAME set of envs/seeds:
  * Hybrid pass: actions = trained policy(obs)
  * LQR pass:    actions = 0  (zero residual -> pure LQR baseline)
and writes two CSVs (PyBullet episode_metrics schema). Aggregate success rates
should match the PyBullet result within ~2 pp if the port is faithful.

    isaaclab -p scripts/compare_hybrid_vs_lqr.py \
        --checkpoint logs/leo_rover_mars_hybrid/<run>/model_final.pt \
        --num_envs 256 --episodes 1000
"""

from __future__ import annotations
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--out", default="logs/compare")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
import leorover_isaac  # noqa: F401  (registers tasks)
from leorover_isaac.envs.leo_rover_mars_hybrid_env import LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg
from leorover_isaac.tasks.leo_rover_agents import LeoRoverMarsHybridPPORunnerCfg
from leorover_isaac.utils.recorder import EpisodeMetricsRecorder
from rsl_rl.runners import OnPolicyRunner
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
except Exception:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper  # type: ignore


def _run_pass(label, action_fn, runner=None):
    env_cfg = LeoRoverMarsHybridEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed                      # identical seeds -> paired terrain
    env = LeoRoverMarsHybridEnv(cfg=env_cfg)
    rec = EpisodeMetricsRecorder(os.path.join(args.out, label), env)
    _orig = env.step
    def _rec(a):
        o, r, te, tr, ex = _orig(a); rec.record_step(r, te | tr); return o, r, te, tr, ex
    env.step = _rec
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    done = 0
    while done < args.episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = action_fn(obs)
            obs_dict, _, term, trunc, _ = env.step(actions)
            obs = obs_dict["policy"]
        done += int((term | trunc).sum().item())
    print(f"[compare:{label}] {rec.episode_count} episodes -> {rec.csv_path}")
    env.close()


def main():
    # Build a temporary wrapped env just to load the policy weights.
    env_cfg = LeoRoverMarsHybridEnvCfg(); env_cfg.scene.num_envs = args.num_envs
    agent_cfg = LeoRoverMarsHybridPPORunnerCfg()
    tmp = RslRlVecEnvWrapper(LeoRoverMarsHybridEnv(cfg=env_cfg))
    runner = OnPolicyRunner(tmp, agent_cfg.to_dict(), log_dir=None, device=str(tmp.unwrapped.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=tmp.unwrapped.device)
    dev = tmp.unwrapped.device
    tmp.close()

    _run_pass("hybrid", lambda obs: policy(obs))
    _run_pass("lqr", lambda obs: torch.zeros(args.num_envs, 2, device=dev))
    simulation_app.close()


if __name__ == "__main__":
    main()
