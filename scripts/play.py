#!/usr/bin/env python
# scripts/play.py
"""
Load a trained Leo Rover checkpoint and roll it out — optionally recording an
mp4 you can download and watch — and print motion diagnostics so you can SEE
(and measure) what the rover is actually doing.

Headless RunPod (record video + print stats):
    isaaclab -p scripts/play.py \
        --task Isaac-LeoRover-Mars-v0 \
        --checkpoint logs/leo_rover_mars/<run>/model_4200.pt \
        --num_envs 4 --headless --enable_cameras --video --video_length 600

Then pull the clip off the pod (run on your Windows machine):
    scp -P <port> -r root@<host>:/workspace/leorover_isaac/logs/leo_rover_mars/<run>/eval/videos .

Quick numbers only (no rendering, very robust — use this if video errors out):
    isaaclab -p scripts/play.py --task Isaac-LeoRover-Mars-v0 \
        --checkpoint logs/leo_rover_mars/<run>/model_4200.pt --num_envs 64 --headless

NOTE: this is self-contained on purpose. It does NOT `import scripts.train`,
because train.py parses CLI args + launches Isaac at import time (that would
crash play.py). It also applies handle_deprecated_rsl_rl_cfg, the same schema
fix train.py needs for rsl-rl-lib 5.x.
"""

from __future__ import annotations
import argparse, os, sys

# Make the repo root importable regardless of launch method.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Roll out a trained Leo Rover policy.")
parser.add_argument("--task", default="Isaac-LeoRover-Mars-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--video", action="store_true",
                    help="record an mp4 (forces --enable_cameras)")
parser.add_argument("--video_length", type=int, default=600,
                    help="number of policy steps (0.2 s each) to record")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Offscreen video recording needs the camera/render pipeline up.
if args.video:
    args.enable_cameras = True
simulation_app = AppLauncher(args).app

# ---- imports that require the running app ----
import importlib.metadata as _metadata
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except Exception:
    handle_deprecated_rsl_rl_cfg = None


_TASKS = {
    "Isaac-LeoRover-Flat-v0":        (LeoRoverFlatEnv, LeoRoverFlatEnvCfg, LeoRoverFlatPPORunnerCfg),
    "Isaac-LeoRover-Mars-v0":        (LeoRoverMarsEnv, LeoRoverMarsEnvCfg, LeoRoverMarsPPORunnerCfg),
    "Isaac-LeoRover-Mars-Hybrid-v0": (LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg, LeoRoverMarsHybridPPORunnerCfg),
}


def main():
    env_cls, env_cfg_cls, runner_cfg_cls = _TASKS[args.task]
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = runner_cfg_cls()

    run_dir = os.path.join(os.path.dirname(args.checkpoint) or ".", "eval")
    os.makedirs(run_dir, exist_ok=True)

    # rgb_array render only when recording (keeps the no-video path light/robust).
    env = env_cls(cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    raw = env  # keep a handle to the unwrapped DirectRLEnv for diagnostics

    # --- PyBullet-schema CSV recorder (non-invasive step hook on the raw env) ---
    recorder = EpisodeMetricsRecorder(os.path.join(run_dir, "csv"), env)
    _orig_step = env.step

    def _step_with_record(action):
        obs, rew, terminated, truncated, extras = _orig_step(action)
        try:
            recorder.record_step(rew, terminated | truncated)
        except Exception:
            pass
        return obs, rew, terminated, truncated, extras

    env.step = _step_with_record

    # --- optional video recording (one clip, starting at step 0) ---
    if args.video:
        video_dir = os.path.join(run_dir, "videos")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
        print(f"[play] recording up to {args.video_length} steps -> {video_dir}")

    # Translate the agent cfg into the installed rsl-rl-lib schema (same step
    # train.py performs; without it OnPolicyRunner raises KeyError: 'class_name').
    if handle_deprecated_rsl_rl_cfg is not None:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))

    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=str(env.unwrapped.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()

    # --- motion diagnostics: are we moving, how fast, do we make progress? ---
    speed_sum = 0.0
    progress_peak = torch.zeros(args.num_envs, device=raw.device)
    n_steps = 0

    done_count = 0
    max_steps = args.video_length if args.video else 10**9
    while done_count < args.episodes and simulation_app.is_running() and n_steps < max_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # tap the raw env for ground truth (body-frame forward speed + progress)
            _, _, fwd_vel, _ = raw._kin()
            speed_sum += fwd_vel.abs().mean().item()
            progress_peak = torch.maximum(progress_peak, raw._path_progress())
        done_count += int(dones.sum().item()) if hasattr(dones, "sum") else 0
        n_steps += 1

    mean_speed = speed_sum / max(n_steps, 1)
    print("\n========== PLAY DIAGNOSTICS ==========")
    print(f"  steps rolled out      : {n_steps}")
    print(f"  episodes finished     : {recorder.episode_count}")
    print(f"  MEAN |forward speed|  : {mean_speed:.4f} m/s   "
          f"(full-throttle ground speed should be ~0.4 m/s for a healthy rover)")
    print(f"  peak path progress    : mean {progress_peak.mean().item():.1f}%  "
          f"max {progress_peak.max().item():.1f}%   (100% = reached goal)")
    print(f"  csv -> {recorder.csv_path}")
    print("======================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
