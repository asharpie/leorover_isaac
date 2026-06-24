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
    try:
        from isaaclab.app import AppLauncher
    except Exception:
        from omni.isaac.lab.app import AppLauncher  # Isaac Sim 4.5
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

import importlib.metadata as _metadata
from rsl_rl.runners import OnPolicyRunner
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
except Exception:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper  # Isaac Sim 4.5
try:
    # Converts the agent cfg into the schema the installed rsl-rl-lib expects
    # (the modular actor/critic format in rsl-rl-lib 5.x). REQUIRED — Isaac Lab's
    # own train.py calls this; without it OnPolicyRunner gets a cfg whose
    # actor/critic dicts are missing 'class_name'.
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except Exception:
    handle_deprecated_rsl_rl_cfg = None


_TASKS = {
    "Isaac-LeoRover-Flat-v0":        (LeoRoverFlatEnv, LeoRoverFlatEnvCfg, LeoRoverFlatPPORunnerCfg),
    "Isaac-LeoRover-Mars-v0":        (LeoRoverMarsEnv, LeoRoverMarsEnvCfg, LeoRoverMarsPPORunnerCfg),
    "Isaac-LeoRover-Mars-Hybrid-v0": (LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg, LeoRoverMarsHybridPPORunnerCfg),
}


@torch.inference_mode()
def _adr_eval(wrapped, raw, det_policy, steps):
    """Run the policy DETERMINISTICALLY (no exploration noise) over the current
    [0, ADR ceiling] terrain band and return (success_rate, mean_cte). Terrain is
    pinned via _eval_levels and CSV logging suppressed via _skip_record; both are
    restored afterward. This is the noise-free competence signal the curriculum
    needs -- the stochastic rollout success understates it badly enough to freeze
    the curriculum, especially for pure PPO."""
    crow = raw.adr_max_level()
    raw._eval_levels = torch.arange(0, crow + 1, device=raw.device, dtype=torch.long)
    raw._skip_record = True
    try:
        raw.reset()
        obs, _ = wrapped.get_observations()
        n_done = torch.zeros((), device=raw.device)
        n_succ = torch.zeros((), device=raw.device)
        cte_sum = torch.zeros(raw.num_envs, device=raw.device)
        cnt = 0
        for _ in range(int(steps)):
            obs, _, dones, _ = wrapped.step(det_policy(obs))
            d = dones.bool()
            # CRITICAL: read the goal flag that _get_dones snapshotted BEFORE Isaac's
            # auto-reset (_log_goal), NOT _is_goal_reached() -- the latter runs after the
            # successful env has already respawned at the start, so it reads False and the
            # eval measures ~0% success, which is what kept the curriculum pinned at 10%.
            goal = getattr(raw, "_log_goal", None)
            if goal is not None:
                n_succ += (goal.bool() & d).sum()
            n_done += d.sum()
            c, _ = raw._true_cte_and_along()
            cte_sum += c.abs()
            cnt += 1
        sr = float((n_succ / torch.clamp(n_done, min=1.0)).item())
        cte = float((cte_sum / max(cnt, 1)).mean().item())
        return sr, cte
    finally:
        raw._eval_levels = None
        raw._skip_record = False


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

    # Translate the agent cfg into the installed rsl-rl-lib schema (the step
    # Isaac Lab's own train.py performs). Without this, rsl-rl-lib 5.x raises
    # KeyError: 'class_name' when constructing the actor/critic.
    if handle_deprecated_rsl_rl_cfg is not None:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))

    # Wrap for rsl_rl and run PPO (clip_actions matches the official workflow)
    raw_env = env  # the underlying DirectRLEnv, before the rsl_rl wrapper
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=run_dir,
                            device=getattr(agent_cfg, "device", None) or str(env.unwrapped.device))
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", "leorover_isaac")

    print(f"[train] task={args.task} num_envs={args.num_envs} -> logging to {run_dir}")

    total_iters = agent_cfg.max_iterations
    # Deterministic-eval-driven ADR: when the env is in external mode, the noisy
    # rollout success must not move the curriculum -- we drive it from a periodic
    # noise-free eval instead (see config.ADR_DETERMINISTIC_EVAL).
    adr_eval_on = getattr(raw_env, "_adr_external", False) and getattr(raw_env, "_adr", None) is not None
    if not adr_eval_on:
        runner.learn(num_learning_iterations=total_iters, init_at_random_ep_len=True)
    else:
        # defaults mirror config.ADR_EVAL_EVERY_ITERS / ADR_EVAL_STEPS; kept as literals here
        # so train.py never does a bare `import config` (which collides with Isaac's bundled
        # cv2/config.py once Isaac has been imported). Override via the env vars below.
        every = int(os.environ.get("LEOROVER_ADR_EVAL_EVERY", "100"))
        eval_steps = int(os.environ.get("LEOROVER_ADR_EVAL_STEPS", "1500"))
        print(f"[train] ADR is DETERMINISTIC-EVAL driven: eval every {every} iters, "
              f"{eval_steps} steps/eval. Stochastic rollout success will NOT move the curriculum.")
        det_policy = runner.get_inference_policy(device=str(env.unwrapped.device))
        done_iters, first = 0, True
        while done_iters < total_iters:
            n = min(every, total_iters - done_iters)
            runner.learn(num_learning_iterations=n, init_at_random_ep_len=first)
            first = False
            done_iters += n
            runner.save(os.path.join(run_dir, f"model_{done_iters}.pt"))
            try:
                sr, cte = _adr_eval(env, raw_env, det_policy, eval_steps)
                ev = raw_env.apply_adr_eval(sr, cte)
                print(f"[ADR-eval] iter {done_iters}: det success {sr:.0%}, CTE {cte:.3f} "
                      f"-> {ev}, terrain_max now {raw_env._adr.terrain_max:.0f}%", flush=True)
            except Exception as exc:  # never let the eval break training
                print(f"[ADR-eval] skipped at iter {done_iters}: {exc}", flush=True)

    runner.save(os.path.join(run_dir, "model_final.pt"))
    print(f"[train] done. checkpoints + episode_metrics.csv in {run_dir}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
