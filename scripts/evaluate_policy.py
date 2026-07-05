#!/usr/bin/env python
# scripts/evaluate_policy.py
"""
Held-out, DETERMINISTIC policy evaluation across a terrain-difficulty sweep.

This is the Isaac-Lab equivalent of the PyBullet evaluation: it takes a trained
policy and runs it over a *plethora of terrain levels x random paths*, logging
one row per finished episode into a CSV in the IDENTICAL episode_metrics schema
that training writes. You then compare algorithms by reading those CSVs.

Why a separate script (not the training CSV):
  * Training logs the STOCHASTIC, still-learning policy. Exploration noise
    (action std ~0.7) shoves good rovers off-path, so the training CSV badly
    understates the deployable policy. Here we run the DETERMINISTIC policy
    (the inference mean, no sampling) -- what you would actually deploy.
  * Training terrain is driven by the ADR curriculum, which collapses to easy
    terrain when success dips. Here we PIN terrain to a fixed sweep so every
    difficulty level gets even coverage (env._eval_levels), giving a clean
    success-vs-terrain curve.

Three algorithms, one script:
  hybrid : --task Isaac-LeoRover-Mars-Hybrid-v0  --checkpoint <hybrid ckpt>
  lqr    : --task Isaac-LeoRover-Mars-Hybrid-v0  --checkpoint <any ckpt>  --zero-residual
           (forces the PPO residual to 0 -> evaluates the bare LQR baseline)
  ppo    : --task Isaac-LeoRover-Mars-v0          --checkpoint <ppo ckpt>

Example (run via the Isaac launcher):
    scripts/run_lab.sh scripts/evaluate_policy.py \
        --task Isaac-LeoRover-Mars-Hybrid-v0 \
        --checkpoint logs/leo_rover_mars_hybrid/<run>/model_600.pt \
        --levels 10,20,30,40,50,60,70,80 \
        --num_envs 1024 --steps 6000 \
        --out evals/hybrid_<stamp>.csv

The output CSV has the same 18 columns as episode_metrics.csv, so leo_report.py,
eval_report.py, and evaluate_training.py all read it unchanged.
"""
from __future__ import annotations
import argparse, os, sys, csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Deterministic terrain-sweep policy evaluation.")
parser.add_argument("--task", default="Isaac-LeoRover-Mars-Hybrid-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=6000,
                    help="total sim steps; episodes accumulate across all envs (~num_envs*steps/1000 episodes)")
parser.add_argument("--levels", default="0,20,40,60,80,100",
                    help="comma-separated terrain intensities (%% of max) to sweep; snapped to the "
                         "nearest terrain rows. Default is row-exact for the 6-row bank (0..100%%) "
                         "so the sweep includes the flat row AND the max row. (The old "
                         "10,20,...,80 default silently snapped 8 values onto rows 0-4 and never "
                         "evaluated 100%% terrain.)")
parser.add_argument("--zero-residual", "--lqr", dest="zero_residual", action="store_true",
                    help="force the PPO residual to 0 = evaluate the pure LQR baseline")
parser.add_argument("--out", default="",
                    help="output CSV path (default: evals/<task-short>_<stamp>.csv under the repo)")

try:
    from isaaclab.app import AppLauncher
except Exception:
    from omni.isaac.lab.app import AppLauncher  # Isaac Sim 4.5
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Headless physics-only kit (the RTX renderer segfaults on driver 595); we never
# render here -- everything we log is physics state.
args.headless = True
args.enable_cameras = False
simulation_app = AppLauncher(args).app

import numpy as np
import torch
import importlib.metadata as _md
from datetime import datetime

# Terrain difficulty is a 0-100% scale (config.ADR_TERRAIN_MAX_LIMIT). Kept as a literal
# so this script never does a bare `import config`, which once Isaac is loaded resolves to
# Isaac's bundled cv2/config.py and crashes.
_TERRAIN_MAX_PCT = 100.0
import leorover_isaac  # noqa: F401  (registers tasks)
from leorover_isaac.tasks.leo_rover_agents import (
    LeoRoverFlatPPORunnerCfg, LeoRoverMarsPPORunnerCfg, LeoRoverMarsHybridPPORunnerCfg,
)
from leorover_isaac.envs.leo_rover_flat_env import LeoRoverFlatEnv, LeoRoverFlatEnvCfg
from leorover_isaac.envs.leo_rover_mars_env import LeoRoverMarsEnv, LeoRoverMarsEnvCfg
from leorover_isaac.envs.leo_rover_mars_hybrid_env import LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg
from leorover_isaac.utils.recorder import EpisodeMetricsRecorder
from rsl_rl.runners import OnPolicyRunner
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
except Exception:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper  # Isaac Sim 4.5
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except Exception:
    handle_deprecated_rsl_rl_cfg = None

_TASKS = {
    "Isaac-LeoRover-Flat-v0":        (LeoRoverFlatEnv, LeoRoverFlatEnvCfg, LeoRoverFlatPPORunnerCfg),
    "Isaac-LeoRover-Mars-v0":        (LeoRoverMarsEnv, LeoRoverMarsEnvCfg, LeoRoverMarsPPORunnerCfg),
    "Isaac-LeoRover-Mars-Hybrid-v0": (LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg, LeoRoverMarsHybridPPORunnerCfg),
}
_SHORT = {"Isaac-LeoRover-Mars-Hybrid-v0": "hybrid", "Isaac-LeoRover-Mars-v0": "ppo", "Isaac-LeoRover-Flat-v0": "flat"}


def _mark(msg):
    print(f"[eval] {msg}", flush=True)


def main():
    mode = "lqr" if args.zero_residual else _SHORT.get(args.task, "policy")
    _mark(f"task={args.task} mode={mode} envs={args.num_envs} steps={args.steps} levels={args.levels}")

    env_cls, cfg_cls, runner_cfg_cls = _TASKS[args.task]
    cfg = cfg_cls(); cfg.scene.num_envs = args.num_envs
    if getattr(args, "device", None):
        cfg.sim.device = args.device
    _mark("creating env (scene build + terrain cook + robot spawn)...")
    env = env_cls(cfg=cfg, render_mode=None)
    raw = env
    _mark("env created OK")

    # --- pin terrain to the requested sweep (rows), disable ADR adaptation ---
    rows_total = max(int(raw._t_rows), 1)
    denom = max(rows_total - 1, 1)
    want = [float(x) for x in str(args.levels).split(",") if x.strip() != ""]
    rows = sorted({min(denom, max(0, round(p / _TERRAIN_MAX_PCT * denom))) for p in want})
    actual = [round(r / denom * _TERRAIN_MAX_PCT, 1) for r in rows]
    raw._adr = None  # stop the curriculum from moving / printing during eval
    raw._eval_levels = torch.tensor(rows, device=raw.device, dtype=torch.long)
    _mark(f"terrain rows {rows} -> intensities {actual}% (of {_TERRAIN_MAX_PCT:.0f}% max)")
    if len(rows) < len(set(want)):
        _mark(f"NOTE: {len(set(want))} requested levels snapped onto only {len(rows)} distinct "
              f"terrain rows (the bank has {rows_total}); use row-exact percents "
              f"(multiples of {100.0 / max(denom, 1):.0f}) to avoid collisions")
    # re-place every env onto an eval level before we attach the recorder. If the
    # reset API balks, fall back to natural auto-resets (only the first ~1 episode
    # per env then starts at the construction level instead of an eval level).
    try:
        raw.reset()
    except Exception as e:
        _mark(f"(pre-roll reset skipped: {e}; eval levels apply from each env's first auto-reset)")

    # --- output CSV (named, so it never clobbers a training episode_metrics.csv) ---
    out = args.out
    if not out:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join("evals", f"{mode}_{stamp}.csv")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # --- attach the PyBullet-schema recorder on the RAW step, exactly like train.py
    # (sees the same reward + terminated|truncated, and the pre-auto-reset progress/
    # goal snapshots). ---
    recorder = EpisodeMetricsRecorder(os.path.dirname(out), env, filename=os.path.basename(out))
    _orig_step = env.step

    def _step_with_record(action):
        obs_, rew_, term_, trunc_, extras_ = _orig_step(action)
        try:
            recorder.record_step(rew_, term_ | trunc_)
        except Exception as e:
            if os.environ.get("LEOROVER_DEBUG"):
                print(f"[eval][recorder] {e}", flush=True)
        return obs_, rew_, term_, trunc_, extras_

    env.step = _step_with_record
    _mark(f"writing -> {out}")

    # --- load the policy (deterministic inference mean; no exploration noise) ---
    agent_cfg = runner_cfg_cls()
    if handle_deprecated_rsl_rl_cfg is not None:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None,
                            device=str(wrapped.unwrapped.device))
    _mark(f"loading checkpoint {os.path.basename(args.checkpoint)}...")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

    obs, _ = wrapped.get_observations()
    _mark(f"evaluating {args.steps} steps (deterministic)...")
    for t in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
            if args.zero_residual:
                actions = torch.zeros_like(actions)   # pure LQR baseline
            obs, _, _, _ = wrapped.step(actions)       # recorder fires inside the raw step
        if (t + 1) % 1000 == 0:
            _mark(f"  step {t+1}/{args.steps}  episodes logged: {recorder.episode_count}")

    env.close()
    simulation_app.close()

    # --- quick per-level summary (full comparison via eval_report.py) ---
    _summarize(out)


def _summarize(path):
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        _mark(f"(could not re-read {path}: {e})"); return
    if not rows:
        _mark("no episodes were logged (try more --steps)"); return
    def col(r, k):
        try: return float(r[k])
        except Exception: return 0.0
    succ = [col(r, "success") for r in rows]
    prog = [col(r, "path_progress") for r in rows]
    terr = [col(r, "terrain_intensity") for r in rows]
    n = len(rows)
    print("\n================= EVAL SUMMARY =================")
    print(f"  file        : {path}")
    print(f"  episodes    : {n}")
    print(f"  SUCCESS     : {100.0*sum(succ)/n:5.1f}%   (deterministic, held-out)")
    print(f"  mean prog   : {sum(prog)/n:5.1f}%   median {sorted(prog)[n//2]:.1f}%")
    print("  by terrain level:")
    levels = sorted(set(round(x, 1) for x in terr))
    print(f"    {'terrain%':>9} {'episodes':>9} {'success%':>9} {'meanprog%':>10}")
    for L in levels:
        idx = [i for i in range(n) if round(terr[i], 1) == L]
        if not idx: continue
        s = 100.0 * sum(succ[i] for i in idx) / len(idx)
        p = sum(prog[i] for i in idx) / len(idx)
        print(f"    {L:>9.1f} {len(idx):>9} {s:>9.1f} {p:>10.1f}")
    print("===============================================\n")
    print(f"Compare algorithms with:  python3 scripts/eval_report.py {mode_label(path)}={path} ...")


def mode_label(path):
    base = os.path.basename(path)
    return base.split("_")[0] if "_" in base else "policy"


if __name__ == "__main__":
    main()
