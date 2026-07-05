#!/usr/bin/env python
# scripts/paired_eval.py
"""
PAIRED, scenario-locked evaluation -- the matched-condition protocol the paper claims.

Every controller (hybrid, pure LQR, pure PPO) is evaluated over the SAME ordered list of
scenarios, where each scenario fixes (path, terrain patch, start pose) deterministically and
friction is held at one controlled value for the whole run. Because the scenario id fully
determines the episode, controller A's scenario j and controller B's scenario j face a
byte-identical world -- so any difference in outcome is the controller alone, and the rows
join on scenario_id for paired statistics (see paired_stats.py). This removes the
luck-of-the-draw skew of running each algorithm over its own random draws.

One controller per process (one Isaac sim context = the safe path, same as evaluate_policy.py).
The three processes share pairing via a scenario file:

    # 1) hybrid builds + saves the scenario list; the other two REUSE it (same --scenarios path)
    run_lab.sh scripts/paired_eval.py --task Isaac-LeoRover-Mars-Hybrid-v0 \
        --checkpoint <hybrid.pt> --paths discrete --levels 10,30,50,70 --friction 1.0 \
        --scenarios evals/paired/scen.npz --out evals/paired/hybrid.csv
    run_lab.sh scripts/paired_eval.py --task Isaac-LeoRover-Mars-Hybrid-v0 \
        --checkpoint <hybrid.pt> --zero-residual \
        --scenarios evals/paired/scen.npz --out evals/paired/lqr.csv --friction 1.0
    run_lab.sh scripts/paired_eval.py --task Isaac-LeoRover-Mars-v0 \
        --checkpoint <ppo.pt> \
        --scenarios evals/paired/scen.npz --out evals/paired/ppo.csv --friction 1.0

    # 2) join + statistics (runs anywhere, pure stdlib)
    python3 scripts/paired_stats.py hybrid=evals/paired/hybrid.csv \
        lqr=evals/paired/lqr.csv ppo=evals/paired/ppo.csv

IMPORTANT: pass the IDENTICAL --friction to all three (it is baked at env import via
LEOROVER_FRICTION); the shared --scenarios file guarantees identical path+terrain+pose.
"""
from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Paired, scenario-locked controller evaluation.")
parser.add_argument("--task", default="Isaac-LeoRover-Mars-Hybrid-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--paths", choices=["discrete", "random"], default="discrete",
                    help="discrete = the 9 zig-zag/curved/polygon templates (path-geometry study); "
                         "random = the curved-path bank (slope-conditioned study)")
parser.add_argument("--num_scenarios", type=int, default=90000,
                    help="number of paired episodes (evenly spread over paths x terrain levels); "
                         "GPU-parallel eval is cheap, so a large N gives tight paired statistics")
parser.add_argument("--levels", default="0,20,40,60,80,100",
                    help="terrain intensities (%% of max) to draw patches from; snapped to the "
                         "nearest terrain rows. Default is row-exact for the 6-row bank. (The old "
                         "10,30,50,70 default snapped onto rows {0,2,4} -- three real levels "
                         "masquerading as four.)")
parser.add_argument("--friction", type=float, default=1.0,
                    help="FIXED wheel friction for the whole run; use the SAME value for all "
                         "three controllers (baked via LEOROVER_FRICTION at env import)")
parser.add_argument("--scenarios", default="",
                    help="path to a scenarios .npz; built + saved if missing, REUSED if present "
                         "(this is what makes the three runs paired)")
parser.add_argument("--zero-residual", "--lqr", dest="zero_residual", action="store_true",
                    help="force the PPO residual to 0 = evaluate the pure LQR baseline")
parser.add_argument("--seed", type=int, default=20260705, help="scenario-construction seed")
parser.add_argument("--max_steps", type=int, default=0,
                    help="hard sim-step cap (0 = auto: enough for every scenario to finish once)")
parser.add_argument("--out", default="", help="output CSV (default evals/paired/<mode>_<stamp>.csv)")

# Friction is read at env-module import, so set it BEFORE anything imports the env.
_pre_args, _ = parser.parse_known_args()
os.environ["LEOROVER_FRICTION"] = str(_pre_args.friction)

try:
    from isaaclab.app import AppLauncher
except Exception:
    from omni.isaac.lab.app import AppLauncher  # Isaac Sim 4.5
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
simulation_app = AppLauncher(args).app

import numpy as np
import torch
import importlib.metadata as _md
from datetime import datetime

_TERRAIN_MAX_PCT = 100.0
import leorover_isaac  # noqa: F401  (registers tasks)
from leorover_isaac.tasks.leo_rover_agents import (
    LeoRoverMarsPPORunnerCfg, LeoRoverMarsHybridPPORunnerCfg,
)
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
    "Isaac-LeoRover-Mars-v0":        (LeoRoverMarsEnv, LeoRoverMarsEnvCfg, LeoRoverMarsPPORunnerCfg),
    "Isaac-LeoRover-Mars-Hybrid-v0": (LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg, LeoRoverMarsHybridPPORunnerCfg),
}
_SHORT = {"Isaac-LeoRover-Mars-Hybrid-v0": "hybrid", "Isaac-LeoRover-Mars-v0": "ppo"}


def _mark(msg):
    print(f"[paired] {msg}", flush=True)


def _build_scenarios(n, bank_size, rows_avail, t_cols, seed):
    """Even coverage of paths x terrain levels; columns (variations) uniform random."""
    rng = np.random.default_rng(seed)
    reps = int(np.ceil(n / max(bank_size, 1)))
    path_idx = np.tile(np.arange(bank_size), reps)[:n]
    rng.shuffle(path_idx)
    row = rng.choice(np.asarray(rows_avail, dtype=np.int64), size=n)
    col = rng.integers(0, max(int(t_cols), 1), size=n)
    return path_idx.astype(np.int64), row.astype(np.int64), col.astype(np.int64)


def main():
    mode = "lqr" if args.zero_residual else _SHORT.get(args.task, "policy")
    _mark(f"task={args.task} mode={mode} paths={args.paths} friction={args.friction} envs={args.num_envs}")

    # if reusing a scenario file, adopt ITS path mode so path_idx indices stay valid
    # (a discrete npz has idx 0..8; a random npz has idx 0..bank-1 -- must not be clamped).
    if args.scenarios and os.path.isfile(args.scenarios):
        try:
            _pk = np.load(args.scenarios)
            if "paths" in _pk and str(_pk["paths"]) in ("discrete", "random"):
                args.paths = str(_pk["paths"])
        except Exception:
            pass
    env_cls, cfg_cls, runner_cfg_cls = _TASKS[args.task]
    cfg = cfg_cls()
    cfg.scene.num_envs = args.num_envs
    cfg.use_random_paths = (args.paths == "random")   # discrete -> the 9 templates
    if getattr(args, "device", None):
        cfg.sim.device = args.device
    _mark("creating env (scene build + terrain cook + robot spawn)...")
    env = env_cls(cfg=cfg, render_mode=None)
    raw = env
    raw._adr = None                 # no curriculum drift during eval
    raw._eval_levels = None
    _mark(f"env created OK  bank_size={raw._bank_size}  terrain rows={raw._t_rows} cols={raw._t_cols}")

    # --- scenario list: build+save if absent, else reuse (this is the pairing key) ---
    denom = max(int(raw._t_rows) - 1, 1)
    want = [float(x) for x in str(args.levels).split(",") if x.strip() != ""]
    rows_avail = sorted({min(denom, max(0, round(p / _TERRAIN_MAX_PCT * denom))) for p in want}) or [0]
    _mark(f"terrain rows {rows_avail} -> intensities "
          f"{[round(r / denom * _TERRAIN_MAX_PCT, 1) for r in rows_avail]}%")
    if len(rows_avail) < len(set(want)):
        _mark(f"NOTE: {len(set(want))} requested levels snapped onto only {len(rows_avail)} distinct "
              f"terrain rows; use multiples of {100.0 / max(denom, 1):.0f}%% to avoid collisions")
    if args.scenarios and os.path.isfile(args.scenarios):
        d = np.load(args.scenarios)
        path_idx, row, col = d["path_idx"], d["row"], d["col"]
        _mark(f"REUSING scenarios {args.scenarios}  (n={len(path_idx)}) -> paired with prior runs")
    else:
        path_idx, row, col = _build_scenarios(args.num_scenarios, raw._bank_size, rows_avail,
                                              raw._t_cols, args.seed)
        if args.scenarios:
            os.makedirs(os.path.dirname(os.path.abspath(args.scenarios)), exist_ok=True)
            np.savez(args.scenarios, path_idx=path_idx, row=row, col=col,
                     paths=args.paths, bank=raw._bank_size)
            _mark(f"BUILT + saved scenarios -> {args.scenarios}  (n={len(path_idx)})")
    S = int(len(path_idx))

    raw._eval_scenarios = {
        "path_idx": torch.as_tensor(path_idx, dtype=torch.long, device=raw.device),
        "row":      torch.as_tensor(row, dtype=torch.long, device=raw.device),
        "col":      torch.as_tensor(col, dtype=torch.long, device=raw.device),
    }
    raw._scenario_ptr = 0
    try:
        raw.reset()   # re-home every env onto scenarios 0..num_envs-1
    except Exception as e:
        _mark(f"(pre-roll reset skipped: {e})")

    # --- output CSV ---
    out = args.out
    if not out:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join("evals", "paired", f"{mode}_{stamp}.csv")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    recorder = EpisodeMetricsRecorder(os.path.dirname(out), env, filename=os.path.basename(out))
    _orig_step = env.step

    def _step_with_record(action):
        obs_, rew_, term_, trunc_, extras_ = _orig_step(action)
        try:
            recorder.record_step(rew_, term_ | trunc_)
        except Exception as e:
            if os.environ.get("LEOROVER_DEBUG"):
                print(f"[paired][recorder] {e}", flush=True)
        return obs_, rew_, term_, trunc_, extras_

    env.step = _step_with_record
    _mark(f"writing -> {out}")

    # --- policy (deterministic mean; zeroed for the pure-LQR pass) ---
    agent_cfg = runner_cfg_cls()
    if handle_deprecated_rsl_rl_cfg is not None:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=str(wrapped.unwrapped.device))
    _mark(f"loading checkpoint {os.path.basename(args.checkpoint)}...")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

    # auto step cap: enough passes for every scenario to reset+finish once, with margin.
    max_steps = args.max_steps if args.max_steps > 0 else int(np.ceil(S / args.num_envs) * 2400 + 4000)
    obs, _ = wrapped.get_observations()
    _mark(f"evaluating {S} scenarios (deterministic), step cap {max_steps}...")
    for t in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            if args.zero_residual:
                actions = torch.zeros_like(actions)
            obs, _, _, _ = wrapped.step(actions)
        if (t + 1) % 1000 == 0:
            _mark(f"  step {t+1}/{max_steps}  episodes logged: {recorder.episode_count}/{S}")
        if recorder.episode_count >= S:
            _mark(f"  all {S} scenarios completed at step {t+1}")
            break

    env.close()
    simulation_app.close()
    _mark(f"done -> {out}  ({recorder.episode_count} episodes)")


if __name__ == "__main__":
    main()
