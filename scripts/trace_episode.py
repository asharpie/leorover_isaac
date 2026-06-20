#!/usr/bin/env python
# scripts/trace_episode.py
"""
Headless, NO-RENDERER episode tracer.

Records the rover's actual trajectory and the reference path for ONE env and
writes CSVs + a 2D top-down plot, WITHOUT ever starting the RTX renderer (which
segfaults on driver 595 in rtx.scenedb). This is how you "watch" an episode when
video recording can't run.

    scripts/run_lab.sh scripts/trace_episode.py \
        --task Isaac-LeoRover-Mars-Hybrid-v0 \
        --checkpoint logs/leo_rover_mars_hybrid/<run>/model_1400.pt \
        --steps 600

Outputs (in eval_trace/ next to the checkpoint):
    trace.csv : per-step x,y,yaw,fwd_vel,cur_idx,cte,progress for the traced env
    path.csv  : reference waypoints (x,y) and the goal
    trace.png : top-down rover-path-vs-reference + speed/progress/waypoint panels
                (only if matplotlib is present; otherwise scp the CSVs)
"""
from __future__ import annotations
import argparse, os, sys, csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-LeoRover-Mars-Hybrid-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--env", type=int, default=0, help="which env index to trace")

try:
    from isaaclab.app import AppLauncher
except Exception:
    from omni.isaac.lab.app import AppLauncher  # Isaac Sim 4.5
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Force HEADLESS + no cameras so AppLauncher loads the physics-only kit
# (isaaclab.python.headless.kit). The full/GUI kit starts the RTX renderer, which
# segfaults on driver 595 in rtx.scenedb. Headless physics runs fine on 595.
args.headless = True
args.enable_cameras = False
simulation_app = AppLauncher(args).app

import numpy as np
import torch
import importlib.metadata as _md

import leorover_isaac  # noqa: F401  (registers tasks)
from leorover_isaac.tasks.leo_rover_agents import (
    LeoRoverFlatPPORunnerCfg, LeoRoverMarsPPORunnerCfg, LeoRoverMarsHybridPPORunnerCfg,
)
from leorover_isaac.envs.leo_rover_flat_env import LeoRoverFlatEnv, LeoRoverFlatEnvCfg
from leorover_isaac.envs.leo_rover_mars_env import LeoRoverMarsEnv, LeoRoverMarsEnvCfg
from leorover_isaac.envs.leo_rover_mars_hybrid_env import LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg
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


def main():
    env_cls, cfg_cls, runner_cfg_cls = _TASKS[args.task]
    cfg = cfg_cls(); cfg.scene.num_envs = args.num_envs
    env = env_cls(cfg=cfg, render_mode=None)
    raw = env

    agent_cfg = runner_cfg_cls()
    if handle_deprecated_rsl_rl_cfg is not None:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None,
                            device=str(wrapped.unwrapped.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

    obs, _ = wrapped.get_observations()
    e = args.env

    # reference path (env-local frame) + goal for the traced env, grabbed now
    K = int(raw._num_wp[e].item())
    wps = raw._wps[e, :K, :2].detach().cpu().numpy()
    goal = raw._goal_xy[e].detach().cpu().numpy()

    rows = []
    for t in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
        pos_local, yaw, fwd_vel, _ = raw._kin()
        cte, _ = raw._true_cte_and_along()
        prog = raw._path_progress()
        rows.append((t, float(pos_local[e, 0]), float(pos_local[e, 1]),
                     float(yaw[e]), float(fwd_vel[e]),
                     int(raw._cur_idx[e]), float(cte[e]), float(prog[e])))
        if bool(dones[e]):
            print(f"  [trace] env {e} episode ended at step {t}")
            break

    outdir = os.path.join(os.path.dirname(args.checkpoint) or ".", "eval_trace")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "trace.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "x", "y", "yaw", "fwd_vel", "cur_idx", "cte", "progress"])
        w.writerows(rows)
    with open(os.path.join(outdir, "path.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wp_x", "wp_y"]); w.writerows(wps.tolist())
        w.writerow([f"goal:{goal[0]:.4f}", f"{goal[1]:.4f}"])

    tr = np.array([[r[1], r[2]] for r in rows])
    net = float(np.linalg.norm(tr[-1] - tr[0])) if len(tr) > 1 else 0.0
    driven = float(np.sum(np.linalg.norm(np.diff(tr, axis=0), axis=1))) if len(tr) > 1 else 0.0
    print("\n================= TRACE SUMMARY (env %d) =================" % e)
    print(f"  steps traced         : {len(rows)}")
    print(f"  start xy             : ({tr[0,0]:.2f}, {tr[0,1]:.2f})")
    print(f"  end   xy             : ({tr[-1,0]:.2f}, {tr[-1,1]:.2f})")
    print(f"  net displacement     : {net:.3f} m")
    print(f"  total path driven    : {driven:.3f} m")
    print(f"  waypoint index       : 0 -> {rows[-1][5]}  (of {K-1})")
    print(f"  final / max progress : {rows[-1][7]:.1f}%  /  {max(r[7] for r in rows):.1f}%")
    print(f"  mean |fwd_vel|        : {np.mean([abs(r[4]) for r in rows]):.4f} m/s "
          f"(stagnation threshold 0.02)")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        ax[0].plot(wps[:, 0], wps[:, 1], "-o", color="#999", ms=3, label="reference path")
        sc = ax[0].scatter(tr[:, 0], tr[:, 1], c=np.arange(len(tr)), cmap="viridis", s=10, label="rover")
        ax[0].scatter([tr[0, 0]], [tr[0, 1]], c="lime", s=120, marker="*", zorder=5, label="start")
        ax[0].scatter([goal[0]], [goal[1]], c="red", s=110, marker="X", zorder=5, label="goal")
        ax[0].set_aspect("equal"); ax[0].grid(alpha=.3); ax[0].legend()
        ax[0].set_title("top-down: rover trajectory vs reference path")
        plt.colorbar(sc, ax=ax[0], label="step")
        st = [r[0] for r in rows]
        ax[1].plot(st, [r[4] for r in rows], label="fwd_vel (m/s)")
        ax[1].plot(st, [r[7] / 100.0 for r in rows], label="progress (fraction)")
        ax[1].plot(st, [r[5] for r in rows], label="waypoint idx")
        ax[1].axhline(0.02, ls=":", c="r", label="stagnation thresh")
        ax[1].grid(alpha=.3); ax[1].legend(); ax[1].set_xlabel("step")
        ax[1].set_title("speed / progress / waypoint over time")
        png = os.path.join(outdir, "trace.png")
        plt.tight_layout(); plt.savefig(png, dpi=110)
        print(f"  PLOT                 : {png}")
    except Exception as ex:
        print(f"  (plot skipped: {ex}; scp trace.csv + path.csv and Claude will plot them)")
    print("=========================================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
