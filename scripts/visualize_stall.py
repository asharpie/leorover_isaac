#!/usr/bin/env python
# scripts/visualize_stall.py
"""
Renderer-FREE visualization of a single stalled rover.

The RTX/Omniverse viewer segfaults on this box's driver, so instead of rendering
pixels we reconstruct the rover GEOMETRICALLY from the physics body poses (which
PhysX always exposes) and draw it with matplotlib. Output PNG has 4 panels:

  (1) every wheel's CENTER height above the ground vs step  -> a wheel sitting at
      z > wheel_radius is OFF the ground (the rover is balancing on the others)
  (2) SIDE view (x-z): wheels as circles + base dot, overlaid through the launch
      -> front/back wheels at different heights = PITCH
  (3) FRONT view (y-z): same -> left/right wheels at different heights = ROLL
  (4) TOP view (x-y): base + per-wheel ground tracks

It auto-picks a parked rover (low final progress + ~zero speed) and also prints
each wheel's height range as TEXT, so even without opening the PNG you can see
which wheels left the ground.

Run (no checkpoint auto-find; pass it):
  scripts/run_lab.sh scripts/visualize_stall.py \
      --checkpoint logs/leo_rover_mars_hybrid/<run>/model_XXXX.pt --terrain 0
"""
from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-LeoRover-Mars-Hybrid-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=200, help="launch window to record (parked rovers jam by ~step 40)")
parser.add_argument("--terrain", type=float, default=0.0)
parser.add_argument("--env", type=int, default=-1, help="force a specific env index; -1 = auto-pick a parked one")
parser.add_argument("--zero-residual", "--lqr", dest="zero_residual", action="store_true")
try:
    from isaaclab.app import AppLauncher
except Exception:
    from omni.isaac.lab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
simulation_app = AppLauncher(args).app

import numpy as np
import torch
import importlib.metadata as _md
import leorover_isaac  # noqa: F401
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
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except Exception:
    handle_deprecated_rsl_rl_cfg = None

_TASKS = {
    "Isaac-LeoRover-Flat-v0":        (LeoRoverFlatEnv, LeoRoverFlatEnvCfg, LeoRoverFlatPPORunnerCfg),
    "Isaac-LeoRover-Mars-v0":        (LeoRoverMarsEnv, LeoRoverMarsEnvCfg, LeoRoverMarsPPORunnerCfg),
    "Isaac-LeoRover-Mars-Hybrid-v0": (LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg, LeoRoverMarsHybridPPORunnerCfg),
}

WHEEL_RADIUS = 0.0625


def main():
    env_cls, cfg_cls, runner_cfg_cls = _TASKS[args.task]
    cfg = cfg_cls(); cfg.scene.num_envs = args.num_envs
    if getattr(args, "device", None):
        cfg.sim.device = args.device
    print(f"[viz] creating env (terrain pinned at {args.terrain:.0f}%)...", flush=True)
    env = env_cls(cfg=cfg, render_mode=None)
    raw = env

    rows_total = max(int(raw._t_rows), 1); denom = max(rows_total - 1, 1)
    row = int(min(denom, max(0, round(args.terrain / 100.0 * denom))))
    raw._adr = None
    raw._eval_levels = torch.tensor([row], device=raw.device, dtype=torch.long)
    try:
        raw.reset()
    except Exception as e:
        print(f"[viz] (pre-roll reset skipped: {e})", flush=True)

    agent_cfg = runner_cfg_cls()
    if handle_deprecated_rsl_rl_cfg is not None:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=str(wrapped.unwrapped.device))
    print(f"[viz] loading {os.path.basename(args.checkpoint)}...", flush=True)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    obs, _ = wrapped.get_observations()

    # --- identify base + wheel body indices by name ---
    try:
        names = list(raw.robot.body_names)
    except Exception:
        names = list(raw.robot.data.body_names)
    wheel_ids = [i for i, n in enumerate(names) if "wheel" in n.lower()]
    base_id = next((i for i, n in enumerate(names) if "base" in n.lower()), 0)
    print(f"[viz] bodies ({len(names)}): {names}")
    print(f"[viz] base = idx {base_id} ({names[base_id]}); wheels = {[names[i] for i in wheel_ids]}", flush=True)

    origin = raw._origins().detach().cpu().numpy()  # [n,3] per-env terrain origin

    P, PROG, VEL, CMDV, CMDW = [], [], [], [], []
    print(f"[viz] recording body poses for {args.steps} steps...", flush=True)
    for t in range(args.steps):
        with torch.inference_mode():
            _, _, fwd_vel, _ = raw._kin()
            prog = raw._path_progress()
            base = raw._last_baseline
            bpos = raw.robot.data.body_pos_w.detach().cpu().numpy()   # [n, nb, 3] world
            P.append(bpos)
            PROG.append(prog.detach().cpu().numpy())
            VEL.append(fwd_vel.detach().cpu().numpy())
            CMDV.append(base[:, 0].detach().cpu().numpy())
            CMDW.append(base[:, 1].detach().cpu().numpy())
            actions = policy(obs)
            if args.zero_residual:
                actions = torch.zeros_like(actions)
            obs, _, dones, _ = wrapped.step(actions)
    P = np.stack(P)        # [T, n, nb, 3]
    PROG = np.stack(PROG)  # [T, n]
    VEL = np.stack(VEL)    # [T, n]

    # --- pick the rover to draw ---
    finalprog = PROG[-1]
    meanspd = np.abs(VEL[-50:]).mean(0)
    if args.env >= 0:
        e = args.env
    else:
        parked = np.where((finalprog < 6.0) & (meanspd < 0.01))[0]
        e = int(parked[0]) if len(parked) else int(np.argsort(finalprog)[0])
    print(f"[viz] drawing env {e}: final prog {finalprog[e]:.2f}%, mean speed {meanspd[e]:.4f} m/s", flush=True)

    pe = P[:, e, :, :] - origin[e][None, None, :]   # ground-relative [T, nb, 3]
    Tn = pe.shape[0]
    wz = pe[:, wheel_ids, 2]    # [T, nW] wheel center heights
    bz = pe[:, base_id, 2]
    wlbl = [names[i].replace("_link", "").replace("wheel_", "") for i in wheel_ids]

    # --- TEXT readout (works even without opening the PNG) ---
    print(f"\n[viz] env {e} wheel CENTER height above ground (rest = {WHEEL_RADIUS}):")
    for j, lab in enumerate(wlbl):
        off = "  <-- LIFTS OFF" if wz[:, j].max() > WHEEL_RADIUS + 0.02 else ""
        print(f"   {lab:4s}  min {wz[:,j].min():.3f}  max {wz[:,j].max():.3f}  final {wz[-1,j]:.3f}{off}")
    print(f"   base  min {bz.min():.3f}  max {bz.max():.3f}  final {bz[-1]:.3f}")
    print(f"   final wheel-height spread = {wz[-1].max()-wz[-1].min():.3f} m "
          f"(0 = flat/level, large = tilted onto a subset of wheels)", flush=True)

    # CONTRAST: a SUCCESS rover should sit ~LEVEL (small spread). If the parked rover's
    # spread is much larger, the parked rovers are rocking/tilting on the triangulated
    # terrain mesh (a rigid rover cannot rest tilted on a truly flat floor).
    succ = np.where(finalprog > 50.0)[0]
    if len(succ):
        es = int(succ[0])
        pes = P[:, es, :, :] - origin[es][None, None, :]
        wzs = pes[:, wheel_ids, 2]
        sp_p = float(wz[-1].max() - wz[-1].min())
        sp_s = float(wzs[-1].max() - wzs[-1].min())
        print(f"\n[viz] CONTRAST  parked env {e} spread {sp_p:.3f} m   vs   "
              f"success env {es} (prog {finalprog[es]:.0f}%) spread {sp_s:.3f} m")
        print(f"[viz]   success wheel heights {[round(float(wzs[-1,j]),3) for j in range(len(wheel_ids))]}")
        print(f"[viz]   => if parked >> success, the rover is rocking/tilting on the terrain MESH "
              f"(PyBullet used a smooth heightfield).", flush=True)

    # --- matplotlib reconstruction (no renderer) ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 10))
    frames = np.linspace(0, Tn - 1, 6).astype(int)
    cols = plt.cm.viridis(np.linspace(0, 1, len(frames)))

    ax1 = fig.add_subplot(2, 2, 1)
    for j, lab in enumerate(wlbl):
        ax1.plot(wz[:, j], label=f"{lab}")
    ax1.plot(bz, "k--", lw=1, label="base")
    ax1.axhline(WHEEL_RADIUS, color="gray", ls=":", label=f"rest z={WHEEL_RADIUS}")
    ax1.set_title("(1) wheel center height vs step  (z>rest = wheel off the ground)")
    ax1.set_xlabel("step"); ax1.set_ylabel("height (m)"); ax1.legend(fontsize=8); ax1.grid(alpha=.3)

    ax2 = fig.add_subplot(2, 2, 2)
    for ti, c in zip(frames, cols):
        for x, z in zip(pe[ti, wheel_ids, 0], pe[ti, wheel_ids, 2]):
            ax2.add_patch(plt.Circle((x, z), WHEEL_RADIUS, fill=False, color=c, alpha=.85))
        ax2.plot([pe[ti, base_id, 0]], [pe[ti, base_id, 2]], "o", color=c, ms=5, label=f"step {ti}")
    ax2.axhline(0, color="saddlebrown", lw=2)
    ax2.set_title("(2) SIDE (x-z): front/back wheel height gap = PITCH")
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("z (m)"); ax2.legend(fontsize=7); ax2.set_aspect("equal"); ax2.grid(alpha=.3)

    ax3 = fig.add_subplot(2, 2, 3)
    for ti, c in zip(frames, cols):
        for y, z in zip(pe[ti, wheel_ids, 1], pe[ti, wheel_ids, 2]):
            ax3.add_patch(plt.Circle((y, z), WHEEL_RADIUS, fill=False, color=c, alpha=.85))
        ax3.plot([pe[ti, base_id, 1]], [pe[ti, base_id, 2]], "o", color=c, ms=5)
    ax3.axhline(0, color="saddlebrown", lw=2)
    ax3.set_title("(3) FRONT (y-z): left/right wheel height gap = ROLL")
    ax3.set_xlabel("y (m)"); ax3.set_ylabel("z (m)"); ax3.set_aspect("equal"); ax3.grid(alpha=.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(pe[:, base_id, 0], pe[:, base_id, 1], "-k", lw=1, label="base")
    for j, lab in enumerate(wlbl):
        ax4.plot(pe[:, wheel_ids[j], 0], pe[:, wheel_ids[j], 1], lw=.8, label=lab)
    ax4.scatter([pe[0, base_id, 0]], [pe[0, base_id, 1]], c="lime", s=70, marker="*", zorder=5)
    ax4.set_title("(4) TOP (x-y): base + wheel ground tracks")
    ax4.set_xlabel("x (m)"); ax4.set_ylabel("y (m)"); ax4.legend(fontsize=7); ax4.set_aspect("equal"); ax4.grid(alpha=.3)

    fig.suptitle(f"Stalled rover env {e} | terrain {args.terrain:.0f}% | final prog {finalprog[e]:.1f}% "
                 f"| residual {'OFF' if args.zero_residual else 'ON'}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    outdir = os.path.join(os.path.dirname(args.checkpoint) or ".", "stall_viz")
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, "stall_viz.png")
    plt.savefig(png, dpi=120)
    print(f"\n[viz] wrote {png}", flush=True)
    print(f"[viz] pull it (run ON THE LAPTOP):  scp irl@10.115.102.210:{os.path.abspath(png)} "
          f'"C:\\Users\\Aaron\\Downloads\\leorover_isaac\\"', flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
