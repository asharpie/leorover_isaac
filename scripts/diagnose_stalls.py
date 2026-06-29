#!/usr/bin/env python
# scripts/diagnose_stalls.py
"""
Diagnose WHY rovers stall.

Runs the policy DETERMINISTICALLY at a pinned terrain level, records a sample of
rovers step-by-step, and for the ones that get stuck it compares what the LQR
CONTROLLER COMMANDED against what the rover ACTUALLY DID. That comparison is the
diagnosis:

  * commanded |v| >> actual |v|   -> WEDGED   (it's pushing but not moving = physics/terrain)
  * commanded |v| ~ 0             -> IDLE     (the controller itself gave up / v_ref collapsed)
  * high cross-track error        -> OFF-PATH (oversteered, drove off, can't recover)
  * moving but below stagnation   -> SLOW     (creeping under the 0.02 kill line)

Renderer-free (safe on driver 595). Writes a top-down PNG of the stuck rovers and
a JSON of their full trajectories, and prints a classification summary.

Run via:
  scripts/run_lab.sh scripts/diagnose_stalls.py \
      --checkpoint logs/leo_rover_mars_hybrid/<run>/model_XXXX.pt
"""
from __future__ import annotations
import argparse, os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-LeoRover-Mars-Hybrid-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=1700, help="~one episode; stops early once all sampled rovers finish")
parser.add_argument("--sample", type=int, default=128, help="how many rovers to record in full detail")
parser.add_argument("--terrain", type=float, default=20.0, help="terrain intensity %% to pin (stalls are terrain-independent)")
parser.add_argument("--zero-residual", "--lqr", dest="zero_residual", action="store_true",
                    help="force residual to 0 -> diagnose the bare LQR baseline's stalls")

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


def main():
    env_cls, cfg_cls, runner_cfg_cls = _TASKS[args.task]
    cfg = cfg_cls(); cfg.scene.num_envs = args.num_envs
    if getattr(args, "device", None):
        cfg.sim.device = args.device
    print(f"[diag] creating env (terrain pinned at {args.terrain:.0f}%)...", flush=True)
    env = env_cls(cfg=cfg, render_mode=None)
    raw = env

    rows_total = max(int(raw._t_rows), 1); denom = max(rows_total - 1, 1)
    row = int(min(denom, max(0, round(args.terrain / 100.0 * denom))))
    raw._adr = None
    raw._eval_levels = torch.tensor([row], device=raw.device, dtype=torch.long)
    try:
        raw.reset()
    except Exception as e:
        print(f"[diag] (pre-roll reset skipped: {e})", flush=True)

    agent_cfg = runner_cfg_cls()
    if handle_deprecated_rsl_rl_cfg is not None:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=str(wrapped.unwrapped.device))
    print(f"[diag] loading {os.path.basename(args.checkpoint)}...", flush=True)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    obs, _ = wrapped.get_observations()

    K = min(args.sample, raw.num_envs)
    keys = ["x", "y", "vel", "cmd_v", "cmd_w", "res_v", "res_w", "cte", "prog", "wheel_spd", "tilt"]
    try:
        _wids = list(raw._left_ids) + list(raw._right_ids)
    except Exception:
        _wids = None
    rec = [{k: [] for k in keys} for _ in range(K)]
    done_flag = torch.zeros(K, dtype=torch.bool)
    refpaths = []
    for e in range(K):
        kk = int(raw._num_wp[e].item())
        refpaths.append(raw._wps[e, :kk, :2].detach().cpu().numpy().tolist())

    print(f"[diag] stepping up to {args.steps} steps, recording {K} rovers...", flush=True)
    for t in range(args.steps):
        with torch.inference_mode():
            pos, yaw, fwd_vel, _ = raw._kin()
            cte, _ = raw._true_cte_and_along()
            prog = raw._path_progress()
            base = raw._last_baseline           # LQR commanded (v, omega) from the previous apply
            res = raw._last_residual            # scaled residual (v, omega)
            # wheel spin (spinning-but-stuck vs stalled) and body tilt (high-centered vs flat)
            try:
                wsp = raw.robot.data.joint_vel[:, _wids].abs().mean(dim=1) if _wids else torch.zeros(raw.num_envs, device=raw.device)
            except Exception:
                wsp = torch.zeros(raw.num_envs, device=raw.device)
            try:
                _g = raw.robot.data.projected_gravity_b
                tilt = torch.rad2deg(torch.atan2(torch.norm(_g[:, :2], dim=-1), _g[:, 2].abs().clamp(min=1e-6)))
            except Exception:
                tilt = torch.zeros(raw.num_envs, device=raw.device)
            actions = policy(obs)
            if args.zero_residual:
                actions = torch.zeros_like(actions)
            obs, _, dones, _ = wrapped.step(actions)
        d = dones.bool()
        for e in range(K):
            if bool(done_flag[e]):
                continue
            r = rec[e]
            r["x"].append(float(pos[e, 0])); r["y"].append(float(pos[e, 1]))
            r["vel"].append(float(fwd_vel[e]))
            r["cmd_v"].append(float(base[e, 0])); r["cmd_w"].append(float(base[e, 1]))
            r["res_v"].append(float(res[e, 0])); r["res_w"].append(float(res[e, 1]))
            r["cte"].append(float(cte[e])); r["prog"].append(float(prog[e]))
            r["wheel_spd"].append(float(wsp[e])); r["tilt"].append(float(tilt[e]))
            if bool(d[e]):
                done_flag[e] = True
        if bool(done_flag.all()):
            break

    # --- classify each recorded rover + diagnose the stuck ones ---
    out = []
    for e in range(K):
        r = rec[e]
        if not r["prog"]:
            continue
        final = r["prog"][-1]
        tail = slice(-100, None)   # the last 100 recorded steps = the "stuck phase"
        cmdv_t = float(np.mean(np.abs(r["cmd_v"][tail])))
        actv_t = float(np.mean(np.abs(r["vel"][tail])))
        cte_t = float(np.mean(np.abs(r["cte"][tail])))
        wsp_t = float(np.mean(np.abs(r["wheel_spd"][tail]))) if r["wheel_spd"] else 0.0
        tilt_t = float(np.mean(np.abs(r["tilt"][tail]))) if r["tilt"] else 0.0
        kind = ("success" if final >= 90 else
                "parked" if final < 5 else
                "stalled" if final < 50 else "partial")
        mech = ""
        if kind != "success":
            if cte_t > 0.3:
                mech = "OFF-PATH (drove off the path)"
            elif cmdv_t > 0.02 and actv_t < 0.012:
                # WEDGED: commanding motion, not moving. Sub-classify by wheel spin + tilt:
                if wsp_t < 0.3:
                    mech = "WEDGED-BLOCKED (wheels stalled by terrain/torque)"
                elif tilt_t > 18.0:
                    mech = "WEDGED-BEACHED (wheels spinning, body tilted/high-centered)"
                else:
                    mech = "WEDGED-SLIP (wheels spinning on the ground, no traction)"
            elif cmdv_t <= 0.02:
                mech = "IDLE (controller commanding ~0)"
            else:
                mech = "SLOW (creeping below the 0.02 kill line)"
        out.append(dict(env=e, final=final, kind=kind, mech=mech, cmd_v_tail=cmdv_t,
                        act_v_tail=actv_t, cte_tail=cte_t, wheel_spd_tail=wsp_t, tilt_tail=tilt_t,
                        steps=len(r["prog"])))

    import collections
    kinds = collections.Counter(o["kind"] for o in out)
    stuck = [o for o in out if o["kind"] != "success"]
    print("\n==================== STALL DIAGNOSIS ====================")
    print(f"  checkpoint : {os.path.basename(args.checkpoint)}   terrain {args.terrain:.0f}%   "
          f"residual {'ZERO (pure LQR)' if args.zero_residual else 'ON'}")
    print(f"  sampled    : {len(out)} rovers")
    for k in ("success", "parked", "stalled", "partial"):
        print(f"     {k:<8} {kinds.get(k,0):4d}  ({100*kinds.get(k,0)/max(len(out),1):.0f}%)")
    print("  MECHANISM of the stuck rovers (the diagnosis):")
    for m, c in collections.Counter(o["mech"] for o in stuck).most_common():
        print(f"     {m:<42} {c:4d}")
    if stuck:
        cv = float(np.mean([o['cmd_v_tail'] for o in stuck]))
        av = float(np.mean([o['act_v_tail'] for o in stuck]))
        ws = float(np.mean([o['wheel_spd_tail'] for o in stuck]))
        ti = float(np.mean([o['tilt_tail'] for o in stuck]))
        print("  stuck rovers, last 100 steps:")
        print(f"     LQR commanded |v| = {cv:.3f} m/s    actual |v| = {av:.3f} m/s")
        print(f"     wheel spin = {ws:.2f} rad/s         body tilt = {ti:.1f} deg")
        wedged = [o for o in stuck if o['mech'].startswith('WEDGED')]
        if wedged:
            print("     WEDGED sub-type breakdown:")
            for m, c in collections.Counter(o['mech'] for o in wedged).most_common():
                print(f"        {m:<54} {c}")
            print("     => the dominant sub-type names the fix:")
            print("        SLIP    -> raise wheel/terrain friction (wheels spin on ground, no grip)")
            print("        BEACHED -> raise ground clearance / lower terrain feature height (high-centered)")
            print("        BLOCKED -> terrain too steep for the torque; cap difficulty or raise effort/speed")
    print("========================================================\n")

    outdir = os.path.join(os.path.dirname(args.checkpoint) or ".", "stall_diag")
    os.makedirs(outdir, exist_ok=True)
    pick = [o["env"] for o in stuck][:12] + [o["env"] for o in out if o["kind"] == "success"][:3]
    dump = {"terrain": args.terrain, "zero_residual": bool(args.zero_residual),
            "envs": [{"env": e, "cls": next(o for o in out if o["env"] == e),
                      "ref": refpaths[e], "traj": rec[e]} for e in pick]}
    jpath = os.path.join(outdir, "stall_diag.json")
    json.dump(dump, open(jpath, "w"))
    print(f"  data -> {jpath}", flush=True)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        show = [o["env"] for o in stuck][:9]
        ncol = 3; nrow = max(1, (len(show) + ncol - 1) // ncol)
        fig, ax = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
        ax = np.array(ax).reshape(-1)
        for i, e in enumerate(show):
            a = ax[i]; r = rec[e]; ref = np.array(refpaths[e]); o = next(o for o in out if o["env"] == e)
            if len(ref):
                a.plot(ref[:, 0], ref[:, 1], "-", color="#bbb", lw=1, label="ref path")
            a.scatter(r["x"], r["y"], c=r["vel"], cmap="viridis", s=6, vmin=0, vmax=0.1)
            a.scatter([r["x"][0]], [r["y"][0]], c="lime", s=70, marker="*", zorder=5)
            a.scatter([r["x"][-1]], [r["y"][-1]], c="red", s=70, marker="X", zorder=5)
            a.set_title(f"env {e}: {o['kind']}\n{o['mech']}\ncmd|v| {o['cmd_v_tail']:.3f}  act|v| {o['act_v_tail']:.3f}",
                        fontsize=8)
            a.set_aspect("equal"); a.grid(alpha=.3)
        for j in range(len(show), len(ax)):
            ax[j].axis("off")
        plt.tight_layout()
        png = os.path.join(outdir, "stall_diag.png")
        plt.savefig(png, dpi=110)
        print(f"  plot -> {png}   (green * = spawn, red X = where it ended/stuck, color = speed)", flush=True)
    except Exception as ex:
        print(f"  (plot skipped: {ex}; scp the JSON and Claude will plot it)", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
