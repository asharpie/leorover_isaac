#!/usr/bin/env python
# scripts/record_demo.py
"""
Record PAIRED demo episodes for the 3D replay viewer (scripts/demo_to_html.py).

Runs headless on the box (no renderer -> immune to the driver-595 GUI segfault),
plays the SAME scenarios twice in one process (hybrid, then zero-residual = pure
LQR) with identical terrain, path, start pose, soil field, and friction, and
records everything the viewer needs:

  - per control step (5 Hz): root position + orientation, wheel angles,
    per-wheel mean slip, cross-track error, residual command
  - per scenario (static): a raycast height grid of the terrain around the
    episode (+ the soil-softness value at each grid point), the path waypoints,
    and the env origin

Output: a single .npz. Convert to a shareable interactive HTML file with:

    python3 scripts/demo_to_html.py evals/demo_<ts>.npz

Usage on the box (GPU must be free - stop training first):

    scripts/run_lab.sh scripts/record_demo.py --num 2 --level 60
    scripts/run_lab.sh scripts/record_demo.py --num 3 --level 100 --friction 0.7
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Record paired demo episodes for the 3D replay viewer.")
parser.add_argument("--mode", choices=["pair", "ppo"], default="pair",
                    help="pair (default) = hybrid + pure-LQR on identical scenarios in one "
                         "process; ppo = single leg from the pure-PPO task/checkpoint")
parser.add_argument("--checkpoint", default="", help="checkpoint (.pt); default = newest for the mode")
parser.add_argument("--num", type=int, default=2, help="number of demo scenarios (= parallel envs)")
parser.add_argument("--level", type=float, default=60.0, help="terrain intensity %% for all scenarios")
parser.add_argument("--friction", type=float, default=1.0, help="fixed wheel friction (both legs)")
parser.add_argument("--seed", type=int, default=7, help="scenario draw seed (paths/columns)")
parser.add_argument("--steps", type=int, default=2400, help="hard step cap per leg")
parser.add_argument("--grid", type=int, default=81, help="terrain height-grid resolution (NxN)")
parser.add_argument("--extent", type=float, default=8.0, help="terrain grid half-extent (m)")
parser.add_argument("--out", default="", help="output .npz (default evals/demo_<stamp>.npz)")

_pre, _ = parser.parse_known_args()
os.environ["LEOROVER_FRICTION"] = str(_pre.friction)   # pin friction BEFORE env import

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

import leorover_isaac  # noqa: F401 (registers tasks)
from leorover_isaac.tasks.leo_rover_agents import (
    LeoRoverMarsHybridPPORunnerCfg, LeoRoverMarsPPORunnerCfg,
)
from leorover_isaac.envs.leo_rover_mars_hybrid_env import LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg
from leorover_isaac.envs.leo_rover_mars_env import LeoRoverMarsEnv, LeoRoverMarsEnvCfg
from rsl_rl.runners import OnPolicyRunner
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
except Exception:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper  # 4.5
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except Exception:
    handle_deprecated_rsl_rl_cfg = None


def _mark(m):
    print(f"[demo] {m}", flush=True)


def _latest_ckpt(mode):
    exp = "leo_rover_mars_hybrid" if mode == "pair" else "leo_rover_mars"
    runs = sorted(glob.glob(f"logs/{exp}/*/model_*.pt"), key=os.path.getmtime)
    return runs[-1] if runs else ""


def _raycast_grid(origin_xy, z_top, half, n):
    """Height grid via PhysX closest-hit raycasts (version-stable API). Falls back
    to zeros (flat) if the query interface is unavailable."""
    try:
        import carb
        from omni.physx import get_physx_scene_query_interface
        q = get_physx_scene_query_interface()
        xs = np.linspace(-half, half, n)
        h = np.zeros((n, n), dtype=np.float32)
        for iy, dy in enumerate(xs):
            for ix, dx in enumerate(xs):
                hit = q.raycast_closest(
                    carb.Float3(float(origin_xy[0] + dx), float(origin_xy[1] + dy), float(z_top)),
                    carb.Float3(0.0, 0.0, -1.0), 200.0)
                if hit and hit.get("hit", False):
                    h[iy, ix] = float(hit["position"][2])
        return h, True
    except Exception as e:
        _mark(f"raycast unavailable ({e}) -> flat terrain in the viewer")
        return np.zeros((n, n), dtype=np.float32), False


def main():
    ckpt = args.checkpoint or _latest_ckpt(args.mode)
    if not ckpt or not os.path.isfile(ckpt):
        _mark(f"no checkpoint found for mode '{args.mode}' (pass --checkpoint)"); sys.exit(1)
    K = int(args.num)
    _mark(f"mode={args.mode}  scenarios={K}  level={args.level}%  friction={args.friction}  "
          f"ckpt={os.path.basename(ckpt)}")

    if args.mode == "pair":
        env_cls, cfg_cls, runner_cfg_cls = (LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg,
                                            LeoRoverMarsHybridPPORunnerCfg)
    else:
        env_cls, cfg_cls, runner_cfg_cls = (LeoRoverMarsEnv, LeoRoverMarsEnvCfg,
                                            LeoRoverMarsPPORunnerCfg)
    cfg = cfg_cls()
    cfg.scene.num_envs = K
    cfg.use_random_paths = True
    if getattr(args, "device", None):
        cfg.sim.device = args.device
    _mark("creating env (scene build + terrain cook)...")
    env = env_cls(cfg=cfg, render_mode=None)
    raw = env
    raw._adr = None
    raw._eval_levels = None

    # --- scenarios: K random paths on the requested terrain row -----------------
    rng = np.random.default_rng(args.seed)
    denom = max(int(raw._t_rows) - 1, 1)
    row = min(denom, max(0, round(args.level / 100.0 * denom)))
    path_idx = rng.integers(0, raw._bank_size, size=K)
    col = rng.integers(0, max(int(raw._t_cols), 1), size=K)
    raw._eval_scenarios = {
        "path_idx": torch.as_tensor(path_idx, dtype=torch.long, device=raw.device),
        "row":      torch.full((K,), row, dtype=torch.long, device=raw.device),
        "col":      torch.as_tensor(col, dtype=torch.long, device=raw.device),
    }
    raw._scenario_ptr = 0
    env.reset()   # homes env e onto scenario e; builds soil lazily on first step

    # --- policy ------------------------------------------------------------------
    agent_cfg = runner_cfg_cls()
    if handle_deprecated_rsl_rl_cfg is not None:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=str(wrapped.unwrapped.device))
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

    # one zero-action step so lazy pieces (soil model, buffers) exist
    with torch.inference_mode():
        obs, _ = wrapped.get_observations()
        a0 = policy(obs)
        wrapped.step(torch.zeros_like(a0))

    # --- static geometry: terrain height grid + soil zone per scenario ----------
    _mark("lifting rovers + raycasting terrain grids...")
    st = raw.robot.data.root_state_w.clone()
    st[:, 2] += 60.0
    raw.robot.write_root_state_to_sim(st)
    with torch.inference_mode():
        wrapped.step(torch.zeros_like(a0))          # let PhysX ingest the new poses
    origins = raw._terrain.env_origins.detach().cpu().numpy().astype(np.float32)  # [K,3]
    G, half = int(args.grid), float(args.extent)
    heights = np.zeros((K, G, G), dtype=np.float32)
    soil = np.zeros((K, G, G), dtype=np.float32)
    for e in range(K):
        heights[e], ok = _raycast_grid(origins[e, :2], origins[e, 2] + 40.0, half, G)
        if not ok:
            heights[e] += origins[e, 2]
        if raw._soil is not None:
            xs = torch.linspace(-half, half, G, device=raw.device)
            gx, gy = torch.meshgrid(xs, xs, indexing="xy")
            pts = torch.stack([gx.reshape(-1) + float(origins[e, 0]),
                               gy.reshape(-1) + float(origins[e, 1])], dim=1)
            soil[e] = raw._soil.zone_at(pts).reshape(G, G).detach().cpu().numpy()
        _mark(f"  scenario {e}: terrain z {heights[e].min():.2f}..{heights[e].max():.2f} m, "
              f"soil {soil[e].min():.2f}..{soil[e].max():.2f}")

    # --- waypoints (env-local; viewer adds the origin) ---------------------------
    nwp = raw._num_wp.detach().cpu().numpy().astype(np.int32)
    wps = raw._wps[:, :, :2].detach().cpu().numpy().astype(np.float32)   # [K, W, 2]

    # --- record all legs over the SAME scenarios ---------------------------------
    leg_plan = (("hybrid", False), ("lqr", True)) if args.mode == "pair" else (("ppo", False),)
    legs = {}
    for name, zero in leg_plan:
        raw._scenario_ptr = 0
        env.reset()
        obs, _ = wrapped.get_observations()
        P, Q, WH, SL, CT = [], [], [], [], []
        done_step = np.full(K, -1, dtype=np.int32)
        _mark(f"recording {name} leg...")
        for t in range(int(args.steps)):
            with torch.inference_mode():
                act = policy(obs)
                if zero:
                    act = torch.zeros_like(act)
                obs, _, dones, _ = wrapped.step(act)
            P.append(raw.robot.data.root_pos_w.detach().cpu().numpy().copy())
            Q.append(raw.robot.data.root_quat_w.detach().cpu().numpy().copy())   # (w,x,y,z)
            WH.append(raw.robot.data.joint_pos[:, raw._soil_jids].detach().cpu().numpy().copy()
                      if raw._soil is not None else np.zeros((K, 4), dtype=np.float32))
            SL.append(raw._soil_slip.abs().mean(dim=1).detach().cpu().numpy().copy()
                      if raw._soil is not None else np.zeros(K, dtype=np.float32))
            cte, _ = raw._true_cte_and_along()
            CT.append(cte.abs().detach().cpu().numpy().copy())
            d = dones.reshape(-1).detach().cpu().numpy().astype(bool)
            for e in range(K):
                if d[e] and done_step[e] < 0:
                    done_step[e] = t
            if (done_step >= 0).all():
                break
        legs[name] = dict(pos=np.stack(P), quat=np.stack(Q), wheels=np.stack(WH),
                          slip=np.stack(SL), cte=np.stack(CT), done=done_step)
        _mark(f"  {name}: {len(P)} steps, done at {done_step.tolist()}")

    out = args.out or os.path.join("evals", f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    meta = dict(level=args.level, friction=args.friction, num=K, dt=0.2,
                grid=G, extent=half, ckpt=os.path.basename(ckpt),
                legs=[n for n, _ in leg_plan])
    np.savez_compressed(
        out, meta=json.dumps(meta), origins=origins, heights=heights, soil=soil,
        wps=wps, nwp=nwp,
        **{f"{n}_{k}": v for n, L in legs.items() for k, v in L.items()})
    _mark(f"done -> {out}   (convert: python3 scripts/demo_to_html.py {out})")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
