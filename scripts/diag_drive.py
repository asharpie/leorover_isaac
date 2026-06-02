#!/usr/bin/env python
# scripts/diag_drive.py
"""
Drivetrain smoke test — NO policy, NO checkpoint required.

Commands a fixed action (default = full-throttle forward, [1, 0]) for N steps and
measures the ACTUAL ground speed the physics produces. This isolates the rover's
kinematics + actuators from the RL policy, answering the core question directly:
can the rover move, and how fast?

Run on FLAT first (fast startup ~30 s, no terrain bake, isolates the drivetrain):
    /workspace/IsaacLab/isaaclab.sh -p scripts/diag_drive.py \
        --task Isaac-LeoRover-Flat-v0 --num_envs 16 --steps 200 --headless

Then on MARS (slower startup; also checks the terrain collision mesh):
    /workspace/IsaacLab/isaaclab.sh -p scripts/diag_drive.py \
        --task Isaac-LeoRover-Mars-v0 --num_envs 16 --steps 200 --headless

Try a pure turn too (sanity-check steering):  --throttle 0 --turn 1
"""
from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Leo Rover drivetrain smoke test.")
parser.add_argument("--task", default="Isaac-LeoRover-Flat-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--throttle", type=float, default=1.0, help="forward action in [-1,1]")
parser.add_argument("--turn", type=float, default=0.0, help="omega action in [-1,1]")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np
import torch

import leorover_isaac  # registers tasks
from leorover_isaac.envs.leo_rover_flat_env import LeoRoverFlatEnv, LeoRoverFlatEnvCfg
from leorover_isaac.envs.leo_rover_mars_env import LeoRoverMarsEnv, LeoRoverMarsEnvCfg
from leorover_isaac.envs.leo_rover_mars_hybrid_env import LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg

_TASKS = {
    "Isaac-LeoRover-Flat-v0":        (LeoRoverFlatEnv, LeoRoverFlatEnvCfg),
    "Isaac-LeoRover-Mars-v0":        (LeoRoverMarsEnv, LeoRoverMarsEnvCfg),
    "Isaac-LeoRover-Mars-Hybrid-v0": (LeoRoverMarsHybridEnv, LeoRoverMarsHybridEnvCfg),
}


def main():
    env_cls, cfg_cls = _TASKS[args.task]
    cfg = cfg_cls()
    cfg.scene.num_envs = args.num_envs
    env = env_cls(cfg=cfg, render_mode=None)
    raw = env

    obs, _ = env.reset()

    # fixed action for every env, every step
    act = torch.zeros(args.num_envs, 2, device=raw.device)
    act[:, 0] = args.throttle
    act[:, 1] = args.turn

    ctrl = raw._controller
    p0 = raw.robot.data.root_pos_w.clone()
    speeds, zheights = [], []
    wheel_target = None

    for i in range(args.steps):
        obs, rew, term, trunc, info = env.step(act)
        lin_b = raw.robot.data.root_lin_vel_b
        speeds.append(lin_b[:, 0].abs().mean().item())
        zheights.append(raw.robot.data.root_pos_w[:, 2].mean().item())
        if wheel_target is None and getattr(raw, "_wheel_l", None) is not None:
            wheel_target = raw._wheel_l.abs().mean().item()

    p1 = raw.robot.data.root_pos_w
    dist = torch.norm((p1 - p0)[:, :2], dim=-1)
    sp = np.array(speeds)

    # what the physics SHOULD produce if wheels track the target:
    #   ground_speed = wheel_rad/s * physical_wheel_radius
    # (controller kinematic radius is 0.3 m; physical USD wheel ~0.06 m)
    print("\n================= DRIVE DIAGNOSTICS =================")
    print(f"  task                  : {args.task}")
    print(f"  action (throttle,turn): ({args.throttle}, {args.turn})")
    print(f"  steps                 : {args.steps}  (= {args.steps*0.2:.0f} s sim)")
    print(f"  controller kin radius : {ctrl.wheel_radius} m   max_v_clip {ctrl.max_velocity_clip} m/s")
    print(f"  commanded wheel speed : {wheel_target:.3f} rad/s" if wheel_target is not None else "  commanded wheel speed : n/a")
    print(f"  --> MEAN forward speed: {sp.mean():.4f} m/s")
    print(f"  --> MAX  forward speed: {sp.max():.4f} m/s")
    print(f"  distance traveled     : mean {dist.mean().item():.3f} m   max {dist.max().item():.3f} m")
    print(f"  body z-height         : start {zheights[0]:.3f} -> end {zheights[-1]:.3f} m  (big drop = falling through ground)")
    print(f"  HEALTHY would be      : ~{ctrl.max_velocity_clip:.2f} m/s, ~{ctrl.max_velocity_clip*args.steps*0.2:.1f} m traveled")
    print("====================================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
