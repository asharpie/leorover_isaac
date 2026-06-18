#!/usr/bin/env python
# scripts/physx_progress_test.py
"""
Definitive PhysX progress test: is the simulation actually advancing, or hung?

Separates two questions that are easy to conflate:
  1. CPU vs GPU -- the "GPU solver pipeline failed, switching to software"
     warning is PhysX announcing it moved the solver to the CPU. If you see it,
     physics is on the CPU. (Optional cross-check: watch nvidia-smi GPU-Util in
     another terminal; with many bodies, GPU physics shows utilization and CPU
     physics does not.)
  2. Progress vs hang -- this script prints a per-step heartbeat (flushed
     immediately) with the step time and the first cube's height. If the
     heartbeat advances and the cube height falls, physics is RUNNING (and the
     step time shows how slow). If it prints "stepping..." and then nothing,
     it is HUNG, not merely slow -- and you will see exactly which phase stalls.

Run headless on the lab box:
    export DISPLAY=:1
    export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
    python physx_progress_test.py --bodies 8 --steps 30      # quick: progress or hang?
    python physx_progress_test.py --bodies 200 --steps 100   # heavier: CPU vs GPU obvious in nvidia-smi
"""
from __future__ import annotations
import argparse, time

parser = argparse.ArgumentParser()
parser.add_argument("--bodies", type=int, default=8)
parser.add_argument("--steps", type=int, default=30)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
except Exception:
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid

print(">>> building scene", flush=True)
world = World(physics_dt=1.0 / 60.0, device="cuda:0")
world.scene.add_default_ground_plane()
cubes = []
for i in range(args.bodies):
    cubes.append(world.scene.add(DynamicCuboid(
        prim_path=f"/World/c{i}", name=f"c{i}",
        position=np.array([0.3 * (i % 20), 0.3 * (i // 20), 1.0 + 0.05 * i]))))

print(">>> reset  (the 'GPU solver pipeline failed' warning, if any, appears around here)", flush=True)
world.reset()

print(f">>> stepping {args.steps} times -- watch the heartbeat below:", flush=True)
t0 = time.time()
for i in range(args.steps):
    s = time.time()
    world.step(render=False)
    dt_ms = (time.time() - s) * 1000.0
    z = float(cubes[0].get_world_pose()[0][2])
    print(f"  step {i+1:>3}/{args.steps}   step_time = {dt_ms:8.1f} ms   cube_z = {z:6.3f}", flush=True)
total = time.time() - t0

print(f">>> DONE: {args.steps} steps in {total:.2f} s  ({args.steps/total:.1f} steps/s)", flush=True)
print(">>> If cube_z fell over the run, physics RAN. step_time / steps-per-second = how slow.", flush=True)
app.close()
