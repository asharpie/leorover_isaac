#!/usr/bin/env python
"""
Minimal Isaac Sim PhysX-on-GPU check (driver reproducer).

On this machine's current NVIDIA driver (595.x), general CUDA works fine but
Isaac Sim's PhysX refuses the GPU and falls back to CPU. This script shows that
in about a minute, without needing the full Leo Rover project.

Run inside an Isaac Sim venv, headless:

    source ~/Desktop/environments/leo311/bin/activate
    export DISPLAY=:1
    export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
    python physx_gpu_check.py

What to look for in the output:
  - "torch CUDA matmul OK: <number>"  means general GPU compute works.
  - The line "PhysX warning: GPU solver pipeline failed, switching to software"
    means PhysX could NOT use the GPU. That is the bug. On a 580-series driver
    that line does not appear and physics runs on the GPU.

Note: on the broken driver it may stall after that warning (CPU PhysX hangs);
the warning itself is the evidence, so Ctrl+C is fine once you see it.
"""

# 1) Does plain CUDA work?  (independent of Isaac Sim / PhysX)
import torch
print("=" * 64)
print("torch CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU                  :", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    print("torch CUDA matmul OK :", float((x @ x).sum()))
print("=" * 64)

# 2) Does Isaac Sim's PhysX use the GPU?
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
except Exception:
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid

world = World(physics_dt=1.0 / 60.0, device="cuda:0")   # request the GPU pipeline
world.scene.add_default_ground_plane()
for i in range(16):
    world.scene.add(DynamicCuboid(prim_path=f"/World/c{i}", name=f"c{i}",
                    position=np.array([0.3 * i, 0.0, 1.0])))
world.reset()
for _ in range(60):
    world.step(render=False)

print("=" * 64)
print("Stepped 60 physics frames on GPU pipeline request.")
print("If 'GPU solver pipeline failed, switching to software' appeared above,")
print("PhysX fell back to CPU = the driver-595 issue. Otherwise GPU physics works.")
print("=" * 64)
app.close()
