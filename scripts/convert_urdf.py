#!/usr/bin/env python
# scripts/convert_urdf.py
"""
Convert the Leo Rover URDF to USD for Isaac Lab.

This wraps Isaac Lab's URDF->USD converter so you don't have to remember the
exact CLI for your Isaac version. Run it once after install; the generated
`leo_robot.usd` lands next to the asset config and is gitignored.

Usage (inside the Isaac Lab python env):
    isaaclab -p scripts/convert_urdf.py
    # or:  python scripts/convert_urdf.py  (if the isaac python is active)

What it does:
  * Resolves the source URDF at
    leorover_isaac/assets/leo_rover/source/leo_robot_1_ros2_shared.urdf
  * Fixes the base so the rover is free-floating (not fixed to the world).
  * Merges fixed joints (rockers are fixed) and keeps the 4 wheel joints
    as revolute/continuous so they can be velocity-driven.
  * Writes leo_robot.usd next to leorover_isaac/assets/leo_rover/__init__.py.

If your Isaac version ships convert_urdf.py as a standalone tool instead, the
equivalent CLI is:
    isaaclab -p <IsaacLab>/scripts/tools/convert_urdf.py \
        leorover_isaac/assets/leo_rover/source/leo_robot_1_ros2_shared.urdf \
        leorover_isaac/assets/leo_rover/leo_robot.usd \
        --merge-joints --joint-stiffness 0 --joint-damping 10
"""

from __future__ import annotations

import os
import argparse


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    src = os.path.join(repo, "leorover_isaac", "assets", "leo_rover", "source",
                       "leo_robot_1_ros2_shared.urdf")
    dst = os.path.join(repo, "leorover_isaac", "assets", "leo_rover", "leo_robot.usd")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=src)
    ap.add_argument("--output", default=dst)
    ap.add_argument("--headless", action="store_true", default=True)
    args = ap.parse_args()

    # --- Boot a (headless) SimulationApp BEFORE importing isaaclab sim bits ---
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    # Imports that require the app to be running:
    import isaaclab.sim as sim_utils
    try:
        # Isaac Lab >= 2.0 namespace
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    except Exception:
        # Older namespace fallback
        from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg  # type: ignore

    print(f"[convert_urdf] input : {args.input}")
    print(f"[convert_urdf] output: {args.output}")

    # The UrdfConverterCfg field names have shifted slightly across versions;
    # build kwargs defensively.
    cfg_kwargs = dict(
        asset_path=args.input,
        usd_dir=os.path.dirname(args.output),
        usd_file_name=os.path.basename(args.output),
        fix_base=False,            # free-floating rover
        merge_fixed_joints=True,   # collapse the fixed rocker joints
        force_usd_conversion=True,
        make_instanceable=False,
    )
    try:
        cfg = UrdfConverterCfg(**cfg_kwargs)
    except TypeError:
        # Older field names
        cfg_kwargs.pop("merge_fixed_joints", None)
        cfg = UrdfConverterCfg(**cfg_kwargs, merge_joints=True)  # type: ignore

    converter = UrdfConverter(cfg)
    print(f"[convert_urdf] USD written to: {converter.usd_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
