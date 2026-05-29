# assets/leo_rover/ — Robot asset (Phase 1)

This directory holds the Leo Rover articulation as both:

- `source/` — the original URDF + meshes + xacro macros from the PyBullet
  repo. Tracked in git so the conversion is reproducible.
- `leo_robot.usd` — the converted Isaac Sim asset. NOT tracked in git
  (.gitignored). Regenerated from `source/` via Isaac Lab's converter.

## Files to copy from the PyBullet repo

From `leoroverpybullet_share - Checkpoint Working (3) (1)/`:

```
leo_robot_1_ros2_shared.urdf   →  source/leo_robot.urdf
macros.xacro                   →  source/macros.xacro
Antenna.dae                    →  source/Antenna.dae
Chassis.dae                    →  source/Chassis.dae
Chassis_outline.stl            →  source/Chassis_outline.stl
Rocker.dae                     →  source/Rocker.dae
Rocker_outline.stl             →  source/Rocker_outline.stl
Universal_camera_support.stl   →  source/Universal_camera_support.stl
WheelA.dae                     →  source/WheelA.dae
WheelB.dae                     →  source/WheelB.dae
WheelMecanumA.dae              →  source/WheelMecanumA.dae
WheelMecanumB.dae              →  source/WheelMecanumB.dae
Wheel_outline.stl              →  source/Wheel_outline.stl
kinect.dae                     →  source/kinect.dae
realsense.dae                  →  source/realsense.dae
```

## Conversion command

After the files are in `source/`:

```bash
~/IsaacLab/isaaclab.sh -p ~/IsaacLab/source/standalone/tools/convert_urdf.py \
    leorover_isaac/assets/leo_rover/source/leo_robot.urdf \
    leorover_isaac/assets/leo_rover/leo_robot.usd \
    --merge-joints \
    --fix-base=false \
    --make-instanceable
```

Flags:
- `--merge-joints`: collapses redundant fixed joints (most rovers have several)
- `--fix-base=false`: rover should not be world-attached
- `--make-instanceable`: lets Isaac Lab reuse the asset across all parallel envs
  without duplicating geometry — important for memory with 4096 envs

## Validation checklist

After conversion, open the USD in Isaac Sim GUI once and verify:

- [ ] 4 wheel joints, all revolute, all parented to chassis or rocker
- [ ] Wheels rotate freely when scrubbing the joint slider
- [ ] Mesh visuals load (no missing material warnings in the console)
- [ ] Mass/inertia properties look reasonable (chassis ~5–8 kg, wheels ~0.3–0.8 kg)
- [ ] No "collision_0" prim has zero volume (an Isaac importer quirk)

If anything looks wrong, fix the URDF source and reconvert — don't hand-edit
the USD.
