# leorover_isaac/assets/leo_rover/__init__.py
"""Leo Rover articulation config for Isaac Lab.

Exports `LEO_ROVER_CFG`, an `ArticulationCfg` pointing at the USD generated from
the PyBullet URDF (`leo_robot_1_ros2_shared.urdf`). Generate the USD once with:

    python scripts/convert_urdf.py          # wraps Isaac Lab's URDF converter

which writes `leo_robot.usd` next to this file (gitignored — the URDF + meshes
under source/ are the tracked ground truth).

ROVER KINEMATICS (must match the PyBullet Controller2):
  * 6 links: base + rocker_L + rocker_R (fixed) + 4 wheels.
  * Driven joints: wheel_FL_joint, wheel_RL_joint (left side),
                   wheel_FR_joint, wheel_RR_joint (right side).
    In the URDF, wheel_RR mimics wheel_FL and rocker_R mimics rocker_L; after
    URDF->USD conversion these become independent revolute joints, so the env
    drives all four explicitly (left command -> FL,RL; right command -> FR,RR),
    reproducing Controller2's `my_joint_velocities=[0, L, L, 0, R, R]` mapping.
  * Velocity control: wheels are driven by writing joint velocity TARGETS
    (stiffness=0, damping>0), matching PyBullet's VELOCITY_CONTROL. We do NOT
    use the URDF <transmission> blocks (they don't survive URDF->USD).
  * Wheel limits mirror the URDF: effort<=2.0 N·m, velocity<=6.0 rad/s.

The controller's kinematic wheel radius (0.3 m, from Controller2) intentionally
differs from the URDF wheel collision radius (~0.06 m) — this is preserved
exactly as in PyBullet so emergent ground speed matches. See controllers/lqr.py.
"""

from __future__ import annotations

import os

__all__ = ["LEO_ROVER_CFG", "USD_PATH", "WHEEL_JOINTS", "LEFT_WHEELS", "RIGHT_WHEELS"]

_HERE = os.path.dirname(__file__)
USD_PATH = os.path.join(_HERE, "leo_robot.usd")

WHEEL_JOINTS = ["wheel_FL_joint", "wheel_RL_joint", "wheel_FR_joint", "wheel_RR_joint"]
LEFT_WHEELS = ["wheel_FL_joint", "wheel_RL_joint"]
RIGHT_WHEELS = ["wheel_FR_joint", "wheel_RR_joint"]


def _build_cfg():
    """Construct the ArticulationCfg (lazy — needs isaaclab installed)."""
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
    except Exception as exc:  # pragma: no cover
        print(f"[leo_rover] isaaclab unavailable ({exc}); LEO_ROVER_CFG=None. "
              f"Run inside the Isaac Lab env to use the articulation.")
        return None

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=10.0,
                max_angular_velocity=50.0,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=50,   # match PyBullet numSolverIterations=50
                solver_velocity_iteration_count=4,    # match numSubSteps=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Velocity-controlled wheels: stiffness=0, damping>0 -> velocity tracking.
            # The rockers are fixed joints in the URDF and get MERGED into base_link
            # during URDF->USD conversion (merge_fixed_joints), so only the 4 wheel
            # joints exist — no rocker actuator. effort_limit_sim/velocity_limit_sim
            # are the current Isaac Lab names (plain effort_limit/velocity_limit are
            # deprecated and ignored for implicit actuators).
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*_joint"],
                effort_limit_sim=2.0,        # URDF <limit effort="2.0">
                velocity_limit_sim=6.0,      # URDF <limit velocity="6.0">
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )


LEO_ROVER_CFG = _build_cfg()
