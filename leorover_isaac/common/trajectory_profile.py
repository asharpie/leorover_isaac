# trajectory_profile.py
"""
Trajectory velocity + feedforward-omega profiling (CPU/numpy).

Direct port of MyEnv2._compute_trajectory_profile from the PyBullet stack. Runs
once per episode at reset (CPU is fine — resets are infrequent), producing the
per-waypoint reference velocity v_r and feedforward angular velocity omega_r
that the LQR baseline tracks:  v = v_r + feedback,  omega = omega_r + feedback.

Kept bit-for-bit with the original so paths are timed identically.
"""

from __future__ import annotations

import math


def compute_trajectory_profile(
    x_list, y_list, yaw_list,
    v_max: float,
    wheel_base: float = 0.34,
    wheel_radius: float = 0.3,
    max_wheel_speed: float = 1.333,
    omega_max: float = 1.047,
    v_min: float = 0.15,
    max_accel: float = 0.5,
    max_decel: float = 0.5,
    use_trajectory_profiling: bool = True,
):
    """Return (vel_list, omega_list) per waypoint. See MyEnv2._compute_trajectory_profile."""
    n = len(x_list)
    if n < 2:
        return [v_max] * n, [0.0] * n
    if not use_trajectory_profiling:
        return [v_max] * n, [0.0] * n

    curvatures_unsigned = [0.0] * n
    curvatures_signed = [0.0] * n
    for i in range(1, n - 1):
        seg_before = math.hypot(x_list[i] - x_list[i - 1], y_list[i] - y_list[i - 1])
        seg_after = math.hypot(x_list[i + 1] - x_list[i], y_list[i + 1] - y_list[i])
        avg_seg = (seg_before + seg_after) / 2.0
        if avg_seg < 1e-6:
            continue
        angle_change = yaw_list[i + 1] - yaw_list[i - 1]
        angle_change = (angle_change + math.pi) % (2 * math.pi) - math.pi
        curvatures_unsigned[i] = abs(angle_change) / avg_seg
        curvatures_signed[i] = angle_change / avg_seg
    curvatures_unsigned[0] = curvatures_unsigned[1] if n > 1 else 0.0
    curvatures_unsigned[-1] = curvatures_unsigned[-2] if n > 1 else 0.0
    curvatures_signed[0] = curvatures_signed[1] if n > 1 else 0.0
    curvatures_signed[-1] = curvatures_signed[-2] if n > 1 else 0.0

    L = wheel_base
    v_wheel_max = max_wheel_speed * 2.0 * wheel_radius

    vel_list = []
    for kappa in curvatures_unsigned:
        v_ws = v_wheel_max / (1.0 + L * kappa)
        v_omega = omega_max / max(kappa, 0.01)
        v = min(v_max, v_ws, v_omega)
        v = max(v_min, v)
        vel_list.append(v)

    for i in range(1, n):
        seg_len = math.hypot(x_list[i] - x_list[i - 1], y_list[i] - y_list[i - 1])
        max_vel_increase = max_accel * max(seg_len, 0.01)
        vel_list[i] = min(vel_list[i], vel_list[i - 1] + max_vel_increase)
    for i in range(n - 2, -1, -1):
        seg_len = math.hypot(x_list[i + 1] - x_list[i], y_list[i + 1] - y_list[i])
        max_vel_increase = max_decel * max(seg_len, 0.01)
        vel_list[i] = min(vel_list[i], vel_list[i + 1] + max_vel_increase)

    omega_list = [vel_list[i] * curvatures_signed[i] for i in range(n)]
    return vel_list, omega_list
