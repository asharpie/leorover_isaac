"""lqr.py — Phase 4 stub for vectorized LQR.

This module will host the GPU-side LQR controller that the hybrid env
uses to provide the baseline action. The PyBullet implementation lives at
`leoroverpybullet/.../controller2.py` — port the gain computation directly,
but replace the per-step numpy math with torch tensor ops over
`[num_envs, ...]`-shaped state vectors.

Design notes:

1. Compute K_lqr once at construction (Q, R fixed). Register as a buffer
   so it follows the env to cuda:0 on init.
2. Keep _smoothed_target_yaw as a per-env tensor of shape [num_envs].
   Update it via an exponential moving average on every _pre_physics_step.
3. The waypoint frame transform uses isaaclab.utils.math.quat_rotate_inverse.
4. Output is a [num_envs, 2] tensor of (velocity_baseline, omega_baseline)
   that the hybrid env will add the (clipped) residual to.

TODO (Phase 4):
  - Port Controller2.__init__ → VectorizedLQR.__init__ on torch
  - Port _compute_lqr_action with shape [num_envs, state_dim]
  - Port yaw-smoothing EMA
  - Add unit tests in tests/test_lqr.py against a saved PyBullet trajectory
    (load a few state snapshots from a PyBullet rollout, run through both
    controllers, assert action outputs match within float32 epsilon)
"""

from __future__ import annotations

__all__ = ["VectorizedLQR"]


class VectorizedLQR:
    """Stub for Phase 4 — see module docstring."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VectorizedLQR is a Phase 4 stub. See PORTING_ROADMAP.md."
        )
