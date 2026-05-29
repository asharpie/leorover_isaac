"""leo_rover_mars_hybrid_env.py — Phase 4 stub.

Hybrid residual-RL env: LQR baseline carries the bulk of the action,
PPO learns a small bounded residual on top. This is the v33.9 hybrid
configuration ported.

PyBullet equivalent: MyEnv2(use_lqr_baseline=True, use_pure_ppo_reward=True,
use_camera_lookahead=True) with MAX_RESIDUAL_VELOCITY=0.15, MAX_RESIDUAL_OMEGA=0.30,
PPO_W_EFFORT=0.5, etc.

TODO (Phase 4):
  - Subclass LeoRoverMarsEnv
  - Override _apply_action to compute LQR(state) and add clipped residual
  - Add the r_effort L2 penalty as a RewardTerm (weight=0.5)
  - Mirror v33.9 weights exactly: PPO_W_CTE=5.0, PPO_W_VELOCITY=0.5,
    PPO_W_PROGRESS=10.0, PPO_SUCCESS_BONUS=200.0, PPO_FAILURE_PENALTY=50.0
  - Expose per-env residual_v_norm / residual_w_norm in info so the
    Phase 6 Recorder can produce mean_residual_v_norm / mean_residual_w_norm
    columns matching the PyBullet CSV schema

Stub for now.
"""

from __future__ import annotations

__all__ = ["LeoRoverMarsHybridEnv"]


class LeoRoverMarsHybridEnv:
    """Stub for Phase 4 — see module docstring."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LeoRoverMarsHybridEnv is a Phase 4 stub. See PORTING_ROADMAP.md."
        )
