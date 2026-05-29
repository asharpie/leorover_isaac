"""leo_rover_mars_env.py — Phase 3 stub.

Pure-PPO Isaac Lab env on randomized Martian heightfield terrain. Same
action/obs interface as the flat env, but the ground is now a heightfield
collider parameterized by terrain_intensity, and the observation gains the
camera-lookahead slope features used in the PyBullet pure-PPO config.

PyBullet equivalent: MyEnv2(use_lqr_baseline=False, use_pure_ppo_reward=True,
use_camera_lookahead=True) — i.e. the pure-PPO mode that the PyBullet
training scripts call train_ppo on.

TODO (Phase 3):
  - Compose with the Phase 3 terrain importer (see leorover_isaac/terrain/)
  - Wire terrain_intensity into the obs and an EventTerm for ADR
  - Add the camera-lookahead RayCaster sensor (slope_near, slope_mid, slope_far)
  - Port the v33.9 pure-PPO reward terms with parity weights

Stub for now.
"""

from __future__ import annotations

__all__ = ["LeoRoverMarsEnv"]


class LeoRoverMarsEnv:
    """Stub for Phase 3 — see module docstring."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LeoRoverMarsEnv is a Phase 3 stub. See PORTING_ROADMAP.md."
        )
