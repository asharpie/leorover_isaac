"""leorover_isaac — Isaac Lab port of the Leo Rover PyBullet RL stack.

This package is a sibling/successor to leoroverpybullet_share. Both should
remain functional; this one targets GPU-accelerated training on Isaac Sim
4.5+ via Isaac Lab.

Status: scaffolding only. See PORTING_ROADMAP.md for the phase-by-phase plan.

Top-level layout::

    leorover_isaac/
        assets/        — URDF/USD for the Leo Rover (Phase 1)
        envs/          — Isaac Lab DirectRLEnv subclasses (Phase 2+)
        controllers/   — Vectorized LQR + helper controllers (Phase 4)
        tasks/         — Task configs + gymnasium registrations (Phase 2+)
        terrain/       — Mars heightfield generation (Phase 3)
        utils/         — Shared math, conversions, logging glue
"""

__version__ = "0.0.1"


def _register_tasks():
    """Register all gym tasks defined in this package.

    Called once at import time. Registers:
      Isaac-LeoRover-Flat-v0         (pure PPO, flat ground — smoke test)
      Isaac-LeoRover-Mars-v0         (pure PPO, Mars terrain — train_ppo)
      Isaac-LeoRover-Mars-Hybrid-v0  (LQR + residual — train_hybrid_ppo)
    """
    from leorover_isaac.tasks import register_tasks
    register_tasks()


# Guard the import so importing leorover_isaac in a non-Isaac environment
# (e.g. running unit tests outside the conda env) still works. We attempt
# registration whenever gymnasium is available; the env classes themselves
# only need Isaac Lab when actually instantiated.
try:
    _register_tasks()
except Exception as _exc:  # pragma: no cover
    import os as _os
    if _os.environ.get("LEOROVER_DEBUG"):
        print(f"[leorover_isaac] task registration deferred: {_exc}")
