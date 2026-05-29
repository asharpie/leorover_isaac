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

    Called once at import time. Tasks are added incrementally as their
    corresponding port phase is completed. For now this is a no-op — the
    task files exist as stubs.
    """
    # Phase 2: Isaac-LeoRover-Flat-v0
    # Phase 3: Isaac-LeoRover-Mars-v0
    # Phase 4: Isaac-LeoRover-Mars-Hybrid-v0
    pass


# Guard the import so importing leorover_isaac in a non-Isaac environment
# (e.g. running unit tests outside the conda env) still works.
try:
    import isaaclab  # noqa: F401
    _register_tasks()
except ImportError:
    pass
