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

# --- sys.path hygiene -------------------------------------------------------
# The repo root holds the project-wide `config.py`. Isaac Sim 4.5 bundles an
# OpenCV whose `cv2/config.py` otherwise shadows our `import config`, producing
# a `NameError: LOADER_DIR` at env-import time. Force the repo root to the very
# front of sys.path (ahead of Isaac's bundled dirs) so `import config` always
# resolves to ours. This runs before _register_tasks() imports any submodule.
import os as _os, sys as _sys, importlib.util as _ilu
_repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
while _repo_root in _sys.path:
    _sys.path.remove(_repo_root)
_sys.path.insert(0, _repo_root)
# Belt-and-suspenders: explicitly load OUR config.py by absolute path and cache
# it as the top-level `config` module. Python consults sys.modules BEFORE it
# searches sys.path, so pre-populating the cache guarantees every later
# `import config` gets ours, even when Isaac's bundled cv2 dir sits ahead of the
# repo root on sys.path (which is what broke the plain sys.path fix).
_cfg_path = _os.path.join(_repo_root, "config.py")
if _os.path.isfile(_cfg_path):
    try:
        _spec = _ilu.spec_from_file_location("config", _cfg_path)
        _cfg_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_cfg_mod)
        _sys.modules["config"] = _cfg_mod
        if _os.environ.get("LEOROVER_DEBUG"):
            print(f"[leorover_isaac] config -> {_cfg_path}")
    except Exception as _e:  # pragma: no cover
        if _os.environ.get("LEOROVER_DEBUG"):
            print(f"[leorover_isaac] explicit config load failed: {_e}")
elif _os.environ.get("LEOROVER_DEBUG"):
    print(f"[leorover_isaac] WARNING: config.py not found at {_cfg_path}")


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
