"""leorover_isaac.common — engine-agnostic CPU/numpy logic carried over verbatim
from the PyBullet stack (leoroverpybullet).

These modules contain NO physics-engine calls. They are the single source of
truth for path geometry, terrain statistics, and the reference LQR gains, and
are shared by both the PyBullet repo and this Isaac Lab port so the two stay
behaviourally identical.

Modules:
  path_templates        — the 9 fixed evaluation paths (zig-zag / curved / polygon)
  random_path_generator — random curved-path generator with curvature control
  terrain_stats         — slope statistics over a heightfield (max/avg/percentiles)
  lqr_baseline          — reference numpy LQR (used to derive the constant gain K
                          that controllers/lqr.py reuses on the GPU)
  path_generation       — quintic-polynomial 2D planner (QuinticPolynomial*)
  mars_terrain_numpy    — procedural Gaussian-hill heightfield generation +
                          get_height_at bilinear sampler (port of mars_terrain.py
                          with all PyBullet calls removed)
"""

from . import path_templates          # noqa: F401
from . import random_path_generator   # noqa: F401
from . import terrain_stats           # noqa: F401
from . import lqr_baseline            # noqa: F401
from . import mars_terrain_numpy      # noqa: F401
