"""mars_heightfield.py — Phase 3 stub for Mars terrain generation.

Generate a heightfield numpy array matching the statistics of the PyBullet
Mars terrain (slope distribution, height range, feature scale).

The PyBullet generation logic in environment2.py uses Perlin noise + bumps.
Most of that math ports directly to numpy — it doesn't need to vectorize
across envs because terrain is built once at startup, not per episode.

TODO (Phase 3):
  - Port the Perlin noise + bump generation
  - Expose terrain_intensity as a scaling factor on the noise amplitude
  - Return [H, W] numpy float32 array compatible with TerrainImporterCfg
  - Validate slope distribution against a PyBullet reference run

Stub for now.
"""

from __future__ import annotations

import numpy as np


def generate_mars_patch(
    size_m: float = 8.0,
    resolution_m: float = 0.05,
    terrain_intensity: float = 50.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate one Mars-like heightfield patch.

    Args:
        size_m: side length of the square patch in meters.
        resolution_m: distance between heightfield cells in meters.
        terrain_intensity: 0-100, controls amplitude of features.
        seed: optional rng seed for reproducibility.

    Returns:
        ndarray of shape `[H, W]` float32, heights in meters.

    Stub: returns flat ground until Phase 3 is implemented.
    """
    n = int(size_m / resolution_m)
    return np.zeros((n, n), dtype=np.float32)
