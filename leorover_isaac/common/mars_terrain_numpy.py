# mars_terrain_numpy.py
"""
Engine-agnostic Mars terrain generation (numpy-only port of mars_terrain.py).

This is a faithful reproduction of the PyBullet `create_terrain()` heightfield
generation — the SAME procedural Gaussian-hill algorithm, grid resolution, and
intensity/friction mappings — with every `pybullet` call removed. It returns
plain numpy/lists so it can be consumed by:

  * the Isaac Lab terrain importer (leorover_isaac/terrain/mars_heightfield.py),
    which turns the heightfield into a USD/PhysX mesh collider, and
  * `get_height_at(...)`, the bilinear sampler used to place waypoints and the
    rover spawn on the terrain surface (identical to the PyBullet version).

Grid:  512 x 512 cells @ 0.05 m  ->  25.6 m x 25.6 m terrain, centered at origin.
Height: height_scale = intensity/100 * 5   (0% flat, 100% steep)
Friction: 0.3 + friction_intensity/100 * 1.7   (0% -> 0.3, 100% -> 2.0)

The heightfield is stored ROW-MAJOR as a flat array indexed `y * numRows + x`,
exactly like the PyBullet `heightfieldData`, so `common.terrain_stats` works on
it unchanged.
"""

from __future__ import annotations

import math
import random
import numpy as np

# Grid resolution — identical to the PyBullet mars_terrain.py constants.
numHeightfieldRows = 512
numHeightfieldColumns = 512
CELL_SIZE = 0.05            # meters per heightfield cell
MESH_SCALE = (0.05, 0.05, 1.0)   # PyBullet meshScale used for the collision shape

TERRAIN_SIZE_M = numHeightfieldRows * CELL_SIZE   # 25.6 m
TERRAIN_HALF_M = TERRAIN_SIZE_M / 2.0             # 12.8 m


def get_height_at(x_world, y_world, heightfieldData,
                  num_rows: int = numHeightfieldRows,
                  num_cols: int = numHeightfieldColumns,
                  cell_size: float = CELL_SIZE):
    """Return interpolated height at world (x, y) via bilinear interpolation.

    Identical math to mars_terrain.get_height_at — kept bit-for-bit so the rover
    spawn height and waypoint projection match the PyBullet stack.
    """
    scale_z = 1.0

    x_index = (x_world + (num_rows * cell_size / 2.0)) / cell_size
    y_index = (y_world + (num_cols * cell_size / 2.0)) / cell_size

    x_index = max(0.0, min(num_rows - 1.001, x_index))
    y_index = max(0.0, min(num_cols - 1.001, y_index))

    x0 = int(x_index)
    x1 = min(x0 + 1, num_rows - 1)
    y0 = int(y_index)
    y1 = min(y0 + 1, num_cols - 1)

    hx0y0 = heightfieldData[y0 * num_rows + x0]
    hx1y0 = heightfieldData[y0 * num_rows + x1]
    hx0y1 = heightfieldData[y1 * num_rows + x0]
    hx1y1 = heightfieldData[y1 * num_rows + x1]

    sx = x_index - x0
    sy = y_index - y0

    h = (hx0y0 * (1 - sx) * (1 - sy) +
         hx1y0 * sx * (1 - sy) +
         hx0y1 * (1 - sx) * sy +
         hx1y1 * sx * sy)
    return h * scale_z


def friction_from_intensity(friction_intensity: float) -> float:
    """0% -> 0.3, 50% -> ~1.15, 100% -> 2.0  (matches PyBullet create_terrain)."""
    friction_intensity = max(0.0, min(100.0, friction_intensity))
    return 0.3 + (friction_intensity / 100.0) * 1.7


def generate_heightfield(seed=None, intensity: float = 50.0):
    """Generate the procedural Gaussian-hill heightfield (flat row-major array).

    Faithful port of the hill-generation loop in mars_terrain.create_terrain().

    Args:
        seed: RNG seed for reproducible terrain (uses Python `random`, like the
              original). None -> nondeterministic.
        intensity: terrain height intensity 0-100 (%).

    Returns:
        heightfieldData: list[float] of length numRows*numCols, row-major
                         (index = y*numRows + x), heights in meters.
    """
    intensity = max(0.0, min(100.0, intensity))
    height_scale = intensity / 100.0 * 5.0

    if seed is not None:
        random.seed(seed)

    heightfieldData = [0.0] * (numHeightfieldRows * numHeightfieldColumns)

    if intensity > 0:
        num_hills = random.randint(20, 100)
        min_hill_radius = 20
        max_hill_radius = 100

        hill_centers = []
        for _ in range(num_hills):
            radius = random.randint(min_hill_radius, max_hill_radius)
            base_max_height = random.uniform(0.05, 0.25)
            max_height = base_max_height * height_scale
            sigma = radius / 2.5
            cx = random.randint(radius, numHeightfieldRows - radius - 1)
            cy = random.randint(radius, numHeightfieldColumns - radius - 1)
            hill_centers.append((cx, cy, radius, max_height, sigma))

        for cx, cy, radius, max_height, sigma in hill_centers:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x = cx + dx
                    y = cy + dy
                    if 0 <= x < numHeightfieldRows and 0 <= y < numHeightfieldColumns:
                        distance = math.sqrt(dx ** 2 + dy ** 2)
                        if distance <= radius:
                            height = max_height * math.exp(-(distance ** 2) / (2.0 * sigma ** 2))
                            index = y * numHeightfieldRows + x
                            heightfieldData[index] += height

    return heightfieldData


def heightfield_to_grid(heightfieldData) -> np.ndarray:
    """Reshape the flat row-major heightfield into a [num_rows, num_cols] array
    (rows indexed by y, columns by x) — convenient for the Isaac terrain mesh."""
    arr = np.asarray(heightfieldData, dtype=np.float32)
    return arr.reshape(numHeightfieldColumns, numHeightfieldRows)


def generate_mars_terrain(seed=None, intensity: float = 50.0,
                          friction_intensity: float = 50.0):
    """Convenience wrapper returning everything the Isaac terrain builder needs.

    Returns dict:
        heightfield   : flat row-major list[float] (for get_height_at / stats)
        grid          : [num_cols, num_rows] float32 array (for mesh build)
        friction      : lateral friction coefficient
        intensity     : clamped terrain intensity
        cell_size     : meters per cell
        size_m        : terrain side length (m)
    """
    hf = generate_heightfield(seed=seed, intensity=intensity)
    return {
        "heightfield": hf,
        "grid": heightfield_to_grid(hf),
        "friction": friction_from_intensity(friction_intensity),
        "intensity": max(0.0, min(100.0, intensity)),
        "friction_intensity": max(0.0, min(100.0, friction_intensity)),
        "cell_size": CELL_SIZE,
        "size_m": TERRAIN_SIZE_M,
        "num_rows": numHeightfieldRows,
        "num_cols": numHeightfieldColumns,
    }
