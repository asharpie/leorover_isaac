# mars_heightfield.py
"""
Mars terrain generation for Isaac Lab.

Two layers:

1. PURE-NUMPY GENERATION (engine-agnostic) — `generate_mars_patch` and friends
   wrap `leorover_isaac.common.mars_terrain_numpy`, which is a faithful port of
   the PyBullet `create_terrain()` Gaussian-hill algorithm (512x512 @ 0.05 m,
   height_scale = intensity/100 * 5). This is the single source of truth for
   terrain shape, shared with the PyBullet repo so slope statistics match.

2. ISAAC LAB TERRAIN IMPORTER (`MARS_TERRAIN_CFG`) — exposes the generation as
   an Isaac Lab sub-terrain so the GPU heightfield collider is built once at
   startup. Isaac Lab's `TerrainGenerator` lays sub-terrains out on a grid where
   the ROW index is "difficulty" in [0, 1]; we map difficulty -> terrain
   intensity (0-100%), which is exactly the ADR curriculum axis. With
   `curriculum=True`, rovers that succeed are promoted to higher-difficulty rows
   — the Isaac-native version of `adr_curriculum.py`'s terrain ramp.

PARITY NOTE: PhysX heightfield/mesh colliders are static after init (see
PORTING_ROADMAP.md Phase 3 gotcha). Per-episode terrain variety therefore comes
from (a) the num_cols variations within each difficulty row, and (b) respawning
the rover onto a different sub-terrain origin each reset — NOT from regenerating
geometry mid-run, which PyBullet did. The slope *distribution* per difficulty
level is preserved; the exact per-episode hill layout differs, which is within
the 2pp parity ceiling.
"""

from __future__ import annotations

import numpy as np

from leorover_isaac.common.mars_terrain_numpy import (
    generate_heightfield,
    heightfield_to_grid,
    get_height_at,           # re-export for spawn / waypoint projection
    friction_from_intensity,
    CELL_SIZE,
    numHeightfieldRows,
    numHeightfieldColumns,
)

__all__ = [
    "generate_mars_patch",
    "mars_height_field",
    "MARS_TERRAIN_CFG",
    "make_mars_terrain_cfg",
    "get_height_at",
    "friction_from_intensity",
]


# --------------------------------------------------------------------------- #
# Pure-numpy generation
# --------------------------------------------------------------------------- #
def generate_mars_patch(
    size_m: float = 8.0,
    resolution_m: float = CELL_SIZE,
    terrain_intensity: float = 50.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate one Mars-like heightfield patch as a [H, W] float32 array.

    Faithful to the PyBullet Gaussian-hill terrain. `size_m` controls how large
    a square patch to crop from the 25.6 m master field; the hills, amplitude
    (height_scale = intensity/100*5) and feature scale match the original.
    """
    full = heightfield_to_grid(generate_heightfield(seed=seed, intensity=terrain_intensity))
    n = max(2, int(round(size_m / resolution_m)))
    n = min(n, full.shape[0])
    # Center crop the requested patch out of the master field.
    r0 = (full.shape[0] - n) // 2
    c0 = (full.shape[1] - n) // 2
    return np.ascontiguousarray(full[r0:r0 + n, c0:c0 + n], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Isaac Lab sub-terrain function
# --------------------------------------------------------------------------- #
def mars_height_field(difficulty: float, cfg) -> np.ndarray:
    """Isaac Lab height-field sub-terrain function.

    Called by Isaac Lab's TerrainGenerator for each grid cell. `difficulty` in
    [0, 1] is mapped to terrain intensity in [intensity_min, intensity_max] %,
    matching the ADR curriculum axis. Returns a 2D int16/float height array
    sized to cfg.size / cfg.horizontal_scale, as Isaac Lab expects from an
    `@height_field_to_mesh`-decorated function.

    `cfg` is expected to carry: size (tuple m), horizontal_scale (m),
    vertical_scale (m), plus our extras `intensity_min`, `intensity_max`,
    and optional `seed`.
    """
    # difficulty in [0,1] -> terrain intensity 0-100% (same difficulty scaling the
    # built-in sub-terrains use; the ADR ceiling limits which difficulty rows are
    # actually sampled at run time, so this is the per-patch steepness).
    intensity = float(difficulty) * 100.0

    horizontal_scale = float(getattr(cfg, "horizontal_scale", CELL_SIZE))
    vertical_scale = float(getattr(cfg, "vertical_scale", 0.005))
    size = getattr(cfg, "size", (20.0, 20.0))
    width_px = max(2, int(size[0] / horizontal_scale))
    length_px = max(2, int(size[1] / horizontal_scale))

    seed = getattr(cfg, "seed", None)
    patch = generate_mars_patch(
        size_m=max(size) ,
        resolution_m=horizontal_scale,
        terrain_intensity=intensity,
        seed=seed,
    )
    # Resize (nearest) to the exact pixel grid Isaac Lab asked for.
    patch = _resize_nearest(patch, (width_px, length_px))
    # Convert meters -> integer units of vertical_scale (Isaac Lab convention).
    return np.rint(patch / vertical_scale).astype(np.int16)


def _resize_nearest(arr: np.ndarray, shape_wh) -> np.ndarray:
    w, h = shape_wh
    xi = (np.linspace(0, arr.shape[0] - 1, w)).round().astype(int)
    yi = (np.linspace(0, arr.shape[1] - 1, h)).round().astype(int)
    return arr[np.ix_(xi, yi)]


# --------------------------------------------------------------------------- #
# Isaac Lab TerrainImporterCfg builder
# --------------------------------------------------------------------------- #
def make_mars_terrain_cfg(
    num_difficulty_rows: int = 20,        # difficulty levels (the ADR ramp axis)
    num_variations: int = 100,            # variations per level -> 20*100 = 2000 patches
    sub_terrain_size: float = 12.0,       # each patch comfortably holds a ~10 m path
    horizontal_scale: float = 0.1,        # 0.1 m cells keep ~2000 patches GPU-feasible
    vertical_scale: float = 0.005,
    intensity_min: float = 0.0,
    intensity_max: float = 100.0,
    curriculum: bool = True,
    static_friction: float = 1.0,
    dynamic_friction: float = 1.0,
):
    """Build an Isaac Lab TerrainImporterCfg with an EXHAUSTIVE terrain bank.

    The bank is `num_difficulty_rows * num_variations` patches (default
    20*100 = 2000, i.e. 10x the previous 200) drawn from ~8 terrain TYPES —
    Mars Gaussian hills, rough noise, up/down slopes, dunes, scattered
    obstacles/rocks, and stairs. Because `use_cache=False`, every cell gets a
    fresh RNG draw, so even same-type patches differ; combined with 20
    difficulty levels this spans essentially any wheeled-robot terrain (for
    anything you can imagine there's a near-match patch the rover can be spawned
    on). To go even larger, raise `num_variations` (e.g. 200 -> 4000 patches) or
    `num_difficulty_rows` — startup time + GPU memory scale with the product, so
    push it as far as your 24 GB allows and dial back if terrain baking OOMs.

    Lazy-imports isaaclab so this module is importable without Isaac installed
    (e.g. for unit tests of the numpy generation). Returns None with a printed
    warning if isaaclab.terrains is unavailable.

    The friction here is a default; per-episode friction-intensity randomization
    (PyBullet's friction 0.3->2.0 sweep) is applied via an EventTerm that
    overrides the physics material — see leorover_isaac/utils/events.py.
    """
    try:
        from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg, HfTerrainBaseCfg
        from isaaclab.terrains.height_field.utils import height_field_to_mesh
        import isaaclab.terrains as terrain_gen  # noqa: F401
        from dataclasses import MISSING
        import isaaclab.sim as sim_utils
    except Exception as exc:  # pragma: no cover - depends on Isaac install
        print(f"[mars_heightfield] isaaclab.terrains unavailable ({exc}); "
              f"returning None. Generation functions still work for tests.")
        return None

    # ── Build an EXHAUSTIVE bank of terrain TYPES ──────────────────────────
    # num_rows*num_cols cells, each assigned a type by `proportion` and given
    # its OWN rng draw, so even same-type patches differ. Difficulty = row index
    # scales each type's amplitude/roughness/slope. With ~8 types x per-cell
    # randomness x difficulty levels, the bank spans essentially any wheeled-
    # robot terrain — for anything you can imagine there's a near-match cell.
    common = dict(
        size=(sub_terrain_size, sub_terrain_size),
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        border_width=0.25,
    )

    sub_terrains = {}

    # Isaac Lab's built-in height-field terrains (robustly supported) form the
    # backbone of the variety: rough noise, up/down slopes, dunes (waves),
    # scattered obstacles/rocks, and stairs.
    try:
        from isaaclab.terrains.height_field import (
            HfRandomUniformTerrainCfg, HfPyramidSlopedTerrainCfg,
            HfInvertedPyramidSlopedTerrainCfg, HfWaveTerrainCfg,
            HfDiscreteObstaclesTerrainCfg, HfPyramidStairsTerrainCfg,
            HfInvertedPyramidStairsTerrainCfg,
        )
        sub_terrains.update({
            "rough":      HfRandomUniformTerrainCfg(proportion=0.22, noise_range=(0.02, 0.14), noise_step=0.02, **common),
            "slope_up":   HfPyramidSlopedTerrainCfg(proportion=0.12, slope_range=(0.0, 0.45), platform_width=2.0, **common),
            "slope_down": HfInvertedPyramidSlopedTerrainCfg(proportion=0.12, slope_range=(0.0, 0.45), platform_width=2.0, **common),
            "dunes":      HfWaveTerrainCfg(proportion=0.14, amplitude_range=(0.05, 0.6), num_waves=4, **common),
            "obstacles":  HfDiscreteObstaclesTerrainCfg(proportion=0.10, obstacle_height_mode="choice", obstacle_width_range=(0.3, 1.6), obstacle_height_range=(0.05, 0.5), num_obstacles=24, platform_width=1.5, **common),
            "stairs_up":  HfPyramidStairsTerrainCfg(proportion=0.05, step_height_range=(0.02, 0.16), step_width=0.3, platform_width=2.0, **common),
            "stairs_dn":  HfInvertedPyramidStairsTerrainCfg(proportion=0.05, step_height_range=(0.02, 0.16), step_width=0.3, platform_width=2.0, **common),
        })
    except Exception as exc:  # pragma: no cover - depends on Isaac version
        print(f"[mars_heightfield] some built-in terrain types unavailable ({exc}).")

    # Our custom Mars Gaussian-hill height field (the parity terrain). Optional:
    # if the custom HF-subterrain registration needs version tweaks, the bank is
    # still rich from the built-ins above.
    try:
        from dataclasses import dataclass

        decorated = height_field_to_mesh(mars_height_field)

        @dataclass
        class MarsHfCfg(HfTerrainBaseCfg):
            function: object = staticmethod(decorated)
            seed: object = None

        sub_terrains["mars_hills"] = MarsHfCfg(proportion=0.30, **common)
    except Exception as exc:  # pragma: no cover
        print(f"[mars_heightfield] custom Mars-hills sub-terrain unavailable ({exc}); "
              f"using built-in types only.")

    if not sub_terrains:
        raise RuntimeError("No terrain sub-types could be constructed.")

    generator = TerrainGeneratorCfg(
        size=(sub_terrain_size, sub_terrain_size),
        border_width=5.0,
        num_rows=num_difficulty_rows,     # difficulty axis (ADR ramp)
        num_cols=num_variations,          # per-difficulty variety
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        slope_threshold=0.75,
        use_cache=False,                  # fresh rng per cell -> max variety
        curriculum=curriculum,
        sub_terrains=sub_terrains,
    )

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=num_difficulty_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        ),
        debug_vis=False,
    )


# A ready-to-use default config (None until isaaclab is importable).
MARS_TERRAIN_CFG = make_mars_terrain_cfg()
