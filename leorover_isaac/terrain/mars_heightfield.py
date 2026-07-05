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
    resolution_m: float = CELL_SIZE,   # kept for API compat; crop is in MASTER cells
    terrain_intensity: float = 50.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate one Mars-like heightfield patch as a [H, W] float32 array.

    Faithful to the PyBullet Gaussian-hill terrain. `size_m` controls how large
    a square patch to crop from the 25.6 m master field; the hills, amplitude
    (height_scale = intensity/100*AMP) and feature scale match the original.

    BUGFIX (2026-07-05): the crop used to take `size_m / resolution_m` MASTER
    cells (master cell = 0.05 m), so at resolution_m=0.025 a "12 m" patch was
    actually 24 m of terrain relabeled to 12 m -- a 2x horizontal compression
    that DOUBLED every slope vs the PyBullet original. The crop is now in
    master cells (size_m / CELL_SIZE); the caller resamples to its pixel grid.
    """
    full = heightfield_to_grid(generate_heightfield(seed=seed, intensity=terrain_intensity))
    n = max(2, int(round(size_m / CELL_SIZE)))
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

    # PER-CELL seed: Isaac Lab hands every grid cell the SAME generator seed
    # (TerrainGenerator._get_terrain_mesh does cfg.seed = self.cfg.seed), so all
    # cells used one RNG stream -> identical hill LAYOUT everywhere, with only the
    # amplitude scaling by difficulty (and the 2 "variation" columns were copies).
    # Deriving the seed from base_seed + difficulty (unique per cell thanks to the
    # curriculum jitter) gives every patch its own hills while staying fully
    # deterministic and cache-consistent (difficulty is part of the cache hash).
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        seed = (int(seed) + int(round(float(difficulty) * 1e6))) % (2 ** 31)
    patch = generate_mars_patch(
        size_m=max(size) ,
        resolution_m=horizontal_scale,
        terrain_intensity=intensity,
        seed=seed,
    )
    # Resize (nearest) to the exact pixel grid Isaac Lab asked for.
    patch = _resize_nearest(patch, (width_px, length_px))
    # Smooth the nearest-resize / 0.1 m-cell steps so a 6 cm-wheel rover isn't tripped by
    # grid facets. Gaussian sigma in cells (0 disables); blurs local sharp features while
    # keeping the large Gaussian hills. Config default TERRAIN_SMOOTH_SIGMA, env override
    # LEOROVER_TERRAIN_SMOOTH. NOTE: sigma is NOT in the terrain cache key -> clear the
    # cache dir / use LEOROVER_TERRAIN_NOCACHE=1 after changing it (same gotcha as AMP).
    import os as _os_sm
    try:
        import config as _cfg_sm
        _sm_def = float(getattr(_cfg_sm, "TERRAIN_SMOOTH_SIGMA", 1.0))
    except Exception:
        _sm_def = 1.0
    _sm = float(_os_sm.environ.get("LEOROVER_TERRAIN_SMOOTH", _sm_def))
    if _sm > 0.0 and float(intensity) > 0.0:
        patch = _smooth_heightfield(patch, _sm)
    # Convert meters -> integer units of vertical_scale (Isaac Lab convention).
    return np.rint(patch / vertical_scale).astype(np.int16)


def _resize_nearest(arr: np.ndarray, shape_wh) -> np.ndarray:
    w, h = shape_wh
    xi = (np.linspace(0, arr.shape[0] - 1, w)).round().astype(int)
    yi = (np.linspace(0, arr.shape[1] - 1, h)).round().astype(int)
    return arr[np.ix_(xi, yi)]


def _smooth_heightfield(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (pure numpy, no scipy) that softens local grid facets
    while preserving the large Gaussian hills. `sigma` in cells. Edge-padded so borders
    aren't pulled toward zero. Runs at terrain-gen time on a small patch, so speed is fine.
    """
    if sigma <= 0.0:
        return arr
    r = max(1, int(round(3.0 * sigma)))
    xs = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    k = (k / k.sum()).astype(np.float32)
    out = arr.astype(np.float32)
    for axis in (0, 1):
        out = np.apply_along_axis(
            lambda m: np.convolve(np.pad(m, r, mode="edge"), k, mode="valid"), axis, out)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Isaac Lab TerrainImporterCfg builder
# --------------------------------------------------------------------------- #
def make_mars_terrain_cfg(
    num_difficulty_rows: int = None,      # default from config.TERRAIN_NUM_DIFFICULTY_ROWS
    num_variations: int = None,           # default from config.TERRAIN_NUM_VARIATIONS
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

    Bank size + caching come from config.py (TERRAIN_NUM_DIFFICULTY_ROWS,
    TERRAIN_NUM_VARIATIONS, TERRAIN_USE_CACHE) so you can tune them from the GUI /
    EXPERIMENT_OVERRIDES. With caching on, a large bank is generated ONCE and
    reused on later runs. Generation is numpy-vectorized, so even big banks build
    in seconds-to-minutes the first time.
    """
    # ── DIAGNOSTIC: true collision PLANE (geometry-vs-friction bisect) ──────
    # LEOROVER_FLAT_PLANE=1 swaps the whole terrain bank for a single flat ground
    # PLANE — a PhysX half-space that CANNOT be penetrated, dug into, or tunneled
    # through. This isolates "the trimesh terrain causes the stalls" from "it's a
    # friction / kinematics / controller problem a plane would share too":
    #   parking craters on the plane -> the mesh-under-drive is the culprit
    #   parking persists on the plane -> geometry is exonerated; look at traction/LQR
    # Flat only (no hills), so it is a diagnostic, never a shipping terrain.
    import os as _os_plane
    if _os_plane.environ.get("LEOROVER_FLAT_PLANE", "0") not in ("0", "", "false", "False"):
        try:
            from isaaclab.terrains import TerrainImporterCfg
            import isaaclab.sim as sim_utils
        except Exception:
            from omni.isaac.lab.terrains import TerrainImporterCfg
            import omni.isaac.lab.sim as sim_utils
        print("[mars_heightfield] LEOROVER_FLAT_PLANE=1 -> flat collision PLANE "
              "(no hills; geometry-vs-friction bisect)", flush=True)
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
            ),
            debug_vis=False,
        )

    # Resolve bank size + caching from config.py (overridable via EXPERIMENT_OVERRIDES).
    import config as _cfg
    if num_difficulty_rows is None:
        num_difficulty_rows = int(getattr(_cfg, "TERRAIN_NUM_DIFFICULTY_ROWS", 10))
    if num_variations is None:
        num_variations = int(getattr(_cfg, "TERRAIN_NUM_VARIATIONS", 20))
    # Facet size from config (default 0.025 m; the 2026-07-04 wheel-vs-triangle fix). The
    # LEOROVER_TERRAIN_HSCALE env var below still overrides this for runtime sweeps.
    horizontal_scale = float(getattr(_cfg, "TERRAIN_HSCALE", horizontal_scale))
    use_cache = bool(getattr(_cfg, "TERRAIN_USE_CACHE", True))
    # LEOROVER_TERRAIN_NOCACHE=1 forces regeneration. The terrain cache key is the cfg
    # hash, but LEOROVER_TERRAIN_AMP is read INSIDE the generation function, so it is NOT
    # in the key -> changing AMP with the cache ON silently reuses the stale (AMP=5) mesh
    # (this is why an AMP sweep looked like it did nothing). Use NOCACHE while sweeping
    # AMP; once a value is chosen, bake it into config.py AND clear the cache dir once.
    import os as _os_nc
    # config.TERRAIN_AMP is the default hill amplitude; generate_heightfield reads the env
    # var, so publish the config value into it here (an explicit env var still wins).
    _os_nc.environ.setdefault("LEOROVER_TERRAIN_AMP", str(float(getattr(_cfg, "TERRAIN_AMP", 5.0))))
    if _os_nc.environ.get("LEOROVER_TERRAIN_NOCACHE", "0") not in ("0", "", "false", "False"):
        use_cache = False
        print("[mars_heightfield] LEOROVER_TERRAIN_NOCACHE=1 -> terrain cache OFF (regenerating)", flush=True)
    print(f"[mars_heightfield] LEOROVER_TERRAIN_AMP="
          f"{float(_os_nc.environ.get('LEOROVER_TERRAIN_AMP', '5.0'))} m relief per 100% intensity "
          f"(cache {'ON' if use_cache else 'OFF'})", flush=True)

    # Quick-test overrides (no config edit / push needed): shrink the bank or
    # coarsen the mesh straight from the shell. The full 20x100 = 2000-patch bank
    # at 0.1 m cells is a ~57M-triangle mesh that PhysX CANNOT cook -> the terrain
    # gets no collider and the rover falls through into the void. Sweep sizes with
    #   LEOROVER_TERRAIN_ROWS / LEOROVER_TERRAIN_COLS / LEOROVER_TERRAIN_HSCALE
    # to find the largest bank that still cooks, then bake that into config.py.
    import os as _os
    num_difficulty_rows = int(_os.environ.get("LEOROVER_TERRAIN_ROWS", num_difficulty_rows))
    num_variations = int(_os.environ.get("LEOROVER_TERRAIN_COLS", num_variations))
    horizontal_scale = float(_os.environ.get("LEOROVER_TERRAIN_HSCALE", horizontal_scale))
    _tris = int((sub_terrain_size / horizontal_scale) ** 2 * 2)
    print(f"[mars_heightfield] terrain bank: {num_difficulty_rows} x {num_variations} = "
          f"{num_difficulty_rows * num_variations} patches @ {sub_terrain_size} m, "
          f"hscale {horizontal_scale} m  (~{_tris} tris/patch, "
          f"~{_tris * num_difficulty_rows * num_variations / 1e6:.1f}M total)")

    try:
        from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg, HfTerrainBaseCfg
        from isaaclab.terrains.height_field.utils import height_field_to_mesh
        import isaaclab.terrains as terrain_gen  # noqa: F401
        from dataclasses import MISSING
        import isaaclab.sim as sim_utils
    except Exception:
        try:  # Isaac Sim 4.5 / Isaac Lab 1.x
            from omni.isaac.lab.terrains import TerrainGeneratorCfg, TerrainImporterCfg, HfTerrainBaseCfg
            from omni.isaac.lab.terrains.height_field.utils import height_field_to_mesh
            import omni.isaac.lab.terrains as terrain_gen  # noqa: F401
            from dataclasses import MISSING
            import omni.isaac.lab.sim as sim_utils
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
        try:
            from isaaclab.terrains.height_field import (
                HfRandomUniformTerrainCfg, HfPyramidSlopedTerrainCfg,
                HfInvertedPyramidSlopedTerrainCfg, HfWaveTerrainCfg,
                HfDiscreteObstaclesTerrainCfg, HfPyramidStairsTerrainCfg,
                HfInvertedPyramidStairsTerrainCfg,
            )
        except Exception:  # Isaac Sim 4.5 / Isaac Lab 1.x namespace
            from omni.isaac.lab.terrains.height_field import (
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

    # ── DIAGNOSTIC / FAITHFULNESS: restrict the bank to the Mars Gaussian hills ──
    # The full bank mixes in rough/slope/dune/obstacle/stair patches whose feature
    # heights (rough up to 0.14 m, obstacles up to 0.5 m, stairs up to 0.16 m, dunes
    # up to 0.6 m) do NOT shrink at low difficulty, so even "row 0" is NOT flat and
    # beaches the 0.0625 m-wheel rover -- and PyBullet only ever used the smooth
    # Gaussian-hill terrain, so the mixed bank is BOTH harder than the baseline and
    # not apples-to-apples. LEOROVER_HILLS_ONLY=1 keeps only the parity terrain:
    # row 0 is then a genuinely FLAT mesh (isolates "is the trimesh collider itself
    # the bug" -> compare to the 91% plane) and higher rows are the faithful hills.
    import os as _os_h
    _hills_default = "1" if bool(getattr(_cfg, "TERRAIN_HILLS_ONLY", False)) else "0"
    if _os_h.environ.get("LEOROVER_HILLS_ONLY", _hills_default) not in ("0", "", "false", "False"):
        if "mars_hills" in sub_terrains:
            sub_terrains = {"mars_hills": sub_terrains["mars_hills"]}
            print("[mars_heightfield] LEOROVER_HILLS_ONLY=1 -> bank restricted to Mars "
                  "Gaussian hills (PyBullet-faithful; row 0 is flat)", flush=True)
        else:
            print("[mars_heightfield] LEOROVER_HILLS_ONLY=1 requested but the mars_hills "
                  "sub-terrain is unavailable; keeping the full bank", flush=True)

    if not sub_terrains:
        raise RuntimeError("No terrain sub-types could be constructed.")

    # Fold AMP + smoothing sigma INTO the seed so the terrain disk-cache key (a hash of this
    # cfg, seed included) invalidates when they change. Otherwise the cache silently serves a
    # stale mesh after an AMP/smooth tweak -- which bit us hard: trained on smoothed terrain,
    # evaluated on the old cached mesh -> a bogus 36% eval. Now train and eval agree without
    # needing LEOROVER_TERRAIN_NOCACHE. (Changing AMP/smooth also reshuffles the hill RNG, fine.)
    _amp_key = float(_os.environ.get("LEOROVER_TERRAIN_AMP", getattr(_cfg, "TERRAIN_AMP", 5.0)))
    _sm_key = float(_os.environ.get("LEOROVER_TERRAIN_SMOOTH", getattr(_cfg, "TERRAIN_SMOOTH_SIGMA", 0.0)))
    # _GEN_REV: bump on ANY generation-code change. The Isaac Lab disk cache hashes the
    # sub-terrain CFG but never the generator FUNCTION BODY (callable_to_string is just
    # "module:name"), so a code fix silently keeps serving meshes built by the OLD code.
    # Folding a revision int into the seed invalidates every stale entry. rev 2 =
    # 2026-07-05 (crop-compression fix + per-cell seeds).
    _GEN_REV = 2
    _seed = (int(getattr(_cfg, "TERRAIN_SEED", 42)) + int(round(_amp_key * 100))
             + int(round(_sm_key * 100)) * 1000 + _GEN_REV * 1_000_000)
    generator = TerrainGeneratorCfg(
        seed=_seed,  # base seed + AMP + smooth-sigma (cache invalidates when terrain params change)
        size=(sub_terrain_size, sub_terrain_size),
        border_width=5.0,
        num_rows=num_difficulty_rows,     # difficulty axis (ADR ramp)
        num_cols=num_variations,          # per-difficulty variety
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        slope_threshold=0.75,
        use_cache=use_cache,              # generate once -> cache to disk -> fast reuse
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
