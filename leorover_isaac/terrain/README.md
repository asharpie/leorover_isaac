# terrain/ — Mars heightfield generation

The PyBullet code generated Mars-like heightfields in numpy per episode
inside `environment2.py`. In Isaac Lab, terrain works differently — colliders
are baked at sim startup, not regenerated per episode.

## The Isaac Lab terrain model

- A `TerrainImporter` builds a giant grid of `num_envs × num_envs` terrain
  patches at startup.
- Each parallel env is spawned on one of these patches.
- At episode reset, the rover is teleported to a *different patch* instead
  of regenerating the terrain it's on.
- Terrain diversity comes from generating many patches at startup, not from
  per-episode regeneration.

This is fast (build once, reuse forever) and lossless (every patch is real
geometry the policy can actually learn from), but it's a different mental
model than PyBullet's per-episode regen.

## What lives here

| file | port phase | corresponds to in PyBullet |
|------|------------|----------------------------|
| `mars_heightfield.py` | Phase 3 | terrain generation helpers in environment2.py |
| `terrain_bank.py` | Phase 3 | the new "build N patches at startup" logic |
| `slope_query.py` | Phase 3 | port of calculate_local_slope_at_position |

## Bridging to the ADR curriculum

The PyBullet ADR ramps `terrain_intensity` smoothly. In Isaac Lab, you can't
re-bake terrain mid-training, so the right pattern is:

1. At startup, build a **bank of patches at all intensities** (e.g. 100
   patches per intensity bin from 0% to 100%).
2. At episode reset, the ADR module picks an intensity (using the same
   sampling logic from `adr_curriculum.py`) and the EventTerm assigns the
   rover to a random patch from that intensity bin.

This decouples ADR's progression (which is fast and stochastic) from
terrain mesh generation (which is slow and one-time).
