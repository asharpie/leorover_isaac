# tasks/ — Isaac Lab task configurations

Each task config wires together one environment class, one reward manager,
one event manager (for randomization), one curriculum, and registers a
gym ID that the standard train script can launch.

This is the layer where the PyBullet `config.py` flag combinations
(`agent_mode = "PPO"`, `"Hybrid"`, etc.) become distinct gym tasks.

## Task IDs (planned)

| gym ID | port phase | what it does |
|--------|------------|--------------|
| `Isaac-LeoRover-Flat-v0` | Phase 2 | rover on flat ground, pure PPO, smoke test |
| `Isaac-LeoRover-Mars-v0` | Phase 3 | pure PPO on Mars heightfield + ADR |
| `Isaac-LeoRover-Mars-Hybrid-v0` | Phase 4 | LQR baseline + residual PPO + v33.9 rewards |
| `Isaac-LeoRover-Mars-LQR-v0` | Phase 7 | pure LQR (for comparison runs only, no training) |

## File pattern

For each task ID, two files:

- `leo_rover_<name>_cfg.py` — the `EnvCfg`, `RewardManagerCfg`,
  `EventManagerCfg`, etc. dataclasses.
- registration call in `leorover_isaac/__init__.py` via `gym.register`.

The configs are the place where the PyBullet `agent_mode` switch becomes
explicit — each mode is its own task, sharing the underlying env class.

## Why task configs and not config.py-style globals

Isaac Lab is opinionated about config-as-code via dataclasses, not flat
modules. The big benefit: every task config is a typed object with sane
defaults, and you override per-experiment from the CLI:

```bash
./isaaclab.sh -p train.py \
    --task Isaac-LeoRover-Mars-Hybrid-v0 \
    env.rewards.effort.weight=1.0 \
    env.actions.residual.max_velocity=0.10
```

…which replaces the old "edit config.py, restart training, hope you
remembered to revert" workflow. The v33.9 changes file in the PyBullet
repo is unnecessary in Isaac Lab because every experiment serializes its
own config to disk automatically.
