# scripts/ — Standalone entry-point scripts

CLI scripts users run directly. Thin wrappers around Isaac Lab's standard
train/eval flows with project-specific defaults.

| script | port phase | purpose |
|--------|------------|---------|
| `train.py` | Phase 2 | thin wrapper around Isaac Lab's rsl_rl train.py with leorover defaults |
| `eval.py` | Phase 2 | load a checkpoint, run N episodes, output the same CSV schema as the PyBullet evaluate mode |
| `compare_hybrid_vs_lqr.py` | Phase 7 | port of run_comparison mode — paired-seed evaluation |
| `convert_assets.sh` | Phase 1 | one-liner that runs the URDF→USD conversion |

All stubs until their respective phase.
