# utils/ — Shared utilities

Small helpers that don't belong in any one phase:

| file | purpose |
|------|---------|
| `csv_recorder.py` | Phase 6: write per-episode metrics in the same column schema as the PyBullet `EpisodeMetricsCallback`, including v33.9's `mean_residual_v_norm` / `mean_residual_w_norm` |
| `math_helpers.py` | torch math the env classes share (quat → euler, body-frame transforms, CTE computation) |
| `wandb_setup.py` | Phase 6: wandb config glue |

Stubs for now.
