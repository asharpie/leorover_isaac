"""Shared utilities: CSV-schema-compatible episode recorder + ADR glue.

The recorder produces an `episode_metrics.csv` with the EXACT column schema of
the PyBullet `MetricsCallback` (run_experiment.py), so the carried-over
`evaluate_training.py` visualization GUI and `analyze_*.py` work unchanged on
Isaac Lab runs.
"""

from .recorder import EpisodeMetricsRecorder  # noqa: F401
