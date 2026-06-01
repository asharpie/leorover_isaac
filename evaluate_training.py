#!/usr/bin/env python3
"""
evaluate_training.py - GUI for analyzing and visualizing training session results

Usage:
    python evaluate_training.py                    # Launch GUI
    python evaluate_training.py <path_to_csv>      # Launch GUI with file pre-loaded
    python evaluate_training.py --compare --lqr <lqr_csv> --hybrid <hybrid_csv>  # Compare two datasets

Features:
    - Load training CSV files or paste CSV data
    - Statistical summary of training performance
    - Learning curves (MCTE, reward, success over episodes)
    - Performance vs terrain/friction intensity
    - Binned performance analysis by parameter ranges
    - Convergence analysis
    - **NEW** Box & whisker comparison plots for LQR vs Hybrid
    - Export plots to PNG/PDF
    - Export text reports
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from datetime import datetime
from io import StringIO

# GUI imports
try:
    import tkinter as tk
    from tkinter import filedialog, scrolledtext, ttk, messagebox

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    tk = None

# Statistical imports
from scipy import stats

# =============================================================================
# CONFIGURATION FOR PUBLICATION-QUALITY FIGURES
# =============================================================================

# Figure settings for publication quality
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Font sizes - LARGE for legibility (feedback requirement)
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
ANNOTATION_FONTSIZE = 11

# Colors for the two controllers
LQR_COLOR = '#E74C3C'  # Red
HYBRID_COLOR = '#3498DB'  # Blue

# Default intensity caps (feedback requirement: terrain max 70%, friction min 30%)
DEFAULT_TERRAIN_MIN = 0
DEFAULT_TERRAIN_MAX = 70
DEFAULT_FRICTION_MIN = 30
DEFAULT_FRICTION_MAX = 100

# Default Y-axis limits for CTE (fixed for consistency across plots)
DEFAULT_CTE_Y_MIN = 0.0
DEFAULT_CTE_Y_MAX = 4.0

# Box plot settings
SHOW_OUTLIERS = False  # Set to True to show outlier points


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(csv_path: str) -> pd.DataFrame:
    """Load training CSV data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def load_from_string(csv_string: str) -> pd.DataFrame:
    """Load training data from pasted CSV string."""
    df = pd.read_csv(StringIO(csv_string))
    return df


def load_screening_data(path: str) -> pd.DataFrame:
    """
    Load screening episode data from CSV file or session directory.

    Args:
        path: Path to screening_episodes.csv or session directory

    Returns:
        pandas DataFrame with episode data
    """
    if os.path.isdir(path):
        # Look for screening_episodes.csv in directory
        csv_path = os.path.join(path, "screening_episodes.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find screening_episodes.csv in {path}")
        path = csv_path

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    # Ensure numeric columns
    numeric_cols = ['terrain_intensity', 'friction_intensity',
                    'mean_cross_track_error', 'success', 'mean_slip']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# =============================================================================
# STATISTICS
# =============================================================================

def generate_statistics_report(df: pd.DataFrame) -> str:
    """Generate comprehensive statistics summary as a string."""
    lines = []
    lines.append("=" * 70)
    lines.append("TRAINING STATISTICS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Episodes: {len(df)}")
    lines.append(f"Columns: {list(df.columns)}")

    # Determine CTE column
    cte_col = None
    for col in ['mean_cross_track_error', 'cross_track_error', 'mean_cte']:
        if col in df.columns:
            cte_col = col
            break

    # CTE Statistics
    if cte_col:
        lines.append(f"\n--- Cross-Track Error ({cte_col}) ---")
        lines.append(f"  Mean:   {df[cte_col].mean():.4f} m")
        lines.append(f"  Std:    {df[cte_col].std():.4f} m")
        lines.append(f"  Min:    {df[cte_col].min():.4f} m")
        lines.append(f"  Max:    {df[cte_col].max():.4f} m")
        lines.append(f"  Median: {df[cte_col].median():.4f} m")

        # Percentiles
        lines.append(f"  25th percentile: {df[cte_col].quantile(0.25):.4f} m")
        lines.append(f"  75th percentile: {df[cte_col].quantile(0.75):.4f} m")

    # Success Statistics
    if 'success' in df.columns:
        success_rate = df['success'].mean() * 100
        lines.append(f"\n--- Success Rate ---")
        lines.append(f"  Overall: {success_rate:.1f}%")
        lines.append(f"  Total successes: {df['success'].sum()} / {len(df)}")

        # Success rate over time (first half vs second half)
        mid = len(df) // 2
        if mid > 0:
            first_half = df['success'].iloc[:mid].mean() * 100
            second_half = df['success'].iloc[mid:].mean() * 100
            lines.append(f"  First half:  {first_half:.1f}%")
            lines.append(f"  Second half: {second_half:.1f}%")
            lines.append(f"  Improvement: {second_half - first_half:+.1f}%")

    # Reward Per Step Statistics
    reward_per_step = _get_reward_per_step(df)
    if reward_per_step is not None:
        lines.append(f"\n--- Reward Per Step ---")
        lines.append(f"  Mean:   {reward_per_step.mean():.4f}")
        lines.append(f"  Std:    {reward_per_step.std():.4f}")
        lines.append(f"  Min:    {reward_per_step.min():.4f}")
        lines.append(f"  Max:    {reward_per_step.max():.4f}")
        mid = len(reward_per_step) // 2
        if mid > 0:
            first_half = reward_per_step.iloc[:mid].mean()
            second_half = reward_per_step.iloc[mid:].mean()
            lines.append(f"  First half:  {first_half:.4f}")
            lines.append(f"  Second half: {second_half:.4f}")
            lines.append(f"  Improvement: {second_half - first_half:+.4f}")

    # Episode Length
    if 'steps' in df.columns:
        SIM_TIME_PER_STEP = 0.2  # seconds (10 sub-steps × 1/50s)
        dur = df['steps'] * SIM_TIME_PER_STEP
        lines.append(f"\n--- Episode Duration ---")
        lines.append(f"  Mean steps:     {df['steps'].mean():.1f}")
        lines.append(f"  Std steps:      {df['steps'].std():.1f}")
        lines.append(f"  Min steps:      {df['steps'].min()}")
        lines.append(f"  Max steps:      {df['steps'].max()}")
        lines.append(f"  Mean duration:  {dur.mean():.1f}s ({dur.mean()/60:.1f} min)")
        lines.append(f"  Median duration:{dur.median():.1f}s ({dur.median()/60:.1f} min)")

        if 'success' in df.columns:
            succ_dur = df[df['success'] == 1]['steps'] * SIM_TIME_PER_STEP
            fail_dur = df[df['success'] == 0]['steps'] * SIM_TIME_PER_STEP
            if len(succ_dur) > 0:
                lines.append(f"  Success avg:    {succ_dur.mean():.1f}s ({succ_dur.mean()/60:.1f} min)")
            if len(fail_dur) > 0:
                lines.append(f"  Failure avg:    {fail_dur.mean():.1f}s ({fail_dur.mean()/60:.1f} min)")

    # Path Progress (if available)
    if 'path_progress' in df.columns:
        lines.append(f"\n--- Path Progress ---")
        lines.append(f"  Mean: {df['path_progress'].mean():.1f}%")
        lines.append(f"  Completion rate (>95%): {(df['path_progress'] > 95).mean() * 100:.1f}%")

    # Terrain Statistics (if available)
    if 'terrain_intensity' in df.columns:
        lines.append(f"\n--- Terrain Intensity ---")
        lines.append(f"  Mean:  {df['terrain_intensity'].mean():.1f}%")
        lines.append(f"  Range: {df['terrain_intensity'].min():.1f}% - {df['terrain_intensity'].max():.1f}%")

    if 'friction_intensity' in df.columns:
        lines.append(f"\n--- Friction Intensity ---")
        lines.append(f"  Mean:  {df['friction_intensity'].mean():.1f}%")
        lines.append(f"  Range: {df['friction_intensity'].min():.1f}% - {df['friction_intensity'].max():.1f}%")

    # Terrain Slope Statistics (if available)
    if 'terrain_max_slope_deg' in df.columns:
        lines.append(f"\n--- Terrain Slope (Measured) ---")
        lines.append(f"  Max Slope:     Mean={df['terrain_max_slope_deg'].mean():.2f}Â°, "
                     f"Range={df['terrain_max_slope_deg'].min():.2f}Â°-{df['terrain_max_slope_deg'].max():.2f}Â°")
    if 'terrain_avg_slope_deg' in df.columns:
        lines.append(f"  Avg Slope:     Mean={df['terrain_avg_slope_deg'].mean():.2f}Â°, "
                     f"Range={df['terrain_avg_slope_deg'].min():.2f}Â°-{df['terrain_avg_slope_deg'].max():.2f}Â°")
    if 'mean_local_slope_deg' in df.columns:
        lines.append(f"  Local Slope:   Mean={df['mean_local_slope_deg'].mean():.2f}Â° (under rover)")

    # Roll/Pitch (if available)
    if 'roll_max' in df.columns:
        lines.append(f"\n--- Rover Tilt (Max per Episode) ---")
        lines.append(f"  Mean max roll:  {np.degrees(df['roll_max'].mean()):.2f}Â°")
        lines.append(f"  Mean max pitch: {np.degrees(df['pitch_max'].mean()):.2f}Â°")

    # Convergence Analysis
    lines.append(f"\n--- Convergence Analysis ---")
    n_segments = 5
    segment_size = len(df) // n_segments
    if segment_size > 0 and cte_col:
        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else len(df)
            segment_cte = df[cte_col].iloc[start:end].mean()
            segment_success = df['success'].iloc[start:end].mean() * 100 if 'success' in df.columns else 0
            lines.append(
                f"  Segment {i + 1} (ep {start + 1}-{end}): CTE={segment_cte:.4f}m, Success={segment_success:.1f}%")

    # Recommendations
    lines.append(f"\n" + "-" * 70)
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 70)

    if 'success' in df.columns:
        success_rate = df['success'].mean() * 100
        if success_rate < 50:
            lines.append("âš  Low success rate (<50%). Consider:")
            lines.append("  - Increasing training episodes")
            lines.append("  - Reducing terrain difficulty")
            lines.append("  - Adjusting reward weights")
        elif success_rate < 80:
            lines.append("â— Moderate success rate (50-80%). Consider:")
            lines.append("  - More training episodes")
            lines.append("  - Fine-tuning hyperparameters")
        else:
            lines.append("âœ“ Good success rate (>80%).")

    if cte_col:
        mean_cte = df[cte_col].mean()
        if mean_cte > 0.5:
            lines.append("âš  High average CTE (>0.5m). Path following needs improvement.")
        elif mean_cte > 0.2:
            lines.append("â— Moderate CTE (0.2-0.5m). Room for improvement.")
        else:
            lines.append("âœ“ Low CTE (<0.2m). Good path following.")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def generate_comparison_report(lqr_df: pd.DataFrame, hybrid_df: pd.DataFrame,
                               terrain_max: float = DEFAULT_TERRAIN_MAX) -> str:
    """Generate comparison report for LQR vs Hybrid."""
    # Filter to common terrain range
    lqr_filtered = lqr_df[
        lqr_df['terrain_intensity'] <= terrain_max] if 'terrain_intensity' in lqr_df.columns else lqr_df
    hybrid_filtered = hybrid_df[
        hybrid_df['terrain_intensity'] <= terrain_max] if 'terrain_intensity' in hybrid_df.columns else hybrid_df

    # Determine CTE column
    cte_col = 'mean_cross_track_error'
    if cte_col not in lqr_df.columns:
        cte_col = 'cross_track_error' if 'cross_track_error' in lqr_df.columns else 'mean_cte'

    # Steps column
    steps_col = 'steps' if 'steps' in lqr_filtered.columns else 'episode_length'

    # Simulated time per action step
    SIM_TIME_PER_STEP = 0.2  # seconds

    # Terrain band definitions
    terrain_bands = [("Easy <25%", 0, 25), ("Med 25-50%", 25, 50), ("Hard 50%+", 50, 100)]

    # Helper: get filtered subsets by terrain band and success
    def get_band(df, tmin, tmax, success_only=False):
        mask = (df['terrain_intensity'] >= tmin) & (df['terrain_intensity'] < tmax)
        if success_only:
            mask = mask & (df['success'] == 1)
        return df[mask]

    lines = []
    lines.append("=" * 70)
    lines.append("PERFORMANCE COMPARISON: LQR vs HYBRID")
    lines.append(f"(Terrain intensity capped at {terrain_max}%)")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # === OVERALL SUMMARY ===
    lines.append(f"\n{'Metric':<30} {'Pure LQR':<20} {'Hybrid LQR+PPO':<20}")
    lines.append("-" * 70)
    lines.append(f"{'Episodes':<30} {len(lqr_filtered):<20} {len(hybrid_filtered):<20}")

    has_cte = cte_col in lqr_filtered.columns and cte_col in hybrid_filtered.columns
    has_success = 'success' in lqr_filtered.columns and 'success' in hybrid_filtered.columns
    has_steps = steps_col in lqr_filtered.columns and steps_col in hybrid_filtered.columns
    has_terrain = 'terrain_intensity' in lqr_filtered.columns

    if has_success:
        lqr_success = lqr_filtered['success'].mean() * 100
        hybrid_success = hybrid_filtered['success'].mean() * 100
        lines.append(f"{'Success Rate (%)':<30} {lqr_success:<20.1f} {hybrid_success:<20.1f}")

    if has_cte:
        lqr_cte_mean = lqr_filtered[cte_col].mean()
        hybrid_cte_mean = hybrid_filtered[cte_col].mean()
        lines.append(f"{'Mean CTE (m)':<30} {lqr_cte_mean:<20.4f} {hybrid_cte_mean:<20.4f}")
        lines.append(f"{'Median CTE (m)':<30} {lqr_filtered[cte_col].median():<20.4f} {hybrid_filtered[cte_col].median():<20.4f}")
        lines.append(f"{'Max CTE (m)':<30} {lqr_filtered[cte_col].max():<20.4f} {hybrid_filtered[cte_col].max():<20.4f}")

    if has_steps:
        lqr_dur_mean = lqr_filtered[steps_col].mean() * SIM_TIME_PER_STEP
        hybrid_dur_mean = hybrid_filtered[steps_col].mean() * SIM_TIME_PER_STEP
        lines.append(f"{'Mean Duration (s)':<30} {lqr_dur_mean:<20.1f} {hybrid_dur_mean:<20.1f}")

    # === IMPROVEMENT SUMMARY ===
    lines.append("\n" + "-" * 70)
    lines.append("IMPROVEMENT SUMMARY (Hybrid vs LQR)")
    lines.append("-" * 70)

    if has_success:
        lines.append(f"  Success Rate:  {hybrid_success - lqr_success:+.1f} percentage points")
    if has_cte and lqr_cte_mean > 0:
        cte_pct = ((lqr_cte_mean - hybrid_cte_mean) / lqr_cte_mean) * 100
        lines.append(f"  Mean CTE:      {cte_pct:+.1f}% {'(better)' if cte_pct > 0 else '(worse)'}")
    if has_steps and lqr_dur_mean > 0:
        dur_pct = ((lqr_dur_mean - hybrid_dur_mean) / lqr_dur_mean) * 100
        lines.append(f"  Duration:      {dur_pct:+.1f}% {'(faster)' if dur_pct > 0 else '(slower)'}")

    # === BREAKDOWN BY TERRAIN DIFFICULTY ===
    if has_terrain and has_success and has_cte and has_steps:
        lines.append("\n" + "-" * 70)
        lines.append("BREAKDOWN BY TERRAIN DIFFICULTY (all episodes)")
        lines.append("-" * 70)
        lines.append(f"{'Terrain':<14} {'':>3} {'Episodes':>10} {'Success':>10} {'Mean CTE':>10} {'Duration':>12}")
        lines.append(f"{'':14} {'':>3} {'':>10} {'Rate':>10} {'(m)':>10} {'(s)':>12}")
        lines.append("-" * 70)

        for label, tmin, tmax in terrain_bands:
            lqr_band = get_band(lqr_filtered, tmin, tmax)
            hybrid_band = get_band(hybrid_filtered, tmin, tmax)

            if len(lqr_band) > 0:
                lqr_sr = lqr_band['success'].mean() * 100
                lqr_cte = lqr_band[cte_col].mean()
                lqr_dur = lqr_band[steps_col].mean() * SIM_TIME_PER_STEP
                lines.append(f"{label:<14} {'LQR':>3} {len(lqr_band):>10} {lqr_sr:>9.1f}% {lqr_cte:>10.4f} {lqr_dur:>11.1f}s")

            if len(hybrid_band) > 0:
                hyb_sr = hybrid_band['success'].mean() * 100
                hyb_cte = hybrid_band[cte_col].mean()
                hyb_dur = hybrid_band[steps_col].mean() * SIM_TIME_PER_STEP
                lines.append(f"{'':14} {'PPO':>3} {len(hybrid_band):>10} {hyb_sr:>9.1f}% {hyb_cte:>10.4f} {hyb_dur:>11.1f}s")

            # Show deltas
            if len(lqr_band) > 0 and len(hybrid_band) > 0:
                sr_delta = hyb_sr - lqr_sr
                cte_delta_pct = ((lqr_cte - hyb_cte) / lqr_cte * 100) if lqr_cte > 0 else 0
                dur_delta_pct = ((lqr_dur - hyb_dur) / lqr_dur * 100) if lqr_dur > 0 else 0
                lines.append(f"{'':14} {'Δ':>3} {'':>10} {sr_delta:>+9.1f}pp {cte_delta_pct:>+9.1f}% {dur_delta_pct:>+10.1f}%")
            lines.append("")

    # === SUCCESSFUL EPISODES ONLY ===
    if has_success and has_cte and has_steps:
        lqr_succ = lqr_filtered[lqr_filtered['success'] == 1]
        hybrid_succ = hybrid_filtered[hybrid_filtered['success'] == 1]

        if len(lqr_succ) > 0 and len(hybrid_succ) > 0:
            lines.append("-" * 70)
            lines.append("SUCCESSFUL EPISODES ONLY")
            lines.append("-" * 70)
            lines.append(f"{'Metric':<30} {'Pure LQR':<20} {'Hybrid LQR+PPO':<20}")
            lines.append("-" * 70)
            lines.append(f"{'Episodes':<30} {len(lqr_succ):<20} {len(hybrid_succ):<20}")
            lines.append(f"{'Mean CTE (m)':<30} {lqr_succ[cte_col].mean():<20.4f} {hybrid_succ[cte_col].mean():<20.4f}")
            lines.append(f"{'Median CTE (m)':<30} {lqr_succ[cte_col].median():<20.4f} {hybrid_succ[cte_col].median():<20.4f}")

            lqr_succ_dur = lqr_succ[steps_col].mean() * SIM_TIME_PER_STEP
            hyb_succ_dur = hybrid_succ[steps_col].mean() * SIM_TIME_PER_STEP
            lines.append(f"{'Mean Duration (s)':<30} {lqr_succ_dur:<20.1f} {hyb_succ_dur:<20.1f}")
            lines.append(f"{'Median Duration (s)':<30} {(lqr_succ[steps_col].median() * SIM_TIME_PER_STEP):<20.1f} {(hybrid_succ[steps_col].median() * SIM_TIME_PER_STEP):<20.1f}")

            # Terrain breakdown for successful episodes
            if has_terrain:
                lines.append(f"\n{'Terrain':<14} {'':>3} {'Episodes':>10} {'Mean CTE':>10} {'Duration':>12}")
                lines.append("-" * 70)

                for label, tmin, tmax in terrain_bands:
                    lqr_band = get_band(lqr_filtered, tmin, tmax, success_only=True)
                    hybrid_band = get_band(hybrid_filtered, tmin, tmax, success_only=True)

                    if len(lqr_band) > 0:
                        lines.append(f"{label:<14} {'LQR':>3} {len(lqr_band):>10} {lqr_band[cte_col].mean():>10.4f} {(lqr_band[steps_col].mean() * SIM_TIME_PER_STEP):>11.1f}s")
                    if len(hybrid_band) > 0:
                        lines.append(f"{'':14} {'PPO':>3} {len(hybrid_band):>10} {hybrid_band[cte_col].mean():>10.4f} {(hybrid_band[steps_col].mean() * SIM_TIME_PER_STEP):>11.1f}s")

                    if len(lqr_band) > 0 and len(hybrid_band) > 0:
                        l_cte = lqr_band[cte_col].mean()
                        h_cte = hybrid_band[cte_col].mean()
                        l_dur = lqr_band[steps_col].mean() * SIM_TIME_PER_STEP
                        h_dur = hybrid_band[steps_col].mean() * SIM_TIME_PER_STEP
                        cte_d = ((l_cte - h_cte) / l_cte * 100) if l_cte > 0 else 0
                        dur_d = ((l_dur - h_dur) / l_dur * 100) if l_dur > 0 else 0
                        lines.append(f"{'':14} {'Δ':>3} {'':>10} {cte_d:>+9.1f}% {dur_d:>+10.1f}%")
                    lines.append("")

    # === FAILED EPISODES ===
    if has_success:
        lqr_fail = lqr_filtered[lqr_filtered['success'] == 0]
        hybrid_fail = hybrid_filtered[hybrid_filtered['success'] == 0]

        if len(lqr_fail) > 0 and len(hybrid_fail) > 0:
            lines.append("-" * 70)
            lines.append("FAILED EPISODES")
            lines.append("-" * 70)
            lines.append(f"{'Metric':<30} {'Pure LQR':<20} {'Hybrid LQR+PPO':<20}")
            lines.append("-" * 70)
            lines.append(f"{'Episodes':<30} {len(lqr_fail):<20} {len(hybrid_fail):<20}")

            if has_cte:
                lines.append(f"{'Mean CTE (m)':<30} {lqr_fail[cte_col].mean():<20.4f} {hybrid_fail[cte_col].mean():<20.4f}")
            if has_steps:
                lines.append(f"{'Mean Duration (s)':<30} {(lqr_fail[steps_col].mean() * SIM_TIME_PER_STEP):<20.1f} {(hybrid_fail[steps_col].mean() * SIM_TIME_PER_STEP):<20.1f}")
            if 'path_progress' in lqr_fail.columns:
                lines.append(f"{'Mean Progress (%)':<30} {lqr_fail['path_progress'].mean():<20.1f} {hybrid_fail['path_progress'].mean():<20.1f}")

    lines.append("=" * 70)
    return "\n".join(lines)


# =============================================================================
# PLOTTING FUNCTIONS - ORIGINAL
# =============================================================================

def _get_reward_per_step(df: pd.DataFrame):
    """Get reward-per-step series from a DataFrame.

    Prefers mean_reward_per_step column if available, otherwise computes
    total_reward / steps.
    """
    if 'mean_reward_per_step' in df.columns:
        return df['mean_reward_per_step'].astype(float)
    elif 'total_reward' in df.columns and 'steps' in df.columns:
        return df['total_reward'].astype(float) / df['steps'].clip(lower=1).astype(float)
    return None


def _add_trend_line(ax, x, y, color='red', linestyle='--', linewidth=1.5, label_prefix='Trend'):
    """Add a linear trend line to an axes, handling NaNs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return
    x_clean = x[mask].values if hasattr(x, 'values') else np.asarray(x)[mask]
    y_clean = y[mask].values if hasattr(y, 'values') else np.asarray(y)[mask]
    slope, intercept, r_value, _, _ = stats.linregress(x_clean, y_clean)
    x_line = np.array([x_clean.min(), x_clean.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=color, linestyle=linestyle, linewidth=linewidth,
            label=f'{label_prefix} (slope={slope:.4f}, R\u00b2={r_value**2:.3f})')


def compute_rolling_stats(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Compute rolling statistics for smoothed plots."""
    stats_df = pd.DataFrame()
    stats_df['episode'] = df['episode'] if 'episode' in df.columns else range(len(df))

    cte_col = None
    for col in ['mean_cross_track_error', 'cross_track_error', 'mean_cte']:
        if col in df.columns:
            cte_col = col
            break

    if cte_col:
        stats_df['cte_rolling'] = df[cte_col].rolling(window=window, min_periods=1).mean()
        stats_df['cte_rolling_std'] = df[cte_col].rolling(window=window, min_periods=1).std()

    # Reward per step: use mean_reward_per_step if available, else compute from total_reward/steps
    reward_per_step = _get_reward_per_step(df)
    if reward_per_step is not None:
        stats_df['rps_rolling'] = reward_per_step.rolling(window=window, min_periods=1).mean()
        stats_df['rps_rolling_std'] = reward_per_step.rolling(window=window, min_periods=1).std()

    if 'success' in df.columns:
        stats_df['success_rolling'] = df['success'].rolling(window=window, min_periods=1).mean() * 100

    return stats_df


def plot_learning_curves(df: pd.DataFrame, window: int = 50):
    """Plot learning curves (MCTE, reward, success over episodes)."""
    stats_df = compute_rolling_stats(df, window)

    cte_col = None
    for col in ['mean_cross_track_error', 'cross_track_error', 'mean_cte']:
        if col in df.columns:
            cte_col = col
            break

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Learning Curves', fontsize=TITLE_FONTSIZE, fontweight='bold')

    episodes = stats_df['episode']

    # Get reward per step data
    reward_per_step = _get_reward_per_step(df)

    # Plot 1: CTE over episodes
    ax1 = axes[0, 0]
    if cte_col:
        ax1.plot(episodes, df[cte_col], alpha=0.3, color='blue', label='Raw')
        ax1.plot(episodes, stats_df['cte_rolling'], color='blue', linewidth=2,
                 label=f'Rolling Mean (w={window})')
        if 'cte_rolling_std' in stats_df.columns:
            ax1.fill_between(episodes,
                             stats_df['cte_rolling'] - stats_df['cte_rolling_std'],
                             stats_df['cte_rolling'] + stats_df['cte_rolling_std'],
                             alpha=0.2, color='blue')
        _add_trend_line(ax1, episodes, df[cte_col], color='red')
    ax1.set_xlabel('Episode', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Cross-Track Error (m)', fontsize=LABEL_FONTSIZE)
    ax1.set_title('CTE Over Training', fontsize=TITLE_FONTSIZE - 2)
    ax1.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax1.tick_params(labelsize=TICK_FONTSIZE)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reward per step over episodes
    ax2 = axes[0, 1]
    if reward_per_step is not None:
        ax2.plot(episodes, reward_per_step, alpha=0.3, color='green', label='Raw')
        if 'rps_rolling' in stats_df.columns:
            ax2.plot(episodes, stats_df['rps_rolling'], color='green', linewidth=2,
                     label=f'Rolling Mean (w={window})')
            if 'rps_rolling_std' in stats_df.columns:
                ax2.fill_between(episodes,
                                 stats_df['rps_rolling'] - stats_df['rps_rolling_std'],
                                 stats_df['rps_rolling'] + stats_df['rps_rolling_std'],
                                 alpha=0.2, color='green')
        _add_trend_line(ax2, episodes, reward_per_step, color='red')
    ax2.set_xlabel('Episode', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('Reward Per Step', fontsize=LABEL_FONTSIZE)
    ax2.set_title('Reward Per Step Over Training', fontsize=TITLE_FONTSIZE - 2)
    ax2.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax2.tick_params(labelsize=TICK_FONTSIZE)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Success rate over episodes
    ax3 = axes[1, 0]
    if 'success' in df.columns:
        ax3.plot(episodes, stats_df['success_rolling'], color='purple', linewidth=2)
        ax3.axhline(y=df['success'].mean() * 100, color='red', linestyle='--',
                    label=f'Overall: {df["success"].mean() * 100:.1f}%')
        _add_trend_line(ax3, episodes, stats_df['success_rolling'], color='darkred')
    ax3.set_xlabel('Episode', fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel('Success Rate (%)', fontsize=LABEL_FONTSIZE)
    ax3.set_title('Success Rate Over Training', fontsize=TITLE_FONTSIZE - 2)
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax3.tick_params(labelsize=TICK_FONTSIZE)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Episode length over time
    ax4 = axes[1, 1]
    if 'steps' in df.columns:
        steps_rolling = df['steps'].rolling(window=window, min_periods=1).mean()
        ax4.plot(episodes, df['steps'], alpha=0.3, color='orange', label='Raw')
        ax4.plot(episodes, steps_rolling, color='orange', linewidth=2,
                 label=f'Rolling Mean (w={window})')
        _add_trend_line(ax4, episodes, df['steps'].astype(float), color='red')
    ax4.set_xlabel('Episode', fontsize=LABEL_FONTSIZE)
    ax4.set_ylabel('Episode Length (steps)', fontsize=LABEL_FONTSIZE)
    ax4.set_title('Episode Length Over Training', fontsize=TITLE_FONTSIZE - 2)
    ax4.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax4.tick_params(labelsize=TICK_FONTSIZE)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_terrain_analysis(df: pd.DataFrame):
    """Plot performance vs terrain/friction intensity."""
    has_terrain = 'terrain_intensity' in df.columns
    has_friction = 'friction_intensity' in df.columns

    cte_col = None
    for col in ['mean_cross_track_error', 'cross_track_error', 'mean_cte']:
        if col in df.columns:
            cte_col = col
            break

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance vs Terrain Parameters', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Plot 1: CTE vs Terrain Intensity
    ax1 = axes[0, 0]
    if has_terrain and cte_col:
        ax1.scatter(df['terrain_intensity'], df[cte_col], alpha=0.3, s=10)
        z = np.polyfit(df['terrain_intensity'], df[cte_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['terrain_intensity'].min(), df['terrain_intensity'].max(), 100)
        ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.4f})')
        ax1.set_xlabel('Terrain Intensity (%)', fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Cross-Track Error (m)', fontsize=LABEL_FONTSIZE)
        ax1.set_title('CTE vs Terrain Intensity', fontsize=TITLE_FONTSIZE - 2)
        ax1.legend(fontsize=LEGEND_FONTSIZE - 2)
        ax1.tick_params(labelsize=TICK_FONTSIZE)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No terrain data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('CTE vs Terrain Intensity', fontsize=TITLE_FONTSIZE - 2)

    # Plot 2: CTE vs Friction Intensity
    ax2 = axes[0, 1]
    if has_friction and cte_col:
        ax2.scatter(df['friction_intensity'], df[cte_col], alpha=0.3, s=10, color='green')
        z = np.polyfit(df['friction_intensity'], df[cte_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['friction_intensity'].min(), df['friction_intensity'].max(), 100)
        ax2.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.4f})')
        ax2.set_xlabel('Friction Intensity (%)', fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel('Cross-Track Error (m)', fontsize=LABEL_FONTSIZE)
        ax2.set_title('CTE vs Friction Intensity', fontsize=TITLE_FONTSIZE - 2)
        ax2.legend(fontsize=LEGEND_FONTSIZE - 2)
        ax2.tick_params(labelsize=TICK_FONTSIZE)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No friction data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('CTE vs Friction Intensity', fontsize=TITLE_FONTSIZE - 2)

    # Plot 3: Success Rate vs Terrain (binned)
    ax3 = axes[1, 0]
    if has_terrain and 'success' in df.columns:
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
        df_copy = df.copy()
        df_copy['terrain_bin'] = pd.cut(df_copy['terrain_intensity'], bins=bins, labels=labels)
        success_by_terrain = df_copy.groupby('terrain_bin')['success'].mean() * 100
        bars = ax3.bar(success_by_terrain.index, success_by_terrain.values, color='steelblue', edgecolor='black')
        ax3.set_xlabel('Terrain Intensity Range (%)', fontsize=LABEL_FONTSIZE)
        ax3.set_ylabel('Success Rate (%)', fontsize=LABEL_FONTSIZE)
        ax3.set_title('Success Rate by Terrain Intensity', fontsize=TITLE_FONTSIZE - 2)
        ax3.set_ylim(0, 105)
        ax3.tick_params(labelsize=TICK_FONTSIZE)
        for bar, val in zip(bars, success_by_terrain.values):
            if not np.isnan(val):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE)
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No terrain/success data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Success Rate by Terrain Intensity', fontsize=TITLE_FONTSIZE - 2)

    # Plot 4: Success Rate vs Friction (binned)
    ax4 = axes[1, 1]
    if has_friction and 'success' in df.columns:
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
        df_copy = df.copy()
        df_copy['friction_bin'] = pd.cut(df_copy['friction_intensity'], bins=bins, labels=labels)
        success_by_friction = df_copy.groupby('friction_bin')['success'].mean() * 100
        bars = ax4.bar(success_by_friction.index, success_by_friction.values, color='forestgreen', edgecolor='black')
        ax4.set_xlabel('Friction Intensity Range (%)', fontsize=LABEL_FONTSIZE)
        ax4.set_ylabel('Success Rate (%)', fontsize=LABEL_FONTSIZE)
        ax4.set_title('Success Rate by Friction Intensity', fontsize=TITLE_FONTSIZE - 2)
        ax4.set_ylim(0, 105)
        ax4.tick_params(labelsize=TICK_FONTSIZE)
        for bar, val in zip(bars, success_by_friction.values):
            if not np.isnan(val):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No friction/success data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Success Rate by Friction Intensity', fontsize=TITLE_FONTSIZE - 2)

    plt.tight_layout()
    return fig


def plot_distributions(df: pd.DataFrame):
    """Plot histograms of key metrics."""
    cte_col = None
    for col in ['mean_cross_track_error', 'cross_track_error', 'mean_cte']:
        if col in df.columns:
            cte_col = col
            break

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Distribution of Training Metrics', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # CTE distribution
    ax1 = axes[0, 0]
    if cte_col:
        ax1.hist(df[cte_col], bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax1.axvline(df[cte_col].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {df[cte_col].mean():.3f}')
        ax1.axvline(df[cte_col].median(), color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {df[cte_col].median():.3f}')
        ax1.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax1.set_xlabel('Cross-Track Error (m)', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Frequency', fontsize=LABEL_FONTSIZE)
    ax1.set_title('CTE Distribution', fontsize=TITLE_FONTSIZE - 2)
    ax1.tick_params(labelsize=TICK_FONTSIZE)

    # Reward per step distribution
    ax2 = axes[0, 1]
    reward_per_step = _get_reward_per_step(df)
    if reward_per_step is not None:
        ax2.hist(reward_per_step, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(reward_per_step.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {reward_per_step.mean():.4f}')
        ax2.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax2.set_xlabel('Reward Per Step', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('Frequency', fontsize=LABEL_FONTSIZE)
    ax2.set_title('Reward Per Step Distribution', fontsize=TITLE_FONTSIZE - 2)
    ax2.tick_params(labelsize=TICK_FONTSIZE)

    # Steps distribution
    ax3 = axes[0, 2]
    if 'steps' in df.columns:
        ax3.hist(df['steps'], bins=50, color='orange', alpha=0.7, edgecolor='black')
        ax3.axvline(df['steps'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {df["steps"].mean():.0f}')
        ax3.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax3.set_xlabel('Episode Length (steps)', fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel('Frequency', fontsize=LABEL_FONTSIZE)
    ax3.set_title('Episode Length Distribution', fontsize=TITLE_FONTSIZE - 2)
    ax3.tick_params(labelsize=TICK_FONTSIZE)

    # Terrain distribution (if available)
    ax4 = axes[1, 0]
    if 'terrain_intensity' in df.columns:
        ax4.hist(df['terrain_intensity'], bins=30, color='brown', alpha=0.7, edgecolor='black')
        ax4.axvline(df['terrain_intensity'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {df["terrain_intensity"].mean():.1f}%')
        ax4.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax4.set_xlabel('Terrain Intensity (%)', fontsize=LABEL_FONTSIZE)
    ax4.set_ylabel('Frequency', fontsize=LABEL_FONTSIZE)
    ax4.set_title('Terrain Distribution', fontsize=TITLE_FONTSIZE - 2)
    ax4.tick_params(labelsize=TICK_FONTSIZE)

    # Friction distribution (if available)
    ax5 = axes[1, 1]
    if 'friction_intensity' in df.columns:
        ax5.hist(df['friction_intensity'], bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax5.axvline(df['friction_intensity'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {df["friction_intensity"].mean():.1f}%')
        ax5.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax5.set_xlabel('Friction Intensity (%)', fontsize=LABEL_FONTSIZE)
    ax5.set_ylabel('Frequency', fontsize=LABEL_FONTSIZE)
    ax5.set_title('Friction Distribution', fontsize=TITLE_FONTSIZE - 2)
    ax5.tick_params(labelsize=TICK_FONTSIZE)

    # Success pie chart
    ax6 = axes[1, 2]
    if 'success' in df.columns:
        success_count = df['success'].sum()
        fail_count = len(df) - success_count
        ax6.pie([success_count, fail_count], labels=['Success', 'Fail'],
                autopct='%1.1f%%', colors=['#4CAF50', '#f44336'],
                explode=(0.05, 0), shadow=True, textprops={'fontsize': TICK_FONTSIZE})
        ax6.set_title(f'Success Rate ({success_count}/{len(df)})', fontsize=TITLE_FONTSIZE - 2)

    plt.tight_layout()
    return fig


def plot_binned_analysis(df: pd.DataFrame):
    """Plot detailed binned analysis of CTE by parameter ranges."""
    has_terrain = 'terrain_intensity' in df.columns
    has_friction = 'friction_intensity' in df.columns

    cte_col = None
    for col in ['mean_cross_track_error', 'cross_track_error', 'mean_cte']:
        if col in df.columns:
            cte_col = col
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Mean CTE by Parameter Ranges (with std error bars)', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Binned CTE by Terrain
    ax1 = axes[0]
    if has_terrain and cte_col:
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        df_copy = df.copy()
        df_copy['terrain_bin'] = pd.cut(df_copy['terrain_intensity'], bins=bins, labels=labels)

        grouped = df_copy.groupby('terrain_bin')[cte_col].agg(['mean', 'std', 'count'])
        grouped = grouped.dropna()

        x = range(len(grouped))
        bars = ax1.bar(x, grouped['mean'], yerr=grouped['std'],
                       capsize=5, color='steelblue', edgecolor='black', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(grouped.index, fontsize=TICK_FONTSIZE)
        ax1.set_xlabel('Terrain Intensity Range', fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Mean Cross-Track Error (m)', fontsize=LABEL_FONTSIZE)
        ax1.set_title('CTE by Terrain Intensity', fontsize=TITLE_FONTSIZE - 2)
        ax1.tick_params(labelsize=TICK_FONTSIZE)

        for i, (idx, row) in enumerate(grouped.iterrows()):
            ax1.text(i, row['mean'] + row['std'] + 0.01,
                     f'n={int(row["count"])}', ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE)
        ax1.grid(True, alpha=0.3, axis='y')
    else:
        ax1.text(0.5, 0.5, 'No terrain data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('CTE by Terrain Intensity', fontsize=TITLE_FONTSIZE - 2)

    # Binned CTE by Friction
    ax2 = axes[1]
    if has_friction and cte_col:
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        df_copy = df.copy()
        df_copy['friction_bin'] = pd.cut(df_copy['friction_intensity'], bins=bins, labels=labels)

        grouped = df_copy.groupby('friction_bin')[cte_col].agg(['mean', 'std', 'count'])
        grouped = grouped.dropna()

        x = range(len(grouped))
        bars = ax2.bar(x, grouped['mean'], yerr=grouped['std'],
                       capsize=5, color='forestgreen', edgecolor='black', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(grouped.index, fontsize=TICK_FONTSIZE)
        ax2.set_xlabel('Friction Intensity Range', fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel('Mean Cross-Track Error (m)', fontsize=LABEL_FONTSIZE)
        ax2.set_title('CTE by Friction Intensity', fontsize=TITLE_FONTSIZE - 2)
        ax2.tick_params(labelsize=TICK_FONTSIZE)

        for i, (idx, row) in enumerate(grouped.iterrows()):
            ax2.text(i, row['mean'] + row['std'] + 0.01,
                     f'n={int(row["count"])}', ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE)
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No friction data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('CTE by Friction Intensity', fontsize=TITLE_FONTSIZE - 2)

    plt.tight_layout()
    return fig


def plot_convergence_analysis(df: pd.DataFrame):
    """Analyze training convergence."""
    cte_col = None
    for col in ['mean_cross_track_error', 'cross_track_error', 'mean_cte']:
        if col in df.columns:
            cte_col = col
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Convergence Analysis', fontsize=TITLE_FONTSIZE, fontweight='bold')

    n_segments = 10
    segment_size = len(df) // n_segments

    # CTE by training segment
    ax1 = axes[0]
    if cte_col and segment_size > 0:
        segment_means = []
        segment_stds = []

        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else len(df)
            segment = df[cte_col].iloc[start:end]
            segment_means.append(segment.mean())
            segment_stds.append(segment.std())

        x = range(n_segments)
        ax1.errorbar(x, segment_means, yerr=segment_stds, marker='o', capsize=5,
                     linewidth=2, markersize=8, color='blue')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{i * 10}-{(i + 1) * 10}%' for i in range(n_segments)],
                            rotation=45, fontsize=TICK_FONTSIZE - 2)
        ax1.set_xlabel('Training Progress (%)', fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Mean CTE (m)', fontsize=LABEL_FONTSIZE)
        ax1.set_title('CTE Convergence Over Training', fontsize=TITLE_FONTSIZE - 2)
        ax1.tick_params(labelsize=TICK_FONTSIZE)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('CTE Convergence Over Training', fontsize=TITLE_FONTSIZE - 2)

    # Success rate by training segment
    ax2 = axes[1]
    if 'success' in df.columns and segment_size > 0:
        segment_success = []

        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else len(df)
            segment = df['success'].iloc[start:end]
            segment_success.append(segment.mean() * 100)

        x = range(n_segments)
        ax2.plot(x, segment_success, marker='s', linewidth=2, markersize=8, color='green')
        ax2.fill_between(x, 0, segment_success, alpha=0.3, color='green')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{i * 10}-{(i + 1) * 10}%' for i in range(n_segments)],
                            rotation=45, fontsize=TICK_FONTSIZE - 2)
        ax2.set_xlabel('Training Progress (%)', fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel('Success Rate (%)', fontsize=LABEL_FONTSIZE)
        ax2.set_title('Success Rate Convergence Over Training', fontsize=TITLE_FONTSIZE - 2)
        ax2.set_ylim(0, 105)
        ax2.tick_params(labelsize=TICK_FONTSIZE)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Success Rate Convergence Over Training', fontsize=TITLE_FONTSIZE - 2)

    plt.tight_layout()
    return fig


def plot_slope_analysis(df: pd.DataFrame):
    """Plot CTE and success rate vs terrain slope."""
    has_max_slope = 'terrain_max_slope_deg' in df.columns
    has_avg_slope = 'terrain_avg_slope_deg' in df.columns
    has_local_slope = 'mean_local_slope_deg' in df.columns

    cte_col = None
    for col in ['mean_cross_track_error', 'cross_track_error', 'mean_cte']:
        if col in df.columns:
            cte_col = col
            break

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance vs Terrain Slope', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Plot 1: CTE vs Max Slope
    ax1 = axes[0, 0]
    if has_max_slope and cte_col:
        ax1.scatter(df['terrain_max_slope_deg'], df[cte_col], alpha=0.3, s=10, color='blue')
        z = np.polyfit(df['terrain_max_slope_deg'], df[cte_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['terrain_max_slope_deg'].min(), df['terrain_max_slope_deg'].max(), 100)
        ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.4f})')
        ax1.set_xlabel('Max Terrain Slope (Â°)', fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Cross-Track Error (m)', fontsize=LABEL_FONTSIZE)
        ax1.set_title('CTE vs Maximum Terrain Slope', fontsize=TITLE_FONTSIZE - 2)
        ax1.legend(fontsize=LEGEND_FONTSIZE - 2)
        ax1.tick_params(labelsize=TICK_FONTSIZE)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No max slope data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('CTE vs Maximum Terrain Slope', fontsize=TITLE_FONTSIZE - 2)

    # Plot 2: CTE vs Avg Slope
    ax2 = axes[0, 1]
    if has_avg_slope and cte_col:
        ax2.scatter(df['terrain_avg_slope_deg'], df[cte_col], alpha=0.3, s=10, color='green')
        z = np.polyfit(df['terrain_avg_slope_deg'], df[cte_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['terrain_avg_slope_deg'].min(), df['terrain_avg_slope_deg'].max(), 100)
        ax2.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.4f})')
        ax2.set_xlabel('Average Terrain Slope (Â°)', fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel('Cross-Track Error (m)', fontsize=LABEL_FONTSIZE)
        ax2.set_title('CTE vs Average Terrain Slope', fontsize=TITLE_FONTSIZE - 2)
        ax2.legend(fontsize=LEGEND_FONTSIZE - 2)
        ax2.tick_params(labelsize=TICK_FONTSIZE)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No avg slope data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('CTE vs Average Terrain Slope', fontsize=TITLE_FONTSIZE - 2)

    # Plot 3: Success Rate vs Max Slope (binned)
    ax3 = axes[1, 0]
    if has_max_slope and 'success' in df.columns:
        slope_min = df['terrain_max_slope_deg'].min()
        slope_max = df['terrain_max_slope_deg'].max()
        bins = np.linspace(slope_min, slope_max, 6)
        labels = [f'{bins[i]:.0f}-{bins[i + 1]:.0f}Â°' for i in range(len(bins) - 1)]

        df_copy = df.copy()
        df_copy['slope_bin'] = pd.cut(df_copy['terrain_max_slope_deg'], bins=bins, labels=labels)
        success_by_slope = df_copy.groupby('slope_bin')['success'].mean() * 100

        bars = ax3.bar(range(len(success_by_slope)), success_by_slope.values, color='steelblue', edgecolor='black')
        ax3.set_xticks(range(len(success_by_slope)))
        ax3.set_xticklabels(success_by_slope.index, rotation=45, fontsize=TICK_FONTSIZE - 2)
        ax3.set_xlabel('Max Terrain Slope Range', fontsize=LABEL_FONTSIZE)
        ax3.set_ylabel('Success Rate (%)', fontsize=LABEL_FONTSIZE)
        ax3.set_title('Success Rate by Max Terrain Slope', fontsize=TITLE_FONTSIZE - 2)
        ax3.set_ylim(0, 105)
        ax3.tick_params(labelsize=TICK_FONTSIZE)
        for bar, val in zip(bars, success_by_slope.values):
            if not np.isnan(val):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE)
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No slope/success data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Success Rate by Max Terrain Slope', fontsize=TITLE_FONTSIZE - 2)

    # Plot 4: Slope distributions
    ax4 = axes[1, 1]
    if has_max_slope or has_avg_slope:
        if has_max_slope:
            ax4.hist(df['terrain_max_slope_deg'], bins=30, alpha=0.5,
                     label=f'Max Slope (mean={df["terrain_max_slope_deg"].mean():.1f}Â°)', color='blue')
        if has_avg_slope:
            ax4.hist(df['terrain_avg_slope_deg'], bins=30, alpha=0.5,
                     label=f'Avg Slope (mean={df["terrain_avg_slope_deg"].mean():.1f}Â°)', color='green')
        if has_local_slope:
            ax4.hist(df['mean_local_slope_deg'], bins=30, alpha=0.5,
                     label=f'Local Slope (mean={df["mean_local_slope_deg"].mean():.1f}Â°)', color='orange')
        ax4.set_xlabel('Slope (Â°)', fontsize=LABEL_FONTSIZE)
        ax4.set_ylabel('Frequency', fontsize=LABEL_FONTSIZE)
        ax4.set_title('Terrain Slope Distributions', fontsize=TITLE_FONTSIZE - 2)
        ax4.legend(fontsize=LEGEND_FONTSIZE - 2)
        ax4.tick_params(labelsize=TICK_FONTSIZE)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No slope data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Terrain Slope Distributions', fontsize=TITLE_FONTSIZE - 2)

    plt.tight_layout()
    return fig


# =============================================================================
# NEW: BOX & WHISKER COMPARISON PLOTS
# =============================================================================

def plot_residual_analysis(df: pd.DataFrame, window: int = 50):
    """Plot residual-magnitude diagnostics (v33.9+).

    Shows whether PPO is actually using its residual authority over training:
      - top-left: mean |residual_v| per episode (raw + rolling) — 0 means
        "PPO does nothing on velocity", 1 means "saturating the bound"
      - top-right: same for omega
      - bottom-left: |residual| vs terrain_intensity scatter — does PPO use
        more authority on harder terrain (good) or uniformly (less ideal)?
      - bottom-right: |residual| distribution split by success vs failure —
        do successful episodes show different residual patterns?
    """
    has_v = 'mean_residual_v_norm' in df.columns
    has_w = 'mean_residual_w_norm' in df.columns

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Residual Magnitude Analysis (v33.9+)',
                 fontsize=TITLE_FONTSIZE, fontweight='bold')

    if not (has_v or has_w):
        for ax in axes.flatten():
            ax.text(0.5, 0.5,
                    "No residual columns in this CSV.\n"
                    "Re-run training with v33.9+ logging.",
                    ha='center', va='center', fontsize=14,
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        return fig

    episodes = df['episode'] if 'episode' in df.columns else pd.Series(range(len(df)))

    # Plot 1: |residual_v| over episodes
    ax1 = axes[0, 0]
    if has_v:
        roll_v = df['mean_residual_v_norm'].rolling(window=window, min_periods=1).mean()
        ax1.plot(episodes, df['mean_residual_v_norm'], alpha=0.3, color='steelblue', label='Raw')
        ax1.plot(episodes, roll_v, color='steelblue', linewidth=2,
                 label=f'Rolling Mean (w={window})')
        _add_trend_line(ax1, episodes, df['mean_residual_v_norm'], color='red')
    ax1.set_xlabel('Episode', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('|residual_v| / max_authority', fontsize=LABEL_FONTSIZE)
    ax1.set_title('Velocity Residual Magnitude', fontsize=TITLE_FONTSIZE - 2)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax1.tick_params(labelsize=TICK_FONTSIZE)
    ax1.grid(True, alpha=0.3)

    # Plot 2: |residual_w| over episodes
    ax2 = axes[0, 1]
    if has_w:
        roll_w = df['mean_residual_w_norm'].rolling(window=window, min_periods=1).mean()
        ax2.plot(episodes, df['mean_residual_w_norm'], alpha=0.3, color='darkorange', label='Raw')
        ax2.plot(episodes, roll_w, color='darkorange', linewidth=2,
                 label=f'Rolling Mean (w={window})')
        _add_trend_line(ax2, episodes, df['mean_residual_w_norm'], color='red')
    ax2.set_xlabel('Episode', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('|residual_w| / max_authority', fontsize=LABEL_FONTSIZE)
    ax2.set_title('Omega Residual Magnitude', fontsize=TITLE_FONTSIZE - 2)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax2.tick_params(labelsize=TICK_FONTSIZE)
    ax2.grid(True, alpha=0.3)

    # Plot 3: combined |residual| magnitude vs terrain difficulty
    ax3 = axes[1, 0]
    if 'terrain_intensity' in df.columns and (has_v or has_w):
        # Use the L2 norm of the two-component normalized residual
        v_part = df['mean_residual_v_norm'] if has_v else 0.0
        w_part = df['mean_residual_w_norm'] if has_w else 0.0
        combined = (v_part ** 2 + w_part ** 2) ** 0.5
        ax3.scatter(df['terrain_intensity'], combined, alpha=0.3, s=10, color='purple')
        try:
            z = np.polyfit(df['terrain_intensity'], combined, 1)
            x_line = np.linspace(df['terrain_intensity'].min(),
                                 df['terrain_intensity'].max(), 100)
            ax3.plot(x_line, np.poly1d(z)(x_line), color='red', linewidth=2,
                     label=f'Slope: {z[0]:+.4f}/% terrain')
            ax3.legend(fontsize=LEGEND_FONTSIZE - 2)
        except Exception:
            pass
    ax3.set_xlabel('Terrain Intensity (%)', fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel('||residual||₂ (normalized)', fontsize=LABEL_FONTSIZE)
    ax3.set_title('Residual Use vs Terrain Difficulty', fontsize=TITLE_FONTSIZE - 2)
    ax3.tick_params(labelsize=TICK_FONTSIZE)
    ax3.grid(True, alpha=0.3)

    # Plot 4: distribution by success/failure
    ax4 = axes[1, 1]
    if 'success' in df.columns and (has_v or has_w):
        v_part = df['mean_residual_v_norm'] if has_v else 0.0
        w_part = df['mean_residual_w_norm'] if has_w else 0.0
        combined = (v_part ** 2 + w_part ** 2) ** 0.5
        succ_vals = combined[df['success'] == 1]
        fail_vals = combined[df['success'] == 0]
        bins = np.linspace(0, max(0.01, float(combined.max())), 40)
        if len(succ_vals) > 0:
            ax4.hist(succ_vals, bins=bins, alpha=0.6, color='green',
                     label=f'Success (n={len(succ_vals)}, μ={succ_vals.mean():.3f})')
        if len(fail_vals) > 0:
            ax4.hist(fail_vals, bins=bins, alpha=0.6, color='red',
                     label=f'Failure (n={len(fail_vals)}, μ={fail_vals.mean():.3f})')
        ax4.legend(fontsize=LEGEND_FONTSIZE - 2)
    ax4.set_xlabel('||residual||₂ (normalized)', fontsize=LABEL_FONTSIZE)
    ax4.set_ylabel('Episode Count', fontsize=LABEL_FONTSIZE)
    ax4.set_title('Residual Distribution: Success vs Failure',
                  fontsize=TITLE_FONTSIZE - 2)
    ax4.tick_params(labelsize=TICK_FONTSIZE)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def bin_data(df: pd.DataFrame, column: str, bins: list, bin_labels: list) -> pd.DataFrame:
    """
    Add a binned column to the DataFrame.
    """
    df = df.copy()
    bin_col = f'{column}_bin'
    df[bin_col] = pd.cut(df[column], bins=bins, labels=bin_labels,
                         include_lowest=True, right=True)
    return df


def plot_comparison_boxplot(lqr_df: pd.DataFrame, hybrid_df: pd.DataFrame,
                            x_column: str, x_bins: list, x_bin_labels: list,
                            y_column: str = 'mean_cross_track_error',
                            y_label: str = 'Mean Cross-Track Error (m)',
                            x_label: str = 'Terrain Intensity',
                            title: str = 'CTE vs Terrain Intensity',
                            y_min: float = DEFAULT_CTE_Y_MIN,
                            y_max: float = DEFAULT_CTE_Y_MAX,
                            figsize: tuple = (10, 7)):
    """
    Create a side-by-side box plot comparing LQR and Hybrid performance.
    """
    # Determine CTE column if default doesn't exist
    if y_column not in lqr_df.columns:
        for col in ['cross_track_error', 'mean_cte']:
            if col in lqr_df.columns:
                y_column = col
                break

    bin_col = f'{x_column}_bin'
    lqr_binned = bin_data(lqr_df, x_column, x_bins, x_bin_labels)
    hybrid_binned = bin_data(hybrid_df, x_column, x_bins, x_bin_labels)

    fig, ax = plt.subplots(figsize=figsize)

    width = 0.35
    gap = 0.05
    data_lqr = []
    data_hybrid = []
    positions_lqr = []
    positions_hybrid = []

    for i, label in enumerate(x_bin_labels):
        center = i

        lqr_subset = lqr_binned[lqr_binned[bin_col] == label][y_column].dropna()
        data_lqr.append(lqr_subset.values if len(lqr_subset) > 0 else [np.nan])
        positions_lqr.append(center - width / 2 - gap / 2)

        hybrid_subset = hybrid_binned[hybrid_binned[bin_col] == label][y_column].dropna()
        data_hybrid.append(hybrid_subset.values if len(hybrid_subset) > 0 else [np.nan])
        positions_hybrid.append(center + width / 2 + gap / 2)

    # Filter out empty bins
    data_lqr_clean = [d for d in data_lqr if not (len(d) == 1 and np.isnan(d[0]))]
    positions_lqr_clean = [p for d, p in zip(data_lqr, positions_lqr) if not (len(d) == 1 and np.isnan(d[0]))]

    data_hybrid_clean = [d for d in data_hybrid if not (len(d) == 1 and np.isnan(d[0]))]
    positions_hybrid_clean = [p for d, p in zip(data_hybrid, positions_hybrid) if not (len(d) == 1 and np.isnan(d[0]))]

    if data_lqr_clean:
        ax.boxplot(data_lqr_clean, positions=positions_lqr_clean, widths=width,
                   patch_artist=True,
                   showfliers=SHOW_OUTLIERS,
                   boxprops=dict(facecolor=LQR_COLOR, alpha=0.7),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color=LQR_COLOR, linewidth=1.5),
                   capprops=dict(color=LQR_COLOR, linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor=LQR_COLOR, markersize=4, alpha=0.5))

    if data_hybrid_clean:
        ax.boxplot(data_hybrid_clean, positions=positions_hybrid_clean, widths=width,
                   patch_artist=True,
                   showfliers=SHOW_OUTLIERS,
                   boxprops=dict(facecolor=HYBRID_COLOR, alpha=0.7),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color=HYBRID_COLOR, linewidth=1.5),
                   capprops=dict(color=HYBRID_COLOR, linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor=HYBRID_COLOR, markersize=4, alpha=0.5))

    ax.set_xlim(-0.5, len(x_bin_labels) - 0.5)
    ax.set_xticks(range(len(x_bin_labels)))
    ax.set_xticklabels(x_bin_labels, fontsize=TICK_FONTSIZE)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)

    legend_elements = [
        Patch(facecolor=LQR_COLOR, alpha=0.7, edgecolor='black', label='Pure LQR'),
        Patch(facecolor=HYBRID_COLOR, alpha=0.7, edgecolor='black', label='Hybrid LQR+PPO')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=LEGEND_FONTSIZE)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for i, label in enumerate(x_bin_labels):
        lqr_n = len(lqr_binned[lqr_binned[bin_col] == label].dropna(subset=[y_column]))
        hybrid_n = len(hybrid_binned[hybrid_binned[bin_col] == label].dropna(subset=[y_column]))

        ax.text(i - width / 2 - gap / 2, y_min + 0.05 * (y_max - y_min),
                f'n={lqr_n}', ha='center', va='bottom',
                fontsize=ANNOTATION_FONTSIZE - 2, color=LQR_COLOR, fontweight='bold')
        ax.text(i + width / 2 + gap / 2, y_min + 0.05 * (y_max - y_min),
                f'n={hybrid_n}', ha='center', va='bottom',
                fontsize=ANNOTATION_FONTSIZE - 2, color=HYBRID_COLOR, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_combined_comparison(lqr_df: pd.DataFrame, hybrid_df: pd.DataFrame,
                             terrain_min: float = DEFAULT_TERRAIN_MIN,
                             terrain_max: float = DEFAULT_TERRAIN_MAX,
                             friction_min: float = DEFAULT_FRICTION_MIN,
                             friction_max: float = DEFAULT_FRICTION_MAX,
                             y_min: float = DEFAULT_CTE_Y_MIN,
                             y_max: float = DEFAULT_CTE_Y_MAX,
                             figsize: tuple = (14, 6)):
    """
    Create a single figure with both terrain and friction comparisons side by side.
    This is the main figure for the Goldwater essay.

    Args:
        lqr_df: LQR screening data
        hybrid_df: Hybrid screening data
        terrain_min: Minimum terrain intensity to include (default 0)
        terrain_max: Maximum terrain intensity to include (default 70)
        friction_min: Minimum friction intensity to include (default 30)
        friction_max: Maximum friction intensity to include (default 100)
        y_min: Y-axis minimum
        y_max: Y-axis maximum
        figsize: Figure size tuple
    """
    cte_col = 'mean_cross_track_error'
    if cte_col not in lqr_df.columns:
        for col in ['cross_track_error', 'mean_cte']:
            if col in lqr_df.columns:
                cte_col = col
                break

    # Filter data for terrain plot (apply terrain bounds)
    if 'terrain_intensity' in lqr_df.columns:
        lqr_terrain = lqr_df[(lqr_df['terrain_intensity'] >= terrain_min) &
                             (lqr_df['terrain_intensity'] <= terrain_max)].copy()
        hybrid_terrain = hybrid_df[(hybrid_df['terrain_intensity'] >= terrain_min) &
                                   (hybrid_df['terrain_intensity'] <= terrain_max)].copy()
    else:
        lqr_terrain = lqr_df.copy()
        hybrid_terrain = hybrid_df.copy()

    # Filter data for friction plot (apply friction bounds)
    if 'friction_intensity' in lqr_df.columns:
        lqr_friction = lqr_df[(lqr_df['friction_intensity'] >= friction_min) &
                              (lqr_df['friction_intensity'] <= friction_max)].copy()
        hybrid_friction = hybrid_df[(hybrid_df['friction_intensity'] >= friction_min) &
                                    (hybrid_df['friction_intensity'] <= friction_max)].copy()
    else:
        lqr_friction = lqr_df.copy()
        hybrid_friction = hybrid_df.copy()

    # Create terrain bins based on filtered range
    terrain_range = terrain_max - terrain_min
    if terrain_range > 60:
        # 4 bins for large ranges
        bin_size = terrain_range / 4
        terrain_bins = [terrain_min, terrain_min + bin_size, terrain_min + 2 * bin_size,
                        terrain_min + 3 * bin_size, terrain_max]
    else:
        # Fewer bins for smaller ranges
        bin_size = terrain_range / 3
        terrain_bins = [terrain_min, terrain_min + bin_size, terrain_min + 2 * bin_size, terrain_max]

    terrain_labels = []
    for i in range(len(terrain_bins) - 1):
        terrain_labels.append(f'{int(terrain_bins[i])}-{int(terrain_bins[i + 1])}%')

    # Create friction bins based on filtered range
    friction_range = friction_max - friction_min
    if friction_range > 60:
        bin_size = friction_range / 4
        friction_bins = [friction_min, friction_min + bin_size, friction_min + 2 * bin_size,
                         friction_min + 3 * bin_size, friction_max]
    else:
        bin_size = friction_range / 3
        friction_bins = [friction_min, friction_min + bin_size, friction_min + 2 * bin_size, friction_max]

    friction_labels = []
    for i in range(len(friction_bins) - 1):
        friction_labels.append(f'{int(friction_bins[i])}-{int(friction_bins[i + 1])}%')

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: CTE vs Terrain Intensity
    ax = axes[0]
    width = 0.35
    gap = 0.05

    if 'terrain_intensity' in lqr_terrain.columns:
        bin_col = 'terrain_intensity_bin'
        lqr_binned = bin_data(lqr_terrain, 'terrain_intensity', terrain_bins, terrain_labels)
        hybrid_binned = bin_data(hybrid_terrain, 'terrain_intensity', terrain_bins, terrain_labels)

        data_lqr = []
        data_hybrid = []
        positions_lqr = []
        positions_hybrid = []

        for i, label in enumerate(terrain_labels):
            center = i
            lqr_subset = lqr_binned[lqr_binned[bin_col] == label][cte_col].dropna()
            data_lqr.append(lqr_subset.values if len(lqr_subset) > 0 else [0])
            positions_lqr.append(center - width / 2 - gap / 2)

            hybrid_subset = hybrid_binned[hybrid_binned[bin_col] == label][cte_col].dropna()
            data_hybrid.append(hybrid_subset.values if len(hybrid_subset) > 0 else [0])
            positions_hybrid.append(center + width / 2 + gap / 2)

        ax.boxplot(data_lqr, positions=positions_lqr, widths=width,
                   patch_artist=True,
                   showfliers=SHOW_OUTLIERS,
                   boxprops=dict(facecolor=LQR_COLOR, alpha=0.7),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color=LQR_COLOR, linewidth=1.5),
                   capprops=dict(color=LQR_COLOR, linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor=LQR_COLOR, markersize=4, alpha=0.5))

        ax.boxplot(data_hybrid, positions=positions_hybrid, widths=width,
                   patch_artist=True,
                   showfliers=SHOW_OUTLIERS,
                   boxprops=dict(facecolor=HYBRID_COLOR, alpha=0.7),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color=HYBRID_COLOR, linewidth=1.5),
                   capprops=dict(color=HYBRID_COLOR, linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor=HYBRID_COLOR, markersize=4, alpha=0.5))

        ax.set_xlim(-0.5, len(terrain_labels) - 0.5)
        ax.set_xticks(range(len(terrain_labels)))
        ax.set_xticklabels(terrain_labels, fontsize=TICK_FONTSIZE)

        for i, label in enumerate(terrain_labels):
            lqr_n = len(lqr_binned[lqr_binned[bin_col] == label])
            hybrid_n = len(hybrid_binned[hybrid_binned[bin_col] == label])
            ax.text(i - width / 2 - gap / 2, y_min + 0.02 * (y_max - y_min),
                    f'n={lqr_n}', ha='center', va='bottom',
                    fontsize=ANNOTATION_FONTSIZE - 2, color=LQR_COLOR, fontweight='bold')
            ax.text(i + width / 2 + gap / 2, y_min + 0.02 * (y_max - y_min),
                    f'n={hybrid_n}', ha='center', va='bottom',
                    fontsize=ANNOTATION_FONTSIZE - 2, color=HYBRID_COLOR, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No terrain data available', ha='center', va='center', transform=ax.transAxes)

    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.set_xlabel('Terrain Intensity', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Mean Cross-Track Error (m)', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_title(f'(a) CTE vs Terrain Intensity ({int(terrain_min)}-{int(terrain_max)}%)',
                 fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Right plot: CTE vs Friction Intensity
    ax = axes[1]

    if 'friction_intensity' in lqr_friction.columns:
        bin_col = 'friction_intensity_bin'
        lqr_binned = bin_data(lqr_friction, 'friction_intensity', friction_bins, friction_labels)
        hybrid_binned = bin_data(hybrid_friction, 'friction_intensity', friction_bins, friction_labels)

        data_lqr = []
        data_hybrid = []
        positions_lqr = []
        positions_hybrid = []

        for i, label in enumerate(friction_labels):
            center = i
            lqr_subset = lqr_binned[lqr_binned[bin_col] == label][cte_col].dropna()
            data_lqr.append(lqr_subset.values if len(lqr_subset) > 0 else [0])
            positions_lqr.append(center - width / 2 - gap / 2)

            hybrid_subset = hybrid_binned[hybrid_binned[bin_col] == label][cte_col].dropna()
            data_hybrid.append(hybrid_subset.values if len(hybrid_subset) > 0 else [0])
            positions_hybrid.append(center + width / 2 + gap / 2)

        ax.boxplot(data_lqr, positions=positions_lqr, widths=width,
                   patch_artist=True,
                   showfliers=SHOW_OUTLIERS,
                   boxprops=dict(facecolor=LQR_COLOR, alpha=0.7),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color=LQR_COLOR, linewidth=1.5),
                   capprops=dict(color=LQR_COLOR, linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor=LQR_COLOR, markersize=4, alpha=0.5))

        ax.boxplot(data_hybrid, positions=positions_hybrid, widths=width,
                   patch_artist=True,
                   showfliers=SHOW_OUTLIERS,
                   boxprops=dict(facecolor=HYBRID_COLOR, alpha=0.7),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color=HYBRID_COLOR, linewidth=1.5),
                   capprops=dict(color=HYBRID_COLOR, linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor=HYBRID_COLOR, markersize=4, alpha=0.5))

        ax.set_xlim(-0.5, len(friction_labels) - 0.5)
        ax.set_xticks(range(len(friction_labels)))
        ax.set_xticklabels(friction_labels, fontsize=TICK_FONTSIZE)

        for i, label in enumerate(friction_labels):
            lqr_n = len(lqr_binned[lqr_binned[bin_col] == label])
            hybrid_n = len(hybrid_binned[hybrid_binned[bin_col] == label])
            ax.text(i - width / 2 - gap / 2, y_min + 0.02 * (y_max - y_min),
                    f'n={lqr_n}', ha='center', va='bottom',
                    fontsize=ANNOTATION_FONTSIZE - 2, color=LQR_COLOR, fontweight='bold')
            ax.text(i + width / 2 + gap / 2, y_min + 0.02 * (y_max - y_min),
                    f'n={hybrid_n}', ha='center', va='bottom',
                    fontsize=ANNOTATION_FONTSIZE - 2, color=HYBRID_COLOR, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No friction data available', ha='center', va='center', transform=ax.transAxes)

    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.set_xlabel('Friction Intensity', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Mean Cross-Track Error (m)', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_title(f'(b) CTE vs Friction Intensity ({int(friction_min)}-{int(friction_max)}%)',
                 fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    legend_elements = [
        Patch(facecolor=LQR_COLOR, alpha=0.7, edgecolor='black', label='Pure LQR'),
        Patch(facecolor=HYBRID_COLOR, alpha=0.7, edgecolor='black', label='Hybrid LQR+PPO')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    return fig


# =============================================================================
# GUI CLASS
# =============================================================================

class TrainingEvaluationGUI:
    """GUI application for training evaluation analysis."""

    def __init__(self, root):
        self.root = root
        self.root.title("Training Session Evaluation")
        self.root.geometry("1200x900")

        self.df = None
        self.figures = []
        self.csv_path = None
        self.lqr_df = None
        self.hybrid_df = None

        self.create_widgets()

    def create_widgets(self):
        """Create GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)

        title_label = ttk.Label(main_frame, text="Training Session Evaluation",
                                font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, pady=10)

        load_frame = ttk.LabelFrame(main_frame, text="Load Data", padding="5")
        load_frame.grid(row=1, column=0, pady=5, sticky=tk.W + tk.E)

        ttk.Button(load_frame, text="Load CSV File",
                   command=self.load_csv_file).grid(row=0, column=0, padx=5, pady=3)
        ttk.Button(load_frame, text="Load Experiment Directory",
                   command=self.load_experiment_dir).grid(row=0, column=1, padx=5, pady=3)
        ttk.Button(load_frame, text="Paste CSV Data",
                   command=self.paste_csv_data).grid(row=0, column=2, padx=5, pady=3)

        ttk.Separator(load_frame, orient='vertical').grid(row=0, column=3, padx=10, sticky='ns')

        ttk.Label(load_frame, text="Rolling Window:").grid(row=0, column=4, padx=5)
        self.window_var = tk.IntVar(value=50)
        ttk.Entry(load_frame, textvariable=self.window_var, width=6).grid(row=0, column=5, padx=5)

        ttk.Separator(load_frame, orient='vertical').grid(row=0, column=6, padx=10, sticky='ns')
        ttk.Button(load_frame, text="Clear", command=self.clear_all).grid(row=0, column=7, padx=5, pady=3)

        compare_frame = ttk.LabelFrame(main_frame, text="LQR vs Hybrid Comparison (Box Plots)", padding="5")
        compare_frame.grid(row=2, column=0, pady=5, sticky=tk.W + tk.E)

        ttk.Button(compare_frame, text="Load LQR Data",
                   command=self.load_lqr_data).grid(row=0, column=0, padx=5, pady=3)
        ttk.Button(compare_frame, text="Load Hybrid Data",
                   command=self.load_hybrid_data).grid(row=0, column=1, padx=5, pady=3)

        ttk.Separator(compare_frame, orient='vertical').grid(row=0, column=2, padx=10, sticky='ns')

        # Terrain bounds
        ttk.Label(compare_frame, text="Terrain:").grid(row=0, column=3, padx=2)
        self.terrain_min_var = tk.IntVar(value=DEFAULT_TERRAIN_MIN)
        ttk.Entry(compare_frame, textvariable=self.terrain_min_var, width=4).grid(row=0, column=4, padx=2)
        ttk.Label(compare_frame, text="-").grid(row=0, column=5)
        self.terrain_max_var = tk.IntVar(value=DEFAULT_TERRAIN_MAX)
        ttk.Entry(compare_frame, textvariable=self.terrain_max_var, width=4).grid(row=0, column=6, padx=2)
        ttk.Label(compare_frame, text="%").grid(row=0, column=7)

        ttk.Separator(compare_frame, orient='vertical').grid(row=0, column=8, padx=5, sticky='ns')

        # Friction bounds
        ttk.Label(compare_frame, text="Friction:").grid(row=0, column=9, padx=2)
        self.friction_min_var = tk.IntVar(value=DEFAULT_FRICTION_MIN)
        ttk.Entry(compare_frame, textvariable=self.friction_min_var, width=4).grid(row=0, column=10, padx=2)
        ttk.Label(compare_frame, text="-").grid(row=0, column=11)
        self.friction_max_var = tk.IntVar(value=DEFAULT_FRICTION_MAX)
        ttk.Entry(compare_frame, textvariable=self.friction_max_var, width=4).grid(row=0, column=12, padx=2)
        ttk.Label(compare_frame, text="%").grid(row=0, column=13)

        ttk.Separator(compare_frame, orient='vertical').grid(row=0, column=14, padx=5, sticky='ns')

        # Y-axis control
        ttk.Label(compare_frame, text="Y-Max:").grid(row=0, column=15, padx=2)
        self.y_max_var = tk.DoubleVar(value=DEFAULT_CTE_Y_MAX)
        ttk.Entry(compare_frame, textvariable=self.y_max_var, width=4).grid(row=0, column=16, padx=2)

        ttk.Separator(compare_frame, orient='vertical').grid(row=0, column=17, padx=5, sticky='ns')

        ttk.Button(compare_frame, text="Box Plot Comparison",
                   command=self.show_comparison_boxplot).grid(row=0, column=18, padx=5, pady=3)
        ttk.Button(compare_frame, text="Comparison Report",
                   command=self.show_comparison_report).grid(row=0, column=19, padx=5, pady=3)
        ttk.Button(compare_frame, text="Export Comparison",
                   command=self.export_comparison).grid(row=0, column=20, padx=5, pady=3)

        analysis_frame = ttk.LabelFrame(main_frame, text="Single Dataset Analysis", padding="5")
        analysis_frame.grid(row=3, column=0, pady=5, sticky=tk.W + tk.E)

        ttk.Button(analysis_frame, text="Statistics Report",
                   command=self.show_statistics).grid(row=0, column=0, padx=5, pady=3)
        ttk.Button(analysis_frame, text="Learning Curves",
                   command=self.show_learning_curves).grid(row=0, column=1, padx=5, pady=3)
        ttk.Button(analysis_frame, text="Distributions",
                   command=self.show_distributions).grid(row=0, column=2, padx=5, pady=3)
        ttk.Button(analysis_frame, text="Terrain Analysis",
                   command=self.show_terrain_analysis).grid(row=0, column=3, padx=5, pady=3)
        ttk.Button(analysis_frame, text="Binned Analysis",
                   command=self.show_binned_analysis).grid(row=0, column=4, padx=5, pady=3)
        ttk.Button(analysis_frame, text="Slope Analysis",
                   command=self.show_slope_analysis).grid(row=0, column=5, padx=5, pady=3)
        ttk.Button(analysis_frame, text="Convergence",
                   command=self.show_convergence).grid(row=0, column=6, padx=5, pady=3)
        # v33.9: residual-magnitude diagnostics for hybrid PPO runs
        ttk.Button(analysis_frame, text="Residual Analysis",
                   command=self.show_residual_analysis).grid(row=0, column=7, padx=5, pady=3)
        ttk.Button(analysis_frame, text="All Plots",
                   command=self.show_all_plots).grid(row=0, column=8, padx=5, pady=3)

        ttk.Separator(analysis_frame, orient='vertical').grid(row=0, column=9, padx=10, sticky='ns')

        ttk.Button(analysis_frame, text="Export Report",
                   command=self.export_report).grid(row=0, column=10, padx=5, pady=3)
        ttk.Button(analysis_frame, text="Export Plots (PDF)",
                   command=self.export_plots_pdf).grid(row=0, column=11, padx=5, pady=3)
        ttk.Button(analysis_frame, text="Export Plots (PNG)",
                   command=self.export_plots_png).grid(row=0, column=12, padx=5, pady=3)

        self.data_info_label = ttk.Label(main_frame, text="No data loaded",
                                         foreground="gray", font=('Arial', 10))
        self.data_info_label.grid(row=4, column=0, pady=5)

        self.output_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD,
                                                     width=140, height=35,
                                                     font=('Courier', 9))
        self.output_text.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.status_label = ttk.Label(main_frame, text="Ready - Load training data to begin",
                                      relief=tk.SUNKEN)
        self.status_label.grid(row=6, column=0, sticky=(tk.W, tk.E))

    def set_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)

    def append_output(self, text):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def clear_all(self):
        self.df = None
        self.lqr_df = None
        self.hybrid_df = None
        self.figures = []
        self.csv_path = None
        self.clear_output()
        self.data_info_label.config(text="No data loaded", foreground="gray")
        self.set_status("Cleared all data")

    def update_data_info(self):
        info_parts = []
        if self.df is not None:
            info_parts.append(f"Single: {len(self.df)} ep")
        if self.lqr_df is not None:
            info_parts.append(f"LQR: {len(self.lqr_df)} ep")
        if self.hybrid_df is not None:
            info_parts.append(f"Hybrid: {len(self.hybrid_df)} ep")

        if info_parts:
            self.data_info_label.config(text=" | ".join(info_parts), foreground="darkgreen")
        else:
            self.data_info_label.config(text="No data loaded", foreground="gray")

    def load_csv_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Training CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.df = load_training_data(filepath)
                self.csv_path = filepath
                self.update_data_info()
                self.set_status(f"Loaded: {filepath}")
                self.append_output(f"Loaded {len(self.df)} episodes from:\n{filepath}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def load_experiment_dir(self):
        dirpath = filedialog.askdirectory(title="Select Experiment Directory")
        if dirpath:
            possible_paths = [
                os.path.join(dirpath, "csv", "episode_metrics.csv"),
                os.path.join(dirpath, "episode_metrics.csv"),
                os.path.join(dirpath, "screening_episodes.csv"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        self.df = load_training_data(path)
                        self.csv_path = path
                        self.update_data_info()
                        self.set_status(f"Loaded: {path}")
                        self.append_output(f"Loaded {len(self.df)} episodes from:\n{path}\n")
                        return
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load file:\n{e}")
                        return
            messagebox.showwarning("Not Found", "Could not find CSV in selected directory.")

    def paste_csv_data(self):
        paste_window = tk.Toplevel(self.root)
        paste_window.title("Paste CSV Data")
        paste_window.geometry("600x400")
        ttk.Label(paste_window, text="Paste CSV data below (including header):").pack(pady=5)
        text_area = scrolledtext.ScrolledText(paste_window, width=70, height=20)
        text_area.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        def load_pasted():
            csv_string = text_area.get(1.0, tk.END).strip()
            if csv_string:
                try:
                    self.df = load_from_string(csv_string)
                    self.csv_path = None
                    self.update_data_info()
                    self.set_status("Loaded data from pasted CSV")
                    self.append_output(f"Loaded {len(self.df)} episodes from pasted data\n")
                    paste_window.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to parse CSV:\n{e}")

        ttk.Button(paste_window, text="Load Data", command=load_pasted).pack(pady=10)

    def load_lqr_data(self):
        filepath = filedialog.askopenfilename(
            title="Select LQR Screening CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.lqr_df = load_screening_data(filepath)
                self.update_data_info()
                self.set_status(f"Loaded LQR data: {filepath}")
                self.append_output(f"Loaded LQR data: {len(self.lqr_df)} episodes\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load LQR file:\n{e}")

    def load_hybrid_data(self):
        filepath = filedialog.askopenfilename(
            title="Select Hybrid Screening CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.hybrid_df = load_screening_data(filepath)
                self.update_data_info()
                self.set_status(f"Loaded Hybrid data: {filepath}")
                self.append_output(f"Loaded Hybrid data: {len(self.hybrid_df)} episodes\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Hybrid file:\n{e}")

    def check_data_loaded(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please load training data first.")
            return False
        return True

    def check_comparison_data_loaded(self):
        if self.lqr_df is None or self.hybrid_df is None:
            messagebox.showwarning("No Comparison Data",
                                   "Please load both LQR and Hybrid data for comparison.")
            return False
        return True

    def show_statistics(self):
        if not self.check_data_loaded():
            return
        self.set_status("Generating statistics...")
        self.clear_output()
        report = generate_statistics_report(self.df)
        self.append_output(report)
        self.set_status("Statistics report generated")

    def show_comparison_report(self):
        if not self.check_comparison_data_loaded():
            return
        self.set_status("Generating comparison report...")
        self.clear_output()
        terrain_max = self.terrain_max_var.get()
        report = generate_comparison_report(self.lqr_df, self.hybrid_df, terrain_max)
        self.append_output(report)
        self.set_status("Comparison report generated")

    def show_comparison_boxplot(self):
        if not self.check_comparison_data_loaded():
            return
        self.set_status("Generating comparison box plots...")
        terrain_min = self.terrain_min_var.get()
        terrain_max = self.terrain_max_var.get()
        friction_min = self.friction_min_var.get()
        friction_max = self.friction_max_var.get()
        y_max = self.y_max_var.get()
        fig = plot_combined_comparison(self.lqr_df, self.hybrid_df,
                                       terrain_min=terrain_min, terrain_max=terrain_max,
                                       friction_min=friction_min, friction_max=friction_max,
                                       y_min=DEFAULT_CTE_Y_MIN, y_max=y_max)
        self.figures.append(('comparison_boxplot', fig))
        plt.show()
        self.set_status("Comparison box plots displayed")

    def export_comparison(self):
        if not self.check_comparison_data_loaded():
            return
        dirpath = filedialog.askdirectory(title="Select Output Directory")
        if dirpath:
            try:
                terrain_min = self.terrain_min_var.get()
                terrain_max = self.terrain_max_var.get()
                friction_min = self.friction_min_var.get()
                friction_max = self.friction_max_var.get()
                y_max = self.y_max_var.get()
                fig = plot_combined_comparison(self.lqr_df, self.hybrid_df,
                                               terrain_min=terrain_min, terrain_max=terrain_max,
                                               friction_min=friction_min, friction_max=friction_max,
                                               y_min=DEFAULT_CTE_Y_MIN, y_max=y_max)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(dirpath, f'cte_comparison_{timestamp}.png')
                fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                self.set_status(f"Saved: {filepath}")
                messagebox.showinfo("Success", f"Comparison plot saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save:\n{e}")

    def show_learning_curves(self):
        if not self.check_data_loaded():
            return
        self.set_status("Generating learning curves...")
        window = self.window_var.get()
        fig = plot_learning_curves(self.df, window)
        self.figures.append(('learning_curves', fig))
        plt.show()
        self.set_status("Learning curves displayed")

    def show_distributions(self):
        if not self.check_data_loaded():
            return
        self.set_status("Generating distributions...")
        fig = plot_distributions(self.df)
        self.figures.append(('distributions', fig))
        plt.show()
        self.set_status("Distributions displayed")

    def show_terrain_analysis(self):
        if not self.check_data_loaded():
            return
        self.set_status("Generating terrain analysis...")
        fig = plot_terrain_analysis(self.df)
        self.figures.append(('terrain_analysis', fig))
        plt.show()
        self.set_status("Terrain analysis displayed")

    def show_binned_analysis(self):
        if not self.check_data_loaded():
            return
        self.set_status("Generating binned analysis...")
        fig = plot_binned_analysis(self.df)
        self.figures.append(('binned_analysis', fig))
        plt.show()
        self.set_status("Binned analysis displayed")

    def show_slope_analysis(self):
        if not self.check_data_loaded():
            return
        self.set_status("Generating slope analysis...")
        fig = plot_slope_analysis(self.df)
        self.figures.append(('slope_analysis', fig))
        plt.show()
        self.set_status("Slope analysis displayed")

    def show_convergence(self):
        if not self.check_data_loaded():
            return
        self.set_status("Generating convergence analysis...")
        fig = plot_convergence_analysis(self.df)
        self.figures.append(('convergence', fig))
        plt.show()
        self.set_status("Convergence analysis displayed")

    def show_residual_analysis(self):
        # v33.9: residual-magnitude diagnostics for hybrid PPO runs
        if not self.check_data_loaded():
            return
        self.set_status("Generating residual analysis...")
        window = self.window_var.get()
        fig = plot_residual_analysis(self.df, window)
        self.figures.append(('residual_analysis', fig))
        plt.show()
        self.set_status("Residual analysis displayed")

    def show_all_plots(self):
        if not self.check_data_loaded():
            return
        self.set_status("Generating all plots...")
        window = self.window_var.get()
        self.figures = []
        self.figures.append(('learning_curves', plot_learning_curves(self.df, window)))
        self.figures.append(('distributions', plot_distributions(self.df)))
        self.figures.append(('terrain_analysis', plot_terrain_analysis(self.df)))
        self.figures.append(('binned_analysis', plot_binned_analysis(self.df)))
        self.figures.append(('slope_analysis', plot_slope_analysis(self.df)))
        self.figures.append(('convergence', plot_convergence_analysis(self.df)))
        # v33.9: only included if the CSV actually has residual columns; the
        # plot function itself draws a "no data" placeholder for older runs.
        self.figures.append(('residual_analysis', plot_residual_analysis(self.df, window)))
        plt.show()
        self.set_status(f"Generated {len(self.figures)} plots")

    def export_report(self):
        if not self.check_data_loaded():
            return
        filepath = filedialog.asksaveasfilename(title="Save Report", defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt")])
        if filepath:
            try:
                report = generate_statistics_report(self.df)
                with open(filepath, 'w') as f:
                    f.write(report)
                self.set_status(f"Report saved to: {filepath}")
                messagebox.showinfo("Success", f"Report saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report:\n{e}")

    def export_plots_pdf(self):
        if not self.figures:
            if self.df is not None:
                self.show_all_plots()
                plt.close('all')
            else:
                messagebox.showwarning("No Plots", "Generate some plots first.")
                return
        filepath = filedialog.asksaveasfilename(title="Save Plots as PDF", defaultextension=".pdf",
                                                filetypes=[("PDF files", "*.pdf")])
        if filepath:
            try:
                with PdfPages(filepath) as pdf:
                    for name, fig in self.figures:
                        pdf.savefig(fig, bbox_inches='tight')
                self.set_status(f"Plots saved to: {filepath}")
                messagebox.showinfo("Success", f"Plots saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plots:\n{e}")

    def export_plots_png(self):
        if not self.figures:
            if self.df is not None:
                self.show_all_plots()
                plt.close('all')
            else:
                messagebox.showwarning("No Plots", "Generate some plots first.")
                return
        dirpath = filedialog.askdirectory(title="Select Output Directory for PNG files")
        if dirpath:
            try:
                for name, fig in self.figures:
                    filepath = os.path.join(dirpath, f"{name}.png")
                    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
                self.set_status(f"Plots saved to: {dirpath}")
                messagebox.showinfo("Success", f"Saved {len(self.figures)} plots to:\n{dirpath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plots:\n{e}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def run_cli_comparison(args):
    """Run comparison from command line."""
    print("Loading LQR data...")
    lqr_df = load_screening_data(args.lqr)
    print(f"  Loaded {len(lqr_df)} episodes")

    print("Loading Hybrid data...")
    hybrid_df = load_screening_data(args.hybrid)
    print(f"  Loaded {len(hybrid_df)} episodes")

    print(f"\nFiltering parameters:")
    print(f"  Terrain: {args.terrain_min}% - {args.terrain_max}%")
    print(f"  Friction: {args.friction_min}% - {args.friction_max}%")

    report = generate_comparison_report(lqr_df, hybrid_df, args.terrain_max)
    print(report)

    os.makedirs(args.output, exist_ok=True)

    print("\nGenerating comparison box plots...")
    fig = plot_combined_comparison(lqr_df, hybrid_df,
                                   terrain_min=args.terrain_min, terrain_max=args.terrain_max,
                                   friction_min=args.friction_min, friction_max=args.friction_max,
                                   y_min=args.y_min, y_max=args.y_max)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(args.output, f'cte_comparison_{timestamp}.{args.format}')
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filepath}")

    if not args.no_show:
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Training Session Evaluation - Analyze rover training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Launch GUI:
    python evaluate_training.py

    # Launch GUI with file pre-loaded:
    python evaluate_training.py training_data.csv

    # Command-line comparison:
    python evaluate_training.py --compare --lqr lqr_episodes.csv --hybrid hybrid_episodes.csv

    # With terrain/friction filtering (terrain 0-70%, friction 30-100%):
    python evaluate_training.py --compare --lqr lqr.csv --hybrid hybrid.csv \\
        --terrain-min 0 --terrain-max 70 --friction-min 30 --friction-max 100

    # Adjust y-axis:
    python evaluate_training.py --compare --lqr lqr.csv --hybrid hybrid.csv --y-max 3.0
        """
    )

    parser.add_argument('csv_file', nargs='?', help='CSV file to load (optional)')
    parser.add_argument('--compare', action='store_true', help='Run comparison mode')
    parser.add_argument('--lqr', type=str, help='Path to LQR screening data')
    parser.add_argument('--hybrid', type=str, help='Path to Hybrid screening data')
    parser.add_argument('--output', '-o', type=str, default='figures',
                        help='Output directory for figures (default: figures)')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                        help='Output figure format (default: png)')
    parser.add_argument('--terrain-min', type=float, default=DEFAULT_TERRAIN_MIN,
                        help=f'Minimum terrain intensity (default: {DEFAULT_TERRAIN_MIN})')
    parser.add_argument('--terrain-max', type=float, default=DEFAULT_TERRAIN_MAX,
                        help=f'Maximum terrain intensity (default: {DEFAULT_TERRAIN_MAX})')
    parser.add_argument('--friction-min', type=float, default=DEFAULT_FRICTION_MIN,
                        help=f'Minimum friction intensity (default: {DEFAULT_FRICTION_MIN})')
    parser.add_argument('--friction-max', type=float, default=DEFAULT_FRICTION_MAX,
                        help=f'Maximum friction intensity (default: {DEFAULT_FRICTION_MAX})')
    parser.add_argument('--y-min', type=float, default=DEFAULT_CTE_Y_MIN,
                        help=f'Y-axis minimum (default: {DEFAULT_CTE_Y_MIN})')
    parser.add_argument('--y-max', type=float, default=DEFAULT_CTE_Y_MAX,
                        help=f'Y-axis maximum (default: {DEFAULT_CTE_Y_MAX})')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display plots')

    args = parser.parse_args()

    if args.compare:
        if not args.lqr or not args.hybrid:
            print("Error: --compare requires both --lqr and --hybrid arguments")
            parser.print_help()
            sys.exit(1)
        run_cli_comparison(args)
        return

    if not TKINTER_AVAILABLE:
        print("Error: tkinter is not available.")
        print("For comparison mode, use: python evaluate_training.py --compare --lqr <lqr.csv> --hybrid <hybrid.csv>")
        sys.exit(1)

    root = tk.Tk()
    app = TrainingEvaluationGUI(root)

    if args.csv_file and os.path.exists(args.csv_file):
        try:
            app.df = load_training_data(args.csv_file)
            app.csv_path = args.csv_file
            app.update_data_info()
            app.set_status(f"Loaded: {args.csv_file}")
            app.append_output(f"Loaded {len(app.df)} episodes from:\n{args.csv_file}\n")
        except Exception as e:
            app.append_output(f"Error loading file: {e}\n")

    root.mainloop()


if __name__ == "__main__":
    main()