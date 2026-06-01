# factorial_screening.py
"""
Full factorial screening experiment automation.

Generates all 2^4 = 16 combinations of the four binary factors:
- max_slope (low/high)
- avg_slope (low/high)
- friction (low/high)
- sharp_turn (low/high)

Each combination is tested for SCREENING_EPISODES_PER_COMBINATION episodes.
"""

import itertools
import os
import csv
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

import config


class FactorCombination:
    """Represents a single combination of factor levels."""

    def __init__(self, combination_id: int, max_slope: str, avg_slope: str,
                 friction: str, sharp_turn: str):
        self.combination_id = combination_id
        self.max_slope = max_slope  # 'low' or 'high'
        self.avg_slope = avg_slope  # 'low' or 'high'
        self.friction = friction  # 'low' or 'high'
        self.sharp_turn = sharp_turn  # 'low' or 'high'

    def get_terrain_intensity(self) -> float:
        """
        Get terrain intensity parameter based on max_slope and avg_slope factors.
        Combines both factors to determine overall terrain difficulty.
        """
        max_slope_val = config.FACTORS['max_slope'][self.max_slope]
        avg_slope_val = config.FACTORS['avg_slope'][self.avg_slope]

        # Use weighted average: max_slope has more impact
        terrain_intensity = 0.6 * max_slope_val + 0.4 * avg_slope_val
        return float(np.clip(terrain_intensity, 0.0, 100.0))

    def get_friction_intensity(self) -> float:
        """Get friction intensity parameter."""
        return config.FACTORS['friction'][self.friction]

    def get_path_curvature_params(self) -> Tuple[float, float]:
        """
        Get path curvature parameters (min, max angles) based on sharp_turn factor.

        Returns:
            (min_curvature_angle, max_curvature_angle)
        """
        if self.sharp_turn == 'low':
            # Gentle curved paths
            return (config.GENTLE_PATH_CURVATURE_MIN,
                    config.GENTLE_PATH_CURVATURE_MAX)
        else:
            # Sharp turning paths
            return (config.SHARP_PATH_CURVATURE_MIN,
                    config.SHARP_PATH_CURVATURE_MAX)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'combination_id': self.combination_id,
            'max_slope': self.max_slope,
            'avg_slope': self.avg_slope,
            'friction': self.friction,
            'sharp_turn': self.sharp_turn,
            'terrain_intensity': self.get_terrain_intensity(),
            'friction_intensity': self.get_friction_intensity(),
            'min_curvature': self.get_path_curvature_params()[0],
            'max_curvature': self.get_path_curvature_params()[1],
        }

    def __str__(self):
        return (f"Combination {self.combination_id}: "
                f"max_slope={self.max_slope}, avg_slope={self.avg_slope}, "
                f"friction={self.friction}, sharp_turn={self.sharp_turn}")


def generate_factorial_combinations() -> List[FactorCombination]:
    """
    Generate all 2^4 = 16 combinations of the four binary factors.

    Returns:
        List of FactorCombination objects
    """
    factors = ['max_slope', 'avg_slope', 'friction', 'sharp_turn']
    levels = ['low', 'high']

    # Generate all combinations using itertools.product
    combinations = list(itertools.product(levels, repeat=4))

    factorial_combinations = []
    for i, combo in enumerate(combinations):
        max_slope, avg_slope, friction, sharp_turn = combo
        fc = FactorCombination(
            combination_id=i,
            max_slope=max_slope,
            avg_slope=avg_slope,
            friction=friction,
            sharp_turn=sharp_turn
        )
        factorial_combinations.append(fc)

    return factorial_combinations


class ScreeningLogger:
    """Handles logging for screening experiments including terrain slope data."""

    def __init__(self, agent_type: str):
        """
        Initialize screening logger.

        Args:
            agent_type: "PPO", "Hybrid", or "LQR"
        """
        self.agent_type = agent_type

        # Create screening results directory
        os.makedirs(config.SCREENING_LOG_DIR, exist_ok=True)

        # Create timestamped directory for this screening run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(
            config.SCREENING_LOG_DIR,
            f"screening_{agent_type}_{timestamp}"
        )
        os.makedirs(self.session_dir, exist_ok=True)

        # CSV file for per-episode results
        self.episode_csv_path = os.path.join(
            self.session_dir,
            "screening_episodes.csv"
        )

        # CSV file for per-combination summary
        self.summary_csv_path = os.path.join(
            self.session_dir,
            "screening_summary.csv"
        )

        # Initialize episode CSV
        self._init_episode_csv()

        # Initialize summary CSV
        self._init_summary_csv()

        # Storage for computing summaries
        self.combination_data = {}

        print(f"\n{'=' * 70}")
        print(f"SCREENING SESSION INITIALIZED")
        print(f"{'=' * 70}")
        print(f"Agent Type:     {self.agent_type}")
        print(f"Session Dir:    {self.session_dir}")
        print(f"Episode Log:    {self.episode_csv_path}")
        print(f"Summary Log:    {self.summary_csv_path}")
        print(f"{'=' * 70}\n")

    def _init_episode_csv(self):
        """Initialize episode-level CSV with headers including terrain slope data."""
        headers = [
            'combination_id',
            'max_slope_level',
            'avg_slope_level',
            'friction_level',
            'sharp_turn_level',
            'terrain_intensity',
            'friction_intensity',
            'min_curvature',
            'max_curvature',
            'episode_in_combination',
            'global_episode',
            'episode_length',
            'cumulative_reward',
            'mean_cross_track_error',
            'max_cross_track_error',
            'final_distance_to_goal',
            'mean_slip',
            'max_slip',
            'energy_proxy',
            'success',
            'path_name',
            'path_type',
            # NEW: Terrain slope columns
            'terrain_max_slope_deg',
            'terrain_avg_slope_deg',
            'mean_local_slope_deg',
            'max_local_slope_deg',
        ]

        with open(self.episode_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _init_summary_csv(self):
        """Initialize summary CSV with headers including terrain slope data."""
        headers = [
            'combination_id',
            'max_slope_level',
            'avg_slope_level',
            'friction_level',
            'sharp_turn_level',
            'terrain_intensity',
            'friction_intensity',
            'min_curvature',
            'max_curvature',
            'num_episodes',
            'mean_cross_track_error',
            'std_cross_track_error',
            'median_cross_track_error',
            'min_cross_track_error',
            'max_cross_track_error',
            'success_rate',
            'mean_slip',
            'mean_energy',
            'mean_episode_length',
            # NEW: Terrain slope summary columns
            'mean_terrain_max_slope_deg',
            'mean_terrain_avg_slope_deg',
            'mean_local_slope_deg',
        ]

        with open(self.summary_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_episode(self, combination: FactorCombination,
                    episode_in_combination: int, global_episode: int,
                    episode_data: Dict):
        """
        Log a single episode's results including terrain slope data.

        Args:
            combination: FactorCombination being tested
            episode_in_combination: Episode number within this combination (1-100)
            global_episode: Global episode counter
            episode_data: Dictionary with episode metrics including terrain slopes
        """
        combo_dict = combination.to_dict()

        row = [
            combo_dict['combination_id'],
            combination.max_slope,
            combination.avg_slope,
            combination.friction,
            combination.sharp_turn,
            f"{combo_dict['terrain_intensity']:.2f}",
            f"{combo_dict['friction_intensity']:.2f}",
            f"{combo_dict['min_curvature']:.2f}",
            f"{combo_dict['max_curvature']:.2f}",
            episode_in_combination,
            global_episode,
            episode_data.get('episode_length', 0),
            f"{episode_data.get('cumulative_reward', 0.0):.4f}",
            f"{episode_data.get('mean_cross_track_error', 0.0):.4f}",
            f"{episode_data.get('max_cross_track_error', 0.0):.4f}",
            f"{episode_data.get('final_distance_to_goal', 0.0):.4f}",
            f"{episode_data.get('mean_slip', 0.0):.4f}",
            f"{episode_data.get('max_slip', 0.0):.4f}",
            f"{episode_data.get('energy_proxy', 0.0):.4f}",
            episode_data.get('success', 0),
            episode_data.get('path_name', 'unknown'),
            episode_data.get('path_type', 'unknown'),
            # NEW: Terrain slope data
            f"{episode_data.get('terrain_max_slope_deg', 0.0):.2f}",
            f"{episode_data.get('terrain_avg_slope_deg', 0.0):.2f}",
            f"{episode_data.get('mean_local_slope_deg', 0.0):.2f}",
            f"{episode_data.get('max_local_slope_deg', 0.0):.2f}",
        ]

        with open(self.episode_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Accumulate data for summary
        combo_id = combo_dict['combination_id']
        if combo_id not in self.combination_data:
            self.combination_data[combo_id] = {
                'combination': combination,
                'cross_track_errors': [],
                'successes': [],
                'slips': [],
                'energies': [],
                'episode_lengths': [],
                # NEW: Terrain slope accumulators
                'terrain_max_slopes': [],
                'terrain_avg_slopes': [],
                'local_slopes': [],
            }

        self.combination_data[combo_id]['cross_track_errors'].append(
            episode_data.get('mean_cross_track_error', 0.0)
        )
        self.combination_data[combo_id]['successes'].append(
            episode_data.get('success', 0)
        )
        self.combination_data[combo_id]['slips'].append(
            episode_data.get('mean_slip', 0.0)
        )
        self.combination_data[combo_id]['energies'].append(
            episode_data.get('energy_proxy', 0.0)
        )
        self.combination_data[combo_id]['episode_lengths'].append(
            episode_data.get('episode_length', 0)
        )
        # NEW: Accumulate terrain slope data
        self.combination_data[combo_id]['terrain_max_slopes'].append(
            episode_data.get('terrain_max_slope_deg', 0.0)
        )
        self.combination_data[combo_id]['terrain_avg_slopes'].append(
            episode_data.get('terrain_avg_slope_deg', 0.0)
        )
        self.combination_data[combo_id]['local_slopes'].append(
            episode_data.get('mean_local_slope_deg', 0.0)
        )

    def log_combination_summary(self, combination: FactorCombination):
        """
        Log summary statistics for a completed combination including terrain slope data.

        Args:
            combination: FactorCombination that was tested
        """
        combo_dict = combination.to_dict()
        combo_id = combo_dict['combination_id']

        if combo_id not in self.combination_data:
            print(f"Warning: No data for combination {combo_id}")
            return

        data = self.combination_data[combo_id]

        cte_values = data['cross_track_errors']
        success_values = data['successes']
        slip_values = data['slips']
        energy_values = data['energies']
        length_values = data['episode_lengths']
        # NEW
        terrain_max_slopes = data['terrain_max_slopes']
        terrain_avg_slopes = data['terrain_avg_slopes']
        local_slopes = data['local_slopes']

        row = [
            combo_dict['combination_id'],
            combination.max_slope,
            combination.avg_slope,
            combination.friction,
            combination.sharp_turn,
            f"{combo_dict['terrain_intensity']:.2f}",
            f"{combo_dict['friction_intensity']:.2f}",
            f"{combo_dict['min_curvature']:.2f}",
            f"{combo_dict['max_curvature']:.2f}",
            len(cte_values),
            f"{np.mean(cte_values):.4f}",
            f"{np.std(cte_values):.4f}",
            f"{np.median(cte_values):.4f}",
            f"{np.min(cte_values):.4f}",
            f"{np.max(cte_values):.4f}",
            f"{np.mean(success_values):.4f}",
            f"{np.mean(slip_values):.4f}",
            f"{np.mean(energy_values):.4f}",
            f"{np.mean(length_values):.1f}",
            # NEW: Terrain slope summary
            f"{np.mean(terrain_max_slopes):.2f}",
            f"{np.mean(terrain_avg_slopes):.2f}",
            f"{np.mean(local_slopes):.2f}",
        ]

        with open(self.summary_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Print summary with terrain data
        print(f"\n--- Combination {combo_id} Summary ---")
        print(f"  Episodes: {len(cte_values)}")
        print(f"  Mean CTE: {np.mean(cte_values):.4f} m")
        print(f"  Success Rate: {np.mean(success_values)*100:.1f}%")
        print(f"  Terrain Max Slope: {np.mean(terrain_max_slopes):.1f}° (mean across episodes)")
        print(f"  Terrain Avg Slope: {np.mean(terrain_avg_slopes):.1f}° (mean across episodes)")
        print(f"  Local Slope (under rover): {np.mean(local_slopes):.1f}° (mean)")

    def start_combination(self, combination: FactorCombination):
        """Log the start of a new combination."""
        pass  # Handled by callback

    def finish_combination(self, combination: FactorCombination):
        """Log the end of a combination and write summary."""
        self.log_combination_summary(combination)

    def finalize(self):
        """
        Finalize the screening session.
        Write any remaining data and print final summary.
        """
        print(f"\n{'=' * 70}")
        print(f"SCREENING SESSION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Results saved to: {self.session_dir}")
        print(f"  - Episode data: {self.episode_csv_path}")
        print(f"  - Summary data: {self.summary_csv_path}")
        print(f"{'=' * 70}\n")


def print_factorial_design():
    """Print the full factorial design table."""
    combinations = generate_factorial_combinations()

    print("\n" + "=" * 90)
    print("FULL FACTORIAL SCREENING DESIGN")
    print("=" * 90)
    print(f"Total Combinations: {len(combinations)} (2^4 = 16)")
    print(f"Episodes per Combination: {config.SCREENING_EPISODES_PER_COMBINATION}")
    print(f"Total Episodes: {len(combinations) * config.SCREENING_EPISODES_PER_COMBINATION}")
    print("=" * 90)
    print(f"{'ID':<4} {'MaxSlope':<10} {'AvgSlope':<10} {'Friction':<10} {'Turn':<10} "
          f"{'Terrain%':<10} {'Friction%':<10}")
    print("-" * 90)

    for combo in combinations:
        combo_dict = combo.to_dict()
        print(f"{combo_dict['combination_id']:<4} "
              f"{combo.max_slope:<10} "
              f"{combo.avg_slope:<10} "
              f"{combo.friction:<10} "
              f"{combo.sharp_turn:<10} "
              f"{combo_dict['terrain_intensity']:<10.1f} "
              f"{combo_dict['friction_intensity']:<10.1f}")

    print("=" * 90 + "\n")