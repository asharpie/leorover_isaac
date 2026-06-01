# terrain_stats.py
"""
Terrain slope statistics calculation utilities.

Provides functions to measure actual terrain slope statistics after generation:
- max_slope_degrees: Maximum slope angle anywhere on terrain
- avg_slope_degrees: Average slope angle across the terrain
- slope_std_degrees: Standard deviation of slopes

These are measurements of the generated terrain, not control parameters.
"""

import math
import numpy as np


def calculate_terrain_slope_stats(heightfieldData, num_rows=512, num_cols=512, cell_size=0.05):
    """
    Calculate comprehensive slope statistics from a heightfield.

    Args:
        heightfieldData: List/array of height values (row-major order)
        num_rows: Number of rows in heightfield grid
        num_cols: Number of columns in heightfield grid
        cell_size: Physical size of each grid cell (meters)

    Returns:
        dict with keys:
            - max_slope_degrees: Maximum slope angle (degrees)
            - avg_slope_degrees: Average slope angle (degrees)
            - slope_std_degrees: Standard deviation of slopes (degrees)
            - median_slope_degrees: Median slope angle (degrees)
            - slope_percentile_90: 90th percentile slope (degrees)
            - slope_percentile_95: 95th percentile slope (degrees)
            - min_height: Minimum terrain height (meters)
            - max_height: Maximum terrain height (meters)
            - height_range: Difference between max and min height (meters)
            - num_samples: Number of slope samples computed
    """
    if len(heightfieldData) == 0:
        return {
            'max_slope_degrees': 0.0,
            'avg_slope_degrees': 0.0,
            'slope_std_degrees': 0.0,
            'median_slope_degrees': 0.0,
            'slope_percentile_90': 0.0,
            'slope_percentile_95': 0.0,
            'min_height': 0.0,
            'max_height': 0.0,
            'height_range': 0.0,
            'num_samples': 0,
        }

    # Height statistics
    min_height = min(heightfieldData)
    max_height = max(heightfieldData)
    height_range = max_height - min_height

    # Calculate slopes at each interior point
    slope_degrees_list = []

    for y in range(1, num_rows - 1):
        for x in range(1, num_cols - 1):
            idx = y * num_rows + x
            idx_right = y * num_rows + (x + 1)
            idx_left = y * num_rows + (x - 1)
            idx_up = (y + 1) * num_rows + x
            idx_down = (y - 1) * num_rows + x

            # Central difference for better accuracy
            slope_x = (heightfieldData[idx_right] - heightfieldData[idx_left]) / (2 * cell_size)
            slope_y = (heightfieldData[idx_up] - heightfieldData[idx_down]) / (2 * cell_size)

            # Slope magnitude (gradient)
            slope_magnitude = math.sqrt(slope_x ** 2 + slope_y ** 2)

            # Convert to angle in degrees
            slope_radians = math.atan(slope_magnitude)
            slope_degrees = math.degrees(slope_radians)

            slope_degrees_list.append(slope_degrees)

    # Convert to numpy for statistics
    slopes = np.array(slope_degrees_list)

    if len(slopes) == 0:
        return {
            'max_slope_degrees': 0.0,
            'avg_slope_degrees': 0.0,
            'slope_std_degrees': 0.0,
            'median_slope_degrees': 0.0,
            'slope_percentile_90': 0.0,
            'slope_percentile_95': 0.0,
            'min_height': float(min_height),
            'max_height': float(max_height),
            'height_range': float(height_range),
            'num_samples': 0,
        }

    return {
        'max_slope_degrees': float(np.max(slopes)),
        'avg_slope_degrees': float(np.mean(slopes)),
        'slope_std_degrees': float(np.std(slopes)),
        'median_slope_degrees': float(np.median(slopes)),
        'slope_percentile_90': float(np.percentile(slopes, 90)),
        'slope_percentile_95': float(np.percentile(slopes, 95)),
        'min_height': float(min_height),
        'max_height': float(max_height),
        'height_range': float(height_range),
        'num_samples': len(slopes),
    }


def calculate_local_slope_at_position(x_world, y_world, heightfieldData,
                                      num_rows=512, num_cols=512, cell_size=0.05):
    """
    Calculate the local slope at a specific world position.

    Useful for getting the slope under the rover at each timestep.

    Args:
        x_world, y_world: World coordinates
        heightfieldData: Heightfield data array
        num_rows, num_cols: Grid dimensions
        cell_size: Physical cell size

    Returns:
        dict with:
            - slope_degrees: Local slope angle (degrees)
            - slope_x: Slope in x direction
            - slope_y: Slope in y direction
            - height: Height at this position
    """
    # Convert world coordinates to grid indices
    x_index = int((x_world + (num_rows * cell_size / 2.0)) / cell_size)
    y_index = int((y_world + (num_cols * cell_size / 2.0)) / cell_size)

    # Clamp to valid range (with margin for gradient calculation)
    x_index = max(1, min(num_rows - 2, x_index))
    y_index = max(1, min(num_cols - 2, y_index))

    # Get indices for gradient calculation
    idx = y_index * num_rows + x_index
    idx_right = y_index * num_rows + (x_index + 1)
    idx_left = y_index * num_rows + (x_index - 1)
    idx_up = (y_index + 1) * num_rows + x_index
    idx_down = (y_index - 1) * num_rows + x_index

    # Central difference gradient
    slope_x = (heightfieldData[idx_right] - heightfieldData[idx_left]) / (2 * cell_size)
    slope_y = (heightfieldData[idx_up] - heightfieldData[idx_down]) / (2 * cell_size)

    # Slope magnitude and angle
    slope_magnitude = math.sqrt(slope_x ** 2 + slope_y ** 2)
    slope_degrees = math.degrees(math.atan(slope_magnitude))

    return {
        'slope_degrees': float(slope_degrees),
        'slope_x': float(slope_x),
        'slope_y': float(slope_y),
        'height': float(heightfieldData[idx]),
    }


def print_terrain_slope_stats(stats: dict, prefix: str = ""):
    """Print terrain slope statistics in a formatted way."""
    print(f"\n{prefix}TERRAIN SLOPE STATISTICS")
    print(f"{prefix}" + "-" * 40)
    print(f"{prefix}Max Slope:      {stats['max_slope_degrees']:.2f}°")
    print(f"{prefix}Avg Slope:      {stats['avg_slope_degrees']:.2f}°")
    print(f"{prefix}Median Slope:   {stats['median_slope_degrees']:.2f}°")
    print(f"{prefix}Std Dev:        {stats['slope_std_degrees']:.2f}°")
    print(f"{prefix}90th Percentile:{stats['slope_percentile_90']:.2f}°")
    print(f"{prefix}95th Percentile:{stats['slope_percentile_95']:.2f}°")
    print(f"{prefix}Height Range:   {stats['height_range']:.4f} m")
    print(f"{prefix}" + "-" * 40)