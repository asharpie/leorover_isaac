# random_path_generator.py
"""
Random curved path generator with curvature intensity control.

Generates paths composed of random curved segments where:
- Each segment has random length (1-10m)
- Each segment curves with angle change within [min_angle, max_angle]
- Total path length is approximately 10m
- All paths start at origin (0,0) facing +X direction
"""

import numpy as np
import math
from typing import List, Tuple


class RandomCurvedPath:
    """
    Generates random curved paths with controlled curvature intensity.

    Each path consists of multiple curved segments connected end-to-end,
    where each segment:
    - Has a random length between min_segment_length and max_segment_length
    - Curves with a total angle change between min_curvature_angle and max_curvature_angle
    - Has controlled sharpness (maximum curvature κ_max to prevent impossible turns)
    """

    def __init__(
            self,
            min_curvature_angle: float = 45.0,
            max_curvature_angle: float = 120.0,
            total_distance: float = 10.0,
            min_segment_length: float = 1.0,
            max_segment_length: float = 10.0,
            max_curvature: float = 0.8,  # radians per meter (prevents too-sharp turns)
            waypoints_per_meter: int = 8,
            seed: int = None
    ):
        """
        Initialize the random path generator.

        Args:
            min_curvature_angle: Minimum angle change per segment (degrees)
            max_curvature_angle: Maximum angle change per segment (degrees)
            total_distance: Target total path length (meters)
            min_segment_length: Minimum length per segment (meters)
            max_segment_length: Maximum length per segment (meters)
            max_curvature: Maximum curvature κ = dθ/ds (rad/m) to prevent impossible turns
            waypoints_per_meter: Waypoint density for smooth curves
            seed: Random seed for reproducibility
        """
        self.min_curvature_angle = min_curvature_angle
        self.max_curvature_angle = max_curvature_angle
        self.total_distance = total_distance
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.max_curvature = max_curvature
        self.waypoints_per_meter = waypoints_per_meter
        self.seed = seed

        # Path metadata
        self.name = "RandomCurved"
        self.path_type = "curved-random"

    def generate_waypoints(self) -> List[Tuple[float, float, float]]:
        """
        Generate random curved path waypoints.

        Returns:
            List of (x, y, yaw) tuples representing the path
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Convert angles to radians
        min_angle_rad = np.deg2rad(self.min_curvature_angle)
        max_angle_rad = np.deg2rad(self.max_curvature_angle)

        waypoints = []

        # Start at origin facing +X
        current_x = 0.0
        current_y = 0.0
        current_yaw = 0.0

        # Add starting waypoint
        waypoints.append((float(current_x), float(current_y), float(current_yaw)))

        distance_covered = 0.0
        segment_count = 0

        while distance_covered < self.total_distance:
            # Determine segment length
            remaining = self.total_distance - distance_covered
            segment_length = np.random.uniform(self.min_segment_length, self.max_segment_length)
            segment_length = min(segment_length, remaining)

            # Ensure minimum segment length
            if segment_length < 0.5:
                segment_length = remaining

            # Choose random angle change for this segment
            # Can be positive (left turn) or negative (right turn)
            angle_sign = np.random.choice([-1, 1])
            angle_magnitude = np.random.uniform(min_angle_rad, max_angle_rad)
            angle_change = angle_sign * angle_magnitude

            # Calculate curvature and check if it's too sharp
            curvature = angle_change / segment_length

            # If curvature is too sharp, reduce angle change
            if abs(curvature) > self.max_curvature:
                angle_change = angle_sign * self.max_curvature * segment_length
                if abs(angle_change) < min_angle_rad:
                    # If we can't meet minimum angle, reduce segment length
                    segment_length = min_angle_rad / self.max_curvature
                    angle_change = angle_sign * min_angle_rad

            # Generate waypoints along this curved segment
            segment_waypoints = self._generate_arc_waypoints(
                start_x=current_x,
                start_y=current_y,
                start_yaw=current_yaw,
                arc_length=segment_length,
                angle_change=angle_change
            )

            # Add segment waypoints (skip first as it duplicates last waypoint)
            waypoints.extend(segment_waypoints[1:])

            # Update current position for next segment
            current_x, current_y, current_yaw = segment_waypoints[-1]
            distance_covered += segment_length
            segment_count += 1

        return waypoints

    def _generate_arc_waypoints(
            self,
            start_x: float,
            start_y: float,
            start_yaw: float,
            arc_length: float,
            angle_change: float
    ) -> List[Tuple[float, float, float]]:
        """
        Generate waypoints along a circular arc.

        Args:
            start_x: Starting x position
            start_y: Starting y position
            start_yaw: Starting heading angle
            arc_length: Length of the arc
            angle_change: Total angle change over the arc (radians)

        Returns:
            List of (x, y, yaw) waypoints along the arc
        """
        waypoints = []

        # Number of waypoints for this segment
        num_waypoints = max(2, int(arc_length * self.waypoints_per_meter))

        # Handle straight line case (very small angle change)
        if abs(angle_change) < 1e-6:
            for i in range(num_waypoints + 1):
                t = i / num_waypoints
                s = t * arc_length
                x = start_x + s * np.cos(start_yaw)
                y = start_y + s * np.sin(start_yaw)
                yaw = start_yaw
                waypoints.append((float(x), float(y), float(yaw)))
            return waypoints

        # Calculate radius of curvature for circular arc
        radius = arc_length / abs(angle_change)

        # Center of the circular arc
        # Turn left (positive angle): center is to the left
        # Turn right (negative angle): center is to the right
        sign = np.sign(angle_change)
        cx = start_x - radius * np.sin(start_yaw) * sign
        cy = start_y + radius * np.cos(start_yaw) * sign

        # Generate points along the arc
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            theta = t * angle_change  # Current angle along arc

            # Calculate position on arc
            # The angle from center to current point
            angle_from_center = start_yaw + theta - np.pi / 2 * sign

            x = cx + radius * np.cos(angle_from_center)
            y = cy + radius * np.sin(angle_from_center)
            yaw = start_yaw + theta

            # Normalize yaw to [-pi, pi]
            yaw = ((yaw + np.pi) % (2 * np.pi)) - np.pi

            waypoints.append((float(x), float(y), float(yaw)))

        return waypoints

    def get_waypoints(self) -> List[Tuple[float, float, float]]:
        """
        Get waypoints (compatible with path template interface).

        Returns:
            List of (x, y, yaw) waypoints
        """
        return self.generate_waypoints()

    def get_required_checkpoints(self):
        """
        No required checkpoints for random curved paths.

        Returns:
            None
        """
        return None

    def visualize_curvature(self, waypoints: List[Tuple[float, float, float]]) -> None:
        """
        Calculate and print curvature statistics for the generated path.

        Args:
            waypoints: List of (x, y, yaw) waypoints
        """
        if len(waypoints) < 2:
            return

        curvatures = []

        for i in range(len(waypoints) - 1):
            x1, y1, yaw1 = waypoints[i]
            x2, y2, yaw2 = waypoints[i + 1]

            # Distance between waypoints
            ds = np.hypot(x2 - x1, y2 - y1)

            # Angle change (wrapped to [-pi, pi])
            dtheta = ((yaw2 - yaw1 + np.pi) % (2 * np.pi)) - np.pi

            # Curvature
            if ds > 1e-6:
                kappa = dtheta / ds
                curvatures.append(abs(kappa))

        if curvatures:
            print(f"\nPath Curvature Statistics:")
            print(f"  Mean curvature: {np.mean(curvatures):.4f} rad/m")
            print(f"  Max curvature:  {np.max(curvatures):.4f} rad/m")
            print(f"  Min curvature:  {np.min(curvatures):.4f} rad/m")
            print(f"  Std curvature:  {np.std(curvatures):.4f} rad/m")

            # Equivalent angles over 1m distance
            print(f"  Max angle change over 1m: {np.rad2deg(np.max(curvatures)):.1f}°")


def generate_random_curved_path(
        min_curvature_angle: float,
        max_curvature_angle: float,
        total_distance: float = 10.0,
        seed: int = None
) -> RandomCurvedPath:
    """
    Factory function to create a random curved path generator.

    Args:
        min_curvature_angle: Minimum angle change per segment (degrees)
        max_curvature_angle: Maximum angle change per segment (degrees)
        total_distance: Target total path length (meters)
        seed: Random seed for reproducibility

    Returns:
        RandomCurvedPath instance
    """
    return RandomCurvedPath(
        min_curvature_angle=min_curvature_angle,
        max_curvature_angle=max_curvature_angle,
        total_distance=total_distance,
        seed=seed
    )