# path_templates.py
"""
Predefined path templates for consistent experimental evaluation.

Provides 9 path types:
- 3 zig-zag variants (sharp directional changes) - ENHANCED with more waypoints
- 3 curved variants (smooth arcs) - ENHANCED with more waypoints
- 3 closed polygon variants (rectangular/triangular loops) - ENHANCED with more waypoints

All paths start at origin (0,0) facing +X direction to avoid rover turnaround.
UPDATED: All paths now have strategically placed additional waypoints for better tracking.
"""

import numpy as np
import math


class PathTemplate:
    """Base class for path templates."""

    def __init__(self, name: str, path_type: str):
        self.name = name
        self.path_type = path_type  # 'zig-zag', 'curved', or 'polygon'

    def get_waypoints(self):
        """
        Return waypoints as list of (x, y, yaw) tuples.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_required_checkpoints(self):
        """
        Return required checkpoints that must be visited before goal completion.
        Only applicable for closed polygon paths. Returns None for other path types.

        Returns:
            List of (x, y) tuples representing checkpoint positions, or None
        """
        return None  # Default: no required checkpoints


# =============================================================================
# ZIG-ZAG PATHS (3 variants) - ENHANCED WITH MORE WAYPOINTS
# =============================================================================

class ZigZag1(PathTemplate):
    """Zig-zag with moderate amplitude and spacing - ENHANCED."""

    def __init__(self):
        super().__init__("ZigZag-Moderate", "zig-zag")

    def get_waypoints(self):
        waypoints = [
            (0.0, 0.0, 0.0),  # Start at origin
            (1.0, 0.0, 0.0),  # +1 waypoint on approach
            (2.0, 0.0, 0.0),  # Original: Forward
            (2.5, 0.5, 0.2),  # +1 waypoint: turn entry
            (3.0, 1.0, 0.3),  # +1 waypoint: mid-turn
            (3.5, 1.5, 0.2),  # +1 waypoint: turn exit
            (4.0, 2.0, 0.0),  # Original: Right turn (45°)
            (5.0, 2.0, 0.0),  # +1 waypoint: stabilize
            (6.0, 2.0, 0.0),  # Original: Straight
            (6.5, 1.0, -0.2),  # +1 waypoint: turn entry
            (7.0, 0.0, -0.3),  # +1 waypoint: mid-turn
            (7.5, -1.0, -0.2),  # +1 waypoint: turn exit
            (8.0, -2.0, 0.0),  # Original: Left turn (-45°)
            (9.0, -2.0, 0.0),  # +1 waypoint: stabilize
            (10.0, -2.0, 0.0),  # Original: Goal
        ]
        return waypoints


class ZigZag2(PathTemplate):
    """Zig-zag with sharp turns and tight spacing - ENHANCED."""

    def __init__(self):
        super().__init__("ZigZag-Sharp", "zig-zag")

    def get_waypoints(self):
        waypoints = [
            (0.0, 0.0, 0.0),
            (0.75, 0.0, 0.0),  # +1 waypoint
            (1.5, 0.0, 0.0),
            (2.0, 0.8, 0.5),  # +1 waypoint: turn entry
            (2.5, 1.6, 0.9),  # +1 waypoint: mid-turn
            (3.0, 2.5, 1.047),  # Original: 60° turn
            (3.75, 2.5, 0.5),  # +1 waypoint: turn exit
            (4.5, 2.5, 0.0),
            (5.0, 0.8, -0.5),  # +1 waypoint: turn entry
            (5.5, -0.8, -0.9),  # +1 waypoint: mid-turn
            (6.0, -2.5, -1.047),  # Original: -60° turn
            (6.75, -2.5, -0.5),  # +1 waypoint: turn exit
            (7.5, -2.5, 0.0),
            (8.0, -0.8, 0.3),  # +1 waypoint: turn entry
            (8.5, 0.8, 0.6),  # +1 waypoint: mid-turn
            (9.0, 2.0, 0.785),  # Original: 45° turn
            (9.5, 2.0, 0.4),  # +1 waypoint: turn exit
            (10.0, 2.0, 0.0),
        ]
        return waypoints


class ZigZag3(PathTemplate):
    """Zig-zag with wide amplitude and gentle transitions - ENHANCED."""

    def __init__(self):
        super().__init__("ZigZag-Wide", "zig-zag")

    def get_waypoints(self):
        waypoints = [
            (0.0, 0.0, 0.0),
            (1.5, 0.0, 0.0),  # +1 waypoint
            (3.0, 0.0, 0.0),
            (3.5, 0.5, 0.2),  # +1 waypoint: turn entry
            (4.0, 1.5, 0.4),  # +1 waypoint: mid-turn
            (4.5, 2.5, 0.5),  # +1 waypoint: turn exit
            (5.0, 3.0, 0.524),  # Original: 30° turn
            (6.0, 3.0, 0.3),  # +1 waypoint: stabilize
            (7.0, 3.0, 0.0),
            (7.5, 2.2, -0.2),  # +1 waypoint: turn entry
            (8.0, 1.5, -0.4),  # +1 waypoint: mid-turn
            (8.5, 0.5, -0.5),  # +1 waypoint: turn exit
            (9.0, 0.0, -0.524),  # Original: -30° turn
            (10.5, 0.0, -0.3),  # +1 waypoint: stabilize
            (12.0, 0.0, 0.0),
        ]
        return waypoints


# =============================================================================
# CURVED PATHS (3 variants) - ENHANCED WITH MORE WAYPOINTS
# =============================================================================

class Curved1(PathTemplate):
    """Gentle S-curve - ENHANCED."""

    def __init__(self):
        super().__init__("Curved-Gentle", "curved")

    def get_waypoints(self):
        # Create smooth S-curve using parametric points
        # Increased density: 50 waypoints (was 30)
        waypoints = []
        t_values = np.linspace(0, 2 * np.pi, 50)

        for i, t in enumerate(t_values):
            x = t * 2.0  # Forward progress
            y = 2.0 * np.sin(t)  # Sinusoidal lateral movement

            # Calculate tangent angle
            dx = 2.0
            dy = 2.0 * np.cos(t)
            yaw = math.atan2(dy, dx)

            # Override first waypoint to face forward
            if i == 0:
                yaw = 0.0

            waypoints.append((x, y, yaw))

        return waypoints


class Curved2(PathTemplate):
    """Tight arc - sustained right turn - ENHANCED."""

    def __init__(self):
        super().__init__("Curved-TightArc", "curved")

    def get_waypoints(self):
        waypoints = []
        radius = 4.0
        angle_span = np.pi  # 180° arc

        # Increased density: 60 waypoints (was 36)
        num_points = 60
        angles = np.linspace(0, angle_span, num_points)

        for angle in angles:
            # Arc centered at (0, radius)
            x = radius * np.sin(angle)
            y = radius - radius * np.cos(angle)
            yaw = angle

            waypoints.append((x, y, yaw))

        return waypoints


class Curved3(PathTemplate):
    """Double curve - chicane pattern with minimal waypoints."""

    def __init__(self):
        super().__init__("Curved-Chicane", "curved")

    def get_waypoints(self):
        waypoints = []

        # First curve (right) - 8 waypoints
        radius1 = 3.0
        angles1 = np.linspace(0, np.pi / 2, 8)
        for angle in angles1:
            x = radius1 * np.sin(angle)
            y = radius1 - radius1 * np.cos(angle)
            yaw = angle
            waypoints.append((x, y, yaw))

        x1, y1, yaw1 = waypoints[-1]

        x_offset = radius1
        y_offset = radius1
        radius2 = 3.0
        angle0 = 0.0
        x2 = x_offset + radius2 * np.sin(angle0)
        y2 = y_offset + radius2 * np.cos(angle0)
        yaw2 = np.pi / 2 - angle0

        # Straight segment - 2 waypoints
        num_straight_points = 2
        for i in range(1, num_straight_points + 1):
            t = i / (num_straight_points + 1)
            x = (1.0 - t) * x1 + t * x2
            y = (1.0 - t) * y1 + t * y2
            yaw = (1.0 - t) * yaw1 + t * yaw2
            waypoints.append((x, y, yaw))

        # Second curve (left) - 8 waypoints
        angles2 = np.linspace(0, np.pi / 2, 8)[1:]
        for angle in angles2:
            x = x_offset + radius2 * np.sin(angle)
            y = y_offset + radius2 * np.cos(angle)
            yaw = np.pi / 2 - angle
            waypoints.append((x, y, yaw))

        return waypoints


# =============================================================================
# CLOSED POLYGON PATHS (3 variants) - ENHANCED WITH MORE WAYPOINTS
# =============================================================================

class Polygon1(PathTemplate):
    """Rectangle loop (4m x 6m) - ENHANCED."""

    def __init__(self):
        super().__init__("Polygon-Rectangle", "polygon")

    def get_waypoints(self):
        waypoints = [
            # Bottom edge (start to front-right corner)
            (0.0, 0.0, 0.0),  # Start
            (2.0, 0.0, 0.0),  # +1 mid-edge waypoint
            (4.0, 0.0, 0.0),  # +1 mid-edge waypoint
            (6.0, 0.0, 0.0),  # Front-right corner
            (6.0, 0.0, np.pi / 2),  # Turn right

            # Right edge (front-right to back-right corner)
            (6.0, 1.0, np.pi / 2),  # +1 mid-edge waypoint
            (6.0, 2.0, np.pi / 2),  # +1 mid-edge waypoint
            (6.0, 3.0, np.pi / 2),  # +1 mid-edge waypoint
            (6.0, 4.0, np.pi / 2),  # Back-right corner
            (6.0, 4.0, np.pi),  # Turn right

            # Top edge (back-right to back-left corner)
            (4.0, 4.0, np.pi),  # +1 mid-edge waypoint
            (2.0, 4.0, np.pi),  # +1 mid-edge waypoint
            (0.0, 4.0, np.pi),  # Back-left corner
            (0.0, 4.0, -np.pi / 2),  # Turn right

            # Left edge (back-left to origin)
            (0.0, 3.0, -np.pi / 2),  # +1 mid-edge waypoint
            (0.0, 2.0, -np.pi / 2),  # +1 mid-edge waypoint
            (0.0, 1.0, -np.pi / 2),  # +1 mid-edge waypoint
            (0.0, 0.0, -np.pi / 2),  # Close loop at origin
            (0.0, 0.0, 0.0),  # Final orientation
        ]
        return waypoints

    def get_required_checkpoints(self):
        """Return the 3 non-origin corners of the rectangle as required checkpoints."""
        return [
            (6.0, 0.0),  # Front-right corner
            (6.0, 4.0),  # Back-right corner
            (0.0, 4.0),  # Back-left corner
        ]


class Polygon2(PathTemplate):
    """Triangular loop - ENHANCED."""

    def __init__(self):
        super().__init__("Polygon-Triangle", "polygon")

    def get_waypoints(self):
        # Equilateral triangle with 8m sides
        side_length = 8.0
        height = side_length * np.sqrt(3) / 2

        waypoints = [
            # Bottom edge (origin to bottom-right corner)
            (0.0, 0.0, 0.0),  # Start
            (2.0, 0.0, 0.0),  # +1 mid-edge waypoint
            (4.0, 0.0, 0.0),  # +1 mid-edge waypoint
            (6.0, 0.0, 0.0),  # +1 mid-edge waypoint
            (side_length, 0.0, 0.0),  # Bottom-right corner
            (side_length, 0.0, 2 * np.pi / 3),  # Turn 120°

            # Right edge (bottom-right to top corner)
            (side_length - 1.0, height * 0.25, 2 * np.pi / 3),  # +1 mid-edge waypoint
            (side_length - 2.0, height * 0.50, 2 * np.pi / 3),  # +1 mid-edge waypoint
            (side_length - 3.0, height * 0.75, 2 * np.pi / 3),  # +1 mid-edge waypoint
            (side_length / 2, height, 2 * np.pi / 3),  # Top corner
            (side_length / 2, height, -2 * np.pi / 3),  # Turn 120°

            # Left edge (top corner back to origin)
            (side_length / 2 - 1.0, height * 0.75, -2 * np.pi / 3),  # +1 mid-edge waypoint
            (side_length / 2 - 2.0, height * 0.50, -2 * np.pi / 3),  # +1 mid-edge waypoint
            (side_length / 2 - 3.0, height * 0.25, -2 * np.pi / 3),  # +1 mid-edge waypoint
            (0.0, 0.0, -2 * np.pi / 3),  # Return to origin
            (0.0, 0.0, 0.0),  # Close loop with correct orientation
        ]
        return waypoints

    def get_required_checkpoints(self):
        """Return the 2 non-origin corners of the triangle as required checkpoints."""
        side_length = 8.0
        height = side_length * np.sqrt(3) / 2
        return [
            (side_length, 0.0),  # Bottom-right corner
            (side_length / 2, height),  # Top corner
        ]


class Polygon3(PathTemplate):
    """Pentagon loop - ENHANCED."""

    def __init__(self):
        super().__init__("Polygon-Pentagon", "polygon")

    def get_waypoints(self):
        waypoints = []
        radius = 5.0  # Pentagon radius
        num_sides = 5

        # Start with angle offset so first point faces +X
        angle_offset = -np.pi / 2

        # Generate waypoints around the pentagon with 8 points per edge
        # This gives us approximately 40 waypoints total (8 per side × 5 sides)
        points_per_edge = 8
        total_points = num_sides * points_per_edge

        for i in range(total_points):
            # Map i to a continuous angle around the pentagon
            angle_frac = i / (total_points - 1)
            angle = angle_frac * num_sides * (2 * np.pi / num_sides) + angle_offset

            x = radius * np.cos(angle)
            y = radius * np.sin(angle) + radius  # Offset to start at origin

            # Yaw points toward next vertex
            next_angle_frac = (i + 1) / (total_points - 1)
            next_angle = next_angle_frac * num_sides * (2 * np.pi / num_sides) + angle_offset
            yaw = math.atan2(
                radius * np.sin(next_angle) + radius - y,
                radius * np.cos(next_angle) - x
            )

            # Override first waypoint to face forward
            if i == 0:
                yaw = 0.0

            waypoints.append((x, y, yaw))

        return waypoints

    def get_required_checkpoints(self):
        """Return 4 of the 5 pentagon vertices as required checkpoints (excluding origin vertex)."""
        return [
            (4.76, 3.45),  # Vertex 1
            (2.94, 9.05),  # Vertex 2
            (-2.94, 9.05),  # Vertex 3
            (-4.76, 3.45),  # Vertex 4
        ]


# =============================================================================
# PATH REGISTRY
# =============================================================================

# Create all 9 path templates
ALL_PATHS = [
    # Zig-zags (enhanced)
    ZigZag1(),
    ZigZag2(),
    ZigZag3(),
    # Curves (enhanced)
    Curved1(),
    Curved2(),
    Curved3(),
    # Polygons (enhanced)
    Polygon1(),
    Polygon2(),
    Polygon3(),
]


def get_path_by_index(index: int) -> PathTemplate:
    """
    Get path template by index (0-8).

    Args:
        index: Path index (0-2 = zig-zag, 3-5 = curved, 6-8 = polygon)

    Returns:
        PathTemplate instance
    """
    if not 0 <= index < len(ALL_PATHS):
        raise ValueError(f"Path index must be 0-{len(ALL_PATHS) - 1}, got {index}")
    return ALL_PATHS[index]


def get_path_by_name(name: str) -> PathTemplate:
    """Get path template by name."""
    for path in ALL_PATHS:
        if path.name == name:
            return path
    raise ValueError(f"Path '{name}' not found")


def get_paths_by_type(path_type: str) -> list:
    """
    Get all paths of a specific type.

    Args:
        path_type: One of 'zig-zag', 'curved', 'polygon'

    Returns:
        List of PathTemplate instances
    """
    return [p for p in ALL_PATHS if p.path_type == path_type]