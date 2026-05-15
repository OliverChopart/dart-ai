"""Geometry helpers for dart score calculation."""

import math

# Dartboard segment order clockwise from top
SEGMENTS: list[int] = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Normalised radii (relative to board radius = 1.0)
RADIUS_BULLSEYE = 0.05
RADIUS_BULL = 0.10
RADIUS_TRIPLE_INNER = 0.60
RADIUS_TRIPLE_OUTER = 0.68
RADIUS_DOUBLE_INNER = 0.95
RADIUS_DOUBLE_OUTER = 1.00


def polar_to_segment(x: float, y: float) -> tuple[int, str]:
    """Convert normalised board coordinates to dart score.

    Args:
        x: Normalised x coordinate (centre = 0, board edge = +/-1).
        y: Normalised y coordinate (centre = 0, board edge = +/-1).

    Returns:
        Tuple of (score, segment_label) e.g. (60, 'T20') or (25, 'Bull').
    """
    radius = math.sqrt(x**2 + y**2)

    if radius <= RADIUS_BULLSEYE:
        return 50, "Bullseye"
    if radius <= RADIUS_BULL:
        return 25, "Bull"
    if radius > RADIUS_DOUBLE_OUTER:
        return 0, "Miss"

    # Angle in degrees, 0 = top (12 o'clock), clockwise
    angle_rad = math.atan2(x, -y)
    angle_deg = math.degrees(angle_rad) % 360

    # Each segment spans 18 degrees
    segment_index = int((angle_deg + 9) / 18) % 20
    base_score = SEGMENTS[segment_index]

    if RADIUS_TRIPLE_INNER <= radius <= RADIUS_TRIPLE_OUTER:
        return base_score * 3, f"T{base_score}"
    if RADIUS_DOUBLE_INNER <= radius <= RADIUS_DOUBLE_OUTER:
        return base_score * 2, f"D{base_score}"

    return base_score, str(base_score)
