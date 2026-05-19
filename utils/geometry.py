"""Geometry helpers for dart score calculation."""

import math

# ---------------------------------------------------------------------------
# Scoring mode
# ---------------------------------------------------------------------------
# "4-zone"     – divide the board into 4 quadrants (Top/Right/Bottom/Left).
#                Only scores 20, 6, 3, 11 (or Bull/Miss).
# "20-segment" – standard dartboard with 20 precise segments.
SCORING_MODE: str = "4-zone"

# ---------------------------------------------------------------------------
# Dartboard segment order clockwise from top (used in 20-segment mode)
# ---------------------------------------------------------------------------
SEGMENTS: list[int] = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# ---------------------------------------------------------------------------
# 4-zone quadrant mapping
# ---------------------------------------------------------------------------
# Angle 0° = top (12 o'clock), increasing clockwise.
#   Top    315° – 45°  → segment 20
#   Right   45° – 135° → segment 6  (where YOLO detects cal_11)
#   Bottom 135° – 225° → segment 3  (where YOLO detects cal_6)
#   Left   225° – 315° → segment 11 (where YOLO detects cal_3)
_ZONE_SEGMENTS: list[tuple[float, float, int]] = [
    (315.0, 45.0,  20),
    (45.0,  135.0,  6),
    (135.0, 225.0,  3),
    (225.0, 315.0, 11),
]

# ---------------------------------------------------------------------------
# Normalised radii (relative to board radius = 1.0)
# ---------------------------------------------------------------------------
RADIUS_BULLSEYE = 0.05
RADIUS_BULL = 0.10
RADIUS_DOUBLE_OUTER = 1.02  # slight margin to absorb float rounding at board edge


def _angle_to_zone_segment(angle_deg: float) -> int:
    """Return the segment number for the 4-zone quadrant containing angle_deg."""
    for start, end, segment in _ZONE_SEGMENTS:
        if start > end:
            # Wraps around 0° (Top zone: 315–360 and 0–45)
            if angle_deg >= start or angle_deg < end:
                return segment
        else:
            if start <= angle_deg < end:
                return segment
    return 20  # fallback


def polar_to_segment(x: float, y: float) -> tuple[int, str]:
    """Convert normalised board coordinates to a dart score.

    In "4-zone" mode (default): only returns 20, 6, 3, 11, Bull, Bullseye or Miss.
    No double or triple multipliers — just the plain segment number.

    In "20-segment" mode: standard 20-segment dartboard, also no multipliers.

    Args:
        x: Normalised x coordinate (centre = 0, board edge = +/-1).
        y: Normalised y coordinate (centre = 0, board edge = +/-1).

    Returns:
        Tuple of (score, label) e.g. (20, "20"), (50, "Bullseye"), (0, "Miss").
    """
    radius = math.sqrt(x**2 + y**2)

    if radius <= RADIUS_BULLSEYE:
        return 50, "Bullseye"
    if radius <= RADIUS_BULL:
        return 25, "Bull"
    if radius > RADIUS_DOUBLE_OUTER:
        return 0, "Miss"

    # Angle: 0° = top (12 o'clock), clockwise
    angle_rad = math.atan2(x, -y)
    angle_deg = math.degrees(angle_rad) % 360

    if SCORING_MODE == "4-zone":
        base_score = _angle_to_zone_segment(angle_deg)
    else:
        segment_index = int((angle_deg + 9) / 18) % 20
        base_score = SEGMENTS[segment_index]

    return base_score, str(base_score)
