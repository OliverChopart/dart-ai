"""Dart score calculator.

Takes a dart tip pixel coordinate from YOLO, transforms it to the canonical
top-down board coordinate using the homography matrix, and returns the score.

Pipeline::

    YOLO pixel (x, y)
        -> apply_homography()       # perspective correction
        -> normalise to [-1, 1]     # board centre = (0, 0), edge = +-1
        -> polar_to_segment()       # geometry -> score
        -> ScoreResult

Usage::

    scorer = DartScorer()
    result = scorer.score_pixel(x=462, y=118)
    print(result.score, result.segment)  # e.g. 60  T20
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
from utils.geometry import polar_to_segment
from utils.logging import get_logger
from vision.calibration import (
    DEFAULT_HOMOGRAPHY_PATH,
    DEFAULT_OUTPUT_SIZE,
    load_homography,
)

logger = get_logger(__name__)


@dataclass
class ScoreResult:
    """Result of scoring a single dart tip."""

    score: int          # e.g. 60
    segment: str        # e.g. 'T20', 'D16', 'Bull', 'Bullseye', 'Miss'
    board_x: float      # normalised board coordinate [-1, 1]
    board_y: float
    pixel_x: float      # original pixel coordinate
    pixel_y: float
    confidence: float   # YOLO detection confidence

    def __str__(self) -> str:
        return (
            f"{self.segment} ({self.score} pts) "
            f"| board=({self.board_x:.3f}, {self.board_y:.3f}) "
            f"| pixel=({self.pixel_x:.0f}, {self.pixel_y:.0f}) "
            f"| conf={self.confidence:.2f}"
        )


class DartScorer:
    """Converts YOLO dart tip detections to dart scores.

    Loads the homography matrix computed by the calibration tool and uses it
    to transform raw pixel coordinates into normalised board coordinates before
    applying polar geometry to determine the segment.
    """

    def __init__(
        self,
        homography_path: Path | str = DEFAULT_HOMOGRAPHY_PATH,
        output_size: int = DEFAULT_OUTPUT_SIZE,
    ) -> None:
        self._output_size = output_size
        self._H: np.ndarray | None = None
        self._homography_path = Path(homography_path)

    def load(self) -> "DartScorer":
        """Load the homography matrix from disk."""
        self._H = load_homography(self._homography_path)
        logger.info("scorer ready", homography=str(self._homography_path))
        return self

    @property
    def is_ready(self) -> bool:
        """True if the homography has been loaded."""
        return self._H is not None

    def score_pixel(
        self,
        x: float,
        y: float,
        confidence: float = 1.0,
    ) -> ScoreResult:
        """Score a single dart tip given its pixel coordinate.

        Args:
            x: Pixel x coordinate in the original (uncorrected) frame.
            y: Pixel y coordinate in the original (uncorrected) frame.
            confidence: YOLO detection confidence (0-1).

        Returns:
            ScoreResult with score, segment and coordinate details.
        """
        if self._H is None:
            raise RuntimeError("Homography not loaded. Call scorer.load() first.")

        # Transform pixel to corrected board coordinate
        bx, by = self._pixel_to_board(x, y)

        # Normalise: board centre = (0,0), board radius = 1.0
        half = self._output_size / 2
        nx = (bx - half) / half
        ny = (by - half) / half

        score, segment = polar_to_segment(nx, ny)

        logger.debug(
            "dart scored",
            segment=segment,
            score=score,
            board_x=round(nx, 3),
            board_y=round(ny, 3),
            confidence=round(confidence, 2),
        )

        return ScoreResult(
            score=score,
            segment=segment,
            board_x=nx,
            board_y=ny,
            pixel_x=x,
            pixel_y=y,
            confidence=confidence,
        )

    def score_detections(
        self,
        tips: list[tuple[float, float]],
        confidences: list[float] | None = None,
    ) -> list[ScoreResult]:
        """Score a list of dart tip detections from a single frame.

        Args:
            tips: List of (x, y) pixel coordinates.
            confidences: Optional list of confidence scores, one per tip.

        Returns:
            List of ScoreResult, one per dart tip.
        """
        if confidences is None:
            confidences = [1.0] * len(tips)

        return [
            self.score_pixel(x, y, conf)
            for (x, y), conf in zip(tips, confidences)
        ]

    def _pixel_to_board(
        self, px: float, py: float
    ) -> tuple[float, float]:
        """Apply homography to a single pixel coordinate."""
        pt = np.array([[[px, py]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self._H)
        return float(warped[0, 0, 0]), float(warped[0, 0, 1])
