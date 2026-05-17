"""Dart score calculator.

Converts YOLO dart tip detections to dart scores using either a dynamic
homography (computed per-frame from calibration points) or a static
homography loaded from disk.

Pipeline::

    YOLO pixel (x, y)
        -> perspectiveTransform(H)  # perspective correction
        -> normalise to [-1, 1]     # board centre = (0, 0), edge = +-1
        -> polar_to_segment()       # geometry -> score
        -> ScoreResult

Preferred usage with 5-class model (dynamic homography)::

    scorer = DartScorer()
    scorer.load()  # loads saved H if available, does not crash if missing
    results = scorer.score_detections_with_homography(tips, H)

Fallback usage with manual calibration (static homography)::

    scorer = DartScorer()
    scorer.load()  # requires config/homography.npy to exist
    result = scorer.score_pixel(x=462, y=118)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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

    Supports two modes:
    - Dynamic homography: pass H directly to score_detections_with_homography().
      This is the preferred mode when using the 5-class YOLO model, as H is
      recomputed on every frame from the detected calibration points.
    - Static homography: call load() to read H from disk, then use
      score_pixel() / score_detections(). Used as a fallback when a trained
      5-class model is not yet available.
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
        """Attempt to load a saved homography matrix from disk.

        Unlike the previous version this method does NOT raise FileNotFoundError
        when the file is absent.  The pipeline will simply start without a
        static H and rely on the per-frame homography from YOLO cal points.
        """
        if self._homography_path.exists():
            try:
                self._H = load_homography(self._homography_path)
                logger.info("scorer ready with saved homography", path=str(self._homography_path))
            except Exception as exc:
                logger.warning("failed to load homography from disk", error=str(exc))
        else:
            logger.info(
                "no saved homography found — awaiting dynamic calibration from YOLO",
                path=str(self._homography_path),
            )
        return self

    @property
    def is_ready(self) -> bool:
        """True if a static homography has been loaded from disk."""
        return self._H is not None

    # ------------------------------------------------------------------
    # Dynamic homography (preferred with 5-class model)
    # ------------------------------------------------------------------

    def score_detections_with_homography(
        self,
        tips: list[tuple[float, float]],
        H: np.ndarray,
        confidences: list[float] | None = None,
    ) -> list[ScoreResult]:
        """Score dart tips using a dynamically computed homography matrix.

        This is the preferred method when using the 5-class YOLO model.
        H is recomputed each frame by compute_homography_from_detections()
        in the pipeline, so the scoring adapts to any camera angle without
        manual recalibration.

        Args:
            tips: List of (x, y) pixel coordinates from dart detections.
            H: 3x3 homography matrix computed from the current frame's
               calibration points.
            confidences: Optional confidence scores, one per tip.

        Returns:
            List of ScoreResult, one per dart tip.
        """
        if confidences is None:
            confidences = [1.0] * len(tips)

        half = self._output_size / 2.0
        results: list[ScoreResult] = []

        for (px, py), conf in zip(tips, confidences):
            bx, by = self._apply_homography(px, py, H)
            nx = (bx - half) / half
            ny = (by - half) / half
            score, segment = polar_to_segment(nx, ny)
            results.append(
                ScoreResult(
                    score=score,
                    segment=segment,
                    board_x=nx,
                    board_y=ny,
                    pixel_x=px,
                    pixel_y=py,
                    confidence=conf,
                )
            )
            logger.debug(
                "dart scored",
                segment=segment,
                score=score,
                board_x=round(nx, 3),
                board_y=round(ny, 3),
            )

        return results

    # ------------------------------------------------------------------
    # Static homography (fallback / manual calibration)
    # ------------------------------------------------------------------

    def score_pixel(
        self,
        x: float,
        y: float,
        confidence: float = 1.0,
    ) -> ScoreResult:
        """Score a single dart tip using the static (disk-loaded) homography.

        Args:
            x: Pixel x coordinate in the original (uncorrected) frame.
            y: Pixel y coordinate in the original (uncorrected) frame.
            confidence: YOLO detection confidence (0-1).

        Returns:
            ScoreResult with score, segment and coordinate details.

        Raises:
            RuntimeError: If no homography has been loaded via load().
        """
        if self._H is None:
            raise RuntimeError(
                "Static homography not loaded. Call scorer.load() first, or use "
                "score_detections_with_homography() with a dynamically computed H."
            )
        return self.score_detections_with_homography(
            [(x, y)], self._H, [confidence]
        )[0]

    def score_detections(
        self,
        tips: list[tuple[float, float]],
        confidences: list[float] | None = None,
    ) -> list[ScoreResult]:
        """Score a list of dart tips using the static (disk-loaded) homography.

        Args:
            tips: List of (x, y) pixel coordinates.
            confidences: Optional list of confidence scores, one per tip.

        Returns:
            List of ScoreResult, one per dart tip.

        Raises:
            RuntimeError: If no homography has been loaded via load().
        """
        if self._H is None:
            raise RuntimeError(
                "Static homography not loaded. Call scorer.load() first, or use "
                "score_detections_with_homography() with a dynamically computed H."
            )
        return self.score_detections_with_homography(tips, self._H, confidences)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_homography(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
        """Apply homography H to a single pixel coordinate."""
        pt = np.array([[[px, py]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, H)
        return float(warped[0, 0, 0]), float(warped[0, 0, 1])
