"""Live dart detection pipeline.

Reads frames from VideoStream, runs YOLO detection, scores each dart tip
via homography, and emits ScoreEvents to a callback.

Calibration strategy
--------------------
With the 5-class YOLO model the pipeline recomputes the homography on every
frame where all four calibration points (cal_20, cal_6, cal_3, cal_11) are
detected.  If only a subset is visible the last known homography is reused.
If no homography has ever been computed the pipeline falls back to a matrix
loaded from disk (config/homography.npy) produced by run_calibration.py.

FIFO debouncing
---------------
Dart tips are only emitted as score events when the same tip location appears
in at least FIFO_MIN_HITS of the last FIFO_SIZE frames (position-matched
within FIFO_TOLERANCE pixels).  This eliminates single-frame false positives
and stabilises detection across lighting fluctuations.

Usage::

    def on_score(event):
        print(event)

    pipeline = DartPipeline(on_score_callback=on_score)
    pipeline.start()
    # ... play darts ...
    pipeline.stop()
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from config.settings import settings
from utils.logging import get_logger
from vision.calibration import (
    DEFAULT_HOMOGRAPHY_PATH,
    DEFAULT_OUTPUT_SIZE,
    compute_homography_from_detections,
)
from vision.detector import DartDetector
from vision.scorer import DartScorer, ScoreResult
from vision.stream import VideoStream

logger = get_logger(__name__)


@dataclass
class ScoreEvent:
    """Emitted when a new dart is detected and scored."""

    results: list[ScoreResult]
    dart_count: int
    homography_source: str = "unknown"  # 'yolo' | 'disk' | 'none'
    frame: np.ndarray | None = None


class DartPipeline:
    """Ties together VideoStream -> YOLO -> Homography -> Score.

    Runs inference in a background thread.  Calls on_score_callback whenever
    the number of stably-detected darts increases (new dart thrown).
    """

    # Debounce: how many consecutive frames must show the same dart count
    # before emitting a score event
    DEBOUNCE_FRAMES: int = 8

    # FIFO: sliding window size and minimum hit count for dart tip stability
    FIFO_SIZE: int = 5
    FIFO_MIN_HITS: int = 3
    FIFO_TOLERANCE: float = 15.0  # pixels — max distance between matched tips

    def __init__(
        self,
        on_score_callback: Callable[[ScoreEvent], None],
        model_path: str | None = None,
        show_preview: bool = True,
    ) -> None:
        self._callback = on_score_callback
        self._show_preview = show_preview
        self._running = False
        self._thread: threading.Thread | None = None

        self._stream = VideoStream()
        self._detector = DartDetector(model_path=model_path)
        self._scorer = DartScorer()

        # Homography state
        self._homography: np.ndarray | None = None
        self._homography_source: str = "none"

        # FIFO queue: each entry is the list of dart tips detected in one frame
        self._dart_fifo: deque[list[tuple[float, float]]] = deque(maxlen=self.FIFO_SIZE)

        # Debounce state
        self._last_dart_count: int = 0
        self._debounce_counter: int = 0
        self._pending_count: int = 0
        self._pending_results: list[ScoreResult] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load models and start the pipeline thread."""
        logger.info("loading models...")
        self._detector.load()
        self._scorer.load()  # soft load — does not crash if homography.npy absent

        # Use saved static homography as initial fallback
        if self._scorer.is_ready:
            self._homography = np.load(str(DEFAULT_HOMOGRAPHY_PATH))
            self._homography_source = "disk"
            logger.info("initial homography loaded from disk")
        else:
            logger.info("no saved homography — will calibrate automatically from YOLO")

        self._stream.start()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("pipeline started")

    def stop(self) -> None:
        """Stop the pipeline and release resources."""
        self._running = False
        self._stream.stop()
        if self._thread:
            self._thread.join(timeout=3.0)
        cv2.destroyAllWindows()
        logger.info("pipeline stopped")

    def reset_dart_count(self) -> None:
        """Call this when darts are removed from the board (new turn)."""
        self._last_dart_count = 0
        self._debounce_counter = 0
        self._pending_count = 0
        self._dart_fifo.clear()
        logger.info("dart count reset")

    @property
    def has_homography(self) -> bool:
        """True if a homography matrix is available (from YOLO or disk)."""
        return self._homography is not None

    @property
    def homography_source(self) -> str:
        """Where the current homography came from: 'yolo', 'disk', or 'none'."""
        return self._homography_source

    # ------------------------------------------------------------------
    # Background inference loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main inference loop — runs in a daemon thread."""
        while self._running:
            frame = self._stream.read(timeout=0.5)
            if frame is None:
                continue

            # 1. Run YOLO detection (darts + calibration points)
            detection = self._detector.detect(frame, annotate=self._show_preview)

            # 2. Update homography from YOLO calibration points when available
            if detection.has_full_calibration:
                H_new = compute_homography_from_detections(
                    detection.cal_points,
                    output_size=settings.homography_output_size,
                )
                if H_new is not None:
                    self._homography = H_new
                    self._homography_source = "yolo"

            if self._homography is None:
                # No homography available yet — keep buffering frames
                logger.debug("waiting for calibration points...")
                continue

            # 3. FIFO debounce: only keep tips stable across multiple frames
            stable_tips = self._fifo_filter(detection.dart_tips)
            stable_confs = [1.0] * len(stable_tips)  # FIFO doesn't preserve conf

            # 4. Score stable dart tips
            results = self._scorer.score_detections_with_homography(
                stable_tips,
                self._homography,
                stable_confs,
            )

            dart_count = len(results)

            # 5. Emit score event when dart count increases and is stable
            if dart_count != self._last_dart_count:
                if dart_count == self._pending_count:
                    self._debounce_counter += 1
                else:
                    self._pending_count = dart_count
                    self._pending_results = results
                    self._debounce_counter = 1

                if self._debounce_counter >= self.DEBOUNCE_FRAMES:
                    if dart_count > self._last_dart_count:
                        event = ScoreEvent(
                            results=self._pending_results,
                            dart_count=dart_count,
                            homography_source=self._homography_source,
                            frame=frame.copy() if self._show_preview else None,
                        )
                        self._callback(event)
                    self._last_dart_count = dart_count
                    self._debounce_counter = 0

            # 6. Show preview with calibration status overlay
            if self._show_preview and detection.annotated_frame is not None:
                preview = detection.annotated_frame.copy()
                cal_status = (
                    f"Cal: {len(detection.cal_points)}/4 "
                    f"[{self._homography_source}]"
                )
                color = (0, 255, 0) if detection.has_full_calibration else (0, 165, 255)
                cv2.putText(
                    preview, f"Pile: {dart_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                )
                cv2.putText(
                    preview, cal_status,
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )
                cv2.imshow("Dart-AI", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False

    # ------------------------------------------------------------------
    # FIFO debounce filter
    # ------------------------------------------------------------------

    def _fifo_filter(
        self,
        tips: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Return only dart tips that appear stably across the FIFO window.

        A tip is considered stable if a matching tip (within FIFO_TOLERANCE
        pixels) was present in at least FIFO_MIN_HITS of the last FIFO_SIZE
        frames.  This eliminates single-frame noise and false positives.

        Args:
            tips: Dart tips detected in the current frame.

        Returns:
            Filtered list of stable dart tip coordinates.
        """
        self._dart_fifo.append(tips)

        if len(self._dart_fifo) < self.FIFO_MIN_HITS:
            return []

        stable: list[tuple[float, float]] = []
        for tip in tips:
            hit_count = sum(
                any(
                    math.hypot(tip[0] - t[0], tip[1] - t[1]) < self.FIFO_TOLERANCE
                    for t in frame_tips
                )
                for frame_tips in self._dart_fifo
            )
            if hit_count >= self.FIFO_MIN_HITS:
                stable.append(tip)

        return stable
