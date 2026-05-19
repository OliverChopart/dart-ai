"""Live dart detection pipeline.

Reads frames from VideoStream, runs YOLO detection, scores each dart tip
via homography, and emits ScoreEvents to a callback.

Calibration strategy
--------------------
On startup the pipeline shows a live camera feed but does NOT attempt
calibration automatically.  The user must press SPACE in the preview window
to trigger a calibration attempt.  This gives the user time to position
the camera correctly before calibration is locked in.

Once calibrated (Cal: OK shown in green) the homography is stored and
reused for all subsequent frames.  Pressing SPACE again re-calibrates.

macOS / OpenCV note
-------------------
cv2.imshow() must be called from the main thread on macOS.  The background
inference thread therefore stores the latest annotated frame in
self._latest_preview, and the caller (play_301.py) must call
pipeline.tick_preview() in its main loop to actually display the window
and handle keyboard events.

Usage::

    pipeline = DartPipeline(on_score_callback=on_score)
    pipeline.start()
    while running:
        pipeline.tick_preview()   # call this from the main thread
    pipeline.stop()
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from config.settings import settings
from utils.logging import get_logger
from vision.calibration import (
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
    homography_source: str = "unknown"
    frame: np.ndarray | None = None


class DartPipeline:
    """Ties together VideoStream -> YOLO -> Homography -> Score."""

    DEBOUNCE_FRAMES: int = 8
    FIFO_SIZE: int = 5
    FIFO_MIN_HITS: int = 3
    FIFO_TOLERANCE: float = 15.0

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

        # Calibration trigger — set from main thread, consumed in bg thread
        self._calibration_requested: bool = False

        # Latest frame for display — written by bg thread, read by main thread
        self._latest_preview: np.ndarray | None = None
        self._preview_lock = threading.Lock()

        # FIFO queue
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
        """Load models and start the background inference thread."""
        logger.info("loading models...")
        self._detector.load()
        self._scorer.load()
        self._stream.start()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("pipeline started — press SPACE in preview window to calibrate")

    def stop(self) -> None:
        """Stop the pipeline and release resources."""
        self._running = False
        self._stream.stop()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._show_preview:
            cv2.destroyAllWindows()
        logger.info("pipeline stopped")

    def tick_preview(self) -> bool:
        """Display the latest frame and handle keyboard input.

        Must be called from the MAIN THREAD on every iteration of the
        caller's loop.  Returns False if the user pressed Q (quit).
        """
        if not self._show_preview:
            return True

        with self._preview_lock:
            frame = self._latest_preview

        if frame is not None:
            cv2.imshow("Dart-AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            self._calibration_requested = True
            logger.info("calibration requested by user")
        elif key == ord("q"):
            return False

        return True

    def reset_dart_count(self) -> None:
        """Call this when darts are removed from the board (new turn)."""
        self._last_dart_count = 0
        self._debounce_counter = 0
        self._pending_count = 0
        self._dart_fifo.clear()
        logger.info("dart count reset")

    @property
    def has_homography(self) -> bool:
        return self._homography is not None

    @property
    def homography_source(self) -> str:
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

            detection = self._detector.detect(frame, annotate=self._show_preview)

            # Handle calibration request
            if self._calibration_requested:
                self._calibration_requested = False
                if detection.has_full_calibration:
                    H_new = compute_homography_from_detections(
                        detection.cal_points,
                        output_size=settings.homography_output_size,
                    )
                    if H_new is not None:
                        self._homography = H_new
                        self._homography_source = "yolo"
                        logger.info("calibration successful")
                        print("\n✅ Kalibrering lykkedes! Tryk ENTER for at starte spillet.")
                    else:
                        print("\n❌ Kalibrering fejlede — prøv igen med SPACE.")
                else:
                    missing = [s for s in (20, 6, 3, 11) if s not in detection.cal_points]
                    print(
                        f"\n❌ Kalibrering fejlede — {len(detection.cal_points)}/4 punkter fundet."
                        f"\n   Mangler: cal_{'  cal_'.join(str(s) for s in missing)}"
                        f"\n   Sørg for at hele dartskiven er synlig og tryk SPACE igen."
                    )

            # Build preview frame (written here, displayed in main thread)
            if self._show_preview:
                preview = (
                    detection.annotated_frame.copy()
                    if detection.annotated_frame is not None
                    else frame.copy()
                )

                cal_count = len(detection.cal_points)
                if self._homography is not None:
                    cal_text = f"Cal: OK [{self._homography_source}]"
                    cal_color = (0, 255, 0)
                else:
                    cal_text = f"Cal: {cal_count}/4 — tryk SPACE for at kalibrere"
                    cal_color = (0, 165, 255) if cal_count > 0 else (0, 0, 255)

                dart_count_display = len(detection.dart_tips)
                cv2.putText(
                    preview, f"Pile: {dart_count_display}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                )
                cv2.putText(
                    preview, cal_text,
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cal_color, 2,
                )
                instruction = "SPACE = kalibrer  |  Q = afslut"
                cv2.putText(
                    preview, instruction,
                    (10, preview.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                )

                with self._preview_lock:
                    self._latest_preview = preview

            # Skip scoring until calibrated
            if self._homography is None:
                continue

            # FIFO debounce
            stable_tips = self._fifo_filter(detection.dart_tips)
            stable_confs = [1.0] * len(stable_tips)

            results = self._scorer.score_detections_with_homography(
                stable_tips, self._homography, stable_confs,
            )

            dart_count = len(results)

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

    # ------------------------------------------------------------------
    # FIFO debounce filter
    # ------------------------------------------------------------------

    def _fifo_filter(self, tips: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Return only dart tips stable across the FIFO window."""
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
