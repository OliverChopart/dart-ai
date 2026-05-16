"""Live dart detection pipeline.

Reads frames from VideoStream, runs YOLO detection, scores each dart tip
via homography, and emits ScoreEvents to a callback.

Usage::

    def on_score(event):
        print(event)

    pipeline = DartPipeline(on_score_callback=on_score)
    pipeline.start()
    # ... play darts ...
    pipeline.stop()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from config.settings import settings
from utils.logging import get_logger
from vision.calibration import apply_homography, load_homography, DEFAULT_OUTPUT_SIZE
from vision.detector import DartDetector
from vision.scorer import DartScorer, ScoreResult
from vision.stream import VideoStream

logger = get_logger(__name__)


@dataclass
class ScoreEvent:
    """Emitted when a new dart is detected and scored."""
    results: list[ScoreResult]
    dart_count: int
    frame: np.ndarray | None = None


class DartPipeline:
    """Ties together VideoStream → YOLO → Homography → Score.

    Runs inference in a background thread. Calls on_score_callback
    whenever the number of detected darts changes (new dart thrown).

    Args:
        on_score_callback: Called with a ScoreEvent when darts change.
        model_path: Path to trained YOLO model.
        show_preview: If True, shows live OpenCV preview window.
    """

    # How many consecutive frames must agree on dart count before
    # we emit a score event (debounce for stability)
    DEBOUNCE_FRAMES = 8

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

        # State tracking
        self._last_dart_count = 0
        self._debounce_counter = 0
        self._pending_count = 0
        self._pending_results: list[ScoreResult] = []

    def start(self) -> None:
        """Load models and start the pipeline thread."""
        logger.info("loading models...")
        self._detector.load()
        self._scorer.load()
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
        logger.info("dart count reset")

    def _run(self) -> None:
        """Main inference loop."""
        while self._running:
            frame = self._stream.read(timeout=0.5)
            if frame is None:
                continue

            # Run detection
            detection = self._detector.detect(frame, annotate=self._show_preview)

            # Score detected tips
            results = self._scorer.score_detections(
                detection.dart_tips,
                detection.confidences,
            )

            dart_count = len(results)

            # Debounce: require DEBOUNCE_FRAMES consecutive frames
            # with the same dart count before emitting
            if dart_count != self._last_dart_count:
                if dart_count == self._pending_count:
                    self._debounce_counter += 1
                else:
                    self._pending_count = dart_count
                    self._pending_results = results
                    self._debounce_counter = 1

                if self._debounce_counter >= self.DEBOUNCE_FRAMES:
                    if dart_count > self._last_dart_count:
                        # New dart(s) detected
                        event = ScoreEvent(
                            results=self._pending_results,
                            dart_count=dart_count,
                            frame=frame.copy() if self._show_preview else None,
                        )
                        self._callback(event)
                    self._last_dart_count = dart_count
                    self._debounce_counter = 0

            # Show preview
            if self._show_preview and detection.annotated_frame is not None:
                # Overlay dart count
                preview = detection.annotated_frame.copy()
                cv2.putText(
                    preview,
                    f"Pile: {dart_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Dart-AI", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False