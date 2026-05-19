"""Live dart detection pipeline.

Reads frames from VideoStream, runs YOLO detection, scores each dart tip
via homography, and emits ScoreEvents to a callback.

Calibration strategy
--------------------
On startup the pipeline shows a live camera feed but does NOT attempt
calibration automatically.  The user must press SPACE in the preview window
to trigger a calibration attempt.

Calibration requires minimum 1 cal point — the user can force calibration
at any time by pressing SPACE, even with partial detection.  More points
= more accurate homography.  4/4 is ideal, 3/4 is good, 1-2/4 is rough.

macOS / OpenCV note
-------------------
cv2.imshow() must be called from the main thread on macOS.  The background
inference thread stores the latest annotated frame in self._latest_preview,
and the caller must call pipeline.tick_preview() in its main loop.
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

MIN_CAL_POINTS = 1


@dataclass
class ScoreEvent:
    """Emitted when a new dart is detected and scored."""

    results: list[ScoreResult]
    dart_count: int
    homography_source: str = "unknown"
    frame: np.ndarray | None = None


@dataclass
class ScoreOverlay:
    """Score info passed from GameSession to pipeline for display in preview."""

    player_name: str = ""
    score_remaining: int = 0      # total score remaining e.g. 245
    hand_scores: list[str] = None  # scores this hand e.g. ["T20", "5", "1"]
    hand_total: int = 0            # sum this hand e.g. 66

    def __post_init__(self):
        if self.hand_scores is None:
            self.hand_scores = []


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

        # Calibration trigger
        self._calibration_requested: bool = False

        # Score overlay — updated by GameSession after each throw
        self._score_overlay: ScoreOverlay = ScoreOverlay()
        self._overlay_lock = threading.Lock()

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

        Must be called from the MAIN THREAD. Returns False if user pressed Q.
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
        elif key == ord("q"):
            return False

        return True

    def update_score_overlay(self, overlay: ScoreOverlay) -> None:
        """Update the score info shown in the preview. Thread-safe."""
        with self._overlay_lock:
            self._score_overlay = overlay

    def reset_dart_count(self) -> None:
        """Call this when darts are removed from the board (new turn)."""
        self._last_dart_count = 0
        self._debounce_counter = 0
        self._pending_count = 0
        self._dart_fifo.clear()

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
                cal_count = len(detection.cal_points)

                if cal_count >= MIN_CAL_POINTS:
                    H_new = compute_homography_from_detections(
                        detection.cal_points,
                        output_size=settings.homography_output_size,
                    )
                    if H_new is not None:
                        self._homography = H_new
                        self._homography_source = "yolo"
                        print(
                            f"\n✅ Kalibrering lykkedes! ({cal_count}/4 punkter fundet)"
                            + ("" if cal_count == 4 else "\n   ⚠️  Delvis kalibrering — scorer kan være unøjagtige")
                        )
                    else:
                        print("\n❌ Homografi-beregning fejlede — prøv igen med SPACE.")
                else:
                    print(
                        "\n❌ Ingen kalibreringspunkter fundet."
                        "\n   Sørg for at dartskiven er synlig og prøv igen med SPACE."
                    )

            # Build preview frame
            if self._show_preview:
                preview = (
                    detection.annotated_frame.copy()
                    if detection.annotated_frame is not None
                    else frame.copy()
                )

                h, w = preview.shape[:2]

                # --- Calibration status (top left) ---
                cal_count = len(detection.cal_points)
                if self._homography is not None:
                    cal_text = f"Cal: OK [{self._homography_source}]"
                    cal_color = (0, 255, 0)
                else:
                    cal_text = f"Cal: {cal_count}/4 — SPACE for at kalibrere"
                    cal_color = (0, 165, 255) if cal_count > 0 else (0, 0, 255)

                cv2.putText(preview, cal_text, (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, cal_color, 2)

                # --- Score overlay (top right) ---
                with self._overlay_lock:
                    overlay = self._score_overlay

                if overlay.player_name:
                    # Player name
                    name_text = overlay.player_name
                    (tw, _), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(preview, name_text, (w - tw - 10, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Score remaining
                    score_text = f"{overlay.score_remaining} tilbage"
                    (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                    cv2.putText(preview, score_text, (w - tw - 10, 72),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

                    # This hand
                    if overlay.hand_scores:
                        hand_text = "  +  ".join(overlay.hand_scores)
                        if overlay.hand_total > 0:
                            hand_text += f"  =  {overlay.hand_total}"
                        (tw, _), _ = cv2.getTextSize(hand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.putText(preview, hand_text, (w - tw - 10, 108),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 100), 2)

                # --- Bottom instruction ---
                cv2.putText(preview, "SPACE = kalibrer  |  Q = afslut",
                            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

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
