"""Live dart detection pipeline.

Reads frames from VideoStream, runs YOLO detection, and emits ScoreEvents.

Scoring flow
------------
Scoring is MANUAL and per-dart:

1. Kast en pil
2. Tryk SPACE → snapshot: scorer kun den NYE pil (ikke dem der allerede var der)
3. Kast næste pil, tryk SPACE igen → scorer kun den seneste pil
4. Tryk ENTER i terminalen → fjern pile, ny tur, hukommelse nulstilles

Teknisk: pipelinen husker pixel-positionerne på alle pile der er scoret denne tur.
Ved hvert SPACE-tryk scores kun pile der IKKE matcher en tidligere scoret position.

Kalibrering:
  SPACE (før kalibrering) → kalibrer
  C (når som helst)       → rekalibrér

macOS note: cv2.imshow() skal kaldes fra main thread — tick_preview() håndterer dette.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from config.settings import settings
from utils.logging import get_logger
from vision.calibration import (
    DEFAULT_OUTPUT_SIZE,
    MIN_HOMOGRAPHY_POINTS,
    compute_homography_from_detections,
)
from vision.detector import DartDetector
from vision.scorer import DartScorer, ScoreResult
from vision.stream import VideoStream

logger = get_logger(__name__)


@dataclass
class ScoreEvent:
    """Emitted when user presses SPACE and a new dart is detected."""

    results: list[ScoreResult]       # kun de NYE pile siden forrige snapshot
    dart_count: int                  # antal pile på skiven i alt
    homography_source: str = "unknown"
    frame: np.ndarray | None = None


@dataclass
class ScoreOverlay:
    """Score info til display i preview-vinduet."""

    player_name: str = ""
    score_remaining: int = 0
    hand_scores: list[str] = None
    hand_total: int = 0

    def __post_init__(self):
        if self.hand_scores is None:
            self.hand_scores = []


class DartPipeline:
    """VideoStream -> YOLO -> manuel snapshot-scoring per pil."""

    # Hvor langt en pil må have flyttet sig og stadig regnes som "samme pil" (pixels)
    MATCH_TOLERANCE: float = 60.0

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

        self._homography: np.ndarray | None = None
        self._homography_source: str = "none"

        self._calibration_requested: bool = False
        self._snapshot_requested: bool = False

        # Hukommelse: pixel-positioner på pile der allerede er scoret denne tur
        self._scored_tips: list[tuple[float, float]] = []

        self._latest_preview: np.ndarray | None = None
        self._preview_lock = threading.Lock()

        self._score_overlay: ScoreOverlay = ScoreOverlay()
        self._overlay_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        logger.info("loading models...")
        self._detector.load()
        self._scorer.load()
        self._stream.start()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("pipeline started")

    def stop(self) -> None:
        self._running = False
        self._stream.stop()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._show_preview:
            cv2.destroyAllWindows()
        logger.info("pipeline stopped")

    def tick_preview(self) -> bool:
        """Vis seneste frame og håndtér tastatur. SKAL kaldes fra main thread.
        Returnerer False hvis brugeren trykker Q.
        """
        if not self._show_preview:
            return True

        with self._preview_lock:
            frame = self._latest_preview

        if frame is not None:
            cv2.imshow("Dart-AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if self._homography is None:
                self._calibration_requested = True
            else:
                self._snapshot_requested = True
        elif key == ord("c"):
            self._calibration_requested = True
        elif key == ord("q"):
            return False

        return True

    def reset_dart_count(self) -> None:
        """Kald når pile fjernes fra skiven (ny tur). Nulstiller hukommelsen."""
        self._scored_tips = []

    def update_score_overlay(self, overlay: ScoreOverlay) -> None:
        with self._overlay_lock:
            self._score_overlay = overlay

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
        while self._running:
            frame = self._stream.read(timeout=0.5)
            if frame is None:
                continue

            detection = self._detector.detect(frame, annotate=self._show_preview)

            # --- Kalibrering ---
            if self._calibration_requested:
                self._calibration_requested = False
                cal_count = len(detection.cal_points)

                if cal_count < MIN_HOMOGRAPHY_POINTS:
                    print(
                        f"\n❌ Kalibrering fejlede — {cal_count}/4 punkter fundet."
                        f"\n   Kræver 4 punkter. Sørg for at hele skiven er synlig og prøv igen."
                    )
                else:
                    H_new = compute_homography_from_detections(
                        detection.cal_points,
                        output_size=settings.homography_output_size,
                    )
                    if H_new is not None:
                        self._homography = H_new
                        self._homography_source = "yolo"
                        self._scored_tips = []
                        print("\n✅ Kalibrering lykkedes! Kast en pil og tryk SPACE.")
                    else:
                        print("\n❌ Homografi-beregning fejlede — prøv igen.")

            # --- Snapshot scoring ---
            if self._snapshot_requested and self._homography is not None:
                self._snapshot_requested = False
                self._score_snapshot(frame, detection)

            # --- Preview ---
            if self._show_preview:
                self._build_preview(frame, detection)

    def _score_snapshot(self, frame: np.ndarray, detection) -> None:
        """Score kun pile der ikke allerede er scoret denne tur."""
        if not detection.dart_tips:
            print("\n⚠️  Ingen pile fundet — er pilen landet på skiven?")
            return

        # Score alle synlige pile
        all_results = self._scorer.score_detections_with_homography(
            detection.dart_tips,
            self._homography,
            [1.0] * len(detection.dart_tips),
        )

        print(f"\n🔍 Snapshot: {len(detection.dart_tips)} pile fundet i alt, {len(self._scored_tips)} allerede scoret")

        # Filtrer: behold kun pile der IKKE matcher en allerede scoret position
        new_results: list[ScoreResult] = []
        for result in all_results:
            tip = (result.pixel_x, result.pixel_y)
            distances = [
                math.hypot(tip[0] - prev[0], tip[1] - prev[1])
                for prev in self._scored_tips
            ]
            min_dist = min(distances) if distances else float("inf")
            already_scored = min_dist < self.MATCH_TOLERANCE

            if already_scored:
                print(f"   ↩ Springer over {result.segment} — matcher tidligere pil (dist={min_dist:.0f}px)")
            else:
                new_results.append(result)
                self._scored_tips.append(tip)
                print(f"   ✅ Ny pil: {result.segment} ({result.score}p) på pixel ({tip[0]:.0f}, {tip[1]:.0f})")

        if not new_results:
            print("⚠️  Ingen nye pile siden forrige snapshot.")
            return

        event = ScoreEvent(
            results=new_results,
            dart_count=len(detection.dart_tips),
            homography_source=self._homography_source,
            frame=frame.copy() if self._show_preview else None,
        )
        self._callback(event)

    def _build_preview(self, frame: np.ndarray, detection) -> None:
        """Byg annoteret preview-frame til display."""
        preview = (
            detection.annotated_frame.copy()
            if detection.annotated_frame is not None
            else frame.copy()
        )

        h, w = preview.shape[:2]

        cal_count = len(detection.cal_points)
        if self._homography is not None:
            cal_text = f"Cal: OK [{self._homography_source}]"
            cal_color = (0, 255, 0)
            instruction = "SPACE = score pil  |  C = rekalibrér  |  Q = afslut"
        else:
            if cal_count >= MIN_HOMOGRAPHY_POINTS:
                cal_text = f"Cal: {cal_count}/4 klar — SPACE for at kalibrere"
                cal_color = (0, 255, 165)
            else:
                cal_text = f"Cal: {cal_count}/4 — mangler {MIN_HOMOGRAPHY_POINTS - cal_count} punkt(er)"
                cal_color = (0, 165, 255) if cal_count > 0 else (0, 0, 255)
            instruction = "SPACE = kalibrér (kræver 4/4)  |  Q = afslut"

        cv2.putText(preview, cal_text, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cal_color, 2)

        with self._overlay_lock:
            overlay = self._score_overlay

        if overlay.player_name:
            (tw, _), _ = cv2.getTextSize(overlay.player_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(preview, overlay.player_name, (w - tw - 10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            score_text = f"{overlay.score_remaining} tilbage"
            (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.putText(preview, score_text, (w - tw - 10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            if overlay.hand_scores:
                hand_text = "  +  ".join(overlay.hand_scores)
                if overlay.hand_total > 0:
                    hand_text += f"  =  {overlay.hand_total}"
                (tw, _), _ = cv2.getTextSize(hand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(preview, hand_text, (w - tw - 10, 108),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 100), 2)

        cv2.putText(preview, instruction,
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        with self._preview_lock:
            self._latest_preview = preview
