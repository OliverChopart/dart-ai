"""YOLO-based dart tip and calibration point detector.

Class mapping for the 5-class model:
    0  dart    — dart tip
    1  cal_20  — upper-left corner of double-20 segment
    2  cal_6   — upper-left corner of double-6 segment
    3  cal_3   — upper-left corner of double-3 segment
    4  cal_11  — upper-left corner of double-11 segment

The four calibration point classes allow automatic per-frame homography
computation without any manual user interaction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Class indices — must match the label order used during training
# ---------------------------------------------------------------------------
CLASS_DART = 0
CLASS_CAL_20 = 1
CLASS_CAL_6 = 2
CLASS_CAL_3 = 3
CLASS_CAL_11 = 4

CLASS_NAMES: dict[int, str] = {
    CLASS_DART: "dart",
    CLASS_CAL_20: "cal_20",
    CLASS_CAL_6: "cal_6",
    CLASS_CAL_3: "cal_3",
    CLASS_CAL_11: "cal_11",
}

# Maps calibration class index -> dartboard segment number
CAL_CLASS_TO_SEGMENT: dict[int, int] = {
    CLASS_CAL_20: 20,
    CLASS_CAL_6: 6,
    CLASS_CAL_3: 3,
    CLASS_CAL_11: 11,
}


@dataclass
class DartDetection:
    """Result of detecting darts and calibration points in a single frame."""

    dart_tips: list[tuple[float, float]]  # (x, y) in pixel coords
    confidences: list[float]

    # Calibration points detected by YOLO.
    # Key = dartboard segment number (20, 6, 3, 11).
    # Value = (x, y) pixel coordinate — best confidence detection per class.
    cal_points: dict[int, tuple[float, float]] = field(default_factory=dict)

    board_bbox: Optional[tuple[int, int, int, int]] = None
    annotated_frame: Optional[np.ndarray] = None

    @property
    def has_calibration(self) -> bool:
        return len(self.cal_points) >= 1

    @property
    def has_full_calibration(self) -> bool:
        return all(k in self.cal_points for k in (20, 6, 3, 11))


class DartDetector:
    """Wraps YOLOv8/v11 for dart tip and calibration point detection."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path or settings.yolo_model_path
        self._device = self._select_device(settings.detection_device)
        self._model: Optional[YOLO] = None
        logger.info("detector initialised", model=self._model_path, device=self._device)

    def load(self) -> "DartDetector":
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self._model_path}. "
                "Download a pretrained base: "
                "curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt "
                "-o models/yolo11n.pt"
            )
        self._model = YOLO(self._model_path)
        self._model.to(self._device)
        logger.info("model loaded", model=self._model_path)
        return self

    def detect(self, frame: np.ndarray, annotate: bool = True) -> DartDetection:
        """Run detection on a single frame.

        For calibration points: if YOLO detects the same class multiple times
        (e.g. two cal_11 boxes), only the one with the highest confidence is
        kept. This prevents double-detections from blocking calibration.

        For dart tips: all detections above the confidence threshold are kept.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call detector.load() first.")

        results = self._model(
            frame,
            conf=settings.detection_confidence,
            verbose=False,
        )

        dart_tips: list[tuple[float, float]] = []
        confidences: list[float] = []

        # Track best confidence per cal segment to handle duplicate detections
        # Key = segment number, Value = (confidence, x, y)
        cal_best: dict[int, tuple[float, float, float]] = {}

        annotated_frame = None

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls == CLASS_DART:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    dart_tips.append((cx, cy))
                    confidences.append(conf)

                elif cls in CAL_CLASS_TO_SEGMENT:
                    segment = CAL_CLASS_TO_SEGMENT[cls]
                    # Keep only the highest-confidence detection per segment
                    if segment not in cal_best or conf > cal_best[segment][0]:
                        cal_best[segment] = (conf, x1, y1)

            if annotate:
                annotated_frame = result.plot()

        # Convert cal_best to cal_points
        cal_points = {
            segment: (x, y)
            for segment, (conf, x, y) in cal_best.items()
        }

        logger.debug(
            "detection complete",
            darts_found=len(dart_tips),
            cal_points_found=list(cal_points.keys()),
            full_calibration=all(k in cal_points for k in (20, 6, 3, 11)),
        )

        return DartDetection(
            dart_tips=dart_tips,
            confidences=confidences,
            cal_points=cal_points,
            annotated_frame=annotated_frame,
        )

    def detect_from_file(self, image_path: str, annotate: bool = True) -> DartDetection:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect(frame, annotate=annotate)

    @staticmethod
    def _select_device(preferred: str) -> str:
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
