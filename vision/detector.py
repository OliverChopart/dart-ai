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
    # Value = (x, y) pixel coordinate of the upper-left corner of the double ring.
    cal_points: dict[int, tuple[float, float]] = field(default_factory=dict)

    board_bbox: Optional[tuple[int, int, int, int]] = None  # kept for HoughCircles fallback
    annotated_frame: Optional[np.ndarray] = None

    @property
    def has_calibration(self) -> bool:
        """True if at least one calibration point was detected."""
        return len(self.cal_points) >= 1

    @property
    def has_full_calibration(self) -> bool:
        """True if all four calibration points were detected."""
        return all(k in self.cal_points for k in (20, 6, 3, 11))


class DartDetector:
    """Wraps YOLOv8/v11 for dart tip and calibration point detection.

    With the 5-class model the detector separates dart tips from the four
    calibration corner classes so that the pipeline can compute a fresh
    homography matrix on every frame without manual intervention.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path or settings.yolo_model_path
        self._device = self._select_device(settings.detection_device)
        self._model: Optional[YOLO] = None
        logger.info("detector initialised", model=self._model_path, device=self._device)

    def load(self) -> "DartDetector":
        """Load the YOLO model into memory."""
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

        Boxes with class 0 (dart) are collected as dart tips.
        Boxes with classes 1-4 (cal_20, cal_6, cal_3, cal_11) are collected
        as calibration points using the upper-left corner of their bounding box.

        Args:
            frame: BGR numpy array from OpenCV.
            annotate: If True, draw bounding boxes on a copy of the frame.

        Returns:
            DartDetection with dart tips, calibration points, and optional
            annotated frame.
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
        cal_points: dict[int, tuple[float, float]] = {}
        board_bbox: Optional[tuple[int, int, int, int]] = None
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
                    # Use bounding box centre as dart tip coordinate
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    dart_tips.append((cx, cy))
                    confidences.append(conf)

                elif cls in CAL_CLASS_TO_SEGMENT:
                    # Use upper-left corner of bbox as the calibration point.
                    # This matches dart-sense's convention: the corner of the
                    # double-ring segment, not its centre.
                    segment = CAL_CLASS_TO_SEGMENT[cls]
                    cal_points[segment] = (x1, y1)

            if annotate:
                annotated_frame = result.plot()

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
            board_bbox=board_bbox,
            annotated_frame=annotated_frame,
        )

    def detect_from_file(self, image_path: str, annotate: bool = True) -> DartDetection:
        """Run detection on an image file.

        Args:
            image_path: Path to a JPG/PNG image.
            annotate: If True, draw bounding boxes on the result.

        Returns:
            DartDetection result.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect(frame, annotate=annotate)

    @staticmethod
    def _select_device(preferred: str) -> str:
        """Select best available compute device."""
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
