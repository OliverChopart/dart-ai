"""YOLO-based dartboard and dart tip detector."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DartDetection:
    """Result of detecting darts in a single frame."""

    dart_tips: list[tuple[float, float]]  # (x, y) in pixel coords
    confidences: list[float]
    board_bbox: Optional[tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    annotated_frame: Optional[np.ndarray] = None


class DartDetector:
    """Wraps YOLOv11 for dartboard and dart tip detection.

    On first run with a COCO-pretrained model, the detector will attempt
    to find objects visually similar to darts. Once fine-tuned on the
    DeepDarts dataset, detection will be dart-specific.
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
                "Run: curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -o models/yolo11n.pt"
            )
        self._model = YOLO(self._model_path)
        self._model.to(self._device)
        logger.info("model loaded", model=self._model_path)
        return self

    def detect(self, frame: np.ndarray, annotate: bool = True) -> DartDetection:
        """Run detection on a single frame.

        Args:
            frame: BGR numpy array from OpenCV.
            annotate: If True, draw bounding boxes on a copy of the frame.

        Returns:
            DartDetection with dart tip coordinates and optional annotated frame.
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

                # Class 0 = dart tip, Class 1+ = board segments
                # (After fine-tuning; with COCO weights we use all detections)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                dart_tips.append((cx, cy))
                confidences.append(conf)

            if annotate:
                annotated_frame = result.plot()

        logger.debug(
            "detection complete",
            darts_found=len(dart_tips),
            board_found=board_bbox is not None,
        )

        return DartDetection(
            dart_tips=dart_tips,
            confidences=confidences,
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
