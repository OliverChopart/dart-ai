"""Video stream reader - runs in a dedicated thread."""

import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)


class VideoStream:
    """Reads frames from a camera source in a background thread.

    Frames are placed in a queue (maxsize=2). If full, the oldest
    frame is dropped so the consumer always gets the latest.
    """

    def __init__(self, source: Optional[str] = None) -> None:
        raw = source or settings.camera_source
        self._source: int | str = int(raw) if raw.isdigit() else raw
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> "VideoStream":
        """Start the background reader thread."""
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        logger.info("video stream started", source=self._source)
        return self

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read the latest frame. Returns None if no frame available."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop the reader thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("video stream stopped")

    def _reader(self) -> None:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            logger.error("failed to open camera", source=self._source)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
        cap.set(cv2.CAP_PROP_FPS, settings.camera_fps)

        logger.info(
            "camera opened",
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        while self._running:
            ok, frame = cap.read()
            if not ok:
                logger.warning("frame read failed - retrying")
                time.sleep(0.1)
                continue

            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            self._queue.put(frame)

        cap.release()
