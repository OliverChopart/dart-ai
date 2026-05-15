"""Video stream reader - runs in a dedicated thread.

Supports any source OpenCV can open:
  - Integer index (0, 1, ...) for local webcams
  - RTSP URL  e.g. rtsp://192.168.1.42:8554/live
  - HTTP MJPEG URL  e.g. http://192.168.1.42:4747/video  (DroidCam / EpocCam)
  - File path for offline testing

If a homography matrix has been saved by vision/calibration.py, every frame
is automatically warped to the canonical top-down view before being placed in
the queue.
"""

import queue
import threading
import time
from pathlib import Path
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

    If ``apply_homography`` is True (the default) and a homography file
    exists at ``homography_path``, each frame is warped to the top-down
    canonical view before being enqueued.
    """

    def __init__(
        self,
        source: Optional[str] = None,
        apply_homography: bool = True,
        homography_path: str | Path = "config/homography.npy",
    ) -> None:
        raw = source or settings.camera_source
        self._source: int | str = int(raw) if str(raw).isdigit() else raw
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Homography (perspective correction)
        self._H: Optional[np.ndarray] = None
        self._output_size: int = 800
        if apply_homography:
            self._H = self._try_load_homography(Path(homography_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "VideoStream":
        """Start the background reader thread."""
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        logger.info("video stream started", source=self._source,
                    homography=self._H is not None)
        return self

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read the latest frame (warped if homography is loaded).

        Returns None if no frame is available within *timeout* seconds.
        """
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

    @property
    def homography_active(self) -> bool:
        """True if perspective correction is being applied to frames."""
        return self._H is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

            if self._H is not None:
                frame = cv2.warpPerspective(
                    frame, self._H, (self._output_size, self._output_size)
                )

            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            self._queue.put(frame)

        cap.release()

    @staticmethod
    def _try_load_homography(path: Path) -> Optional[np.ndarray]:
        """Load homography from disk, or return None if not found."""
        if not path.exists():
            logger.info("no homography file found - streaming raw frames",
                        path=str(path))
            return None
        H = np.load(str(path))
        logger.info("homography loaded - frames will be perspective-corrected",
                    path=str(path))
        return H
