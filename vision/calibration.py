"""Homography calibration for dart board perspective correction.

Two calibration modes are supported:

1. **Automatic (preferred)** — ``compute_homography_from_detections()``
   Uses the four calibration-point classes detected by the 5-class YOLO model
   (cal_20, cal_6, cal_3, cal_11) to compute the homography on every frame
   without any user interaction.  The model must be trained with these classes.

2. **Manual (fallback)** — ``run_calibration()``
   Opens an interactive OpenCV window and lets the user click 4 points on the
   double-ring (top=20, right=6, bottom=3, left=11).  The resulting matrix is
   saved to disk and can be reloaded until a trained 5-class model is available.

Usage — automatic (from pipeline)::

    from vision.calibration import compute_homography_from_detections
    H = compute_homography_from_detections(detection.cal_points)

Usage — manual (from script)::

    from vision.calibration import run_calibration
    run_calibration(image_path="dataset/.../IMG_1322.JPG")
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default paths and sizes
# ---------------------------------------------------------------------------
DEFAULT_HOMOGRAPHY_PATH = Path("config/homography.npy")

# The canonical output square (pixels). The board is mapped into an
# (OUTPUT_SIZE x OUTPUT_SIZE) image with the bull at the centre.
DEFAULT_OUTPUT_SIZE = 800

# ---------------------------------------------------------------------------
# Calibration point geometry
# ---------------------------------------------------------------------------
# Angles (degrees, clockwise from 12 o'clock) for the upper-left corner of
# each double-ring segment used as a calibration landmark.
# These match dart-sense's convention and assume the standard dartboard layout.
# Validate against your annotated dataset and adjust if needed.
_CAL_SEGMENT_ANGLES: dict[int, float] = {
    20: 351.0,   # just left of 12 o'clock (upper-left of D20 bbox)
    6:   62.0,   # upper-right quadrant
    3:  102.0,   # right side
    11: 138.0,   # lower-right quadrant
}

# Radius of the double ring in the canonical 800x800 output image (pixels)
_DOUBLE_RING_RADIUS = 380.0


def _cal_destination(segment: int, output_size: int = DEFAULT_OUTPUT_SIZE) -> tuple[float, float]:
    """Return the canonical pixel coordinate for a calibration segment corner.

    Args:
        segment: Dartboard segment number — one of 20, 6, 3, 11.
        output_size: Side length of the canonical square output image.

    Returns:
        (x, y) pixel coordinate in the canonical top-down view.
    """
    if segment not in _CAL_SEGMENT_ANGLES:
        raise ValueError(f"Unknown calibration segment: {segment}. Must be one of 20, 6, 3, 11.")
    angle_deg = _CAL_SEGMENT_ANGLES[segment]
    angle_rad = math.radians(angle_deg)
    scale = output_size / DEFAULT_OUTPUT_SIZE
    cx = cy = output_size / 2.0
    r = _DOUBLE_RING_RADIUS * scale
    x = cx + r * math.sin(angle_rad)
    y = cy - r * math.cos(angle_rad)
    return x, y


def compute_homography_from_detections(
    cal_points: dict[int, tuple[float, float]],
    output_size: int = DEFAULT_OUTPUT_SIZE,
    method: int = cv2.RANSAC,
) -> Optional[np.ndarray]:
    """Compute a homography matrix from YOLO-detected calibration points.

    This is the primary calibration path when using the 5-class model.
    It is called on every frame so the homography adapts automatically to
    camera movement, zoom changes, and any perspective shift.

    Args:
        cal_points: Dict mapping segment number (20, 6, 3, 11) to the
                    (x, y) pixel coordinate detected by YOLO in the current
                    frame.  Partial detection (fewer than 4 points) returns
                    None — the caller should fall back to the last known H.
        output_size: Side length of the canonical output image in pixels.
        method: Homography estimation method passed to cv2.findHomography.
                RANSAC is recommended for robustness against detection noise.

    Returns:
        3x3 float32 homography matrix, or None if fewer than 4 points are
        available or cv2.findHomography fails.
    """
    valid_segments = [s for s in (20, 6, 3, 11) if s in cal_points]

    if len(valid_segments) < 4:
        logger.debug(
            "insufficient calibration points for homography",
            found=len(valid_segments),
            segments=valid_segments,
        )
        return None

    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []
    for segment in valid_segments:
        px, py = cal_points[segment]
        dx, dy = _cal_destination(segment, output_size)
        src_pts.append([px, py])
        dst_pts.append([dx, dy])

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    H, mask = cv2.findHomography(src, dst, method)
    if H is None:
        logger.error("cv2.findHomography failed — check that calibration points are not collinear")
        return None

    inliers = int(mask.sum()) if mask is not None else len(src_pts)
    logger.debug("homography computed from detections", inliers=inliers, total=len(src_pts))
    return H


# ---------------------------------------------------------------------------
# Shared homography utilities
# ---------------------------------------------------------------------------

def _destination_points_manual(output_size: int) -> np.ndarray:
    """Return the 4 destination corners for the manual (top/right/bottom/left) calibration."""
    half = output_size / 2
    margin = output_size * 0.02
    r = half - margin
    cx, cy = half, half
    return np.array(
        [
            [cx,      cy - r],  # top    (20)
            [cx + r,  cy    ],  # right  (6)
            [cx,      cy + r],  # bottom (3)
            [cx - r,  cy    ],  # left   (11)
        ],
        dtype=np.float32,
    )


def compute_homography(
    src_points: List[Tuple[float, float]],
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> np.ndarray:
    """Compute a 3x3 homography matrix from 4 manually selected source points.

    Use this for the manual calibration fallback (run_calibration).
    For automatic calibration from YOLO detections use
    compute_homography_from_detections() instead.
    """
    if len(src_points) != 4:
        raise ValueError(f"Expected exactly 4 source points, got {len(src_points)}")
    src = np.array(src_points, dtype=np.float32)
    dst = _destination_points_manual(output_size)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError(
            "cv2.findHomography failed — check that the 4 points are not collinear."
        )
    return H


def save_homography(
    H: np.ndarray,
    path: Path | str = DEFAULT_HOMOGRAPHY_PATH,
    src_points: List[Tuple[float, float]] | None = None,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> None:
    """Persist the homography matrix and metadata to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), H)
    logger.info("homography saved", path=str(path))

    sidecar = path.with_suffix(".json")
    meta = {
        "output_size": output_size,
        "src_points": src_points if src_points is not None else [],
        "dst_points": _destination_points_manual(output_size).tolist(),
        "homography": H.tolist(),
    }
    sidecar.write_text(json.dumps(meta, indent=2))
    logger.info("homography metadata saved", path=str(sidecar))


def load_homography(path: Path | str = DEFAULT_HOMOGRAPHY_PATH) -> np.ndarray:
    """Load a previously saved homography matrix from a .npy file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No homography file found at '{path}'. "
            "Run 'scripts/run_calibration.py' or wait for automatic calibration "
            "from the 5-class YOLO model."
        )
    H = np.load(str(path))
    logger.info("homography loaded", path=str(path))
    return H


def apply_homography(
    frame: np.ndarray,
    H: np.ndarray,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> np.ndarray:
    """Warp a frame to the canonical top-down view using homography H."""
    return cv2.warpPerspective(frame, H, (output_size, output_size))


# ---------------------------------------------------------------------------
# Interactive manual calibration
# ---------------------------------------------------------------------------

class _ClickCollector:
    """OpenCV mouse-click callback that collects up to n points."""

    LABELS = [
        "TOP — double-20 (12 o'clock)",
        "RIGHT — double-6",
        "BOTTOM — double-3",
        "LEFT — double-11",
    ]
    COLORS = [
        (0,   255, 0  ),
        (255, 128, 0  ),
        (0,   128, 255),
        (0,   0,   255),
    ]

    def __init__(self, image: np.ndarray, n: int = 4):
        self.image = image.copy()
        self.display = image.copy()
        self.n = n
        self.points: List[Tuple[int, int]] = []

    def __call__(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) >= self.n:
            return
        idx = len(self.points)
        self.points.append((x, y))
        color = self.COLORS[idx]
        cv2.drawMarker(
            self.display, (x, y), color,
            markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2,
        )
        label = f"{idx + 1}. {self.LABELS[idx]}  ({x}, {y})"
        cv2.putText(
            self.display, label, (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
        logger.info("calibration point collected", index=idx + 1, label=self.LABELS[idx], x=x, y=y)

    def overlay_instructions(self) -> None:
        if len(self.points) < self.n:
            next_idx = len(self.points)
            msg = f"Click {next_idx + 1}/4: {self.LABELS[next_idx]}"
        else:
            msg = "4/4 points collected — press ENTER to confirm, R to reset."
        cv2.rectangle(self.display, (0, 0), (self.display.shape[1], 36), (30, 30, 30), -1)
        cv2.putText(
            self.display, msg, (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    def reset(self) -> None:
        self.points = []
        self.display = self.image.copy()


def run_calibration(
    image_path: str | Path | None = None,
    camera_index: int = 0,
    output_path: Path | str = DEFAULT_HOMOGRAPHY_PATH,
    output_size: int = DEFAULT_OUTPUT_SIZE,
    window_name: str = "Dart Board Calibration",
) -> np.ndarray:
    """Open an interactive window and let the user click 4 calibration points.

    This is the manual fallback for use before a 5-class YOLO model is
    available.  Click the outermost point of the double ring at:
        1. TOP    (double-20, 12 o'clock)
        2. RIGHT  (double-6)
        3. BOTTOM (double-3)
        4. LEFT   (double-11)

    The computed homography is saved to output_path and will be loaded
    automatically by DartPipeline as a fallback when automatic calibration
    from YOLO is unavailable.
    """
    if image_path is not None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: '{image_path}'")
        logger.info("using still image for calibration", path=str(image_path))
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError("Failed to grab a frame from the camera.")
        logger.info("captured still from live camera for calibration")

    collector = _ClickCollector(frame)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame.shape[1], 1280), min(frame.shape[0], 900))
    cv2.setMouseCallback(window_name, collector)

    print("[calibration] -----------------------------------------------")
    print("[calibration] Click the outermost point of the double ring")
    print("[calibration] Order:  TOP (D20) -> RIGHT (D6) -> BOTTOM (D3) -> LEFT (D11)")
    print("[calibration] ENTER to confirm, R to reset, Q to quit.")
    print("[calibration] -----------------------------------------------")

    while True:
        collector.overlay_instructions()
        cv2.imshow(window_name, collector.display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # ENTER
            if len(collector.points) < 4:
                print("[calibration] Need 4 points before confirming.")
            else:
                break
        elif key in (ord('r'), ord('R')):
            print("[calibration] Resetting — click 4 new points.")
            collector.reset()
        elif key in (ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration aborted by user (pressed Q).")

    cv2.destroyAllWindows()

    H = compute_homography(collector.points, output_size=output_size)
    save_homography(H, path=output_path, src_points=collector.points, output_size=output_size)

    warped = apply_homography(frame, H, output_size=output_size)
    cv2.imshow("Calibration result — top-down view (press any key to close)", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H
