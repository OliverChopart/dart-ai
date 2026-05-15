"""Homography calibration for dart board perspective correction.

The camera hangs at an angle above the dartboard, causing the board to appear
distorted in the image.  This module lets the user click 4 points on the
double-ring (top, right, bottom, left) and computes the homography matrix that
transforms the camera view into a perfect top-down view of the board.

The resulting matrix is saved to disk and can be reloaded by every subsequent
frame-processing step.

Usage (interactive, requires a display)::

    from vision.calibration import run_calibration
    run_calibration(image_path="dataset/cropped_images/800/d1_02_04_2020/IMG_1322.JPG")

Usage (headless / from saved points)::

    from vision.calibration import compute_and_save_homography
    src_pts = [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]  # top, right, bottom, left
    compute_and_save_homography(src_pts, output_size=800)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_HOMOGRAPHY_PATH = Path("config/homography.npy")

# The canonical output square (pixels).  The board will be mapped into a
# (OUTPUT_SIZE x OUTPUT_SIZE) image where the centre is the bull.
DEFAULT_OUTPUT_SIZE = 800


def _destination_points(output_size: int) -> np.ndarray:
    """Return the 4 destination corners in the canonical top-down view."""
    half = output_size / 2
    margin = output_size * 0.02
    r = half - margin
    cx, cy = half, half
    return np.array(
        [
            [cx,      cy - r],  # top
            [cx + r,  cy    ],  # right
            [cx,      cy + r],  # bottom
            [cx - r,  cy    ],  # left
        ],
        dtype=np.float32,
    )


def compute_homography(
    src_points: List[Tuple[float, float]],
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> np.ndarray:
    """Compute the 3x3 homography matrix from 4 source points."""
    if len(src_points) != 4:
        raise ValueError(f"Expected exactly 4 source points, got {len(src_points)}")
    src = np.array(src_points, dtype=np.float32)
    dst = _destination_points(output_size)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed - check that the 4 points are not collinear.")
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
    print(f"[calibration] Homography saved to {path}")

    sidecar = path.with_suffix(".json")
    meta = {
        "output_size": output_size,
        "src_points": src_points if src_points is not None else [],
        "dst_points": _destination_points(output_size).tolist(),
        "homography": H.tolist(),
    }
    sidecar.write_text(json.dumps(meta, indent=2))
    print(f"[calibration] Metadata saved to {sidecar}")


def load_homography(path: Path | str = DEFAULT_HOMOGRAPHY_PATH) -> np.ndarray:
    """Load a previously saved homography matrix from a .npy file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No homography file found at '{path}'. "
            "Run 'scripts/run_calibration.py' first."
        )
    H = np.load(str(path))
    print(f"[calibration] Homography loaded from {path}")
    return H


def apply_homography(
    frame: np.ndarray,
    H: np.ndarray,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> np.ndarray:
    """Warp frame to the canonical top-down view using homography H."""
    return cv2.warpPerspective(frame, H, (output_size, output_size))


class _ClickCollector:
    """OpenCV mouse-click callback that collects up to n points."""

    LABELS = ["TOP (double-ring)", "RIGHT (double-ring)", "BOTTOM (double-ring)", "LEFT (double-ring)"]
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
        cv2.drawMarker(self.display, (x, y), color,
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        label = f"{idx + 1}. {self.LABELS[idx]}  ({x}, {y})"
        cv2.putText(self.display, label, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        print(f"[calibration] Point {idx + 1}/4 - {self.LABELS[idx]}: ({x}, {y})")

    def overlay_instructions(self) -> None:
        if len(self.points) < self.n:
            next_idx = len(self.points)
            msg = f"Click {next_idx + 1}/4: {self.LABELS[next_idx]}"
        else:
            msg = "4/4 points collected - press ENTER to confirm, R to reset."
        cv2.rectangle(self.display, (0, 0), (self.display.shape[1], 36), (30, 30, 30), -1)
        cv2.putText(self.display, msg, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

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
    """Open an interactive window and let the user click 4 calibration points."""
    if image_path is not None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: '{image_path}'")
        print(f"[calibration] Using still image: {image_path}")
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError("Failed to grab a frame from the camera.")
        print("[calibration] Captured still from live camera.")

    collector = _ClickCollector(frame)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame.shape[1], 1280), min(frame.shape[0], 900))
    cv2.setMouseCallback(window_name, collector)

    print("[calibration] -----------------------------------------------")
    print("[calibration] Click 4 outermost points of the double ring")
    print("[calibration] Order:  TOP -> RIGHT -> BOTTOM -> LEFT")
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
            print("[calibration] Resetting - click 4 new points.")
            collector.reset()
        elif key in (ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration aborted by user (pressed Q).")

    cv2.destroyAllWindows()

    H = compute_homography(collector.points, output_size=output_size)
    save_homography(H, path=output_path, src_points=collector.points, output_size=output_size)

    warped = apply_homography(frame, H, output_size=output_size)
    cv2.imshow("Calibration result - top-down view (press any key to close)", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H
