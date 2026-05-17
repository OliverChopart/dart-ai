"""Quick visual test of the trained 5-class dart detector.

Runs the model on a single image and shows:
- Detected dart tips (class 0)
- Detected calibration points (classes 1-4)
- Computed homography and warped top-down view
- Scored dart tips

Usage:
    # Test on a specific image
    uv run python scripts/test_detector.py --image dataset/own/images/IMG_2221.jpeg

    # Test on a random image from your own dataset
    uv run python scripts/test_detector.py --random

    # Test on live camera
    uv run python scripts/test_detector.py --camera
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from vision.calibration import DEFAULT_OUTPUT_SIZE, apply_homography, compute_homography_from_detections
from vision.detector import DartDetector
from vision.scorer import DartScorer


def draw_results(frame: np.ndarray, detection, H, results) -> np.ndarray:
    """Draw detections and scores onto the frame."""
    vis = frame.copy()

    # Draw calibration points
    cal_colors = {20: (0, 255, 255), 6: (255, 128, 0), 3: (0, 128, 255), 11: (128, 0, 255)}
    for segment, (x, y) in detection.cal_points.items():
        color = cal_colors.get(segment, (255, 255, 0))
        cv2.drawMarker(vis, (int(x), int(y)), color, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis, f"cal_{segment}", (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw dart tips and scores
    for i, (tip, result) in enumerate(zip(detection.dart_tips, results)):
        x, y = int(tip[0]), int(tip[1])
        cv2.drawMarker(vis, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(vis, result.segment, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Status overlay
    cal_count = len(detection.cal_points)
    cal_color = (0, 255, 0) if detection.has_full_calibration else (0, 165, 255)
    cv2.putText(vis, f"Cal: {cal_count}/4  Darts: {len(detection.dart_tips)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cal_color, 2)

    if results:
        total = sum(r.score for r in results)
        scores_str = " + ".join(r.segment for r in results) + f" = {total}"
        cv2.putText(vis, scores_str, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


def test_image(image_path: str) -> None:
    """Run detection on a single image and display results."""
    print(f"\nLoading model: {settings.yolo_model_path}")
    detector = DartDetector()
    detector.load()
    scorer = DartScorer()

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    print(f"Running detection on: {image_path}")
    detection = detector.detect(frame, annotate=False)

    print(f"\nResults:")
    print(f"  Dart tips:        {len(detection.dart_tips)}")
    print(f"  Cal points found: {list(detection.cal_points.keys())}")
    print(f"  Full calibration: {detection.has_full_calibration}")

    H = None
    results = []

    if detection.has_full_calibration:
        H = compute_homography_from_detections(detection.cal_points)
        if H is not None:
            results = scorer.score_detections_with_homography(
                detection.dart_tips, H
            )
            print(f"\nScores:")
            for r in results:
                print(f"  {r}")
        else:
            print("  WARNING: Homography computation failed")
    else:
        print(f"\n  WARNING: Only {len(detection.cal_points)}/4 cal points detected")
        print("  Scores cannot be computed without full calibration")
        print("  Try retraining with more annotated images of your setup")

    # Show annotated frame
    vis = draw_results(frame, detection, H, results)

    # Show warped top-down view if homography available
    if H is not None:
        warped = apply_homography(frame, H, output_size=DEFAULT_OUTPUT_SIZE)
        # Draw scored dart positions on warped view
        half = DEFAULT_OUTPUT_SIZE / 2
        for r in results:
            bx = int(r.board_x * half + half)
            by = int(r.board_y * half + half)
            cv2.drawMarker(warped, (bx, by), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(warped, r.segment, (bx + 6, by - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Top-down view (homography corrected)", warped)

    cv2.imshow("Detection results", vis)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_random() -> None:
    """Pick a random image from dataset/own/images/ and test it."""
    own_dir = Path("dataset/own/images")
    if not own_dir.exists():
        print(f"ERROR: {own_dir} not found")
        return

    images = [f for f in own_dir.iterdir()
              if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        print(f"ERROR: No images found in {own_dir}")
        return

    chosen = random.choice(images)
    print(f"Random image: {chosen}")
    test_image(str(chosen))


def test_camera() -> None:
    """Run detection on live camera feed. Press Q to quit, SPACE to score."""
    print(f"\nLoading model: {settings.yolo_model_path}")
    detector = DartDetector()
    detector.load()
    scorer = DartScorer()

    cap = cv2.VideoCapture(int(settings.camera_source)
                           if settings.camera_source.isdigit()
                           else settings.camera_source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {settings.camera_source}")
        return

    print("Camera open. Press SPACE to score current frame, Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection = detector.detect(frame, annotate=False)
        H = None
        results = []

        if detection.has_full_calibration:
            H = compute_homography_from_detections(detection.cal_points)
            if H is not None:
                results = scorer.score_detections_with_homography(
                    detection.dart_tips, H
                )

        vis = draw_results(frame, detection, H, results)
        cv2.imshow("Dart detector — live (Q=quit, SPACE=score)", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and results:
            print("--- SCORE ---")
            for r in results:
                print(f"  {r}")
            total = sum(r.score for r in results)
            print(f"  Total: {total}")

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the trained 5-class dart detector")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", type=str, help="Path to a specific image")
    group.add_argument("--random", action="store_true", help="Pick a random image from dataset/own")
    group.add_argument("--camera", action="store_true", help="Test on live camera feed")
    args = parser.parse_args()

    if not Path(settings.yolo_model_path).exists():
        print(f"ERROR: Model not found: {settings.yolo_model_path}")
        print("Has training finished? Update .env with:")
        print("  YOLO_MODEL_PATH=runs/train/dart_5class/weights/best.pt")
        return

    if args.image:
        test_image(args.image)
    elif args.random:
        test_random()
    elif args.camera:
        test_camera()
    else:
        # Default: pick a random own image
        test_random()


if __name__ == "__main__":
    main()
