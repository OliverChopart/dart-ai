"""Quick test: run YOLO detection on a sample image from the DeepDarts dataset.

Usage:
    uv run python scripts/test_detector.py
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.detector import DartDetector


def main() -> None:
    # Find a sample image from the dataset
    dataset_path = Path("dataset/cropped_images/800")
    sample_images = list(dataset_path.rglob("*.JPG"))

    if not sample_images:
        print("No images found in dataset/cropped_images/800/")
        print("Make sure you have extracted cropped_images.zip")
        return

    sample = str(sample_images[0])
    print(f"Testing on: {sample}")

    # Load detector and run
    detector = DartDetector()
    detector.load()

    result = detector.detect_from_file(sample, annotate=True)

    print(f"Detections found: {len(result.dart_tips)}")
    for i, (tip, conf) in enumerate(zip(result.dart_tips, result.confidences)):
        print(f"  [{i+1}] x={tip[0]:.1f}, y={tip[1]:.1f}, confidence={conf:.2f}")

    # Show annotated image
    if result.annotated_frame is not None:
        output_path = "test_detection_output.jpg"
        cv2.imwrite(output_path, result.annotated_frame)
        print(f"\nAnnotated image saved to: {output_path}")
        print("Open it in Finder to see the detections")


if __name__ == "__main__":
    main()
