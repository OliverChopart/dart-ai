"""Fine-tune YOLOv11n on the DeepDarts dataset.

Usage:
    uv run python scripts/train.py

The model will be saved to runs/train/dart_detector/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def main() -> None:
    dataset_yaml = Path("dataset/yolo/dataset.yaml")
    if not dataset_yaml.exists():
        print("Dataset not converted yet. Run first:")
        print("  uv run python scripts/convert_dataset.py")
        return

    print("Loading YOLOv11n base model...")
    model = YOLO("models/yolo11n.pt")

    print("Starting fine-tuning on DeepDarts dataset...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=100,
        imgsz=800,
        batch=16,
        device="mps",
        project="runs/train",
        name="dart_detector",
        save=True,
        plots=True,
        patience=20,       # early stopping
        lr0=0.001,
        warmup_epochs=3,
    )

    print("\nTraining complete.")
    print(f"Best model saved to: runs/train/dart_detector/weights/best.pt")
    print("Update YOLO_MODEL_PATH in .env to use the trained model:")
    print("  YOLO_MODEL_PATH=runs/train/dart_detector/weights/best.pt")


if __name__ == "__main__":
    main()
