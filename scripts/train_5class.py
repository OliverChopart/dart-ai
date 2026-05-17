"""Train 5-class YOLOv8n on the converted DeepDarts dataset.

Run convert_dataset_5class.py first, then:

    uv run python scripts/train_5class.py

The best model is saved to:
    runs/train/dart_5class/weights/best.pt

When training is done, update .env:
    YOLO_MODEL_PATH=runs/train/dart_5class/weights/best.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def main() -> None:
    dataset_yaml = Path("dataset/yolo_5class/dataset.yaml")

    if not dataset_yaml.exists():
        print("ERROR: dataset not converted yet. Run first:")
        print("  uv run python scripts/convert_dataset_5class.py")
        return

    print("Loading YOLOv8n base model...")
    # Use YOLOv8n — faster to train than v11 and well-tested for this task
    model = YOLO("models/yolo11n.pt")

    print("Starting training...")
    print(f"Dataset: {dataset_yaml}")
    print("Classes: dart, cal_20, cal_6, cal_3, cal_11\n")

    model.train(
        data=str(dataset_yaml),
        epochs=100,
        imgsz=800,
        batch=16,
        device="mps",          # change to 'cuda' if on Linux/Windows with GPU
        project="runs/train",
        name="dart_5class",
        save=True,
        plots=True,
        patience=20,           # stop early if no improvement for 20 epochs
        lr0=0.001,
        warmup_epochs=3,
        # Class weights: upweight cal points slightly since there are fewer
        # dart tips than cal points per image (3 darts vs 4 cal points)
        # — leave at default (1.0) for first run, tune if cal mAP is low
    )

    best = Path("runs/train/dart_5class/weights/best.pt")
    print("\n=== Training complete ===")
    print(f"Best model: {best}")
    print("\nUpdate .env to use the new model:")
    print(f"  YOLO_MODEL_PATH={best}")
    print("\nThen run the game:")
    print("  uv run python scripts/play_301.py")


if __name__ == "__main__":
    main()
