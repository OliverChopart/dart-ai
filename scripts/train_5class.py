"""Train 5-class YOLO on McNally's dataset + your own images.

Workflow
--------
1. Convert McNally's dataset (once):
       uv run python scripts/convert_dataset_5class.py

2. (Optional but recommended) Add your own images:
       - Take 50-100 photos with your camera
       - Annotate them at https://roboflow.com  (export: YOLOv8, 5 classes)
       - Place images in:  dataset/own/images/
       - Place labels in:  dataset/own/labels/
       See README or run with --check to verify the folder structure.

3. Train:
       uv run python scripts/train_5class.py

   To train WITHOUT your own images (McNally only):
       uv run python scripts/train_5class.py --no-own

4. Update .env when done:
       YOLO_MODEL_PATH=runs/train/dart_5class/weights/best.pt

Class mapping (must match Roboflow export order):
    0: dart
    1: cal_20
    2: cal_6
    3: cal_3
    4: cal_11
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MCNALLY_DIR = Path("dataset/yolo_5class")
OWN_DIR = Path("dataset/own")
MERGED_DIR = Path("dataset/yolo_5class_merged")

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def check_own_images() -> bool:
    """Return True if own images folder exists and contains annotated images."""
    img_dir = OWN_DIR / "images"
    lbl_dir = OWN_DIR / "labels"

    if not img_dir.exists():
        return False

    images = [f for f in img_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS]
    labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    if images and labels:
        print(f"Own images found: {len(images)} images, {len(labels)} labels in {OWN_DIR}")
        return True

    if images and not labels:
        print(f"WARNING: {len(images)} images found in {img_dir} but no labels in {lbl_dir}")
        print("Annotate your images first: https://roboflow.com")
        print("Export as YOLOv8 format with these 5 classes in order:")
        print("  0: dart  1: cal_20  2: cal_6  3: cal_3  4: cal_11")
        return False

    return False


def merge_datasets(use_own: bool) -> Path:
    """Merge McNally dataset with optional own images into MERGED_DIR.

    Own images are always added to the train split only — they are too few
    to split further, and keeping them in train maximises their impact on
    learning your specific camera angle.
    """
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    for split in SPLITS:
        (MERGED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy McNally splits
    total_mcnally = 0
    for split in SPLITS:
        src_imgs = MCNALLY_DIR / "images" / split
        src_lbls = MCNALLY_DIR / "labels" / split
        if not src_imgs.exists():
            continue
        for img in src_imgs.iterdir():
            if img.suffix not in IMAGE_EXTENSIONS:
                continue
            shutil.copy2(img, MERGED_DIR / "images" / split / img.name)
            lbl = src_lbls / img.with_suffix(".txt").name
            if lbl.exists():
                shutil.copy2(lbl, MERGED_DIR / "labels" / split / lbl.name)
                total_mcnally += 1

    print(f"McNally images copied: {total_mcnally}")

    # Copy own images into train split
    total_own = 0
    if use_own:
        own_imgs = OWN_DIR / "images"
        own_lbls = OWN_DIR / "labels"
        for img in own_imgs.iterdir():
            if img.suffix not in IMAGE_EXTENSIONS:
                continue
            dst_name = f"own_{img.name}"
            shutil.copy2(img, MERGED_DIR / "images" / "train" / dst_name)
            # Roboflow label filename matches image filename with .txt extension
            lbl = own_lbls / img.with_suffix(".txt").name
            if lbl.exists():
                shutil.copy2(lbl, MERGED_DIR / "labels" / "train" / f"own_{lbl.name}")
                total_own += 1

        print(f"Own images added to train: {total_own}")

    # Write merged dataset.yaml
    yaml = f"""# Merged dataset: McNally ({total_mcnally}) + own ({total_own})
path: {MERGED_DIR.absolute()}
train: images/train
val:   images/val
test:  images/test

nc: 5
names:
  0: dart
  1: cal_20
  2: cal_6
  3: cal_3
  4: cal_11
"""
    (MERGED_DIR / "dataset.yaml").write_text(yaml)
    print(f"Merged dataset.yaml written to {MERGED_DIR / 'dataset.yaml'}")
    return MERGED_DIR / "dataset.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 5-class dart detector")
    parser.add_argument(
        "--no-own",
        action="store_true",
        help="Train on McNally dataset only, skip own images",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check folder structure and exit without training",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    if not MCNALLY_DIR.exists() or not (MCNALLY_DIR / "dataset.yaml").exists():
        print("ERROR: McNally dataset not converted yet. Run first:")
        print("  uv run python scripts/convert_dataset_5class.py")
        return

    has_own = check_own_images()

    if args.check:
        print("\nMcNally dataset: OK" if MCNALLY_DIR.exists() else "\nMcNally dataset: MISSING")
        print(f"Own images:      {'OK' if has_own else 'not found (optional)'}")
        print(f"\nOwn images folder: {OWN_DIR}/")
        print("  images/   <-- put your .jpg/.jpeg/.png files here")
        print("  labels/   <-- put Roboflow-exported .txt files here")
        return

    use_own = has_own and not args.no_own
    if not use_own and has_own:
        print("Skipping own images (--no-own flag set)")
    elif not has_own:
        print(f"No own images found in {OWN_DIR} — training on McNally only")
        print("To add your own images later, see the docstring at the top of this file")

    # ------------------------------------------------------------------
    # Merge datasets
    # ------------------------------------------------------------------
    print("\nPreparing merged dataset...")
    dataset_yaml = merge_datasets(use_own=use_own)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\nLoading base model...")
    model = YOLO("models/yolo11n.pt")

    print("Starting training...")
    print("Classes: dart, cal_20, cal_6, cal_3, cal_11\n")

    model.train(
        data=str(dataset_yaml),
        epochs=100,
        imgsz=800,
        batch=16,
        device="mps",       # change to 'cuda' on Linux/Windows with GPU
        project="runs/train",
        name="dart_5class",
        save=True,
        plots=True,
        patience=20,
        lr0=0.001,
        warmup_epochs=3,
    )

    best = Path("runs/train/dart_5class/weights/best.pt")
    print("\n=== Training complete ===")
    print(f"Best model: {best}")
    print("\nUpdate .env:")
    print(f"  YOLO_MODEL_PATH={best}")
    print("\nThen play:")
    print("  uv run python scripts/play_301.py")


if __name__ == "__main__":
    main()
