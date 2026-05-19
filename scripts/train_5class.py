"""Train 5-class YOLO on McNally's dataset + your own images.

Workflow
--------
1. Convert McNally's dataset (once):
       uv run python scripts/convert_dataset_5class.py

2. Add your own images:
       - Place images in:  dataset/own/images/
       - Place labels in:  dataset/own/labels/

3. Train (McNally + own images):
       caffeinate -i uv run python scripts/train_5class.py

   Train on OWN IMAGES ONLY (recommended if McNally causes confusion):
       caffeinate -i uv run python scripts/train_5class.py --own-only

   Train on McNally only (no own images):
       caffeinate -i uv run python scripts/train_5class.py --no-own

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
    img_dir = OWN_DIR / "images"
    lbl_dir = OWN_DIR / "labels"
    if not img_dir.exists():
        return False
    images = [f for f in img_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS]
    labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
    if images and labels:
        print(f"Own images found: {len(images)} images, {len(labels)} labels")
        return True
    return False


def build_own_only_dataset() -> Path:
    """Build a dataset from own images only — 80% train, 20% val."""
    import random

    out = MERGED_DIR
    if out.exists():
        shutil.rmtree(out)
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    own_imgs = OWN_DIR / "images"
    own_lbls = OWN_DIR / "labels"
    all_imgs = [f for f in own_imgs.iterdir() if f.suffix in IMAGE_EXTENSIONS]
    random.shuffle(all_imgs)
    split_idx = int(len(all_imgs) * 0.8)
    train_imgs = all_imgs[:split_idx]
    val_imgs = all_imgs[split_idx:]

    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        for img in imgs:
            shutil.copy2(img, out / "images" / split / img.name)
            lbl = own_lbls / img.with_suffix(".txt").name
            if lbl.exists():
                shutil.copy2(lbl, out / "labels" / split / lbl.name)

    print(f"Own-only dataset: {len(train_imgs)} train, {len(val_imgs)} val")

    yaml = f"""# Own images only dataset
path: {out.absolute()}
train: images/train
val:   images/val

nc: 5
names:
  0: dart
  1: cal_20
  2: cal_6
  3: cal_3
  4: cal_11
"""
    (out / "dataset.yaml").write_text(yaml)
    return out / "dataset.yaml"


def merge_datasets(use_own: bool) -> Path:
    """Merge McNally dataset with optional own images into MERGED_DIR."""
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    for split in SPLITS:
        (MERGED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

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

    total_own = 0
    if use_own:
        own_imgs = OWN_DIR / "images"
        own_lbls = OWN_DIR / "labels"
        for img in own_imgs.iterdir():
            if img.suffix not in IMAGE_EXTENSIONS:
                continue
            dst_name = f"own_{img.name}"
            shutil.copy2(img, MERGED_DIR / "images" / "train" / dst_name)
            lbl = own_lbls / img.with_suffix(".txt").name
            if lbl.exists():
                shutil.copy2(lbl, MERGED_DIR / "labels" / "train" / f"own_{lbl.name}")
                total_own += 1
        print(f"Own images added to train: {total_own}")

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
    parser.add_argument("--no-own", action="store_true", help="McNally only, skip own images")
    parser.add_argument("--own-only", action="store_true", help="Own images only, skip McNally")
    parser.add_argument("--check", action="store_true", help="Check folder structure and exit")
    args = parser.parse_args()

    has_own = check_own_images()

    if args.check:
        print("\nMcNally dataset: OK" if MCNALLY_DIR.exists() else "\nMcNally dataset: MISSING")
        print(f"Own images:      {'OK' if has_own else 'not found'}")
        return

    if args.own_only:
        if not has_own:
            print("ERROR: No own images found in dataset/own/")
            return
        print("\n⚠️  Training on OWN IMAGES ONLY — McNally excluded")
        print("This gives best results for your specific camera angle.\n")
        dataset_yaml = build_own_only_dataset()
    else:
        if not MCNALLY_DIR.exists() or not (MCNALLY_DIR / "dataset.yaml").exists():
            print("ERROR: McNally dataset not converted yet. Run first:")
            print("  uv run python scripts/convert_dataset_5class.py")
            return
        use_own = has_own and not args.no_own
        if not use_own and has_own:
            print("Skipping own images (--no-own flag set)")
        elif not has_own:
            print(f"No own images found in {OWN_DIR} — training on McNally only")
        print("\nPreparing merged dataset...")
        dataset_yaml = merge_datasets(use_own=use_own)

    print("\nLoading base model...")
    model = YOLO("models/yolo11n.pt")
    print("Starting training...")
    print("Classes: dart, cal_20, cal_6, cal_3, cal_11\n")

    model.train(
        data=str(dataset_yaml),
        epochs=100,
        imgsz=640,
        batch=8,
        device="mps",
        project="runs/train",
        name="dart_5class",
        save=True,
        plots=True,
        patience=20,
        lr0=0.001,
        warmup_epochs=3,
        workers=0,
        cache=False,
    )

    best = Path("runs/train/dart_5class/weights/best.pt")
    print("\n=== Training complete ===")
    print(f"Best model: {best}")
    print("\nUpdate .env:")
    print(f"  YOLO_MODEL_PATH={best}")
    print("\nThen test:")
    print("  uv run python scripts/test_detector.py --random")
    print("\nThen play:")
    print("  uv run python scripts/play_301.py")


if __name__ == "__main__":
    main()
