"""Convert DeepDarts labels.pkl to YOLO keypoint format.

McNally's labels.pkl contains:
  - 4 calibration points (top, bottom, left, right of board)
  - 0-3 dart tip coordinates
  All in normalised coordinates (0-1) relative to the cropped image.

We convert to YOLO keypoint format:
  - One .txt file per image
  - One row per dart tip
  - Format: class cx cy w h kp_x kp_y kp_visible
  - class = 0 (dart)
  - bbox is a small box around the dart tip
  - keypoint is the dart tip itself

Also generates dataset.yaml for ultralytics training.

Usage:
    uv run python scripts/convert_dataset.py
"""

import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
DATASET_ROOT = Path("dataset")
IMAGES_ROOT = DATASET_ROOT / "cropped_images" / "800"
LABELS_PKL = DATASET_ROOT / "labels.pkl"
OUTPUT_ROOT = DATASET_ROOT / "yolo"

# YOLO bbox size around dart tip (normalised, relative to 800x800 image)
TIP_BOX_SIZE = 0.05  # 5% of image = 40px box around each tip


def convert() -> None:
    print("Loading labels.pkl...")
    df = pickle.load(open(LABELS_PKL, "rb"))
    print(f"Total annotations: {len(df)}")

    # Filter to only images that exist on disk
    valid_rows = []
    missing = 0
    for _, row in df.iterrows():
        img_path = IMAGES_ROOT / row["img_folder"] / row["img_name"]
        if img_path.exists():
            valid_rows.append(row)
        else:
            missing += 1

    print(f"Valid images: {len(valid_rows)} (missing: {missing})")
    df_valid = pd.DataFrame(valid_rows)

    # Filter to images with at least 1 dart tip
    df_valid["xy_arr"] = df_valid["xy"].apply(np.array)
    df_valid["n_darts"] = df_valid["xy_arr"].apply(lambda x: max(0, len(x) - 4))
    df_darts = df_valid[df_valid["n_darts"] > 0].copy()
    print(f"Images with darts: {len(df_darts)}")

    # Train/val/test split (70/15/15)
    train_df, temp_df = train_test_split(df_darts, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    print(f"Split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Create output directories
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Convert each split
    for split, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"Converting {split}...")
        for _, row in split_df.iterrows():
            src_img = IMAGES_ROOT / row["img_folder"] / row["img_name"]
            dst_img = OUTPUT_ROOT / "images" / split / f"{row['img_folder']}_{row['img_name']}"
            dst_lbl = OUTPUT_ROOT / "labels" / split / f"{row['img_folder']}_{row['img_name'].replace('.JPG', '.txt')}"

            # Copy image
            shutil.copy2(src_img, dst_img)

            # Write YOLO label
            xy = np.array(row["xy"])
            dart_tips = xy[4:]  # skip first 4 calibration points

            lines = []
            for tip in dart_tips:
                cx, cy = float(tip[0]), float(tip[1])
                w = h = TIP_BOX_SIZE
                # YOLO keypoint format: class cx cy w h kp_x kp_y kp_visible
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cx:.6f} {cy:.6f} 2")

            dst_lbl.write_text("\n".join(lines))

    # Write dataset.yaml
    yaml_content = f"""# DeepDarts dataset — converted for YOLOv11 keypoint training
path: {OUTPUT_ROOT.absolute()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: dart

# Keypoints: 1 per dart (the tip)
kpt_shape: [1, 3]  # [num_keypoints, (x, y, visible)]
"""
    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nDataset YAML written to: {yaml_path}")

    print("\n=== Conversion complete ===")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Run training with:")
    print(f"  uv run python scripts/train.py")


if __name__ == "__main__":
    convert()
