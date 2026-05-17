"""Convert DeepDarts labels.pkl to 5-class YOLO detection format.

McNally's labels.pkl contains per-image:
  - xy[0]  top    calibration point (double-20 area, 12 o'clock)
  - xy[1]  right  calibration point (double-6 area)
  - xy[2]  bottom calibration point (double-3 area)
  - xy[3]  left   calibration point (double-11 area)
  - xy[4+] dart tip coordinates (0-3 darts)
  All in normalised coordinates (0.0-1.0) relative to the 800x800 cropped image.

Output format — one .txt file per image, one row per object:
  class cx cy w h
  class 0 = dart
  class 1 = cal_20  (top,    double-20)
  class 2 = cal_6   (right,  double-6)
  class 3 = cal_3   (bottom, double-3)
  class 4 = cal_11  (left,   double-11)

Calibration points are encoded as small bounding boxes (CAL_BOX_SIZE)
centred on the detected corner coordinate.

Usage:
    uv run python scripts/convert_dataset_5class.py

Output:
    dataset/yolo_5class/
        images/train|val|test/  (symlinks or copies)
        labels/train|val|test/  (.txt files)
        dataset.yaml
"""

import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("dataset")
IMAGES_ROOT = DATASET_ROOT / "cropped_images" / "800"
LABELS_PKL = DATASET_ROOT / "labels.pkl"
OUTPUT_ROOT = DATASET_ROOT / "yolo_5class"

# ---------------------------------------------------------------------------
# Bounding box sizes (normalised, relative to 800x800 image)
# ---------------------------------------------------------------------------
TIP_BOX_SIZE = 0.05   # 40px box around each dart tip
CAL_BOX_SIZE = 0.04   # 32px box around each calibration corner

# ---------------------------------------------------------------------------
# Calibration point index -> (class_id, class_name)
# McNally's order: 0=top(20), 1=right(6), 2=bottom(3), 3=left(11)
# ---------------------------------------------------------------------------
CAL_CLASSES = [
    (1, "cal_20"),   # xy[0] top
    (2, "cal_6"),    # xy[1] right
    (3, "cal_3"),    # xy[2] bottom
    (4, "cal_11"),   # xy[3] left
]


def convert() -> None:
    if not LABELS_PKL.exists():
        print(f"ERROR: {LABELS_PKL} not found.")
        print("Download the DeepDarts dataset and place labels.pkl in dataset/")
        return

    print("Loading labels.pkl...")
    df = pickle.load(open(LABELS_PKL, "rb"))
    print(f"Total annotations: {len(df)}")

    # ------------------------------------------------------------------
    # Filter to images that exist on disk
    # ------------------------------------------------------------------
    valid_rows = []
    missing = 0
    for _, row in df.iterrows():
        img_path = IMAGES_ROOT / row["img_folder"] / row["img_name"]
        if img_path.exists():
            valid_rows.append(row)
        else:
            missing += 1

    print(f"Valid images: {len(valid_rows)} (missing on disk: {missing})")
    df_valid = pd.DataFrame(valid_rows)

    # ------------------------------------------------------------------
    # Filter to images that have at least 1 dart tip AND all 4 cal points
    # ------------------------------------------------------------------
    df_valid["xy_arr"] = df_valid["xy"].apply(np.array)
    df_valid["n_darts"] = df_valid["xy_arr"].apply(lambda x: max(0, len(x) - 4))
    df_valid["n_cal"] = df_valid["xy_arr"].apply(lambda x: min(4, len(x)))

    df_usable = df_valid[(df_valid["n_darts"] > 0) & (df_valid["n_cal"] == 4)].copy()
    print(f"Usable images (>=1 dart + 4 cal points): {len(df_usable)}")

    # ------------------------------------------------------------------
    # Train / val / test split  70 / 15 / 15
    # ------------------------------------------------------------------
    train_df, temp_df = train_test_split(df_usable, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    print(f"Split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ------------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------------
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Convert each split
    # ------------------------------------------------------------------
    for split, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"Converting {split} ({len(split_df)} images)...")
        for _, row in split_df.iterrows():
            img_name = f"{row['img_folder']}_{row['img_name']}"
            lbl_name = img_name.replace(".JPG", ".txt").replace(".jpg", ".txt")

            src_img = IMAGES_ROOT / row["img_folder"] / row["img_name"]
            dst_img = OUTPUT_ROOT / "images" / split / img_name
            dst_lbl = OUTPUT_ROOT / "labels" / split / lbl_name

            shutil.copy2(src_img, dst_img)

            xy = np.array(row["xy"])
            lines = []

            # 4 calibration points (indices 0-3)
            for cal_idx, (cls_id, _) in enumerate(CAL_CLASSES):
                cx, cy = float(xy[cal_idx][0]), float(xy[cal_idx][1])
                w = h = CAL_BOX_SIZE
                # Clamp to valid range
                cx = max(w / 2, min(1 - w / 2, cx))
                cy = max(h / 2, min(1 - h / 2, cy))
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Dart tips (indices 4+)
            dart_tips = xy[4:]
            for tip in dart_tips:
                cx, cy = float(tip[0]), float(tip[1])
                w = h = TIP_BOX_SIZE
                cx = max(w / 2, min(1 - w / 2, cx))
                cy = max(h / 2, min(1 - h / 2, cy))
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            dst_lbl.write_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Write dataset.yaml
    # ------------------------------------------------------------------
    yaml_content = f"""# DeepDarts dataset — 5-class YOLO detection format
# Class 0: dart      — dart tip (bbox centre)
# Class 1: cal_20    — upper-left corner of double-20 segment
# Class 2: cal_6     — upper-left corner of double-6 segment
# Class 3: cal_3     — upper-left corner of double-3 segment
# Class 4: cal_11    — upper-left corner of double-11 segment

path: {OUTPUT_ROOT.absolute()}
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
    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\ndataset.yaml written to: {yaml_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Conversion complete ===")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Total images: {len(df_usable)}")
    print("\nNext step — train the 5-class model:")
    print("  uv run python scripts/train_5class.py")


if __name__ == "__main__":
    convert()
