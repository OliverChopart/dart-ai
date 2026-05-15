"""CLI entry-point for the dart board homography calibration tool.

Examples
--------
Test mode (still image from the dataset, no camera needed)::

    python scripts/run_calibration.py --image dataset/cropped_images/800/d1_02_04_2020/IMG_1322.JPG

Live camera mode::

    python scripts/run_calibration.py --camera 0

Custom output path and resolution::

    python scripts/run_calibration.py \\
        --image dataset/cropped_images/800/d1_02_04_2020/IMG_1322.JPG \\
        --output config/homography.npy \\
        --size 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.calibration import (
    DEFAULT_HOMOGRAPHY_PATH,
    DEFAULT_OUTPUT_SIZE,
    run_calibration,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_calibration",
        description=(
            "Interactive homography calibration for the dart board camera.\n"
            "Click 4 points on the double-ring (top, right, bottom, left) "
            "to compute and save the perspective-correction matrix."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image", "-i",
        metavar="PATH",
        default="dataset/cropped_images/800/d1_02_04_2020/IMG_1322.JPG",
        help="Path to a still image (default: sample dataset image).",
    )
    source.add_argument(
        "--camera", "-c",
        metavar="INDEX",
        type=int,
        default=None,
        help="OpenCV camera index to use (overrides --image).",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=str(DEFAULT_HOMOGRAPHY_PATH),
        help=f"Where to write the homography .npy file (default: {DEFAULT_HOMOGRAPHY_PATH}).",
    )
    parser.add_argument(
        "--size", "-s",
        metavar="PIXELS",
        type=int,
        default=DEFAULT_OUTPUT_SIZE,
        help=f"Side length of the canonical output square (default: {DEFAULT_OUTPUT_SIZE}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.camera is not None:
            H = run_calibration(
                image_path=None,
                camera_index=args.camera,
                output_path=args.output,
                output_size=args.size,
            )
        else:
            H = run_calibration(
                image_path=args.image,
                output_path=args.output,
                output_size=args.size,
            )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[run_calibration] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[run_calibration] Calibration complete.")
    print("[run_calibration] Homography matrix:")
    for row in H:
        print("  ", "  ".join(f"{v:10.5f}" for v in row))


if __name__ == "__main__":
    main()
