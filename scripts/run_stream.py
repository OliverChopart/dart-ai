"""Live dart detection pipeline with perspective correction.

Streams video from any OpenCV-compatible source (iPhone via DroidCam/EpocCam,
USB webcam, RTSP camera, etc.), applies the saved homography, runs YOLO
detection on every frame, and displays the result in a window.

Typical usage
-------------
# iPhone via DroidCam (USB or Wi-Fi):
    python scripts/run_stream.py --source http://192.168.1.42:4747/video

# iPhone via EpocCam (appears as webcam index 1 or 2):
    python scripts/run_stream.py --source 1

# Built-in webcam, no homography:
    python scripts/run_stream.py --source 0 --no-homography

How to stream from iPhone
-------------------------
Option A - DroidCam (free, Wi-Fi or USB):
  1. Install 'DroidCam' on iPhone and 'DroidCam Client' on Mac.
  2. Connect and note the URL shown in the app, e.g. http://192.168.1.x:4747/video
  3. Pass that URL as --source.

Option B - EpocCam (Elgato, better quality):
  1. Install EpocCam on iPhone and EpocCam drivers on Mac.
  2. The iPhone appears as a virtual webcam (try --source 1 or --source 2).

Option C - Continuity Camera (macOS 13+, no extra app needed):
  1. Lock your iPhone near the Mac - it appears automatically as a webcam.
  2. Try --source 1 or --source 2.

Press Q in the window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from config.settings import settings
from vision.stream import VideoStream


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_stream",
        description="Live dart detection with perspective correction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", "-s",
        default=settings.camera_source,
        help="Camera index, RTSP URL, or HTTP MJPEG URL (default: settings.camera_source).",
    )
    parser.add_argument(
        "--homography", "-H",
        default="config/homography.npy",
        metavar="PATH",
        help="Path to the homography .npy file (default: config/homography.npy).",
    )
    parser.add_argument(
        "--no-homography",
        action="store_true",
        help="Disable perspective correction even if a homography file exists.",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Skip YOLO detection (useful for checking the stream/homography only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # Start the video stream
    # ------------------------------------------------------------------
    stream = VideoStream(
        source=str(args.source),
        apply_homography=not args.no_homography,
        homography_path=args.homography,
    ).start()

    if stream.homography_active:
        print("[run_stream] Perspective correction: ON")
    else:
        print("[run_stream] Perspective correction: OFF (run scripts/run_calibration.py first)")

    # ------------------------------------------------------------------
    # Optionally load detector
    # ------------------------------------------------------------------
    detector = None
    if not args.no_detect:
        try:
            from vision.detector import DartDetector
            detector = DartDetector().load()
            print("[run_stream] YOLO detector: ON")
        except Exception as exc:
            print(f"[run_stream] YOLO detector could not be loaded ({exc}) - showing raw stream.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    print("[run_stream] Streaming... press Q in the window to quit.")
    fps_t0 = time.time()
    fps_count = 0

    cv2.namedWindow("Dart AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dart AI", 800, 800)

    while True:
        frame = stream.read(timeout=2.0)
        if frame is None:
            print("[run_stream] No frame received - check your source URL/index.")
            continue

        display = frame

        if detector is not None:
            try:
                result = detector.detect(frame, annotate=True)
                if result.annotated_frame is not None:
                    display = result.annotated_frame
                # Log detected dart tips to console
                for i, (tip, conf) in enumerate(zip(result.dart_tips, result.confidences)):
                    print(f"  dart {i+1}: ({tip[0]:.1f}, {tip[1]:.1f})  conf={conf:.2f}")
            except Exception as exc:
                print(f"[run_stream] Detection error: {exc}")

        # FPS overlay
        fps_count += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_t0 = time.time()
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Dart AI", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()
    print("[run_stream] Stopped.")


if __name__ == "__main__":
    main()
