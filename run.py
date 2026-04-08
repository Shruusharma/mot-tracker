"""
run.py — Command-line entry point for the MOT pipeline.

Examples
--------
# Use defaults from src/config.py
python run.py

# Override input / output / device
python run.py --video path/to/input.mp4 --output path/to/output.mp4 --device cuda

# Quiet mode (no per-frame log)
python run.py --quiet
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-Object Tracking — YOLOv8m + ByteTrack"
    )
    p.add_argument("--video",  type=str, default=None, help="Path to input video")
    p.add_argument("--output", type=str, default=None, help="Path for annotated output video")
    p.add_argument("--model",  type=str, default=None, help="YOLO model path or name (default: yolov8m.pt)")
    p.add_argument("--device", type=str, default="cpu", help="Inference device: cpu | cuda | mps")
    p.add_argument("--quiet",  action="store_true",    help="Suppress per-frame logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy import so the module is importable even before deps are installed
    from src.tracker import MOTPipeline
    import src.config as CFG

    kwargs: dict = {"device": args.device}
    if args.video:
        kwargs["video_path"]  = args.video
    if args.output:
        kwargs["output_path"] = args.output
    if args.model:
        kwargs["model_path"]  = args.model

    pipeline = MOTPipeline(**kwargs)
    stats    = pipeline.run(verbose=not args.quiet)

    print("\nSummary")
    print("─" * 30)
    for key, val in stats.items():
        print(f"  {key:<18}: {val}")


if __name__ == "__main__":
    main()
