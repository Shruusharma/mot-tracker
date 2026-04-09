"""
run.py — CLI entry point AND importable run_tracker() function for Streamlit.

CLI usage
---------
    python run.py --video input.mp4 --output /tmp/output.mp4 --device cpu

Streamlit / programmatic usage
--------------------------------
    from run import run_tracker
    output_path = run_tracker("/tmp/uploaded_video.mp4")
    # output_path is always under /tmp/ — safe on Streamlit Cloud
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def run_tracker(
    video_path: str,
    output_path: str | None = None,
    model_path: str = "yolov8m.pt",
    device: str = "cpu",
    verbose: bool = False,
) -> str:
    """
    Run the full MOT pipeline on *video_path* and return the path to the
    annotated output video.

    Parameters
    ----------
    video_path : str
        Path to the input video (e.g. a NamedTemporaryFile.name from Streamlit).
    output_path : str | None
        Where to write the annotated video.  Defaults to a new file under
        /tmp/ — always writable, even on Streamlit Cloud.
    model_path : str
        YOLO model weights.
    device : str
        "cpu", "cuda", or "mps".
    verbose : bool
        Print per-frame progress to stdout.

    Returns
    -------
    str
        Absolute path to the annotated output video.
    """
    # If caller didn't specify an output path, create one in /tmp so it's
    # writable on Streamlit Cloud (repo mount at /mount/src/ is read-only).
    if output_path is None:
        tmp_dir     = Path(tempfile.mkdtemp())
        output_path = str(tmp_dir / "output.mp4")

    from src.tracker import MOTPipeline

    pipeline = MOTPipeline(
        video_path  = video_path,
        output_path = output_path,
        model_path  = model_path,
        device      = device,
    )
    stats = pipeline.run(verbose=verbose)
    return stats["output_path"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Object Tracking — YOLOv8m + ByteTrack")
    p.add_argument("--video",  default=None, help="Input video path")
    p.add_argument("--output", default=None, help="Output video path (default: /tmp/output.mp4)")
    p.add_argument("--model",  default="yolov8m.pt", help="YOLO model")
    p.add_argument("--device", default="cpu",        help="cpu | cuda | mps")
    p.add_argument("--quiet",  action="store_true",  help="Suppress per-frame logging")
    return p.parse_args()


def main() -> None:
    import src.config as CFG
    args = _parse_args()

    out = run_tracker(
        video_path  = args.video or CFG.VIDEO_PATH,
        output_path = args.output,
        model_path  = args.model,
        device      = args.device,
        verbose     = not args.quiet,
    )
    print(f"\nOutput saved to: {out}")


if __name__ == "__main__":
    main()
