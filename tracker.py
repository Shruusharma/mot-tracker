"""
tracker.py — Multi-Object Tracking pipeline.

Pipeline:
  Frame → YOLOv8m detection → area filter → ByteTrack → stationary filter
       → annotated frame → VideoWriter

Usage (as a module):
    from src.tracker import MOTPipeline
    pipe = MOTPipeline(video_path="input.mp4")
    stats = pipe.run()

Usage (CLI):
    python run.py --video input.mp4

Streamlit Cloud notes
---------------------
* /mount/src/ is READ-ONLY — never write output there.
* output_path always defaults to a fresh /tmp/ directory.
* VideoWriter codec falls back through avc1 → mp4v → XVID automatically.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from src import config as CFG
from src.utils import draw_hud, draw_track, filter_by_area, is_stationary

# ---------------------------------------------------------------------------
# Codec fallback list
# avc1 (H.264) works on most Linux systems and plays in browsers.
# mp4v is the universal fallback. XVID produces .avi-compatible streams.
# ---------------------------------------------------------------------------
_CODEC_PRIORITY = ["avc1", "mp4v", "XVID"]


def _open_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Try codecs in priority order until one opens successfully."""
    for codec in _CODEC_PRIORITY:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"cv2.VideoWriter could not open '{path}' with any of {_CODEC_PRIORITY}.\n"
        "Ensure the output path is inside a writable directory (e.g. /tmp/)."
    )


class MOTPipeline:
    """
    End-to-end detection + persistent-ID tracking pipeline.

    Parameters
    ----------
    video_path : str | Path
        Source video file.
    output_path : str | Path | None
        Destination for the annotated video.
        Defaults to a fresh file under /tmp/ — always writable, even on
        Streamlit Cloud where the repo mount is read-only.
    model_path : str
        YOLO weights file or Ultralytics model name.
    device : str
        Inference device: "cpu", "cuda", or "mps".
    """

    def __init__(
        self,
        video_path: str | Path = CFG.VIDEO_PATH,
        output_path: str | Path | None = None,
        model_path: str = CFG.MODEL_PATH,
        device: str = "cpu",
    ):
        self.video_path = Path(video_path)

        # Always write to /tmp so the code works on Streamlit Cloud (read-only
        # repo mount) as well as locally.
        if output_path is None:
            tmp_dir = Path(tempfile.mkdtemp())
            self.output_path = tmp_dir / "output.mp4"
        else:
            self.output_path = Path(output_path)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.device = device

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # ── Video metadata ────────────────────────────────────────────────────
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = YOLO(model_path)

        # ── Tracker ───────────────────────────────────────────────────────────
        self.tracker = sv.ByteTrack(
            track_activation_threshold = CFG.TRACK_THRESH,
            lost_track_buffer          = CFG.BUFFER_SIZE,
            minimum_matching_threshold = CFG.MATCH_THRESH,
            minimum_consecutive_frames = CFG.MIN_HITS,
            frame_rate                 = int(self.fps),
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.centroid_history: dict[int, deque] = {}
        self.all_ids: set[int] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run YOLO on *frame*, apply area filter, return filtered Detections."""
        results    = self.model(frame, conf=CFG.CONF_THRESH, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        return filter_by_area(detections, CFG.MIN_BOX_AREA)

    def update_centroid_history(self, tracked: sv.Detections) -> None:
        """Append current centroids to per-track history deque."""
        if tracked.tracker_id is None:
            return
        for box, tid in zip(tracked.xyxy, tracked.tracker_id):
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            if tid not in self.centroid_history:
                self.centroid_history[tid] = deque(maxlen=CFG.STATIONARY_FRAMES)
            self.centroid_history[tid].append((cx, cy))

    def annotate(self, frame: np.ndarray, tracked: sv.Detections, frame_idx: int) -> np.ndarray:
        """Draw bounding boxes, IDs, and HUD onto a copy of *frame*."""
        annotated = frame.copy()
        active = 0

        if tracked.tracker_id is not None:
            for box, tid, conf in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
                history = self.centroid_history.get(int(tid), deque())
                if is_stationary(history):
                    continue  # suppress non-moving spectators

                self.all_ids.add(int(tid))
                active += 1
                draw_track(annotated, tuple(map(int, box)), int(tid), float(conf))

        elapsed  = max(time.time() - self._start, 1e-6)
        fps_live = (frame_idx + 1) / elapsed
        draw_hud(annotated, frame_idx, active, len(self.all_ids), fps_live)
        return annotated

    def run(self, verbose: bool = True) -> dict:
        """
        Process the full video and write the annotated output.

        Returns
        -------
        dict
            ``total_frames``, ``unique_ids``, ``elapsed_s``, ``avg_fps``,
            ``output_path`` (string — ready to pass to st.video()).
        """
        cap    = cv2.VideoCapture(str(self.video_path))
        writer = _open_writer(str(self.output_path), self.fps, (self.width, self.height))

        self._start = time.time()
        frame_idx   = 0

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Input  : {self.video_path}")
            print(f"  Output : {self.output_path}")
            print(f"  Video  : {self.width}×{self.height} @ {self.fps:.0f} fps  ({self.total} frames)")
            print(f"  Device : {self.device}")
            print(f"{'─'*60}")
            print(f"{'Frame':<10}{'Dets':<8}{'Active':<10}{'Total IDs':<12}{'FPS'}")
            print("─" * 50, flush=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect(frame)
            tracked    = self.tracker.update_with_detections(detections)
            self.update_centroid_history(tracked)
            annotated  = self.annotate(frame, tracked, frame_idx)
            writer.write(annotated)

            if verbose and frame_idx % 30 == 0:
                elapsed  = max(time.time() - self._start, 1e-6)
                fps_live = (frame_idx + 1) / elapsed
                eta_s    = (self.total - frame_idx) / fps_live
                active   = sum(
                    1 for tid in (tracked.tracker_id or [])
                    if not is_stationary(self.centroid_history.get(int(tid), deque()))
                )
                print(
                    f"{frame_idx:<10}{len(detections):<8}{active:<10}"
                    f"{len(self.all_ids):<12}{fps_live:<8.1f}"
                    f"ETA {eta_s:.0f}s",
                    flush=True,
                )

            frame_idx += 1

        cap.release()
        writer.release()

        elapsed = time.time() - self._start
        avg_fps = frame_idx / max(elapsed, 1e-6)

        stats = {
            "total_frames": frame_idx,
            "unique_ids":   len(self.all_ids),
            "elapsed_s":    round(elapsed, 1),
            "avg_fps":      round(avg_fps, 2),
            "output_path":  str(self.output_path),
        }

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Done!  {frame_idx} frames in {elapsed:.0f}s  ({avg_fps:.1f} FPS avg)")
            print(f"  Unique IDs (active players) : {len(self.all_ids)}")
            print(f"  Output : {self.output_path}")
            print(f"{'─'*60}\n")

        return stats
