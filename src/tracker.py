"""
tracker.py — Multi-Object Tracking pipeline.

Pipeline:
  Frame → YOLOv8m detection → area filter → ByteTrack → stationary filter
       → annotated frame → VideoWriter

Usage (as a module):
    from src.tracker import MOTPipeline
    pipe = MOTPipeline()
    pipe.run()

Usage (CLI):
    python run.py --video input.mp4 --output output.mp4
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from src import config as CFG
from src.utils import draw_hud, draw_track, filter_by_area, is_stationary


class MOTPipeline:
    """
    End-to-end detection + persistent-ID tracking pipeline.

    Parameters
    ----------
    video_path : str | Path
        Source video.  Defaults to ``config.VIDEO_PATH``.
    output_path : str | Path
        Where the annotated video is written.  Defaults to ``config.OUTPUT_PATH``.
    model_path : str
        YOLO weights file / Ultralytics model name.  Defaults to ``config.MODEL_PATH``.
    device : str
        Inference device, e.g. ``"cpu"``, ``"cuda"``, ``"mps"``.
    """

    def __init__(
        self,
        video_path: str | Path = CFG.VIDEO_PATH,
        output_path: str | Path = CFG.OUTPUT_PATH,
        model_path: str = CFG.MODEL_PATH,
        device: str = "cpu",
    ):
        self.video_path  = Path(video_path)
        self.output_path = Path(output_path)
        self.device      = device

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # ── Video metadata ───────────────────────────────────────────────────
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps    = cap.get(cv2.CAP_PROP_FPS)
        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ── Model ────────────────────────────────────────────────────────────
        self.model = YOLO(model_path)

        # ── Tracker ──────────────────────────────────────────────────────────
        self.tracker = sv.ByteTrack(
            track_activation_threshold = CFG.TRACK_THRESH,
            lost_track_buffer          = CFG.BUFFER_SIZE,
            minimum_matching_threshold = CFG.MATCH_THRESH,
            minimum_consecutive_frames = CFG.MIN_HITS,
            frame_rate                 = int(self.fps),
        )

        # ── State ────────────────────────────────────────────────────────────
        # centroid_history maps track_id → deque of (cx, cy) observations
        self.centroid_history: dict[int, deque] = {}
        self.all_ids: set[int] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run YOLO on *frame*, apply area filter, return filtered Detections."""
        results    = self.model(frame, conf=CFG.CONF_THRESH, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        return filter_by_area(detections, CFG.MIN_BOX_AREA)

    def update_centroid_history(self, tracked: sv.Detections) -> None:
        """Append current centroids to per-track history."""
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
                    continue   # suppress non-moving spectators

                self.all_ids.add(int(tid))
                active += 1
                draw_track(annotated, tuple(map(int, box)), int(tid), float(conf))

        elapsed = max(time.time() - self._start, 1e-6)
        fps_live = (frame_idx + 1) / elapsed
        draw_hud(annotated, frame_idx, active, len(self.all_ids), fps_live)
        return annotated

    def run(self, verbose: bool = True) -> dict:
        """
        Process the full video and write the annotated output.

        Returns
        -------
        dict with keys: total_frames, unique_ids, elapsed_s, avg_fps
        """
        cap    = cv2.VideoCapture(str(self.video_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height))

        if not writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for: {self.output_path}")

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

            annotated = self.annotate(frame, tracked, frame_idx)
            writer.write(annotated)

            if verbose and frame_idx % 30 == 0:
                active = (
                    sum(
                        1 for tid in (tracked.tracker_id or [])
                        if not is_stationary(self.centroid_history.get(int(tid), deque()))
                    )
                )
                elapsed = max(time.time() - self._start, 1e-6)
                fps_live = (frame_idx + 1) / elapsed
                eta_s    = (self.total - frame_idx) / fps_live
                print(
                    f"{frame_idx:<10}{len(detections):<8}{active:<10}"
                    f"{len(self.all_ids):<12}{fps_live:<8.1f}"
                    f"ETA {eta_s:.0f}s",
                    flush=True,
                )

            frame_idx += 1

        cap.release()
        writer.release()

        elapsed   = time.time() - self._start
        avg_fps   = frame_idx / max(elapsed, 1e-6)
        stats = {
            "total_frames": frame_idx,
            "unique_ids":   len(self.all_ids),
            "elapsed_s":    round(elapsed, 1),
            "avg_fps":      round(avg_fps, 2),
        }

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Done!  {frame_idx} frames in {elapsed:.0f}s  ({avg_fps:.1f} FPS avg)")
            print(f"  Unique IDs (active players) : {len(self.all_ids)}")
            print(f"  Output : {self.output_path}")
            print(f"{'─'*60}\n")

        return stats
