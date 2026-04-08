"""
utils.py — Stateless helper functions used by the tracking pipeline.
"""

from __future__ import annotations
from collections import deque

import cv2
import numpy as np
import supervision as sv

from src.config import MIN_BOX_AREA, STATIONARY_FRAMES, STATIONARY_PIXELS


# ── Detection helpers ──────────────────────────────────────────────────────────

def filter_by_area(det: sv.Detections, min_area: float = MIN_BOX_AREA) -> sv.Detections:
    """Remove detections whose bounding-box area is below *min_area* pixels²."""
    w = det.xyxy[:, 2] - det.xyxy[:, 0]
    h = det.xyxy[:, 3] - det.xyxy[:, 1]
    return det[w * h >= min_area]


# ── Stationary-track filter ────────────────────────────────────────────────────

def is_stationary(history: deque) -> bool:
    """
    Return True when a track's centroid has barely moved over its entire history.
    Only fires once *STATIONARY_FRAMES* observations have been collected so that
    briefly-visible players are not accidentally suppressed.
    """
    if len(history) < STATIONARY_FRAMES:
        return False
    pts = np.array(history)
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    return x_range < STATIONARY_PIXELS and y_range < STATIONARY_PIXELS


# ── Annotation helpers ─────────────────────────────────────────────────────────

def id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, visually distinct BGR colour per track ID."""
    rng = np.random.default_rng(int(track_id) * 7)
    return tuple(int(c) for c in rng.integers(80, 230, 3))


def draw_track(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    track_id: int,
    conf: float | None = None,
) -> None:
    """
    Draw a coloured bounding box and ID label (with filled background) on *frame*
    in-place.  Optionally appends the detection confidence score to the label.
    """
    x1, y1, x2, y2 = box
    color = id_color(track_id)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label text
    label = f"ID {track_id}" if conf is None else f"ID {track_id}  {conf:.2f}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)

    # Filled label background
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), font, scale, (255, 255, 255), thick)


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    active: int,
    total_ids: int,
    fps: float,
) -> None:
    """Overlay a HUD banner (frame counter, active tracks, total IDs, speed)."""
    h, w = frame.shape[:2]
    banner_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    text = (
        f"Frame {frame_idx:>5}  |  Active: {active:>3}  |"
        f"  Total IDs: {total_ids:>3}  |  {fps:>5.1f} FPS"
    )
    cv2.putText(frame, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
