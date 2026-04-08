"""
config.py — All pipeline parameters in one place.
Edit this file to change behaviour without touching pipeline code.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
VIDEO_PATH  = os.path.join(os.getcwd(), "input.mp4")
OUTPUT_PATH = os.path.join(os.getcwd(), "output.mp4")
MODEL_PATH  = "yolov8m.pt"

# ── Detection ─────────────────────────────────────────────────────────────────
CONF_THRESH  = 0.35    # Min detection confidence (below → ignored entirely)
MIN_BOX_AREA = 1500    # Min bounding-box area in px² (strips partial bodies at edges)

# ── ByteTrack ─────────────────────────────────────────────────────────────────
TRACK_THRESH = 0.65    # Min conf to SPAWN a new track ID
                       # Re-entering players at low conf won't create ghost IDs
MATCH_THRESH = 0.75    # IoU association threshold (higher = stricter matching)
BUFFER_SIZE  = 60      # Frames to keep a lost track alive (2 s @ 30 fps)
MIN_HITS     = 3       # Frames a detection must appear before receiving an ID

# ── Stationary Person Filter ───────────────────────────────────────────────────
# Spectators / crowds in stands are detected but never move.
# Tracks whose centroid has not moved more than STATIONARY_PIXELS over
# STATIONARY_FRAMES frames are suppressed from the annotated output.
STATIONARY_FRAMES = 300   # 10 s @ 30 fps
STATIONARY_PIXELS = 20    # Centroid drift threshold (px)
