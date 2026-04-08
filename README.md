# Multi-Object Detection & Persistent ID Tracking

A computer vision pipeline that detects all persons in a sports video and assigns each one a **persistent unique ID** that remains consistent across the full video, even under occlusion, motion blur, rapid movement, and camera panning.

**Stack:** YOLOv8m (detection) · ByteTrack (tracking) · OpenCV (I/O) · supervision (annotation)

---

## Demo

> 📹 **Source video:** _[Add your public video URL here]_  
> 🎬 **Output video:** _[Add your output/demo link here — e.g. Google Drive, YouTube unlisted]_

Sample annotated frame:

![Sample frame](assets/sample_frame.png)

---

## Pipeline

```
Input Video
    │
    ▼
YOLOv8m Detection  ──► Area Filter (< 1500 px² dropped)
    │
    ▼
ByteTrack Association
  · Two-stage IoU matching
  · Lost-track buffer: 60 frames
  · Track activation threshold: 0.65
    │
    ▼
Stationary Person Filter  ──► Suppresses crowd / spectators
    │
    ▼
Annotated Output Video
```

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/<your-username>/mot-tracker.git
cd mot-tracker
pip install -r requirements.txt
```

---

## Usage

### Option A — Command line

```bash
# Place your video as input.mp4 in the project root, then:
python run.py

# Or specify paths explicitly:
python run.py --video path/to/video.mp4 --output path/to/output.mp4

# GPU inference (much faster):
python run.py --device cuda
```

### Option B — Jupyter notebook

```bash
jupyter notebook notebooks/mot_pipeline.ipynb
```

Run all cells top to bottom. `output.mp4` is written to the working directory.

### Option C — Python API

```python
from src.tracker import MOTPipeline

pipe = MOTPipeline(
    video_path  = "input.mp4",
    output_path = "output.mp4",
    device      = "cpu",   # or "cuda"
)
stats = pipe.run()
print(stats)
# {'total_frames': 1800, 'unique_ids': 47, 'elapsed_s': 360.2, 'avg_fps': 5.0}
```

---

## Configuration

All parameters live in `src/config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `CONF_THRESH` | `0.35` | Min YOLO detection confidence |
| `MIN_BOX_AREA` | `1500` | Min bounding-box area in px² |
| `TRACK_THRESH` | `0.65` | Min confidence to spawn a new track ID |
| `MATCH_THRESH` | `0.75` | IoU threshold for track-detection association |
| `BUFFER_SIZE` | `60` | Frames a lost track stays alive (2 s @ 30 fps) |
| `MIN_HITS` | `3` | Consecutive frames before an ID is assigned |
| `STATIONARY_FRAMES` | `300` | History window for stationary filter (10 s) |
| `STATIONARY_PIXELS` | `20` | Max centroid drift to be classed as stationary |

---

## Repository Structure

```
mot-tracker/
├── src/
│   ├── config.py          # All tunable parameters
│   ├── tracker.py         # MOTPipeline class (detect → track → annotate)
│   └── utils.py           # Stateless helpers (filters, drawing, colors)
├── notebooks/
│   └── mot_pipeline.ipynb # Self-contained notebook (same pipeline)
├── assets/
│   └── sample_frame.png   # Screenshot for README
├── run.py                 # CLI entry point
├── requirements.txt
├── REPORT.md              # Technical report
└── README.md
```

---

## Results

| Metric | Value |
|---|---|
| Processing speed | ~5 FPS (CPU) · ~30 FPS (GPU) |
| Unique active IDs | ~47 (stationary filter applied) |
| Lost-track buffer | 60 frames (2 s) |

---

## Known Limitations

- **ID switches under heavy occlusion** — IoU-only association cannot re-identify players after a long overlap. OSNet ReID embeddings would fix this.
- **Same-jersey confusion** — identical uniforms confuse position-only trackers.
- **Camera pans** — fast pans cause IoU matching to fail; Global Motion Compensation (BoT-SORT) partially mitigates this.

See [`REPORT.md`](REPORT.md) for the full technical analysis.

---

## License

MIT
