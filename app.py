"""
app.py — Streamlit UI for the MOT pipeline.

Deploy to Streamlit Cloud:
  1. Push this repo to GitHub
  2. Go to share.streamlit.io → New app → select this repo
  3. Set Main file path: app.py
  4. Deploy

Key design decisions for Streamlit Cloud compatibility
-------------------------------------------------------
* All file I/O happens in /tmp/ (the repo mount is read-only).
* NamedTemporaryFile(delete=False) is used for the uploaded video so the
  path remains valid during the long pipeline.run() call.
* st.video() receives the /tmp/ output path as bytes — no disk-to-browser
  transfer issues.
"""

import os
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "MOT Pipeline — YOLOv8m + ByteTrack",
    page_icon   = "🎯",
    layout      = "wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Multi-Object Detection & Persistent ID Tracking")
st.markdown(
    "Upload a sports video and the pipeline will detect all persons, "
    "assign each one a **persistent unique ID**, and return the annotated video.  \n"
    "**Stack:** YOLOv8m · ByteTrack · OpenCV · supervision"
)

# ── Sidebar — config overrides ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Config")
    conf_thresh   = st.slider("Detection confidence",    0.10, 0.80, 0.35, 0.05)
    track_thresh  = st.slider("Track activation thresh", 0.30, 0.90, 0.65, 0.05)
    match_thresh  = st.slider("IoU match threshold",     0.30, 0.95, 0.75, 0.05)
    buffer_size   = st.slider("Lost-track buffer (frames)", 10, 120, 60, 5)
    min_hits      = st.slider("Min hits before ID assigned", 1, 6, 3, 1)
    device        = st.selectbox("Inference device", ["cpu", "cuda", "mps"], index=0)

    st.divider()
    st.markdown(
        "**Tip:** Lowering *Track activation thresh* below 0.5 will increase "
        "unique IDs (more ghost tracks). Raising it reduces noise."
    )

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a video (MP4 / AVI / MOV — max ~200 MB recommended)",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded is None:
    st.info("👆 Upload a video to get started.")
    st.stop()

# ── Save upload to /tmp (the repo mount is read-only) ─────────────────────────
tmp_input = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
tmp_input.write(uploaded.read())
tmp_input.flush()
tmp_input.close()
input_path = tmp_input.name

# Show a preview of the original
col1, col2 = st.columns(2)
with col1:
    st.subheader("📹 Original")
    st.video(input_path)

# ── Run pipeline ──────────────────────────────────────────────────────────────
with col2:
    st.subheader("🎯 Annotated Output")
    run_btn = st.button("▶ Run Tracker", type="primary", use_container_width=True)

if run_btn:
    # Apply sidebar overrides to config BEFORE importing the pipeline
    import src.config as CFG
    CFG.CONF_THRESH  = conf_thresh
    CFG.TRACK_THRESH = track_thresh
    CFG.MATCH_THRESH = match_thresh
    CFG.BUFFER_SIZE  = buffer_size
    CFG.MIN_HITS     = min_hits

    # Output always goes to /tmp/
    tmp_output = Path(tempfile.mkdtemp()) / "output.mp4"

    progress = st.progress(0, text="Loading model…")
    status   = st.empty()

    try:
        from src.tracker import MOTPipeline

        @st.cache_resource
        def load_model():
            from ultralytics import YOLO
            return YOLO(CFG.MODEL_PATH)

        pipeline = MOTPipeline(
            video_path  = input_path,
            output_path = str(tmp_output),
            device      = device,
        )
        # Patch the cached model in so we don't re-download on every run
        pipeline.model = load_model()

        total = pipeline.total

        # ── Frame-level loop so we can update the progress bar ──────────────
        import supervision as sv
        from src.utils import draw_hud, draw_track, filter_by_area, is_stationary
        from src.tracker import _open_writer
        from collections import deque

        tracker = sv.ByteTrack(
            track_activation_threshold = CFG.TRACK_THRESH,
            lost_track_buffer          = CFG.BUFFER_SIZE,
            minimum_matching_threshold = CFG.MATCH_THRESH,
            minimum_consecutive_frames = CFG.MIN_HITS,
            frame_rate                 = int(pipeline.fps),
        )

        cap    = cv2.VideoCapture(input_path)
        writer = _open_writer(
            str(tmp_output), pipeline.fps, (pipeline.width, pipeline.height)
        )

        centroid_history: dict = {}
        all_ids: set           = set()
        frame_idx              = 0
        start                  = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = pipeline.detect(frame)
            tracked    = tracker.update_with_detections(detections)

            if tracked.tracker_id is not None:
                for box, tid in zip(tracked.xyxy, tracked.tracker_id):
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    if tid not in centroid_history:
                        centroid_history[tid] = deque(maxlen=CFG.STATIONARY_FRAMES)
                    centroid_history[tid].append((cx, cy))

            annotated = frame.copy()
            active    = 0
            if tracked.tracker_id is not None:
                for box, tid, conf in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
                    if is_stationary(centroid_history.get(int(tid), deque())):
                        continue
                    all_ids.add(int(tid))
                    active += 1
                    draw_track(annotated, tuple(map(int, box)), int(tid), float(conf))

            elapsed  = max(time.time() - start, 1e-6)
            fps_live = (frame_idx + 1) / elapsed
            draw_hud(annotated, frame_idx, active, len(all_ids), fps_live)
            writer.write(annotated)

            frame_idx += 1
            pct = int(frame_idx / max(total, 1) * 100)
            if frame_idx % 10 == 0:
                progress.progress(
                    pct,
                    text=f"Frame {frame_idx}/{total}  |  Active: {active}  |"
                         f"  IDs: {len(all_ids)}  |  {fps_live:.1f} FPS",
                )

        cap.release()
        writer.release()
        progress.progress(100, text="✅ Done!")

        elapsed = time.time() - start

        # ── Results ─────────────────────────────────────────────────────────
        with col2:
            st.video(str(tmp_output))

        st.success(
            f"**Done!** {frame_idx} frames in {elapsed:.0f}s "
            f"({frame_idx/elapsed:.1f} FPS avg) — "
            f"**{len(all_ids)} unique active IDs** detected."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Frames",  frame_idx)
        m2.metric("Unique IDs",    len(all_ids))
        m3.metric("Avg FPS",       f"{frame_idx/elapsed:.1f}")
        m4.metric("Elapsed",       f"{elapsed:.0f}s")

        # Download button
        with open(str(tmp_output), "rb") as f:
            st.download_button(
                "⬇️ Download annotated video",
                data        = f,
                file_name   = "tracked_output.mp4",
                mime        = "video/mp4",
                use_container_width = True,
            )

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        raise

    finally:
        # Clean up the uploaded temp file
        try:
            os.unlink(input_path)
        except OSError:
            pass
