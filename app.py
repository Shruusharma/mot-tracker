"""
Streamlit app — Multi-Object Tracking Sports Video
YOLOv8m + ByteTrack pipeline with analytics dashboard.
"""

import os
import tempfile
import time
import collections

import cv2
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOT — Sports Tracker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Multi-Object Tracking — Sports Video")
st.markdown(
    "Upload a sports video to detect and persistently track every player "
    "using **YOLOv8m + ByteTrack**. Includes analytics: trajectory maps, "
    "ID count over time, and stationary-viewer suppression."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def filter_by_area(det, min_area: int):
    areas = (det.xyxy[:, 2] - det.xyxy[:, 0]) * (det.xyxy[:, 3] - det.xyxy[:, 1])
    return det[areas >= min_area]


def id_color(track_id: int) -> tuple:
    rng = np.random.default_rng(int(track_id) * 7)
    return tuple(int(c) for c in rng.integers(80, 230, 3))


@st.cache_resource(show_spinner="Loading YOLOv8m …")
def load_model(model_name: str = "yolov8m.pt"):
    return YOLO(model_name)


def open_writer(path: str, fps: float, w: int, h: int):
    for fourcc_str in ("avc1", "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Cannot open VideoWriter at {path}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    st.subheader("Detection")
    conf_thresh  = st.slider("Confidence threshold", 0.10, 0.90, 0.30, 0.05)
    min_box_area = st.slider("Min box area (px²)", 100, 5000, 1000, 100)

    st.subheader("ByteTrack")
    track_thresh = st.slider("Track activation threshold", 0.10, 0.90, 0.65, 0.05)
    match_thresh = st.slider("IoU match threshold", 0.10, 0.99, 0.75, 0.05)
    buffer_size  = st.slider("Lost-track buffer (frames)", 10, 150, 60, 5)
    min_hits     = st.slider("Min consecutive frames for ID", 1, 10, 3, 1)

    st.subheader("Stationary filter")
    enable_stationary = st.toggle("Suppress stationary viewers", value=True)
    stationary_px     = st.slider("Movement threshold (px)", 5, 100, 20, 5,
                                  disabled=not enable_stationary)
    stationary_after  = st.slider("Apply after N frames", 50, 600, 300, 50,
                                  disabled=not enable_stationary)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your video (MP4 / AVI / MOV / MKV)",
    type=["mp4", "avi", "mov", "mkv"],
)

if not uploaded:
    st.info("👆 Upload a sports video to get started.")
    st.stop()

tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
tmp_in.write(uploaded.read())
tmp_in.flush()
VIDEO_PATH = tmp_in.name

cap   = cv2.VideoCapture(VIDEO_PATH)
FPS   = cap.get(cv2.CAP_PROP_FPS) or 30.0
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Resolution", f"{W}×{H}")
c2.metric("FPS", f"{FPS:.0f}")
c3.metric("Duration", f"{TOTAL/FPS:.1f}s")
c4.metric("Total frames", TOTAL)

# ── Detection preview ─────────────────────────────────────────────────────────
with st.expander("🔍 Detection preview (mid-video frame)"):
    if st.button("Run detection check"):
        model = load_model()
        cap2  = cv2.VideoCapture(VIDEO_PATH)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, TOTAL // 2)
        _, frame = cap2.read()
        cap2.release()
        results = model(frame, conf=conf_thresh, classes=[0], verbose=False)[0]
        dets    = sv.Detections.from_ultralytics(results)
        dets    = filter_by_area(dets, min_box_area)
        viz     = frame.copy()
        for box, conf in zip(dets.xyxy, dets.confidence):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 200, 80), 2)
            cv2.putText(viz, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1)
        st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB),
                 caption=f"{len(dets)} persons detected")

# ── Main run ──────────────────────────────────────────────────────────────────
st.divider()
run_btn = st.button("🚀 Run Tracking Pipeline", type="primary", use_container_width=True)

if not run_btn:
    st.stop()

model = load_model()
tracker = sv.ByteTrack(
    track_activation_threshold=track_thresh,
    lost_track_buffer=buffer_size,
    minimum_matching_threshold=match_thresh,
    minimum_consecutive_frames=min_hits,
    frame_rate=int(FPS),
)

tmp_dir    = tempfile.mkdtemp()
OUTPUT_PATH = os.path.join(tmp_dir, "tracked.mp4")
writer     = open_writer(OUTPUT_PATH, FPS, W, H)
cap        = cv2.VideoCapture(VIDEO_PATH)

progress_bar  = st.progress(0, text="Starting…")
stat_cols     = st.columns(4)
frame_m       = stat_cols[0].empty()
active_m      = stat_cols[1].empty()
total_m       = stat_cols[2].empty()
fps_m         = stat_cols[3].empty()

# Analytics collectors
centroid_history: dict[int, list] = {}
trajectory_map   = np.zeros((H, W, 3), dtype=np.uint8)
active_over_time: list[int]       = []
unique_over_time: list[int]       = []
frame_times:      list[float]     = []

all_ids  = set()
grey_ids = set()
frame_idx = 0
start     = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results  = model(frame, conf=conf_thresh, classes=[0], verbose=False)[0]
    dets     = sv.Detections.from_ultralytics(results)
    dets     = filter_by_area(dets, min_box_area)
    tracked  = tracker.update_with_detections(dets)

    if tracked.tracker_id is not None:
        all_ids.update(tracked.tracker_id.tolist())

        for box, tid in zip(tracked.xyxy, tracked.tracker_id):
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            centroid_history.setdefault(int(tid), []).append((cx, cy))

        # Build trajectory map
        for tid, pts in centroid_history.items():
            if len(pts) >= 2:
                color = id_color(tid)
                cv2.polylines(
                    trajectory_map,
                    [np.array(pts[-30:], dtype=np.int32)],  # last 30 pts
                    False, color, 1, cv2.LINE_AA,
                )

        # Stationary filter
        if enable_stationary and frame_idx >= stationary_after:
            for tid, pts in centroid_history.items():
                if len(pts) >= 10:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    drift = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
                    if drift < stationary_px:
                        grey_ids.add(tid)

    # Annotate frame
    annotated = frame.copy()
    # Blend trajectory
    mask = trajectory_map.sum(axis=2) > 0
    annotated[mask] = cv2.addWeighted(annotated, 0.6, trajectory_map, 0.4, 0)[mask]

    active = 0
    if tracked.tracker_id is not None:
        for box, tid in zip(tracked.xyxy, tracked.tracker_id):
            x1, y1, x2, y2 = map(int, box)
            is_viewer = enable_stationary and int(tid) in grey_ids
            color     = (160, 160, 160) if is_viewer else id_color(int(tid))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tid}" + (" ●" if is_viewer else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 7), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            if not is_viewer:
                active += 1

    cv2.putText(annotated,
                f"Frame {frame_idx}  Active: {active}  Total IDs: {len(all_ids)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 230), 2)

    writer.write(annotated)
    active_over_time.append(active)
    unique_over_time.append(len(all_ids))
    frame_times.append(frame_idx / FPS)
    frame_idx += 1

    if frame_idx % 10 == 0 or frame_idx == TOTAL:
        pct     = frame_idx / max(TOTAL, 1)
        elapsed = time.time() - start
        fps_now = frame_idx / max(elapsed, 1e-6)
        eta_s   = (TOTAL - frame_idx) / max(fps_now, 1e-6)
        progress_bar.progress(pct, text=f"Frame {frame_idx}/{TOTAL} — ETA {eta_s:.0f}s")
        frame_m.metric("Frame", f"{frame_idx}/{TOTAL}")
        active_m.metric("Active IDs", active)
        total_m.metric("Unique IDs", len(all_ids))
        fps_m.metric("Speed", f"{fps_now:.1f} FPS")

cap.release()
writer.release()
elapsed = time.time() - start

progress_bar.progress(1.0, text="✅ Done!")
st.success(
    f"Processed **{frame_idx} frames** in **{elapsed:.0f}s** "
    f"({frame_idx/elapsed:.1f} FPS) — **{len(all_ids)} unique IDs** total"
)

# ── Results tabs ──────────────────────────────────────────────────────────────
tab_video, tab_traj, tab_chart, tab_stats = st.tabs(
    ["📹 Output Video", "🗺️ Trajectory Map", "📈 Analytics", "📊 Stats"]
)

with tab_video:
    cap_out = cv2.VideoCapture(OUTPUT_PATH)
    cap_out.set(cv2.CAP_PROP_POS_FRAMES, frame_idx // 2)
    _, sample = cap_out.read()
    cap_out.release()
    if sample is not None:
        st.image(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB),
                 caption="Sample frame — mid-video")

    with open(OUTPUT_PATH, "rb") as f:
        st.download_button(
            "⬇️ Download tracked video",
            data=f.read(),
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_tracked.mp4",
            mime="video/mp4",
            use_container_width=True,
            type="primary",
        )

with tab_traj:
    st.subheader("Full-video trajectory map")
    # Build full trajectory from all collected history
    full_traj = np.zeros((H, W, 3), dtype=np.uint8)
    for tid, pts in centroid_history.items():
        if len(pts) < 2:
            continue
        color = (160, 160, 160) if tid in grey_ids else id_color(tid)
        cv2.polylines(full_traj, [np.array(pts, dtype=np.int32)],
                      False, color, 2, cv2.LINE_AA)
        # Mark start (circle) and end (square)
        cv2.circle(full_traj, pts[0], 5, color, -1)
        cv2.circle(full_traj, pts[-1], 4, (255, 255, 255), -1)

    # Overlay on a dark background with the first video frame
    cap_bg = cv2.VideoCapture(VIDEO_PATH)
    _, bg   = cap_bg.read()
    cap_bg.release()
    if bg is not None:
        bg_dark = (bg.astype(np.float32) * 0.25).astype(np.uint8)
        mask    = full_traj.sum(axis=2) > 0
        bg_dark[mask] = full_traj[mask]
        st.image(cv2.cvtColor(bg_dark, cv2.COLOR_BGR2RGB),
                 caption="Coloured by player ID · grey = viewer · ○ start  ● end")
    else:
        st.image(cv2.cvtColor(full_traj, cv2.COLOR_BGR2RGB))

with tab_chart:
    st.subheader("Active vs cumulative IDs over time")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor("#0e1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    ax1.plot(frame_times, active_over_time, color="#4f86e8", linewidth=1.2)
    ax1.fill_between(frame_times, active_over_time, alpha=0.15, color="#4f86e8")
    ax1.set_title("Active IDs per frame (players visible)")
    ax1.set_ylabel("Active IDs")

    ax2.plot(frame_times, unique_over_time, color="#e84f4f", linewidth=1.5)
    ax2.set_title("Cumulative unique IDs over time")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Unique IDs")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Track length distribution
    lengths = [len(pts) for pts in centroid_history.values()]
    if lengths:
        fig2, ax = plt.subplots(figsize=(12, 3))
        fig2.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.hist(lengths, bins=min(30, len(lengths)), color="#4f86e8", edgecolor="#0e1117")
        ax.axvline(min_hits, color="#ff9800", linestyle="--", label=f"min_hits={min_hits}")
        ax.set_title("Track length distribution (frames per ID)")
        ax.set_xlabel("Track length (frames)")
        ax.set_ylabel("Count")
        ax.legend(labelcolor="white", facecolor="#1a1a2e")
        st.pyplot(fig2)
        plt.close(fig2)

with tab_stats:
    player_ids = all_ids - grey_ids
    st.subheader("Summary statistics")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Frames processed", frame_idx)
    r2.metric("Player IDs", len(player_ids))
    r3.metric("Viewer IDs suppressed", len(grey_ids))
    r4.metric("Processing speed", f"{frame_idx/elapsed:.1f} FPS")

    st.subheader("Parameter configuration used")
    params = {
        "conf_thresh": conf_thresh,
        "track_activation_threshold": track_thresh,
        "match_thresh": match_thresh,
        "lost_track_buffer (frames)": buffer_size,
        "min_consecutive_frames": min_hits,
        "min_box_area (px²)": min_box_area,
        "stationary_filter": enable_stationary,
        "stationary_movement_threshold (px)": stationary_px if enable_stationary else "—",
    }
    for k, v in params.items():
        st.markdown(f"- **{k}**: `{v}`")

    if lengths:
        avg_len   = np.mean(lengths)
        noise_ct  = sum(l < min_hits for l in lengths)
        stable_ct = len(lengths) - noise_ct
        st.subheader("Track quality")
        q1, q2, q3 = st.columns(3)
        q1.metric("Avg track length (frames)", f"{avg_len:.1f}")
        q2.metric("Stable tracks", stable_ct)
        q3.metric("Noise tracks (< min_hits)", noise_ct)

# Cleanup
try:
    os.unlink(VIDEO_PATH)
except Exception:
    pass
