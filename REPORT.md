# Technical Report — Multi-Object Detection & Persistent ID Tracking

**Project:** Sports Video MOT Pipeline  
**Author:** Shruti Sharma  
**Date:** April 2026

---

## 1. Objective

Build a computer vision pipeline that detects all persons in a sports video and assigns each one a unique ID that remains consistent across the full video, even under occlusion, motion blur, rapid movement, and camera motion.

---

## 2. Model & Detector

**YOLOv8m (Ultralytics)**

YOLOv8m was selected as the detector after considering the speed/accuracy tradeoff:

- `yolov8n` — fast but misses partially occluded players at distance
- `yolov8m` — runs at ~5 FPS on CPU with sufficient recall for crowded scenes
- `yolov8x` — too slow for practical video processing on CPU

The model is used with `classes=[0]` (persons only) to eliminate irrelevant detections.

---

## 3. Tracking Algorithm

**ByteTrack (via the `supervision` library)**

ByteTrack was selected over DeepSORT for the following reasons:

| Criterion | ByteTrack | DeepSORT |
|---|---|---|
| Per-frame cost | IoU only — cheap | ReID network per detection — expensive |
| Occlusion handling | Two-stage matching recovers low-conf detections | Single-stage — drops occluded tracks sooner |
| CPU viability | Yes | Marginal |

ByteTrack's two-stage matching strategy is the key advantage: high-confidence detections are matched first; remaining unmatched tracks are then re-matched against low-confidence detections (0.1–0.35 conf) that would otherwise be discarded. This directly recovers tracks of partially occluded players.

---

## 4. ID Consistency Strategy

Four layers maintain ID stability:

### 4.1 Lost-track buffer (`BUFFER_SIZE = 60`)
When a track drops below confidence, it is held alive for 60 frames (2 s at 30 fps). If the person reappears within that window, the original ID is restored rather than spawning a new one.

### 4.2 Track activation threshold (`TRACK_THRESH = 0.65`)
A new track ID is only spawned for detections above 0.65 confidence. Re-entering players who first appear at moderate confidence (0.5–0.6) cannot start a new track — they must reach the buffer's Kalman-predicted position before being confirmed. This was the single most impactful parameter: lowering it from 0.65 back to 0.5 increased unique IDs from ~47 to ~79 in testing.

### 4.3 Minimum consecutive frames (`MIN_HITS = 3`)
A detection must appear in 3 consecutive frames before receiving an ID. This kills single-frame ghost detections (motion-blur artefacts, partial limb detections at the frame edge).

### 4.4 Area filter (`MIN_BOX_AREA = 1500 px²`)
Bounding boxes smaller than 1500 px² are discarded before tracking. These are reliably partial bodies at the edge of frame that would otherwise spawn ghost IDs.

---

## 5. Stationary Person Filter

Spectators and crowd members in the stands are valid person detections but are not subjects of interest. A post-tracking filter suppresses any track whose centroid has not moved more than 20 pixels over the last 300 frames (10 s). This reduces the reported unique ID count to active players only and cleans up the annotated output.

---

## 6. Results

| Metric | Value |
|---|---|
| Video resolution | 1920×1080 |
| Processing speed | ~5 FPS (CPU) |
| Unique active IDs | ~47 (after stationary filter) |
| Confidence threshold | 0.35 |
| Track activation threshold | 0.65 |
| Lost-track buffer | 60 frames |

---

## 7. Challenges & Failure Cases

**ID switches under heavy occlusion**  
When two players fully overlap for several seconds and the lost-track buffer expires, ByteTrack cannot re-match by IoU alone and assigns a new ID on re-entry. This is a fundamental limitation of IoU-only association.

**Same-jersey confusion**  
Players in identical uniforms have similar appearance. IoU-based trackers have no appearance signal to distinguish them — only spatial coherence (Kalman-predicted position) separates them. Rapid direction changes break this.

**Camera motion**  
Fast pans cause IoU matching to fail because the Kalman-predicted and actual positions diverge. Global Motion Compensation (GMC via sparse optical flow) was tested using BoT-SORT but did not produce a consistent improvement over tuned ByteTrack parameters on the test video.

---

## 8. Possible Improvements

**Appearance-based ReID (OSNet embeddings)**  
The most impactful next step. OSNet runs fast enough (~10ms per crop on GPU) to afford per-detection appearance vectors. DeepSORT and StrongSORT use this — the correct approach once GPU is available.

**Global Motion Compensation**  
BoT-SORT's `sparseOptFlow` compensates Kalman predictions for camera motion. Useful for broadcast footage with frequent pans and zooms.

**Confidence-weighted IoU (GHOST)**  
GHOST (2023) replaces hard IoU thresholds with soft confidence-weighted matching, reducing ID switches at boundaries without the ReID compute cost.

**Evaluation metrics**  
HOTA, MOTA, and IDF1 computed against manual ground-truth annotations would quantify improvement across parameter sweeps rather than relying on unique-ID count as a proxy.
