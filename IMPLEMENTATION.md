# Implementation Details

## Overview

This document provides a comprehensive explanation of the bird counting and weight estimation methods used in the Poultry Monitoring System.

## Table of Contents

1. [Bird Counting Method](#bird-counting-method)
2. [Weight Estimation Approach](#weight-estimation-approach)
3. [System Architecture](#system-architecture)
4. [Performance Considerations](#performance-considerations)

---

## Bird Counting Method

### Detection + Tracking Pipeline

The bird counting system combines **object detection** (YOLOv8) with **multi-object tracking** (ByteTrack) to accurately count birds while preventing double-counting.

```
┌─────────────────────────────────────────────────────────────┐
│                      VIDEO INPUT                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               FRAME SAMPLING (Optional)                      │
│  • Reduce computational load for long videos                 │
│  • Sample every Nth frame (e.g., 5 FPS from 30 FPS)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   YOLOv8 DETECTION                           │
│  • Model: YOLOv8n (nano) pretrained on COCO                 │
│  • Class Filter: bird (class_id = 14)                       │
│  • Confidence Threshold: 0.25 (configurable)                │
│  • NMS IoU Threshold: 0.45 (configurable)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  BYTETRACK TRACKING                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ New Tracks                                           │   │
│  │  • Activation threshold: 0.25                        │   │
│  │  • Creates new track ID for new detection           │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Matched Tracks                                       │   │
│  │  • Motion prediction for position estimation         │   │
│  │  • IoU matching threshold: 0.8                       │   │
│  │  • Maintains existing track ID                       │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Lost Tracks                                          │   │
│  │  • Buffer: 30 frames                                 │   │
│  │  • Maintains ID during brief occlusions              │   │
│  │  • Removes after buffer expires                      │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  TRACK FILTERING                             │
│  • Minimum track length: 5 frames                           │
│  • Filters transient false positives                        │
│  • Only stable tracks counted                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 COUNT AGGREGATION                            │
│  • Count unique track IDs per frame                         │
│  • Generate time-series with timestamps                     │
│  • Calculate statistics (avg, max, unique)                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. YOLOv8 Detection

**What it does:**
- Detects birds in each frame using deep learning
- Returns bounding boxes with confidence scores

**Model Selection:**
- **YOLOv8n (nano)**: Lightweight, fast, good accuracy
- Pretrained on COCO dataset (80 object classes)
- Bird class (class_id = 14) filtered from all detections

**Configuration:**
```python
results = model(
    frame,
    conf=0.25,      # Minimum confidence for detection
    iou=0.45,       # IoU threshold for NMS
    verbose=False   # Suppress output
)[0]
```

**Why YOLOv8?**
- ✅ State-of-the-art accuracy
- ✅ Real-time performance
- ✅ Pretrained weights available
- ✅ Easy to use with Ultralytics library

#### 2. ByteTrack Multi-Object Tracking

**What it does:**
- Assigns stable IDs to detected birds across frames
- Maintains identity even during occlusions
- Prevents double-counting

**Algorithm Overview:**

ByteTrack is a state-of-the-art tracking algorithm that handles occlusions and ID switches better than traditional trackers like SORT or DeepSORT.

**Key Innovation:** ByteTrack uses BOTH high and low confidence detections:
- **High confidence detections** → Primary tracking
- **Low confidence detections** → Recover occluded objects

**Tracking Process (per frame):**

1. **First Association (High Confidence)**
   ```python
   high_conf_detections = detections[confidence > 0.7]
   match_tracks_with_detections(high_conf_detections)
   ```
   - Match high-confidence detections to existing tracks
   - Use Kalman filter for motion prediction
   - IoU matching with threshold 0.8

2. **Second Association (Low Confidence)**
   ```python
   low_conf_detections = detections[0.25 < confidence < 0.7]
   unmatched_tracks = tracks_without_match
   match_tracks_with_detections(low_conf_detections, unmatched_tracks)
   ```
   - Match low-confidence detections to unmatched tracks
   - Recovers occluded birds that have low detection confidence

3. **Track Management**
   ```python
   # Create new tracks for unmatched high-conf detections
   new_tracks = create_tracks(unmatched_high_conf)
   
   # Remove lost tracks after buffer expires
   remove_tracks(age > lost_track_buffer)
   ```

**Configuration:**
```python
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,  # Min conf to create track
    lost_track_buffer=30,             # Frames before removing lost track
    minimum_matching_threshold=0.8,   # IoU threshold for matching
    frame_rate=30,                    # Video FPS
    minimum_consecutive_frames=5      # Min frames to confirm track
)
```

**Why ByteTrack?**
- ✅ Handles occlusions robustly
- ✅ Reduces ID switches
- ✅ Better than SORT/DeepSORT for crowded scenes
- ✅ No need for re-identification model

#### 3. Anti-Double-Counting Mechanisms

**Problem:** How do we ensure each bird is counted only once?

**Solution: Multi-layered approach**

1. **Stable Tracking IDs**
   - Each bird gets a unique ID when first detected
   - ID persists across all frames where bird is visible
   - Same bird = same ID = counted once per frame

2. **Track Activation Threshold (0.25)**
   - Prevents spurious detections from creating new tracks
   - Only confident detections (≥25%) can start a new track
   - Filters out noise and artifacts

3. **Minimum Track Length (5 frames)**
   - Tracks must exist for at least 5 frames to be counted
   - Filters transient false positives
   - Ensures only stable detections are counted

4. **Temporal Consistency**
   - Count unique track IDs, not detections
   - If bird appears in 100 frames, counted 100 times (once per frame)
   - But same bird never counted twice in same frame

**Example:**
```python
# Frame 1: Detect 3 birds → IDs: [1, 2, 3] → Count: 3
# Frame 2: Detect 3 birds → IDs: [1, 2, 3] → Count: 3
# Frame 3: Detect 4 birds → IDs: [1, 2, 3, 4] → Count: 4
# Frame 4: Detect 3 birds → IDs: [1, 2, 4] → Count: 3 (bird 3 left)

# Total unique birds in video: 4 (IDs 1, 2, 3, 4)
# Average count: 3.25 birds per frame
```

#### 4. Occlusion Handling

**Problem:** Birds overlap or hide behind objects. How do we maintain tracking?

**Solution: Three-tier approach**

1. **Motion Prediction (Kalman Filter)**
   ```
   Frame 100: Bird visible at (x=100, y=200) with velocity (vx=2, vy=0)
   Frame 101: Bird occluded → Predict position: (x=102, y=200)
   Frame 102: Bird occluded → Predict position: (x=104, y=200)
   Frame 103: Bird visible at (x=105, y=201) → Match to predicted position
   ```
   - Predicts bird position based on previous motion
   - Allows matching even when bird is occluded
   - Works for brief occlusions (2-3 frames)

2. **Lost Track Buffer (30 frames)**
   ```
   Frame 100: Bird detected (ID=5)
   Frame 101-130: Bird fully occluded (no detection)
     → Track maintained in "lost" state
     → Continues motion prediction
     → Ready to match if bird reappears
   Frame 131: If bird reappears → Re-matched to ID=5
   Frame 131: If bird doesn't reappear → Track removed
   ```
   - Maintains track for up to 30 frames without detection
   - Handles long occlusions (up to 1 second at 30 FPS)
   - Prevents premature ID removal

3. **High IoU Matching Threshold (0.8)**
   - Requires 80% overlap for match
   - Reduces false matches during crowded scenes
   - Ensures correct re-identification

**Real-World Scenario:**
```
Time: 0.0s - Bird A (ID=1) walking, visible
Time: 0.5s - Bird B (ID=2) walks in front of Bird A
            Bird A occluded, last position: (150, 200)
            Tracker predicts: (155, 200), (160, 200), ...
Time: 1.0s - Bird B moves away
            Bird A visible again at (180, 205)
            IoU with predicted position: 0.85 > 0.8 ✓
            Successfully matched to ID=1 (no new ID created)
```

### Output Format

**Per-Frame Counts:**
```json
{
  "timestamp": 1.5,
  "frame_idx": 45,
  "count": 12
}
```

**Track Sample:**
```json
{
  "track_id": 1,
  "bbox": [100.5, 200.3, 150.8, 280.6],
  "confidence": 0.85,
  "frame_idx": 45,
  "timestamp": 1.5
}
```

**Metadata:**
```json
{
  "unique_tracks": 15,
  "avg_bird_count": 12.3,
  "max_bird_count": 18
}
```

---

## Weight Estimation Approach

### Feature-Based Regression Method

The weight estimation uses **bounding box geometry** as a proxy for bird weight. This is based on the principle that larger birds occupy more pixels in a fixed-camera setup.

```
┌─────────────────────────────────────────────────────────────┐
│           BOUNDING BOX (x1, y1, x2, y2)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE EXTRACTION                           │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Area = (x2 - x1) × (y2 - y1)                       │     │
│  │   • Pixel area of bounding box                      │     │
│  │   • Proxy for bird size                             │     │
│  └────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Aspect Ratio = height / width                      │     │
│  │   • Body shape information                          │     │
│  │   • Distinguishes crouching vs. standing            │     │
│  └────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Confidence = detection_confidence                   │     │
│  │   • How confident detector is                       │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            WEIGHT PROXY CALCULATION                          │
│                                                              │
│  weight_index = area × density × (1 + 0.1 × aspect)        │
│                                                              │
│  where:                                                      │
│    • area = bbox area in pixels                             │
│    • density = 1.0 (baseline factor)                        │
│    • aspect = height / width                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         TEMPORAL SMOOTHING (Per Track)                       │
│  • Average weight estimates over bird's lifetime            │
│  • Reduces frame-to-frame noise                             │
│  • Only tracks with ≥5 observations included                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT                                    │
│  • Weight Index (relative measure) OR                       │
│  • Weight in Grams (if calibrated)                          │
│  • Confidence Score                                          │
│  • Number of Observations                                    │
└─────────────────────────────────────────────────────────────┘
```

### Mathematical Formula

#### Weight Index Calculation

```python
# Step 1: Extract bbox dimensions
width = x2 - x1
height = y2 - y1
area = width * height

# Step 2: Calculate aspect ratio
aspect_ratio = height / width

# Step 3: Compute weight index
weight_index = area * density_factor * (1 + 0.1 * aspect_ratio)
```

**Intuition:**
- **Base weight**: Proportional to bbox area
- **Aspect adjustment**: Taller birds (higher aspect) weigh slightly more
- **Density factor**: Baseline scale (1.0), can be calibrated

**Example:**
```python
# Bird 1: Standing
bbox = [100, 150, 150, 250]  # width=50, height=100
area = 50 * 100 = 5000
aspect = 100 / 50 = 2.0
weight_index = 5000 * 1.0 * (1 + 0.1 * 2.0) = 6000

# Bird 2: Crouching
bbox = [200, 180, 260, 240]  # width=60, height=60
area = 60 * 60 = 3600
aspect = 60 / 60 = 1.0
weight_index = 3600 * 1.0 * (1 + 0.1 * 1.0) = 3960

# Bird 1 has higher weight index (larger + standing)
```

#### Confidence Calculation

```python
# Size confidence: Larger birds → higher confidence
size_confidence = min(area / 5000.0, 1.0)

# Final confidence: Combine detection and size confidence
final_confidence = detection_conf * size_confidence * 0.8
```

**Why 0.8 multiplier?**
- Conservative estimate
- Acknowledges limitations of proxy method
- Leaves room for uncertainty

#### Temporal Smoothing

```python
# Collect all weight estimates for a track
track_weights = [6000, 6100, 5950, 6050, 6020]

# Average over lifetime
avg_weight = mean(track_weights) = 6024

# This is the final weight estimate for the bird
```

**Benefits:**
- Reduces noise from frame-to-frame variations
- Handles partial occlusions (some frames have smaller bbox)
- More stable and reliable estimates

### Assumptions

The weight estimation method relies on several key assumptions:

#### 1. Fixed Camera Perspective

**Assumption:** Camera position and angle remain constant throughout video.

**Why it matters:**
- Birds at same distance appear same size
- Consistent pixel-to-size ratio
- Enables comparative weight estimates

**Violation:** If camera moves or zooms, scale changes and weight estimates become unreliable.

#### 2. Standing Birds

**Assumption:** Birds are on the ground (not flying) with relatively consistent posture.

**Why it matters:**
- Flying birds have different bbox aspect ratios
- Crouching vs. standing affects bbox size
- Consistent posture → consistent area-to-weight mapping

**Handling:** Tracking filters (minimum 5 frames) help filter out flying/jumping birds.

#### 3. Linear Relationship

**Assumption:** Bounding box area correlates linearly with bird mass.

**Why it matters:**
- Larger bbox → heavier bird
- Proportional relationship simplifies calculation
- Works reasonably well for similar bird types

**Reality:** Relationship may not be perfectly linear, but approximation is useful.

#### 4. Uniform Density

**Assumption:** All birds have similar density (mass per volume).

**Why it matters:**
- Uses single density factor for all birds
- Simplifies calculation
- Reasonable for same species/age group

**Limitation:** Different breeds or ages may have different densities.

### Limitations

1. **Perspective Effects**
   - Birds closer to camera appear larger
   - Same-weight birds at different distances have different bbox sizes
   - **Mitigation:** Calibrate with reference birds at fixed distance

2. **Posture Variations**
   - Crouching birds have smaller bboxes
   - Standing/stretching birds have larger bboxes
   - **Mitigation:** Temporal smoothing averages over different postures

3. **Occlusions**
   - Partially occluded birds have smaller bboxes
   - Underestimates weight if heavily occluded
   - **Mitigation:** Temporal smoothing, minimum observations filter

4. **Breed Differences**
   - Different breeds have different body shapes
   - Same weight may have different bbox sizes
   - **Mitigation:** Calibrate separately for each breed

### Calibration to Grams

The system outputs "weight index" by default (relative measure). To convert to actual grams, calibration is required:

#### Calibration Process

**Step 1: Reference Object**
```python
# Place 10cm × 10cm square in camera view
ref_bbox_pixels = 856  # Measured in pixels
ref_size_cm = 10
pixels_per_cm = ref_bbox_pixels / ref_size_cm  # 85.6 px/cm
```

**Step 2: Ground Truth Collection**
```python
# Weigh 10+ birds with scale
ground_truth = [
    {"bird_id": 1, "actual_weight_g": 2500, "video_segment": "bird1.mp4"},
    {"bird_id": 2, "actual_weight_g": 2800, "video_segment": "bird2.mp4"},
    ...
]

# Process videos to get weight indices
weight_indices = [
    {"bird_id": 1, "weight_index": 2847.3},
    {"bird_id": 2, "weight_index": 3124.8},
    ...
]
```

**Step 3: Linear Regression**
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data
X = np.array([w["weight_index"] for w in weight_indices]).reshape(-1, 1)
y = np.array([g["actual_weight_g"] for g in ground_truth])

# Fit model: weight_g = a × weight_index + b
model = LinearRegression()
model.fit(X, y)

calibration_factor = model.coef_[0]      # e.g., 0.025
calibration_bias = model.intercept_      # e.g., 50.0

# R² score (goodness of fit)
r2_score = model.score(X, y)            # e.g., 0.85
```

**Step 4: Apply Calibration**
```python
# Update weight estimator
weight_estimator = WeightEstimator(
    calibrated=True,
    calibration_factor=0.025,
    calibration_bias=50.0
)

# Now weight estimates are in grams
weight_g = weight_estimator.estimate_weight(bbox, confidence)
# Output: {"weight_value": 2621.5, "unit": "grams", ...}
```

#### Example Calibration Results

```
Calibration Dataset:
  10 birds weighed on scale
  Range: 2400g - 3200g
  
Linear Regression:
  weight_g = 0.0246 × weight_index + 48.3
  R² = 0.87
  RMSE = 89.2g
  
Validation (5 new birds):
  Bird A: Actual=2650g, Predicted=2621g, Error=29g (1.1%)
  Bird B: Actual=2890g, Predicted=2915g, Error=25g (0.9%)
  ...
  Mean Absolute Error: 34.2g (1.3%)
```

### Alternative Methods (Future)

1. **Depth-Based Estimation**
   - Use depth camera or stereo vision
   - Estimate 3D volume directly
   - More accurate but requires special hardware

2. **Silhouette-Based**
   - Extract bird silhouette from multiple angles
   - Compute 3D volume
   - Requires multi-camera setup

3. **Deep Learning Regression**
   - Train CNN: bird image → weight
   - End-to-end learning
   - Requires large dataset of labeled bird weights

4. **Floor Pressure Sensors**
   - Measure actual weight when bird steps on sensor
   - Ground truth validation
   - Combine with vision for best accuracy

---

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Endpoints:                                        │ │
│  │    • GET  /health                                  │ │
│  │    • POST /analyze_video                           │ │
│  └────────────────────────────────────────────────────┘ │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│                 PoultryTracker                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │  • YOLOv8 Model                                    │ │
│  │  • ByteTrack Tracker                               │ │
│  │  • Weight Estimator                                │ │
│  │  • Video Processing Pipeline                       │ │
│  └────────────────────────────────────────────────────┘ │
└─────────┬───────────────────────────────┬───────────────┘
          │                               │
          ▼                               ▼
┌──────────────────────┐    ┌─────────────────────────────┐
│   Detection Module   │    │   Weight Estimator Module   │
│  ┌────────────────┐  │    │  ┌───────────────────────┐  │
│  │ YOLOv8         │  │    │  │ Feature Extraction    │  │
│  │  • Bird class  │  │    │  │  • Area               │  │
│  │  • Confidence  │  │    │  │  • Aspect ratio       │  │
│  │  • Bounding box│  │    │  │ Weight Calculation    │  │
│  └────────────────┘  │    │  │  • Proxy formula      │  │
│                      │    │  │  • Calibration        │  │
│  ┌────────────────┐  │    │  │ Temporal Smoothing    │  │
│  │ ByteTrack      │  │    │  │  • Track averaging    │  │
│  │  • Track IDs   │  │    │  └───────────────────────┘  │
│  │  • Motion pred │  │    └─────────────────────────────┘
│  │  • Occlusion   │  │
│  └────────────────┘  │
└──────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                  Output Generation                       │
│  • Counts time-series (JSON)                            │
│  • Track samples (JSON)                                 │
│  • Weight estimates (JSON)                              │
│  • Annotated video (MP4)                                │
│  • Metadata (JSON)                                      │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. User uploads video
       ↓
2. API receives video, saves to temp directory
       ↓
3. PoultryTracker.process_video() called
       ↓
4. For each frame:
   a. Optional frame sampling (skip if needed)
   b. YOLOv8 detection → bounding boxes
   c. Filter for bird class
   d. ByteTrack tracking → track IDs
   e. Count unique track IDs
   f. Weight estimation per bird
   g. Annotate frame (if output video requested)
       ↓
5. After all frames:
   a. Filter tracks (min 5 frames)
   b. Average weights per track
   c. Calculate statistics
   d. Generate JSON response
       ↓
6. Return results to user
```

---

## Performance Considerations

### Computational Complexity

**Per Frame:**
- YOLOv8 detection: O(1) [constant for fixed resolution]
- ByteTrack matching: O(n×m) [n=tracks, m=detections]
- Weight estimation: O(m) [linear in detections]

**Total:** O(F) where F = number of processed frames

### Optimization Strategies

1. **Frame Sampling**
   ```python
   # Process 5 FPS instead of 30 FPS
   # 6× speedup, minimal accuracy loss
   fps_sample = 5
   ```

2. **GPU Acceleration**
   ```bash
   # 10× faster with CUDA GPU
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Smaller Model**
   ```python
   # YOLOv8n (nano): Fast, good accuracy
   # YOLOv8s (small): Balanced
   # YOLOv8m (medium): Slower, better accuracy
   model = YOLO("yolov8n.pt")  # Use nano for speed
   ```

4. **Lower Resolution**
   ```bash
   # Pre-process video to 1280×720 instead of 1920×1080
   ffmpeg -i input.mp4 -vf scale=1280:720 output.mp4
   ```

### Typical Performance

**CPU (Intel i7)**:
- Processing speed: ~5-10 FPS
- 2-minute video (30 FPS): ~5 minutes

**GPU (NVIDIA RTX 3060)**:
- Processing speed: ~50-100 FPS
- 2-minute video (30 FPS): ~30 seconds

**Frame Sampling (5 FPS)**:
- Processing speed: 6× faster
- 2-minute video: ~50 seconds (CPU)

### Memory Requirements

- **Minimum:** 4GB RAM
- **Recommended:** 8GB RAM
- **GPU:** 4GB+ VRAM (if using GPU)
- **Disk:** 2GB for models, temp space for videos

---

## Conclusion

The Poultry Monitoring System combines state-of-the-art computer vision techniques (YOLOv8, ByteTrack) with practical heuristics (bbox-based weight estimation) to provide a robust solution for automated poultry monitoring.

**Key Strengths:**
- Accurate bird counting with anti-double-counting
- Robust tracking through occlusions
- Practical weight estimation without specialized hardware
- RESTful API for easy integration

**Future Improvements:**
- Real-time processing with GPU optimization
- Multi-camera support for better coverage
- Deep learning-based weight estimation
- Behavior analysis (feeding, resting, movement patterns)
