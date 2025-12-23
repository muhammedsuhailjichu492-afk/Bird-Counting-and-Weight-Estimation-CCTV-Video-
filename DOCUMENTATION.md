# Bird/Poultry Counting System - Documentation

## Overview
This system uses YOLOv8 object detection with ByteTrack tracking to count and monitor individual birds/poultry in video footage. It provides real-time tracking, weight estimation, and comprehensive analytics.

## Key Improvements in Enhanced Version

### 1. **Better Error Handling**
- File existence validation
- Proper exception handling
- Informative error messages

### 2. **Configurable Parameters**
- Centralized CONFIG dictionary
- Easy adjustment without code modification
- Model selection options (yolov8n through yolov8x)

### 3. **Class Filtering**
- Filters for bird class only (COCO class 14)
- Reduces false positives from other objects
- More accurate bird counting

### 4. **Enhanced Statistics**
- Mean, median, standard deviation
- Min/max values for areas and weights
- Per-bird tracking metrics

### 5. **Progress Tracking**
- Real-time progress display
- Frame-by-frame counting
- Total unique birds counter

### 6. **Better Visualization**
- Frame-level information overlay
- Unique ID tracking across frames
- Configurable colors and styles

## Configuration Parameters

### Input/Output
```python
"video_path": str          # Path to input video
"output_video": str        # Path for annotated output
"output_json": str         # Path for results JSON
```

### Model Settings
```python
"model_name": str          # YOLOv8 model variant
                          # Options: yolov8n (fastest), yolov8s, yolov8m, 
                          #          yolov8l, yolov8x (most accurate)
"conf_threshold": float   # Confidence threshold (0.0-1.0)
"iou_threshold": float    # IoU threshold for NMS (0.0-1.0)
"tracker": str            # Tracking algorithm (bytetrack.yaml or botsort.yaml)
```

### Filtering Parameters
```python
"min_area": int           # Minimum bbox area (pixels²)
                         # Typical chicken: 5,000-15,000 px²
                         # Adjust based on video resolution

"max_area": int          # Maximum bbox area (pixels²)
                         # Prevents false positives from large objects

"min_y_ratio": float     # Ignore top portion of frame (0.0-1.0)
                         # 0.4 = ignore top 40%
                         # Useful to filter flying birds or background
```

### Weight Estimation
```python
"weight_area_factor": int # Calibration factor for area→weight conversion
                          # Formula: weight_proxy = area / factor
                          # Lower value = higher weight estimates
```

## Calibrating for Your Setup

### 1. **Area Thresholds**
Record actual bird sizes in your video:
```python
# Measure a few birds manually
actual_bird_area = (x2 - x1) * (y2 - y1)
print(f"Bird area: {actual_bird_area} pixels²")

# Set thresholds with 20-30% margin
CONFIG["min_area"] = actual_bird_area * 0.7
CONFIG["max_area"] = actual_bird_area * 1.3
```

### 2. **Weight Calibration**
If you have actual weight data:
```python
# Example: 2kg bird has 10,000 px² area
known_weight_kg = 2.0
known_area = 10000

# Calculate calibration factor
weight_factor = known_area / known_weight_kg
# Result: 5000 means 1.0 proxy = ~2kg

CONFIG["weight_area_factor"] = weight_factor
```

### 3. **Position Filtering**
Adjust based on camera angle:
```python
# Ground-level camera: use lower ratio
CONFIG["min_y_ratio"] = 0.2  # Ignore top 20%

# High-angle camera: use higher ratio
CONFIG["min_y_ratio"] = 0.5  # Ignore top 50%
```

## Output JSON Structure

```json
{
  "video_info": {
    "path": "video_path",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "total_frames": 9000,
    "duration_sec": 300.0
  },
  "processing_config": {
    "model": "yolov8n.pt",
    "confidence_threshold": 0.25,
    "min_area": 2500,
    "max_area": 40000
  },
  "summary": {
    "total_unique_birds": 15,
    "area_stats": {
      "mean": 8500.5,
      "median": 8200.0,
      "std": 1500.3,
      "min": 5000,
      "max": 12000
    },
    "weight_stats": {
      "mean": 0.472,
      "median": 0.456,
      "std": 0.083,
      "min": 0.278,
      "max": 0.667
    }
  },
  "count_over_time": [
    {
      "frame": 0,
      "time_sec": 0.0,
      "bird_count": 5,
      "unique_birds_so_far": 5
    }
  ],
  "individual_birds": {
    "1": {
      "track_length": 150,
      "avg_area": 8500.0,
      "std_area": 200.5,
      "min_area": 8000,
      "max_area": 9000,
      "avg_weight_proxy": 0.472
    }
  }
}
```

## Common Issues & Solutions

### Issue 1: Too Many False Positives
**Solution:**
- Increase `conf_threshold` (try 0.35 or 0.4)
- Tighten `min_area` and `max_area` ranges
- Adjust `min_y_ratio` to ignore more background

### Issue 2: Missing Birds
**Solution:**
- Lower `conf_threshold` (try 0.2 or 0.15)
- Expand `min_area` and `max_area` ranges
- Use larger model (yolov8m or yolov8l)

### Issue 3: ID Switching
**Solution:**
- Try `botsort.yaml` tracker instead of `bytetrack.yaml`
- Increase `iou_threshold` (try 0.6 or 0.7)
- Use more powerful model for better feature extraction

### Issue 4: Poor Performance
**Solution:**
- Use smaller model (yolov8n)
- Enable `frame_skip` (process every 2nd or 3rd frame)
- Reduce video resolution before processing

### Issue 5: Inaccurate Weight Estimates
**Solution:**
- Calibrate with actual bird measurements
- Ensure consistent camera distance/angle
- Consider using 3D area estimation for better accuracy

## Advanced Usage

### Process Specific Time Range
```python
# Add to main loop
start_frame = 300  # Start at 10 seconds (300 frames @ 30fps)
end_frame = 900    # End at 30 seconds

if frame_id < start_frame:
    frame_id += 1
    continue
if frame_id > end_frame:
    break
```

### Multiple Video Processing
```python
video_paths = [
    "/path/to/video1.mp4",
    "/path/to/video2.mp4",
    "/path/to/video3.mp4"
]

for i, video_path in enumerate(video_paths):
    config = CONFIG.copy()
    config["video_path"] = video_path
    config["output_video"] = f"/output/video_{i}.mp4"
    config["output_json"] = f"/output/results_{i}.json"
    
    process_video(config)
```

### Custom Weight Formula
```python
def compute_weight_proxy_advanced(area, aspect_ratio):
    """
    More sophisticated weight estimation using bbox shape.
    
    Args:
        area: Bounding box area
        aspect_ratio: Width/height ratio
    """
    # Normalize area
    base_weight = area / 18000
    
    # Adjust for bird posture
    # Standing birds (tall/narrow) vs sitting (wide/short)
    posture_factor = 1.0
    if aspect_ratio < 0.8:  # Tall and narrow (standing)
        posture_factor = 1.1
    elif aspect_ratio > 1.2:  # Wide and short (sitting)
        posture_factor = 0.9
    
    return round(min(base_weight * posture_factor, 1.0), 3)
```

### Export Results to CSV
```python
import pandas as pd

# After process_video()
df = pd.DataFrame(results["count_over_time"])
df.to_csv("/output/count_timeline.csv", index=False)

# Individual bird data
bird_data = []
for bird_id, stats in results["individual_birds"].items():
    bird_data.append({
        "bird_id": bird_id,
        **stats
    })

df_birds = pd.DataFrame(bird_data)
df_birds.to_csv("/output/bird_statistics.csv", index=False)
```

## Performance Optimization

### Model Selection Guide
| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| yolov8n | Fastest | Good | Real-time, many videos |
| yolov8s | Fast | Better | Balanced performance |
| yolov8m | Medium | Great | High accuracy needed |
| yolov8l | Slow | Excellent | Best quality |
| yolov8x | Slowest | Best | Research/analysis |

### Processing Speed Tips
1. **Use GPU**: Ensure CUDA is available for 10-50x speedup
2. **Lower Resolution**: Resize video before processing
3. **Frame Skipping**: Process every 2-3 frames for static scenes
4. **Batch Processing**: Process multiple videos in parallel

### Memory Optimization
```python
# Limit tracking history to recent frames
MAX_HISTORY_SIZE = 100

for tid, areas in track_history.items():
    if len(areas) > MAX_HISTORY_SIZE:
        track_history[tid] = areas[-MAX_HISTORY_SIZE:]
```

## Best Practices

1. **Test on Sample First**: Process 10-30 seconds before full video
2. **Validate Parameters**: Check annotated frames visually
3. **Use Ground Truth**: Compare against manual counts
4. **Document Settings**: Save CONFIG with results
5. **Version Control**: Track parameter changes over time

## References

- YOLOv8 Documentation: https://docs.ultralytics.com/
- ByteTrack Paper: https://arxiv.org/abs/2110.06864
- COCO Dataset: https://cocodataset.org/

## Support

For issues or questions:
1. Check video resolution and lighting conditions
2. Verify model installation: `yolo checks`
3. Test with different confidence thresholds
4. Review annotated video for false positives/negatives
