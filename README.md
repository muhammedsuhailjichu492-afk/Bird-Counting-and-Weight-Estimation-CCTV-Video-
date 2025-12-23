# Bird Counting System - Quick Start Guide

Hey there! This system helps you count chickens/birds in videos automatically.

## What's Inside?

1. **bird_counting_improved.py** - The main script (use this one!)
2. **bird_counting_original.py** - Your original code (kept for reference)
3. **DOCUMENTATION.md** - Detailed technical docs
4. **COMPARISON.md** - What changed and why

## How to Use (Super Simple)

### Step 1: Upload to Google Colab
1. Go to https://colab.research.google.com/
2. Upload `bird_counting_improved.py`
3. Mount your Google Drive (the video should be in your Drive)

### Step 2: Change 3 Things
Open the script and find this section near the top:

```python
CONFIG = {
    # Change these 3 lines:
    "video_path": "/content/drive/MyDrive/YOUR_VIDEO.MP4",  # Your video path
    "output_video": "/content/annotated_output.mp4",         # Where to save result
    "output_json": "/content/results.json",                  # Where to save data
```

### Step 3: Run It
Click **Runtime > Run all** and wait. You'll see progress like:
```
Processing: 45.2% | Frame 1356/3000 | Current count: 8
```

### Step 4: Get Your Results
The script will automatically download:
- **annotated_output.mp4** - Video with boxes around each bird
- **results.json** - All the counting data

## What You Get

### The Video Shows:
- Green boxes around each bird
- ID number for tracking
- Weight estimate
- Total count at top of screen

### The JSON Contains:
- Total unique birds detected
- Count at each moment in time
- Individual stats for each bird (average size, weight, etc.)
- Video information

## Fine-Tuning (Optional)

If you're getting too many false detections or missing birds:

```python
CONFIG = {
    # ... other settings ...
    
    # If detecting too many wrong things:
    "conf_threshold": 0.35,    # Increase this (default: 0.25)
    "min_area": 3500,          # Increase this (default: 2500)
    
    # If missing real birds:
    "conf_threshold": 0.20,    # Decrease this
    "min_area": 2000,          # Decrease this
    "max_area": 50000,         # Increase this
}
```

## Quick Tips

- Start with a short video clip (30 seconds) to test
- Verify the video path is correct before running
- Check the first few frames of output to validate detection
- Adjust area thresholds based on your camera distance
- Use the JSON for data analysis, the video for visual verification

## Example Results Structure

```json
{
  "summary": {
    "total_unique_birds": 15,
    "area_stats": {
      "mean": 8500.5,
      "median": 8200.0
    }
  },
  "individual_birds": {
    "1": {
      "track_length": 150,
      "avg_area": 8500.0,
      "avg_weight_proxy": 0.472
    }
  }
}
```

## System Requirements

- Google Colab (free tier works fine)
- Video file in MP4 format
- Decent internet connection (for model download first time)

## Credits

Built with YOLOv8 object detection and ByteTrack multi-object tracking.

---

That's it! Upload, change 3 lines, run, and get results. Good luck!
