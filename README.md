
Poultry Monitoring System with Bird Counting, Tracking, and Weight Estimation

Requirements:
- pip install fastapi uvicorn python-multipart
- pip install ultralytics opencv-python numpy pillow
- pip install supervision scikit-learn

Run: uvicorn main:app --reload

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# Deep learning imports
from ultralytics import YOLO
import supervision as sv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Poultry Monitoring API", version="1.0")

# DATA MODELS

@dataclass
class BirdDetection:
    """Single bird detection with tracking info"""
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    frame_idx: int
    timestamp: float

@dataclass
class CountData:
    """Bird count at specific timestamp"""
    timestamp: float
    frame_idx: int
    count: int
    
@dataclass
class WeightEstimate:
    """Weight estimate for a bird or group"""
    track_id: Optional[int]
    weight_value: float
    unit: str  # 'g' or 'index'
    confidence: float
    method: str

@dataclass
class AnalysisResult:
    """Complete analysis output"""
    counts: List[Dict[str, Any]]
    tracks_sample: List[Dict[str, Any]]
    weight_estimates: List[Dict[str, Any]]
    artifacts: Dict[str, str]
    metadata: Dict[str, Any]



# BIRD DETECTOR AND TRACKER

class BirdDetectorTracker:
    """Handles bird detection and tracking using YOLOv8 + ByteTrack"""
    
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.25):
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        
    def detect_and_track(self, frame: np.ndarray) -> sv.Detections:
        """Run detection and tracking on a frame"""
        # Run YOLO detection (class 14 = bird in COCO)
        results = self.model(frame, conf=self.conf_thresh, classes=[14], verbose=False)[0]
        
        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results)
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        return detections

# WEIGHT ESTIMATOR


class WeightEstimator:
    """
    Estimates bird weight using visual features.
    
    Method: Feature-based proxy using bbox area and aspect ratio.
    
    Weight Proxy Formula:
        weight_index = bbox_area * density_factor * scale_correction
    
    To convert to grams, calibration data is needed:
        - Ground truth weights for sample birds
        - Camera height and angle
        - Reference object of known size in frame
        - Pixel-to-cm conversion factor
        
    With calibration:
        weight_g = (weight_index * calibration_factor) + bias
    """
    
    def __init__(self, calibrated: bool = False, 
                 calibration_factor: float = 1.0,
                 calibration_bias: float = 0.0):
        self.calibrated = calibrated
        self.calibration_factor = calibration_factor
        self.calibration_bias = calibration_bias
        
        # Heuristic parameters for weight proxy
        self.density_factor = 1.0
        self.min_area_threshold = 500  # pixels
        
    def estimate_weight_from_bbox(self, bbox: np.ndarray, 
                                  confidence: float) -> WeightEstimate:
        """
        Estimate weight from bounding box dimensions.
        
        Returns weight proxy (relative index) unless calibrated.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = height / width if width > 0 else 1.0
        
        # Weight proxy based on area and shape
        # Taller birds (higher aspect) tend to be heavier
        weight_proxy = area * self.density_factor * (1 + 0.1 * aspect_ratio)
        
        if self.calibrated:
            # Convert to grams using calibration
            weight_g = (weight_proxy * self.calibration_factor) + self.calibration_bias
            unit = "g"
        else:
            weight_g = weight_proxy
            unit = "index"
        
        # Confidence decreases with smaller birds (harder to measure accurately)
        size_confidence = min(area / 5000, 1.0)
        final_confidence = confidence * size_confidence * 0.8
        
        return WeightEstimate(
            track_id=None,
            weight_value=float(weight_g),
            unit=unit,
            confidence=float(final_confidence),
            method="bbox_area_proxy"
        )
    
    def estimate_aggregate_weight(self, detections: sv.Detections) -> List[WeightEstimate]:
        """Estimate weights for all detected birds"""
        estimates = []
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            conf = detections.confidence[i] if detections.confidence is not None else 0.5
            track_id = detections.tracker_id[i] if detections.tracker_id is not None else None
            
            estimate = self.estimate_weight_from_bbox(bbox, conf)
            estimate.track_id = int(track_id) if track_id is not None else None
            estimates.append(estimate)
        
        return estimates


# VIDEO PROCESSOR

class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self, conf_thresh: float = 0.25, iou_thresh: float = 0.45):
        self.detector = BirdDetectorTracker(conf_thresh=conf_thresh)
        self.weight_estimator = WeightEstimator(calibrated=False)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
    def process_video(self, video_path: str, fps_sample: Optional[int] = None,
                     output_dir: str = "/tmp") -> AnalysisResult:
        """
        Process video and generate analysis results.
        
        Occlusion Handling:
        - ByteTrack maintains IDs during brief occlusions using motion prediction
        - Lost tracks buffer keeps IDs for 30 frames before reassignment
        - High matching threshold (0.8) reduces ID switches
        
        Double-counting Prevention:
        - Stable tracking IDs prevent counting same bird multiple times
        - Track activation threshold filters spurious detections
        - Minimum track length filtering (post-process)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Sampling rate
        frame_skip = 1
        if fps_sample and fps_sample < fps:
            frame_skip = int(fps / fps_sample)
        
        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
        logger.info(f"Frame skip: {frame_skip} (effective FPS: {fps/frame_skip:.1f})")
        
        # Output video setup
        output_path = Path(output_dir) / f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps/frame_skip, (width, height))
        
        # Annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)
        
        # Data collection
        counts_data = []
        all_detections = []
        weight_estimates_all = []
        track_history = defaultdict(list)  # track_id -> list of detections
        
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame sampling
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            timestamp = frame_idx / fps
            
            # Detection and tracking
            detections = self.detector.detect_and_track(frame)
            bird_count = len(detections)
            
            # Store count
            counts_data.append(CountData(
                timestamp=timestamp,
                frame_idx=frame_idx,
                count=bird_count
            ))
            
            # Collect detections for sample
            if detections.tracker_id is not None:
                for i in range(len(detections)):
                    track_id = int(detections.tracker_id[i])
                    bbox = detections.xyxy[i].tolist()
                    conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                    
                    detection = BirdDetection(
                        track_id=track_id,
                        bbox=bbox,
                        confidence=conf,
                        frame_idx=frame_idx,
                        timestamp=timestamp
                    )
                    all_detections.append(detection)
                    track_history[track_id].append(detection)
            
            # Weight estimation (every 30 frames to reduce computation)
            if processed_frames % 30 == 0:
                weights = self.weight_estimator.estimate_aggregate_weight(detections)
                weight_estimates_all.extend(weights)
            
            # Annotate frame
            annotated_frame = frame.copy()
            
            if len(detections) > 0:
                # Draw boxes
                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
                
                # Draw labels with track IDs
                labels = []
                if detections.tracker_id is not None:
                    for i in range(len(detections)):
                        track_id = int(detections.tracker_id[i])
                        conf = detections.confidence[i] if detections.confidence is not None else 0
                        labels.append(f"ID:{track_id} {conf:.2f}")
                else:
                    labels = [f"{conf:.2f}" for conf in detections.confidence]
                
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )
            
            # Add count overlay
            cv2.rectangle(annotated_frame, (10, 10), (300, 80), (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"Bird Count: {bird_count}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Time: {timestamp:.1f}s", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            out.write(annotated_frame)
            
            processed_frames += 1
            frame_idx += 1
            
            if processed_frames % 100 == 0:
                logger.info(f"Processed {processed_frames} frames...")
        
        cap.release()
        out.release()
        
        # Post-process: Filter short tracks (likely false positives)
        min_track_length = 5
        valid_tracks = {tid for tid, dets in track_history.items() 
                       if len(dets) >= min_track_length}
        
        # Sample representative tracks
        tracks_sample = []
        for track_id in sorted(valid_tracks)[:20]:  # Top 20 tracks
            track_dets = track_history[track_id]
            # Take middle detection as representative
            rep_det = track_dets[len(track_dets)//2]
            tracks_sample.append(asdict(rep_det))
        
        # Aggregate weight estimates by track
        weight_by_track = defaultdict(list)
        for w in weight_estimates_all:
            if w.track_id in valid_tracks:
                weight_by_track[w.track_id].append(w)
        
        final_weights = []
        for track_id, weights in weight_by_track.items():
            if weights:
                avg_weight = np.mean([w.weight_value for w in weights])
                avg_conf = np.mean([w.confidence for w in weights])
                final_weights.append(WeightEstimate(
                    track_id=track_id,
                    weight_value=float(avg_weight),
                    unit=weights[0].unit,
                    confidence=float(avg_conf),
                    method="bbox_area_proxy_averaged"
                ))
        
        # Create result
        result = AnalysisResult(
            counts=[asdict(c) for c in counts_data],
            tracks_sample=tracks_sample,
            weight_estimates=[asdict(w) for w in final_weights],
            artifacts={
                "annotated_video": str(output_path),
                "video_exists": output_path.exists()
            },
            metadata={
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "fps": fps,
                "effective_fps": fps/frame_skip,
                "resolution": [width, height],
                "unique_tracks": len(valid_tracks),
                "avg_bird_count": float(np.mean([c.count for c in counts_data])),
                "max_bird_count": max([c.count for c in counts_data]),
                "weight_calibration_needed": not self.weight_estimator.calibrated,
                "calibration_requirements": [
                    "Ground truth weights for 10+ sample birds",
                    "Reference object of known size in frame (e.g., 10cm marker)",
                    "Camera height and angle measurements",
                    "Pixel-to-cm conversion factor"
                ]
            }
        )
        
        return result



# API ENDPOINTS

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK", "service": "Poultry Monitoring System"}

@app.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
    fps_sample: Optional[int] = Form(None),
    conf_thresh: Optional[float] = Form(0.25),
    iou_thresh: Optional[float] = Form(0.45)
):
    """
    Analyze poultry video for bird counting, tracking, and weight estimation.
    
    Parameters:
    - file: Video file (mp4, avi, etc.)
    - fps_sample: Optional frame sampling rate (e.g., 5 = process 5 FPS)
    - conf_thresh: Detection confidence threshold (0-1)
    - iou_thresh: IoU threshold for NMS (0-1)
    
    Returns:
    - counts: Time series of bird counts
    - tracks_sample: Sample of tracked birds with IDs and bboxes
    - weight_estimates: Weight estimates per bird (unit: index or g)
    - artifacts: Generated output file paths
    - metadata: Processing statistics and calibration requirements
    """
    
    # Validate file type
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')):
        raise HTTPException(status_code=400, detail="Invalid file type. Use mp4, avi, or mov.")
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # Process video
        logger.info(f"Processing video: {file.filename}")
        processor = VideoProcessor(conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        
        result = processor.process_video(
            video_path=tmp_path,
            fps_sample=fps_sample,
            output_dir=tempfile.gettempdir()
        )
        
        return JSONResponse(content=asdict(result))
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)

@app.get("/download_artifact/{filename}")
async def download_artifact(filename: str):
    """Download generated artifacts (annotated videos, etc.)"""
    file_path = Path(tempfile.gettempdir()) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename
    )

# MAIN

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
