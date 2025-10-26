"""
run_video.py
Main video processing pipeline.
Uses COCO YOLO detector + small helmet classifier + rule engine.
"""

import os
import cv2
import json
import numpy as np
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from detection.detector import YOLODetector
from detection.tracker import SimpleTrackerWrapper
from detection.helmet_classifier import HelmetClassifier
from violations.rules import evaluate_frame
from violations.evidence_generator import save_violation_package
from backend.integration_helper import send_violation_to_backend, check_backend_health

# Output directories
os.makedirs("output/debug_tracking/json", exist_ok=True)
os.makedirs("output/debug_tracking/video_overlays", exist_ok=True)
os.makedirs("output/violations", exist_ok=True)

def to_serializable(obj):
    """
    Convert numpy types to plain Python types for JSON serialization.
    """
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set,)):
        return list(obj)
    return obj

# Configuration
CAMERA_ID = "CAM_01"
LOCATION_NAME = "MG Road Junction"
VIDEO_PATH = "data/raw/intersection_test.mp4"
OUTPUT_DIR = "output/violations"

# Polygons (example stop-line)
STOP_LINE_POLYGON = [(100, 500), (400, 500), (400, 520), (100, 520)]
TRAFFIC_LIGHT_STATE = "RED"  # simulate red light

# Initialize models
print("🚀 Initializing models...")
detector = YOLODetector(weights_path="yolov8n.pt")
tracker = SimpleTrackerWrapper()
helmet_model = HelmetClassifier()

# Check backend connection
backend_available = check_backend_health()
if backend_available:
    print("✅ Backend is connected")
else:
    print("⚠️  Backend is not available - violations will be saved locally only")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Could not open video at {VIDEO_PATH}")
    exit(1)
else:
    print(f"✅ Opened video at {VIDEO_PATH}")

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None
frame_idx = 0

# Frame buffer for evidence clips (stores last 45 frames = ~3 seconds at 15fps)
frame_buffer = []
MAX_BUFFER_SIZE = 45

# Track which violations we've already processed to avoid duplicates
processed_violations = set()

print("🎬 Starting video processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Detection + tracking
    raw_dets = detector.detect(frame)
    tracked_objects = tracker.update(raw_dets, frame)

    # 2️⃣ Helmet inference for persons
    for obj in tracked_objects:
        if obj["class"] == "person":
            helmet, h_conf = helmet_model.classify_person_crop(frame, obj["bbox"])
            obj["helmet"] = helmet
            obj["helmet_confidence"] = h_conf
        else:
            # Keep keys so downstream code doesn't KeyError
            obj["helmet"] = None
            obj["helmet_confidence"] = None

    # 3️⃣ Violation logic
    violation_events = evaluate_frame(
        detections=tracked_objects,
        traffic_light_state=TRAFFIC_LIGHT_STATE,
        stop_line_polygon=STOP_LINE_POLYGON,
    )

    # 4️⃣ Visual overlay for debug video
    vis_frame = frame.copy()
    
    # Create a set of track IDs that are violating for quick lookup
    violating_track_ids = set()
    violation_types_map = {}  # track_id -> violation_type
    
    for event in violation_events:
        track_id = event.get("track_id")
        if track_id is not None:
            violating_track_ids.add(track_id)
            violation_types_map[track_id] = event.get("violation_type", "")
    
    # Draw all tracked objects
    for t in tracked_objects:
        x1, y1, x2, y2 = t["bbox"]
        track_id = t.get("track_id")
        
        # Default color: green
        color = (0, 255, 0)
        thickness = 2
        
        # Check if this object is violating
        if track_id in violating_track_ids:
            violation_type = violation_types_map.get(track_id, "")
            
            # Red for violations
            color = (0, 0, 255)  # BGR: Red
            thickness = 3  # Thicker box for violations
            
            # Add violation label
            violation_label = violation_type.replace("_", " ").upper()
            cv2.putText(vis_frame, f"⚠️ {violation_label}", 
                       (x1, y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Special handling for persons without helmets
        elif t["class"] == "person" and t.get("helmet") is False:
            color = (0, 0, 255)  # Red
            thickness = 3
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Build label text
        label = f"{t['class']}:{t['track_id']}"
        if t["helmet"] is not None:
            label += f" {'H' if t['helmet'] else 'NH'}"
        
        # Draw label
        cv2.putText(vis_frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw stop line polygon with label
    pts = np.array(STOP_LINE_POLYGON, np.int32)
    cv2.polylines(vis_frame, [pts], True, (0, 255, 255), 2)
    cv2.putText(vis_frame, "STOP LINE", 
                (STOP_LINE_POLYGON[0][0], STOP_LINE_POLYGON[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Draw traffic light state indicator
    light_color = (0, 0, 255) if TRAFFIC_LIGHT_STATE == "RED" else (0, 255, 0)
    cv2.putText(vis_frame, f"Light: {TRAFFIC_LIGHT_STATE}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, light_color, 2)
    
    # Draw violation count
    if violation_events:
        cv2.putText(vis_frame, f"Violations: {len(violation_events)}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 5️⃣ Maintain frame buffer for evidence clips
    frame_buffer.append((cv2.getTickCount(), frame.copy()))
    if len(frame_buffer) > MAX_BUFFER_SIZE:
        frame_buffer.pop(0)

    # 6️⃣ Process and save new violations
    for event in violation_events:
        # Create unique key to avoid duplicate processing
        violation_key = f"{event['violation_type']}_{event['track_id']}_{frame_idx}"
        
        if violation_key not in processed_violations:
            processed_violations.add(violation_key)
            
            print(f"🚨 Violation detected: {event['violation_type']} (Track ID: {event['track_id']})")
            
            # Generate evidence package
            try:
                violation_record = save_violation_package(
                    frame=frame,
                    past_frames_buffer=frame_buffer.copy(),
                    event=event,
                    camera_id=CAMERA_ID,
                    location_name=LOCATION_NAME,
                    plate_text=None,  # TODO: Integrate ANPR
                    output_root=OUTPUT_DIR
                )
                
                # Send to backend if available
                if backend_available:
                    result = send_violation_to_backend(
                        violation_id=violation_record["violation_id"],
                        vtype=violation_record["type"],
                        timestamp_utc=violation_record["timestamp_utc"],
                        camera_id=violation_record["camera_id"],
                        location=violation_record["location"],
                        plate=violation_record["plate"],
                        confidence=violation_record["confidence"],
                        media_paths=violation_record["media"],
                        extra=violation_record.get("extra")
                    )
                    
                    if result:
                        print(f"✅ Sent to backend: {violation_record['violation_id']}")
                    else:
                        print(f"⚠️  Failed to send to backend: {violation_record['violation_id']}")
            
            except Exception as e:
                print(f"❌ Error processing violation: {e}")

    # 7️⃣ Save frame JSON for debugging
    safe_objects = []
    for obj in tracked_objects:
        safe_obj = {
            "track_id": int(obj.get("track_id", -1)),
            "class": str(obj.get("class", "")),
            "bbox": [int(v) for v in obj.get("bbox", [0, 0, 0, 0])],
            "confidence": float(obj.get("confidence", 0.0)),
            "helmet": (None if obj.get("helmet") is None else bool(obj.get("helmet"))),
            "helmet_confidence": (
                None if obj.get("helmet_confidence") is None
                else float(obj.get("helmet_confidence"))
            ),
        }
        safe_objects.append(safe_obj)

    safe_violations = []
    for v in violation_events:
        safe_v = {
            "violation_type": str(v.get("violation_type", "")),
            "track_id": int(v.get("track_id", -1)),
            "confidence": float(v.get("confidence", 0.0)),
            "timestamp_utc": str(v.get("timestamp_utc", "")),
            "details": v.get("details", {}),
        }
        safe_violations.append(safe_v)

    frame_json = {
        "frame": int(frame_idx),
        "timestamp": datetime.utcnow().isoformat(),
        "traffic_light_state": str(TRAFFIC_LIGHT_STATE),
        "objects": safe_objects,
        "violations": safe_violations,
    }

    json_path = f"output/debug_tracking/json/frame_{frame_idx:06d}.json"
    with open(json_path, "w") as f:
        json.dump(frame_json, f, indent=2, default=to_serializable)

    frame_idx += 1
    
    # Progress indicator
    if frame_idx % 30 == 0:
        print(f"📊 Processed {frame_idx} frames...")

# Cleanup
cap.release()
if out:
    out.release()

print(f"\n✅ Pipeline completed!")
print(f"📊 Total frames processed: {frame_idx}")
print(f"🚨 Total violations detected: {len(processed_violations)}")
print(f"📁 Output saved to: output/")