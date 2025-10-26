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
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from detection.detector import YOLODetector
from detection.tracker import SimpleTrackerWrapper
from detection.helmet_classifier import HelmetClassifier
from violations.rules import evaluate_frame
from backend.integration_helper import send_violation_to_backend

# output directories
os.makedirs("output/debug_tracking/json", exist_ok=True)
os.makedirs("output/debug_tracking/video_overlays", exist_ok=True)
os.makedirs("output/violations", exist_ok=True)

def to_serializable(obj):
    """
    Convert anything (numpy types, numpy arrays) into plain Python types
    so json.dump won't crash.
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

# polygons (example stop-line)
STOP_LINE_POLYGON = [(100, 500), (400, 500), (400, 520), (100, 520)]
TRAFFIC_LIGHT_STATE = "RED"  # simulate red light

# initialize models
detector = YOLODetector(weights_path="yolov8n.pt")
tracker = SimpleTrackerWrapper()
helmet_model = HelmetClassifier()

# input video
VIDEO_PATH = "data/raw/intersection_test.mp4"  # <-- make sure this matches your actual file name
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Could not open video at {VIDEO_PATH}")
else:
    print(f"✅ Opened video at {VIDEO_PATH}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None
frame_idx = 0

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
            # keep keys so downstream code doesn't KeyError
            obj["helmet"] = obj.get("helmet", None)
            obj["helmet_confidence"] = obj.get("helmet_confidence", None)

    # 3️⃣ Violation logic
    violations = evaluate_frame(
        detections=tracked_objects,
        traffic_light_state=TRAFFIC_LIGHT_STATE,
        stop_line_polygon=STOP_LINE_POLYGON,
    )

    # 4️⃣ Visual overlay
    for t in tracked_objects:
        x1, y1, x2, y2 = t["bbox"]
        color = (0, 255, 0)
        if t["class"] == "person" and t.get("helmet") is False:
            color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{t['class']}:{t['track_id']}"
        if t["helmet"] is not None:
            label += f" {'H' if t['helmet'] else 'NH'}"
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    pts = np.array(STOP_LINE_POLYGON, np.int32)
    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
    cv2.putText(frame, f"Light: {TRAFFIC_LIGHT_STATE}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter("output/debug_tracking/video_overlays/demo_output.mp4",
                              fourcc, 20.0, (w, h))
    out.write(frame)

    # convert tracked_objects and violations into JSON-safe plain types
safe_objects = []
for obj in tracked_objects:
    safe_obj = {
        "track_id": int(obj.get("track_id", -1)),
        "class": str(obj.get("class", "")),
        "bbox": [int(v) for v in obj.get("bbox", [0,0,0,0])],
        "confidence": float(obj.get("confidence", 0.0)),
        "helmet": (None if obj.get("helmet") is None else bool(obj.get("helmet"))),
        "helmet_confidence": (
            None if obj.get("helmet_confidence") is None
            else float(obj.get("helmet_confidence"))
        ),
    }
    safe_objects.append(safe_obj)

safe_violations = []
for v in violations:
    safe_v = {
        "violation_type": str(v.get("violation_type", "")),
        "track_id": int(v.get("track_id", -1)),
        "confidence": float(v.get("confidence", 0.0)),
        "timestamp_utc": str(v.get("timestamp_utc", "")),
        "details": v.get("details", {}),  # assuming only simple types inside
    }
    safe_violations.append(safe_v)

frame_json = {
    "frame": int(frame_idx),
    "timestamp": datetime.utcnow().isoformat(),  # still fine for now
    "traffic_light_state": str(TRAFFIC_LIGHT_STATE),
    "objects": safe_objects,
    "violations": safe_violations,
}

json_path = f"output/debug_tracking/json/frame_{frame_idx:06d}.json"
with open(json_path, "w") as f:
    json.dump(frame_json, f, indent=2, default=to_serializable)

    # 6️⃣ Send any new violations to backend
    for v in violations:
        send_violation_to_backend(v)

    frame_idx += 1

cap.release()
if out:
    out.release()

print(f"Total frames processed: {frame_idx}")
print("✅ Pipeline completed. Check output folders for results.")