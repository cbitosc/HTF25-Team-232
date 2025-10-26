"""
rules.py
This module inspects tracked objects for the current frame and returns a list
of violation events.

Input expected by evaluate_frame():
    detections: list of objects, where each object is a dict:
        {
            "track_id": int,
            "class": str,
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "helmet": bool or None,
            "helmet_confidence": float or None
        }

    traffic_light_state: str  -> "RED" / "GREEN" / etc.
    stop_line_polygon: list[(x,y)]  -> polygon representing stop area

Returns:
    List[dict] of violation events. Each event has:
        {
            "violation_type": str,
            "track_id": int,
            "confidence": float,
            "timestamp_utc": "...",
            "details": {...}
        }
"""

from typing import List, Dict, Any
from datetime import datetime
import numpy as np
import cv2


#############
# Utilities #
#############

def _inside_polygon(point, polygon) -> bool:
    """Return True if point (cx, cy) lies inside polygon."""
    poly_np = np.array(polygon, dtype=np.int32)
    # cv2.pointPolygonTest returns >=0 if inside or on edge
    res = cv2.pointPolygonTest(poly_np, (float(point[0]), float(point[1])), False)
    return res >= 0

def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)

def _now_utc_iso():
    return datetime.utcnow().isoformat() + "Z"


##############################
# Violation check functions  #
##############################

def check_helmetless(detections: List[Dict[str, Any]],
                     min_conf: float = 0.6) -> List[Dict[str, Any]]:
    """
    Helmetless riding violation:
    - object.class == 'person'
    - helmet == False
    - helmet_confidence >= min_conf
    """
    events = []

    for obj in detections:
        if obj.get("class") != "person":
            continue

        helmet_flag = obj.get("helmet")
        helmet_conf = obj.get("helmet_confidence", 0.0)

        # skip if we didn't classify / low confidence
        if helmet_flag is None:
            continue

        # helmet_flag == False means "no helmet"
        if helmet_flag is False and float(helmet_conf) >= min_conf:
            evt = {
                "violation_type": "helmetless",
                "track_id": int(obj.get("track_id", -1)),
                "confidence": float(helmet_conf),
                "timestamp_utc": _now_utc_iso(),
                "details": {
                    "bbox": obj.get("bbox"),
                    "class": obj.get("class"),
                    "helmet_confidence": helmet_conf
                }
            }
            events.append(evt)

    return events


def check_triple_riding(detections: List[Dict[str, Any]],
                        iou_threshold: float = 0.3,
                        min_people: int = 3) -> List[Dict[str, Any]]:
    """
    Very simple heuristic:
    - Group people that overlap (IOU) with the same motorbike bbox area.
    - If >=3 people overlap the same bike -> triple riding.
    NOTE: This is a naive placeholder.
    """
    events = []

    # Collect bikes/motorbikes
    bikes = []
    persons = []

    for obj in detections:
        clsname = obj.get("class", "")
        if clsname in ["motorbike", "motorcycle", "bike", "bicycle", "scooter"]:
            bikes.append(obj)
        elif clsname == "person":
            persons.append(obj)

    # helper: IoU
    def iou(b1, b2):
        x1a,y1a,x2a,y2a = b1
        x1b,y1b,x2b,y2b = b2
        inter_x1 = max(x1a, x1b)
        inter_y1 = max(y1a, y1b)
        inter_x2 = min(x2a, x2b)
        inter_y2 = min(y2a, y2b)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter_area = iw * ih
        area_a = (x2a - x1a) * (y2a - y1a)
        area_b = (x2b - x1b) * (y2b - y1b)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    # for each bike, count how many persons overlap enough
    for bike in bikes:
        bike_bbox = bike.get("bbox", [0,0,0,0])
        riders = []
        for p in persons:
            if iou(bike_bbox, p.get("bbox",[0,0,0,0])) >= iou_threshold:
                riders.append(p)

        if len(riders) >= min_people:
            # we mark this as triple riding
            evt = {
                "violation_type": "triple_riding",
                "track_id": int(bike.get("track_id", -1)),
                "confidence": 1.0,
                "timestamp_utc": _now_utc_iso(),
                "details": {
                    "bike_bbox": bike_bbox,
                    "rider_count": len(riders),
                    "rider_ids": [int(r.get("track_id",-1)) for r in riders],
                }
            }
            events.append(evt)

    return events


def check_signal_jump(detections: List[Dict[str, Any]],
                      traffic_light_state: str,
                      stop_line_polygon,
                      min_conf_vehicle: float = 0.4) -> List[Dict[str, Any]]:
    """
    Red light jump:
    - traffic_light_state == "RED"
    - vehicle center is inside stop_line_polygon
    """
    events = []
    if traffic_light_state.upper() != "RED":
        return events

    VEHICLE_CLASSES = [
        "car", "truck", "bus", "motorbike", "motorcycle", "bike", "bicycle", "scooter"
    ]

    for obj in detections:
        clsname = obj.get("class", "")
        conf = float(obj.get("confidence", 0.0))

        if clsname not in VEHICLE_CLASSES:
            continue
        if conf < min_conf_vehicle:
            continue

        center = _bbox_center(obj.get("bbox",[0,0,0,0]))
        if _inside_polygon(center, stop_line_polygon):
            evt = {
                "violation_type": "signal_jump",
                "track_id": int(obj.get("track_id", -1)),
                "confidence": conf,
                "timestamp_utc": _now_utc_iso(),
                "details": {
                    "bbox": obj.get("bbox"),
                    "traffic_light_state": traffic_light_state,
                    "center": center,
                }
            }
            events.append(evt)

    return events


##############################
# Master evaluation function #
##############################

def evaluate_frame(
    detections: List[Dict[str, Any]],
    traffic_light_state: str,
    stop_line_polygon: List[List[int]],
) -> List[Dict[str, Any]]:
    """
    This is the ONLY function run_video.py should call.
    It aggregates all violation checks.

    Returns: list of violation dicts.
    """

    # Debug info (optional for tuning)
    print(f"[rules] evaluating frame with {len(detections)} tracked objects")
    print(f"[rules] traffic_light_state={traffic_light_state}")

    all_events = []

    # 1. Helmetless
    helmet_events = check_helmetless(detections, min_conf=0.6)
    all_events.extend(helmet_events)

    # 2. Triple riding
    triple_events = check_triple_riding(detections, iou_threshold=0.3, min_people=3)
    all_events.extend(triple_events)

    # 3. Signal jump
    jump_events = check_signal_jump(
        detections,
        traffic_light_state=traffic_light_state,
        stop_line_polygon=stop_line_polygon,
        min_conf_vehicle=0.4
    )
    all_events.extend(jump_events)

    # You said: no overspeed, no wrong-lane. Skipped.

    # Optional: print what we found
    if all_events:
        print(f"[rules] violations found: {[e['violation_type'] for e in all_events]}")
    else:
        print("[rules] no violations this frame")

    return all_events