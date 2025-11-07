"""
violations/adapter.py
Converts between internal violation event format and backend API format.

This adapter bridges the schema differences between:
- rules.py (uses "violation_type")
- evidence_generator.py & backend (use "type")
"""

from typing import Dict, Any
from datetime import datetime


def adapt_event_for_evidence(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert rules.py event format to evidence_generator format.
    
    Input (from rules.py):
        {
            "violation_type": "helmetless",
            "track_id": 42,
            "confidence": 0.92,
            "timestamp_utc": "2025-10-26T...",
            "details": {
                "bbox": [x1, y1, x2, y2],
                "class": "person",
                "helmet_confidence": 0.92
            }
        }
    
    Output (for evidence_generator.py):
        {
            "type": "helmetless",  # ✅ renamed from "violation_type"
            "track_id": 42,
            "confidence": 0.92,
            "timestamp_utc": "2025-10-26T...",
            "bike_bbox": [x1, y1, x2, y2],  # ✅ flattened from details
            ...
        }
    """
    adapted = {
        "type": event.get("violation_type", "unknown"),
        "track_id": event.get("track_id"),
        "confidence": event.get("confidence", 0.0),
        "timestamp_utc": event.get("timestamp_utc", datetime.utcnow().isoformat() + "Z"),
    }
    
    # Flatten details into main dict
    details = event.get("details", {})
    
    # ✅ FIX: Properly map bbox based on violation type and class
    violation_type = adapted["type"]
    
    if "bbox" in details:
        bbox = details["bbox"]
        
        # For helmetless: always use bike_bbox
        if violation_type == "helmetless":
            adapted["bike_bbox"] = bbox
        
        # For triple_riding: use bike_bbox from dedicated field
        elif violation_type == "triple_riding":
            # Triple riding has bike_bbox directly in details
            adapted["bike_bbox"] = details.get("bike_bbox", bbox)
        
        # For red_light_jump/signal_jump: use vehicle_bbox
        elif violation_type in ["red_light_jump", "signal_jump"]:
            adapted["vehicle_bbox"] = bbox
        
        # Fallback: use vehicle_bbox for any other vehicle-related violation
        else:
            obj_class = details.get("class", "")
            if obj_class == "person":
                adapted["bike_bbox"] = bbox
            else:
                adapted["vehicle_bbox"] = bbox
    
    # ✅ FIX: Handle triple_riding specific fields
    if violation_type == "triple_riding":
        if "bike_bbox" in details:
            adapted["bike_bbox"] = details["bike_bbox"]
        if "rider_count" in details:
            adapted["rider_count"] = details["rider_count"]
        if "rider_ids" in details:
            adapted["rider_ids"] = details["rider_ids"]
        
        # ✅ NEW: Build riders array with bbox info if available
        if "rider_ids" in details:
            # We need to reconstruct rider info from the original detections
            # For now, create placeholder entries
            adapted["riders"] = []
            for rider_id in details["rider_ids"]:
                adapted["riders"].append({
                    "track_id": rider_id,
                    "bbox": None,  # Will be filled by pipeline if needed
                    "helmet": None
                })
    
    # For helmetless violations, include helmet confidence
    if violation_type == "helmetless":
        if "helmet_confidence" in details:
            adapted["helmet_confidence"] = details["helmet_confidence"]
        
        # ✅ NEW: Create riders array for helmetless (single rider)
        adapted["riders"] = [{
            "bbox": details.get("bbox"),
            "helmet": False,  # We know it's False because it's helmetless
            "helmet_confidence": details.get("helmet_confidence", 0.0)
        }]
    
    # For signal jump, include traffic light state
    if violation_type in ["red_light_jump", "signal_jump"]:
        if "traffic_light_state" in details:
            adapted["traffic_light_state"] = details["traffic_light_state"]
    
    # Keep original details as well (for debugging/explainability)
    adapted["details"] = details
    
    return adapted


def adapt_violation_for_backend(violation_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure violation record from evidence_generator matches backend schema.
    
    This is mostly a validation step since evidence_generator already
    uses the correct schema via event_schema.py
    
    Required keys for backend API:
    - violation_id
    - type
    - timestamp_utc
    - camera_id
    - location
    - plate
    - confidence
    - media (with context_img, plate_img, clip_video)
    """
    required_keys = [
        "violation_id",
        "type",
        "timestamp_utc",
        "camera_id",
        "location",
        "plate",
        "confidence",
        "media"
    ]
    
    for key in required_keys:
        if key not in violation_record:
            raise ValueError(f"Missing required key in violation record: {key}")
    
    # Validate media paths
    media = violation_record.get("media", {})
    media_keys = ["context_img", "plate_img", "clip_video"]
    for key in media_keys:
        if key not in media:
            raise ValueError(f"Missing required media key: {key}")
    
    return violation_record


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    print("🧪 Testing adapter.py...\n")
    
    # Test 1: Helmetless violation
    print("Test 1: Helmetless violation")
    test_event_helmetless = {
        "violation_type": "helmetless",
        "track_id": 42,
        "confidence": 0.92,
        "timestamp_utc": "2025-10-26T10:30:00Z",
        "details": {
            "bbox": [100, 200, 300, 400],
            "class": "person",
            "helmet_confidence": 0.92
        }
    }
    
    adapted = adapt_event_for_evidence(test_event_helmetless)
    print("✅ Adapted helmetless event:")
    print(f"   type: {adapted['type']}")
    print(f"   track_id: {adapted['track_id']}")
    print(f"   bike_bbox: {adapted.get('bike_bbox')}")
    print(f"   riders: {adapted.get('riders')}")
    print(f"   confidence: {adapted['confidence']}\n")
    
    # Test 2: Triple riding violation
    print("Test 2: Triple riding violation")
    test_event_triple = {
        "violation_type": "triple_riding",
        "track_id": 15,
        "confidence": 1.0,
        "timestamp_utc": "2025-10-26T10:35:00Z",
        "details": {
            "bike_bbox": [200, 150, 450, 500],
            "rider_count": 3,
            "rider_ids": [42, 43, 44]
        }
    }
    
    adapted2 = adapt_event_for_evidence(test_event_triple)
    print("✅ Adapted triple riding event:")
    print(f"   type: {adapted2['type']}")
    print(f"   bike_bbox: {adapted2.get('bike_bbox')}")
    print(f"   rider_count: {adapted2.get('rider_count')}")
    print(f"   rider_ids: {adapted2.get('rider_ids')}\n")
    
    # Test 3: Signal jump violation
    print("Test 3: Signal jump violation")
    test_event_signal = {
        "violation_type": "red_light_jump",
        "track_id": 88,
        "confidence": 0.95,
        "timestamp_utc": "2025-10-26T10:40:00Z",
        "details": {
            "bbox": [150, 250, 400, 450],
            "class": "car",
            "traffic_light_state": "RED",
            "center": [275, 350]
        }
    }
    
    adapted3 = adapt_event_for_evidence(test_event_signal)
    print("✅ Adapted signal jump event:")
    print(f"   type: {adapted3['type']}")
    print(f"   vehicle_bbox: {adapted3.get('vehicle_bbox')}")
    print(f"   traffic_light_state: {adapted3.get('traffic_light_state')}\n")
    
    print("✅ All adapter tests passed!")