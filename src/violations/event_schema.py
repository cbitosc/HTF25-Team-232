"""
event_schema.py
Defines the canonical data structure for all traffic violation events.

Each violation is represented as a structured dictionary containing:
- Metadata (timestamp, camera, location)
- Violation type and confidence
- Media links (images, clips)
- Optional extra data for explainability

This schema ensures consistency between:
- Violation logic layer (rules.py)
- Evidence generator (evidence_generator.py)
- Backend & dashboard (app.py)
"""

import uuid
import datetime
from typing import Dict, Optional, Tuple


# ============================================================
# Unique ID and timestamp generation
# ============================================================

def generate_violation_id() -> str:
    """
    Generates a unique readable violation ID.
    Example: 'V-12AB34CD'
    """
    uid = uuid.uuid4().hex[:8].upper()
    return f"V-{uid}"


def current_iso_timestamp() -> str:
    """
    Returns the current timestamp in ISO8601 (UTC) format.
    Example: '2025-10-26T14:45:31.200Z'
    """
    return datetime.datetime.utcnow().isoformat() + "Z"


# ============================================================
# Core schema builder
# ============================================================

def build_violation_record(
    violation_type: str,
    camera_id: str,
    location_name: str,
    plate_text: Optional[str],
    confidence: float,
    media_paths: Dict[str, Optional[str]],
    extra: Optional[Dict] = None
) -> Tuple[str, Dict]:
    """
    Builds a standardized violation record dictionary.

    Parameters
    ----------
    violation_type : str
        The type of violation (helmetless, triple_riding, red_light_jump, etc.)
    camera_id : str
        Unique camera identifier (e.g., 'CAM_01')
    location_name : str
        Human-readable camera location (e.g., 'MG Road Intersection')
    plate_text : str
        Vehicle number plate (if recognized) or 'UNREADABLE'
    confidence : float
        Confidence score for the detected violation
    media_paths : dict
        File paths for context image, plate crop, and video clip
        {
            "context_img": "output/violations/V-XXXX/context.jpg",
            "plate_img": "output/violations/V-XXXX/plate.jpg",
            "clip_video": "output/violations/V-XXXX/clip.mp4"
        }
    extra : dict, optional
        Any additional data (rider count, bbox, class labels, etc.)

    Returns
    -------
    (violation_id, record)
        - violation_id : str (unique)
        - record : dict (JSON-compatible data)
    """

    violation_id = generate_violation_id()

    record = {
        "violation_id": violation_id,
        "type": violation_type,
        "timestamp_utc": current_iso_timestamp(),
        "camera_id": camera_id,
        "location": location_name,
        "plate": plate_text if plate_text else "UNREADABLE",
        "confidence": round(float(confidence), 3),
        "media": {
            "context_img": media_paths.get("context_img"),
            "plate_img": media_paths.get("plate_img"),
            "clip_video": media_paths.get("clip_video"),
        },
    }

    if extra:
        record["extra"] = extra

    return violation_id, record


# ============================================================
# Validation helper (optional)
# ============================================================

def validate_violation_record(record: Dict) -> bool:
    """
    Quick validation check for required keys.
    Returns True if valid, False otherwise.
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
        if key not in record:
            print(f" Missing key in record: {key}")
            return False

    return True


# ============================================================
# Quick manual test
# ============================================================

if __name__ == "__main__":
    media = {
        "context_img": "output/violations/V-TEST/context.jpg",
        "plate_img": "output/violations/V-TEST/plate.jpg",
        "clip_video": "output/violations/V-TEST/clip.mp4"
    }

    _, rec = build_violation_record(
        violation_type="helmetless",
        camera_id="CAM_01",
        location_name="MG Road Junction",
        plate_text="KA05MN1234",
        confidence=0.92,
        media_paths=media,
        extra={"riders": 2, "bike_id": 15}
    )

    print("✅ Generated Violation Record:")
    print(rec)
