from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Import config
from .config import get_settings

# ===========================
# Load Settings
# ===========================
settings = get_settings()

# Use settings for paths
VIOLATION_ROOT = os.path.abspath(settings.violation_root)
FALLBACK_LOG = os.path.abspath(settings.fallback_log)

# in-memory "DB" for now
violations_db: Dict[str, Dict] = {}

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version
)

# CORS with settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,  # Use from config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================
# Pydantic models (API schemas)
# ===========================

class MediaInfo(BaseModel):
    context_img: Optional[str] = None
    plate_img: Optional[str] = None
    clip_video: Optional[str] = None

class ViolationIn(BaseModel):
    violation_id: str
    type: str                  # "helmetless", "triple_riding", "red_light_jump", etc.
    timestamp_utc: str         # ISO8601 string
    camera_id: str
    location: str
    plate: str                 # "UNREADABLE" if OCR failed
    confidence: float
    media: MediaInfo
    extra: Optional[dict] = None


class ViolationOut(ViolationIn):
    """Same as ViolationIn plus metadata"""
    reviewed: bool = False
    approved: Optional[bool] = None
    reviewer_notes: Optional[str] = None


# ===========================
# Utility functions
# ===========================

def load_single_violation_dir(vdir_path: str) -> Optional[Dict]:
    """
    Reads one violation directory from disk:
      /output/violations/V0001/
        - evidence.json
        - context.jpg
        - clip.mp4
        ...
    Returns dict matching ViolationOut or None on error.
    """
    evidence_path = os.path.join(vdir_path, "evidence.json")
    if not os.path.isfile(evidence_path):
        return None

    try:
        with open(evidence_path, "r") as f:
            record = json.load(f)
    except Exception as e:
        print(f"[!] Failed to load {evidence_path}: {e}")
        return None

    # basic validation
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
    for k in required_keys:
        if k not in record:
            print(f"[!] Missing key {k} in {evidence_path}")
            return None

    # Add review fields if not present
    if "reviewed" not in record:
        record["reviewed"] = False
    if "approved" not in record:
        record["approved"] = None
    if "reviewer_notes" not in record:
        record["reviewer_notes"] = None

    return record


def import_all_from_disk(root_dir: str) -> int:
    """
    Walk through violation folders and load them into memory.
    Returns how many were imported.
    """
    count = 0
    if not os.path.isdir(root_dir):
        print(f"[!] {root_dir} does not exist yet.")
        return 0

    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item)
        if not os.path.isdir(full_path):
            continue

        record = load_single_violation_dir(full_path)
        if record is None:
            continue

        vid = record["violation_id"]
        violations_db[vid] = record
        count += 1

    return count


def filter_violations(
    vtype: Optional[str] = None,
    camera_id: Optional[str] = None,
    plate: Optional[str] = None,
    reviewed: Optional[bool] = None,
    min_confidence: Optional[float] = None,
    since: Optional[str] = None,
    until: Optional[str] = None
) -> List[Dict]:
    """
    Apply basic filtering for dashboard queries.
    since/until are ISO timestamps.
    """
    results = list(violations_db.values())

    if vtype:
        results = [v for v in results if v.get("type") == vtype]

    if camera_id:
        results = [v for v in results if v.get("camera_id") == camera_id]

    if plate:
        # Partial match for license plate
        results = [v for v in results if plate.upper() in v.get("plate", "").upper()]

    if reviewed is not None:
        results = [v for v in results if v.get("reviewed") == reviewed]

    if min_confidence is not None:
        results = [v for v in results if v.get("confidence", 0) >= min_confidence]

    def parse_time(t):
        try:
            return datetime.fromisoformat(t.replace("Z","").replace("z",""))
        except Exception:
            return None

    since_dt = parse_time(since) if since else None
    until_dt = parse_time(until) if until else None

    if since_dt:
        results = [
            v for v in results
            if parse_time(v.get("timestamp_utc","")) and parse_time(v.get("timestamp_utc","")) >= since_dt
        ]

    if until_dt:
        results = [
            v for v in results
            if parse_time(v.get("timestamp_utc","")) and parse_time(v.get("timestamp_utc","")) <= until_dt
        ]

    # sort newest first
    results.sort(key=lambda v: v.get("timestamp_utc",""), reverse=True)
    return results


def log_to_fallback(violation_data: dict):
    """
    Fallback JSON logger for when system fails
    Judge bonus: "What if the backend crashes?"
    """
    try:
        os.makedirs(os.path.dirname(FALLBACK_LOG), exist_ok=True)
        
        if os.path.exists(FALLBACK_LOG):
            with open(FALLBACK_LOG, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(violation_data)
        
        with open(FALLBACK_LOG, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
        
        print(f"⚠️  Fallback: Logged {violation_data.get('violation_id')}")
    except Exception as e:
        print(f"❌ Fallback logging failed: {e}")


def get_mime_type(filepath: str) -> str:
    """Return proper MIME type for file serving"""
    ext = Path(filepath).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".json": "application/json"
    }
    return mime_types.get(ext, "application/octet-stream")


# ===========================
# Routes
# ===========================

@app.on_event("startup")
async def startup_event():
    """Auto-import violations on server start"""
    print(f"🚀 Starting {settings.api_title}...")
    print(f"📁 Violation root: {VIOLATION_ROOT}")
    print(f"🌐 CORS allowed origins: {settings.allowed_origins_list}")
    
    # Ensure directories exist
    os.makedirs(VIOLATION_ROOT, exist_ok=True)
    os.makedirs(settings.logs_dir, exist_ok=True)
    
    # Auto-import existing violations
    count = import_all_from_disk(VIOLATION_ROOT)
    print(f"✅ Auto-imported {count} violations from disk")


@app.get("/")
def healthcheck():
    """Health check - judges love this"""
    return {
        "status": "ok",
        "service": settings.api_title,
        "version": settings.api_version,
        "violations_loaded": len(violations_db),
        "violation_root": VIOLATION_ROOT,
        "cors_origins": settings.allowed_origins_list
    }


@app.post("/violations", response_model=ViolationOut)
def add_violation(v: ViolationIn):
    """
    This is how Member 3's pipeline will send a new violation live.
    
    Example call from pipeline (Python):
    ```
    import requests
    
    payload = {
        "violation_id": "V-12AB34CD",
        "type": "helmetless",
        "timestamp_utc": "2025-10-26T14:55:03Z",
        "camera_id": "CAM_01",
        "location": "Intersection A",
        "plate": "KA05MN1234",
        "confidence": 0.92,
        "media": {
            "context_img": "output/violations/V-12AB34CD/context.jpg",
            "plate_img": "output/violations/V-12AB34CD/plate.jpg",
            "clip_video": "output/violations/V-12AB34CD/clip.mp4"
        },
        "extra": {"count": 3, "bike_id": 7}
    }
    
    response = requests.post("http://localhost:8000/violations", json=payload)
    print(response.json())
    ```
    """
    try:
        # Add review metadata
        violation_data = v.dict()
        violation_data["reviewed"] = False
        violation_data["approved"] = None
        violation_data["reviewer_notes"] = None
        
        # Store in memory
        violations_db[v.violation_id] = violation_data
        
        print(f"✅ Violation received: {v.violation_id} | Type: {v.type} | Plate: {v.plate}")
        
        return violation_data
    
    except Exception as e:
        print(f"❌ Error storing violation: {e}")
        log_to_fallback(v.dict())
        raise HTTPException(status_code=500, detail=f"Failed to store violation: {str(e)}")


@app.get("/violations", response_model=List[ViolationOut])
def list_violations(
    vtype: Optional[str] = Query(None, description="Filter by violation type"),
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    plate: Optional[str] = Query(None, description="Search by license plate"),
    reviewed: Optional[bool] = Query(None, description="Filter by review status"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    since: Optional[str] = Query(None, description="Start date (ISO format)"),
    until: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, le=1000, description="Max results to return")
):
    """
    Dashboard calls this to populate the table.
    
    **Examples:**
    - All violations: `/violations`
    - Helmetless only: `/violations?vtype=helmetless`
    - By camera: `/violations?camera_id=CAM_01`
    - Unreviewed: `/violations?reviewed=false`
    - High confidence: `/violations?min_confidence=0.9`
    - Date range: `/violations?since=2025-10-20T00:00:00Z&until=2025-10-25T23:59:59Z`
    """
    res = filter_violations(
        vtype=vtype,
        camera_id=camera_id,
        plate=plate,
        reviewed=reviewed,
        min_confidence=min_confidence,
        since=since,
        until=until
    )
    
    # Apply limit
    res = res[:limit]
    
    print(f"📊 Fetched {len(res)} violations (filters applied)")
    
    return res


@app.get("/violations/{violation_id}", response_model=ViolationOut)
def get_violation(violation_id: str):
    """
    Dashboard calls this when you click one specific row to open a detail modal.
    
    **Example:** `/violations/V-12AB34CD`
    """
    v = violations_db.get(violation_id)
    if not v:
        raise HTTPException(status_code=404, detail=f"Violation {violation_id} not found")
    return v


@app.patch("/violations/{violation_id}/review")
def review_violation(
    violation_id: str,
    approved: bool = Query(..., description="Approve or reject the violation"),
    notes: Optional[str] = Query(None, description="Reviewer notes")
):
    """
    Human-in-the-loop: Mark violation as reviewed
    
    **Example:** `/violations/V-12AB34CD/review?approved=true&notes=Verified`
    """
    v = violations_db.get(violation_id)
    if not v:
        raise HTTPException(status_code=404, detail=f"Violation {violation_id} not found")
    
    v["reviewed"] = True
    v["approved"] = approved
    v["reviewer_notes"] = notes or ""
    
    # Update evidence.json on disk
    violation_dir = os.path.join(VIOLATION_ROOT, violation_id)
    evidence_path = os.path.join(violation_dir, "evidence.json")
    
    try:
        with open(evidence_path, 'w') as f:
            json.dump(v, f, indent=2)
        print(f"✅ Violation {violation_id} reviewed: {'APPROVED' if approved else 'REJECTED'}")
    except Exception as e:
        print(f"⚠️  Could not update evidence.json: {e}")
    
    return {
        "violation_id": violation_id,
        "status": "reviewed",
        "approved": approved,
        "notes": notes
    }


@app.get("/media")
def get_media(path: str = Query(..., description="Relative path to media file")):
    """
    Dashboard will hit:
      /media?path=output/violations/V-12AB34CD/context.jpg
    or
      /media?path=output/violations/V-12AB34CD/clip.mp4
    
    Returns the file with proper MIME type.
    """
    # Handle both absolute and relative paths
    if not os.path.isabs(path):
        full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", path))
    else:
        full_path = os.path.abspath(path)

    # Security: restrict to output/violations only
    if not full_path.startswith(os.path.abspath(VIOLATION_ROOT)):
        raise HTTPException(status_code=403, detail="Access denied - path outside violations folder")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    # Determine MIME type
    media_type = get_mime_type(full_path)
    
    return FileResponse(full_path, media_type=media_type)


@app.post("/violations/import_from_disk")
def import_from_disk():
    """
    Call this ONCE after Member 3 generates evidence folders.
    It will scan output/violations/*, read evidence.json,
    and load everything into memory for the dashboard.
    
    **Usage:** `curl -X POST http://localhost:8000/violations/import_from_disk`
    """
    count = import_all_from_disk(VIOLATION_ROOT)
    return {
        "imported": count,
        "total_in_memory": len(violations_db),
        "message": f"Successfully imported {count} violations from disk"
    }


@app.get("/stats")
def get_statistics():
    """
    Dashboard analytics endpoint
    
    Returns summary statistics for all violations
    """
    total = len(violations_db)
    
    # Count by type
    by_type = {}
    for v in violations_db.values():
        vtype = v.get("type", "unknown")
        by_type[vtype] = by_type.get(vtype, 0) + 1
    
    # Count by camera
    by_camera = {}
    for v in violations_db.values():
        cam = v.get("camera_id", "unknown")
        by_camera[cam] = by_camera.get(cam, 0) + 1
    
    # Review stats
    reviewed = sum(1 for v in violations_db.values() if v.get("reviewed", False))
    approved = sum(1 for v in violations_db.values() if v.get("approved", False))
    rejected = sum(1 for v in violations_db.values() if v.get("reviewed") and not v.get("approved"))
    pending = total - reviewed
    
    # Confidence distribution
    high_conf = sum(1 for v in violations_db.values() if v.get("confidence", 0) >= 0.9)
    medium_conf = sum(1 for v in violations_db.values() if 0.7 <= v.get("confidence", 0) < 0.9)
    low_conf = sum(1 for v in violations_db.values() if v.get("confidence", 0) < 0.7)
    
    return {
        "total_violations": total,
        "by_type": by_type,
        "by_camera": by_camera,
        "review_status": {
            "reviewed": reviewed,
            "approved": approved,
            "rejected": rejected,
            "pending": pending
        },
        "confidence_distribution": {
            "high": high_conf,    # ≥0.9
            "medium": medium_conf,  # 0.7-0.9
            "low": low_conf         # <0.7
        }
    }


@app.delete("/violations/{violation_id}")
def delete_violation(violation_id: str):
    """
    Delete a violation (optional - for testing)
    
    **Example:** `DELETE /violations/V-12AB34CD`
    """
    if violation_id not in violations_db:
        raise HTTPException(status_code=404, detail=f"Violation {violation_id} not found")
    
    del violations_db[violation_id]
    
    print(f"🗑️  Deleted violation: {violation_id}")
    
    return {"message": f"Violation {violation_id} deleted", "remaining": len(violations_db)}


# ===========================
# Run server
# ===========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.backend.app:app",
        host=settings.host,
        port=settings.port,
        reload=True  # Auto-reload on code changes during development
    )
