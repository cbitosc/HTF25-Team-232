"""
backend_server.py
FastAPI backend for traffic violations management

Run with: uvicorn backend_server:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
import json
import os

# ============================================================
# Helper: Convert numpy types to native Python types
# ============================================================
def to_serializable(obj):
    """
    Recursively convert numpy types to native Python types.
    This prevents 'Object of type int64 is not JSON serializable' errors.
    """
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


# ============================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================
class MediaInfo(BaseModel):
    context_img: Optional[str] = None
    crop_img: Optional[str] = None
    plate_img: Optional[str] = None
    clip_video: Optional[str] = None


class ViolationDetails(BaseModel):
    bbox: Optional[List[float]] = None
    camera_id: Optional[str] = None
    location: Optional[str] = None
    plate: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class ViolationCreate(BaseModel):
    violation_type: str
    track_id: int
    confidence: float
    timestamp_utc: str
    details: ViolationDetails
    media: MediaInfo
    status: str = "Pending"


class ViolationResponse(BaseModel):
    id: str
    violation_type: str
    track_id: int
    confidence: float
    timestamp_utc: str
    details: Dict[str, Any]
    media: Dict[str, Any]
    status: str
    # Flattened fields for dashboard convenience
    plate: Optional[str] = None
    type: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    image: Optional[str] = None  # Primary image for preview


class StatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(Pending|Approved|Rejected)$")


# ============================================================
# In-Memory Storage
# ============================================================
violations_store: Dict[str, Dict[str, Any]] = {}


# ============================================================
# FastAPI App Setup
# ============================================================
app = FastAPI(
    title="Traffic Violations API",
    description="Backend API for traffic violation detection system",
    version="1.0.0"
)

# CORS middleware - allows Streamlit frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helper Functions
# ============================================================
def generate_violation_id(violation_type: str, track_id: int, timestamp_utc: str) -> str:
    """Generate unique violation ID"""
    # Clean timestamp for filename-safe format
    clean_timestamp = timestamp_utc.replace(":", "-").replace(".", "-").replace("Z", "")
    return f"vio_{clean_timestamp}_{violation_type}_{track_id}"


def flatten_violation(violation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten violation data for easier frontend consumption.
    Adds convenience fields like 'plate', 'type', 'time', 'location', 'lat', 'lon', 'image'
    """
    flattened = violation_data.copy()
    
    # Flatten commonly used fields from details
    details = violation_data.get("details", {})
    flattened["plate"] = details.get("plate", "UNKNOWN")
    flattened["location"] = details.get("location", "Unknown Location")
    flattened["lat"] = details.get("lat")
    flattened["lon"] = details.get("lon")
    
    # Map violation_type to type for consistency
    flattened["type"] = violation_data.get("violation_type", "Unknown")
    
    # Format time for display
    timestamp = violation_data.get("timestamp_utc", "")
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", ""))
        flattened["time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        flattened["time"] = timestamp
    
    # Set primary image for preview (prefer context_img, fallback to crop_img)
    media = violation_data.get("media", {})
    flattened["image"] = media.get("context_img") or media.get("crop_img")
    
    return flattened


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Traffic Violations API",
        "version": "1.0.0",
        "violations_count": len(violations_store),
        "endpoints": {
            "list": "GET /violations",
            "get": "GET /violations/{id}",
            "create": "POST /violations",
            "update_status": "PATCH /violations/{id}"
        }
    }


@app.get("/violations", response_model=List[ViolationResponse])
def list_violations(
    status: Optional[str] = None,
    violation_type: Optional[str] = None,
    location: Optional[str] = None,
    limit: Optional[int] = None
):
    """
    Get all violations with optional filters
    
    Query Parameters:
    - status: Filter by status (Pending/Approved/Rejected)
    - violation_type: Filter by type (helmetless/signal_jump/triple_riding)
    - location: Filter by location
    - limit: Maximum number of results
    """
    violations_list = list(violations_store.values())
    
    # Apply filters
    if status:
        violations_list = [v for v in violations_list if v.get("status") == status]
    if violation_type:
        violations_list = [v for v in violations_list if v.get("violation_type") == violation_type]
    if location:
        violations_list = [v for v in violations_list 
                          if v.get("details", {}).get("location") == location]
    
    # Apply limit
    if limit:
        violations_list = violations_list[:limit]
    
    # Flatten and serialize
    flattened_list = [flatten_violation(v) for v in violations_list]
    serialized_list = [to_serializable(v) for v in flattened_list]
    
    return serialized_list


@app.get("/violations/{violation_id}", response_model=ViolationResponse)
def get_violation(violation_id: str):
    """Get single violation by ID"""
    if violation_id not in violations_store:
        raise HTTPException(status_code=404, detail=f"Violation {violation_id} not found")
    
    violation = violations_store[violation_id]
    flattened = flatten_violation(violation)
    serialized = to_serializable(flattened)
    
    return serialized


@app.post("/violations", response_model=ViolationResponse, status_code=201)
def create_violation(violation: ViolationCreate):
    """
    Create new violation (called by detection pipeline)
    
    IMPORTANT: This endpoint automatically converts numpy types to native Python types
    before storing, so you won't get JSON serialization errors.
    """
    # Convert input to dict and serialize numpy types
    violation_dict = violation.dict()
    violation_dict = to_serializable(violation_dict)
    
    # Generate unique ID
    violation_id = generate_violation_id(
        violation_dict["violation_type"],
        violation_dict["track_id"],
        violation_dict["timestamp_utc"]
    )
    
    # Add ID to the record
    violation_dict["id"] = violation_id
    
    # Store in memory
    violations_store[violation_id] = violation_dict
    
    # Return flattened response
    flattened = flatten_violation(violation_dict)
    serialized = to_serializable(flattened)
    
    print(f"✅ Created violation: {violation_id} | Type: {violation_dict['violation_type']} | Plate: {violation_dict.get('details', {}).get('plate', 'N/A')}")
    
    return serialized


@app.patch("/violations/{violation_id}", response_model=ViolationResponse)
def update_violation_status(violation_id: str, status_update: StatusUpdate):
    """
    Update violation status (called by dashboard)
    
    Allowed statuses: Pending, Approved, Rejected
    """
    if violation_id not in violations_store:
        raise HTTPException(status_code=404, detail=f"Violation {violation_id} not found")
    
    # Update status
    violations_store[violation_id]["status"] = status_update.status
    
    # Return updated record
    violation = violations_store[violation_id]
    flattened = flatten_violation(violation)
    serialized = to_serializable(flattened)
    
    print(f"✅ Updated violation {violation_id}: Status → {status_update.status}")
    
    return serialized


@app.delete("/violations/{violation_id}")
def delete_violation(violation_id: str):
    """Delete violation (optional, for testing)"""
    if violation_id not in violations_store:
        raise HTTPException(status_code=404, detail=f"Violation {violation_id} not found")
    
    deleted = violations_store.pop(violation_id)
    print(f"🗑️ Deleted violation: {violation_id}")
    
    return {"message": f"Violation {violation_id} deleted", "deleted": to_serializable(deleted)}


@app.get("/stats")
def get_statistics():
    """Get dashboard statistics"""
    violations_list = list(violations_store.values())
    
    total = len(violations_list)
    approved = sum(1 for v in violations_list if v.get("status") == "Approved")
    rejected = sum(1 for v in violations_list if v.get("status") == "Rejected")
    pending = sum(1 for v in violations_list if v.get("status") == "Pending")
    
    # Count by type
    by_type = {}
    for v in violations_list:
        vtype = v.get("violation_type", "unknown")
        by_type[vtype] = by_type.get(vtype, 0) + 1
    
    # Count by location
    by_location = {}
    for v in violations_list:
        loc = v.get("details", {}).get("location", "unknown")
        by_location[loc] = by_location.get(loc, 0) + 1
    
    stats = {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "pending": pending,
        "by_type": by_type,
        "by_location": by_location
    }
    
    return to_serializable(stats)


# ============================================================
# Optional: Seed with sample data for testing
# ============================================================
@app.post("/seed-sample-data")
def seed_sample_data():
    """Populate with sample violations for testing"""
    sample_violations = [
        {
            "violation_type": "helmetless",
            "track_id": 8414,
            "confidence": 0.92,
            "timestamp_utc": "2025-10-26T14:10:00Z",
            "details": {
                "bbox": [220, 130, 300, 260],
                "camera_id": "CAM_01",
                "location": "MG Road Junction",
                "plate": "TS09AB1234",
                "lat": 17.3850,
                "lon": 78.4867
            },
            "media": {
                "context_img": "output/violations/vio_2025-10-26T14-10-00Z_helmetless_8414/context.jpg",
                "crop_img": "output/violations/vio_2025-10-26T14-10-00Z_helmetless_8414/crop.jpg"
            },
            "status": "Pending"
        },
        {
            "violation_type": "signal_jump",
            "track_id": 9201,
            "confidence": 0.88,
            "timestamp_utc": "2025-10-26T14:15:00Z",
            "details": {
                "bbox": [400, 200, 550, 450],
                "camera_id": "CAM_02",
                "location": "Banjara Hills Signal",
                "plate": "TS10CD5678",
                "lat": 17.4239,
                "lon": 78.4738
            },
            "media": {
                "context_img": "output/violations/vio_2025-10-26T14-15-00Z_signal_jump_9201/context.jpg"
            },
            "status": "Pending"
        },
        {
            "violation_type": "triple_riding",
            "track_id": 7532,
            "confidence": 0.95,
            "timestamp_utc": "2025-10-26T14:20:00Z",
            "details": {
                "bbox": [150, 180, 320, 480],
                "camera_id": "CAM_01",
                "location": "MG Road Junction",
                "plate": "TS11EF9012",
                "lat": 17.3850,
                "lon": 78.4867
            },
            "media": {
                "context_img": "output/violations/vio_2025-10-26T14-20-00Z_triple_riding_7532/context.jpg"
            },
            "status": "Approved"
        }
    ]
    
    count = 0
    for sample in sample_violations:
        sample = to_serializable(sample)
        violation_id = generate_violation_id(
            sample["violation_type"],
            sample["track_id"],
            sample["timestamp_utc"]
        )
        sample["id"] = violation_id
        violations_store[violation_id] = sample
        count += 1
    
    return {
        "message": f"Seeded {count} sample violations",
        "total_violations": len(violations_store)
    }


# ============================================================
# Run Server
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Traffic Violations Backend...")
    print("📡 API will be available at: http://localhost:8000")
    print("📖 API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)