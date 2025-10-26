"""
Violation management endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

router = APIRouter(prefix="/violations", tags=["violations"])

class MediaInfo(BaseModel):
    context_img: Optional[str] = None
    plate_img: Optional[str] = None
    clip_video: Optional[str] = None

class ViolationIn(BaseModel):
    violation_id: str
    type: str
    timestamp_utc: str
    camera_id: str
    location: str
    plate: str
    confidence: float
    media: MediaInfo
    extra: Optional[dict] = None

class ViolationOut(ViolationIn):
    reviewed: bool = False
    approved: Optional[bool] = None
    reviewer_notes: Optional[str] = None

# You'll need to access violations_db from app.py
# For now, import it or pass as dependency

@router.post("", response_model=ViolationOut)
def create_violation(v: ViolationIn):
    """Create new violation"""
    # Your existing logic
    pass

@router.get("", response_model=List[ViolationOut])
def list_violations(
    vtype: Optional[str] = None,
    camera_id: Optional[str] = None,
    # ... other filters
):
    """List violations with filters"""
    # Your existing logic
    pass

@router.get("/{violation_id}", response_model=ViolationOut)
def get_violation(violation_id: str):
    """Get single violation"""
    # Your existing logic
    pass

@router.patch("/{violation_id}/review")
def review_violation(violation_id: str, approved: bool, notes: Optional[str] = None):
    """Review violation (human-in-the-loop)"""
    # Your existing logic
    pass
