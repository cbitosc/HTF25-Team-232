"""
Media file serving endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import os
from pathlib import Path

router = APIRouter(prefix="/media", tags=["media"])

VIOLATION_ROOT = os.path.abspath("output/violations")

def get_mime_type(filename: str) -> str:
    """Get MIME type for file"""
    ext = Path(filename).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".json": "application/json"
    }
    return mime_types.get(ext, "application/octet-stream")

@router.get("")
def get_media(path: str = Query(...)):
    """
    Serve media files
    Example: /media?path=output/violations/V-123/context.jpg
    """
    if not os.path.isabs(path):
        full_path = os.path.abspath(os.path.join("..", "..", path))
    else:
        full_path = os.path.abspath(path)

    # Security check
    if not full_path.startswith(os.path.abspath(VIOLATION_ROOT)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    media_type = get_mime_type(full_path)
    return FileResponse(full_path, media_type=media_type)
