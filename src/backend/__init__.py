"""
backend package

FastAPI-based REST API for the AI Traffic Violation Detection System.

Provides:
- POST /violations → Save new violation (from pipeline)
- GET /violations → List violations with filters
- GET /violations/{id} → Get single violation
- PATCH /violations/{id}/review → Review violation (human-in-the-loop)
- GET /media → Serve evidence files (images, videos)
- GET /stats → Dashboard analytics
- POST /violations/import_from_disk → Load existing violations

Owned by Member 4 (Backend API & Dashboard Integration).
"""

from .app import app
from .config import get_settings

__all__ = ["app", "get_settings"]

__version__ = "1.0.0"
