"""
Helper functions for pipeline team (Member 3) to send violations to backend
Import this in src/pipeline/run_video.py

This module now auto-detects which backend is running and adapts the
payload accordingly to avoid 422 errors:
- "app"    -> src/backend/app.py (expects ViolationIn with keys: type, media, ...)
- "server" -> backend_server.py (expects ViolationCreate with keys: violation_type, details, media)
"""
import requests
from typing import Dict, Optional, Any
from datetime import datetime
import json

# Backend URL (change if backend is on different host)
BACKEND_URL = "http://localhost:8000"

# Cached backend kind to avoid repeated detection calls
_BACKEND_KIND: Optional[str] = None  # "app" | "server" | None


def _detect_backend_kind() -> str:
    """
    Detect which backend is running by inspecting the root endpoint.
    Returns: "server" for backend_server.py, "app" for src/backend/app.py
    Fallback to "app" if uncertain (to preserve previous behavior).
    """
    global _BACKEND_KIND
    if _BACKEND_KIND:
        return _BACKEND_KIND

    try:
        resp = requests.get(f"{BACKEND_URL}/", timeout=2)
        if resp.ok:
            data = resp.json()
            # Heuristics:
            # backend_server.py returns keys: status, service: "Traffic Violations API", endpoints: {...}
            # src/backend/app.py returns keys: status, service: settings.api_title ("AI Traffic Violation Backend"), violations_loaded, ...
            service = str(data.get("service", ""))
            endpoints = data.get("endpoints")
            if endpoints or "Traffic Violations API" in service:
                _BACKEND_KIND = "server"
            else:
                _BACKEND_KIND = "app"
        else:
            _BACKEND_KIND = "app"
    except Exception:
        # If detection fails (backend down), default to app to preserve legacy behavior
        _BACKEND_KIND = "app"

    return _BACKEND_KIND

def check_backend_health() -> bool:
    """
    Check if backend server is running
    
    Usage:
    ```
    if check_backend_health():
        print("✅ Backend is ready")
    else:
        print("❌ Backend is down")
    ```
    """
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=2)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return False

def send_violation_to_backend(
    violation_id: str,
    vtype: str,
    timestamp_utc: str,
    camera_id: str,
    location: str,
    plate: str,
    confidence: float,
    media_paths: Dict[str, str],
    extra: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Send violation to backend API
    
    Parameters:
    - violation_id: Unique ID (e.g., "V-12AB34CD")
    - vtype: Type of violation ("helmetless", "triple_riding", etc.)
    - timestamp_utc: ISO format timestamp with Z suffix
    - camera_id: Camera identifier (e.g., "CAM_01")
    - location: Location description
    - plate: License plate number (or "UNREADABLE")
    - confidence: Detection confidence (0.0 to 1.0)
    - media_paths: Dict with keys: context_img, plate_img, clip_video
    - extra: Additional metadata (optional)
    
    Returns:
    - Response dict from backend or None if failed
    
    Usage in pipeline:
    ```
    from backend.integration_helper import send_violation_to_backend
    
    result = send_violation_to_backend(
        violation_id="V-12AB34CD",
        vtype="helmetless",
        timestamp_utc=datetime.utcnow().isoformat() + "Z",
        camera_id="CAM_01",
        location="MG Road Junction",
        plate="KA05MN1234",
        confidence=0.92,
        media_paths={
            "context_img": "output/violations/V-12AB34CD/context.jpg",
            "plate_img": "output/violations/V-12AB34CD/plate.jpg",
            "clip_video": "output/violations/V-12AB34CD/clip.mp4"
        },
        extra={"vehicle_type": "motorcycle", "riders_count": 3}
    )
    
    if result:
        print(f"✅ Sent to backend: {result['violation_id']}")
    ```
    """
    try:
        backend_kind = _detect_backend_kind()

        if backend_kind == "server":
            # Map to backend_server.py schema (ViolationCreate)
            # Extract best-available bbox from extra
            details_bbox = None
            if isinstance(extra, dict):
                details_bbox = extra.get("bike_bbox") or extra.get("vehicle_bbox") or extra.get("bbox")
            
            payload = {
                "violation_type": vtype,
                "track_id": int((extra or {}).get("track_id", 0)),
                "confidence": float(confidence),
                "timestamp_utc": timestamp_utc,
                "details": {
                    "bbox": details_bbox,
                    "camera_id": camera_id,
                    "location": location,
                    "plate": plate,
                    # lat/lon can be added here if present in `extra`
                    "lat": (extra or {}).get("lat"),
                    "lon": (extra or {}).get("lon"),
                },
                "media": {
                    "context_img": media_paths.get("context_img"),
                    "crop_img": (extra or {}).get("crop_img"),
                    "plate_img": media_paths.get("plate_img"),
                    "clip_video": media_paths.get("clip_video"),
                },
                "status": "Pending",
            }

            response = requests.post(
                f"{BACKEND_URL}/violations",
                json=payload,
                timeout=10
            )

            if response.status_code in (200, 201):
                print(f"✅ Violation {violation_id} sent to backend_server")
                return response.json()
            else:
                print(f"❌ Backend (server) error {response.status_code}: {response.text}")
                return None

        else:
            # Default: src/backend/app.py schema (ViolationIn)
            payload = {
                "violation_id": violation_id,
                "type": vtype,
                "timestamp_utc": timestamp_utc,
                "camera_id": camera_id,
                "location": location,
                "plate": plate,
                "confidence": confidence,
                "media": media_paths,
                "extra": extra or {}
            }

            response = requests.post(
                f"{BACKEND_URL}/violations",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                print(f"✅ Violation {violation_id} sent to backend app")
                return response.json()
            else:
                print(f"❌ Backend (app) error {response.status_code}: {response.text}")
                return None

    except requests.exceptions.Timeout:
        print(f"❌ Backend request timed out for {violation_id}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to backend at {BACKEND_URL}")
        return None
    except Exception as e:
        print(f"❌ Failed to send violation {violation_id}: {e}")
        return None

def send_violation_simple(violation_dict: Dict) -> Optional[Dict]:
    """
    Simplified version - send pre-formatted violation dict
    
    Usage:
    ```
    violation = {
        "violation_id": "V-123",
        "type": "helmetless",
        "timestamp_utc": "2025-10-25T14:30:00Z",
        "camera_id": "CAM_01",
        "location": "Junction A",
        "plate": "KA01AB1234",
        "confidence": 0.95,
        "media": {
            "context_img": "output/violations/V-123/context.jpg",
            "plate_img": "output/violations/V-123/plate.jpg",
            "clip_video": "output/violations/V-123/clip.mp4"
        }
    }
    
    result = send_violation_simple(violation)
    ```
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/violations",
            json=violation_dict,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Backend error: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"❌ Failed to send violation: {e}")
        return None

def get_violation_stats() -> Optional[Dict]:
    """
    Get statistics from backend
    
    Returns:
    - Stats dict or None if failed
    """
    try:
        response = requests.get(f"{BACKEND_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")
        return None

def trigger_import_from_disk() -> Optional[Dict]:
    """
    Trigger backend to import existing violations from disk
    
    Usage:
    ```
    result = trigger_import_from_disk()
    if result:
        print(f"Imported {result['imported']} violations")
    ```
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/violations/import_from_disk",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"❌ Failed to trigger import: {e}")
        return None

# Example usage for testing
if __name__ == "__main__":
    print("🧪 Testing backend integration...")
    
    # Check if backend is running
    if not check_backend_health():
        print("❌ Backend is not running. Start it with: python src/backend/app.py")
        exit(1)
    
    print("✅ Backend is running")
    
    # Create test violation
    test_violation = {
        "violation_id": f"V-TEST-{int(datetime.now().timestamp())}",
        "type": "helmetless",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "camera_id": "CAM_TEST",
        "location": "Test Location",
        "plate": "TEST1234",
        "confidence": 0.99,
        "media": {
            "context_img": "output/violations/V-TEST/context.jpg",
            "plate_img": "output/violations/V-TEST/plate.jpg",
            "clip_video": "output/violations/V-TEST/clip.mp4"
        },
        "extra": {"test": True}
    }
    
    result = send_violation_simple(test_violation)
    
    if result:
        print(f"✅ Test violation created: {result['violation_id']}")
    else:
        print("❌ Failed to create test violation")
    
    # Get stats
    stats = get_violation_stats()
    if stats:
        print(f"\n📊 Current stats:")
        print(f"  Total violations: {stats.get('total_violations', 0)}")
        print(f"  By type: {stats.get('by_type', {})}")
