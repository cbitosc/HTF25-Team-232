"""
frame_utils.py
Shared utilities for working with frames in the traffic violation pipeline.

This module is used by:
- detection/tracking team (Members 1 & 2)
- violation logic / evidence generation (Member 3)
- pipeline runner (run_video.py)

Features:
1. Drawing detection bounding boxes
2. Drawing violation overlays
3. Drawing and checking ROI polygons
4. Managing rolling frame buffer for evidence clip creation
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


# ============================================================
# 1) Detection visualization (owned mainly by Member 2)
# ============================================================

def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    show_ids: bool = True,
    show_helmet: bool = True
) -> np.ndarray:
    """
    Draws raw model detections (people, bikes, cars...) on the frame.

    detections format expectation:
    {
        "track_id": 7,
        "class": "motorbike",
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.95,
        "helmet": False,                # only for persons
        "helmet_confidence": 0.92
    }

    Returns: annotated frame copy (does not modify original)
    """

    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        obj_class = det.get("class", "object")
        track_id = det.get("track_id", None)
        conf = det.get("confidence", None)

        # Choose color per object class
        if obj_class in ("person",):
            color = (0, 255, 0)   # green
        elif obj_class in ("motorbike", "motorcycle", "bike", "scooter"):
            color = (255, 0, 0)   # blue-ish
        elif obj_class in ("car", "truck", "bus"):
            color = (0, 0, 255)   # red-ish
        else:
            color = (255, 255, 0) # yellow

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Build label text
        label_parts = [obj_class]
        if show_ids and track_id is not None:
            label_parts.append(f"ID:{track_id}")
        if conf is not None:
            label_parts.append(f"{conf:.2f}")

        # Helmet info shown only for person class
        if show_helmet and obj_class == "person":
            if "helmet" in det:
                if det["helmet"] is True:
                    label_parts.append("HELMET")
                elif det["helmet"] is False:
                    label_parts.append("NO-HELMET")

        label = " ".join(label_parts)

        # Put the label just above the box
        cv2.putText(
            out,
            label,
            (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return out


# ============================================================
# 2) Violation visualization (owned mainly by Member 3)
# ============================================================

def draw_violations(
    frame: np.ndarray,
    events: List[Dict[str, Any]],
    draw_riders: bool = True
) -> np.ndarray:
    """
    Draws violation-level overlays. This mirrors what
    evidence_generator._draw_event_overlay() does for context.jpg,
    but works for multi-violation-per-frame preview.

    events come from rules.evaluate_frame(), e.g.
    {
        "type": "helmetless",
        "bike_id": 7,
        "bike_bbox": [...],
        "riders": [...],
        "confidence": 0.91
    }
    or
    {
        "type": "red_light_jump",
        "vehicle_id": 12,
        "vehicle_bbox": [...],
        "confidence": 0.95
    }

    Returns: annotated frame copy.
    """

    out = frame.copy()

    for ev in events:
        vtype = ev["type"].upper()

        # Choose color per violation type
        if ev["type"] == "helmetless":
            color = (0, 0, 255)  # red
        elif ev["type"] == "triple_riding":
            color = (0, 165, 255)  # orange
        elif ev["type"] == "red_light_jump":
            color = (255, 0, 255)  # magenta
        else:
            color = (255, 255, 255)  # white fallback

        # Pick bbox to highlight
        bbox = ev.get("bike_bbox") or ev.get("vehicle_bbox")
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                vtype,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Optionally draw individual riders for explainability
        if draw_riders and ev["type"] in ("helmetless", "triple_riding"):
            riders = ev.get("riders", [])
            for r in riders:
                rb = r.get("bbox")
                if rb:
                    rx1, ry1, rx2, ry2 = map(int, rb)
                    cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                    tag = "RIDER"
                    if r.get("helmet") is True:
                        tag = "HELMET"
                    elif r.get("helmet") is False:
                        tag = "NO-HELMET"

                    cv2.putText(
                        out,
                        tag,
                        (rx1, max(ry1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

    return out


# ============================================================
# 3) ROI helpers (shared between Member 2 and Member 3)
# ============================================================

def load_rois_from_dict(roi_dict: Dict[str, List[List[int]]]) -> Dict[str, np.ndarray]:
    """
    Convert raw ROI coordinates (from JSON / config) into numpy arrays
    suitable for OpenCV polygon ops.

    roi_dict example:
    {
        "lane_poly": [[100,200],[300,200],[350,400],[50,400]],
        "stop_line_zone": [[150,350],[350,350],[350,380],[150,380]]
    }

    Returns:
    {
        "lane_poly": np.ndarray([[100,200],[300,200]...], dtype=int32),
        "stop_line_zone": np.ndarray([...], dtype=int32)
    }
    """
    out = {}
    for name, pts in roi_dict.items():
        poly = np.array(pts, dtype=np.int32)
        out[name] = poly
    return out


def inside_polygon(polygon: Optional[np.ndarray], point: Tuple[int, int]) -> bool:
    """
    Check if a point (cx, cy) is inside a given ROI polygon.
    Uses cv2.pointPolygonTest: >=0 means on or inside.
    """
    if polygon is None or len(polygon) == 0:
        return False
    x, y = point
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0


def draw_roi(
    frame: np.ndarray,
    polygon: Optional[np.ndarray],
    label: str = "ROI",
    color: Tuple[int, int, int] = (255, 255, 0)
) -> np.ndarray:
    """
    Overlay a polygon ROI on the frame with a label.
    """
    out = frame.copy()
    if polygon is None or len(polygon) == 0:
        return out

    cv2.polylines(out, [polygon], isClosed=True, color=color, thickness=2)

    # put label near the first vertex
    x0, y0 = polygon[0]
    cv2.putText(
        out,
        label,
        (int(x0), int(y0) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    return out


def annotate_with_rois(
    frame: np.ndarray,
    rois: Dict[str, np.ndarray],
    order: Optional[List[str]] = None
) -> np.ndarray:
    """
    Convenience helper: draw all ROIs on a frame at once.

    rois is a dict like:
    {
      "lane_poly": <np.ndarray>,
      "stop_line_zone": <np.ndarray>,
    }

    order lets you define draw order if you want, e.g.
    ["lane_poly", "stop_line_zone"]
    """
    out = frame.copy()
    if order is None:
        order = list(rois.keys())

    for name in order:
        poly = rois.get(name)
        out = draw_roi(out, poly, label=name)

    return out


# ============================================================
# 4) Frame buffer helper (shared)
# ============================================================

def maintain_buffer(
    buffer_frames: List[Tuple[float, Any]],
    frame: np.ndarray,
    max_len: int = 45
) -> List[Tuple[float, Any]]:
    """
    Keeps a rolling buffer of the last N frames.
    Used later to build clip.mp4 in evidence_generator.

    buffer_frames format:
        [ (timestamp, frame_bgr), ... ]
    """
    buffer_frames.append((cv2.getTickCount(), frame.copy()))
    if len(buffer_frames) > max_len:
        buffer_frames.pop(0)
    return buffer_frames