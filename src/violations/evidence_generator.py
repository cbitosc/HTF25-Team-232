"""
evidence_generator.py
Responsible for creating the actual evidence package for each violation:
- cropped proof images
- short temporal clip (before/after event)
- metadata JSON

Called by run_video.py whenever rules.py reports a new violation event.
"""

import os
import cv2
import json
import shutil
import time
from typing import Dict, List, Tuple, Optional, Any

from .event_schema import build_violation_record
from .adapter import adapt_event_for_evidence


# ============================================================
# Helper: ensure output directory exists
# ============================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Helper: draw debug boxes on frame for context image
# ============================================================

def _draw_event_overlay(frame, event: Dict[str, Any], color=(0, 0, 255)):
    """
    Draws bounding boxes / text to make context.jpg understandable
    for humans / police during review.

    ✅ ENHANCED: Now draws thick RED boxes and prominent labels
    """
    annotated = frame.copy()

    label = event["type"].upper().replace("_", " ")
    
    # ✅ Use THICK RED boxes for violations
    violation_color = (0, 0, 255)  # BGR: Red
    thickness = 4  # Thick boxes for evidence

    if event["type"] in ("helmetless", "triple_riding"):
        bbox = event.get("bike_bbox")
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            
            # ✅ Draw thick red rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), violation_color, thickness)
            
            # ✅ Draw semi-transparent red overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), violation_color, -1)
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
            
            # ✅ Draw large warning label with background
            label_text = f"⚠️ {label}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            
            # Background rectangle for text
            cv2.rectangle(
                annotated, 
                (x1, y1 - text_height - 15), 
                (x1 + text_width + 10, y1 - 5),
                violation_color, 
                -1
            )
            
            # White text on red background
            cv2.putText(
                annotated,
                label_text,
                (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        # ✅ Mark riders with smaller boxes
        riders = event.get("riders", [])
        for r in riders:
            rb = r.get("bbox")
            if rb:
                rx1, ry1, rx2, ry2 = map(int, rb)
                
                # Green box for riders
                rider_color = (0, 255, 0)
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), rider_color, 2)
                
                helmet = r.get("helmet", None)
                if helmet is False:
                    tag = "NO HELMET"
                    tag_color = (0, 0, 255)
                elif helmet is True:
                    tag = "HELMET"
                    tag_color = (0, 255, 0)
                else:
                    tag = "RIDER"
                    tag_color = (255, 255, 0)
                
                cv2.putText(
                    annotated,
                    tag,
                    (rx1, ry1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    tag_color,
                    2,
                )

    elif event["type"] in ("red_light_jump", "signal_jump"):
        bbox = event.get("vehicle_bbox")
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            
            # ✅ Draw thick red rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), violation_color, thickness)
            
            # ✅ Draw semi-transparent red overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), violation_color, -1)
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
            
            # ✅ Draw large warning label with background
            label_text = f"🚨 {label}"
            vehicle_class = event.get("details", {}).get("class", "VEHICLE")
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )
            
            # Background rectangle for text
            cv2.rectangle(
                annotated, 
                (x1, y1 - text_height - 20), 
                (x1 + text_width + 10, y1 - 5),
                violation_color, 
                -1
            )
            
            # White text on red background
            cv2.putText(
                annotated,
                label_text,
                (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            
            # ✅ Add vehicle type label
            cv2.putText(
                annotated,
                vehicle_class.upper(),
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                violation_color,
                2,
            )

    else:
        # ✅ Fallback for unknown violation types
        cv2.putText(
            annotated,
            f"⚠️ {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            violation_color,
            3,
        )

    # ✅ Add timestamp and confidence overlay
    timestamp = event.get("timestamp_utc", "")
    confidence = event.get("confidence", 0.0)
    
    info_text = f"Time: {timestamp[:19]} | Confidence: {confidence:.2%}"
    cv2.putText(
        annotated,
        info_text,
        (10, annotated.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    
    # ✅ Add "EVIDENCE" watermark
    cv2.putText(
        annotated,
        "EVIDENCE",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    return annotated


# ============================================================
# Helper: save a short mp4 clip from buffer
# ============================================================

def _save_clip_from_buffer(
    buffer_frames: List[Tuple[float, Any]],
    out_path: str,
    fps: int = 15
):
    """
    buffer_frames should be a list of (timestamp, frame_bgr).
    We'll just dump them to mp4 in time order.

    NOTE: assumes frames in buffer_frames are already in order (oldest first).
    """
    if not buffer_frames:
        return

    # sort just in case
    buffer_frames = sorted(buffer_frames, key=lambda x: x[0])

    # video writer config
    h, w, _ = buffer_frames[0][1].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for (_, frm) in buffer_frames:
        writer.write(frm)

    writer.release()


# ============================================================
# Helper: optional plate crop placeholder
# ============================================================

def _extract_plate_crop(frame, event: Dict[str, Any]) -> Optional[Any]:
    """
    Placeholder for ANPR integration.
    For now we just return None.
    Later:
    - run plate detector on the region around the bike/vehicle bbox
    - crop that region
    """
    return None


# ============================================================
# MAIN: save_violation_package
# ============================================================

def save_violation_package(
    frame,
    past_frames_buffer: List[Tuple[float, Any]],
    event: Dict[str, Any],
    camera_id: str,
    location_name: str,
    plate_text: Optional[str],
    output_root: str = "output/violations"
) -> Dict:
    """
    Create the full evidence bundle for this violation event.

    Steps:
    1. Create folder output/violations/<violation_id>/
    2. Annotate and save context image (context.jpg)
    3. Build small video clip from buffered frames (clip.mp4)
    4. Attempt plate crop (plate.jpg) [placeholder for now]
    5. Build violation record (JSON)
    6. Save evidence.json
    7. Return the record (for backend sending)

    Parameters
    ----------
    frame : np.ndarray (BGR OpenCV frame for the current moment)
    past_frames_buffer : list[(timestamp, frame_bgr)]
        rolling buffer of recent frames (2-3s window)
    event : dict
        one violation event from rules.evaluate_frame()
    camera_id : str
        e.g. "CAM_01"
    location_name : str
        e.g. "MG Road Junction"
    plate_text : str or None
        license plate OCR result, or None/"UNREADABLE"
    output_root : str
        root output directory for all violations
    """

    # ✅ FIXED: Convert event format from rules.py to evidence format
    adapted_event = adapt_event_for_evidence(event)

    # 1. Draw annotated frame for context
    annotated_frame = _draw_event_overlay(frame, adapted_event, color=(0, 0, 255))

    # 2. Try to crop plate (placeholder for now)
    plate_crop_img = _extract_plate_crop(frame, adapted_event)

    # 3. We'll fill in media_paths AFTER we know violation_id and file paths
    media_paths = {
        "context_img": None,
        "plate_img": None,
        "clip_video": None
    }

    # 4. Build the structured violation record (generates violation_id for us)
    # ✅ FIXED: Use adapted_event["type"] instead of event["type"]
    violation_id, record = build_violation_record(
        violation_type=adapted_event["type"],
        camera_id=camera_id,
        location_name=location_name,
        plate_text=plate_text if plate_text else "UNREADABLE",
        confidence=adapted_event.get("confidence", 0.9),
        media_paths=media_paths,
        extra=adapted_event  # keep adapted event info for explainability
    )

    # Make directory for this specific violation
    vdir = os.path.join(output_root, violation_id)
    _ensure_dir(vdir)

    # 5. Save context image
    context_path = os.path.join(vdir, "context.jpg")
    cv2.imwrite(context_path, annotated_frame)

    # 6. Save plate crop if we had one
    plate_path = None
    if plate_crop_img is not None:
        plate_path = os.path.join(vdir, "plate.jpg")
        cv2.imwrite(plate_path, plate_crop_img)

    # 7. Save short clip from buffered frames
    clip_path = os.path.join(vdir, "clip.mp4")
    _save_clip_from_buffer(past_frames_buffer, clip_path, fps=15)

    # 8. Update record media paths now that we know final paths
    record["media"]["context_img"] = context_path
    record["media"]["plate_img"] = plate_path
    record["media"]["clip_video"] = clip_path

    # 9. Save evidence.json
    json_path = os.path.join(vdir, "evidence.json")
    with open(json_path, "w") as jf:
        json.dump(record, jf, indent=2)

    # 10. Debug log for you
    print(f"[EVIDENCE] Saved {violation_id} -> {vdir}")

    # 11. Return record (this is what you can POST to backend)
    return record