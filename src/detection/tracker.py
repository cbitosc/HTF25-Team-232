"""
tracker.py
Object tracking wrapper (ByteTrack / DeepSORT / etc.)

Goal:
- Take raw detections from YOLO (per frame)
- Assign stable track IDs across frames
- Return detections in the exact format expected by the pipeline

Current state:
- Uses a simple placeholder tracker that just assigns incremental IDs
  per frame, so integration/testing can proceed before real tracking
  is wired up.

Later:
- Member 2 can replace the internals with ByteTrack, DeepSORT, etc.
"""

from typing import List, Dict, Any
import os
import json
import cv2
import numpy as np

# If you later integrate a real tracker:
# from yolox.tracker.byte_tracker import BYTETracker
# or DeepSORT, etc.


class SimpleTrackerWrapper:
    def __init__(self, max_age: int = 30):
        """
        max_age: how many frames we keep 'lost' objects alive (for real tracker)
        For now, we don't actually use it because this is a stub.
        """
        self.max_age = max_age
        self.tracker = None  # placeholder for real tracker state
        self.next_id = 0     # persistent counter so IDs remain consistent across frames

    def update(self, detections: List[Dict[str, Any]], frame) -> List[Dict[str, Any]]:
        """
        Input:
            detections: list of dicts from YOLODetector.detect(), e.g.
            {
                "track_id": None,
                "class": "motorbike",
                "bbox": [x1,y1,x2,y2],
                "confidence": 0.95,
                "helmet": False,
                "helmet_confidence": 0.9
            }

        Output:
            tracks: list of dicts in the final canonical format used by Member 3:
            {
                "track_id": 12,                # stable across frames
                "class": "motorbike",
                "bbox": [x1,y1,x2,y2],
                "confidence": 0.95,
                "helmet": False,
                "helmet_confidence": 0.9
            }

        Right now we just assign IDs ourselves.
        Later: plug in real ByteTrack output instead.
        """

        tracked_outputs: List[Dict[str, Any]] = []

        # STUB MODE:
        # just assign/reuse IDs in a dumb way.
        # Each frame, give every detection a new ID.
        # This is obviously not persistent tracking yet.
        for det in detections:
            # assign track id
            tid = self.next_id
            self.next_id += 1

            tracked_outputs.append({
                "track_id": tid,
                "class": det.get("class", "unknown"),
                "bbox": det.get("bbox", [0, 0, 0, 0]),
                "confidence": det.get("confidence", det.get("score", 0.0)),
                "helmet": det.get("helmet", None),
                "helmet_confidence": det.get("helmet_confidence", None),
            })

        return tracked_outputs


def draw_tracks_on_frame(frame, tracks: List[Dict[str, Any]]) -> any:
    """
    Helper for debugging: draw tracked objects on the frame.
    """
    vis = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        tid = t["track_id"]
        cls_name = t["class"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{cls_name} ID:{tid}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return vis


if __name__ == "__main__":
    """
    Debug/demo mode:
    - Runs detector
    - Runs tracker
    - Saves a debug overlay video + per-frame JSON

    This is for Member 1 & 2 to QA detection+tracking WITHOUT the full pipeline.
    """

    # local import here to avoid circular import at runtime
    from detection.detector import YOLODetector

    cap = cv2.VideoCapture("data/raw/sample_intersection.mp4")
    det_model = YOLODetector(weights_path="src/detection/models/yolo_best.pt")
    tracker = SimpleTrackerWrapper()

    # prepare debug output dirs
    debug_video_dir = "output/debug_tracking/video_overlays"
    debug_json_dir = "output/debug_tracking/json"
    os.makedirs(debug_video_dir, exist_ok=True)
    os.makedirs(debug_json_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. detect
        dets = det_model.detect(frame)

        # 2. track
        tracks = tracker.update(dets, frame)

        # 3. draw overlays for debugging ONLY
        vis_frame = draw_tracks_on_frame(frame, tracks)

        # 4. init video writer
        if writer is None:
            h, w = frame.shape[:2]
            writer_path = os.path.join(debug_video_dir, "demo.mp4")
            writer = cv2.VideoWriter(writer_path, fourcc, 20.0, (w, h))

        # 5. save frame to debug video
        writer.write(vis_frame)

        # 6. dump JSON for this frame (debug)
        frame_record = {
            "frame_idx": frame_idx,
            "objects": tracks  # already in pipeline-friendly format
        }
        json_path = os.path.join(debug_json_dir, f"frame_{frame_idx:06d}.json")
        with open(json_path, "w") as f:
            json.dump(frame_record, f, indent=2)

        frame_idx += 1

        # Live preview window (optional)
        cv2.imshow("det+track preview", vis_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("✅ Tracking demo complete. Debug output saved.")
