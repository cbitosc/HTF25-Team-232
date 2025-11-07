"""
detector.py
YOLOv8 COCO detector wrapper.
Detects vehicles, persons, and traffic lights.
"""

from typing import List, Dict, Any
import numpy as np
import torch
import cv2

# COCO → Our internal class mapping
COCO_NAME_MAP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}


def normalize_class(name: str) -> str:
    """Normalize COCO class names to project convention."""
    if name == "motorcycle":
        return "motorbike"
    if name == "bicycle":
        return "bike"
    if name == "traffic light":
        return "traffic_light"
    return name


class YOLODetector:
    def __init__(self, weights_path: str = "yolov8n.pt",
                 device: str = "cuda", conf_thresh: float = 0.4, imgsz: int = 640):
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        print(f"[YOLODetector] Loaded {weights_path} (COCO pretrained) on {self.device}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection on one frame."""
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf_thresh, device=self.device)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        detections = []
        for i in range(len(boxes)):
            cls_id = int(classes[i])
            if cls_id not in COCO_NAME_MAP:
                continue
            name = normalize_class(COCO_NAME_MAP[cls_id])
            x1, y1, x2, y2 = boxes[i].astype(int)
            detections.append({
                "track_id": None,
                "class": name,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(scores[i]),
                "helmet": None,
                "helmet_confidence": None
            })
        return detections


if __name__ == "__main__":
    cap = cv2.VideoCapture("data/raw/sample_intersection.mp4")
    det = YOLODetector()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = det.detect(frame)
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            label = f"{d['class']} {d['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("detector preview", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
