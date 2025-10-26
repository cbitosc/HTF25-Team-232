import torch
import cv2
import numpy as np
from torchvision import models, transforms
from torch import nn
from PIL import Image  # <-- ADD THIS

class HelmetClassifier:
    def __init__(self,
                 weights_path: str = "src/detection/models/helmet_classifier.pt",
                 device: str = "cuda"):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        # same model architecture as training script
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval().to(self.device)

        # matches ImageFolder order: 0 -> helmet, 1 -> no_helmet
        self.id_to_label = {0: "helmet", 1: "no_helmet"}

        # same transforms as training
        self.tfms = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
        ])

    def classify_person_crop(self, frame_bgr, person_bbox):
        """
        frame_bgr: full frame (BGR from OpenCV)
        person_bbox: [x1,y1,x2,y2]
        returns: (helmet_bool, confidence_float)
        """
        x1, y1, x2, y2 = map(int, person_bbox)

        # take top 60% of bbox (head/upper body)
        h = max(1, y2 - y1)
        head_y2 = y1 + int(0.6 * h)
        head_crop = frame_bgr[y1:head_y2, x1:x2]

        # if crop is empty or invalid
        if head_crop.size == 0:
            return None, None

        # BGR -> RGB
        head_rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)

        # convert to PIL Image for torchvision
        pil_img = Image.fromarray(head_rgb)

        # apply transforms
        tensor = self.tfms(pil_img).unsqueeze(0).to(self.device)

        # inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = self.id_to_label[pred_idx]  # "helmet" or "no_helmet"
        pred_conf = float(probs[pred_idx])

        helmet_bool = (pred_label == "helmet")
        return helmet_bool, pred_conf
