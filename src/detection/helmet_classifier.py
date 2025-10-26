import torch
import cv2
import numpy as np
from PIL import Image  # ✅ FIXED: Moved to top imports
from torchvision import models, transforms
from torch import nn

class HelmetClassifier:
    def __init__(self,
                 weights_path: str = "src/detection/models/helmet_classifier.pt",
                 device: str = "cuda"):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        # same model architecture as training script
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)
        
        # Load weights with error handling
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"✅ Loaded helmet classifier from {weights_path}")
        except FileNotFoundError:
            print(f"⚠️  Warning: Helmet classifier weights not found at {weights_path}")
            print("    Using untrained model (for testing only)")
        except Exception as e:
            print(f"❌ Error loading helmet classifier: {e}")
            raise
        
        self.model.eval().to(self.device)

        # matches ImageFolder order: 0 -> helmet, 1 -> no_helmet
        self.id_to_label = {0: "helmet", 1: "no_helmet"}

        # same transforms as training
        self.tfms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def classify_person_crop(self, frame_bgr, person_bbox):
        """
        Classify whether a person is wearing a helmet.
        
        Args:
            frame_bgr: full frame (BGR from OpenCV)
            person_bbox: [x1, y1, x2, y2]
        
        Returns:
            (helmet_bool, confidence_float) or (None, None) if classification fails
        """
        try:
            x1, y1, x2, y2 = map(int, person_bbox)
            
            # Validate bbox coordinates
            h, w = frame_bgr.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            
            if x2 <= x1 or y2 <= y1:
                return None, None

            # take top 60% of bbox (head/upper body)
            bbox_h = max(1, y2 - y1)
            head_y2 = y1 + int(0.6 * bbox_h)
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
            
        except Exception as e:
            print(f"⚠️  Helmet classification error: {e}")
            return None, None