import os
import cv2
import xml.etree.ElementTree as ET

# --- Adjust your dataset paths ---
IMAGES_DIR = "data/pascal_voc_helmet_dataset/images"
ANN_DIR    = "data/pascal_voc_helmet_dataset/annotations"

# --- Output paths for cropped data ---
OUT_HELMET_DIR     = "data/helmet_classifier_data/helmet"
OUT_NO_HELMET_DIR  = "data/helmet_classifier_data/no_helmet"

os.makedirs(OUT_HELMET_DIR, exist_ok=True)
os.makedirs(OUT_NO_HELMET_DIR, exist_ok=True)

counter_helmet = 0
counter_nohelmet = 0

for ann_file in os.listdir(ANN_DIR):
    if not ann_file.endswith(".xml"):
        continue

    ann_path = os.path.join(ANN_DIR, ann_file)
    tree = ET.parse(ann_path)
    root = tree.getroot()

    # Find matching image
    filename_tag = root.find("filename")
    img_name = filename_tag.text if filename_tag is not None else os.path.splitext(ann_file)[0] + ".jpg"
    img_path = os.path.join(IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"[WARN] image not found: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] cannot open {img_path}")
        continue

    for obj in root.findall("object"):
        cls = obj.find("name").text.strip().lower()  # convert to lowercase

        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        # Ensure box inside image
        h, w = img.shape[:2]
        xmin = max(0, min(xmin, w - 1))
        xmax = max(0, min(xmax, w - 1))
        ymin = max(0, min(ymin, h - 1))
        ymax = max(0, min(ymax, h - 1))
        if xmax <= xmin or ymax <= ymin:
            continue

        crop = img[ymin:ymax, xmin:xmax]

        # ✅ Update these to match your class names
        if cls in ["with helmet", "helmet", "with_helmet", "helmet_on"]:
            out_path = os.path.join(OUT_HELMET_DIR, f"helmet_{counter_helmet:06d}.jpg")
            cv2.imwrite(out_path, crop)
            counter_helmet += 1

        elif cls in ["without helmet", "no_helmet", "without_helmet", "no helmet"]:
            out_path = os.path.join(OUT_NO_HELMET_DIR, f"nohelmet_{counter_nohelmet:06d}.jpg")
            cv2.imwrite(out_path, crop)
            counter_nohelmet += 1

print("\n✅ DONE.")
print(f"Saved {counter_helmet} helmet crops to {OUT_HELMET_DIR}")
print(f"Saved {counter_nohelmet} no-helmet crops to {OUT_NO_HELMET_DIR}")