"""STEP 8: Extract tooth patches for classification using the fine-tuned YOLO model."""
import os
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def main():
    model = YOLO(r"C:\Users\enosh\oralguard\src\detector\weights\oralguard_finetuned\weights\best.pt")

    output_patch_dir = r"C:\Users\enosh\oralguard\data\patches_v2\images"
    os.makedirs(output_patch_dir, exist_ok=True)

    CLASS_NAMES = ['caries', 'deep_caries', 'periapical_lesion', 'impacted_tooth']
    csv_rows = []
    patch_count = 0

    img_dirs = [
        r"C:\Users\enosh\oralguard\data\combined\images\train",
        r"C:\Users\enosh\oralguard\data\finetune\periapical_yolo\images",
    ]

    for img_dir in img_dirs:
        if not os.path.exists(img_dir):
            continue
        imgs = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
        print(f"Processing {len(imgs)} images from {img_dir}")

        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            results = model.predict(
                source=str(img_path),
                imgsz=1024,
                conf=0.25,
                iou=0.45,
                agnostic_nms=True,
                device=0,
                verbose=False
            )

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                patch = img[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                patch_resized = cv2.resize(patch, (128, 128))
                patch_name = f"patch_{patch_count:07d}.jpg"
                cv2.imwrite(
                    os.path.join(output_patch_dir, patch_name),
                    patch_resized
                )

                # Get label from original annotation
                label_path = str(img_path).replace(
                    'images', 'labels'
                ).replace('.jpg', '.txt').replace('.png', '.txt')

                labels = [0, 0, 0, 0]
                if os.path.exists(label_path):
                    with open(label_path) as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if parts:
                                cls = int(parts[0])
                                if 0 <= cls <= 3:
                                    labels[cls] = 1

                csv_rows.append([
                    os.path.join(output_patch_dir, patch_name),
                    *labels
                ])
                patch_count += 1

    print(f"Total patches extracted: {patch_count}")

    csv_path = r"C:\Users\enosh\oralguard\data\patches_v2\labels.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Labels CSV written: {csv_path}")

    arr = np.array([row[1:] for row in csv_rows])
    for i, name in enumerate(CLASS_NAMES):
        count = int(arr[:, i].sum())
        pct = count / len(arr) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
