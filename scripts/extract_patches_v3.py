import os, cv2, glob, csv, torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def main():
    model = YOLO(
        'C:/Users/enosh/oralguard/src/detector/weights'
        '/oralguard_v3/weights/best.pt'
    )

    output_patch_dir = r'C:\Users\enosh\oralguard\data\patches_v3\images'
    os.makedirs(output_patch_dir, exist_ok=True)

    CLASS_NAMES = ['caries','deep_caries','periapical_lesion','impacted_tooth']
    csv_rows = []
    patch_count = 0

    img_dirs = [
        r'C:\Users\enosh\oralguard\data\combined\images\train',
        r'C:\Users\enosh\oralguard\data\finetune\periapical_yolo\images',
    ]

    for img_dir in img_dirs:
        if not os.path.exists(img_dir):
            print(f'Directory {img_dir} does not exist, skipping.')
            continue
        imgs = (
            list(Path(img_dir).glob('*.jpg')) + 
            list(Path(img_dir).glob('*.png')) +
            list(Path(img_dir).glob('*.JPG')) +
            list(Path(img_dir).glob('*.PNG'))
        )
        imgs = list(set(imgs))
        print(f'Processing {len(imgs)} images from {img_dir}')

        for idx, img_path in enumerate(imgs):
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

            for box in boxes.xyxy.cpu().numpy():
                x1,y1,x2,y2 = map(int, box)
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(w,x2), min(h,y2)
                patch = img[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                patch_resized = cv2.resize(patch, (128,128))
                patch_name = f'patch_{patch_count:07d}.jpg'
                cv2.imwrite(
                    os.path.join(output_patch_dir, patch_name),
                    patch_resized
                )

                label_path = (
                    str(img_path)
                    .replace('images','labels')
                    .replace('.jpg','.txt')
                    .replace('.png','.txt')
                    .replace('.JPG','.txt')
                    .replace('.PNG','.txt')
                )
                labels = [0,0,0,0]
                if os.path.exists(label_path):
                    with open(label_path) as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if parts:
                                try:
                                    cls = int(parts[0])
                                    if 0 <= cls <= 3:
                                        labels[cls] = 1
                                except ValueError:
                                    continue

                csv_rows.append([
                    os.path.join(output_patch_dir, patch_name),
                    *labels
                ])
                patch_count += 1

            if idx > 0 and idx % 1000 == 0:
                print(f'Processed {idx}/{len(imgs)} images. Patches extracted so far: {patch_count}')

    print(f'Total patches: {patch_count}')

    csv_path = r'C:\Users\enosh\oralguard\data\patches_v3\labels.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f'Labels CSV written: {csv_path}')

    import numpy as np
    arr = np.array([row[1:] for row in csv_rows])
    if len(arr) > 0:
        for i, name in enumerate(CLASS_NAMES):
            count = int(arr[:,i].sum())
            print(f'{name}: {count} ({count/len(arr)*100:.1f}%)')
    else:
        print('No patches extracted.')

if __name__ == '__main__':
    main()
