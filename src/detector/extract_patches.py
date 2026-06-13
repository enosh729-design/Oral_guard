"""
OralGuard — DENTEX Tooth Patch Extractor
Automatically crops individual tooth patches from DENTEX panoramic X-rays
using the COCO bounding box annotations, and creates a CSV label file
for training the ResNet50 multi-label classifier.

Run once before classifier training:
    python src/detector/extract_patches.py

Output:
    data/patches/images/   ← cropped 128×128 tooth patches
    data/patches/labels.csv ← image_path, caries, deep_caries, periapical_lesion, impacted_tooth
"""

from __future__ import annotations

import json
import csv
import cv2
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent.parent
DENTEX_DIR = ROOT / "data" / "dentex"
PATCHES_DIR = ROOT / "data" / "patches" / "images"
CSV_PATH    = ROOT / "data" / "patches" / "labels.csv"

# DENTEX annotation JSON (training split with disease labels)
TRAIN_JSON = (DENTEX_DIR / "training_data" / "training_data"
              / "quadrant-enumeration-disease"
              / "train_quadrant_enumeration_disease.json")
TRAIN_IMGS = (DENTEX_DIR / "training_data" / "training_data"
              / "quadrant-enumeration-disease" / "xrays")

PATCH_SIZE  = 128    # pixels (square output)
PADDING     = 10     # extra pixels around each bbox
NUM_CLASSES = 4

# DENTEX categories_3 disease IDs:
# 0=Impacted, 1=Caries, 2=Periapical Lesion, 3=Deep Caries
# → OralGuard class order: caries, deep_caries, periapical_lesion, impacted_tooth
CATEGORY_MAP = {1: 0, 3: 1, 2: 2, 0: 3}   # dentex_id → oralguard_idx
CLASS_NAMES  = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]
DISEASE_FIELD = "category_id_3"   # DENTEX uses 3 separate label fields


def extract_patches() -> int:
    """
    Extract all tooth patches from DENTEX training images.
    Returns the number of patches saved.
    """
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading DENTEX annotations...")
    with open(TRAIN_JSON) as f:
        coco = json.load(f)

    # Build lookups
    id_to_image   = {img["id"]: img for img in coco["images"]}
    # Group annotations by image_id
    anns_by_image: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    logger.info(f"Found {len(id_to_image)} images, "
                f"{len(coco.get('annotations',[]))} annotations")

    rows = []
    saved = 0
    skipped = 0

    for img_id, img_info in id_to_image.items():
        img_path = TRAIN_IMGS / img_info["file_name"]
        if not img_path.exists():
            skipped += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            skipped += 1
            continue

        h, w = bgr.shape[:2]
        anns = anns_by_image.get(img_id, [])

        for i, ann in enumerate(anns):
            # COCO bbox: [x, y, width, height]
            x, y, bw, bh = ann.get("bbox", [0, 0, 1, 1])
            # DENTEX uses category_id_3 for disease label
            cat_id = ann.get(DISEASE_FIELD, 1)

            # Add padding
            x1 = max(0, int(x) - PADDING)
            y1 = max(0, int(y) - PADDING)
            x2 = min(w, int(x + bw) + PADDING)
            y2 = min(h, int(y + bh) + PADDING)

            if x2 <= x1 or y2 <= y1:
                continue

            patch = bgr[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            patch_resized = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))

            # File name: imageid_annotationindex.png
            fname = f"{img_info['file_name'].replace('.png','').replace('.jpg','')}_{i:03d}.png"
            out_path = PATCHES_DIR / fname
            cv2.imwrite(str(out_path), patch_resized)

            # Build multi-label row — 1 for the detected class, 0 for others
            label = [0] * NUM_CLASSES
            cls_idx = CATEGORY_MAP.get(cat_id, 0)
            label[cls_idx] = 1

            rows.append([str(out_path)] + label)
            saved += 1

    # Write CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        # No header — ToothPatchDataset reads headerless CSV
        writer.writerows(rows)

    logger.info(f"✅ Saved {saved} patches to {PATCHES_DIR}")
    logger.info(f"✅ Labels CSV written to {CSV_PATH}")
    logger.info(f"   Skipped {skipped} images (not found)")

    # Class distribution
    labels_array = [r[1:] for r in rows]
    for i, name in enumerate(CLASS_NAMES):
        count = sum(1 for r in labels_array if r[i] == 1)
        logger.info(f"   {name:<22}: {count} patches")

    return saved


if __name__ == "__main__":
    n = extract_patches()
    print(f"\nDone! {n} patches ready for classifier training.")
    print(f"Next step:\n  python src/classifier/train.py --csv data/patches/labels.csv")
