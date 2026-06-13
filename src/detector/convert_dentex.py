"""
OralGuard — DENTEX COCO → YOLO Format Converter
Converts the DENTEX Challenge 2023 annotations (COCO JSON) into
YOLO segmentation format (.txt) expected by dental.yaml.

Run once before training:
    python src/detector/convert_dentex.py

Output structure created under data/dentex/:
    images/train/   ← symlinked/copied from training_data xrays
    images/val/     ← symlinked/copied from validation_data xrays
    images/test/    ← subset of val (no separate test split in DENTEX)
    labels/train/   ← converted YOLO .txt files
    labels/val/
    labels/test/
"""

from __future__ import annotations

import json
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT      = Path(__file__).resolve().parent.parent.parent
DENTEX    = ROOT / "data" / "dentex"

# Source paths (as extracted from the Kaggle zip)
TRAIN_IMGS   = DENTEX / "training_data"   / "training_data"   / "quadrant-enumeration-disease" / "xrays"
VAL_IMGS     = DENTEX / "validation_data" / "validation_data" / "quadrant_enumeration_disease" / "xrays"
TRAIN_JSON   = DENTEX / "training_data"   / "training_data"   / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json"

# DENTEX disease class ID → OralGuard class index
# DENTEX uses: 0=caries, 1=deep caries, 2=periapical lesion, 3=impacted tooth
DENTEX_TO_ORALGUARD = {0: 0, 1: 1, 2: 2, 3: 3}

# Output YOLO dirs
YOLO_TRAIN_IMGS   = DENTEX / "images"  / "train"
YOLO_VAL_IMGS     = DENTEX / "images"  / "val"
YOLO_TEST_IMGS    = DENTEX / "images"  / "test"
YOLO_TRAIN_LABELS = DENTEX / "labels"  / "train"
YOLO_VAL_LABELS   = DENTEX / "labels"  / "val"
YOLO_TEST_LABELS  = DENTEX / "labels"  / "test"

VAL_TEST_SPLIT = 0.5   # half of val becomes test


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x,y,w,h] to YOLO [cx,cy,w,h] normalised."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def convert_split(json_path: Path, src_img_dir: Path,
                  dst_img_dir: Path, dst_lbl_dir: Path,
                  image_ids: list[int] = None) -> int:
    """
    Convert a COCO JSON split to YOLO format.

    Args:
        json_path:    COCO annotation JSON file.
        src_img_dir:  Source image directory.
        dst_img_dir:  Destination image directory (YOLO layout).
        dst_lbl_dir:  Destination label directory (YOLO layout).
        image_ids:    If given, only convert these image IDs.

    Returns:
        Number of images converted.
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        logger.warning(f"JSON not found: {json_path} — skipping split.")
        return 0

    with open(json_path) as f:
        coco = json.load(f)

    # Build lookup maps
    id_to_image = {img["id"]: img for img in coco.get("images", [])}
    annotations_by_image: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    ids_to_process = image_ids if image_ids else list(id_to_image.keys())
    converted = 0

    for img_id in ids_to_process:
        if img_id not in id_to_image:
            continue
        img_info = id_to_image[img_id]
        file_name = img_info["file_name"]
        img_w     = img_info["width"]
        img_h     = img_info["height"]

        # Copy image
        src_img = src_img_dir / file_name
        dst_img = dst_img_dir / file_name
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Write YOLO label
        anns = annotations_by_image.get(img_id, [])
        label_file = dst_lbl_dir / (Path(file_name).stem + ".txt")
        lines = []
        for ann in anns:
            cat_id = ann.get("category_id", 0)
            # Map to 0-indexed OralGuard class
            yolo_cls = DENTEX_TO_ORALGUARD.get(cat_id - 1, 0)
            bbox = ann.get("bbox", [0, 0, 1, 1])
            cx, cy, nw, nh = coco_bbox_to_yolo(bbox, img_w, img_h)
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_file, "w") as lf:
            lf.write("\n".join(lines))
        converted += 1

    logger.info(f"Converted {converted} images → {dst_lbl_dir}")
    return converted


def main():
    import random
    import shutil
    logger.info("=== OralGuard — DENTEX COCO→YOLO Conversion (v2) ===")

    with open(TRAIN_JSON) as f:
        coco = json.load(f)

    id_to_image = {img["id"]: img for img in coco.get("images", [])}
    annotations_by_image: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    # ── DENTEX disease field is category_id_3 ──────────────────────────────
    # categories_3: {0:Impacted, 1:Caries, 2:Periapical Lesion, 3:Deep Caries}
    # OralGuard order: caries(0), deep_caries(1), periapical(2), impacted(3)
    DISEASE_MAP = {1: 0, 3: 1, 2: 2, 0: 3}

    all_ids = list(id_to_image.keys())
    # ── Keep only images that have ≥1 annotation (skip pure backgrounds) ──
    all_ids = [img_id for img_id in all_ids if img_id in annotations_by_image]
    logger.info(f"Images with annotations: {len(all_ids)} / {len(id_to_image)}")
    random.seed(42)
    random.shuffle(all_ids)

    # 80% train / 15% val / 5% test  — all with real labels
    n_total = len(all_ids)
    n_train = int(n_total * 0.80)
    n_val   = int(n_total * 0.15)

    train_ids = all_ids[:n_train]
    val_ids   = all_ids[n_train:n_train + n_val]
    test_ids  = all_ids[n_train + n_val:]

    logger.info(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    def write_split(ids, img_dir, lbl_dir):
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            img_info  = id_to_image[img_id]
            file_name = img_info["file_name"]
            img_w, img_h = img_info["width"], img_info["height"]

            src = TRAIN_IMGS / file_name
            dst = img_dir / file_name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

            anns = annotations_by_image.get(img_id, [])
            lines = []
            for ann in anns:
                cat_id   = ann.get("category_id_3", 1)   # ← correct field
                yolo_cls = DISEASE_MAP.get(cat_id, 0)
                x, y, w, h = ann.get("bbox", [0, 0, 1, 1])
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            lbl_file = lbl_dir / (Path(file_name).stem + ".txt")
            lbl_file.write_text("\n".join(lines))

    # Clear old splits
    for d in [YOLO_TRAIN_IMGS, YOLO_VAL_IMGS, YOLO_TEST_IMGS,
              YOLO_TRAIN_LABELS, YOLO_VAL_LABELS, YOLO_TEST_LABELS]:
        if d.exists():
            shutil.rmtree(d)

    write_split(train_ids, YOLO_TRAIN_IMGS, YOLO_TRAIN_LABELS)
    write_split(val_ids,   YOLO_VAL_IMGS,   YOLO_VAL_LABELS)
    write_split(test_ids,  YOLO_TEST_IMGS,  YOLO_TEST_LABELS)

    logger.info("Conversion complete with proper train/val/test splits.")
    logger.info("Next: python src/detector/yolo_trainer.py")


if __name__ == "__main__":
    main()
