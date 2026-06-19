"""
OralGuard — Full Test-Split Evaluation
Runs the entire detection + classification + uncertainty estimation pipeline
on all 35 test-split images, logs the results to a CSV file, and calculates summary statistics.
"""

import os
import sys
import csv
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector.fdi_mapper import map_to_fdi, fdi_label
from src.classifier.model import get_model, CLASS_NAMES
from src.classifier.uncertainty import mc_uncertainty, is_uncertain

# Paths
TEST_IMGS_DIR = ROOT / "data" / "dentex" / "images" / "test"
YOLO_WEIGHTS = ROOT / "src" / "detector" / "weights" / "oralguard_finetuned" / "weights" / "best.pt"
CLASSIFIER_WEIGHTS = ROOT / "src" / "classifier" / "checkpoints" / "best.pt"
CSV_OUT = ROOT / "outputs" / "test_evaluation_results.csv"

# Configuration
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MC_T = 30
UNCERTAINTY_THRESHOLD = 2.0
INPUT_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def crop_tooth_patch(bgr_img: np.ndarray, bbox: list) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    patch = bgr_img[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    patch = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE))
    return patch

def patch_to_tensor(patch_bgr: np.ndarray) -> torch.Tensor:
    from torchvision import transforms
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(patch_rgb)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tfm(pil_img).unsqueeze(0).to(DEVICE)

def run_evaluation():
    print("=== OralGuard Test-Split Full Evaluation ===")
    print(f"Device: {DEVICE}")
    
    # 1. Load models
    from ultralytics import YOLO
    print("Loading YOLO detector...")
    detector = YOLO(str(YOLO_WEIGHTS))
    
    print("Loading ResNet classifier...")
    classifier = get_model(pretrained=False, weights_path=str(CLASSIFIER_WEIGHTS)).to(DEVICE)
    classifier.eval()
    
    # 2. Get test images
    img_paths = sorted(TEST_IMGS_DIR.glob("*.png")) + sorted(TEST_IMGS_DIR.glob("*.jpg"))
    print(f"Found {len(img_paths)} test images.")
    
    rows = []
    total_detections = 0
    uncertain_count = 0
    
    # Track predicted classes (probability >= 0.5)
    class_pred_counts = {name: 0 for name in CLASS_NAMES}
    
    for idx, img_path in enumerate(img_paths):
        print(f"[{idx+1}/{len(img_paths)}] Processing {img_path.name}...")
        
        bgr_img = cv2.imread(str(img_path))
        if bgr_img is None:
            print(f"Error reading image: {img_path}")
            continue
            
        img_h, img_w = bgr_img.shape[:2]
        
        # Run detector
        res = detector(
            bgr_img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            agnostic_nms=True,
            verbose=False,
        )[0]
        
        if res.boxes is None:
            continue
            
        boxes = res.boxes.xyxy.cpu().numpy()
        
        # FDI mapping and classification
        image_findings = []
        for box in boxes:
            x1, y1, x2, y2 = box
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
            
            fdi = map_to_fdi(xc, yc, bw, bh, img_w, img_h)
            patch_bgr = crop_tooth_patch(bgr_img, box)
            tensor = patch_to_tensor(patch_bgr)
            
            # MC Uncertainty estimation
            mean_pred, entropy = mc_uncertainty(classifier, tensor, T=MC_T)
            probs = mean_pred[0].cpu().tolist()
            ent_val = entropy[0].item()
            uncertain = ent_val > UNCERTAINTY_THRESHOLD
            
            image_findings.append({
                "fdi": fdi,
                "probs": probs,
                "uncertain": uncertain,
                "entropy": ent_val,
            })
            
        # Deduplicate findings by FDI number (same as api/main.py)
        deduped = {}
        for f in image_findings:
            fdi = f["fdi"]
            if fdi not in deduped:
                deduped[fdi] = f
            else:
                f_max = max(f["probs"])
                existing_max = max(deduped[fdi]["probs"])
                if f_max > existing_max:
                    deduped[fdi] = f
                    
        # Log and accumulate stats
        for fdi, f in deduped.items():
            probs = f["probs"]
            uncertain = f["uncertain"]
            ent_val = f["entropy"]
            
            rows.append([
                img_path.name,
                fdi,
                round(probs[0], 6),
                round(probs[1], 6),
                round(probs[2], 6),
                round(probs[3], 6),
                1 if uncertain else 0,
                round(ent_val, 6)
            ])
            
            total_detections += 1
            if uncertain:
                uncertain_count += 1
                
            # Count class predictions (threshold 0.5)
            for j, p in enumerate(probs):
                if p >= 0.5:
                    class_pred_counts[CLASS_NAMES[j]] += 1
                    
    # 3. Write CSV file
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_filename", "tooth_fdi", 
            "prob_caries", "prob_deep_caries", "prob_periapical_lesion", "prob_impacted_tooth",
            "uncertain", "entropy"
        ])
        writer.writerows(rows)
        
    print("\n=== EVALUATION COMPLETED ===")
    print(f"Results written to: {CSV_OUT}")
    print(f"Total Tooth Detections (deduplicated): {total_detections}")
    print(f"Uncertain Predictions: {uncertain_count} / {total_detections} ({uncertain_count/total_detections*100:.2f}%)")
    print("Class Predictions Distribution (p >= 0.5):")
    for name, count in class_pred_counts.items():
        print(f"  {name:<22}: {count} ({count/total_detections*100:.2f}%)")

if __name__ == "__main__":
    run_evaluation()
