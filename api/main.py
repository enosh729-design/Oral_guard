"""
OralGuard — FastAPI Inference Server
POST /predict  → accepts panoramic X-ray → returns per-tooth pathology JSON

Pipeline:
    1. YOLOv8 detects & crops individual teeth from the panoramic OPG
    2. FDI mapper assigns tooth numbers to each detection
    3. ResNet50 classifier predicts pathologies per tooth patch
    4. MC Dropout estimates uncertainty per tooth
    5. GradCAM++ generates heatmaps for uncertain predictions
    6. JSON response with full structured findings
"""

from __future__ import annotations

import io
import os
import time
import uuid
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# OralGuard modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector.fdi_mapper import map_to_fdi, fdi_label
from src.classifier.model import get_model, CLASS_NAMES
from src.classifier.uncertainty import mc_uncertainty, is_uncertain
from src.explainability.gradcam import generate_gradcam

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS   = ROOT / "src" / "detector" / "weights" / "oralguard_det" / "weights" / "best.pt"
CLASSIFIER_WEIGHTS = ROOT / "src" / "classifier" / "checkpoints" / "best.pt"
OUTPUT_DIR     = ROOT / "outputs" / "gradcam"
CONF_THRESHOLD = 0.25
MC_T           = 30
UNCERTAINTY_THRESHOLD = 0.5
INPUT_SIZE     = 128

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OralGuard API",
    description="Uncertainty-aware multi-task dental pathology detection from panoramic X-rays.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loaders (lazy-loaded once at startup)
# ---------------------------------------------------------------------------

_yolo_model   = None
_clf_model    = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        if not YOLO_WEIGHTS.exists():
            raise RuntimeError(
                f"YOLO weights not found at {YOLO_WEIGHTS}. "
                "Please train the detector first (src/detector/yolo_trainer.py)."
            )
        _yolo_model = YOLO(str(YOLO_WEIGHTS))
        logger.info(f"YOLO model loaded from {YOLO_WEIGHTS}")
    return _yolo_model


def get_classifier():
    global _clf_model
    if _clf_model is None:
        weights_path = str(CLASSIFIER_WEIGHTS) if CLASSIFIER_WEIGHTS.exists() else None
        _clf_model = get_model(
            pretrained=(weights_path is None),
            weights_path=weights_path,
        ).to(DEVICE)
        logger.info(
            f"Classifier loaded | weights={'checkpoint' if weights_path else 'ImageNet pretrained'}"
        )
    return _clf_model


# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------

class ToothFinding(BaseModel):
    tooth_id: int
    tooth_label: str
    findings: list[str]
    confidence: dict[str, float]
    uncertain: bool
    gradcam_path: Optional[str]


class PredictResponse(BaseModel):
    request_id: str
    image_filename: str
    num_teeth_detected: int
    processing_time_ms: float
    findings: list[ToothFinding]


class HealthResponse(BaseModel):
    status: str
    device: str
    yolo_ready: bool
    classifier_ready: bool


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to BGR numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure it is a valid PNG/JPG.")
    return img


def crop_tooth_patch(bgr_img: np.ndarray, bbox: dict) -> np.ndarray:
    """Crop a tooth region from the full panoramic image."""
    h, w = bgr_img.shape[:2]
    x1 = max(0, int(bbox["x1"]))
    y1 = max(0, int(bbox["y1"]))
    x2 = min(w, int(bbox["x2"]))
    y2 = min(h, int(bbox["y2"]))
    patch = bgr_img[y1:y2, x1:x2]
    patch = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE))
    return patch


def patch_to_tensor(patch_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR uint8 patch to normalised float tensor (1, 3, H, W)."""
    from torchvision import transforms
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(patch_rgb)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tfm(pil_img).unsqueeze(0).to(DEVICE)   # (1, 3, 128, 128)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check API and model readiness."""
    return HealthResponse(
        status="ok",
        device=DEVICE,
        yolo_ready=YOLO_WEIGHTS.exists(),
        classifier_ready=CLASSIFIER_WEIGHTS.exists(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Run the full OralGuard inference pipeline on an uploaded panoramic X-ray.

    - Accepts JPEG or PNG images.
    - Returns structured JSON with per-tooth pathology findings,
      confidence scores, uncertainty flags, and GradCAM heatmap paths.
    """
    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]

    # ---- Read and decode image ----
    contents = await file.read()
    try:
        bgr_img = bytes_to_cv2(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    img_h, img_w = bgr_img.shape[:2]
    logger.info(f"[{request_id}] Received '{file.filename}' — {img_w}×{img_h}px")

    # ---- YOLO detection ----
    try:
        yolo = get_yolo()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    yolo_results = yolo.predict(
        source=rgb_img,
        imgsz=1024,
        conf=CONF_THRESHOLD,
        device=0 if DEVICE == "cuda" else "cpu",
        verbose=False,
    )

    # Parse detections
    detections = []
    result = yolo_results[0]
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()   # (N, 4) x1y1x2y2
        for box in boxes:
            x1, y1, x2, y2 = box
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "x_center": xc, "y_center": yc,
                "width": bw, "height": bh,
            })

    logger.info(f"[{request_id}] YOLO detected {len(detections)} teeth")

    if not detections:
        elapsed = (time.perf_counter() - t_start) * 1000
        return PredictResponse(
            request_id=request_id,
            image_filename=file.filename or "unknown",
            num_teeth_detected=0,
            processing_time_ms=round(elapsed, 1),
            findings=[],
        )

    # ---- Per-tooth classification ----
    clf = get_classifier()
    findings: list[ToothFinding] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, det in enumerate(detections):
        # FDI mapping
        fdi = map_to_fdi(
            det["x_center"], det["y_center"],
            det["width"],    det["height"],
            img_w, img_h,
        )

        # Crop patch
        patch_bgr = crop_tooth_patch(bgr_img, det)
        tensor    = patch_to_tensor(patch_bgr)

        # MC uncertainty
        mean_pred, entropy = mc_uncertainty(clf, tensor, T=MC_T)
        probs      = mean_pred[0].detach().cpu().tolist()   # list[float] len=4
        ent_val    = entropy[0].item()
        uncertain  = bool(is_uncertain(ent_val, threshold=UNCERTAINTY_THRESHOLD))

        # Detected findings (probability > 0.5)
        detected = [CLASS_NAMES[j] for j, p in enumerate(probs) if p >= 0.5]

        # GradCAM (only for uncertain or positive predictions)
        gradcam_path: Optional[str] = None
        if uncertain or detected:
            prefix = f"{request_id}_tooth{i}_fdi{fdi}"
            try:
                overlays = generate_gradcam(
                    model=clf,
                    patch_tensor=tensor,
                    save_path=OUTPUT_DIR,
                    filename_prefix=prefix,
                    show_all_classes=False,
                )
                gradcam_path = str(OUTPUT_DIR / f"{prefix}_{CLASS_NAMES[0]}.png")
            except Exception as e:
                logger.warning(f"GradCAM failed for tooth {fdi}: {e}")

        findings.append(ToothFinding(
            tooth_id=fdi,
            tooth_label=fdi_label(fdi),
            findings=detected,
            confidence={
                CLASS_NAMES[j]: round(float(probs[j]), 4)
                for j in range(len(CLASS_NAMES))
            },
            uncertain=uncertain,
            gradcam_path=gradcam_path,
        ))

    # Sort by FDI number
    findings.sort(key=lambda x: x.tooth_id)

    elapsed = (time.perf_counter() - t_start) * 1000
    logger.info(f"[{request_id}] Done in {elapsed:.1f}ms")

    return PredictResponse(
        request_id=request_id,
        image_filename=file.filename or "unknown",
        num_teeth_detected=len(findings),
        processing_time_ms=round(elapsed, 1),
        findings=findings,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
