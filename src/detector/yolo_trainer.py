"""
OralGuard — YOLOv8 Instance Segmentation Trainer
Trains a YOLOv8m-seg model on dental panoramic X-ray datasets.
"""

import os
import sys
import logging
from pathlib import Path

# Switch MLflow to SQLite backend (file store deprecated in latest MLflow)
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow/mlflow.db")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.pytorch
from ultralytics import YOLO, settings

# Disable built-in MLflow callbacks to avoid duplicate runs and callback crashes
settings.update({"mlflow": False})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # oralguard/
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_YAML = Path(__file__).resolve().parent / "dental.yaml"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "train_detector.log"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
TRAIN_CFG = dict(
    model="yolov8m.pt",           # detection model — works with 5-col bbox labels
    task="detect",
    data=str(DATASET_YAML),
    imgsz=1024,
    epochs=50,
    batch=4,                  # conservative for 8GB VRAM on RTX 4060 Laptop
    device=0,                 # GPU 0
    workers=4,
    patience=15,              # early stopping patience
    save=True,
    project=str(WEIGHTS_DIR),
    name="oralguard_det",
    exist_ok=True,
    pretrained=True,
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=1.0,
    # label_smoothing removed — deprecated in ultralytics 8.4.62+
    nbs=64,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    amp=True,                 # automatic mixed precision
    verbose=True,
    seed=42,
    deterministic=True,
    single_cls=False,
    rect=False,
    cos_lr=False,
    close_mosaic=10,
    resume=False,
    fraction=1.0,
    profile=False,
    freeze=None,
    multi_scale=False,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    val=True,
    plots=True,
)


def train(cfg: dict = None) -> str:
    """
    Train YOLOv8m-seg on the dental dataset.

    Args:
        cfg: Override dict merged on top of TRAIN_CFG defaults.

    Returns:
        Path to the best weights file.
    """
    config = {**TRAIN_CFG, **(cfg or {})}

    logger.info("=" * 60)
    logger.info("OralGuard — YOLOv8 Segmentation Trainer")
    logger.info("=" * 60)
    logger.info(f"Dataset  : {config['data']}")
    logger.info(f"Image sz : {config['imgsz']}")
    logger.info(f"Epochs   : {config['epochs']}")
    logger.info(f"Device   : cuda:{config['device']}")
    logger.info(f"Output   : {config['project']}/{config['name']}")

    # MLflow tracking — SQLite backend
    db_path = ROOT / "mlflow" / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("oralguard-detector")
    with mlflow.start_run(run_name="yolov8m-seg-dental"):
        mlflow.log_params({k: v for k, v in config.items()
                           if isinstance(v, (int, float, str, bool))})

        # Load and train
        model = YOLO(config["model"])
        results = model.train(**config)

        # Derive best weights path
        run_dir = Path(config["project"]) / config["name"]
        best_weights = run_dir / "weights" / "best.pt"
        if best_weights.exists():
            logger.info(f"✅ Best weights saved at: {best_weights}")
            mlflow.log_artifact(str(best_weights), artifact_path="weights")
        else:
            logger.warning("best.pt not found — check training output directory.")

        # Log final metrics
        import re
        if hasattr(results, "results_dict"):
            for k, v in results.results_dict.items():
                if isinstance(v, (int, float)):
                    # Sanitize metric name for MLflow (alphanumeric, dash, dot, space, underscore)
                    clean_k = re.sub(r'[^\w\-\. ]', '_', k)
                    mlflow.log_metric(clean_k, v)

        return str(best_weights)


def validate(weights_path: str) -> dict:
    """
    Validate the trained model on the test split.

    Args:
        weights_path: Path to .pt weights file.

    Returns:
        Validation metrics dict.
    """
    model = YOLO(weights_path)
    metrics = model.val(
        data=str(DATASET_YAML),
        imgsz=TRAIN_CFG["imgsz"],
        device=TRAIN_CFG["device"],
        split="test",
    )
    logger.info("Validation metrics:")
    for k, v in metrics.results_dict.items():
        logger.info(f"  {k}: {v:.4f}")
    return metrics.results_dict


def predict_single(weights_path: str, image_path: str, conf: float = 0.25):
    """
    Run inference on a single image.

    Args:
        weights_path: Path to trained .pt weights.
        image_path:   Path to input image.
        conf:         Confidence threshold.

    Returns:
        ultralytics Results object.
    """
    model = YOLO(weights_path)
    results = model.predict(
        source=image_path,
        imgsz=TRAIN_CFG["imgsz"],
        device=TRAIN_CFG["device"],
        conf=conf,
        save=True,
    )
    return results


if __name__ == "__main__":
    best = train()
    validate(best)
