"""
OralGuard — Multi-Label Classifier Training Loop
Trains the ResNet50 classifier with BCE loss, LR scheduler,
early stopping, checkpoint saving, and MLflow logging.
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import mlflow
import mlflow.pytorch
from sklearn.metrics import f1_score

from src.classifier.model import get_model, CLASS_NAMES, INPUT_SIZE, NUM_CLASSES

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS_DIR = ROOT / "src" / "classifier" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ToothPatchDataset(Dataset):
    """
    Minimal dataset loader for tooth patch images with multi-label CSV.

    Expected CSV format (no header):
        image_path, caries, deep_caries, periapical_lesion, impacted_tooth
    """

    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    VAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, csv_path: str, transform=None) -> None:
        import csv
        self.samples: list[tuple[str, list[float]]] = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < NUM_CLASSES + 1:
                    continue
                img_path = row[0].strip()
                labels = [float(v) for v in row[1: NUM_CLASSES + 1]]
                self.samples.append((img_path, labels))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_f1(preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    """Compute macro-averaged F1 over all classes."""
    binary_preds = (preds >= threshold).astype(int)
    return float(f1_score(targets, binary_preds, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(
    csv_path: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 10,
    val_split: float = 0.2,
    dropout_p: float = 0.4,
    experiment_name: str = "oralguard-classifier",
    run_name: Optional[str] = None,
) -> str:
    """
    Full training loop for the OralGuard multi-label classifier.

    Args:
        csv_path:         Path to CSV with image paths + labels.
        epochs:           Maximum training epochs.
        batch_size:       Mini-batch size.
        lr:               Initial learning rate.
        patience:         Early stopping patience (epochs without val improvement).
        val_split:        Fraction of data used for validation.
        dropout_p:        MC Dropout probability.
        experiment_name:  MLflow experiment name.
        run_name:         MLflow run name (auto-generated if None).

    Returns:
        Path to the best saved checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ---- Dataset ----
    full_dataset = ToothPatchDataset(csv_path, transform=ToothPatchDataset.TRAIN_TRANSFORMS)
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # Override val transforms
    val_ds.dataset.transform = ToothPatchDataset.VAL_TRANSFORMS

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    logger.info(f"Dataset  : {len(train_ds)} train / {len(val_ds)} val samples")

    # ---- Model ----
    model = get_model(dropout_p=dropout_p, pretrained=True).to(device)

    # ---- Loss / Optimizer / Scheduler ----
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # ---- MLflow ----
    mlflow.set_experiment(experiment_name)
    run_name = run_name or f"resnet50-bce-{time.strftime('%Y%m%d-%H%M%S')}"

    best_val_loss = float("inf")
    early_stop_counter = 0
    best_ckpt_path = str(CHECKPOINTS_DIR / "best.pt")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "val_split": val_split,
            "dropout_p": dropout_p,
            "optimizer": "AdamW",
            "loss": "BCELoss",
            "scheduler": "ReduceLROnPlateau",
        })

        for epoch in range(1, epochs + 1):
            # ---- Train phase ----
            model.train()
            train_losses = []
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))

            # ---- Val phase ----
            model.eval()
            val_losses, all_preds, all_targets = [], [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    preds = model(images)
                    loss = criterion(preds, labels)
                    val_losses.append(loss.item())
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(labels.cpu().numpy())

            val_loss = float(np.mean(val_losses))
            all_preds   = np.concatenate(all_preds,   axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            val_f1 = compute_f1(all_preds, all_targets)

            # ---- LR scheduler step ----
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # ---- Logging ----
            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_F1={val_f1:.4f} | "
                f"lr={current_lr:.2e}"
            )
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "lr": current_lr,
            }, step=epoch)

            # ---- Checkpoint & early stopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                }, best_ckpt_path)
                logger.info(f"  ✅ New best model saved (val_loss={val_loss:.4f})")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

        mlflow.log_artifact(best_ckpt_path, artifact_path="checkpoints")
        mlflow.log_metric("best_val_loss", best_val_loss)

    logger.info(f"Training complete. Best checkpoint: {best_ckpt_path}")
    return best_ckpt_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train OralGuard classifier")
    parser.add_argument("--csv",      type=str, required=True, help="Path to label CSV")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=32)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    best = train(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        patience=args.patience,
    )
    print(f"Best model: {best}")
