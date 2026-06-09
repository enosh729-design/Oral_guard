"""
OralGuard — Active Learning Sampler
Selects the most uncertain unlabeled images for annotation
using Monte Carlo Dropout uncertainty estimation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.classifier.uncertainty import mc_uncertainty

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default transforms for unlabeled pool (no augmentation)
# ---------------------------------------------------------------------------

POOL_TRANSFORMS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Unlabeled pool dataset
# ---------------------------------------------------------------------------

class UnlabeledPool(Dataset):
    """
    Minimal dataset wrapper for a folder of unlabeled tooth patch images.

    Args:
        image_paths: List of absolute paths to image files.
        transform:   torchvision transform pipeline (default: POOL_TRANSFORMS).
    """

    def __init__(
        self,
        image_paths: list[str | Path],
        transform=None,
    ) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform or POOL_TRANSFORMS

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx   # return original index for tracking

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff"),
    ) -> "UnlabeledPool":
        """Build pool from all images in a directory (recursive)."""
        directory = Path(directory)
        paths = sorted([
            p for p in directory.rglob("*")
            if p.suffix.lower() in extensions
        ])
        logger.info(f"Unlabeled pool: {len(paths)} images from {directory}")
        return cls(image_paths=paths)


# ---------------------------------------------------------------------------
# Active learning query function
# ---------------------------------------------------------------------------

def query_most_uncertain(
    model: torch.nn.Module,
    pool: UnlabeledPool | list[str | Path],
    K: int = 50,
    T: int = 30,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> tuple[list[int], np.ndarray]:
    """
    Select the top-K most uncertain images from the unlabeled pool.

    Uncertainty is measured by predictive entropy from MC Dropout:
        H = -Σ p * log(p + ε)

    Higher entropy → model is more confused → best candidate for labeling.

    Args:
        model:      Trained OralGuardClassifier with MC Dropout in forward().
        pool:       UnlabeledPool dataset OR list of image file paths.
        K:          Number of images to select (default 50).
        T:          Number of MC Dropout passes per image (default 30).
        batch_size: Batch size for pool inference (default 16).
        device:     "cuda" or "cpu". Auto-detected if None.

    Returns:
        selected_indices: List of K pool indices with highest entropy,
                          sorted descending by uncertainty.
        entropy_scores:   Array of shape (len(pool),) with entropy per image.

    Example:
        >>> from src.classifier.model import get_model
        >>> model = get_model(weights_path="checkpoints/best.pt").cuda()
        >>> pool  = UnlabeledPool.from_directory("data/unlabeled/")
        >>> indices, scores = query_most_uncertain(model, pool, K=50, T=30)
        >>> print(f"Top uncertain image: {pool.image_paths[indices[0]]}")
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Accept raw path list
    if not isinstance(pool, UnlabeledPool):
        pool = UnlabeledPool(image_paths=pool)

    loader = DataLoader(
        pool,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,    # 0 workers for inference pool (avoid multiprocessing issues)
        pin_memory=(str(device) == "cuda"),
    )

    model = model.to(device)
    all_entropies = np.zeros(len(pool), dtype=np.float32)

    logger.info(
        f"Running MC Uncertainty (T={T}) on {len(pool)} unlabeled images "
        f"[batch_size={batch_size}, device={device}]"
    )

    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        _, entropy = mc_uncertainty(model, images, T=T)
        ent_np = entropy.cpu().numpy()
        for i, pool_idx in enumerate(indices.numpy()):
            all_entropies[pool_idx] = ent_np[i]

    # Descending sort by entropy: highest uncertainty first
    ranked_indices = np.argsort(all_entropies)[::-1].tolist()
    selected_indices = ranked_indices[:K]

    logger.info(
        f"Selected top-{K} uncertain images. "
        f"Entropy range: [{all_entropies[selected_indices[-1]]:.4f}, "
        f"{all_entropies[selected_indices[0]]:.4f}]"
    )

    return selected_indices, all_entropies


def export_query_manifest(
    pool: UnlabeledPool,
    selected_indices: list[int],
    entropy_scores: np.ndarray,
    output_path: str | Path = "active_learning_manifest.csv",
) -> Path:
    """
    Save selected images + entropy scores to a CSV for annotation workflows.

    Args:
        pool:             The unlabeled pool dataset.
        selected_indices: List of selected indices from query_most_uncertain.
        entropy_scores:   Full entropy array (all pool images).
        output_path:      Where to save the CSV.

    Returns:
        Path to the written CSV file.
    """
    import csv
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "pool_index", "image_path", "entropy"])
        for rank, idx in enumerate(selected_indices, 1):
            writer.writerow([
                rank,
                idx,
                str(pool.image_paths[idx]),
                f"{entropy_scores[idx]:.6f}",
            ])

    logger.info(f"Active learning manifest saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.classifier.model import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(pretrained=False).to(device)

    # Synthetic pool of random tensors for testing
    class _FakePool(UnlabeledPool):
        def __init__(self, n=100):
            self.image_paths = [Path(f"fake_{i}.png") for i in range(n)]
            self.transform = None

        def __getitem__(self, idx):
            return torch.randn(3, 128, 128), idx

    pool = _FakePool(n=100)
    indices, scores = query_most_uncertain(model, pool, K=10, T=10, batch_size=16)
    print(f"Top-10 uncertain indices: {indices}")
    print(f"Their entropy scores:     {[f'{scores[i]:.4f}' for i in indices]}")
    print("✅ Active learning sampler OK")
