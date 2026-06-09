"""
OralGuard — GradCAM++ Explainability
Generates GradCAM++ heatmap overlays for dental pathology predictions.

Uses the grad-cam library (pytorch-grad-cam) targeting the last
convolutional layer of the ResNet50 backbone.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# pytorch-grad-cam
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.classifier.model import OralGuardClassifier, CLASS_NAMES, INPUT_SIZE

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_target_layer(model: OralGuardClassifier) -> nn.Module:
    """
    Return the last convolutional layer of the ResNet50 backbone.

    ResNet50 architecture (via model.features):
        features[0]  = Conv2d (stem)
        features[4]  = Layer1
        features[5]  = Layer2
        features[6]  = Layer3
        features[7]  = Layer4   ← target
        features[8]  = AdaptiveAvgPool2d
    The last residual block inside Layer4 contains the deepest conv layer.
    """
    # model.features is nn.Sequential(*list(resnet.children())[:-1])
    # Index 7 = layer4 (the last residual group)
    layer4 = model.features[7]
    # Last bottleneck block → last Conv2d in the last Bottleneck
    last_bottleneck = list(layer4.children())[-1]
    target_layer = last_bottleneck.conv3   # 2048-channel conv
    return target_layer


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (3, H, W) normalised tensor back to RGB uint8 numpy array.
    Denormalises using ImageNet mean/std.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = img * std + mean
    img = np.clip(img, 0, 1).astype(np.float32)
    return img


# ---------------------------------------------------------------------------
# Main GradCAM++ function
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: OralGuardClassifier,
    patch_tensor: torch.Tensor,
    class_idx: Optional[int] = None,
    save_path: Optional[str | Path] = None,
    filename_prefix: str = "gradcam",
    show_all_classes: bool = True,
) -> dict[str, np.ndarray]:
    """
    Generate GradCAM++ heatmaps for a tooth patch and save side-by-side images.

    Args:
        model:          Trained OralGuardClassifier.
        patch_tensor:   Input tensor of shape (1, 3, H, W) on any device.
        class_idx:      If given, only compute CAM for this class index.
                        If None (default), compute for all 4 classes.
        save_path:      Directory to save output images (default: outputs/gradcam/).
        filename_prefix: Prefix for saved filenames.
        show_all_classes: If True, produces a 2×2 panel with all class CAMs.

    Returns:
        Dict mapping class name to the RGB overlay numpy array (H, W, 3).

    Example:
        >>> model  = get_model(weights_path="best.pt").cuda()
        >>> patch  = preprocess_patch("tooth.png").cuda()
        >>> output = generate_gradcam(model, patch, save_path="outputs/gradcam/")
    """
    save_dir = Path(save_path) if save_path else OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    input_tensor = patch_tensor.to(device)

    target_layer = _get_target_layer(model)

    # grad-cam expects model in eval mode (we set it temporarily)
    model.eval()

    overlays: dict[str, np.ndarray] = {}

    classes_to_run = [class_idx] if class_idx is not None else list(range(len(CLASS_NAMES)))

    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        for idx in classes_to_run:
            class_name = CLASS_NAMES[idx]
            targets = [ClassifierOutputTarget(idx)]

            # grayscale_cam shape: (1, H, W)
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0]   # (H, W)

            # Convert tensor to RGB float image for overlay
            rgb_img = _tensor_to_rgb(input_tensor[0])
            rgb_img_resized = cv2.resize(rgb_img, (grayscale_cam.shape[1], grayscale_cam.shape[0]))

            # Create coloured overlay
            overlay = show_cam_on_image(rgb_img_resized, grayscale_cam, use_rgb=True)
            overlays[class_name] = overlay

            # Save individual side-by-side
            _save_side_by_side(
                original=rgb_img_resized,
                overlay=overlay,
                class_name=class_name,
                save_path=save_dir / f"{filename_prefix}_{class_name}.png",
            )

    # Optionally save a combined 2×2 panel
    if show_all_classes and len(overlays) == 4:
        _save_panel(
            overlays=overlays,
            original=_tensor_to_rgb(input_tensor[0]),
            save_path=save_dir / f"{filename_prefix}_panel.png",
        )

    return overlays


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _save_side_by_side(
    original: np.ndarray,
    overlay: np.ndarray,
    class_name: str,
    save_path: Path,
) -> None:
    """Save original patch | heatmap overlay side-by-side."""
    h, w = original.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w]  = (original * 255).astype(np.uint8)
    canvas[:, w:]  = overlay

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(canvas[:, :w])
    axes[0].set_title("Original Patch", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(canvas[:, w:])
    axes[1].set_title(f"GradCAM++ — {class_name}", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def _save_panel(
    overlays: dict[str, np.ndarray],
    original: np.ndarray,
    save_path: Path,
) -> None:
    """Save a 1 + 4 panel: original + all class GradCAM++ overlays."""
    fig = plt.figure(figsize=(14, 7))
    spec = gridspec.GridSpec(2, 3, figure=fig)

    # Original — center top
    ax_orig = fig.add_subplot(spec[0, 1])
    ax_orig.imshow((original * 255).astype(np.uint8))
    ax_orig.set_title("Original Patch", fontsize=12, fontweight="bold")
    ax_orig.axis("off")

    positions = [(0, 0), (0, 2), (1, 0), (1, 2)]
    for (row, col), (class_name, overlay) in zip(positions, overlays.items()):
        ax = fig.add_subplot(spec[row, col])
        ax.imshow(overlay)
        ax.set_title(class_name.replace("_", " ").title(), fontsize=10)
        ax.axis("off")

    plt.suptitle("OralGuard — GradCAM++ Explainability", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"GradCAM panel saved: {save_path}")


# ---------------------------------------------------------------------------
# Convenience: preprocess a raw image path → patch tensor
# ---------------------------------------------------------------------------

def preprocess_patch(image_path: str | Path, device: str = "cpu") -> torch.Tensor:
    """
    Load an image file and convert to normalised tensor ready for the model.

    Args:
        image_path: Path to image file.
        device:     Target device string.

    Returns:
        Tensor of shape (1, 3, 128, 128).
    """
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = tfm(img).unsqueeze(0).to(device)   # (1, 3, 128, 128)
    return tensor


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.classifier.model import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(pretrained=True).to(device)

    dummy_patch = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    print("Generating GradCAM++ for dummy patch...")
    overlays = generate_gradcam(
        model=model,
        patch_tensor=dummy_patch,
        save_path=OUTPUT_DIR,
        filename_prefix="test",
    )
    print(f"Generated CAMs for: {list(overlays.keys())}")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("✅ GradCAM++ module OK")
