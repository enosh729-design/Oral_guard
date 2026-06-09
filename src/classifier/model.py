"""
OralGuard — ResNet50 Multi-Label Classifier
Classifies dental pathologies from 128×128 tooth patch images.

Output classes (multi-label):
    0: caries
    1: deep_caries
    2: periapical_lesion
    3: impacted_tooth

Design notes:
  - Dropout(p=0.4) is placed INSIDE forward() so it stays active
    during both training AND inference (needed for Monte Carlo dropout
    uncertainty estimation — see src/classifier/uncertainty.py).
  - Final activation is torch.sigmoid for independent per-class probabilities.
  - Pretrained ImageNet weights are loaded via torchvision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 4
CLASS_NAMES = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]
INPUT_SIZE = 128   # tooth patch size in pixels (square)
DROPOUT_P = 0.4


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class OralGuardClassifier(nn.Module):
    """
    ResNet50-based multi-label dental pathology classifier.

    MC Dropout is enabled by keeping Dropout INSIDE forward() —
    calling model.eval() does NOT disable it, which is intentional.
    This allows uncertainty estimation via repeated stochastic passes.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_p: float = DROPOUT_P,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Load ResNet50 backbone with optional ImageNet weights
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Extract feature layers (everything except the original FC head)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # After backbone: tensor of shape (B, 2048, 1, 1)

        # Classification head
        self.fc = nn.Linear(2048, num_classes)

        # Dropout stored as a module so it appears in state_dict
        # but NOTE: it is called explicitly in forward() to ensure
        # it is active regardless of model.train() / model.eval() state.
        self.dropout = nn.Dropout(p=dropout_p)

        self.num_classes = num_classes
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 128, 128).

        Returns:
            Sigmoid-activated predictions of shape (B, num_classes).
            Values in [0, 1]; each class is independent.
        """
        # Feature extraction
        features = self.features(x)          # (B, 2048, 1, 1)
        features = features.flatten(start_dim=1)   # (B, 2048)

        # Dropout INSIDE forward — active even when model.eval() is called
        # This is intentional for MC Dropout uncertainty estimation.
        features = self.dropout(features)    # (B, 2048)

        # Classification
        logits = self.fc(features)           # (B, num_classes)

        # Sigmoid for multi-label (not mutually exclusive)
        return torch.sigmoid(logits)         # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward(); returns class probabilities."""
        return self.forward(x)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"dropout_p={self.dropout_p}, "
            f"input_size={INPUT_SIZE}x{INPUT_SIZE}"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_model(
    num_classes: int = NUM_CLASSES,
    dropout_p: float = DROPOUT_P,
    pretrained: bool = True,
    weights_path: str | None = None,
) -> OralGuardClassifier:
    """
    Build and return the OralGuard classifier.

    Args:
        num_classes:   Number of output pathology classes (default 4).
        dropout_p:     MC Dropout probability (default 0.4).
        pretrained:    Load ImageNet weights for backbone if True.
        weights_path:  Optional path to a .pt checkpoint to load
                       (overrides pretrained backbone weights with
                       fine-tuned weights).

    Returns:
        OralGuardClassifier instance.

    Example:
        >>> model = get_model()
        >>> model = model.cuda()
        >>> dummy = torch.randn(1, 3, 128, 128).cuda()
        >>> out = model(dummy)
        >>> assert out.shape == torch.Size([1, 4])
    """
    model = OralGuardClassifier(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
    )

    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Support checkpoints saved as {'model_state_dict': ...} or plain state dict
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"Loaded weights from: {weights_path}")

    return model


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    model = get_model(pretrained=True).to(device)
    model.eval()   # Note: dropout stays active — this is intentional

    dummy = torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    out = model(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == torch.Size([4, NUM_CLASSES]), "Shape mismatch!"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Outputs not in [0, 1]!"
    print("✅ Model OK — output shape and range verified.")
    print(f"\nClass probabilities for batch[0]:")
    for name, prob in zip(CLASS_NAMES, out[0].detach().cpu().tolist()):
        print(f"  {name:<22}: {prob:.4f}")
