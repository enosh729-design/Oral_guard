"""
OralGuard — Monte Carlo Dropout Uncertainty Estimation

Uses MC Dropout to estimate predictive uncertainty for multi-label
dental pathology classification. Dropout stays active during inference
because it is placed inside forward() in OralGuardClassifier.

References:
    Gal & Ghahramani (2016) — "Dropout as a Bayesian Approximation"
    Kendall & Gal (2017)    — "What Uncertainties Do We Need in Bayesian
                               Deep Learning for Computer Vision?"
"""

from __future__ import annotations

import torch
import numpy as np


# ---------------------------------------------------------------------------
# MC Uncertainty estimation
# ---------------------------------------------------------------------------

def mc_uncertainty(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    T: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate predictive uncertainty via Monte Carlo Dropout.

    The model's Dropout is already in forward(); calling model.train()
    ensures BatchNorm layers (if any) also behave stochastically,
    but the key effect is that Dropout remains active for T passes.

    Args:
        model:        OralGuardClassifier (or any model with dropout in forward).
        image_tensor: Input patch tensor of shape (B, 3, H, W).
                      Should already be on the same device as the model.
        T:            Number of stochastic forward passes (default 30).

    Returns:
        mean_prediction: Tensor of shape (B, num_classes) — averaged
                         sigmoid probabilities across T passes.
        entropy_score:   Tensor of shape (B,) — predictive entropy
                         (higher = more uncertain) summed over classes.

    Example:
        >>> model = get_model().cuda()
        >>> patch = torch.randn(1, 3, 128, 128).cuda()
        >>> mean_pred, entropy = mc_uncertainty(model, patch, T=30)
        >>> print(mean_pred.shape)   # (1, 4)
        >>> print(entropy.shape)     # (1,)
    """
    # Set to train mode so Dropout is active (BN uses running stats in eval,
    # but we force train mode for full stochasticity)
    model.train()

    with torch.no_grad():
        # Collect T stochastic predictions
        stochastic_preds = []
        for _ in range(T):
            logits = model(image_tensor)   # (B, num_classes)
            preds = torch.sigmoid(logits)
            stochastic_preds.append(preds.unsqueeze(0))   # (1, B, num_classes)

        # Stack → (T, B, num_classes)
        stacked = torch.cat(stochastic_preds, dim=0)

    # Mean prediction across T passes → (B, num_classes)
    mean_prediction = stacked.mean(dim=0)

    # Predictive entropy: H = -Σ p * log(p + ε), summed over classes
    # Higher entropy = higher uncertainty
    eps = 1e-8
    p = mean_prediction.clamp(eps, 1.0 - eps)
    entropy_per_class = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    entropy_score = entropy_per_class.sum(dim=1)   # (B,)

    return mean_prediction, entropy_score


# ---------------------------------------------------------------------------
# Uncertainty threshold helper
# ---------------------------------------------------------------------------

def is_uncertain(
    entropy: torch.Tensor | float,
    threshold: float = 1.5,
) -> bool | list[bool]:
    # Threshold calibrated for 4-class multi-label BCE. 
    # Max possible entropy = 4 * ln(2) ≈ 2.77. 
    # 1.5 flags predictions where average class confidence is genuinely low.
    """
    Determine whether a prediction is uncertain based on entropy threshold.

    Args:
        entropy:   Scalar entropy value or tensor of shape (B,).
        threshold: Entropy threshold above which a prediction is uncertain.
                   Default 1.5 (tuned for 4-class binary entropy range [0, 2.77]).

    Returns:
        Single bool (if scalar/0-d tensor) or list of bools (if batch tensor).

    Example:
        >>> flag = is_uncertain(1.8, threshold=1.5)
        >>> assert flag is True
    """
    if isinstance(entropy, (int, float)):
        return float(entropy) > threshold

    if isinstance(entropy, torch.Tensor):
        if entropy.dim() == 0:
            return bool(entropy.item() > threshold)
        return [bool(e.item() > threshold) for e in entropy]

    # numpy array
    return list(np.array(entropy) > threshold)


# ---------------------------------------------------------------------------
# Variance-based uncertainty (alternative metric)
# ---------------------------------------------------------------------------

def mc_variance(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    T: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute variance-based uncertainty (epistemic uncertainty proxy).

    Args:
        model:        Classifier with dropout in forward().
        image_tensor: Input tensor (B, 3, H, W).
        T:            Number of MC passes.

    Returns:
        mean_prediction: (B, num_classes)
        variance:        (B, num_classes) — variance across T passes per class.
    """
    model.train()

    with torch.no_grad():
        preds_list = []
        for _ in range(T):
            logits = model(image_tensor)
            probs = torch.sigmoid(logits)
            preds_list.append(probs.unsqueeze(0))

        stacked = torch.cat(preds_list, dim=0)   # (T, B, num_classes)

    mean_prediction = stacked.mean(dim=0)
    variance = stacked.var(dim=0)
    return mean_prediction, variance


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
    from src.classifier.model import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = get_model(pretrained=False).to(device)

    # Single-image test
    patch = torch.randn(1, 3, 128, 128, device=device)
    mean_pred, entropy = mc_uncertainty(model, patch, T=30)

    print(f"mean_prediction shape : {mean_pred.shape}")    # (1, 4)
    print(f"entropy_score shape   : {entropy.shape}")      # (1,)
    print(f"mean_prediction       : {mean_pred}")
    print(f"entropy_score         : {entropy.item():.4f}")
    print(f"is_uncertain (0.5)    : {is_uncertain(entropy)}")

    # Batch test
    batch = torch.randn(4, 3, 128, 128, device=device)
    mean_b, ent_b = mc_uncertainty(model, batch, T=20)
    print(f"\nBatch mean shape : {mean_b.shape}")   # (4, 4)
    print(f"Batch entropy    : {ent_b}")
    flags = is_uncertain(ent_b, threshold=0.5)
    print(f"Uncertain flags  : {flags}")
    print("✅ Uncertainty module OK")
