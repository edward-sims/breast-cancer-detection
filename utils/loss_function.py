import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ProbabilisticF1Loss(nn.Module):
    """
    Probabilistic F1 Loss for breast cancer detection

    This loss function directly optimizes the probabilistic F1 score,
    which is the evaluation metric for the competition.
    """

    def __init__(self, beta: float = 1.0, epsilon: float = 1e-8):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilistic F1 loss

        Args:
            predictions: Predicted logits [batch_size]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)

        # Probabilistic True Positives
        p_tp = torch.sum(probs * targets)

        # Probabilistic False Positives
        p_fp = torch.sum(probs * (1 - targets))

        # True Positives and False Negatives (for recall denominator)
        tp = torch.sum((probs > 0.5).float() * targets)
        fn = torch.sum((probs <= 0.5).float() * targets)

        # Probabilistic Precision
        p_precision = p_tp / (p_tp + p_fp + self.epsilon)

        # Probabilistic Recall
        p_recall = p_tp / (tp + fn + self.epsilon)

        # Probabilistic F1
        if p_precision + p_recall > 0:
            p_f1 = (
                (1 + self.beta**2)
                * (p_precision * p_recall)
                / ((self.beta**2 * p_precision) + p_recall + self.epsilon)
            )
        else:
            p_f1 = torch.tensor(0.0, device=predictions.device)

        # Return negative F1 as loss (we want to maximize F1)
        return 1.0 - p_f1


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    Focal loss down-weights easy examples and focuses on hard examples,
    which is particularly useful for medical imaging where positive cases are rare.
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss

        Args:
            predictions: Predicted logits [batch_size]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)

        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )

        # Focal loss components
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that combines multiple loss types

    This allows us to benefit from the strengths of different loss functions:
    - Focal loss for handling class imbalance
    - Probabilistic F1 loss for direct metric optimization
    - BCE loss for stability
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pf1_weight: float = 0.3,
        focal_weight: float = 0.5,
        bce_weight: float = 0.2,
    ):
        super().__init__()

        self.pf1_weight = pf1_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight

        self.pf1_loss = ProbabilisticF1Loss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss

        Args:
            predictions: Predicted logits [batch_size]
            targets: Ground truth labels [batch_size]

        Returns:
            Combined loss value
        """
        pf1_loss = self.pf1_loss(predictions, targets)
        focal_loss = self.focal_loss(predictions, targets)
        bce_loss = self.bce_loss(predictions, targets)

        total_loss = (
            self.pf1_weight * pf1_loss
            + self.focal_weight * focal_loss
            + self.bce_weight * bce_loss
        )

        return total_loss


class UncertaintyLoss(nn.Module):
    """
    Loss function that incorporates uncertainty estimation

    This loss encourages the model to be uncertain when it's not confident,
    which is important for medical applications.
    """

    def __init__(self, base_loss: nn.Module, uncertainty_weight: float = 0.1):
        super().__init__()
        self.base_loss = base_loss
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with uncertainty regularization

        Args:
            predictions: Predicted logits [batch_size]
            targets: Ground truth labels [batch_size]
            uncertainty: Predicted uncertainty [batch_size]

        Returns:
            Loss value with uncertainty regularization
        """
        # Base loss
        base_loss = self.base_loss(predictions, targets)

        # Uncertainty regularization (encourage uncertainty when wrong)
        probs = torch.sigmoid(predictions)
        prediction_errors = torch.abs(probs - targets)

        # Uncertainty should be high when prediction error is high
        uncertainty_loss = torch.mean(uncertainty * prediction_errors)

        total_loss = base_loss + self.uncertainty_weight * uncertainty_loss

        return total_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization

    Label smoothing helps prevent overconfidence and improves generalization
    by softening the target labels.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss

        Args:
            predictions: Predicted logits [batch_size]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        # Apply label smoothing
        smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        # BCE loss with smoothed targets
        loss = F.binary_cross_entropy_with_logits(predictions, smoothed_targets)

        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss for handling different costs of false positives vs false negatives

    In medical imaging, false negatives (missing cancer) are typically more costly
    than false positives (false alarm).
    """

    def __init__(self, fn_weight: float = 2.0, fp_weight: float = 1.0):
        super().__init__()
        self.fn_weight = fn_weight  # False negative weight
        self.fp_weight = fp_weight  # False positive weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss

        Args:
            predictions: Predicted logits [batch_size]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        probs = torch.sigmoid(predictions)

        # False negatives: predicted negative but actually positive
        fn_mask = (probs < 0.5) & (targets > 0.5)
        fn_loss = torch.sum(
            fn_mask.float()
            * self.fn_weight
            * F.binary_cross_entropy_with_logits(
                predictions[fn_mask], targets[fn_mask], reduction="none"
            )
        )

        # False positives: predicted positive but actually negative
        fp_mask = (probs > 0.5) & (targets < 0.5)
        fp_loss = torch.sum(
            fp_mask.float()
            * self.fp_weight
            * F.binary_cross_entropy_with_logits(
                predictions[fp_mask], targets[fp_mask], reduction="none"
            )
        )

        # True predictions
        true_mask = ~(fn_mask | fp_mask)
        true_loss = torch.sum(
            true_mask.float()
            * F.binary_cross_entropy_with_logits(
                predictions[true_mask], targets[true_mask], reduction="none"
            )
        )

        total_loss = (fn_loss + fp_loss + true_loss) / predictions.size(0)

        return total_loss


def get_loss_function(loss_type: str = "combined", **kwargs) -> nn.Module:
    """
    Factory function to create loss functions

    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function instance
    """
    if loss_type == "pf1":
        return ProbabilisticF1Loss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(**kwargs)
    elif loss_type == "asymmetric":
        return AsymmetricLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Legacy loss function for compatibility
def l2_norm(input, axis=1):
    """Legacy L2 normalization function"""
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output
