"""
Loss functions optimized for severe class imbalance in graph fraud detection.
Includes:
- Binary Focal Loss (focuses learning on hard camouflaged fraud nodes)
- Class-Weighted Cross Entropy (inverse frequency weighting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits or probabilities [N, 2] or [N]
            targets: Binary labels [N] (0 or 1)
        """
        if inputs.dim() > 1 and inputs.shape[1] == 2:
            probs = F.softmax(inputs, dim=1)
            p_t = probs[:, 1] * targets + probs[:, 0] * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = -alpha_t * torch.pow(1.0 - p_t, self.gamma) * torch.log(p_t.clamp(min=1e-8))
        else:
            p = torch.sigmoid(inputs)
            p_t = p * targets + (1 - p) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = -alpha_t * torch.pow(1.0 - p_t, self.gamma) * torch.log(p_t.clamp(min=1e-8))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_criterion(loss_type: str = "focal", fraud_weight: float = 5.0) -> nn.Module:
    """Factory function for loss functions."""
    if loss_type == "focal":
        return FocalLoss(alpha=0.75, gamma=2.0)
    elif loss_type == "weighted_ce":
        weight = torch.tensor([1.0, fraud_weight])
        return nn.CrossEntropyLoss(weight=weight)
    else:
        return nn.CrossEntropyLoss()
