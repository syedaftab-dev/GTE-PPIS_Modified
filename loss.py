import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weight(labels_list):
    """
    Compute the negative-to-positive ratio from a list of per-protein label arrays.
    Returns a float scalar. Use to build class_weights = [1.0, pos_weight].

    Args:
        labels_list: list of numpy/list label arrays, one per protein in the fold.
    """
    labels = [int(l) for row in labels_list for l in row]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for 2-class (binary) PPIS prediction.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (Tensor | None): Per-class weight tensor of shape [num_classes].
            Set to [1.0, neg_pos_ratio] to up-weight the minority (positive) class.
            If None, uses uniform weights (equivalent to standard CrossEntropyLoss).
        gamma (float): Focusing parameter. 0 -> standard (weighted) CE loss.
            Typical range: 0.5 - 5.0. Default: 2.0 (Lin et al. 2017, RetinaNet).
        reduction (str): 'mean' | 'sum' | 'none'.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha          # Tensor [C] or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) float tensor -- raw logits (NOT softmax output).
            targets: (N,) long tensor -- ground truth class indices.
        Returns:
            Scalar loss (if reduction='mean' or 'sum').
        """
        # Standard cross-entropy per sample, shape (N,)
        # weight=self.alpha handles the alpha_t per-class weighting.
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')

        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce_loss)

        # Focal modulation: down-weight easy examples
        focal_weight = (1.0 - pt) ** self.gamma

        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss  # 'none'
