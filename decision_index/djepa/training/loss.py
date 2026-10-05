"""
Loss functions for D-JEPA: CrossEntropy, Brier Score calibration, and Anti-Lure contrastive loss.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DJEPACombinedLoss(nn.Module):
    """
    Multi-objective loss function enforcing:
    1. Correct decision selection (NLL / Cross-Entropy)
    2. Strict probability calibration (Brier Score)
    3. Immunity to surface lexical lures (Anti-Lure Margin Loss)
    """

    def __init__(
        self,
        brier_weight: float = 1.0,
        anti_lure_weight: float = 1.5,
        anti_lure_margin: float = 0.5,
    ):
        super().__init__()
        self.brier_weight = brier_weight
        self.anti_lure_weight = anti_lure_weight
        self.anti_lure_margin = anti_lure_margin

    def forward(
        self,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
        scores: torch.Tensor,
        targets: torch.Tensor,
        trap_indices: Optional[torch.Tensor] = None,
        options_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, K] calibrated decision logits
            probabilities: [B, K] output decision probabilities
            scores: [B, K] normalized compatibility scores in [-1, 1]
            targets: [B] ground truth option indices
            trap_indices: [B] (optional, index of trap distractor, -100 if none)
            options_valid_mask: [B, K]
        Returns:
            Dict containing 'loss', 'loss_nll', 'loss_brier', 'loss_anti_lure'
        """
        B, K = logits.shape
        device = logits.device

        # 1. Negative Log-Likelihood (CrossEntropy)
        loss_nll = F.cross_entropy(logits, targets)

        # 2. Brier Score Loss (Proper scoring rule for calibration)
        one_hot_targets = F.one_hot(targets, num_classes=K).float()
        if options_valid_mask is not None:
            # Zero out padded option dimensions
            valid_float = options_valid_mask.float()
            brier_diff = (probabilities - one_hot_targets) * valid_float
        else:
            brier_diff = probabilities - one_hot_targets
        loss_brier = torch.mean(torch.sum(brier_diff ** 2, dim=-1))

        # 3. Anti-Lure Margin Loss
        loss_anti_lure = torch.tensor(0.0, device=device)
        if trap_indices is not None:
            has_trap = (trap_indices >= 0) & (trap_indices < K)
            if has_trap.any():
                trap_b = torch.nonzero(has_trap).squeeze(-1)
                t_idx = targets[trap_b]
                trap_idx = trap_indices[trap_b]

                target_scores = scores[trap_b, t_idx]
                trap_scores = scores[trap_b, trap_idx]

                # Correct target score should exceed trap score by at least margin:
                # trap_score - target_score + margin <= 0
                lure_violations = F.relu(trap_scores - target_scores + self.anti_lure_margin)
                loss_anti_lure = torch.mean(lure_violations)

        total_loss = (
            loss_nll
            + self.brier_weight * loss_brier
            + self.anti_lure_weight * loss_anti_lure
        )

        return {
            "loss": total_loss,
            "loss_nll": loss_nll,
            "loss_brier": loss_brier,
            "loss_anti_lure": loss_anti_lure,
        }
