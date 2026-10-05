"""
Latent Compatibility Scorer and Decision Head for D-JEPA.
Maps latent predictions and target option representations to typed decision probabilities.
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentCompatibilityScorer(nn.Module):
    """
    Computes semantic compatibility scores between predicted latent state representations
    and candidate option representations, outputting calibrated decision probabilities.
    """

    def __init__(self, d_model: int = 768, init_temperature: float = 10.0):
        super().__init__()
        self.d_model = d_model
        # Learnable log-temperature parameter for probability calibration
        self.log_temperature = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(init_temperature))))
        )
        self.bias = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(
        self,
        s_hat: torch.Tensor,
        s_options: torch.Tensor,
        options_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            s_hat: [B, K, D] latent state representation predicted for each option
            s_options: [B, K, D] target representations of the candidate options
            options_valid_mask: [B, K] boolean mask (True for valid options)
        Returns:
            Dict containing:
                - 'scores': [B, K] raw normalized compatibility scores in [-1, 1]
                - 'logits': [B, K] calibrated logits
                - 'probabilities': [B, K] probability distribution summing to 1.0
        """
        # Cosine similarity in latent representation space
        norm_s_hat = F.normalize(s_hat, p=2, dim=-1)
        norm_s_options = F.normalize(s_options, p=2, dim=-1)
        scores = torch.sum(norm_s_hat * norm_s_options, dim=-1)  # [B, K]

        # Scaled logits for calibrated decision readout
        logits = self.temperature * scores + self.bias  # [B, K]

        # Mask out padded options when option count varies across records
        if options_valid_mask is not None:
            mask_val = -1e9
            logits = torch.where(
                options_valid_mask,
                logits,
                torch.tensor(mask_val, device=logits.device, dtype=logits.dtype),
            )

        probabilities = F.softmax(logits, dim=-1)

        return {
            "scores": scores,
            "logits": logits,
            "probabilities": probabilities,
            "temperature": self.temperature,
        }
