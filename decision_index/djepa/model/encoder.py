"""
Encoders for State Context and Candidate Option Criteria.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel


class StateContextEncoder(nn.Module):
    """
    Encodes long document state and question instructions into contextual sequence embeddings.
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        pretrained: bool = True,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)

        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        self.d_model = self.backbone.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
        Returns:
            h_state: [B, L, D]
            attention_mask: [B, L]
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h_state = outputs.last_hidden_state  # [B, L, D]
        return h_state, attention_mask


class CriteriaTargetEncoder(nn.Module):
    """
    Encodes candidate option criteria into compact semantic target embeddings s_O in latent space.
    """

    def __init__(self, backbone: PreTrainedModel, d_model: int):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        option_input_ids: torch.Tensor,
        option_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            option_input_ids: [B, K, L_opt]
            option_attention_mask: [B, K, L_opt]
        Returns:
            s_options: [B, K, D] target embeddings in latent space
        """
        B, K, L = option_input_ids.shape
        flat_ids = option_input_ids.view(B * K, L)
        flat_mask = (
            option_attention_mask.view(B * K, L)
            if option_attention_mask is not None
            else None
        )

        outputs = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
        hidden = outputs.last_hidden_state  # [B*K, L, D]

        # Mean pooling over valid tokens
        if flat_mask is not None:
            mask_expanded = flat_mask.unsqueeze(-1).expand_as(hidden).float()
            sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            pooled = hidden.mean(dim=1)

        s_options = self.pool_proj(pooled)  # [B*K, D]
        return s_options.view(B, K, self.d_model)
