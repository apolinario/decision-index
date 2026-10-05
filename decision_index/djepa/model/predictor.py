"""
Latent Predictive Verifier (JEPA Predictor) performing M-step recurrent rollouts in latent space.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictorStep(nn.Module):
    """
    A single recurrent step in the JEPA latent thought space.
    Performs cross-attention to contextual state facts followed by latent feedforward deduction.
    """

    def __init__(self, d_model: int, nhead: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.norm0 = nn.LayerNorm(d_model)
        self.option_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        # Recurrent update gate (GRU style) for stable multi-step reasoning
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        h_state: torch.Tensor,
        state_mask: Optional[torch.Tensor] = None,
        option_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: [B, K, D] current latent hypothesis vectors
            h_state: [B, L, D] encoded state context
            state_mask: [B, L] boolean mask (True for padding, or standard PyTorch key_padding_mask)
            option_mask: [B, K] boolean mask (True for padding)
        Returns:
            z_next: [B, K, D]
        """
        # 1. Option self-attention: inter-candidate listwise interaction
        norm_z0 = self.norm0(z)
        opt_attn, _ = self.option_self_attn(
            query=norm_z0,
            key=norm_z0,
            value=norm_z0,
            key_padding_mask=option_mask,
        )
        z = z + opt_attn

        # 2. Cross-attention to extract facts conditioned on option hypothesis
        norm_z = self.norm1(z)
        attn_out, _ = self.cross_attn(
            query=norm_z,
            key=h_state,
            value=h_state,
            key_padding_mask=state_mask,
        )
        z_attn = z + attn_out

        # 3. Latent rule deduction
        norm_z2 = self.norm2(z_attn)
        ffn_out = self.ffn(norm_z2)

        # 4. Gated integration
        candidate = z_attn + ffn_out
        g = self.gate(torch.cat([z, candidate], dim=-1))
        z_next = g * candidate + (1.0 - g) * z
        return z_next


class LatentPredictiveVerifier(nn.Module):
    """
    M-step Recurrent Latent Predictive Verifier (D-JEPA Predictor).
    Simulates condition satisfaction in abstract embedding space without generating output tokens.
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        d_ff: int = 2048,
        num_steps: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_steps

        # Recurrent step cell
        self.cell = LatentPredictorStep(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout)

        # Final projection into target embedding metric space
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        h_state: torch.Tensor,
        s_options: torch.Tensor,
        state_attention_mask: Optional[torch.Tensor] = None,
        options_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_state: [B, L, D]
            s_options: [B, K, D] target embeddings of candidate criteria
            state_attention_mask: [B, L] (1 for valid token, 0 for pad)
            options_valid_mask: [B, K] (1 for valid option, 0 for pad)
        Returns:
            s_hat: [B, K, D] predicted condition representation in target space
        """
        # PyTorch MultiheadAttention key_padding_mask expects True for positions to be ignored (pad)
        key_padding_mask = None
        if state_attention_mask is not None:
            key_padding_mask = (state_attention_mask == 0)

        opt_key_padding_mask = None
        if options_valid_mask is not None:
            opt_key_padding_mask = (options_valid_mask == 0)

        # Initialize reasoning state from candidate criteria target embeddings
        z = s_options

        # Execute M-step recurrent latent rollout
        for _ in range(self.num_steps):
            z = self.cell(z, h_state, key_padding_mask, opt_key_padding_mask)

        s_hat = self.out_proj(z)  # [B, K, D]
        return s_hat
