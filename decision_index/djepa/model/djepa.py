"""
D-JEPA (Decision Joint Embedding Predictive Architecture) Model.
A lightweight, non-autoregressive System 1 decision engine.
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

from djepa.model.encoder import StateContextEncoder, CriteriaTargetEncoder
from djepa.model.predictor import LatentPredictiveVerifier
from djepa.model.scorer import LatentCompatibilityScorer
from djepa.dataset.formatter import JevFormatter


class DJEPA(nn.Module):
    """
    D-JEPA: Non-autoregressive System 1 decision model based on Yann LeCun's
    Joint Embedding Predictive Architecture (JEPA).
    
    Operates in latent representation space to evaluate candidate options directly,
    eliminating autoregressive text generation, CoT decoding latency, and parsing fragility.
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        pretrained: bool = True,
        num_steps: int = 4,
        d_ff: int = 2048,
        nhead: int = 8,
        dropout: float = 0.1,
        init_temperature: float = 10.0,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.state_encoder = StateContextEncoder(
            model_name=model_name,
            pretrained=pretrained,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.d_model = self.state_encoder.d_model

        # Shared backbone criteria encoder with projection to target embedding space
        self.criteria_encoder = CriteriaTargetEncoder(
            backbone=self.state_encoder.backbone,
            d_model=self.d_model,
        )

        # M-step Latent Predictive Verifier (reasoning in latent space)
        self.latent_predictor = LatentPredictiveVerifier(
            d_model=self.d_model,
            nhead=nhead,
            d_ff=d_ff,
            num_steps=num_steps,
            dropout=dropout,
        )

        # Latent Compatibility Scorer & Decision Head
        self.scorer = LatentCompatibilityScorer(
            d_model=self.d_model,
            init_temperature=init_temperature,
        )

    def forward(
        self,
        state_input_ids: torch.Tensor,
        state_attention_mask: torch.Tensor,
        option_input_ids: torch.Tensor,
        option_attention_mask: torch.Tensor,
        options_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single-pass forward evaluation of all candidate options.
        """
        # 1. Encode state context
        h_state, _ = self.state_encoder(
            input_ids=state_input_ids,
            attention_mask=state_attention_mask,
        )

        # 2. Encode candidate option targets
        s_options = self.criteria_encoder(
            option_input_ids=option_input_ids,
            option_attention_mask=option_attention_mask,
        )

        # 3. M-step latent predictive rollout in thought space
        s_hat = self.latent_predictor(
            h_state=h_state,
            s_options=s_options,
            state_attention_mask=state_attention_mask,
            options_valid_mask=options_valid_mask,
        )

        # 4. Latent compatibility scoring and probability readout
        decision_out = self.scorer(
            s_hat=s_hat,
            s_options=s_options,
            options_valid_mask=options_valid_mask,
        )

        return {
            "logits": decision_out["logits"],
            "probabilities": decision_out["probabilities"],
            "scores": decision_out["scores"],
            "s_hat": s_hat,
            "s_options": s_options,
            "temperature": decision_out["temperature"],
        }

    @torch.no_grad()
    def decide(
        self,
        record: Dict[str, Any],
        tokenizer,
        max_state_length: int = 4096,
        max_option_length: int = 256,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        High-level decision method conforming to the JevBench / System 1 contract.
        Returns the chosen label, exact probability distribution, and decision metadata.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        state_prompt, option_prompts, target_idx, trap_idx = JevFormatter.format_record(record)
        labels = record.get("labels", [])

        state_enc = tokenizer(
            state_prompt,
            max_length=max_state_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        opt_encs = [
            tokenizer(
                opt,
                max_length=max_option_length,
                truncation=True,
                return_tensors="pt",
            )
            for opt in option_prompts
        ]

        K = len(opt_encs)
        max_opt_len = max(o["input_ids"].size(1) for o in opt_encs)
        pad_id = tokenizer.pad_token_id or 0

        opt_ids = torch.full((1, K, max_opt_len), pad_id, dtype=torch.long, device=device)
        opt_mask = torch.zeros((1, K, max_opt_len), dtype=torch.long, device=device)

        for k, o in enumerate(opt_encs):
            l = o["input_ids"].size(1)
            opt_ids[0, k, :l] = o["input_ids"][0].to(device)
            opt_mask[0, k, :l] = o["attention_mask"][0].to(device)

        amp_context = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.no_grad()
        )

        with amp_context:
            out = self.forward(
                state_input_ids=state_enc["input_ids"],
                state_attention_mask=state_enc["attention_mask"],
                option_input_ids=opt_ids,
                option_attention_mask=opt_mask,
            )

        probs = out["probabilities"][0].float().cpu().numpy().tolist()
        scores = out["scores"][0].float().cpu().numpy().tolist()

        pred_idx = int(torch.argmax(out["probabilities"][0]).item())
        pred_label = labels[pred_idx] if pred_idx < len(labels) else "unknown"

        prob_dist = {str(label): float(probs[i]) for i, label in enumerate(labels)}
        score_dist = {str(label): float(scores[i]) for i, label in enumerate(labels)}

        # Hume confidence formula: c = (p_max - 1/K) / (1 - 1/K)
        p_max = float(probs[pred_idx])
        if K > 1:
            norm_conf = max(0.0, min(1.0, (p_max - (1.0 / K)) / (1.0 - (1.0 / K))))
        else:
            norm_conf = p_max

        return {
            "prediction": pred_label,
            "prediction_idx": pred_idx,
            "target_idx": target_idx,
            "trap_idx": trap_idx,
            "is_correct": (str(pred_label) == str(record.get("expected"))),
            "confidence": norm_conf,
            "raw_confidence": p_max,
            "probabilities": prob_dist,
            "scores": score_dist,
        }
