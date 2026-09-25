"""decision_index.engines.gevva_engine
======================================
In-process Decision Index Engine adapter for Gevva (Gemma 4 Multimodal System 1 Decision Engine).
Non-autoregressive cross-encoder: scores typed decisions in a single forward pass without token generation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from decision_index.engines.base import Engine, Unsupported, text


def _softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    arr = np.array(logits, dtype=np.float64) / max(temperature, 1e-4)
    exp = np.exp(arr - np.max(arr))
    probs = exp / np.sum(exp)
    probs = probs / np.sum(probs)
    return probs.tolist()


class GevvaEngine(Engine):
    name = "gevva"
    latency = "Device-synchronized in-process request wall time including NLI formatting and cross-encoder inference."

    def __init__(
        self,
        model: str = "davidburhans/gevva-e4b",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        max_length: int = 4096,
        temperature: float = 1.0,
        scoring: str = "margin",
        batch_size: int = 16,
        **options,
    ):
        super().__init__(**options)
        try:
            import torch
            from gevva import load
        except ImportError as e:
            raise ImportError(
                "The gevva package and PyTorch are required to run GevvaEngine. "
                "Install with: pip install gevva"
            ) from e

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = float(temperature)
        self.scoring = scoring
        self.model_id = model
        self.max_length = max_length
        self.batch_size = batch_size

        self.model = load(
            model,
            device=self.device,
            dtype=dtype,
            max_length=max_length,
            batch_size=batch_size,
        )

        self.provenance = {
            "kind": "gevva_cross_encoder",
            "repo": model,
            "device": self.device,
            "dtype": dtype or "bfloat16",
            "scoring": scoring,
            "temperature": self.temperature,
            "max_length": max_length,
            "architecture": "Gemma 4 Multimodal System 1 Decision Engine",
            "policy": "Non-autoregressive cross-encoder scoring each candidate option via calibrated NLI entailment-contradiction margin. No autoregressive generation tokens.",
        }

    def runtime(self) -> Dict[str, Any]:
        info = {
            "torch": self.torch.__version__,
            "device": self.device,
            "model": self.model_id,
        }
        if self.device == "cuda" and self.torch.cuda.is_available():
            info.update(
                cuda=self.torch.version.cuda,
                gpu=self.torch.cuda.get_device_name(),
            )
        try:
            import gevva
            info["gevva"] = getattr(gevva, "__version__", "1.0.0")
        except Exception:
            pass
        return info

    def synchronize(self):
        if self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def warmup(self):
        warm = {
            "warmup": {
                "type": "choice",
                "instructions": "Which color is named?",
                "criteria": {"red": "red", "blue": "blue"},
            }
        }
        self("The color is red.", warm)

    def __call__(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        state_str = text(state) if state not in ("", None, {}, []) else ""
        answers: Dict[str, Dict[str, Any]] = {}
        raw: Dict[str, Any] = {}
        total_tokens = 0

        for q_id, q_dict in questions.items():
            q_type = q_dict.get("type", "choice")
            if q_type not in ("choice", "noul"):
                raise Unsupported(f"Unsupported question type: {q_type}")

            instructions = text(q_dict.get("instructions", ""))
            criteria = q_dict.get("criteria", {})

            if q_type == "choice":
                keys = list(criteria.keys())
                if not keys:
                    raise Unsupported(f"Question '{q_id}' has no criteria options.")

                if state_str and instructions:
                    premise = f"{state_str}\n\nQuestion: {instructions}"
                elif state_str:
                    premise = state_str
                else:
                    premise = instructions

                pairs = []
                for k in keys:
                    desc = text(criteria[k])
                    if desc and desc.lower() != str(k).lower():
                        hyp = f"The correct answer is {k}: {desc}."
                    else:
                        hyp = f"The correct answer is: {k}."
                    pairs.append((premise, hyp))

                probs = self.model.predict(pairs)
                # NLI indices: 0=contradiction, 1=entailment, 2=neutral
                p_con = probs[:, 0]
                p_ent = probs[:, 1]

                if self.scoring == "contrastive":
                    scores = (p_ent / (p_ent + p_con + 1e-6)).tolist()
                elif self.scoring == "entailment":
                    scores = p_ent.tolist()
                else:  # margin
                    scores = (p_ent - p_con).tolist()

                probs_list = _softmax(scores, temperature=self.temperature)
                choice = keys[int(np.argmax(probs_list))]
                prob_map = {k: float(p) for k, p in zip(keys, probs_list)}

                answers[q_id] = {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": prob_map,
                }
                raw[q_id] = {
                    "option_scores": dict(zip(keys, scores)),
                    "pair_count": len(pairs),
                }

            elif q_type == "noul":
                if state_str and instructions:
                    premise = state_str
                    hypothesis = instructions
                elif state_str:
                    premise = state_str
                    hypothesis = "The claim is true."
                else:
                    premise = instructions
                    hypothesis = "The statement is true."

                probs = self.model.predict([(premise, hypothesis)])
                p_con, p_ent, p_neu = probs[0]
                if p_ent + p_con > 1e-6:
                    p_true = float(p_ent / (p_ent + p_con))
                else:
                    p_true = float(p_ent)
                p_true = float(np.clip(p_true, 0.0, 1.0))

                answers[q_id] = {
                    "type": "noul",
                    "noul": p_true,
                }
                raw[q_id] = {
                    "p_entailment": float(p_ent),
                    "p_contradiction": float(p_con),
                    "p_neutral": float(p_neu),
                }

        response = {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": total_tokens},
        }
        return response, raw
