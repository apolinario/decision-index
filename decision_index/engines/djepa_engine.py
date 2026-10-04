"""
DM-JEPA Decision Engine Adapter for Decision Index.
Wraps the trained DM-JEPA checkpoint into the standard Decision Index Engine protocol.
Published by Danger Labs: https://huggingface.co/DangerLabs/DM-JEPA
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from decision_index.engines.base import Engine, Unsupported, validate
from djepa.model.djepa import DJEPA
from djepa.dataset.formatter import JevFormatter


class DJEPAEngine(Engine):
    name = "dm-jepa"
    latency = "Device-synchronized in-process System 1 forward pass latency."

    def __init__(
        self,
        checkpoint: str = "DangerLabs/DM-JEPA",
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        device: Optional[str] = None,
        max_state_length: int = 2048,
        max_option_length: int = 256,
        **options,
    ):
        super().__init__(**options)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_state_length = max_state_length
        self.max_option_length = max_option_length
        self.checkpoint_spec = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = DJEPA().to(self.device)

        local_path = None
        if Path(checkpoint).exists():
            local_path = checkpoint
        elif Path("/home/jerrick/decision-model-workspace/checkpoints/djepa_production_best.pt").exists():
            local_path = "/home/jerrick/decision-model-workspace/checkpoints/djepa_production_best.pt"
        else:
            from huggingface_hub import hf_hub_download
            try:
                local_path = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
            except Exception:
                local_path = hf_hub_download(repo_id=checkpoint, filename="pytorch_model.bin")

        if local_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(local_path)
        else:
            ckpt = torch.load(local_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[DM-JEPA Engine] Successfully loaded model weights from {local_path}")

        self.provenance = {
            "kind": "dm-jepa",
            "model_type": "Joint-Embedding Predictive Architecture (System 1)",
            "organization": "Danger Labs",
            "model_id": "DangerLabs/DM-JEPA",
            "state_encoder": tokenizer_name,
            "latent_predictor": "6-layer bidirectional cross-attention rollout",
            "criteria_encoder": "ModernBERT criteria projection",
            "policy": "Deterministic non-autoregressive latent compatibility scoring in thought space.",
            "declared_limits": {
                "max_state_length": self.max_state_length,
                "max_option_length": self.max_option_length,
            },
        }

    def runtime(self) -> Dict[str, Any]:
        info = {
            "model": "DM-JEPA (Danger Labs)",
            "torch": torch.__version__,
            "device": str(self.device),
            "declared_limits": {
                "max_state_length": self.max_state_length,
                "max_option_length": self.max_option_length,
            },
        }
        if self.device.type == "cuda":
            info.update(
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),
                allocated_mb=round(torch.cuda.memory_allocated() / (1024 * 1024), 1),
            )
        return info

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def warmup(self):
        warm_question = {
            "warmup": {
                "type": "choice",
                "instructions": "Which color is named?",
                "criteria": {"red": "red", "blue": "blue"},
            }
        }
        self("The color is red.", warm_question)
        self.synchronize()

    @torch.no_grad()
    def _score_single_question(
        self,
        state: Any,
        instructions: str,
        labels: list,
        criteria_dict: dict,
    ) -> list:
        state_prompt = JevFormatter.format_state_prompt(state, instructions)
        option_prompts = [
            JevFormatter.format_option_prompt(lbl, criteria_dict.get(lbl))
            for lbl in labels
        ]

        # Check declared capacity limits without truncation (Decision Index Rule: No Truncation)
        state_enc = self.tokenizer(
            state_prompt,
            truncation=False,
            return_tensors="pt",
        )
        if state_enc["input_ids"].size(1) > self.max_state_length:
            raise Unsupported(
                f"State token length {state_enc['input_ids'].size(1)} exceeds declared capacity {self.max_state_length}"
            )
        state_enc = {k: v.to(self.device) for k, v in state_enc.items()}

        opt_encs = []
        for opt in option_prompts:
            o_enc = self.tokenizer(opt, truncation=False, return_tensors="pt")
            if o_enc["input_ids"].size(1) > self.max_option_length:
                raise Unsupported(
                    f"Option token length {o_enc['input_ids'].size(1)} exceeds declared capacity {self.max_option_length}"
                )
            opt_encs.append(o_enc)

        K = len(opt_encs)
        max_opt_len = max(o["input_ids"].size(1) for o in opt_encs)
        pad_id = self.tokenizer.pad_token_id or 0

        opt_ids = torch.full((1, K, max_opt_len), pad_id, dtype=torch.long, device=self.device)
        opt_mask = torch.zeros((1, K, max_opt_len), dtype=torch.long, device=self.device)

        for k, o in enumerate(opt_encs):
            l = o["input_ids"].size(1)
            opt_ids[0, k, :l] = o["input_ids"][0].to(self.device)
            opt_mask[0, k, :l] = o["attention_mask"][0].to(self.device)

        amp_context = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.no_grad()
        )

        with amp_context:
            out = self.model.forward(
                state_input_ids=state_enc["input_ids"],
                state_attention_mask=state_enc["attention_mask"],
                option_input_ids=opt_ids,
                option_attention_mask=opt_mask,
            )

        probs = out["probabilities"][0].float().cpu().tolist()
        return probs

    def __call__(self, state: Any, questions: dict) -> tuple:
        answers = {}
        for q_id, q_data in questions.items():
            q_type = q_data.get("type", "choice")
            instructions = q_data.get("instructions", "")

            if q_type == "choice":
                criteria_dict = q_data.get("criteria", {})
                labels = list(criteria_dict.keys())
                probs = self._score_single_question(state, instructions, labels, criteria_dict)

                total_p = sum(probs)
                if total_p > 0:
                    probs = [p / total_p for p in probs]
                else:
                    probs = [1.0 / len(labels)] * len(labels)

                prob_map = {lbl: float(p) for lbl, p in zip(labels, probs)}
                best_choice = labels[max(range(len(labels)), key=lambda i: probs[i])]

                answers[q_id] = {
                    "type": "choice",
                    "choice": best_choice,
                    "probabilities": prob_map,
                }
            elif q_type == "noul":
                noul_question = {
                    "instructions": instructions,
                    "criteria": {"yes": "Yes, condition met", "no": "No, condition not met"},
                }
                probs = self._score_single_question(state, instructions, ["yes", "no"], noul_question["criteria"])
                p_yes = float(probs[0]) / (float(probs[0]) + float(probs[1]) + 1e-9)
                answers[q_id] = {
                    "type": "noul",
                    "noul": max(0.0, min(1.0, p_yes)),
                }
            else:
                raise Unsupported(f"Unsupported question type: {q_type}")

        response = {
            "model": "DM-JEPA",
            "answers": answers,
        }
        validate(questions, response)
        return response, None
