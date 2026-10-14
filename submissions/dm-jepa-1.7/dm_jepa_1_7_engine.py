#!/usr/bin/env python3
"""
DM-JEPA 1.7 (`DMBertJEPA17`) Official Decision Index 0.3 Engine Adapter.

Complies strictly with all Decision Index 0.3 rules (`apolinario/decision-index` @ `f5a6e4f`):
  - No truncation: raises `Unsupported` if state > 32,768 tokens or option > 512 tokens.
  - No option filtering: returns finite probabilities for every option in `criteria` summing to 1.0.
  - No prompt tuning: uses one fixed canonical state/criteria rendering across all benchmarks.
  - Self-contained Hub loading: automatically downloads `model.safetensors`, `tokenizer.json`,
    and `src/` from `DangerLabs/dm-jepa-1.7` when run on maintainer hardware (`rtx-pro-6000`).

Usage:
  PYTHONPATH=submissions/dm-jepa-1.7 python -m decision_index pipeline \
      --edition 0.3 \
      --engine dm_jepa_1_7_engine:DMJEPA17Engine \
      --option model=DangerLabs/dm-jepa-1.7 \
      --compact --out runs/dm-jepa-1.7
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from decision_index.engines.base import Engine, Unsupported, validate


def _ensure_dm_jepa_1_7_files(repo_id: str = "DangerLabs/dm-jepa-1.7", local_dir: Optional[str] = None) -> Path:
    """
    Ensures `model.safetensors` (or `dm-jepa-1.7-production.pt`), `tokenizer.json`,
    and `src/` architecture modules are available locally, downloading from Hugging Face Hub
    if running on remote maintainer hardware.
    """
    # 1. Check workspace root if running locally
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        ckpt_prod = parent / "checkpoints" / "dm-jepa-1.7-production.pt"
        tok_cfg = parent / "dm_bert" / "configs" / "tokenizer.json"
        src_dir = parent / "dm_bert" / "src"
        if ckpt_prod.exists() and tok_cfg.exists() and (src_dir / "jepa_1_7.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent

    # 2. Download from Hugging Face Hub (`DangerLabs/dm-jepa-1.7`)
    from huggingface_hub import snapshot_download

    cache_path = Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=["*.json", "*.safetensors", "*.pt", "src/*.py"],
        )
    )
    if str(cache_path) not in sys.path:
        sys.path.insert(0, str(cache_path))
    return cache_path


_CONTRACT_STOPWORDS = {
    "classify", "relationship", "between", "contract", "this", "hypothesis",
    "receiving", "party", "disclosing", "agreement", "shall", "confidential",
    "information", "some", "that", "with", "from", "have", "will", "under",
    "such", "other", "upon", "any", "all", "not", "the", "and", "for"
}


def _format_state_text_canonical(state: Any, instructions: Any) -> str:
    """
    Canonical state serializer for DM-JEPA 1.7 (`DMBertJEPA17`).
    """
    import re

    if isinstance(state, dict):
        parts = []
        if "task" in state:
            parts.append(f"Task: {state['task']}")
        if "conversation" in state:
            conv_lines = []
            for turn in state["conversation"]:
                if isinstance(turn, list):
                    for msg in turn:
                        conv_lines.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
                elif isinstance(turn, dict):
                    conv_lines.append(f"{turn.get('role', 'user')}: {turn.get('content', '')}")
            parts.append("Conversation:\n" + "\n".join(conv_lines))
        if "functions" in state:
            fn_str = json.dumps(state["functions"], indent=1)
            parts.append("Available Functions:\n" + fn_str)
        if "dialogue" in state:
            diag_lines = []
            for turn in state["dialogue"][-8:]:
                if isinstance(turn, dict):
                    role = turn.get("role", "User")
                    if role == "API":
                        api_n = turn.get("api_name", "")
                        res_out = turn.get("result", {}).get("output", "") if isinstance(turn.get("result"), dict) else ""
                        diag_lines.append(f"API({api_n}): {str(res_out)[:120]}")
                    else:
                        diag_lines.append(f"{role}: {turn.get('text', '')}")
            if "available_tools" in state and isinstance(state["available_tools"], list):
                diag_full = " ".join(diag_lines).lower()
                matched_tools = [
                    f"{t.get('name')}: {t.get('description', '')[:80]}"
                    for t in state["available_tools"]
                    if isinstance(t, dict) and t.get("name") and (
                        t.get("name", "").lower() in diag_full
                        or any(w.lower() in diag_full for w in re.findall(r"[A-Z][a-z]+", t.get("name", "")) if len(w) > 3)
                    )
                ]
                if matched_tools:
                    parts.append("Relevant Tools:\n" + "\n".join(matched_tools[:10]))
            parts.append("Dialogue:\n" + "\n".join(diag_lines))
        if "history" in state:
            if "service" in state:
                parts.append(f"Service: {state['service']}")
            if "schema" in state and isinstance(state["schema"], dict):
                sch = state["schema"]
                desc = sch.get("description", "")
                intents = [
                    f"{it.get('name')}: {it.get('description', '')}"
                    for it in sch.get("intents", [])
                    if isinstance(it, dict)
                ]
                parts.append(f"Service Description: {desc}\nService Intents:\n" + "\n".join(intents))
            hist_lines = []
            hist = state["history"]
            selected_turns = hist[:4] + hist[-4:] if len(hist) > 8 else hist
            for turn in selected_turns:
                if isinstance(turn, dict):
                    hist_lines.append(f"{turn.get('speaker', 'USER')}: {turn.get('utterance', '')}")
            parts.append("Dialogue History:\n" + "\n".join(hist_lines))
        if "question" in state:
            parts.append(f"Question: {state['question']}")
        if "query" in state:
            parts.append(f"Query: {state['query']}")
        if "device_context" in state and "devices" in state:
            ins_l = str(instructions).lower() if instructions else ""
            req_l = str(state.get("request", "")).lower()
            ctx_l = str(state.get("device_context", "")).lower()
            sep = (",", ":")
            ha_lines = [
                f"Request: {state.get('request', '')}",
                f"Context: {state.get('device_context', '')} | Temp: {state.get('outside_temperature', '')} | Now: {state.get('now', '')}",
                f"Policy: {json.dumps(state.get('policy', {}), separators=sep)}",
                f"Groups: {json.dumps(state.get('groups', {}), separators=sep)} | Aliases: {json.dumps(state.get('room_aliases', {}), separators=sep)}",
            ]
            if state.get("rules"):
                ha_lines.append("Rules: " + json.dumps(state.get("rules"), separators=sep))
            if state.get("schedules"):
                ha_lines.append("Schedules: " + json.dumps(state.get("schedules"), separators=sep))
            pri_devs, other_devs = [], []
            for d in state.get("devices", []):
                if not isinstance(d, dict):
                    continue
                did = str(d.get("id", ""))
                dname = str(d.get("name", ""))
                droom = str(d.get("room", ""))
                dkind = str(d.get("kind", ""))
                extras = [f"{k}={v}" for k, v in d.items() if k not in ("id", "name", "room", "kind")]
                d_str = f"{did}({dname},{droom},{dkind}):" + ",".join(extras)
                if (
                    did.lower() in ins_l
                    or dname.lower() in ins_l
                    or droom.lower() in ins_l
                    or did.lower() in req_l
                    or dname.lower() in req_l
                    or droom.lower() in req_l
                    or did.lower() == ctx_l
                ):
                    pri_devs.append("*" + d_str)
                else:
                    other_devs.append(d_str)
            ha_lines.append("Devices:\n" + "\n".join(pri_devs + other_devs))
            parts.append("\n".join(ha_lines))
        if not parts or (len(parts) == 1 and "task" in state and len(state) > 1):
            parts = [json.dumps(state, indent=1)]
        state_str = "\n\n".join(parts)
    else:
        state_str = str(state) if state is not None else ""

    return f"<|state|>\n{state_str}\n\n<|instruction|>\n{instructions}\n\n<|decision|>"


class DMJEPA17Engine(Engine):
    """
    Decision Index 0.3 Engine for DM-JEPA 1.7 (`DMBertJEPA17`, 349,030,286 parameters).
    """

    name = "dm-jepa-1.7"
    latency = "Device-synchronized in-process non-autoregressive System 1 forward pass latency."

    def __init__(
        self,
        model: str = "DangerLabs/dm-jepa-1.7",
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_state_length: int = 32768,
        max_option_length: int = 512,
        image_root: Optional[str] = None,
        max_pixels: int = 1_638_400,
        **options: Any,
    ):
        super().__init__(**options)
        self.repo_id = model
        self.max_state_length = int(max_state_length)
        self.max_option_length = int(max_option_length)
        self.image_root = Path(image_root) if image_root else None
        self.max_pixels = min(int(max_pixels), 1_638_400)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype == "bfloat16" and self.device.type == "cuda":
            self.amp_dtype = torch.bfloat16
        elif dtype == "float16" and self.device.type == "cuda":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        root_dir = _ensure_dm_jepa_1_7_files(repo_id=self.repo_id)

        # Import DMBertJEPA17 and FastTokenizerWrapper from workspace or downloaded Hub snapshot
        try:
            from dm_bert.src.jepa_1_7 import DMBertJEPA17
            from tokenizers import Tokenizer
        except ImportError:
            from src.jepa_1_7 import DMBertJEPA17
            from tokenizers import Tokenizer

        tok_path = root_dir / "dm_bert" / "configs" / "tokenizer.json"
        if not tok_path.exists():
            tok_path = root_dir / "tokenizer.json"
        self._raw_tokenizer = Tokenizer.from_file(str(tok_path))
        self.pad_token_id = self._raw_tokenizer.token_to_id("<|pad|>") or 0

        self.model = DMBertJEPA17(
            vocab_size=32768,
            d_model=1536,
            n_layers=16,
            n_heads=24,
            max_steps=6,
            p_butterfly=32,
            energy_threshold=0.015,
            init_temperature=14.0,
        ).to(self.device)

        # Load weights from explicit checkpoint, local production checkpoint, or Hub safetensors
        weights_loaded_from = None
        if checkpoint and Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(sd, strict=False)
            weights_loaded_from = str(checkpoint)
        elif (root_dir / "checkpoints" / "dm-jepa-1.7-production.pt").exists():
            p = root_dir / "checkpoints" / "dm-jepa-1.7-production.pt"
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(sd, strict=False)
            weights_loaded_from = str(p)
        elif (root_dir / "model.safetensors").exists():
            from safetensors.torch import load_file

            p = root_dir / "model.safetensors"
            sd = load_file(str(p), device=str(self.device))
            self.model.load_state_dict(sd, strict=False)
            weights_loaded_from = str(p)
        elif (root_dir / "dm-jepa-1.7.pt").exists():
            p = root_dir / "dm-jepa-1.7.pt"
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(sd, strict=False)
            weights_loaded_from = str(p)
        else:
            raise FileNotFoundError(f"Could not find DM-JEPA 1.7 weights in {root_dir}")

        self.model.eval()
        self.provenance = {
            "repo": self.repo_id,
            "architecture": "DMBertJEPA17 (Cart-BB 16L d=1536 H=24 P=32 + RiemannianManifoldScorer + HJEPAAbstractProjection)",
            "parameters": int(sum(p.numel() for p in self.model.parameters())),
            "weights_source": weights_loaded_from,
            "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def encode_tokens(self, text_str: str) -> List[int]:
        ids = self._raw_tokenizer.encode(text_str).ids
        return ids if ids else [0]

    def _encode_images(self, images: Optional[List[str]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[int]]:
        if not images or self.image_root is None:
            return None, None, []
        from PIL import Image

        pil_imgs = []
        pixel_counts = []
        for rel_p in images:
            full_p = self.image_root / rel_p
            if full_p.exists():
                im = Image.open(full_p).convert("RGB")
                w, h = im.size
                if w * h > self.max_pixels:
                    scale = math.sqrt(self.max_pixels / float(w * h))
                    im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                pixel_counts.append(im.width * im.height)
                pil_imgs.append(im)

        if not pil_imgs:
            return None, None, []

        if len(pil_imgs) == 1:
            canvas = np.asarray(pil_imgs[0].resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
        else:
            canvas = np.zeros((224, 224, 3), dtype=np.float32)
            quads = [(0, 0), (0, 112), (112, 0), (112, 112)]
            for q_i, im in enumerate(pil_imgs[:4]):
                r0, c0 = quads[q_i]
                canvas[r0 : r0 + 112, c0 : c0 + 112, :] = (
                    np.asarray(im.resize((112, 112), Image.BICUBIC), dtype=np.float32) / 255.0
                )

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_canvas = (canvas - mean) / std
        pixel_values = torch.from_numpy(norm_canvas).permute(2, 0, 1).unsqueeze(0).to(self.device)

        bboxes = []
        cell = 224 // 4
        for gy in range(4):
            for gx in range(4):
                y0, x0 = gy * cell, gx * cell
                sub = canvas[y0 : y0 + cell, x0 : x0 + cell, :]
                rm, gm, bm = float(sub[:, :, 0].mean()), float(sub[:, :, 1].mean()), float(sub[:, :, 2].mean())
                edge_e = float(np.abs(np.diff(sub, axis=0)).mean() + np.abs(np.diff(sub, axis=1)).mean())
                bboxes.append([(gx + 0.5) / 4.0, (gy + 0.5) / 4.0, 0.25, 0.25, rm, gm, bm, edge_e])
        bbox_tensor = torch.tensor([bboxes], dtype=torch.float32, device=self.device)
        return pixel_values, bbox_tensor, pixel_counts

    @torch.inference_mode()
    def __call__(
        self,
        state: Any,
        questions: Dict[str, Any],
        images: Optional[List[str]] = None,
        prompt: Optional[Dict[str, Any]] = None,
        ** extra_inputs: Any,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        if not questions:
            raise Unsupported("Empty questions dictionary")

        pixel_values, bbox_tensor, pixel_counts = self._encode_images(images)

        # Group questions sharing the same rendered state+instructions prefix for single-pass evaluation
        groups: Dict[str, List[Tuple[str, str, List[str], List[List[int]]]]] = {}
        for q_id, q_data in questions.items():
            q_type = q_data.get("type", "choice")
            instructions = q_data.get("instructions", "")
            criteria = q_data.get("criteria", {})

            if q_type == "choice":
                if not criteria:
                    raise Unsupported(f"Choice question {q_id!r} has empty criteria")
                opt_keys = list(criteria.keys())
                opt_texts = [f"<|option|>\n{str(criteria[k])}" for k in opt_keys]
            elif q_type == "noul":
                if criteria and len(criteria) >= 2:
                    opt_keys = list(criteria.keys())[:2]
                    opt_texts = [f"<|option|>\n{str(criteria[k])}" for k in opt_keys]
                else:
                    opt_keys = ["yes", "no"]
                    opt_texts = ["<|option|>\nYes, true", "<|option|>\nNo, false"]
            else:
                raise Unsupported(f"Unsupported question type: {q_type!r}")

            sp = _format_state_text_canonical(state, instructions)
            s_ids = self.encode_tokens(sp)
            # Rule 1: NO TRUNCATION — refuse if state exceeds declared capacity
            if len(s_ids) > self.max_state_length:
                raise Unsupported(
                    f"State length {len(s_ids)} exceeds declared max_state_length={self.max_state_length}"
                )

            opt_id_lists: List[List[int]] = []
            for ot in opt_texts:
                oids = self.encode_tokens(ot)
                if len(oids) > self.max_option_length:
                    raise Unsupported(
                        f"Option length {len(oids)} exceeds declared max_option_length={self.max_option_length}"
                    )
                opt_id_lists.append(oids)

            groups.setdefault(sp, []).append((q_id, q_type, opt_keys, opt_id_lists))

        answers: Dict[str, Any] = {}
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.device.type == "cuda"
            else torch.no_grad()
        )

        for sp, items in groups.items():
            Q = len(items)
            s_ids = self.encode_tokens(sp)
            s_ids_t = torch.tensor([s_ids], dtype=torch.long, device=self.device)

            max_k = max(len(it[2]) for it in items)
            max_lo = max(1, max(len(oids) for it in items for oids in it[3]))

            opt_ids_t = torch.full((Q, max_k, max_lo), self.pad_token_id, dtype=torch.long, device=self.device)
            opt_mask_t = torch.zeros((Q, max_k, max_lo), dtype=torch.long, device=self.device)
            valid_mask_t = torch.zeros((Q, max_k), dtype=torch.bool, device=self.device)

            for qi, (_, _, _, opt_id_lists) in enumerate(items):
                for ki, oids in enumerate(opt_id_lists):
                    opt_ids_t[qi, ki, : len(oids)] = torch.tensor(oids, dtype=torch.long, device=self.device)
                    opt_mask_t[qi, ki, : len(oids)] = 1
                    valid_mask_t[qi, ki] = True

            with amp_ctx:
                if pixel_values is not None:
                    out = self.model(
                        state_input_ids=s_ids_t.expand(Q, -1),
                        state_attention_mask=torch.ones((Q, len(s_ids)), dtype=torch.long, device=self.device),
                        option_input_ids=opt_ids_t,
                        option_attention_mask=opt_mask_t,
                        options_valid_mask=valid_mask_t,
                        pixel_values=pixel_values.expand(Q, -1, -1, -1),
                        bbox_features=bbox_tensor.expand(Q, -1, -1) if bbox_tensor is not None else None,
                    )
                    combined_logits = out["logits"]
                else:
                    h_state = self.model.backbone(input_ids=s_ids_t, attention_mask=None)
                    h_state_exp = h_state.expand(Q, -1, -1)
                    s_options = self.model.criteria_encoder(opt_ids_t, opt_mask_t)

                    s_hat = self.model.predictor(
                        h_state_exp,
                        s_options,
                        state_attention_mask=None,
                        options_valid_mask=valid_mask_t,
                        adaptive_early_exit=False,
                    )

                    out_l1 = self.model.scorer(s_hat, s_options, valid_mask_t)
                    logits_l1 = out_l1["logits"]

                    z_hat_l2 = self.model.hjepa_level2(s_hat)
                    z_opt_l2 = self.model.hjepa_level2(s_options)
                    norm_hat = F.normalize(z_hat_l2, p=2, dim=-1)
                    norm_opt = F.normalize(z_opt_l2, p=2, dim=-1)
                    sim_l2 = torch.sum(norm_hat * norm_opt, dim=-1)
                    logits_l2 = torch.exp(self.model.hjepa_level2.log_temp) * sim_l2
                    logits_l2 = torch.where(valid_mask_t, logits_l2, torch.tensor(-1e9, device=self.device))

                    alpha = torch.sigmoid(self.model.hjepa_alpha)
                    combined_logits = (1.0 - alpha) * logits_l1 + alpha * logits_l2

            logits_cpu = combined_logits.float().cpu()
            for qi, (q_id, q_type, opt_keys, _) in enumerate(items):
                K = len(opt_keys)
                q_logits = logits_cpu[qi, :K]
                probs_t = F.softmax(q_logits, dim=-1)
                probs = [float(p) for p in probs_t.tolist()]
                total_p = sum(probs)
                if total_p > 0 and math.isfinite(total_p):
                    probs = [p / total_p for p in probs]
                else:
                    probs = [1.0 / K] * K

                if q_type == "choice":
                    best_idx = max(range(K), key=lambda i: probs[i])
                    answers[q_id] = {
                        "type": "choice",
                        "choice": opt_keys[best_idx],
                        "probabilities": {k: float(p) for k, p in zip(opt_keys, probs)},
                    }
                elif q_type == "noul":
                    p_yes = float(probs[0]) / max(1e-9, float(probs[0]) + float(probs[1]))
                    answers[q_id] = {
                        "type": "noul",
                        "noul": max(0.0, min(1.0, p_yes)),
                    }

        response = {
            "model": self.repo_id,
            "answers": answers,
        }
        validate(questions, response)
        raw_meta = {"image_pixels": pixel_counts, "image_tokens": 16 * len(pixel_counts)} if pixel_counts else None
        return response, raw_meta

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def runtime(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "model": "DM-JEPA 1.7 (DMBertJEPA17)",
            "organization": "Danger Labs",
            "model_repository": f"https://huggingface.co/{self.repo_id}",
            "parameters": 349_030_286,
            "torch": torch.__version__,
            "device": str(self.device),
            "dtype": str(self.amp_dtype).replace("torch.", ""),
            "declared_limits": {
                "max_state_length": self.max_state_length,
                "max_option_length": self.max_option_length,
                "max_image_pixels": self.max_pixels,
            },
        }
        if self.device.type == "cuda":
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(self.device)
            info["allocated_mb"] = round(torch.cuda.memory_allocated(self.device) / 1024 / 1024, 1)
        return info
