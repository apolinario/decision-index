"""Decision Index vision engine for AutoTrust-JEV-27B-v1.1: in-process vLLM, the model's own prompt and code read-out.

The model repository's engine (jev_vllm_engine.py) is text only. This engine reads images with the same weights: the
vision tower is d3's, unchanged (all 333 visual tensors are byte-identical to vllm-sr/d3), so the images go in d3's
image-prompt layout, one image placeholder per image before the user text (d3_runtime.image_messages), with every prompt
string from the repository's jev_format.py:

    system: jev_format.SYSTEM_PROMPT
    user:   [image, ...] + jev_format.user_prompt(state, question, codes)     (state "{}" -> "(empty)" as JSON "{}")
    assistant turn opened with the empty think block (enable_thinking=False), where the read-out happens.

One decoding step restricted to the question's first n answer-code tokens (vllm_readout.json), processed log-probabilities,
softmax at the read-out temperature (1.0). The row's `prompt.readout` prefix ("The answer is (", R-Bench-M and MMMU-Pro
vision) is not used: the model reads its answer at its own read-out position, as for every other row.

vLLM is configured like the repository's serve_vllm.sh: StartLux prompt-bidirectional plugin on (STARTLUX_PREFIX_BIDIR=1),
bf16, no prefix caching, no chunked prefill, max_num_batched_tokens = max_model_len = 65536, FLASH_ATTN, processed
log-probabilities (up to 255); max_num_seqs 8 so that every prompt, images included, is prefilled whole in one step. Images: the kit's cap (native resolution up to 1,638,400 pixels; larger images downscaled by
the kit), then the model's own processor, whose limits (longest_edge 16,777,216 pixels) are above the cap.

    decision-index vision pipeline --dir work/vision --engine jev_vllm_vision_engine:JevVllmVision \
        --option model=AutoTrust-JEV-27B-v1.1 --out runs/autotrust-jev-27b-v1.1-vision
"""

from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("STARTLUX_PREFIX_BIDIR", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from decision_index.engines.base import Unsupported  # noqa: E402
from decision_index.vision import images as img  # noqa: E402
from decision_index.vision.engines import CAPACITY_MARKERS, VllmVisionChat, answer, pixels, single_choice, softmax  # noqa: E402


class JevVllmVision(VllmVisionChat):
    name = "autotrust-jev-27b-v1.1-vision"
    latency = "Batched llm.chat wall time per chunk divided by the rows in the chunk; excludes model loading."

    def __init__(self, model, max_model_len=65536, max_images=8, gpu_memory_utilization=0.85, **options):
        sys.path.insert(0, model)
        import jev_format

        self.fmt = jev_format
        rd = json.load(open(os.path.join(model, "vllm_readout.json")))
        llm_options = {
            "dtype": "bfloat16",
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "max_num_batched_tokens": int(max_model_len),
            "max_logprobs": 255,
            "attention_backend": "FLASH_ATTN",
            "generation_config": "vllm",
            # The plugin needs every prompt prefilled whole in one step. vLLM sizes the encoder cache to
            # max_num_batched_tokens and, when an image does not fit in it, schedules only the text before the image;
            # 8 sequences x 4 images x <= 1,617 image tokens (1.6 MP) always fit in 65,536.
            "max_num_seqs": 8,
        }
        super().__init__(model=model, max_model_len=int(max_model_len), max_images=int(max_images),
                         gpu_memory_utilization=float(gpu_memory_utilization), llm_options=llm_options, **options)
        self.codes = rd["codes"]
        self.letter_ids = rd["readout_token_ids"]
        self.temperature = float(rd.get("temperature", 1.0))
        self.provenance.update({
            "kind": "vllm-chat, model prompt and code read-out",
            "readout": rd.get("note"),
            "temperature": self.temperature,
            "prompt_attention": "bidirectional (startlux_prefix_bidir)",
            "llm_options": llm_options,
            "policy": "system = jev_format.SYSTEM_PROMPT; user = one image placeholder per image, then "
                      "jev_format.user_prompt(state, question, codes); assistant turn opened with the empty think block; "
                      "one decoding step restricted to the question's first n answer-code tokens; prompt.readout not used",
        })

    def conversation(self, payload):
        key, q, keys = single_choice(payload["questions"])
        content = [{"type": "image_pil", "image_pil": img.pil(p, self.image_root, self.max_pixels).convert("RGB")}
                   for p in payload.get("images") or []]
        if len(keys) > len(self.letter_ids):
            raise Unsupported(f"{len(keys)} options exceed the {len(self.letter_ids)} answer codes")
        content.append({"type": "text", "text": self.fmt.user_prompt(payload.get("state"), q, self.codes)})
        msgs = [{"role": "system", "content": self.fmt.SYSTEM_PROMPT}, {"role": "user", "content": content}]
        return msgs, key, keys, None

    def read(self, o, key, keys):
        k = len(keys)
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        vals = [float(lp[t].logprob) if t in lp else -math.inf for t in self.letter_ids[:k]]
        probs = softmax([v / self.temperature for v in vals])
        choice = keys[max(range(k), key=lambda i: probs[i])]
        raw = {"code_logprobs": vals}
        if self.image_token_id is not None:
            raw["image_tokens"] = sum(t == self.image_token_id for t in o.prompt_token_ids)
        return answer(key, keys, choice, probs, self.model, {"input_tokens": len(o.prompt_token_ids)}), raw

    def batch(self, payloads):
        from vllm import SamplingParams

        out = [None] * len(payloads)
        members = []
        for i, p in enumerate(payloads):
            try:
                members.append((i, *self.conversation(p)))
            except Exception as e:
                out[i] = e
        if not members:
            return out
        convs = [m[1] for m in members]
        sps = [SamplingParams(max_tokens=1, temperature=1.0, logprobs=len(m[3]), allowed_token_ids=self.letter_ids[:len(m[3])])
               for m in members]
        kw = dict(add_generation_prompt=True, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        try:
            outs = self.llm.chat(convs, sps, **kw)
        except Exception:
            outs = []
            for cv, sp in zip(convs, sps):
                try:
                    outs.append(self.llm.chat([cv], [sp], **kw)[0])
                except Exception as e:
                    outs.append(e)
        for (i, _, key, keys, _), o in zip(members, outs):
            if isinstance(o, Exception):
                out[i] = Unsupported(str(o)) if any(m in str(o) for m in CAPACITY_MARKERS) else o
            else:
                out[i] = self.read(o, key, keys)
                out[i][1]["image_pixels"] = [pixels(p, self.image_root, self.max_pixels) for p in payloads[i].get("images") or []]
        return out
