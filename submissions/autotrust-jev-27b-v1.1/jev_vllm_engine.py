"""Decision Index engine for AutoTrust-JEV-27B-v1.1 served by vLLM (StartLux prefix-bidirectional plugin).

Engine: ``jev_vllm_engine:JevVllmEngine`` with ``--option model=<vllm folder> --option url=http://127.0.0.1:8090/v1``.
The vLLM folder has the decision readout folded into ``lm_head`` rows of the 255 code tokens and
``jev_prompt_attention: bidirectional``; the server must run with the plugin on (STARTLUX_PREFIX_BIDIR=1) and
``--logprobs-mode processed_logprobs --max-logprobs 255 --no-enable-prefix-caching --no-enable-chunked-prefill``.

Per request: every text question is rendered with the model's prompt (jev_format.render, the model's codes), sent as one
completion with ``max_tokens=1``, ``allowed_token_ids`` = the first n code tokens and ``logprobs=n``; the processed
logprobs over the allowed set are the readout softmax (temperature from vllm_readout.json). Questions of one request are sent
concurrently. Image/video requests are declared unsupported (text-only engine, as our training).
"""

from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from decision_index.engines.base import Engine, Unsupported

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import jev_format  # noqa: E402  (prompt renderer shipped with the model)


class JevVllmEngine(Engine):
    name = "autotrust-jev-27b-v1.1-vllm"
    latency = (
        "Wall time of one request against a local vLLM server (HTTP on localhost): prompt rendering, one completion "
        "per question (max_tokens 1, processed logprobs over the option codes), questions of a request in parallel; "
        "excludes server start-up."
    )

    def __init__(
        self,
        model: str | None = None,
        url: str = "http://127.0.0.1:8090/v1",
        workers: str = "8",
        timeout: str = "600",
        **options,
    ):
        if options:
            raise TypeError(f"unknown engine options {sorted(options)}")
        if not model:
            raise ValueError("pass --option model=<vllm folder>")
        super().__init__(model=model, url=url, workers=workers)
        self.model_dir = model
        self.url = url.rstrip("/")
        self.workers = int(workers)
        self.timeout = float(timeout)
        rd = json.load(open(os.path.join(model, "vllm_readout.json")))
        self.token_ids = rd["readout_token_ids"]
        self.codes = rd["codes"]
        self.temperature = float(rd.get("temperature", 1.0))
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.session = requests.Session()
        self.pool = ThreadPoolExecutor(self.workers)
        info = self.session.get(self.url + "/models", timeout=30).json()
        self.served = info["data"][0]["id"]
        self.provenance = {
            "vllm_folder": model,
            "served_model": self.served,
            "readout": rd.get("note"),
        }

    def runtime(self):
        try:
            import vllm  # noqa: F401

            v = vllm.__version__
        except Exception:
            v = "server"
        return {
            "backend": "vllm",
            "vllm": v,
            "url": self.url,
            "prompt_attention": "bidirectional (startlux_prefix_bidir)",
            "readout": "lm_head rows = decision readout (bf16)",
            "temperature": self.temperature,
        }

    def _readout(self, prompt: str, n: int) -> list[float]:
        body = {
            "model": self.served,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": n,
            "allowed_token_ids": self.token_ids[:n],
            "return_tokens_as_token_ids": True,
        }
        r = self.session.post(
            self.url + "/completions", json=body, timeout=self.timeout
        )
        r.raise_for_status()
        top = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
        z = [top.get(f"token_id:{t}", -1e9) for t in self.token_ids[:n]]
        m = max(z)
        e = [math.exp((x - m) / self.temperature) for x in z]
        s = sum(e)
        return [x / s for x in e]

    def __call__(self, state, questions, images=None, videos=None):
        if images or videos:
            raise Unsupported(
                "image and video inputs are not supported by this text engine"
            )
        keys = list(questions)
        specs = []
        for k in keys:
            q = questions[k]
            opts = jev_format.options(q)[0]
            if len(opts) > len(self.token_ids):
                raise Unsupported(
                    f"{k}: {len(opts)} options exceed the {len(self.token_ids)} answer codes"
                )
            specs.append(
                (k, jev_format.render(self.tokenizer, state, q, self.codes), len(opts))
            )
        probs = list(self.pool.map(lambda s: self._readout(s[1], s[2]), specs))
        answers = {
            k: jev_format.to_answer(questions[k], p)
            for (k, _, _), p in zip(specs, probs)
        }
        return {
            "model": self.served,
            "answers": answers,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }, None
