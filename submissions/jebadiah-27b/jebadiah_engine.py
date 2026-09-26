"""Jebadiah 27B (merged bf16 chat checkpoint on Qwen3.8-27B, thinking off) as a Decision Index engine.

The model ships merged weights (the LoRA folded into Qwen/Qwen3.8-27B), so it loads directly with no
base+LoRA step. It is read with its own published inference code: the `scripts/` directory of
frontier-infra/jebadiah-27b at a pinned revision (jebadiah_model.py, jebadiah_prompt.py,
ainode_prompt_verbatim.py), imported from the Hub snapshot as-is. The engine refuses to load if the
chat template, the prompt source or the chat template kwargs differ from the model's
prompt_contract.json. Nothing in the prompt, the label alphabet, the logit read or the temperatures
is changed here. What this file adds:

- mechanical schema translation (recorded per question in raw_output): an `instructions` or
  option description that is not a string is flattened with the model's own `flatten_text`
  (the rule its training conversion used), because AINode's wire requires strings;
- capacity refusals instead of truncation: the published Renderer cuts the state when the
  prompt is over its token budget, so the budget is set to the declared context and any
  rendered prompt that came back truncated is refused as `Unsupported`, never scored;
  a question with more options than the single-token label alphabet holds is refused too;
- on Apple MPS only, a faster exact triangular inverse inside transformers' reference Gated
  DeltaNet prefill (mps_delta.py; the CUDA path is untouched and uses flash-linear-attention);
- batching by padded token budget rather than by a fixed 8 questions, so a long prompt runs
  alone (probabilities do not depend on the batch beyond bf16 rounding);
- optional, off by default: a shared-prefix cache (prefix_cache=true) that runs a request's common
  system + state token prefix once and each question's suffix from its cached keys, values and
  Gated DeltaNet state; a padded-token batch budget honoured on every device (batch_tokens); a
  one-row-per-forward retry of a request that runs out of device memory; a per-process CUDA memory
  cap (mem_cap_gb) for a shared box;
- the answer shape the kit validates: choice -> {type, choice, probabilities, confidence},
  noul -> {type, noul = P(true)}, from the published `answer_from_probs`, with the probabilities
  left unrounded (the helper rounds to 6 decimals for display).

Run:  python -m decision_index run --engine jebadiah_engine:JebadiahEngine [--option device=mps]
Faster on a large GPU, same answers to bf16 rounding:
      --option prefix_cache=true --option batch_tokens=16384 --option max_batch=64
"""
from __future__ import annotations

import hashlib
import importlib
import os
import platform
import sys

from decision_index.engines import Engine, Unsupported

MODEL_REPO = "frontier-infra/jebadiah-27b"
MODEL_REVISION = "a5d7c80a084ba470da5b16fbff72907d76a7a075"
SCRIPTS = ("ainode_prompt_verbatim.py", "jebadiah_prompt.py", "jebadiah_model.py")


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _flag(v):
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


class _PrefixState:
    """What a forward over the shared prefix leaves behind, in the interface Qwen3.5's layers call on
    transformers' cache: per full-attention layer the keys and values (after RoPE), per Gated DeltaNet
    layer the last conv_kernel-1 inputs of the causal conv1d and the recurrent state. Recording
    (batch None) returns every input unchanged, so the prefix forward is the plain forward; with a
    batch it serves B suffix rows that continue from that one prefix."""

    def __init__(self, n_layers, length, batch=None, source=None):
        self.length, self.batch = length, batch
        if source is None:
            self.layers = [_PrefixLayer() for _ in range(n_layers)]
        else:
            self.layers = [lay.expanded(batch) for lay in source.layers]

    def for_batch(self, batch):
        return _PrefixState(len(self.layers), self.length, batch, self)

    def has_previous_state(self, layer_idx=0, state_idx=0):
        return self.batch is not None

    def get_seq_length(self, layer_idx=0):
        return self.length if self.batch is not None else 0

    def update_conv_state(self, x, layer_idx, conv_kernel_size=4, **kwargs):
        lay = self.layers[layer_idx]
        if self.batch is None:
            lay.conv_states[0] = x[:, :, -(conv_kernel_size - 1):].clone()
            return x
        import torch
        return torch.cat([lay.conv_states[0], x], dim=-1)

    def update_recurrent_state(self, state, layer_idx, **kwargs):
        if self.batch is None:
            self.layers[layer_idx].recurrent_states[0] = state

    def update(self, key, value, layer_idx, *args, **kwargs):
        lay = self.layers[layer_idx]
        if self.batch is None:
            lay.keys, lay.values = key, value
            return key, value
        import torch
        b = key.shape[0]
        cat = torch.cat
        return cat([lay.keys.expand(b, -1, -1, -1), key], dim=2), cat([lay.values.expand(b, -1, -1, -1), value], dim=2)


class _PrefixLayer:
    def __init__(self):
        self.conv_states, self.recurrent_states, self.keys, self.values = [None], [None], None, None
        self.record_past = False

    def expanded(self, b):
        out = _PrefixLayer()
        out.keys, out.values = self.keys, self.values
        if self.conv_states[0] is not None:
            out.conv_states = [self.conv_states[0].expand(b, -1, -1).contiguous()]
            out.recurrent_states = [self.recurrent_states[0].expand(b, *self.recurrent_states[0].shape[1:]).contiguous()]
        return out


class JebadiahEngine(Engine):
    name = "jebadiah-27b"

    def __init__(self, model=MODEL_REPO, revision=MODEL_REVISION, device=None, dtype=None, attn="sdpa", temperatures=True,
                 max_tokens=None, token_budget=32768, max_batch=8, mps_fast_delta=True, prefix_cache=False,
                 batch_tokens=None, prefix_align=64, pad_waste=0.15, cuda_alloc_conf=None, mem_cap_gb=None, **options):
        super().__init__(**options)
        if cuda_alloc_conf:
            # read by PyTorch's CUDA caching allocator at its first allocation, so it has to be set before the model loads
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", str(cuda_alloc_conf))
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        if self.device == "mps":
            # transformers' threaded weight loader intermittently segfaults moving tensors to MPS (1 load in 6 here)
            os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        self.mem_cap_gb = float(mem_cap_gb) if mem_cap_gb not in (None, "", 0, "0") else None
        if self.device == "cuda" and self.mem_cap_gb:
            # a hard ceiling for this process on a shared unified-memory box: the caching allocator raises
            # OutOfMemoryError at the cap instead of taking memory the resident servers need
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, self.mem_cap_gb * 2**30 / total), 0)
        dtype = dtype or "bfloat16"
        kernel_note = None
        if self.device == "mps" and mps_fast_delta:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            kernel_note = importlib.import_module("mps_delta").install()
        local = os.path.isdir(model)
        model_dir = model if local else snapshot_download(model, revision=revision)
        scripts = os.path.join(model_dir, "scripts")
        sys.path.insert(0, scripts)
        jm = importlib.import_module("jebadiah_model")
        jp = importlib.import_module("jebadiah_prompt")
        self.jp = jp
        self.jm = jm
        self.answer_from_probs = jp.answer_from_probs
        self.flatten_text = jp.flatten_text

        cfg = AutoConfig.from_pretrained(model_dir)
        window = getattr(getattr(cfg, "text_config", cfg), "max_position_embeddings")
        self.context = int(min(max_tokens or window, window))
        self.token_budget = max(int(token_budget), self.context) if self.device == "cuda" else int(token_budget)
        self.max_batch = int(max_batch)
        # Both off by default, which is the published behaviour above. batch_tokens, when set, is the padded-token
        # budget on every device (the CUDA override to the full context does not apply to it).
        self.prefix_cache = _flag(prefix_cache)
        self.batch_tokens = int(batch_tokens) if batch_tokens not in (None, "", 0, "0", "none", "None") else None
        self.prefix_align = max(1, int(prefix_align))
        self.pad_waste = float(pad_waste)
        self.trace = None
        self.oom_retries = 0
        if self.prefix_cache:
            self.latency = self.latency_prefix

        # the 27B is a merged checkpoint: the LoRA is already folded into the weights, so the model's own loader
        # reads it directly (as its scripts/decide_standalone.py does) and no adapter is applied
        tok = jm.load_tokenizer(model_dir)
        contract = {}
        cpath = os.path.join(model_dir, "prompt_contract.json")
        if os.path.exists(cpath):
            import json
            contract = json.load(open(cpath))
        # the prompt must be byte-identical to training: same chat template, same prompt source, thinking off
        if contract.get("chat_template_sha256") and jm.template_sha256(tok) != contract["chat_template_sha256"]:
            raise RuntimeError("chat template differs from the model's prompt_contract.json")
        if contract.get("prompt_source_sha256") and jp.PROMPT_SOURCE_SHA256 != contract["prompt_source_sha256"]:
            raise RuntimeError("prompt source differs from the model's prompt_contract.json")
        if contract.get("chat_template_kwargs") and dict(jp.CHAT_TEMPLATE_KWARGS) != contract["chat_template_kwargs"]:
            raise RuntimeError("chat template kwargs differ from the model's prompt_contract.json")
        net = jm.load_base(model_dir, attn_implementation=attn, dtype=getattr(torch, dtype), device=self.device)
        temps = jm.read_temperatures(model_dir) if temperatures else {}
        self.scorer = jm.Scorer(net, tok, max_tokens=self.context, temperatures=temps, device=self.device)
        self.tok = tok
        self.model_id = f"{MODEL_REPO}@{revision[:8]}"

        import transformers
        self.provenance = {
            "kind": "merged",
            "model_repo": MODEL_REPO, "model_revision": revision,
            "loaded_from": "local directory" if local else "hub snapshot",
            "inference_code": {name: _sha256(os.path.join(scripts, name)) for name in SCRIPTS},
            "inference_code_source": f"https://huggingface.co/{MODEL_REPO}/tree/{revision}/scripts",
            "prompt_source_sha256": jp.PROMPT_SOURCE_SHA256,
            "prompt_source_commit": jp.PROMPT_SOURCE_COMMIT,
            "chat_template_sha256": jm.template_sha256(tok),
            "chat_template_sha256_expected": contract.get("chat_template_sha256"),
            "chat_template_kwargs": dict(jp.CHAT_TEMPLATE_KWARGS),
            "temperatures": temps,
            "fp32_candidate_logits": jm.FP32_CANDIDATE_LOGITS,
            "declared_context_tokens": self.context,
            "declared_max_options": self.scorer.renderer.max_options_extended,
            "ainode_label_alphabet": self.scorer.renderer.max_options,
            "device": self.device, "dtype": dtype, "attn": attn,
            "batching": {"max_batch": self.max_batch, "batch_tokens": self.batch_tokens, "token_budget": self.token_budget,
                         "prefix_cache": self.prefix_cache, "prefix_align": self.prefix_align, "pad_waste": self.pad_waste,
                         "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "mem_cap_gb": self.mem_cap_gb},
            "delta_rule_kernel": kernel_note or "transformers default (flash-linear-attention when installed, else the torch reference)",
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version()},
        }

    latency = ("In-process wall time of one request: AINode rendering, tokenization and one label-token "
               "logit read per question (questions batched by padded token budget, no prefix cache), "
               "device-synchronized; excludes model loading.")

    latency_prefix = ("In-process wall time of one request: AINode rendering, tokenization and one label-token "
                      "logit read per question; when a request has two or more questions their shared system + state "
                      "token prefix runs once and each question's suffix runs from its cached keys, values and "
                      "Gated DeltaNet state (suffixes batched by padded token budget); device-synchronized; "
                      "excludes model loading.")

    def runtime(self):
        t = self.torch
        info = {"device": self.device, "torch": t.__version__, "platform": platform.platform(), "machine": platform.machine()}
        if self.device == "cuda":
            info["gpu"] = t.cuda.get_device_name(0)
        return info

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def _translate(self, q):
        """Mechanical schema translation to AINode's string-only wire; returns (question, notes)."""
        notes = []
        out = dict(q)
        if not isinstance(q.get("instructions"), str):
            out["instructions"] = self.flatten_text(q.get("instructions"))
            notes.append("instructions_flattened")
        crit = q.get("criteria")
        if isinstance(crit, dict) and any(v is not None and not isinstance(v, str) for v in crit.values()):
            out["criteria"] = {k: (v if v is None or isinstance(v, str) else self.flatten_text(v)) for k, v in crit.items()}
            notes.append("descriptions_flattened")
        return out, notes

    def _tick(self):
        if self.trace is None:
            return None
        import time
        self.synchronize()
        return time.perf_counter()

    def _tock(self, t0, kind, rows, padded, useful):
        if t0 is not None:
            import time
            self.synchronize()
            self.trace.append({"kind": kind, "rows": rows, "padded_tokens": padded, "useful_tokens": useful,
                               "seconds": time.perf_counter() - t0})

    def _pack(self, lengths, extra):
        """Batches over lengths sorted longest first, as (start, size): a row joins while the batch stays within
        max_batch rows, within the padded-token budget (rows x (longest + extra)) and within pad_waste of padding."""
        budget = self.batch_tokens or self.token_budget
        out, i = [], 0
        while i < len(lengths):
            w, n, used = lengths[i], 1, lengths[i]
            while (i + n < len(lengths) and n < self.max_batch and (w + extra) * (n + 1) <= budget
                   and w * (n + 1) - (used + lengths[i + n]) <= self.pad_waste * w * (n + 1)):
                used += lengths[i + n]
                n += 1
            out.append((i, n))
            i += n
        return out

    def _score_shared_prefix(self, rendered, token_ids):
        """Every question of a request renders the same system + state text first, so their token ids
        share a prefix. The prefix runs once; each question's suffix then runs from the keys, values,
        conv inputs and recurrent state the prefix left, batched by padded tokens. The cut is the longest
        common token prefix (at least one token left per question), rounded down to a multiple of
        prefix_align (64, the Gated DeltaNet chunk) so the suffix's chunks fall where they fall in the full
        forward. Returns None, meaning score the plain way, when the rounded prefix is empty."""
        torch = self.torch
        qids = list(rendered)
        seqs = [token_ids[k] for k in qids]
        cut = min(len(x) for x in seqs) - 1
        first = seqs[0]
        for x in seqs[1:]:
            j = 0
            while j < cut and x[j] == first[j]:
                j += 1
            cut = j
        cut -= cut % self.prefix_align
        if cut <= 0:
            return None
        core = self.jm.core_of(self.scorer.model)
        body = core.model
        dev = self.device
        n_layers = len(body.layers)
        with torch.no_grad():
            t0 = self._tick()
            state = _PrefixState(n_layers, cut)
            body(input_ids=torch.tensor([first[:cut]], device=dev),
                 attention_mask={"full_attention": None, "linear_attention": None},
                 position_ids=torch.arange(cut, device=dev)[None], past_key_values=state, use_cache=False)
            self._tock(t0, "prefix", 1, cut, cut)
            order = sorted(qids, key=lambda k: -len(token_ids[k]))
            kv_cost = cut // 16      # a row's share of the prefix keys and values it attends over, in token units
            probs = {}
            for i, n in self._pack([len(token_ids[k]) - cut for k in order], kv_cost):
                batch = order[i:i + n]
                width = len(token_ids[batch[0]]) - cut
                t0 = self._tick()
                b = len(batch)
                pad = self.tok.pad_token_id
                ids = torch.full((b, width), pad, dtype=torch.long)
                mask = torch.zeros((b, width), dtype=torch.long)
                for r, k in enumerate(batch):
                    suf = token_ids[k][cut:]
                    ids[r, :len(suf)] = torch.tensor(suf)
                    mask[r, :len(suf)] = 1
                ids, mask = ids.to(dev), mask.to(dev)
                # full attention: every suffix row sees the whole prefix, then itself causally, never padding
                causal = torch.ones((width, width), dtype=torch.bool, device=dev).tril()
                own = causal[None] & mask.bool()[:, None, :]
                full = torch.cat([torch.ones((b, width, cut), dtype=torch.bool, device=dev), own], dim=-1)[:, None]
                hidden = body(input_ids=ids,
                              attention_mask={"full_attention": full, "linear_attention": mask if bool((mask == 0).any()) else None},
                              position_ids=(cut + torch.arange(width, device=dev))[None].expand(b, width),
                              past_key_values=state.for_batch(b), use_cache=False).last_hidden_state
                last = mask.sum(dim=1) - 1
                h_last = hidden[torch.arange(b, device=dev), last]
                kmax = max(len(rendered[k][0].cand_ids) for k in batch)
                cand = torch.full((b, kmax), -1, dtype=torch.long, device=dev)
                for r, k in enumerate(batch):
                    cand[r, :len(rendered[k][0].cand_ids)] = torch.tensor(rendered[k][0].cand_ids, device=dev)
                logits = self._candidate_logits(core, h_last, cand)
                for r, k in enumerate(batch):
                    rr, qtype = rendered[k][0], rendered[k][1]
                    t = float(self.scorer.temperatures.get(qtype, 1.0))
                    probs[k] = torch.softmax(logits[r, :len(rr.cand_ids)] / t, dim=-1).tolist()
                self._tock(t0, "suffix", b, (width + kv_cost) * b, sum(len(token_ids[k]) - cut for k in batch))
        return probs

    def _candidate_logits(self, core, h_last, cand_ids):
        """The published option_logits' read after the forward, unchanged: fp32 W[cand] @ h."""
        torch = self.torch
        safe = cand_ids.clamp(min=0)
        if self.jm.FP32_CANDIDATE_LOGITS:
            head = core.lm_head
            w = head.weight[safe].float()
            cand = torch.einsum("bkh,bh->bk", w, h_last.float())
            if getattr(head, "bias", None) is not None:
                cand = cand + head.bias[safe].float()
        else:
            cand = core.lm_head(h_last).float().gather(1, safe)
        return cand.masked_fill(cand_ids < 0, float("-inf"))

    def _probs(self, rendered, token_ids):
        probs = self._score_shared_prefix(rendered, token_ids) if self.prefix_cache and len(rendered) > 1 else None
        if probs is None:
            # batches by padded token budget, longest first, so memory is bounded and a long prompt runs alone
            order = sorted(rendered, key=lambda k: -rendered[k][2])
            probs = {}
            if self.batch_tokens:
                batches = self._pack([rendered[k][2] for k in order], 0)
            else:
                batches, i = [], 0
                while i < len(order):
                    n = 1
                    while i + n < len(order) and n < self.max_batch and rendered[order[i]][2] * (n + 1) <= self.token_budget:
                        n += 1
                    batches.append((i, n))
                    i += n
            for i, n in batches:
                batch = order[i:i + n]
                t0 = self._tick()
                out = self.scorer.score_rendered([(rendered[k][0], rendered[k][1]) for k in batch])
                self._tock(t0, "full", len(batch), rendered[order[i]][2] * len(batch), sum(rendered[k][2] for k in batch))
                for k, p in zip(batch, out):
                    probs[k] = p
        return probs

    def __call__(self, state, questions):
        renderer = self.scorer.renderer
        rendered, raw, token_ids = {}, {}, {}
        for qid, q in questions.items():
            if q["type"] not in ("choice", "noul"):
                raise Unsupported(f"question type {q['type']!r} is not one Jebadiah answers")
            q2, notes = self._translate(q)
            try:
                r = self.scorer.render(state, q2)
            except ValueError as exc:
                if "single-token labels" in str(exc):
                    raise Unsupported(f"{len(q['criteria'])} options exceed the model's {renderer.max_options_extended} single-token labels") from exc
                raise
            if r.truncated:
                raise Unsupported(f"prompt exceeds the declared context of {self.context} tokens; refused, not truncated")
            ids = self.tok.encode(r.prompt, add_special_tokens=False)
            n = len(ids)
            rendered[qid] = (r, q2["type"], n)
            token_ids[qid] = ids
            raw[qid] = {"labels": r.letters, "label_scheme": r.label_scheme, "prompt_tokens": n}
            if notes:
                raw[qid]["translation"] = notes
        try:
            probs = self._probs(rendered, token_ids)
        except self.torch.OutOfMemoryError:
            # the box is shared, so free memory moves; retry this request one row per forward (same values to
            # bf16 batch noise) before letting the runner halt on a device error
            if self.device == "cuda":
                self.torch.cuda.empty_cache()
            elif self.device == "mps":
                self.torch.mps.empty_cache()
            self.oom_retries += 1
            saved = (self.max_batch, self.batch_tokens)
            self.max_batch, self.batch_tokens = 1, 1
            try:
                probs = self._probs(rendered, token_ids)
            finally:
                self.max_batch, self.batch_tokens = saved
            for qid in raw:
                raw[qid]["oom_retry"] = True
        answers = {}
        for qid, q in questions.items():
            r = rendered[qid][0]
            a = self.answer_from_probs(q, r.keys, probs[qid])
            if q["type"] == "choice":
                # the published helper rounds to 6 decimals for the route; the kit ranks ToolRet and
                # BRIGHT candidates by probability across requests, where rounding would create ties,
                # so the answer carries the unrounded softmax (same argmax, same values to 1e-6)
                a["probabilities"] = dict(zip(r.keys, probs[qid]))
            else:
                a["noul"] = float(probs[qid][r.keys.index("true")])
            answers[qid] = a
            raw[qid]["probabilities"] = dict(zip(r.keys, probs[qid]))
        usage = {"input_tokens": sum(v[2] for v in rendered.values())}
        return {"model": self.model_id, "answers": answers, "usage": usage}, raw
