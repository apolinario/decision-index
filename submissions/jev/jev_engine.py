"""Decision Index engine for the JEV models (autotrust/JEV-9B, autotrust/JEV-27B).

    python -m decision_index pipeline --engine jev_engine:JevEngine \
        --option model=autotrust/JEV-9B --option revision=<commit> --out runs/jev-9b

What it runs is the model's own System 1 path, exactly as published on the model card and used for every number the
card reports: the Qwen3.5/3.8 text backbone (bf16, fp32 tensors kept fp32), the System 1 LoRA from `adapter/` merged
in memory, and the 24-slot fp32 decision head from `head.safetensors`, with the per-kind temperatures from
`calibration.json`. One prefill pass per question, read-out at the last token of the bare-v1 template:

    [kind] choice
    [state] <state>
    [question] <instructions>
    [options]
    A) <option 1>
    ...
    [decision]:

Mechanical schema translation (recorded here, identical for every benchmark):
  * `state`, `instructions` and option descriptions are passed verbatim when they are strings, otherwise as JSON
    (`json.dumps`, default separators, as in the training corpus).
  * A choice option is rendered `"<key>: <description>"`, or just `"<key>"` when the description is empty or equal to
    the key (the two forms the training corpus uses).
  * `noul` questions use the fixed `false` / `true` options; the answer is P(true). Their `instructions` are a statement
    to judge; it is phrased `"Is this scenario one where: <statement>?"` (first letter lowercased unless the second is
    uppercase, one trailing period dropped), the form in which all 150,515 TypeSafe Jev 1.13-labelled `noul` rows of the
    training corpus present a statement. Instructions that already end in `?` are passed verbatim. The noul `criteria`
    descriptions are not rendered (the template's noul options are fixed).

Declared capacity: the decision head has 16 choice slots (A-P). A choice question with more than 16 options is answered
with every option read by the model and nothing pruned, in ceil(n/16) + 1 forward passes:
  1. the options are split, in the given order, into ceil(n/16) contiguous groups of near-equal size (each <= 16),
     and each group is read whole;
  2. a final of up to 16 options is read, in original order: the top option of every group, with the free places
     going to the next most likely options by in-group probability;
  3. finalists keep the final's distribution times the chance that the answer is among them; every other option
     gets its group's share of the final times its in-group probability. The result sums to 1 over all options.
No extra temperature is applied and the argmax of the combined distribution is the answer.

No truncation: a prompt longer than the backbone's context window is refused (`Unsupported`), never shortened.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import math
import os
import platform

import torch

from decision_index.engines.base import Engine, Unsupported

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_CHOICE = len(LETTERS)
SLOTS = {"noul": (0, 2), "choice": (8, 24)}
TEMPLATE_VERSION = "bare-v1"
NEEDED = ["config.json", "model*.safetensors", "model.safetensors.index.json", "tokenizer*", "chat_template.jinja",
          "special_tokens_map.json", "adapter/*", "head.safetensors", "judge_config.json", "calibration.json"]


def as_text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def option_text(key: str, desc) -> str:
    d = "" if desc is None else as_text(desc).strip()
    return key if (not d or d == key) else f"{key}: {d}"


def noul_question(instructions) -> str:
    s = as_text(instructions).strip()
    if not s or s.endswith("?"):
        return s
    if s.endswith(".") and not s.endswith(".."):
        s = s[:-1]
    if len(s) > 1 and s[0].isupper() and not s[1].isupper():
        s = s[0].lower() + s[1:]
    return f"Is this scenario one where: {s}?"


def render(kind: str, state: str, question: str, options: list[str]) -> str:
    lines = ["false", "true"] if kind == "noul" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
    return f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


def plan_groups(n: int) -> list[list[int]]:
    """Contiguous groups of near-equal size, each <= 16, in the given order."""
    k = math.ceil(n / MAX_CHOICE)
    base, extra = divmod(n, k)
    groups, i = [], 0
    for g in range(k):
        size = base + (1 if g < extra else 0)
        groups.append(list(range(i, i + size)))
        i += size
    return groups


def pick_finalists(groups: list[list[int]], in_group: dict[int, float]) -> list[int]:
    chosen = [max(g, key=lambda o: (in_group[o], -o)) for g in groups]
    rest = sorted((o for g in groups for o in g if o not in set(chosen)), key=lambda o: (-in_group[o], o))
    chosen += rest[: max(0, MAX_CHOICE - len(chosen))]
    return sorted(chosen)


def combine(groups: list[list[int]], in_group: dict[int, float], finalists: list[int], final: dict[int, float]) -> dict[int, float]:
    group_of = {o: gi for gi, g in enumerate(groups) for o in g}
    share = [0.0] * len(groups)      # final mass of each group's finalists
    captured = [0.0] * len(groups)   # in-group mass of each group's finalists
    for f in finalists:
        share[group_of[f]] += final[f]
        captured[group_of[f]] += in_group[f]
    among = sum(s * c for s, c in zip(share, captured))  # P(answer is a finalist)
    fin = set(finalists)
    return {o: (final[o] * among if o in fin else share[group_of[o]] * in_group[o]) for g in groups for o in g}


class JevEngine(Engine):
    autocast = False  # True for unmerged fp32-LoRA bundles (judge_config.merge_adapter = false)
    name = "jev"
    latency = ("In-process request wall time, CUDA-synchronized: template rendering, tokenization, every backbone "
               "forward pass of the request (all questions batched) and the head read-out; excludes model loading.")

    def __init__(self, model: str = "autotrust/JEV-9B", revision: str | None = None, local_dir: str | None = None,
                 device: str = "cuda", token_budget: int = 32768, attn: str = "sdpa", pad_mask: bool = True, **options):
        super().__init__(model=model, revision=revision, local_dir=local_dir, device=device, token_budget=token_budget,
                         attn=attn, pad_mask=pad_mask, **options)
        # pad_mask=False (batched driver only): right padding without an attention mask. The model is causal
        # (attention, gated-delta recurrence and short convolution only look back), so pad tokens placed after a
        # prompt cannot change the hidden state at its last real token; skipping the mask lets SDPA use the fused
        # causal kernel instead of materialising a [B, T, T] mask.
        self.pad_mask = pad_mask
        from huggingface_hub import HfApi, snapshot_download
        from peft import PeftModel
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device

        sha = None
        if local_dir is None:
            sha = HfApi().model_info(model, revision=revision).sha
            local_dir = snapshot_download(model, revision=sha, allow_patterns=NEEDED)
        self.device = torch.device(device)
        self.token_budget = int(token_budget)
        self.tok = AutoTokenizer.from_pretrained(local_dir)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        cfg = AutoConfig.from_pretrained(local_dir)
        text_cfg = getattr(cfg, "text_config", cfg)
        text_cfg._attn_implementation = attn
        self.max_len = int(getattr(text_cfg, "max_position_embeddings"))
        jc = json.load(open(os.path.join(local_dir, "judge_config.json")))
        ro = jc.get("readout") or {}
        self.readout = ro.get("type", "last_token")
        self.softcap = jc.get("softcap")
        if self.readout == "diffusion_canvas":
            # google/diffusiongemma-*: encoder reads [bos] + prompt (causal); the decoder denoises a fixed pad canvas
            # bidirectionally over prompt + canvas; the head reads canvas position 0. The encoder and decoder share
            # their base weights but carry separate LoRA adapters, so the adapter is kept UNMERGED.
            from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion
            self.canvas_len, self.canvas_token = int(ro["canvas_len"]), int(ro["canvas_token"])
            self.prefix_token = ro.get("prefix_token")
            m = DiffusionGemmaForBlockDiffusion.from_pretrained(local_dir, dtype=torch.bfloat16, device_map=str(self.device))
            m = PeftModel.from_pretrained(m, os.path.join(local_dir, jc.get("adapter_subfolder", "adapter")), is_trainable=False)
            m.eval()
            self.backbone = m.get_base_model().model
            self.max_len -= self.canvas_len + 1
            self._finish_init(local_dir, jc, model, sha, revision)
            return
        self.prefix_token = ro.get("prefix_token")
        if getattr(cfg, "model_type", None) == "gemma4":
            # google/gemma-4-*-it: autoregressive; the head reads the last real token of [bos] + prompt (softcap from
            # judge_config). One causal pass, so the System 1 adapter is merged in memory like the Qwen bundles.
            from transformers import Gemma4ForConditionalGeneration
            m = Gemma4ForConditionalGeneration.from_pretrained(local_dir, dtype=torch.bfloat16, device_map=str(self.device),
                                                               attn_implementation=attn)
            m.lm_head = None
            m = PeftModel.from_pretrained(m, os.path.join(local_dir, jc.get("adapter_subfolder", "adapter")), is_trainable=False)
            if jc.get("merge_adapter", True):
                m = m.merge_and_unload()
                self.backbone = m.model
            else:
                # judge_config.merge_adapter = false: run exactly as trained -- adapter unmerged with fp32 LoRA weights,
                # bf16 backbone under bf16 autocast (merging into bf16 weights would round the small LoRA deltas)
                for n, p in m.named_parameters():
                    if "lora_" in n:
                        p.data = p.data.float()
                self.backbone = m.get_base_model().model
                self.autocast = True
            m.eval()
            self.max_len -= 1
            self._finish_init(local_dir, jc, model, sha, revision)
            return
        with init_empty_weights(include_buffers=False):
            lm = Qwen3_5ForCausalLM(text_cfg)
        expected = set(lm.state_dict())
        seen = set()
        for shard in sorted(glob.glob(os.path.join(local_dir, "model*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    nk = k.replace("model.language_model.", "model.", 1)
                    if nk == "lm_head.weight" or nk not in expected:  # decisions never use the vocab projection
                        continue
                    t = f.get_tensor(k)
                    set_module_tensor_to_device(lm, nk, self.device, value=t,
                                                dtype=torch.float32 if t.dtype == torch.float32 else torch.bfloat16)
                    seen.add(nk)
        missing = expected - seen - {"lm_head.weight"}
        if missing:
            raise RuntimeError(f"{len(missing)} backbone weights missing, e.g. {sorted(missing)[:3]}")
        lm.lm_head = None
        lm.to(self.device)
        lm = PeftModel.from_pretrained(lm, os.path.join(local_dir, jc.get("adapter_subfolder", "adapter")),
                                       is_trainable=False).merge_and_unload()
        lm.eval()
        self.backbone = lm.model
        self._finish_init(local_dir, jc, model, sha, revision)

    def _finish_init(self, local_dir, jc, model, sha, revision):
        from safetensors.torch import load_file

        head = load_file(os.path.join(local_dir, "head.safetensors"))
        self.W = head["proj.weight"].to(self.device, torch.float32)
        self.b = head["proj.bias"].to(self.device, torch.float32)
        self.T = json.load(open(os.path.join(local_dir, "calibration.json")))["per_kind"]
        if jc.get("slots", {}).get("ranges"):
            assert {k: tuple(v) for k, v in jc["slots"]["ranges"].items() if k in SLOTS} == SLOTS
        tv = jc.get("slots", {}).get("template_version") or jc.get("template_version")
        if tv and tv != TEMPLATE_VERSION:
            raise RuntimeError(f"bundle template {tv!r} != engine template {TEMPLATE_VERSION!r}")
        files = {}
        for name in ("head.safetensors", "calibration.json", "judge_config.json", "adapter/adapter_model.safetensors"):
            p = os.path.join(local_dir, name)
            if os.path.exists(p):
                files[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        self.provenance = {
            "kind": "trained", "repo": model, "revision": sha or revision, "local_dir": local_dir,
            "path": ("System 1: bf16 block-diffusion model + adapter/ LoRA (unmerged) + 24-slot fp32 head (softcap) read at "
                     f"canvas position 0 of a {getattr(self, 'canvas_len', 0)}-token pad canvas + calibration.json")
                    if self.readout == "diffusion_canvas" else
                    ("System 1: bf16 backbone + adapter/ LoRA merged in memory + 24-slot fp32 head"
                     + (" (softcap)" if self.softcap else "") + (" read at the last token of [bos] + prompt" if self.prefix_token is not None else "")
                     + " + calibration.json"),
            "template": TEMPLATE_VERSION, "temperatures": self.T, "max_choice_options_per_pass": MAX_CHOICE,
            "over_16_options": "ceil(n/16) contiguous groups + one final of 16; every option read, none pruned",
            "truncation": "none; prompts over max_position_embeddings are refused", "max_position_embeddings": self.max_len,
            "sha256": files,
        }

    # ---------------------------------------------------------------------------------------------------------------
    def runtime(self):
        import peft
        import transformers
        return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else None,
                "python": platform.python_version(), "dtype": "bfloat16 backbone, float32 head"}

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def _logits(self, prompts: list[str]) -> list[torch.Tensor]:
        """Head logits [24] (fp32, before temperature) for each prompt; length-sorted micro-batches, right padding."""
        ids = [self.tok.encode(p, add_special_tokens=False) for p in prompts]
        for x in ids:
            if len(x) > self.max_len:
                raise Unsupported(f"prompt of {len(x)} tokens exceeds the {self.max_len}-token context window")
        return self._logits_ids(ids)

    @torch.inference_mode()
    def _logits_ids(self, ids: list[list[int]]) -> list[torch.Tensor]:
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out: list[torch.Tensor | None] = [None] * len(ids)
        batches, cur = [], []
        for i in order:
            extra = (self.canvas_len + 1) if self.readout == "diffusion_canvas" else 0
            if cur and (len(cur) + 1) * (len(ids[i]) + extra) > self.token_budget:
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        pending = list(batches)
        while pending:
            batch = pending.pop(0)
            try:
                if self.readout == "diffusion_canvas":
                    z = self._diffusion_logits([ids[i] for i in batch])
                    for r, i in enumerate(batch):
                        out[i] = z[r]
                    continue
                pre = [self.prefix_token] if self.prefix_token is not None else []
                seqs = [pre + list(ids[i]) for i in batch]
                T = max(len(x) for x in seqs)
                inp = torch.full((len(batch), T), self.tok.pad_token_id, dtype=torch.long)
                am = torch.zeros((len(batch), T), dtype=torch.long)
                for r, x in enumerate(seqs):
                    inp[r, : len(x)] = torch.tensor(x)
                    am[r, : len(x)] = 1
                mask = am.to(self.device) if (self.pad_mask or len(batch) == 1) else None  # causal backbones only
                with (torch.autocast("cuda", dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext()):
                    hs = self.backbone(input_ids=inp.to(self.device), attention_mask=mask, use_cache=False).last_hidden_state
                last = torch.tensor([len(x) - 1 for x in seqs], device=self.device)
                h = hs[torch.arange(len(batch), device=self.device), last].to(torch.float32)
                z = h @ self.W.T + self.b
                if self.softcap:
                    z = self.softcap * torch.tanh(z / self.softcap)
                for r, i in enumerate(batch):
                    out[i] = z[r]
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(batch) == 1:
                    raise Unsupported(f"a {len(ids[batch[0]])}-token prompt does not fit in GPU memory")
                pending[:0] = [batch[: len(batch) // 2], batch[len(batch) // 2:]]
        return out  # type: ignore[return-value]

    def _diffusion_logits(self, seqs: list[list[int]]) -> torch.Tensor:
        pre = [self.prefix_token] if self.prefix_token is not None else []
        seqs = [pre + list(x) for x in seqs]
        B, T, C, dev = len(seqs), max(map(len, seqs)), self.canvas_len, self.device
        inp = torch.full((B, T), self.tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((B, T), dtype=torch.long)
        for r, x in enumerate(seqs):
            inp[r, : len(x)] = torch.tensor(x)
            am[r, : len(x)] = 1
        lengths = am.sum(1).to(dev)
        inp, am = inp.to(dev), am.to(dev)
        canvas = torch.full((B, C), self.canvas_token, dtype=torch.long, device=dev)
        pos = lengths.unsqueeze(1) + torch.arange(C, device=dev).unsqueeze(0)
        dmask = torch.cat([am, torch.ones((B, C), dtype=am.dtype, device=dev)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.backbone(input_ids=inp, attention_mask=am, decoder_input_ids=canvas, decoder_attention_mask=dmask,
                               decoder_position_ids=pos).last_hidden_state
        z = hs[:, 0].to(torch.float32) @ self.W.T + self.b
        if self.softcap:
            z = self.softcap * torch.tanh(z / self.softcap)
        return z

    def _probs(self, kind: str, z: torch.Tensor, n: int) -> list[float]:
        s = SLOTS[kind][0]
        return torch.softmax(z[s : s + n] / float(self.T[kind]), dim=0).tolist()

    def __call__(self, state, questions):
        st = as_text(state) if state is not None else ""
        jobs = []  # (question key, kind, rendered prompt, option indices)
        plans = {}
        for key, q in questions.items():
            qtext = as_text(q.get("instructions", ""))
            if q["type"] == "noul":
                jobs.append((key, "noul", render("noul", st, noul_question(q.get("instructions", "")), []), None))
            elif q["type"] == "choice":
                keys = list(q["criteria"])
                texts = [option_text(k, q["criteria"][k]) for k in keys]
                if len(keys) < 2:
                    raise Unsupported("choice needs at least 2 options")
                groups = [list(range(len(keys)))] if len(keys) <= MAX_CHOICE else plan_groups(len(keys))
                plans[key] = (keys, texts, groups)
                for gi, g in enumerate(groups):
                    jobs.append((key, "choice", render("choice", st, qtext, [texts[o] for o in g]), (gi, g)))
            else:
                raise Unsupported(f"unsupported question type {q['type']!r}")
        zs = self._logits([j[2] for j in jobs])
        answers, raw = {}, {"passes": len(jobs)}
        in_group: dict[str, dict[int, float]] = {k: {} for k in plans}
        for (key, kind, _, extra), z in zip(jobs, zs):
            if kind == "noul":
                answers[key] = {"type": "noul", "noul": self._probs("noul", z, 2)[1]}
            else:
                _, g = extra
                for o, p in zip(g, self._probs("choice", z, len(g))):
                    in_group[key][o] = p
        finals = []
        for key, (keys, texts, groups) in plans.items():
            if len(groups) > 1:
                fin = pick_finalists(groups, in_group[key])
                qtext = as_text(questions[key].get("instructions", ""))
                finals.append((key, fin, render("choice", st, qtext, [texts[o] for o in fin])))
        final_probs = {}
        if finals:
            for (key, fin, _), z in zip(finals, self._logits([f[2] for f in finals])):
                final_probs[key] = (fin, dict(zip(fin, self._probs("choice", z, len(fin)))))
            raw["passes"] += len(finals)
        for key, (keys, texts, groups) in plans.items():
            if len(groups) == 1:
                p = in_group[key]
            else:
                fin, fp = final_probs[key]
                p = combine(groups, in_group[key], fin, fp)
            probs = {keys[o]: float(v) for o, v in p.items()}
            tot = sum(probs.values())
            probs = {k: v / tot for k, v in probs.items()}
            answers[key] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
        return {"model": self.provenance["repo"], "answers": answers}, raw
