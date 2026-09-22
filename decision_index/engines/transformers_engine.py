import json
import os

from decision_index.engines.base import Engine, Unsupported, text

SYSTEM_PROMPT = "You are a decision engine. Read the state, then answer the question by replying with exactly one option key from the list. Do not explain."


def render_prompt(state, question):
    lines = ["State:", text(state) if state not in ("", None, {}, []) else "(empty)", "", "Question:", text(question.get("instructions", "")), "", "Options:"]
    criteria = question["criteria"] if question["type"] == "choice" else {"false": "False", "true": "True"}
    for key, description in criteria.items():
        lines.append(f"{key}: {key if description is None else text(description)}")
    lines += ["", "Reply with exactly one option key."]
    return "\n".join(lines)


class TransformersEngine(Engine):
    name = "transformers"
    latency = "Device-synchronized in-process request wall time including prompt construction and state-prefix cache reuse; excludes model loading. First shapes may include kernel warmup."

    def __init__(self, model="Qwen/Qwen2.5-0.5B-Instruct", revision=None, device=None, dtype=None, attn="sdpa", max_tokens=None, cache_prefix=True, chat_template=True, **options):
        super().__init__(**options)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        if dtype is None:
            dtype = "bfloat16" if self.device == "cuda" else "float32"
        self.dtype = getattr(torch, dtype)
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=self.dtype, attn_implementation=attn).to(self.device).eval()
        self.attn = attn
        limit = getattr(self.model.config, "max_position_embeddings", None) or getattr(self.tok, "model_max_length", None)
        self.limit = min(limit, max_tokens) if max_tokens else limit
        self.cache_prefix = cache_prefix
        self.chat_template = chat_template and getattr(self.tok, "chat_template", None) is not None
        self.model_id = model
        self.cache = None
        self.cached_ids = []
        self.provenance = {
            "kind": "transformers",
            "repo": model,
            "revision": revision or getattr(self.model.config, "_commit_hash", None),
            "device": self.device,
            "dtype": dtype,
            "attn_implementation": attn,
            "context_limit_tokens": self.limit,
            "chat_template": self.chat_template,
            "prefix_cache": cache_prefix,
            "policy": "Every option is scored in one forward pass with a block-diagonal attention mask over a shared prompt; option score is the summed log-probability of the option key tokens; probabilities are the softmax over options. Prompts that would exceed the context limit are refused as unsupported, never truncated. No option is filtered and the prompt template is fixed.",
        }

    def runtime(self):
        torch = self.torch
        info = {"torch": torch.__version__, "device": self.device}
        if self.device == "cuda":
            info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        try:
            import transformers

            info["transformers"] = transformers.__version__
        except Exception:
            pass
        return info

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def _prompt_ids(self, state, question):
        body = render_prompt(state, question)
        if self.chat_template:
            prompt = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
            return self.tok(prompt, add_special_tokens=False)["input_ids"]
        return self.tok(SYSTEM_PROMPT + "\n\n" + body + "\nAnswer:", add_special_tokens=True)["input_ids"]

    def _option_ids(self, keys):
        out = []
        for key in keys:
            ids = self.tok((" " if not self.chat_template else "") + key, add_special_tokens=False)["input_ids"]
            if not ids:
                raise Unsupported(f"option key {key!r} tokenizes to nothing")
            out.append(ids)
        return out

    def _common_prefix(self, ids):
        n = min(len(ids), len(self.cached_ids), len(ids) - 1)
        p = 0
        while p < n and ids[p] == self.cached_ids[p]:
            p += 1
        return p

    def _forward(self, p, q_ids, options):
        torch = self.torch
        from transformers import DynamicCache

        if self.cache is None or p == 0:
            self.cache = DynamicCache()
            p = 0
        else:
            self.cache.crop(p)
        lengths = [len(o) for o in options]
        q_len = len(q_ids)
        flat = list(q_ids)
        starts = []
        for o in options:
            starts.append(len(flat))
            flat.extend(o)
        Q = len(flat)
        K = p + Q
        allowed = torch.zeros(Q, K, dtype=torch.bool)
        rows = torch.arange(Q)
        cols = torch.arange(K)
        allowed[:q_len] = cols[None, :] <= (p + rows[:q_len])[:, None]
        for s, m in zip(starts, lengths):
            block = allowed[s:s + m]
            block[:, : p + q_len] = True
            local = torch.arange(m)
            block[:, p + s : p + s + m] = local[None, :] <= local[:, None]
        positions = list(range(p, p + q_len))
        for m in lengths:
            positions.extend(range(p + q_len, p + q_len + m))
        keep = [q_len - 1]
        for s, m in zip(starts, lengths):
            keep.extend(range(s, s + m - 1))
        if self.attn == "eager":
            mask = torch.zeros(Q, K, dtype=self.dtype).masked_fill(~allowed, torch.finfo(self.dtype).min)
        else:
            mask = allowed
        mask = mask[None, None].to(self.device)
        input_ids = torch.tensor([flat], device=self.device)
        position_ids = torch.tensor([positions], device=self.device)
        keep_t = torch.tensor(keep, device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids, past_key_values=self.cache, use_cache=True, logits_to_keep=keep_t)
            logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
        self.cache.crop(p + q_len)
        scores = []
        cursor = 1
        for o, s, m in zip(options, starts, lengths):
            total = logprobs[0, o[0]].item()
            for i in range(1, m):
                total += logprobs[cursor + i - 1, o[i]].item()
            cursor += m - 1
            scores.append(total)
        return scores

    def _score_options(self, p, q_ids, options):
        torch = self.torch
        chunk = len(options)
        while True:
            try:
                scores = []
                for start in range(0, len(options), chunk):
                    scores.extend(self._forward(p, q_ids, options[start:start + chunk]))
                return scores
            except torch.OutOfMemoryError:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if chunk <= 1:
                    raise Unsupported(f"out of memory scoring one option after a {len(q_ids)}-token prompt")
                chunk = max(1, chunk // 2)

    def __call__(self, state, questions):
        answers, raw = {}, {}
        for k, q in questions.items():
            if q["type"] not in ("choice", "noul"):
                raise Unsupported("Unsupported question type " + str(q["type"]))
            keys = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"]
            ids = self._prompt_ids(state, q)
            options = self._option_ids(keys)
            if len(ids) + max(len(o) for o in options) > self.limit:
                raise Unsupported(f"prompt of {len(ids)} tokens plus option exceeds the {self.limit}-token context limit")
            p = self._common_prefix(ids) if (self.cache_prefix and self.cache is not None) else 0
            scores = self._score_options(p, ids[p:], options)
            self.cached_ids = ids
            top = max(scores)
            weights = [self.torch.exp(self.torch.tensor(s - top)).item() for s in scores]
            z = sum(weights)
            probs = {key: w / z for key, w in zip(keys, weights)}
            choice = max(keys, key=lambda x: probs[x])
            raw[k] = {"option_logprobs": dict(zip(keys, scores)), "prompt_tokens": len(ids), "cached_prefix_tokens": p}
            if q["type"] == "choice":
                answers[k] = {"type": "choice", "choice": choice, "probabilities": probs}
            else:
                answers[k] = {"type": "noul", "noul": probs["true"]}
        return {"model": self.model_id, "answers": answers, "usage": {"input_tokens": sum(v["prompt_tokens"] for v in raw.values())}}, raw

    def reset_cache(self):
        self.cache = None
        self.cached_ids = []
