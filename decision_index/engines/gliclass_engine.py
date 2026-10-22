"""GLiClass engine: runs GLiClass zero-shot classifiers (https://github.com/Knowledgator/GLiClass) in process.

    pip install -e ".[gliclass]"
    python -m decision_index run --engine gliclass --model knowledgator/gliclass-v4-edge \
        --option dtype=bfloat16 --option max_length=16384 --option batch_size=8 \
        --rows sample-100.jsonl.gz --out runs/gliclass-v4-edge-sample

Every question of a request becomes one GLiClass classification over the same text:
- text: the request `state` (a string as-is; a JSON object as `key: value` lines, list values as `- item` lines)
- prompt: the question `instructions`
- labels: one per option, the option description (`--option labels=key|key_description` to change)
- probabilities: softmax over the option logits (`--option activation=sigmoid` normalizes sigmoid scores instead)

`noul` questions are answered the same way over their `criteria` (yes/no when none are given), reporting the
probability of the `true` option. Requests that do not fit `max_length` tokens are refused as unsupported, never
truncated.
"""

from decision_index.engines.base import Engine, Unsupported, text

NO_LIMIT = 10**9


def render_state(state):
    if state in ("", None, {}, []):
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "\n".join(f"- {text(item)}" for item in state)
    lines = []
    for key, value in state.items():
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            lines.append(f"{key}:")
            lines.extend(f"- {item}" for item in value)
        else:
            lines.append(f"{key}: {text(value)}")
    return "\n".join(lines)


def option_criteria(question):
    if question["type"] == "choice":
        return question["criteria"]
    if question["type"] == "noul":
        if question.get("criteria"):
            return question["criteria"]
        return {"true": "yes", "false": "no"}
    raise Unsupported("Unsupported question type " + str(question["type"]))


def option_label(key, description, mode="description"):
    description = key if description is None else text(description)
    if mode == "key":
        return key
    if mode == "key_description":
        return f"{key}: {description}"
    return description


class GLiClassEngine(Engine):
    name = "gliclass"
    latency = "Device-synchronized in-process request wall time including input construction; excludes model loading."

    def __init__(
        self,
        model="knowledgator/gliclass-v4-edge",
        tokenizer=None,
        device=None,
        dtype="float32",
        max_length=1024,
        labels="description",
        activation="softmax",
        max_options=None,
        batch_size=1,
        batch_tokens=65536,
        **options,
    ):
        super().__init__(**options)
        if labels not in ("description", "key", "key_description"):
            raise ValueError("labels must be description, key or key_description")
        if activation not in ("softmax", "sigmoid"):
            raise ValueError("activation must be softmax or sigmoid")
        import torch
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline

        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GLiClassModel.from_pretrained(model).to(dtype=getattr(torch, dtype))
        self.pipeline = ZeroShotClassificationPipeline(
            self.model,
            tokenizer or model,
            max_length=NO_LIMIT,  # length is checked here and refused instead of truncated
            classification_type="single-label",
            device=self.device,
            progress_bar=False,
        )
        self.pipe = self.pipeline.pipe
        self.limit = self._text_limit(int(max_length))
        self.label_limit = self._label_limit()
        self.labels = labels
        self.activation = activation
        self.max_options = int(max_options) if max_options else None
        self.batch_size = int(batch_size)
        self.batch_tokens = int(batch_tokens)
        self.model_id = model
        self.provenance = {
            "kind": "gliclass",
            "repo": model,
            "revision": getattr(self.model.config, "_commit_hash", None),
            "architecture_type": self.model.config.architecture_type,
            "device": str(self.pipe.device),
            "dtype": dtype,
            "context_limit_tokens": self.limit,
            "label_limit_tokens": self.label_limit,
            "labels": labels,
            "activation": activation,
            "batch_size": self.batch_size,
            "batch_tokens": self.batch_tokens,
            "policy": "Each question is one GLiClass classification: the state is the text, the instructions are the "
            "task prompt and every option is a label; probabilities are the "
            + ("softmax over option logits" if activation == "softmax" else "normalized per-option sigmoid scores")
            + ". Inputs over the token limit are refused as unsupported, never truncated. No option is filtered "
            "and the rendering is fixed for every benchmark.",
        }

    def _text_limit(self, max_length):
        """max_length, capped by the encoder's absolute position table when it has one."""
        encoder = getattr(self.model.config, "encoder_config", None)
        if encoder is None or getattr(encoder, "relative_attention", False):
            return max_length
        positions = getattr(encoder, "max_position_embeddings", None)
        return min(max_length, positions) if positions else max_length

    def _label_limit(self):
        label_config = getattr(self.model.config, "label_model_config", None)
        if self.model.config.architecture_type in ("bi-encoder", "bi-encoder-fused") and label_config is not None:
            return getattr(label_config, "max_position_embeddings", None) or self.limit
        return self.limit

    def runtime(self):
        torch = self.torch
        info = {"torch": torch.__version__, "device": str(self.pipe.device)}
        if self.pipe.device.type == "cuda":
            info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(self.pipe.device))
        try:
            import gliclass
            import transformers

            info.update(gliclass=gliclass.__version__, transformers=transformers.__version__)
        except Exception:
            pass
        return info

    def synchronize(self):
        if self.pipe.device.type == "cuda":
            self.torch.cuda.synchronize(self.pipe.device)

    def _logits(self, state_text, prompt, labels):
        torch = self.torch
        inputs = self.pipe.prepare_inputs([state_text], labels, same_labels=True, prompt=prompt)
        length = inputs["input_ids"].shape[-1]
        if length > self.limit:
            raise Unsupported(f"input of {length} tokens exceeds the {self.limit}-token limit")
        label_length = inputs["class_input_ids"].shape[-1] if "class_input_ids" in inputs else 0
        if label_length > self.label_limit:
            raise Unsupported(f"an option of {label_length} tokens exceeds the {self.label_limit}-token label limit")
        max_num_classes = self.pipe._resolve_max_num_classes(labels, True)
        try:
            out = self.model(**inputs, max_num_classes=max_num_classes)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise Unsupported(f"out of memory on a {length}-token input with {len(labels)} options") from None
        logits = out.logits[0, : len(labels)].float()
        if logits.shape[0] != len(labels):
            raise Unsupported(f"model returned {logits.shape[0]} scores for {len(labels)} options")
        return logits, length

    def _logits_batch(self, state_text, items):
        """items: [(prompt, labels)]. One padded forward pass per chunk of up to batch_size questions whose
        padded size stays under batch_tokens; falls back to one pass per question on out-of-memory."""
        torch = self.torch
        if self.batch_size <= 1 or len(items) == 1:
            return [self._logits(state_text, prompt, labels) for prompt, labels in items]
        out = [None] * len(items)
        lengths = []
        for prompt, labels in items:  # length check per question, as in _logits
            ids = self.pipe.prepare_inputs([state_text], labels, same_labels=True, prompt=prompt)["input_ids"]
            if ids.shape[-1] > self.limit:
                raise Unsupported(f"input of {ids.shape[-1]} tokens exceeds the {self.limit}-token limit")
            lengths.append(ids.shape[-1])
        order = sorted(range(len(items)), key=lambda i: lengths[i])
        chunks, chunk = [], []
        for i in order:
            if chunk and (len(chunk) >= self.batch_size or (len(chunk) + 1) * lengths[i] > self.batch_tokens):
                chunks.append(chunk)
                chunk = []
            chunk.append(i)
        chunks.append(chunk)
        for chunk in chunks:
            if len(chunk) == 1:
                i = chunk[0]
                out[i] = self._logits(state_text, *items[i])
                continue
            labels = [items[i][1] for i in chunk]
            inputs = self.pipe.prepare_inputs(
                [state_text] * len(chunk), labels, same_labels=False, prompt=[items[i][0] or "" for i in chunk]
            )
            try:
                logits = self.model(**inputs, max_num_classes=max(len(group) for group in labels)).logits.float()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                for i in chunk:
                    out[i] = self._logits(state_text, *items[i])
                continue
            for row, i in enumerate(chunk):
                out[i] = (logits[row, : len(items[i][1])], lengths[i])
        return out

    def __call__(self, state, questions):
        torch = self.torch
        state_text = render_state(state)
        answers, raw = {}, {}
        names, items, keys_of = [], [], {}
        for k, q in questions.items():
            criteria = option_criteria(q)
            keys = list(criteria)
            if self.max_options and len(keys) > self.max_options:
                raise Unsupported(f"{len(keys)} options exceed the declared limit of {self.max_options}")
            names.append(k)
            keys_of[k] = keys
            labels = [option_label(key, criteria[key], self.labels) for key in keys]
            items.append((text(q.get("instructions", "")) or None, labels))
        with torch.no_grad():
            scored = self._logits_batch(state_text, items)
        for k, (logits, length) in zip(names, scored):
            q, keys = questions[k], keys_of[k]
            if self.activation == "softmax":
                scores = torch.softmax(logits, dim=-1)
            else:
                scores = torch.sigmoid(logits)
                scores = scores / scores.sum() if scores.sum() > 0 else torch.full_like(scores, 1 / len(keys))
            probs = dict(zip(keys, scores.tolist()))
            raw[k] = {"option_logits": dict(zip(keys, logits.tolist())), "input_tokens": length}
            if q["type"] == "choice":
                answers[k] = {"type": "choice", "choice": max(keys, key=probs.get), "probabilities": probs}
            else:
                answers[k] = {"type": "noul", "noul": probs.get("true", 1 - probs.get("false", 0.5))}
        usage = {"input_tokens": sum(v["input_tokens"] for v in raw.values())}
        return {"model": self.model_id, "answers": answers, "usage": usage}, raw
