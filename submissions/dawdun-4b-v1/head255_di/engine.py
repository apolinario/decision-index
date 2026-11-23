"""Inference-only engine for the public Decision Index runner."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from threading import Lock

from decision_index.engines.base import Engine, Unsupported
from . import prompt
from .prompt import MAX_OPTIONS, messages, render_question

VISION_MAX_PIXELS = 1_638_400
MODEL_FILES = (
    "config.json", "classification_config.json", "decision_config.json", "readout.safetensors",
    "model.safetensors", "model.safetensors.index.json", "model-*.safetensors",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
    "processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
)


def option_keys(row):
    return [option["id"] for option in row["options"]]


def make_answer(row, probabilities):
    kind = row["type"]
    if kind == "noul":
        return {"type": "noul", "noul": probabilities[1]}
    keys = option_keys(row)
    distribution = dict(zip(keys, probabilities, strict=True))
    result = {"type": kind, "probabilities": distribution}
    if kind == "choice":
        result["choice"] = max(distribution, key=distribution.__getitem__)
    else:
        result["score"] = sum(i * p for i, p in enumerate(probabilities))
        result["legend"] = {o["id"]: o["criterion"] or o["id"] for o in row["options"]}
    return result


def noncausal_full_attention(module, args, kwargs):
    from transformers.masking_utils import create_recurrent_attention_mask

    if args or kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
        raise ValueError("this attention path requires an uncached forward")
    padding = kwargs["attention_mask"]
    embeddings = kwargs.get("inputs_embeds")
    if embeddings is None:
        embeddings = module.embed_tokens(kwargs["input_ids"])
    kwargs["attention_mask"] = {
        "full_attention": padding[:, None, None, :].bool(),
        "linear_attention": create_recurrent_attention_mask(
            config=module.config, inputs_embeds=embeddings, attention_mask=padding
        ),
    }
    return args, kwargs


def read_bundle(model, revision):
    local = Path(model).expanduser()
    if local.is_dir():
        if revision:
            raise ValueError("revision only applies to a Hugging Face model repository")
        return local, "local model export", None
    from huggingface_hub import snapshot_download

    if not revision:
        raise ValueError("pin the model repository with --option revision=<commit>")
    snapshot = Path(snapshot_download(model, revision=revision, allow_patterns=MODEL_FILES))
    return snapshot, model, snapshot.name


def read_head_config(ckpt):
    head = json.loads((ckpt / "classification_config.json").read_text(encoding="utf-8"))
    codes = tuple(head["codes"])
    token_ids = head["token_ids"]
    num_letters = len(prompt.LETTERS)
    if (len(codes) != MAX_OPTIONS or any(not isinstance(code, str) or not code for code in codes)
            or len(set(codes)) != MAX_OPTIONS or codes[:num_letters] != tuple(prompt.LETTERS)
            or len(token_ids) != MAX_OPTIONS or any(type(token_id) is not int for token_id in token_ids)
            or len(set(token_ids)) != MAX_OPTIONS):
        raise ValueError("expected a 255-class single-pass readout")
    letter_token_ids = token_ids[:num_letters]

    if "temperature_by_type" in head:
        return codes, letter_token_ids, head["temperature_by_type"]

    # Older exports stored calibration in a separate file.
    decision = json.loads((ckpt / "decision_config.json").read_text(encoding="utf-8"))
    if (letter_token_ids != decision["letter_token_ids"]
            or decision.get("prompt_attention") != "bidirectional" or decision.get("auto_think")):
        raise ValueError("incompatible legacy readout configuration")
    return codes, letter_token_ids, decision["temperature_by_type"]


class Head255Engine(Engine):
    """One forward per question and one probability per supplied choice."""

    name = "head255-4b"
    latency = "In-process wall time for the whole request on a single device."

    def __init__(self, model, *, revision=None, device="cuda", context_tokens=None,
                 cpu_threads=4, memory_fraction=0.9, image_root=None,
                 min_pixels=65_536, max_pixels=VISION_MAX_PIXELS):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoProcessor
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

        super().__init__(model=model, revision=revision, device=device)
        self.torch = torch
        self.device = torch.device(device)
        torch.set_num_threads(int(cpu_threads))
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_per_process_memory_fraction(float(memory_fraction), self.device)
        self.image_root = image_root
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        if not 0 < self.min_pixels <= self.max_pixels <= VISION_MAX_PIXELS:
            raise ValueError("invalid vision pixel bounds")
        self._lock = Lock()
        ckpt, source, resolved_revision = read_bundle(model, revision)
        self.codes, letter_token_ids, self.temperatures = read_head_config(ckpt)
        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=False)
        tokenizer = self.processor.tokenizer
        if prompt.check_tokenizer(tokenizer) != letter_token_ids:
            raise ValueError("unexpected tokenizer/decision-letter mapping")
        tokenizer.padding_side = "left"
        bf16 = self.device.type == "cuda"
        placement = {"device_map": {"": str(self.device)}} if bf16 else {}
        self.backbone = Qwen3_5Model.from_pretrained(
            ckpt, dtype=torch.bfloat16 if bf16 else torch.float32,
            attn_implementation="sdpa", **placement
        ).eval()
        if getattr(self.backbone.config, "jev_prompt_attention", None) != "bidirectional":
            raise ValueError("model bundle is missing the bidirectional prompt attention setting")
        if len(tokenizer) > self.backbone.get_input_embeddings().num_embeddings:
            raise ValueError("tokenizer vocabulary exceeds model embeddings")
        self.backbone.language_model.register_forward_pre_hook(noncausal_full_attention, with_kwargs=True)
        text_config = self.backbone.config.text_config
        self.context_tokens = int(context_tokens or text_config.max_position_embeddings)
        if not 1 <= self.context_tokens <= text_config.max_position_embeddings:
            raise ValueError("context exceeds the checkpoint's native limit")
        weight_file = ckpt / "readout.safetensors"
        self.weight = load_file(str(weight_file))["weight"].to(self.device, dtype=torch.float32)
        if self.weight.shape != (MAX_OPTIONS, text_config.hidden_size):
            raise ValueError("invalid 255-class readout shape")
        self.provenance = {
            "model": source,
            "revision": resolved_revision,
            "readout_sha256": sha256(weight_file.read_bytes()).hexdigest(),
            "temperatures": self.temperatures,
            "protocol": "single-question-head255-v1",
        }

    def _open_images(self, refs):
        from decision_index.vision.images import load
        from PIL import Image

        images = []
        for ref in refs:
            try:
                with Image.open(BytesIO(load(ref, self.image_root))) as im:
                    images.append(im.convert("RGB"))
            except (OSError, ValueError, TypeError) as exc:
                raise Unsupported("unable to read a supplied image") from exc
        return images

    def _forward(self, row, pictures):
        torch = self.torch
        tokenizer = self.processor.tokenizer
        try:
            rendered = prompt.plain_prompt(tokenizer, messages(row, self.codes))
            if pictures:
                inputs = self.processor(
                    text=[rendered], images=pictures, return_tensors="pt",
                    size={"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels},
                )
                image_processor = self.processor.image_processor
                grids = inputs["image_grid_thw"].tolist()
                pixels = [h * w * int(image_processor.patch_size) ** 2 for _, h, w in grids]
                image_tokens = sum(t * h * w // int(image_processor.merge_size) ** 2 for t, h, w in grids)
                if len(pixels) != len(pictures) or any(p > self.max_pixels for p in pixels):
                    raise ValueError("processed image exceeds the declared pixel budget")
            else:
                ids = tokenizer.encode(rendered, add_special_tokens=False)
                inputs = {"input_ids": torch.tensor([ids], dtype=torch.long),
                          "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
                pixels, image_tokens = [], None
        except ValueError as exc:
            raise Unsupported(str(exc)) from exc
        length = inputs["input_ids"].shape[-1]
        if length > self.context_tokens:
            raise Unsupported(f"expanded prompt has {length} tokens, capacity is {self.context_tokens}")
        args = {key: value.to(self.device) for key, value in inputs.items() if torch.is_tensor(value)}
        hidden = self.backbone(**args, use_cache=False, return_dict=True).last_hidden_state[:, -1]
        with torch.autocast(device_type=self.device.type, enabled=False):
            logits = torch.nn.functional.linear(hidden.float(), self.weight)[0, :len(row["options"])]
        if row["type"] == "noul":
            logits = logits.flip(0)
        temperature = torch.tensor(self.temperatures[row["type"]], dtype=torch.float32, device=self.device)
        probs = (logits / temperature).softmax(-1).cpu().tolist()
        return probs, length, pixels, image_tokens

    def __call__(self, state, questions, images=(), prompt=None):
        if prompt and set(prompt) - {"options_in_image", "readout"}:
            raise Unsupported("unrecognized vision prompt metadata")
        pictures = self._open_images(images or ())
        answers, tokens, pixels, image_tokens = {}, 0, [], None
        with self._lock, self.torch.inference_mode():
            for key, spec in questions.items():
                try:
                    row = render_question(state, spec, len(pictures))
                except ValueError as exc:
                    raise Unsupported(str(exc)) from exc
                odds, length, pixels, image_tokens = self._forward(row, pictures)
                answers[key] = make_answer(row, odds)
                tokens += length
        response = {"model": self.name, "answers": answers,
                    "usage": {"input_tokens": tokens, "output_tokens": 0}}
        return response, {"image_pixels": pixels, "image_tokens": image_tokens} if pictures else None

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def runtime(self):
        import transformers

        return {
            "torch": self.torch.__version__, "transformers": transformers.__version__,
            "device": self.torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else str(self.device),
            "max_options": MAX_OPTIONS, "context_tokens": self.context_tokens,
            "min_image_pixels": self.min_pixels, "max_image_pixels": self.max_pixels,
            "one_forward_per_question": True, "grouping": False, "truncation": False,
        }

    def close(self):
        del self.backbone
        del self.weight
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
