"""Laws for the wyat-2 adapter on a real, tiny ``wyat-ouro-decision-v2`` artifact.

A random 24-layer Ouro with 16-wide states is written as a v2 artifact once per
run, then loaded through the adapter, wyat's loader and its Ouro scorer on CPU.
The laws pin what the kit and the maintainers rely on:
- every request the kit can send is answered as the model answers it;
- a request beyond the artifact's limit is unsupported, never truncated;
- only a uniform v2 artifact is accepted;
- the run records the artifact and runtime it used.

Needs ``WYAT_OURO_SOURCE``: a directory holding Ouro's ``configuration_ouro.py``
and ``modeling_ouro.py`` (ByteDance/Ouro-1.4B, or wyat's test fixture).

    WYAT_OURO_SOURCE=... tlaw check tests/wyat2_runtime_laws.py
"""

import functools
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import tlaw
import torch
from hypothesis import settings
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast
from wyat.input_rendering import input_rendering_policy
from wyat.loading import ouro_artifact
from wyat.loading.ouro import (
    ScalarScoreHead,
    attach_step_lora,
    load_ouro_classes,
    write_ouro_artifact,
)

from decision_index.engines import Unsupported, validate
from examples.wyat2_engine import Wyat2Engine

SOURCE = Path(os.environ["WYAT_OURO_SOURCE"])
MAX_LENGTH = 1024
CONTEXT_PREFILL = {
    "chunk": 64,
    "single_chunk_max_tokens": 128,
    "chunked_attention": "lean",
    "host_cache_above_tokens": 512,
}

Word = tlaw.one_of(["alpha", "beta", "délai", "日本語", "x", "a longer option text"])
Words = tlaw.list_of(Word, min_size=1, max_size=4)
Question = tlaw.record(
    kind=tlaw.one_of(["choice", "noul"]),
    options=tlaw.integer(2, 5),
    instructions=Words,
    texts=tlaw.list_of(Words, min_size=5, max_size=5),
)
Questions = tlaw.list_of(Question, min_size=1, max_size=2)
Repeats = tlaw.integer(200, 400)
Refused = tlaw.one_of(["v1", "rendered", "unknown-option", "mps"])


def _tokenizer(path: Path) -> Path:
    path.mkdir(parents=True)
    alphabet = ByteLevel.alphabet()
    vocabulary = {symbol: index for index, symbol in enumerate(sorted(alphabet))}
    vocabulary["[PAD]"] = len(vocabulary)
    tokenizer = Tokenizer(BPE(vocabulary, [], unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]"
    ).save_pretrained(path)
    special = path / "special_tokens_map.json"
    if not special.is_file():
        special.write_text(json.dumps({"pad_token": "[PAD]"}) + "\n")
    return path


def _config() -> dict:
    return {
        "architectures": ["OuroForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "head_dim": 4,
        "hidden_act": "silu",
        "hidden_size": 16,
        "initializer_range": 0.02,
        "intermediate_size": 32,
        "layer_types": ["full_attention"] * ouro_artifact.NATIVE_LAYERS,
        "max_position_embeddings": 65536,
        "max_window_layers": ouro_artifact.NATIVE_LAYERS,
        "model_type": "ouro",
        "num_attention_heads": 4,
        "num_hidden_layers": ouro_artifact.NATIVE_LAYERS,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 1000000.0,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "total_ut_steps": 4,
        "early_exit_threshold": 1.0,
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 320,
    }


def _artifacts(root: Path) -> dict[str, Path]:
    """One tiny model written as v2, as v1, and as v2 with an input renderer."""
    source = root / "source"
    source.mkdir()
    for name in ("configuration_ouro.py", "modeling_ouro.py"):
        shutil.copy2(SOURCE / name, source / name)
    (source / "config.json").write_text(json.dumps(_config(), indent=2) + "\n")
    tokenizer = _tokenizer(root / "tokenizer")
    config_cls, model_cls = load_ouro_classes(source)
    config = config_cls(**_config())
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(11)
        backbone = model_cls(config).to(dtype=torch.bfloat16).eval()
        base = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
        attach_step_lora(
            backbone,
            rank=ouro_artifact.ADAPTER_RANK,
            alpha=ouro_artifact.ADAPTER_ALPHA,
            steps=ouro_artifact.CANDIDATE_STEPS,
            targets=ouro_artifact.ADAPTER_TARGETS,
        )
        for name, parameter in backbone.named_parameters():
            if ".lora_B." in name:
                parameter.data.normal_(0.0, 0.05)
        adapters = {
            f"backbone.{name}": parameter.detach().float().clone()
            for name, parameter in backbone.named_parameters()
            if ".lora_" in name
        }
        head = {
            f"head.{k}": v.detach().float().clone()
            for k, v in ScalarScoreHead(16, ouro_artifact.HEAD_WIDTH)
            .state_dict()
            .items()
        }
    common = dict(
        contract_prompt=ouro_artifact.PROMPT,
        source_dir=source,
        tokenizer_dir=tokenizer,
        base_weights=base,
        adapters=adapters,
        head=head,
        max_length=MAX_LENGTH,
    )
    v2 = dict(context_prefill=CONTEXT_PREFILL, numerics="deterministic-no-tf32-v1")
    rendering = input_rendering_policy("tool-decision-schema-inline-v5")
    paths = {"v2": root / "v2", "v1": root / "v1", "rendered": root / "rendered"}
    write_ouro_artifact(paths["v2"], **common, **v2)
    write_ouro_artifact(paths["v1"], **common)
    write_ouro_artifact(paths["rendered"], **common, **v2, input_rendering=rendering)
    return paths


_WORKSPACE = tempfile.TemporaryDirectory(prefix="wyat2-laws-")
ARTIFACTS = _artifacts(Path(_WORKSPACE.name))


@functools.cache
def engine() -> Wyat2Engine:
    """Built inside the first law that needs it, so a broken constructor
    fails a law rather than the module import."""
    return Wyat2Engine(model=str(ARTIFACTS["v2"]), device="cpu")


def _questions(drawn):
    questions = {}
    for index, item in enumerate(drawn):
        question = {
            "type": item["kind"],
            "instructions": " ".join(item["instructions"]),
        }
        if item["kind"] == "choice":
            question["criteria"] = {
                f"o{j}": " ".join(item["texts"][j]) for j in range(item["options"])
            }
        else:
            question["criteria"] = {"false": "No.", "true": "Yes."}
        questions[f"q{index}"] = question
    return questions


def _direct(state, questions):
    native = engine().model.predict({"state": state, "questions": questions})
    return {
        key: (
            {
                "type": "choice",
                "choice": native[key].choice,
                "probabilities": native[key].probabilities_by_id,
            }
            if question["type"] == "choice"
            else {"type": "noul", "noul": native[key].probability_true}
        )
        for key, question in questions.items()
    }


# Each example scores a real model twice (about 0.2 s per call on CPU), and
# tlaw's command line gives one law file 60 seconds: 25 examples, no deadline.
@settings(max_examples=25, deadline=None)
@tlaw.law(
    "every request the kit sends is answered as the model answers it",
    drawn=Questions,
    state=Words,
)
def every_request_the_kit_sends_is_answered_as_the_model_answers_it(drawn, state):
    questions = _questions(drawn)
    response, raw = engine()(" ".join(state), questions)
    validate(questions, response)
    return tlaw.same(
        [response["answers"], response["model"], raw],
        [_direct(" ".join(state), questions), engine().model.model_id, None],
    )


@tlaw.law
def a_request_beyond_the_artifact_limit_is_unsupported_not_truncated(
    drawn: Questions, repeats: Repeats
):
    state = "too long " * repeats
    return tlaw.raises(Unsupported, engine(), state, _questions(drawn)).and_then(
        lambda error: tlaw.same(str(error), "wyat-2 declared capacity: input-length")
    )


def _construct(**options) -> str:
    """A stable return value: an engine's repr holds its address, so a failure
    naming it would read differently when tlaw replays the kill."""
    Wyat2Engine(**options)
    return "loaded"


@tlaw.law
def only_a_uniform_v2_artifact_on_cpu_or_cuda_loads(refused: Refused):
    artifact = {"v1": "v1", "rendered": "rendered"}.get(refused, "v2")
    options = {
        "model": str(ARTIFACTS[artifact]),
        "device": "mps" if refused == "mps" else "cpu",
    }
    if refused == "unknown-option":
        options["permutations"] = 4
    expected, message = {
        "v1": (ValueError, "wyat-2 needs a wyat-ouro-decision-v2 artifact"),
        "rendered": (ValueError, "Official evaluation requires uniform inputs"),
        "unknown-option": (TypeError, "Unknown wyat-2 engine options: permutations"),
        "mps": (ValueError, "wyat's Ouro runtime here supports cpu or cuda"),
    }[refused]
    return tlaw.raises(expected, _construct, **options).and_then(
        lambda error: tlaw.same(str(error).startswith(message), True)
    )


Recorded = tlaw.one_of(
    [
        "manifest",
        "model",
        "format",
        "prefill",
        "numerics",
        "limit",
        "runtime",
        "options",
    ]
)


@tlaw.law
def each_recorded_field_matches_the_artifact_and_runtime_used(field: Recorded):
    manifest_path = ARTIFACTS["v2"] / ouro_artifact.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    provenance = engine().provenance
    runtime = Path(ouro_artifact.__file__).parents[1]
    recorded, expected = {
        "manifest": (
            provenance["manifest_sha256"],
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        ),
        "model": (provenance["model_id"], manifest["model_id"]),
        "format": (provenance["artifact"]["format"], ouro_artifact.FORMAT_V2),
        "prefill": (provenance["settings"]["context_prefill"], CONTEXT_PREFILL),
        "numerics": (provenance["settings"]["numerics"], "deterministic-no-tf32-v1"),
        "limit": (provenance["capacity_limits"]["max_sequence_tokens"], MAX_LENGTH),
        "options": (
            engine().options,
            {"model": str(ARTIFACTS["v2"]), "device": "cpu"},
        ),
        "runtime": (
            provenance["runtime_files_sha256"]["loading/ouro.py"],
            hashlib.sha256((runtime / "loading/ouro.py").read_bytes()).hexdigest(),
        ),
    }[field]
    return tlaw.same(recorded, expected)
