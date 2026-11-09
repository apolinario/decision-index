import os

import pytest

from decision_index.engines import REGISTRY, Unsupported, validate
from decision_index.engines.gliclass_engine import GLiClassEngine, option_labels, render_state
from decision_index.hf_job import job_command


def test_registered():
    assert REGISTRY["gliclass"] == "decision_index.engines.gliclass_engine:GLiClassEngine"


def test_render_state():
    assert render_state(None) == ""
    assert render_state("plain text") == "plain text"
    assert render_state(["a", {"b": 1}]) == '- a\n- {"b":1}'
    state = {"scenario": "Checkout errors at 12%.", "facts": ["Rollback takes 2 min.", "Deploy was 10 min ago."], "n": 3}
    assert render_state(state) == "scenario: Checkout errors at 12%.\nfacts:\n- Rollback takes 2 min.\n- Deploy was 10 min ago.\nn: 3"


def test_option_labels():
    assert option_labels({"a": "Roll back", "b": None}) == {"a": "Roll back", "b": "b"}
    assert option_labels({"a": "Same", "b": "Same"}) == {"a": "a: Same", "b": "b: Same"}


def test_distribution():
    engine = GLiClassEngine.__new__(GLiClassEngine)
    labels = {"a": "A", "b": "B"}
    engine.probs = "normalize"
    assert engine._distribution(labels, {"A": 0.6, "B": 0.2}) == pytest.approx({"a": 0.75, "b": 0.25})
    assert engine._distribution(labels, {"A": 0.0, "B": 0.0}) == {"a": 0.5, "b": 0.5}
    engine.probs = "softmax"
    assert engine._distribution(labels, {"A": 0.5, "B": 0.5}) == pytest.approx({"a": 0.5, "b": 0.5})


def test_hf_job_installs_gliclass_extra():
    assert "[transformers,gliclass]" in job_command("me/results", "gliclass", [], "me/suite", "run", [])
    assert "[transformers]'" in job_command("me/results", "transformers", [], "me/suite", "run", [])


@pytest.mark.skipif(not os.environ.get("DECISION_INDEX_GLICLASS_MODEL"), reason="set DECISION_INDEX_GLICLASS_MODEL to run a real model")
def test_real_model_round_trip():
    pytest.importorskip("gliclass")
    from decision_index.engines import load_engine

    model = os.environ["DECISION_INDEX_GLICLASS_MODEL"]
    questions = {
        "action": {"type": "choice", "instructions": "What should we do right now?", "criteria": {"rollback": "Roll back to the last healthy build", "wait": "Watch errors for another hour"}},
        "urgent": {"type": "noul", "instructions": "Is this urgent?", "criteria": {"true": "It is urgent.", "false": "It is not urgent."}},
        "plain": {"type": "noul", "instructions": "Did errors rise after the deploy?"},
    }
    state = {"scenario": "Checkout errors hit 12% after a deploy.", "facts": ["Rolling back takes two minutes."]}
    engine = load_engine("gliclass", model=model, max_length=4096)
    response, raw = engine(state, questions)
    validate(questions, response)
    assert set(raw) == set(questions)
    single, _ = load_engine("gliclass", model=model, max_length=4096, batch_size=1)(state, questions)
    for k in questions:  # batched and one-at-a-time passes agree
        a, b = response["answers"][k], single["answers"][k]
        if a["type"] == "choice":
            assert a["choice"] == b["choice"]
            assert all(abs(a["probabilities"][o] - b["probabilities"][o]) < 1e-2 for o in a["probabilities"])
        else:
            assert abs(a["noul"] - b["noul"]) < 1e-2
    with pytest.raises(Unsupported):
        load_engine("gliclass", model=model, max_length=8)("a long enough state " * 20, questions)
