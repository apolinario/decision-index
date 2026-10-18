import json
import sys
from types import SimpleNamespace

import pytest

from decision_index.engines import Unsupported, validate
from examples.wyat_engine import WyatEngine


class NativeInputError(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__("private input must not appear in a capacity log")


def engine_with(predict):
    engine = WyatEngine.__new__(WyatEngine)
    engine.model = SimpleNamespace(model_id="native-fixture", predict=predict)
    engine.wyat = SimpleNamespace(ModelInputError=NativeInputError)
    return engine


def test_preserves_complete_native_record_and_typed_probabilities():
    state = {"dialogue": [{"user": "A fictional support question"}]}
    questions = {
        "keep-this-id": {
            "type": "choice",
            "instructions": {"instruction": "Choose"},
            "criteria": {"first": "same text", "second": "same text"},
        },
        "truth-id": {"type": "noul", "instructions": "Is the request urgent?"},
    }
    received = []

    def predict(record):
        received.append(record)
        return {
            "keep-this-id": SimpleNamespace(
                choice="second", probabilities_by_id={"first": 0.2, "second": 0.8}
            ),
            "truth-id": SimpleNamespace(probability_true=0.37),
        }

    response, raw = engine_with(predict)(state, questions)
    assert received == [{"state": state, "questions": questions}]
    assert received[0]["state"] is state
    assert received[0]["questions"] is questions
    validate(questions, response)
    assert response["answers"]["keep-this-id"]["probabilities"] == {
        "first": 0.2,
        "second": 0.8,
    }
    assert response["answers"]["truth-id"] == {"type": "noul", "noul": 0.37}
    assert raw is None


@pytest.mark.parametrize(
    "reason", ["input-length", "candidate-count", "unsupported-task"]
)
def test_declared_capacity_is_unsupported_and_does_not_leak_input(reason):
    def predict(record):
        raise NativeInputError(reason)

    engine = engine_with(predict)
    with pytest.raises(Unsupported, match="^Wyat declared capacity: " + reason + "$"):
        engine("state", {"q": {"type": "noul"}})


@pytest.mark.parametrize(
    "error", [NativeInputError("input-segment"), RuntimeError("CUDA failure")]
)
def test_malformed_inputs_and_device_failures_remain_errors(error):
    def predict(record):
        raise error

    with pytest.raises(type(error)) as raised:
        engine_with(predict)("state", {"q": {"type": "noul"}})
    assert raised.value is error


@pytest.mark.parametrize("count", [1, 256])
def test_never_drops_options_to_fit_capacity(count):
    called = []
    engine = engine_with(lambda record: called.append(record))
    with pytest.raises(Unsupported, match="2..255"):
        engine(
            "state",
            {
                "q": {
                    "type": "choice",
                    "criteria": {str(i): "description" for i in range(count)},
                }
            },
        )
    assert called == []


def test_accepts_all_255_options():
    criteria = {str(i): "description" for i in range(255)}
    probabilities = {key: 1 / 255 for key in criteria}
    questions = {"q": {"type": "choice", "criteria": criteria}}
    engine = engine_with(
        lambda record: {
            "q": SimpleNamespace(choice="0", probabilities_by_id=probabilities),
        }
    )
    response, _ = engine({}, questions)
    validate(questions, response)
    assert set(response["answers"]["q"]["probabilities"]) == set(criteria)


def test_active_task_specific_renderer_is_rejected_before_model_load(
    tmp_path, monkeypatch
):
    (tmp_path / "fork_manifest.json").write_text(
        json.dumps(
            {
                "input_rendering": {"mode": "api-bank-schema-next-v1"},
            }
        )
    )
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            device=lambda value: SimpleNamespace(type="cuda"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "wyat",
        SimpleNamespace(
            load=lambda *args, **kw: calls.append((args, kw)),
        ),
    )
    with pytest.raises(ValueError, match="uniform inputs"):
        WyatEngine(str(tmp_path))
    assert calls == []
