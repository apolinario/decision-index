import math
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

pytest.importorskip("torch")

# Ensure mock gevva module exists even in CI environments where gevva is not yet installed
mock_gevva = MagicMock()
sys.modules.setdefault("gevva", mock_gevva)

from decision_index.engines import load_engine, validate
from decision_index.engines.gevva_engine import GevvaEngine, _softmax


def test_softmax_helper():
    probs = _softmax([1.0, 2.0, 3.0])
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-5
    assert probs[2] > probs[1] > probs[0]


@patch("gevva.load")
def test_gevva_engine_registry(mock_load):
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    engine = load_engine("gevva", model="davidburhans/gevva-e4b", device="cpu")
    assert isinstance(engine, GevvaEngine)
    assert engine.name == "gevva"
    assert engine.model_id == "davidburhans/gevva-e4b"


@patch("gevva.load")
def test_gevva_engine_call_and_validate(mock_load):
    mock_model = MagicMock()
    mock_load.return_value = mock_model

    # Option scoring mocks
    # predict returns np.ndarray of shape (N, 3) [p_con, p_ent, p_neu]
    mock_model.predict.side_effect = [
        # Choice question (3 options):
        np.array([
            [0.2, 0.7, 0.1],  # A (high entailment)
            [0.8, 0.1, 0.1],  # B (high contradiction)
            [0.1, 0.1, 0.8],  # C (neutral)
        ]),
        # Noul question (1 pair):
        np.array([
            [0.1, 0.9, 0.0],  # true statement
        ]),
    ]

    engine = GevvaEngine(model="davidburhans/gevva-e4b", device="cpu")

    questions = {
        "q1": {
            "type": "choice",
            "instructions": "Which city is the capital of Australia?",
            "criteria": {
                "A": "Canberra",
                "B": "Sydney",
                "C": "Melbourne",
            },
        },
        "q2": {
            "type": "noul",
            "instructions": "Canberra is the federal capital of Australia.",
            "criteria": {
                "false": "False",
                "true": "True",
            },
        },
    }

    response, raw = engine("Australia context.", questions)

    # Must pass official Decision Index validator
    validate(questions, response)

    # Check choice response
    q1_ans = response["answers"]["q1"]
    assert q1_ans["type"] == "choice"
    assert q1_ans["choice"] == "A"
    assert set(q1_ans["probabilities"]) == {"A", "B", "C"}
    assert abs(sum(q1_ans["probabilities"].values()) - 1.0) < 0.01
    assert q1_ans["probabilities"]["A"] > q1_ans["probabilities"]["B"]

    # Check noul response
    q2_ans = response["answers"]["q2"]
    assert q2_ans["type"] == "noul"
    assert 0.0 <= q2_ans["noul"] <= 1.0
    assert q2_ans["noul"] > 0.5  # Supported by high entailment

    # Check raw diagnostics
    assert "q1" in raw
    assert "q2" in raw


@patch("gevva.load")
def test_gevva_warmup_and_runtime(mock_load):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1, 0.9, 0.0], [0.9, 0.1, 0.0]])
    mock_load.return_value = mock_model

    engine = GevvaEngine(model="davidburhans/gevva-e2b", device="cpu")
    engine.warmup()
    engine.synchronize()

    rt = engine.runtime()
    assert "torch" in rt
    assert "device" in rt
    assert rt["model"] == "davidburhans/gevva-e2b"
