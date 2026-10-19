import os

import pytest

from decision_index.engines import validate

pytest.importorskip("torch")
pytest.importorskip("kieve")

CKPT = os.environ.get("KIEVE_CHECKPOINT")
CFG = os.environ.get("KIEVE_CONFIG")

pytestmark = pytest.mark.skipif(not (CKPT and CFG), reason="set KIEVE_CHECKPOINT and KIEVE_CONFIG")


@pytest.fixture(scope="module")
def engine():
    from decision_index.engines.kieve import KieveEngine

    return KieveEngine(checkpoint=CKPT, config=CFG,
                       device=os.environ.get("KIEVE_DEVICE", "cpu"))


def test_choice_round_trip(engine):
    q = {"q1": {"type": "choice", "instructions": "Which color is named?",
                "criteria": {"A": "red", "B": "blue", "C": "green"}}}
    response, raw = engine("The color is red.", q)
    validate(q, response)
    a = response["answers"]["q1"]
    assert a["type"] == "choice" and a["choice"] in q["q1"]["criteria"]
    assert set(a["probabilities"]) == {"A", "B", "C"}
    assert abs(sum(a["probabilities"].values()) - 1) < 0.01


def test_noul_round_trip(engine):
    q = {"q1": {"type": "noul", "instructions": "Is the state consistent?"}}
    response, _ = engine("A perfectly ordinary sentence.", q)
    validate(q, response)
    a = response["answers"]["q1"]
    assert a["type"] == "noul" and 0.0 <= a["noul"] <= 1.0
