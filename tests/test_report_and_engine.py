import json

import pytest

from decision_index.engines import RandomEngine, validate
from decision_index.scoring.report import score


def rows_for(number, n=3, criteria=None, track="T"):
    criteria = criteria or {"A": "first", "B": "second"}
    out = []
    for i in range(n):
        out.append({"_evaluation": {"run_id": f"{number}:{track}:{i}", "catalog_id": number, "group_id": f"g{i}", "track": track, "dataset": "x"}, "questions": {"q1": {"type": "choice", "instructions": "pick", "criteria": criteria}}, "expected": {"q1": "A"}, "metadata": {}})
    return out


def results_for(rows, choices):
    return {r["_evaluation"]["run_id"]: {"status": "ok", "response": {"answers": {"q1": {"type": "choice", "choice": c, "probabilities": {"A": 0.5, "B": 0.5}}}}, "total_wall_ms": 10.0} for r, c in zip(rows, choices)}


def test_accuracy_report():
    rows = rows_for(24)
    rep = score(rows, results_for(rows, ["A", "B", "A"]))
    assert rep["primary_metric"] == "accuracy" and rep["primary_value"] == pytest.approx(2 / 3)
    assert rep["case_exact_accuracy"] == pytest.approx(2 / 3)


def test_macro_f1_report_uses_semantic_labels():
    rows = rows_for(41)
    rep = score(rows, results_for(rows, ["A", "B", "A"]))
    a = 2 * 2 / (2 * 2 + 0 + 1)
    b = 0.0
    assert rep["primary_metric"] == "macro-F1" and rep["primary_value"] == pytest.approx((a + b) / 2)


def test_partial_results_are_excluded_from_metric_but_counted():
    rows = rows_for(24)
    res = results_for(rows[:2], ["A", "B"])
    rep = score(rows, res)
    assert rep["successful_requests"] == 2 and rep["pending_requests"] == 1
    assert rep["primary_value"] == pytest.approx(0.5)


def test_validate_rejects_bad_distributions():
    q = {"q1": {"type": "choice", "instructions": "", "criteria": {"A": "a", "B": "b"}}}
    validate(q, {"answers": {"q1": {"type": "choice", "choice": "A", "probabilities": {"A": 0.7, "B": 0.3}}}})
    with pytest.raises(ValueError):
        validate(q, {"answers": {"q1": {"type": "choice", "choice": "C", "probabilities": {"A": 0.7, "B": 0.3}}}})
    with pytest.raises(ValueError):
        validate(q, {"answers": {"q1": {"type": "choice", "choice": "A", "probabilities": {"A": 0.7}}}})


def test_random_engine_round_trip():
    e = RandomEngine(seed=1)
    q = {"q1": {"type": "choice", "instructions": "", "criteria": {"A": "a", "B": "b", "C": "c"}}}
    response, _ = e("state", q)
    validate(q, response)
    assert json.dumps(response)
