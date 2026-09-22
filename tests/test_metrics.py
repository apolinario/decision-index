import math

import pytest

from decision_index.scoring import metrics as M


def test_conservative_f1_missing_rule():
    golds = ["a", "b", "a", "b"]
    preds = ["a", "b", None, "a"]
    value = M.conservative_f1(golds, preds, ["a", "b"])
    a = 2 * 1 / (2 * 1 + 1 + 0 + 1)
    b = 2 * 1 / (2 * 1 + 0 + 1 + 1)
    assert value == pytest.approx((a + b) / 2)


def test_conservative_f1_positive_only():
    assert M.conservative_f1(["yes", "no", "yes"], ["yes", "yes", None], ["no", "yes"], "yes") == pytest.approx(2 * 1 / (2 * 1 + 1 + 0 + 1))


def test_macro_f1_uses_observed_label_union():
    pairs = [("a", "a"), ("b", "c")]
    a = 1.0
    b = 0.0
    c = 0.0
    assert M.macro_f1(pairs) == pytest.approx((a + b + c) / 3)


def test_binary_f1():
    assert M.binary_f1([("yes", "yes"), ("no", "yes"), ("yes", "no")]) == pytest.approx(0.5)


def test_score_query_ndcg():
    record = {"scorable_ids": ["d1", "d2", "d3"], "qrels": {"d2": 1}, "retrieved_ids": ["d1", "d2", "d3"], "candidate_recall": 1.0}
    v = M.score_query(record, {"d1": 0.9, "d2": 0.5, "d3": 0.1})
    assert v["valid"] and v["ndcg_at_10"] == pytest.approx(1 / math.log2(3))
    assert v["mrr"] == pytest.approx(0.5)
    assert not M.score_query(record, {"d1": 0.9})["valid"]
    assert M.random_ndcg_baseline(record) == pytest.approx((1 / 3) * sum(1 / math.log2(i + 2) for i in range(3)))


def test_router_chess_consensus_forecast():
    row = {"scoring": {"outcomes": {"a": {"quality": 1.0, "cost_usd": 0.5}, "b": {"quality": 0.6, "cost_usd": 0.01}}, "cost_penalty_per_usd": 10.0}}
    assert M.router_score(row, "a")["oracle_optimal"]
    assert M.router_score(row, "a", "cost_aware")["utility_regret"] == pytest.approx((0.6 - 0.1) - (1.0 - 5.0))
    chess = {"scoring": {"values": {"e2e4": 0.6, "d2d4": 0.6, "a2a3": 0.1}, "accepted": ["d2d4", "e2e4"]}}
    assert M.chess_score(chess, "d2d4")["best_move"] and M.chess_score(chess, "a2a3")["value_regret"] == pytest.approx(0.5)
    cons = {"scoring": {"summed_ranks": {"o0": 3, "o1": 1, "o2": 5}, "accepted": ["o1"], "panel_size": 3}}
    assert M.consensus_score(cons, "o2")["normalized_rank_regret"] == pytest.approx(4 / 6)
    fc = {"expected": {"answer": "yes"}}
    assert M.forecast_score(fc, 0.8)["brier"] == pytest.approx(0.04)
    assert M.forecast_score(fc, 0.3)["correct"] is False


def test_skill_clip():
    assert M.skill(0.5, 0.25) == pytest.approx(1 / 3)
    assert M.skill(0.1, 0.25) == 0.0
    assert M.skill(1.5, 0.25) == 1.0
