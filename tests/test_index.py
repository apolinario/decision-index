import math

import pytest

from decision_index import constants as C
from decision_index.scoring import index as I


def make(n, raw, random, coverage=1.0, tracks=None):
    tracks = tracks or [dict(track="all", raw=raw, random=random, skill=I.skill(raw, random), planned_cases=10, coverage=coverage, pending=0.0, label_universe=[])]
    return dict(catalog_id=n, raw=sum(t["raw"] for t in tracks) / len(tracks), skill=sum(t["skill"] for t in tracks) / len(tracks), coverage=sum(t["coverage"] for t in tracks) / len(tracks), pending=0.0, tracks=tracks)


def panel_scored(value=0.6, random=0.2):
    scored = {n: make(n, value, random) for n in I.panel_ids() if n not in C.INTERACTIVE}
    for n in C.INTERACTIVE:
        scored[n] = I.interactive_placeholder(n)
    return scored


def test_area_membership():
    areas = {a["id"]: a["ids"] for a in I.area_ids()}
    assert areas == {"knowledge": [24, 25, 30, 43, 44, 31], "language": [11, 40, 41], "retrieval": [36, 37], "tools": [1, 2, 6], "arts": [20, 21, 22, 23, 50]}
    assert sum(len(v) for v in areas.values()) == 19


def test_uniform_skill_reproduces_proposal_examples():
    scored = panel_scored(0.6, 0.0)
    scores, cats = I.recompute(scored)
    assert scores["balanced_skill"] == pytest.approx(60.0)
    assert scores["breadth_skill"] == pytest.approx(60.0)
    assert len(cats) == 5


def test_uneven_breadth_matches_proposal():
    values = [0.9, 0.9, 0.9, 0.2, 0.1]
    breadth = 100 * (math.prod((0.1 + 0.9 * x) ** 0.2 for x in values) - 0.1) / 0.9
    assert breadth == pytest.approx(47.28240846043158)


def test_coverage_adjustment_unanswered_counts_as_wrong():
    scored = panel_scored(0.6, 0.2)
    scored[24] = make(24, 0.3, 0.25, coverage=0.5)
    scores, cats = I.recompute(scored)
    knowledge = next(c for c in cats if c["id"] == "knowledge")
    assert knowledge["coverage"] == pytest.approx((0.5 + 5) / 6)
    assert scores["balanced_skill"] < 50.0


def test_headline_track_replaces_isarcasm():
    scored = panel_scored(0.6, 0.2)
    tracks = [dict(track="iSarcasmEval-A-Ar", raw=0.2, random=0.2, skill=0.0, planned_cases=1, coverage=1.0, pending=0.0, label_universe=[]), dict(track="iSarcasmEval-A-En", raw=0.7, random=0.2227, skill=I.skill(0.7, 0.2227), planned_cases=1, coverage=1.0, pending=0.0, label_universe=[]), dict(track="iSarcasmEval-C-En", raw=0.9, random=0.5, skill=0.8, planned_cases=1, coverage=1.0, pending=0.0, label_universe=[])]
    scored[40] = make(40, 0.0, 0.0, tracks=tracks)
    entry = I.index_entry(scored)
    b = entry["benchmarks"]["40"]
    assert b["raw"] == pytest.approx(0.7)
    assert b["random"] == pytest.approx(0.2227)
    assert b["skill"] == pytest.approx(round((0.7 - 0.2227) / (1 - 0.2227), 4))
    assert [t["track"] for t in b["tracks"]] == ["A · Arabic", "A · English", "C · English pairs"]
    assert [t["headline"] for t in b["tracks"]] == [False, True, False]


def test_interactive_dropped_from_index_but_zero_in_frozen_panel():
    scored = panel_scored(1.0, 0.0)
    entry = I.index_entry(scored)
    assert entry["index"] == pytest.approx(100.0)
    frozen = entry["frozen_panel"]["scores"]["balanced_skill"]
    assert frozen == pytest.approx(100 * (1 + 1 + 1 + 3 / 5 + 1 / 5) / 5, abs=0.01)
    assert all(not entry["benchmarks"][str(n)]["in_index"] for n in C.INTERACTIVE)


def test_static_score_accuracy_with_unanswered_rows():
    rows = []
    res = {}
    for i in range(4):
        rid = f"44:CLadder:CLadder:balanced:{i}"
        rows.append({"_evaluation": {"run_id": rid, "catalog_id": 44, "group_id": f"g{i}", "track": "CLadder"}, "questions": {"q1": {"type": "choice", "instructions": "", "criteria": {"A": "yes", "B": "no"}}}, "expected": {"q1": "A"}})
        if i < 3:
            res[rid] = {"status": "ok", "answers": {"q1": {"type": "choice", "choice": "A" if i < 2 else "B", "probabilities": {"A": 0.5, "B": 0.5}}}}
    res["44:CLadder:CLadder:balanced:3"] = {"status": "unsupported", "answers": {}}
    out = I.static_score(44, rows, res, I.chance_baselines())
    assert out["raw"] == pytest.approx(0.5)
    assert out["coverage"] == pytest.approx(0.75)
    assert out["tracks"][0]["random"] == pytest.approx(0.5)
    assert out["skill"] == 0.0


def test_static_score_conservative_f1_track():
    rows = []
    res = {}
    for i, gold in enumerate(["yes", "no", "yes"]):
        rid = f"40:iSarcasmEval-A-En:iSarcasmEval-A-En:test:{i}"
        rows.append({"_evaluation": {"run_id": rid, "catalog_id": 40, "group_id": f"g{i}", "track": "iSarcasmEval-A-En"}, "questions": {"sarcastic": {"type": "choice", "instructions": "", "criteria": {"no": "No", "yes": "Yes"}}}, "expected": {"sarcastic": gold}})
        if i < 2:
            res[rid] = {"status": "ok", "answers": {"sarcastic": {"type": "choice", "choice": "yes", "probabilities": {"yes": 1.0, "no": 0.0}}}}
    out = I.static_score(40, rows, res, I.chance_baselines())
    t = out["tracks"][0]
    assert t["track"] == "iSarcasmEval-A-En"
    assert t["raw"] == pytest.approx(2 * 1 / (2 * 1 + 1 + 0 + 1))
    assert t["random"] == pytest.approx(0.2226862339379383)
    assert t["coverage"] == pytest.approx(2 / 3)
