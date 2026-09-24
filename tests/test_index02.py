import json
from pathlib import Path

import pytest

from decision_index import editions
from decision_index.scoring import added as A
from decision_index.scoring import index02 as X

BOARD = json.loads((Path(__file__).parent / "fixtures/board-0.2.json").read_text())
LIVE = {"jev": 51.67, "autojev-27b": 50.94, "hopper": 30.01, "verdict-8af2496e": 1.82}


def values_from_board(entrant):
    s = X.spec()
    out = {}
    for n, b in entrant["benchmarks"].items():
        n = int(n)
        if n in s["track_scored"]:
            out[n] = X.benchmark_value(n, s, track={"catalog_id": n, "raw": b["board"]["raw"], "skill": b["board"]["skill"], "coverage": b["board"]["coverage"], "tracks": []})
        else:
            out[n] = X.benchmark_value(n, s, native=b["result"])
    return out


def test_panel_matches_live_definition():
    s = X.spec()
    ids = [n for a in s["areas"] for n in a["benchmarks"]]
    assert len(ids) == len(set(ids)) == 40
    assert [a["id"] for a in s["areas"]] == ["knowledge", "language", "retrieval", "tools", "arts"]
    assert sorted(int(n) for n in s["added"]) == [56, 57, 58, 59, 61, 62, 64]
    assert [x["id"] for x in s["not_in_index"]] == [24, 26, 27, 34]
    assert not {24, 26, 27, 34} & set(ids)
    assert s["loss_rules"]["48"]["baseline"] == 0.25


@pytest.mark.parametrize("engine", sorted(LIVE))
def test_reproduces_live_board_index(engine):
    entrant = BOARD["entrants"][engine]
    values = values_from_board(entrant)
    scores, areas = X.aggregate(values)
    assert scores["balanced_skill"] == pytest.approx(LIVE[engine], abs=0.01)
    assert scores["balanced_skill"] == pytest.approx(entrant["scores"]["balanced_skill"], abs=0.01)
    assert scores["balanced_raw"] == pytest.approx(entrant["scores"]["balanced_raw"], abs=0.01)
    assert scores["breadth_skill"] == pytest.approx(entrant["scores"]["breadth_skill"], abs=0.01)
    for n, v in values.items():
        b = entrant["benchmarks"][str(n)]["board"]
        assert v["skill"] == pytest.approx(b["skill"], abs=1.5e-4), n
        assert v["raw"] == pytest.approx(b["raw"], abs=1.5e-4), n


def test_board_order_and_ties():
    r = X.ranks({k: BOARD["entrants"][k]["scores"]["balanced_skill"] for k in LIVE})
    assert [k for k, _ in sorted(r.items(), key=lambda kv: kv[1]["rank"])] == ["jev", "autojev-27b", "hopper", "verdict-8af2496e"]
    t = X.ranks({"a": 30.0, "b": 29.8, "c": 29.6, "d": 29.0})
    assert t["a"] == t["b"] == t["c"] == {"rank": 1, "tied": True}
    assert t["d"] == {"rank": 4, "tied": False}


def test_chance_correction_and_coverage():
    v = X.display_value(0.8, 50, 100, 0.5)
    assert v["raw"] == pytest.approx(0.4)
    assert v["skill"] == 0.0
    v = X.display_value(0.9, 100, 100, 0.25)
    assert v["skill"] == pytest.approx((0.9 - 0.25) / 0.75)
    assert X.display_value(None, 0, 100, 0.25)["skill"] == 0.0


def test_forecastbench_against_baseline():
    rule = X.spec()["loss_rules"]["48"]
    assert X.loss_value(0.25, 10, 10, rule)["skill"] == 0.0
    assert X.loss_value(0.0, 10, 10, rule)["skill"] == 1.0
    assert X.loss_value(0.2, 5, 10, rule)["skill"] == pytest.approx((0.25 - 0.2) / 0.25 * 0.5)
    assert X.loss_value(0.4, 10, 10, rule)["skill"] == 0.0


def rows(n, golds, criteria=None, qtype="choice"):
    out = []
    for i, g in enumerate(golds):
        q = {"type": qtype, "instructions": "", "criteria": criteria or {"A": "a", "B": "b"}}
        out.append({"_evaluation": {"run_id": f"x:{n}:{i}", "catalog_id": n, "dataset": "x", "track": "t", "group_id": f"g{i}"}, "questions": {"q": q}, "expected": {"q": g}, "scoring": {}})
    return out


def test_added_accuracy_and_coverage():
    rr = rows(57, ["A", "B", "A", "B"], {"A": "a", "B": "b", "C": "c", "D": "d"})
    res = {rr[0]["_evaluation"]["run_id"]: {"status": "ok", "response": {"answers": {"q": {"choice": "A", "probabilities": {"A": 1.0, "B": 0, "C": 0, "D": 0}}}}},
           rr[1]["_evaluation"]["run_id"]: {"status": "ok", "response": {"answers": {"q": {"choice": "A", "probabilities": {"A": 1.0, "B": 0, "C": 0, "D": 0}}}}},
           rr[2]["_evaluation"]["run_id"]: {"status": "unsupported"}}
    rep = A.report(57, rr, res)
    assert rep["score"] == pytest.approx(0.5)
    assert (rep["answered"], rep["requests"], rep["pending"]) == (2, 4, 1)
    assert rep["chance"] == pytest.approx(0.25)
    v = X.added_value(rep["score"], rep["answered"], rep["requests"], 0.25)
    assert v["raw"] == pytest.approx(0.25)
    assert v["skill"] == 0.0


def test_ragtruth_f1_and_chance():
    rr = rows(59, [True, False, True, False], {"true": "t", "false": "f"}, qtype="noul")
    preds = [0.9, 0.8, 0.1, 0.2]
    res = {r["_evaluation"]["run_id"]: {"status": "ok", "response": {"answers": {"q": {"type": "noul", "noul": p}}}} for r, p in zip(rr, preds)}
    rep = A.report(59, rr, res)
    assert rep["metric"] == "F1 on hallucinated class"
    assert rep["score"] == pytest.approx(2 * 1 / (2 * 1 + 1 + 1))
    assert rep["chance"] == pytest.approx(0.5 / (0.5 + 0.5))
    assert X.spec()["chance"]["59"]["chance"] == pytest.approx(round(943 / 2700 / (943 / 2700 + 0.5), 4))


def test_phishnchips_scores_verdict_only():
    row = {"_evaluation": {"run_id": "calibration-v1:56:1", "catalog_id": 56, "dataset": "x", "track": "t", "group_id": "g"}, "questions": {"verdict": {"type": "choice", "instructions": "", "criteria": {"phishing": "p", "legitimate": "l"}}, "is_phishing": {"type": "noul", "instructions": "", "criteria": {}}, "sig_urgency": {"type": "noul", "instructions": "", "criteria": {}}}, "expected": {"verdict": "phishing", "is_phishing": True, "sig_urgency": None}, "scoring": {"primary_field": "verdict", "unscored_fields": ["sig_urgency"]}}
    res = {"calibration-v1:56:1": {"status": "ok", "response": {"answers": {"verdict": {"choice": "phishing", "probabilities": {"phishing": 0.7, "legitimate": 0.3}}, "is_phishing": {"noul": 0.1}, "sig_urgency": {"noul": 0.5}}}}}
    rep = A.report(56, [row], res)
    assert rep["scored_fields"] == 1 and rep["score"] == 1.0


def test_editions():
    assert editions.get("0.2")["rows_sha256"].startswith("b2b56d6f")
    assert editions.get("0.2")["rows_gz_sha256"].startswith("25aac5e8")
    assert editions.get("release-v1")["id"] == "0.1"
    e = editions.get("0.2")
    assert e["requests"] == 121057 and e["scoreable"] == 120615 and e["added_requests"] == 30419
    catalog, keep = editions.scoring_subset("0.2")
    assert catalog == 38 and len(keep) == 1565
    assert editions.scoring_subset("0.1") is None
    subsets = editions.retrieval_subsets()["subsets"]
    assert {k: len(v["group_ids"]) for k, v in subsets.items()} == {"ToolRet": 1000, "BRIGHT": 550}
