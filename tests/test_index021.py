import gzip
import json
from pathlib import Path

import pytest

from decision_index import editions
from decision_index.scoring import index02 as X
from decision_index.scoring.report import benchmark_summary, score
from decision_index.suite.io import Suite

BOARD = json.loads((Path(__file__).parent / "fixtures/board-0.2.1.json").read_text())
LIVE = {"jev": 57.89, "rune-26b-a4b-v3": 57.44, "autojev-27b": 56.4, "jebadiah-27b": 54.67, "reflex-27b-v2": 52.16, "jpt-9b": 46.89, "hopper": 39.67, "lev": 38.54, "pngwn-space-uncapped-full": 29.91, "jpt-0.8b": 19.22, "lavoir": 8.69, "verdict-8af2496e": 1.87, "lumma-fev-0.1b": 1.78}
GOLD = [1, 3, 4, 5, 12, 25, 28, 29, 36, 45, 48, 57, 58]


def spec():
    return X.spec("0.2.1")


def values_from_board(entrant):
    s = spec()
    out = {}
    for n, b in entrant["benchmarks"].items():
        n = int(n)
        if n in s["track_scored"]:
            out[n] = X.benchmark_value(n, s, track={"catalog_id": n, "raw": b["board"]["raw"], "skill": b["board"]["skill"], "coverage": b["board"]["coverage"], "tracks": []})
        else:
            out[n] = X.benchmark_value(n, s, native=b["result"])
    return out


def test_panel_matches_live_definition():
    s = spec()
    ids = [n for a in s["areas"] for n in a["benchmarks"]]
    assert len(ids) == len(set(ids)) == 38
    assert [a["id"] for a in s["areas"]] == ["knowledge", "language", "retrieval", "tools", "arts"]
    assert not {6, 10} & set(ids)
    assert [x["id"] for x in s["not_in_index"]] == [6, 10, 24, 26, 27, 34]
    assert sorted(int(n) for n in s["gold"]) == GOLD and set(s["gold"].values()) == {1.2}
    assert 30 not in {int(n) for n in s["gold"]}
    assert s["metrics"]["38"]["key"] == "review_f1"
    assert s["chance"]["59"]["chance"] == 0.5177


def test_area_weights():
    w = X.area_weights(spec())
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["arts"] == 0.1
    assert {k: round(v, 6) for k, v in w.items()} == {"knowledge": 0.258494, "language": 0.258494, "retrieval": 0.200229, "tools": 0.182783, "arts": 0.1}
    assert w["knowledge"] / w["tools"] == pytest.approx((10 / 5) ** 0.5)
    assert X.area_weights(X.spec("0.2")) is None


@pytest.mark.parametrize("engine", sorted(LIVE))
def test_reproduces_live_board_index(engine):
    entrant = BOARD["entrants"][engine]
    values = values_from_board(entrant)
    scores, areas = X.aggregate(values, spec())
    assert scores["balanced_skill"] == pytest.approx(LIVE[engine], abs=0.01)
    assert scores["balanced_skill"] == pytest.approx(entrant["scores"]["balanced_skill"], abs=0.01)
    assert scores["balanced_raw"] == pytest.approx(entrant["scores"]["balanced_raw"], abs=0.01)
    assert scores["breadth_skill"] == pytest.approx(entrant["scores"]["breadth_skill"], abs=0.01)
    for n, v in values.items():
        b = entrant["benchmarks"][str(n)]["board"]
        assert v["skill"] == pytest.approx(b["skill"], abs=1.5e-4), n
        assert v["raw"] == pytest.approx(b["raw"], abs=1.5e-4), n


def test_gold_weighs_inside_area_only():
    s = spec()
    values = {n: dict(raw=0.0, skill=0.0, coverage=1.0) for a in s["areas"] for n in a["benchmarks"]}
    values[4] = dict(raw=1.0, skill=1.0, coverage=1.0)
    scores, areas = X.aggregate(values, s)
    retrieval = next(a for a in areas if a["id"] == "retrieval")
    assert retrieval["skill"] == pytest.approx(1.2 / (1.2 * 3 + 3))
    assert scores["balanced_skill"] == pytest.approx(100 * X.area_weights(s)["retrieval"] * retrieval["skill"])


def acos_rows():
    out = []
    for g, golds in (("r1", ["yes", "no", "no"]), ("r2", ["no", "no"])):
        for i, gold in enumerate(golds):
            q = {"type": "choice", "instructions": "", "criteria": {"yes": "y", "no": "n"}}
            out.append({"_evaluation": {"run_id": f"38:{g}:{i}", "catalog_id": 38, "dataset": "ACOS", "track": "t", "group_id": g}, "questions": {"q": q}, "expected": {"q": gold}, "scoring": {}})
    return out


def test_acos_per_review_f1():
    rows = acos_rows()
    preds = {"38:r1:0": "yes", "38:r1:1": "yes", "38:r1:2": "no", "38:r2:0": "no", "38:r2:1": "no"}
    res = {k: {"status": "ok", "total_wall_ms": 1.0, "response": {"answers": {"q": {"type": "choice", "choice": v, "probabilities": {"yes": float(v == "yes"), "no": float(v == "no")}}}}} for k, v in preds.items()}
    rep = score(rows, res)
    assert rep["review_f1"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert rep["case_exact_accuracy"] == pytest.approx(0.5)
    summary = benchmark_summary(None, res, "x", rows=rows, metrics={38: ("per-review F1", "review_f1")})
    b = summary["benchmarks"][0]
    assert (b["metric"], b["score"]) == ("per-review F1", pytest.approx(rep["review_f1"]))
    assert benchmark_summary(None, res, "x", rows=rows)["benchmarks"][0]["score"] == pytest.approx(0.5)


def test_editions():
    e = editions.get("0.2.1")
    assert editions.get("release-v2.1") is e and editions.DEFAULT == "0.2.1"
    assert e["requests"] == 120340 and e["scoreable"] == 119898 and e["added_requests"] == 30419
    for k in ("rows_sha256", "rows_gz_sha256", "added_sha256", "exclusions_sha256", "dataset", "suite_dir"):
        assert e[k] == editions.get("0.2")[k]
    assert editions.compatible("0.2.1", "0.2") and editions.compatible("0.2", "0.2.1") and not editions.compatible("0.1", "0.2")
    for ed in ("0.2", "0.2.1"):
        pinned = editions.subset_sha256(ed)
        assert pinned and all(editions.get(ed)[k] == v for k, v in pinned.items())
    keep = editions.scoring_subsets("0.2.1")
    assert {n: len(v) for n, v in keep.items()} == {38: 1565, 2: 1013, 36: 307, 9: 88}
    assert set(editions.scoring_subsets("0.2")) == {38}


def test_suite_021_reads_02_files_and_applies_subsets(tmp_path):
    keep = editions.scoring_subsets("0.2.1")
    toolret = sorted(keep[2])[0]
    home = sorted(keep[9])[0]
    rows = [(2, toolret), (2, "2:unanswerable"), (9, home), (9, "9:dropped"), (24, "24:m")]
    with gzip.open(tmp_path / editions.ROWS_FILE, "wt") as f:
        for n, rid in rows:
            f.write(json.dumps({"_evaluation": {"run_id": rid, "catalog_id": n, "group_id": rid}}) + "\n")
    with gzip.open(tmp_path / editions.ADDED_FILE, "wt"):
        pass
    (tmp_path / editions.MANIFEST_FILE).write_text(json.dumps({"edition": "release-v2"}))
    assert [r["_evaluation"]["run_id"] for r in Suite(tmp_path, "0.2.1").rows()] == [toolret, home, "24:m"]
    assert len(list(Suite(tmp_path, "0.2").rows())) == 5
    (tmp_path / editions.MANIFEST_FILE).write_text(json.dumps({"edition": "release-v2.1"}))
    assert len(list(Suite(tmp_path, "0.2").rows())) == 5
    with pytest.raises(ValueError):
        Suite(tmp_path, "0.1")
