import collections
import json
import math
import statistics
from importlib import resources

from decision_index import constants as C
from decision_index.scoring.metrics import avg, conservative_f1, random_ndcg_baseline, score_query, semantic_label, skill


def load_data(name):
    return json.loads(resources.files("decision_index").joinpath("data", name).read_text())


def panel():
    return load_data("index-panel.json")


def panel_ids():
    return {b["catalog_id"] for c in panel()["categories"] for b in c["benchmarks"]}


def chance_baselines():
    return load_data("chance-baselines.json")["benchmarks"]


def pred(q, a):
    return a["choice"] if q["type"] == "choice" else a["noul"] >= 0.5


def static_score(n, rows, res, random_baselines):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["_evaluation"]["group_id"]].append(r)
    good = {g: all(res.get(r["_evaluation"]["run_id"], {}).get("status") == "ok" for r in rs) for g, rs in groups.items()}
    pending = {g: any(r["_evaluation"]["run_id"] not in res or res[r["_evaluation"]["run_id"]]["status"] == "error" for r in rs) for g, rs in groups.items()}
    tracks = collections.defaultdict(list)
    for r in rows:
        t = r["_evaluation"]["track"] if n in (30, 40, 6) else "all"
        if n == 40 and "-B-" in t:
            continue
        tracks[t].append(r)
    parts = []
    for track, rr in tracks.items():
        values, baselines, golds, preds, classes = [], [], [], [], set()
        macro = collections.defaultdict(list)
        visited = set()
        for r in rr:
            e = r["_evaluation"]
            gid = e["group_id"]
            ok = good[gid]
            answers = res[e["run_id"]]["answers"] if ok else {}
            if n in (2, 36):
                if gid in visited:
                    continue
                visited.add(gid)
                s = r["scoring"]
                probs = {}
                if ok:
                    for row in groups[gid]:
                        a = res[row["_evaluation"]["run_id"]]["answers"]
                        probs.update({doc: a[field]["probabilities"]["yes"] for field, doc in row["scoring"]["field_to_document"].items()})
                values.append(score_query(s, probs)["ndcg_at_10"] if ok else 0.0)
                baselines.append(random_ndcg_baseline(s))
                continue
            if n == 6:
                outcomes = r["scoring"]["outcomes"]
                values.append(outcomes[answers["quality"]["choice"]]["quality"] if ok else 0.0)
                baselines.append(avg([o["quality"] for o in outcomes.values()]))
                continue
            if n == 1:
                if gid in visited:
                    continue
                visited.add(gid)
                hits = []
                chance = 1.0
                for row in groups[gid]:
                    a = res[row["_evaluation"]["run_id"]]["answers"] if ok else {}
                    for key, q in row["questions"].items():
                        hits.append(ok and pred(q, a[key]) == row["expected"][key])
                        chance /= len(q["criteria"]) if q["type"] == "choice" else 2
                values.append(float(all(hits)))
                baselines.append(chance)
                continue
            for key, q in r["questions"].items():
                gold = r["expected"][key]
                p = pred(q, answers[key]) if ok else None
                if n in (11, 37, 41) or n == 40 and "-A-" in track:
                    if n == 40:
                        golds.append(gold)
                        preds.append(p)
                        classes.update(q["criteria"])
                    else:
                        golds.append(semantic_label(q, gold))
                        preds.append(semantic_label(q, p) if ok else None)
                        classes.update(semantic_label(q, k) for k in q["criteria"])
                    continue
                accepted = r.get("scoring", {}).get("accepted", [gold]) if n in (31, 50) else [gold]
                value = float(ok and p in accepted)
                chance = len(accepted) / (len(q["criteria"]) if q["type"] == "choice" else 2)
                macro_key = r["provenance"]["subject"] if n == 24 else r["metadata"]["variant"] if n == 20 else r["metadata"]["song_id"] if n == 22 else r["metadata"]["user_id"] if n == 23 else "all"
                macro[macro_key].append((value, chance))
        if golds:
            raw = conservative_f1(golds, preds, sorted(classes), "yes" if n == 40 else None)
            b = random_baselines[str(n)]["tracks"][track] if n == 40 else random_baselines[str(n)]["primary"]
            assert b.get("monte_carlo_standard_error", 0) <= 0.001
            baseline = b["value"]
        elif macro:
            raw = avg([avg([v for v, b in pairs]) for pairs in macro.values()])
            baseline = avg([avg([b for v, b in pairs]) for pairs in macro.values()])
        else:
            raw = avg(values)
            baseline = avg(baselines)
        gids = {r["_evaluation"]["group_id"] for r in rr}
        parts.append(dict(track=track, raw=raw, random=baseline, skill=skill(raw, baseline), planned_cases=len(gids), coverage=avg([int(good[g]) for g in gids]), pending=avg([int(pending[g]) for g in gids]), label_universe=sorted(classes)))
    return dict(catalog_id=n, raw=avg([p["raw"] for p in parts]), skill=avg([p["skill"] for p in parts]), coverage=avg([p["coverage"] for p in parts]), pending=avg([p["pending"] for p in parts]), tracks=parts)


def interactive_placeholder(n):
    return dict(catalog_id=n, raw=0.0, skill=0.0, coverage=0.0, pending=1.0, status="unrun; provisional lower bound", tracks=[])


def score_panel(suite, results):
    ids = panel_ids()
    rows = collections.defaultdict(list)
    for r in suite.rows(apply_exclusions=True):
        n = r["_evaluation"]["catalog_id"]
        if n in ids:
            rows[n].append(r)
    res = {rid: {"status": r["status"], "answers": (r.get("response") or {}).get("answers", {})} for rid, r in results.items() if r.get("catalog_id") in ids}
    baselines = chance_baselines()
    scored = {n: static_score(n, rs, res, baselines) for n, rs in rows.items()}
    for n in C.INTERACTIVE:
        scored[n] = interactive_placeholder(n)
    return scored


def frozen_scores(scored):
    categories = []
    for c in panel()["categories"]:
        values = [scored[b["catalog_id"]] for b in c["benchmarks"]]
        categories.append(dict(id=c["id"], label=c["label"], raw=avg([v["raw"] for v in values]), skill=avg([v["skill"] for v in values]), coverage=avg([v["coverage"] for v in values]), pending=avg([v["pending"] for v in values])))
    scores = dict(
        balanced_raw=100 * avg([c["raw"] for c in categories]),
        balanced_skill=100 * avg([c["skill"] for c in categories]),
        breadth_skill=100 * (math.prod((0.1 + 0.9 * c["skill"]) ** 0.2 for c in categories) - 0.1) / 0.9,
    )
    return scores, categories


def headline(b):
    t = next((t for t in b.get("tracks", []) if t["track"] == C.HEADLINE.get(b["catalog_id"])), None)
    if not t:
        return b
    r = t.get("random") or 0.0
    return {**b, "raw": t["raw"], "skill": min(1.0, max(0.0, (t["raw"] - r) / (1 - r))) if r < 1 else 0.0, "coverage": t.get("coverage", b["coverage"]), "pending": t.get("pending", b.get("pending", 0)), "tracks": b["tracks"]}


def track_list(b):
    tracks = b.get("tracks", [])
    if len(tracks) <= 1:
        return []
    return [{"track": C.TRACK_LABEL.get(t["track"], t["track"]), "score": rnd(t["raw"]), "headline": t["track"] == C.HEADLINE.get(b["catalog_id"])} for t in tracks]


def area_ids():
    drop = set(C.INTERACTIVE)
    out = []
    for area in C.AREAS:
        ids = [i for i in area["panel"] if i not in drop]
        for src, dst in C.FOLD.items():
            if dst == area["id"]:
                ids = ids + [i for i in C.FOLDED_PANEL[src] if i not in drop]
        out.append({"id": area["id"], "label": area["label"], "ids": ids})
    return out


def recompute(scored):
    per = {n: headline(b) for n, b in scored.items()}
    cats = []
    for area in area_ids():
        ids = [i for i in area["ids"] if i in per]
        cats.append({
            "id": area["id"],
            "label": area["label"],
            "raw": statistics.mean(per[i]["raw"] for i in ids),
            "skill": statistics.mean(per[i]["skill"] for i in ids),
            "coverage": statistics.mean(per[i]["coverage"] for i in ids),
            "pending": statistics.mean(per[i].get("pending", 0) for i in ids),
            "n": len(ids),
            "benchmarks": ids,
        })
    skills = [c["skill"] for c in cats]
    scores = {
        "balanced_raw": 100 * statistics.mean(c["raw"] for c in cats),
        "balanced_skill": 100 * statistics.mean(skills),
        "breadth_skill": 100 * (math.prod((0.1 + 0.9 * x) ** (1 / len(skills)) for x in skills) - 0.1) / 0.9,
    }
    return scores, cats


def rnd(x, n=4):
    return None if x is None else round(x, n)


def index_entry(scored, precision=4):
    scores, cats = recompute(scored)
    frozen, frozen_cats = frozen_scores(scored)
    per = {n: headline(b) for n, b in scored.items()}
    benchmarks = {}
    for n, b in sorted(per.items()):
        random = next((t.get("random") for t in b.get("tracks", []) if t["track"] == C.HEADLINE.get(n, t["track"]) and t.get("random") is not None), None)
        benchmarks[str(n)] = {"raw": rnd(b["raw"], precision), "skill": rnd(b["skill"], precision), "coverage": rnd(b["coverage"], precision), "pending": rnd(b.get("pending"), precision), "random": rnd(random, precision), "tracks": track_list(b), "in_index": n not in C.INTERACTIVE}
    return {
        "panel_id": C.PANEL_ID,
        "index": rnd(scores["balanced_skill"], 2),
        "scores": {k: rnd(v, 2) for k, v in scores.items()},
        "areas": [{"id": c["id"], "label": c["label"], "raw": rnd(c["raw"], precision), "skill": rnd(c["skill"], precision), "coverage": rnd(c["coverage"], precision), "pending": rnd(c["pending"], precision), "n": c["n"], "benchmarks": c["benchmarks"]} for c in cats],
        "benchmarks": benchmarks,
        "coverage": rnd(statistics.mean(c["coverage"] for c in cats), precision),
        "pending": rnd(statistics.mean(c["pending"] for c in cats), precision),
        "frozen_panel": {
            "scores": {k: rnd(v, 2) for k, v in frozen.items()},
            "categories": [{"id": c["id"], "raw": rnd(c["raw"], precision), "skill": rnd(c["skill"], precision), "coverage": rnd(c["coverage"], precision), "pending": rnd(c["pending"], precision)} for c in frozen_cats],
            "coverage": rnd(avg([c["coverage"] for c in frozen_cats]), precision),
            "pending": rnd(avg([c["pending"] for c in frozen_cats]), precision),
            "note": "25-benchmark frozen panel with the six interactive benchmarks scored as zero (provisional lower bound); not the headline index.",
        },
        "formulas": panel()["proposals"],
    }
