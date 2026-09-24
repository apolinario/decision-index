import collections
import statistics

from decision_index.scoring.metrics import mean

FIELDS = {56: ("verdict",)}
F1_POSITIVE = {59: True}


def field(q, gold, a):
    if q["type"] == "noul":
        v = a["noul"]
        pred = v >= 0.5
        return dict(conf=max(v, 1 - v), correct=pred == gold, pred=pred, gold=gold, k=2)
    p = a["probabilities"]
    return dict(conf=max(p.values()), correct=a["choice"] == gold, pred=a["choice"], gold=gold, k=len(q["criteria"]))


def f1(items, positive):
    tp = sum(x["pred"] == positive and x["gold"] == positive for x in items)
    fp = sum(x["pred"] == positive and x["gold"] != positive for x in items)
    fn = sum(x["pred"] != positive and x["gold"] == positive for x in items)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def scored_keys(n, row):
    unscored = set((row.get("scoring") or {}).get("unscored_fields", []))
    only = FIELDS.get(n)
    return [k for k in row["questions"] if k not in unscored and (not only or k in only)]


def chance(n, rows):
    items = [dict(gold=r["expected"][k], k=2 if r["questions"][k]["type"] == "noul" else len(r["questions"][k]["criteria"])) for r in rows for k in scored_keys(n, r)]
    if not items:
        return None
    if n in F1_POSITIVE:
        prevalence = mean(x["gold"] == F1_POSITIVE[n] for x in items)
        return prevalence / (prevalence + 0.5)
    return mean(1 / x["k"] for x in items)


def report(n, rows, results):
    status = collections.Counter()
    fields, latency = [], []
    for row in rows:
        r = results.get(row["_evaluation"]["run_id"])
        if r is None:
            status["pending"] += 1
            continue
        status[r["status"]] += 1
        if r["status"] != "ok":
            continue
        latency.append(r.get("total_wall_ms") or 0)
        answers = r["response"]["answers"]
        for key in scored_keys(n, row):
            x = field(row["questions"][key], row["expected"][key], answers[key])
            x["track"] = row["_evaluation"]["track"]
            fields.append(x)
    answered = status["ok"]
    out = dict(catalog_id=n, dataset=rows[0]["_evaluation"]["dataset"], requests=len(rows), answered=answered, unsupported=status["unsupported"], errors=status["error"], abstained=status["abstained"], pending=status["pending"], scored_fields=len(fields), metric="accuracy", score=mean(x["correct"] for x in fields), field_accuracy=mean(x["correct"] for x in fields), median_ms=statistics.median(latency) if latency else None, chance=chance(n, rows))
    if n in F1_POSITIVE:
        out.update(metric="F1 on hallucinated class", score=f1(fields, F1_POSITIVE[n]) if fields else None)
    return out
