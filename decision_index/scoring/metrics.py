import json
import math
import statistics


def mean(xs):
    return statistics.mean(xs) if xs else None


def avg(xs):
    return statistics.mean(xs) if xs else 0.0


def percentile(xs, p):
    if not xs:
        return None
    x = sorted(xs)
    return x[min(len(x) - 1, math.ceil(len(x) * p) - 1)]


def prediction(q, a):
    return a["choice"] if q["type"] == "choice" else a["noul"] >= 0.5


def semantic_label(q, x):
    if q["type"] == "choice":
        return json.dumps(q["criteria"].get(str(x), x), sort_keys=True, ensure_ascii=False)
    return str(x)


def macro_f1(pairs):
    labels = set(x for p in pairs for x in p)
    result = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in pairs)
        fp = sum(g != label and p == label for g, p in pairs)
        fn = sum(g == label and p != label for g, p in pairs)
        result[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    return mean(list(result.values()))


def binary_f1(pairs, positive="yes"):
    tp = sum(g == positive and p == positive for g, p in pairs)
    fp = sum(g != positive and p == positive for g, p in pairs)
    fn = sum(g == positive and p != positive for g, p in pairs)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0


def conservative_f1(golds, preds, classes, positive=None):
    missing = sum(p is None for p in preds)
    values = []
    for k in [positive] if positive is not None else classes:
        tp = sum(g == k and p == k for g, p in zip(golds, preds))
        fp = sum(g != k and p == k for g, p in zip(golds, preds))
        fn = sum(g == k and p is not None and p != k for g, p in zip(golds, preds))
        d = 2 * tp + fp + fn + missing
        values.append(2 * tp / d if d else 0.0)
    return avg(values)


def skill(x, b):
    return min(1.0, max(0.0, (x - b) / (1 - b)))


def score_query(query_record, document_probabilities):
    ids = query_record["scorable_ids"]
    qrels = query_record["qrels"]
    if not all(d in document_probabilities and isinstance(document_probabilities[d], (int, float)) and math.isfinite(document_probabilities[d]) and 0 <= document_probabilities[d] <= 1 for d in ids):
        return {"valid": False, "reason": "missing or invalid candidate probability"}
    ordered = sorted(ids, key=lambda d: (-document_probabilities[d], d))
    ranked = [qrels.get(d, 0) for d in ordered]
    ideal = sorted(qrels.values(), reverse=True)[:10]
    dcg = sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(ranked[:10]))
    idcg = sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(ideal))
    positives = sum(v > 0 for v in qrels.values())
    return {
        "valid": True,
        "ndcg_at_10": dcg / idcg if idcg else None,
        "mrr": next((1 / (i + 1) for i, v in enumerate(ranked) if v > 0), 0.0),
        "recall_at_10": sum(v > 0 for v in ranked[:10]) / positives if positives else None,
        "candidate_recall": query_record["candidate_recall"],
        "scorable_candidate_recall": sum(qrels.get(d, 0) > 0 for d in ids) / positives if positives else None,
        "candidates_scored": len(ids),
        "candidates_retrieved": len(query_record["retrieved_ids"]),
    }


def random_ndcg_baseline(s):
    ideal = sorted(s["qrels"].values(), reverse=True)[:10]
    idcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(ideal))
    mean_gain = avg([2 ** s["qrels"].get(d, 0) - 1 for d in s["scorable_ids"]])
    return mean_gain * sum(1 / math.log2(i + 2) for i in range(min(10, len(s["scorable_ids"])))) / idcg if idcg else 0.0


def router_score(row, choice, question="quality"):
    s = row["scoring"]
    if choice not in s["outcomes"]:
        return {"valid": False, "oracle_optimal": False}
    outcome = s["outcomes"][choice]
    penalty = 0.0 if question == "quality" else s["cost_penalty_per_usd"]
    utility = outcome["quality"] - penalty * outcome["cost_usd"]
    best = max(o["quality"] - penalty * o["cost_usd"] for o in s["outcomes"].values())
    return {"valid": True, **outcome, "utility": utility, "oracle_optimal": math.isclose(best, utility, abs_tol=1e-12, rel_tol=0), "utility_regret": max(0.0, best - utility)}


def chess_score(row, choice):
    values = row["scoring"]["values"]
    if choice not in values:
        return {"valid": False, "best_move": False, "value_regret": 1.0}
    best = max(values.values())
    return {"valid": True, "best_move": choice in row["scoring"]["accepted"], "value_regret": best - values[choice]}


def consensus_score(row, choice):
    scores = row["scoring"]
    if choice not in scores["summed_ranks"]:
        return {"valid": False, "group_preferred": False, "normalized_rank_regret": 1.0}
    cost = scores["summed_ranks"][choice]
    best = min(scores["summed_ranks"].values())
    return {"valid": True, "group_preferred": choice in scores["accepted"], "normalized_rank_regret": (cost - best) / (scores["panel_size"] * (len(scores["summed_ranks"]) - 1))}


def forecast_score(row, p_yes):
    if not isinstance(p_yes, (int, float)) or not math.isfinite(p_yes) or not 0 <= p_yes <= 1:
        return {"valid": False}
    y = int(row["expected"]["answer"] == "yes")
    p = max(1e-15, min(1 - 1e-15, p_yes))
    return {"valid": True, "brier": (p_yes - y) ** 2, "log_loss": -y * math.log(p) - (1 - y) * math.log1p(-p), "correct": int(p_yes >= 0.5) == y}
