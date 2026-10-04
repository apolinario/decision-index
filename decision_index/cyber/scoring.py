"""Coverage-aware metrics for separate, source-grouped cybersecurity tracks."""
import math
import random
from collections import Counter, defaultdict

from decision_index.engines.base import validate as validate_response
from .common import SEED, TRACKS, digest, rank


def metrics(items):
    n = len(items)
    answered = [x for x in items if x["valid"]]
    correct = sum(x["correct"] for x in items)
    classes = sorted({x["gold"] for x in items})
    f1s = []
    for label in classes:
        tp = sum(x["gold"] == label and x["prediction"] == label for x in items)
        fp = sum(x["gold"] != label and x["prediction"] == label for x in items)
        fn = sum(x["gold"] == label and x["prediction"] != label for x in items)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    groups = defaultdict(list)
    for x in items:
        groups[x["group"]].append(x["correct"])
    means = [sum(v) / len(v) for _, v in sorted(groups.items())]
    ci = None
    if len(means) >= 10:
        rng = random.Random(SEED)
        samples = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(1000))
        ci = [samples[24], samples[974]]
    brier, losses = [], []
    bins = defaultdict(list)
    for x in answered:
        probabilities = x["probabilities"]
        brier.append(sum((p - int(k == x["gold"])) ** 2 for k, p in probabilities.items()))
        losses.append(-math.log(max(probabilities[x["gold"]], 1e-15)))
        confidence = probabilities[x["prediction"]]
        bins[min(9, int(confidence * 10))].append((confidence, x["correct"]))
    ece = sum(abs(sum(p for p, _ in bucket) / len(bucket) - sum(c for _, c in bucket) / len(bucket))
              * len(bucket) for bucket in bins.values()) / len(answered) if answered else None
    random_chance = sum(1 / x["option_count"] for x in items) / n if n else 0
    accuracy = correct / n if n else 0
    return {"questions": n, "answered": len(answered), "coverage": len(answered) / n if n else 0,
            "accuracy": accuracy, "macro_f1_on_gold_supported_classes": sum(f1s) / len(f1s) if f1s else 0,
            "uniform_random_expected_accuracy": random_chance,
            "chance_corrected_accuracy": max(0, (accuracy - random_chance) / (1 - random_chance)) if random_chance < 1 else 0,
            "independent_groups": len(groups), "group_macro_accuracy": sum(means) / len(means) if means else 0,
            "group_macro_accuracy_cluster_bootstrap_95pct": ci,
            "confidence_interval_note": "1000 resamples of source-group means; suppressed below 10 groups.",
            "multiclass_brier_answered_only": sum(brier) / len(brier) if brier else None,
            "log_loss_answered_only": sum(losses) / len(losses) if losses else None,
            "ece_10_bins_answered_only": ece,
            "failures": dict(Counter(x["status"] for x in items if not x["valid"])),
            "gold_class_counts": dict(Counter(x["gold"] for x in items))}


def binary_errors(items, positive, negative):
    judged = [x for x in items if x["gold"] in {positive, negative}]
    positives = [x for x in judged if x["gold"] == positive]
    negatives = [x for x in judged if x["gold"] == negative]
    return {"judged_questions": len(judged), "excluded_unjudged": len(items) - len(judged),
            "positive_support": len(positives), "negative_support": len(negatives),
            "false_positive_rate": sum(x["prediction"] == positive for x in negatives) / len(negatives) if negatives else None,
            "miss_rate_including_failed_answers": sum(x["prediction"] != positive for x in positives) / len(positives) if positives else None,
            "failed_answer_rate": sum(not x["valid"] for x in judged) / len(judged) if judged else None,
            "nonbinary_answer_rate": sum(x["valid"] and x["prediction"] not in {positive, negative} for x in judged) / len(judged) if judged else None}


def score(values, predictions):
    from .gate import require
    require(values)
    gold_by_run = {r["_evaluation"]["run_id"]: r for r in values}
    by_run, duplicates = {}, 0
    engines, identities = set(), set()
    for p in predictions:
        rid = p.get("run_id")
        if rid not in gold_by_run:
            raise ValueError("result contains a run outside this frozen partition")
        if p.get("payload_sha256") != gold_by_run[rid]["_evaluation"]["payload_sha256"]:
            raise ValueError("result payload hash differs from frozen input")
        engines.add(p.get("engine"))
        identities.add(p.get("run_identity_sha256"))
        if len(engines) > 1 or len(identities) > 1:
            raise ValueError("Imported cyber results mix engines or run identities")
        if rid in by_run:
            raise ValueError("Repeated finished cyber prediction attempt")
        if "payload" in p and (digest(p["payload"]) if not rid.startswith("esdb-v0.3:") else __import__("decision_index.cyber.v3.common", fromlist=["payload_hash"]).payload_hash(**p["payload"])) != p["payload_sha256"]:
            raise ValueError("result visible payload differs from its receipt")
        duplicates += rid in by_run
        by_run[rid] = p  # Matches runner resume: latest recorded attempt wins.
    items_by_track, case_results = defaultdict(list), defaultdict(list)
    for rid, row in gold_by_run.items():
        p = by_run.get(rid, {"status": "missing"})
        status, valid = p.get("status", "missing"), False
        if status == "ok":
            try:
                validate_response(row["questions"], p["response"])
                valid = True
            except (KeyError, ValueError, TypeError, AttributeError):
                status = "invalid"
        case_correct = True
        for field, gold in row["expected"].items():
            answer = p["response"]["answers"][field] if valid else {}
            prediction = answer.get("choice")
            raw_probs = answer.get("probabilities", {})
            # Runner accepts sum tolerance 0.01; normalize before proper scoring.
            probs = {k: v / sum(raw_probs.values()) for k, v in raw_probs.items()} if valid else {}
            correct = valid and prediction == gold
            case_correct &= correct
            items_by_track[row["family"]].append({"gold": gold, "prediction": prediction, "correct": correct,
                "valid": valid, "status": status, "probabilities": probs, "option_count": len(row["questions"][field]["criteria"]),
                "group": row["metadata"]["split_group"], "task": row["metadata"]["task"], "field": field})
        case_results[row["family"]].append({"correct": case_correct, "valid": valid, "group": row["metadata"]["split_group"]})
    tracks = {}
    for track in TRACKS:
        items = items_by_track[track]
        if not items:
            continue
        cases = case_results[track]
        case_groups = defaultdict(list)
        for x in cases:
            case_groups[x["group"]].append(x["correct"])
        task_fields = sorted({(x["task"], x["field"]) for x in items})
        tracks[track] = {**metrics(items), "cases": len(cases),
                        "case_exact_accuracy": sum(x["correct"] for x in cases) / len(cases),
                        "case_group_macro_accuracy": sum(sum(v) / len(v) for v in case_groups.values()) / len(case_groups),
                        "case_coverage": sum(x["valid"] for x in cases) / len(cases),
                        "task_fields": {f"{task}/{field}": metrics([x for x in items if (x["task"], x["field"]) == (task, field)])
                                        for task, field in task_fields}}
        if track == "authorization_policy":
            tracks[track]["approval_errors"] = binary_errors(items, "requires_approval", "permitted")
            tracks[track]["approval_errors_by_policy"] = {op: binary_errors([x for x in items if x["field"] == op],
                "requires_approval", "permitted") for op in sorted({x["field"] for x in items})}
        elif track == "network_defense":
            tracks[track]["binary_detection_errors"] = binary_errors(items, "Botnet", "Normal")
    return {"tracks": tracks, "result_rows": len(by_run), "duplicate_attempt_rows": duplicates,
            "protocol": "Every expected question and case remains in its denominator. Missing/invalid/failed answers are wrong. "
                        "Probability metrics are conditional on valid answers and accompany coverage. No pooled enterprise index.",
            "complete": len(by_run) == len(values) and all(x["valid"] for v in case_results.values() for x in v)}


def baseline(values, development, mode):
    """Predeclared baseline; parameter estimation uses development only."""
    from .gate import require
    require(values)
    counts = defaultdict(Counter)
    for row in development:
        for field, gold in row["expected"].items():
            counts[row["family"], row["metadata"]["task"], field][gold] += 1
    output = []
    for row in values:
        answers = {}
        for field, q in row["questions"].items():
            keys = list(q["criteria"])
            c = counts[row["family"], row["metadata"]["task"], field]
            if mode == "development_majority":
                selected = min(keys, key=lambda k: (-c[k], k))
                probabilities = {k: float(k == selected) for k in keys}
            elif mode == "uniform":
                selected = min(keys, key=lambda k: rank("uniform-baseline", row["id"], field, k))
                probabilities = {k: 1 / len(keys) for k in keys}
            else:
                raise ValueError("unknown baseline")
            answers[field] = {"type": "choice", "choice": selected, "probabilities": probabilities}
        output.append({**row["_evaluation"], "status": "ok", "engine": mode, "response": {"answers": answers}})
    return output
