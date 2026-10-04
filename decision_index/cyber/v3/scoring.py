"""Separate task scores, matched consistency, organization and class reports."""
from collections import defaultdict

from ..scoring import score as base_score, binary_errors
from ..common import digest
from decision_index.engines.base import validate


def network_detection_errors(values, by_run):
    """Keep provider-specific binary labels separate; Background is unjudged."""
    grouped = defaultdict(list)
    for row in values:
        if row["family"] != "network_defense" or row["metadata"]["source_dataset"] not in ("CTU-13", "IoT-23", "CSE-CIC-IDS2018"):
            continue
        result = by_run.get(row["_evaluation"]["run_id"], {})
        valid = result.get("status") == "ok"
        if valid:
            try:
                validate(row["questions"], result["response"])
            except (KeyError, ValueError, TypeError, AttributeError):
                valid = False
        field = next(iter(row["expected"]))
        grouped[row["metadata"]["source_dataset"]].append({"gold": row["expected"][field], "valid": valid,
            "prediction": result["response"]["answers"][field]["choice"] if valid else None})
    return {source: binary_errors(items, *(('Botnet', 'Normal') if source == 'CTU-13' else ('Malicious', 'Benign')))
            for source, items in grouped.items()}


def score(values, predictions):
    predictions = list(predictions)
    if len({p["run_id"] for p in predictions}) != len(predictions):
        raise ValueError("Repeated finished prediction attempts are not admitted")
    result = base_score(values, predictions)
    by_run = {p["run_id"]: p for p in predictions}
    main = [r for r in values if not r["metadata"].get("diagnostic_only", False)]
    diagnostic = [r for r in values if r["metadata"].get("diagnostic_only", False)]
    result["ranking_tracks"] = base_score(main, [p for p in predictions if p["run_id"] in {r["_evaluation"]["run_id"] for r in main}])["tracks"]
    result["diagnostic_tracks"] = base_score(diagnostic, [p for p in predictions if p["run_id"] in {r["_evaluation"]["run_id"] for r in diagnostic}])["tracks"] if diagnostic else {}
    for reports in (result["tracks"], result["ranking_tracks"]):
        if "network_defense" in reports:
            reports["network_defense"].pop("binary_detection_errors", None)
            reports["network_defense"]["binary_detection_errors_by_source_dataset"] = network_detection_errors(main, by_run)
    result["aggregate_index"] = None
    result["aggregation_note"] = "No pooled cross-track ranking. Endpoint ranking uses case-exact credit including supporting event references."
    grouped, matched = defaultdict(list), defaultdict(list)
    decision_pairs = defaultdict(list)
    class_rows = defaultdict(list)
    for row in main:
        p = by_run.get(row["_evaluation"]["run_id"], {})
        valid = p.get("status") == "ok"
        if valid:
            try:
                validate(row["questions"], p["response"])
            except (KeyError, ValueError, TypeError, AttributeError):
                valid = False
        answers = p.get("response", {}).get("answers", {}) if valid else {}
        correct = valid and all(answers.get(q, {}).get("choice") == y for q, y in row["expected"].items())
        detail = {"id": row["id"], "correct": correct, "valid": valid,
                  "expected": row["expected"], "predicted": {q: a.get("choice") for q, a in answers.items()}}
        m = row["metadata"]
        grouped[(row["family"], m["split_group"])].append(detail)
        if m.get("match_pair"):
            matched[(row["family"], m["match_pair"])].append(detail)
        for pair in m.get("decision_pairs", []):
            decision_pairs[(pair["axis"], pair["id"])].append(detail)
        field = next(iter(row["expected"]))
        class_rows[(row["family"], field, row["expected"][field])].append(detail)
    result["by_source_group"] = {family: {group: {"cases": len(items), "case_accuracy": sum(x["correct"] for x in items) / len(items),
        "coverage": sum(x["valid"] for x in items) / len(items)} for (f, group), items in grouped.items() if f == family} for family in {f for f, _ in grouped}}
    result["by_class"] = {family: {label: {"cases": len(items), "recall_including_failures": sum(x["correct"] for x in items) / len(items),
        "coverage": sum(x["valid"] for x in items) / len(items)} for (f, _, label), items in class_rows.items() if f == family} for family in {f for f, _, _ in class_rows}}
    guide_population = [x for (family, _), cases in grouped.items() if family == "incident_triage" for x in cases]
    for (family, field, label), cases in class_rows.items():
        if family != "incident_triage":
            continue
        tp = sum(x["correct"] for x in cases)
        predicted = sum(x["valid"] and x["predicted"].get(field) == label for x in guide_population)
        fp, fn = predicted - tp, len(cases) - tp
        result["by_class"][family][label].update(precision=tp / predicted if predicted else 0,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            true_positives=tp, false_positives=fp, false_negatives=fn)
    result["matched_all_correct"] = {family: {"groups": len(items), "accuracy": sum(all(r["correct"] for r in pair) for pair in items) / len(items)}
        for family in {f for f, _ in matched} for items in [[v for (f, _), v in matched.items() if f == family]]}
    result["policy_decision_consistency_by_axis"] = {axis: {"pairs": len(items), "both_cases_correct": sum(all(r["correct"] for r in pair) for pair in items) / len(items)}
        for axis in {a for a, _ in decision_pairs} for items in [[v for (a, _), v in decision_pairs.items() if a == axis]]}
    for track in ("network_defense", "authorization_policy", "endpoint_investigation"):
        fields = ("source_dataset", "capture_family") if track == "network_defense" else ("operation", "intervention", "task") if track == "authorization_policy" else ("task",)
        for field in fields:
            key = f"{track}_by_{field}"
            result[key] = {}
            for value in sorted({r["metadata"].get(field, "unknown") for r in main if r["family"] == track}):
                subset = [r for r in main if r["family"] == track and r["metadata"].get(field, "unknown") == value]
                runs = {r["_evaluation"]["run_id"] for r in subset}
                result[key][value] = base_score(subset, [p for p in predictions if p["run_id"] in runs])["tracks"][track]
                if track == "network_defense":
                    result[key][value].pop("binary_detection_errors", None)
                    result[key][value]["binary_detection_errors_by_source_dataset"] = network_detection_errors(subset, by_run)
    return result
