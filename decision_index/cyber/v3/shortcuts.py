"""Construction checks for the reviewed shortcuts, on dev/cal only."""
from collections import Counter, defaultdict
from ..common import digest


def program_blind(row):
    return {"state": {k: v for k, v in row["state"].items() if k != "program"}, "questions": row["questions"]}


def endpoint_shape(row):
    events = row["state"]["events"]
    return {"target": row["state"]["target_event"], "event_count": len(events),
            "schemas": [(e["EventID"], sorted(e)) for e in events],
            "image_counts": dict(Counter(e.get("Image", "") for e in events)),
            "questions": row["questions"], "request": row["state"]["request"]}


def ceiling(values, feature):
    buckets = defaultdict(Counter)
    for r in values:
        buckets[digest(feature(r))][digest(r["expected"])] += 1
    return sum(max(c.values()) for c in buckets.values()) / len(values) if values else None


def audit(values):
    output = {}
    for split in ("development", "calibration"):
        program = [r for r in values if r["split"] == split and r["metadata"]["task"] == "program_approval"]
        endpoint = [r for r in values if r["split"] == split and r["family"] == "endpoint_investigation" and not r["metadata"]["diagnostic_only"]]
        output[split] = {"program_blind_case_exact_ceiling": ceiling(program, program_blind),
                         "endpoint_schema_count_options_image_presence_case_exact_ceiling": ceiling(endpoint, endpoint_shape),
                         "program_cases": len(program), "endpoint_cases": len(endpoint)}
        if output[split]["program_blind_case_exact_ceiling"] != .5 or output[split]["endpoint_schema_count_options_image_presence_case_exact_ceiling"] != .5:
            raise ValueError("Reviewed program-blind or endpoint-shape shortcut exceeds balanced-pair ceiling")
    return {"scope": "Construction upper bounds for these specified features, not proof that every shortcut is absent. Test is not scored.",
            "partitions": output}
