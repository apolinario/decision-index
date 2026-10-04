"""Whole-incident GUIDE selection with source-wide EOF and hash verification."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from .common import choice, digest, file_hash, rank, receipt, record
from .protection import legacy_guide_group

SOURCE = "external/microsoft-guide-source-v2/GUIDE_Train.csv"
PIN = "3d6c286e1353236b3457a011a20cb9a8cec26a589ab685dab31dad261d47b8c3"
FIELDS = ("Category", "MitreTechniques", "EntityType", "ResourceType", "AntispamDirection", "DetectorId", "AlertTitle")
LABELS = ("TruePositive", "BenignPositive", "FalsePositive")


def visible_incident(native_rows):
    """Retain native event chronology and alert relationships without source IDs."""
    alerts, pairs = {}, defaultdict(set)
    evidence = []
    for row in native_rows:
        alert = row["AlertId"]
        local_alert = alerts.setdefault(alert, f"alert-{len(alerts) + 1}")
        evidence.append({"alert": local_alert, "Timestamp": row["Timestamp"],
                         **{field: row[field] for field in FIELDS if row[field]}})
        for field in FIELDS:
            values = row[field].split(";") if field == "MitreTechniques" else [row[field]]
            pairs[field].update((alert, v.strip()) for v in values if v.strip())
    return {"scope": "Complete retrospective incident metadata in native source order; event timestamps retained. "
                     "Alert references are local to this incident. Response actions and verdict fields excluded.",
            "alerts": len(alerts), "evidence_rows": len(native_rows), "alert_evidence": evidence,
            "counts": {field: dict(sorted(Counter(v for _, v in pairs[field]).items())) for field in FIELDS}}


def build(collection, protected, *, organizations=300, per_org=10, max_rows=500, class_cap=400, log=print):
    path = Path(collection) / SOURCE
    if file_hash(path) != PIN:
        raise ValueError("GUIDE source hash differs from the pinned official training file")
    stat = path.stat()
    org_rank, selected, worst_incident = {}, {}, {}
    worst_org = None
    scan = 0
    rejected = Counter()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        required = {"OrgId", "IncidentId", "AlertId", "Timestamp", "IncidentGrade", *FIELDS}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("GUIDE schema missing required fields")
        for scan, row in enumerate(reader, 1):
            if scan % 1000000 == 0:
                log(f"GUIDE: scanned {scan:,} source rows")
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"malformed GUIDE CSV record {scan}")
            org, incident = row["OrgId"], row["IncidentId"]
            if not org or not incident or not row["AlertId"]:
                raise ValueError("GUIDE record lacks grouping identity")
            if org not in org_rank:
                org_rank[org] = rank("guide-org", org)
            if legacy_guide_group(org) in protected["groups"]:
                rejected["known_sev_organization_rows"] += 1
                continue
            if org not in selected:
                if len(selected) == organizations:
                    if (org_rank[org], org) >= (org_rank[worst_org], worst_org):
                        continue
                    del selected[worst_org]
                    worst_incident.pop(worst_org, None)
                selected[org] = {}
                if len(selected) == organizations:
                    worst_org = max(selected, key=lambda k: (org_rank[k], k))
            bucket = selected[org]
            if incident not in bucket:
                value = rank("guide-incident", org, incident)
                if len(bucket) == per_org:
                    old = worst_incident[org]
                    if (value, incident) >= (bucket[old]["rank"], old):
                        continue
                    del bucket[old]
                bucket[incident] = {"rank": value, "n": 0, "grades": set(), "rows": [], "ordinals": []}
                if len(bucket) == per_org:
                    worst_incident[org] = max(bucket, key=lambda k: (bucket[k]["rank"], k))
            item = bucket[incident]
            item["n"] += 1
            item["grades"].add(row["IncidentGrade"])
            if item["n"] > max_rows:
                continue
            item["rows"].append({key: row[key] for key in ("AlertId", "Timestamp", *FIELDS)})
            item["ordinals"].append(scan)
    if (path.stat().st_size, path.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns) or file_hash(path) != PIN:
        raise ValueError("GUIDE source changed during full scan")
    if scan != 9516837:
        raise ValueError("GUIDE EOF record count differs from the source receipt")
    ordered = sorted(selected, key=lambda org: (org_rank[org], org))
    from .common import SPLITS
    assignments = {org: SPLITS[i % 3] for i, org in enumerate(ordered)}
    output = defaultdict(list)
    for org, bucket in selected.items():
        for incident, item in bucket.items():
            if item["n"] > max_rows:
                rejected["whole_incident_over_row_bound"] += 1
                continue
            if len(item["grades"]) != 1 or not item["grades"] <= set(LABELS):
                rejected["missing_or_conflicting_incident_grade"] += 1
                continue
            grade = next(iter(item["grades"]))
            state = visible_incident(item["rows"])
            if digest(state) in protected["states"]:
                rejected["known_sev_state"] += 1
                continue
            q, gold = choice([org, incident], "incident_grade",
                             "Predict the provider-adjudicated incident grade from the supplied metadata. "
                             "The grade applies to the incident, not to each evidence row.",
                             {k: k for k in LABELS}, grade)
            entry = record("incident_triage", [PIN, org, incident], assignments[org], "guide-org/" + digest(org),
                           state, {"incident_grade": q}, {"incident_grade": gold},
                           {"task": "provider_incident_grade", "source_kind": "anonymized_real_incident_evidence",
                            "organization": org, "incident": incident, "label_basis": "IncidentGrade"},
                           {"path": SOURCE, "sha256": PIN, "source_partition": "official_train",
                            "row_ordinals": item["ordinals"], "license": "CDLA-Permissive-2.0"})
            output[assignments[org], grade].append(entry)
    entries = [r for key in sorted(output) for r in sorted(output[key], key=lambda r: rank("guide-final", r["id"]))[:class_cap]]
    report = {"source": receipt(path, collection), "source_rows_scanned": scan, "eof_verified": True,
              "selected_organizations_before_incident_checks": len(assignments), "rejections": dict(rejected),
              "selection": {"organizations": organizations, "incidents_per_org": per_org,
                            "max_whole_incident_rows": max_rows, "per_split_class_cap": class_cap},
              "limitations": ["Only the official training source is used; ESDB splits are independent local splits.",
                              "Detector-assisted retrospective grade prediction, not raw event maliciousness or online response.",
                              "Class caps change prevalence; do not interpret dataset class proportions as operational prevalence."]}
    return entries, report
