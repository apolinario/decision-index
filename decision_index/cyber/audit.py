"""Fail-closed projection/split checks and complete selected-source replay."""
import csv
import gzip
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .common import SPLITS, TRACKS, digest, file_hash, rank, rows, summary
from .endpoint import EFFECTS, UNKNOWN, VISIBLE, joined, native_records, xml_event
from .guide import FIELDS as GUIDE_FIELDS, visible_incident
from .network import FIELDS as FLOW_FIELDS, label_class
from .policy import OPERATIONS, valid_contract
from .protection import legacy_guide_group


def deduplicate(values):
    """Choose a single owner of identical evidence without consulting gold."""
    buckets = defaultdict(list)
    for row in values:
        buckets[row["family"], digest(row["state"])].append(row)
    kept, dropped, conflicts = [], Counter(), Counter()
    for (family, _), candidates in buckets.items():
        ordered = sorted(candidates, key=lambda r: (rank("duplicate-owner", r["id"]), r["id"]))
        kept.append(ordered[0])
        dropped[family] += len(ordered) - 1
        if len({digest(r["expected"]) for r in candidates}) > 1:
            conflicts[family] += 1
    for row in kept:
        for field, q in row["questions"].items():
            q["criteria"] = {k: q["criteria"][k] for k in sorted(
                q["criteria"], key=lambda k: rank("freeze-options", row["id"], field, k))}
        row["_evaluation"]["payload_sha256"] = digest({"state": row["state"], "questions": row["questions"]})
    return sorted(kept, key=lambda r: (r["family"], rank("frozen-order", r["id"]))), {
        "removed_identical_evidence": dict(dropped),
        "duplicate_evidence_with_different_answers_before_removal": dict(conflicts),
        "policy": "One deterministic evidence owner across all splits; owner selection does not inspect answers."}


def validate(values, protected=None):
    protected = protected or defaultdict(set)
    identifiers, groups, states, native_owners, source_owners = set(), {}, {}, {}, {}
    order_slots = defaultdict(Counter)
    for row in values:
        family, split = row["family"], row["split"]
        if family not in TRACKS or split not in SPLITS or row["id"] in identifiers:
            raise ValueError("invalid or duplicate record identity")
        identifiers.add(row["id"])
        group = row["metadata"]["split_group"]
        if groups.setdefault(group, split) != split:
            raise ValueError("source group crosses splits")
        state_sha = digest(row["state"])
        if states.setdefault(state_sha, split) != split:
            raise ValueError("identical evidence crosses splits")
        if state_sha in protected["states"] or group in protected["groups"]:
            raise ValueError("known Sev evidence admitted")
        payload = {"state": row["state"], "questions": row["questions"]}
        if digest(payload) != row["_evaluation"]["payload_sha256"]:
            raise ValueError("payload receipt mismatch")
        if set(row["expected"]) != set(row["questions"]):
            raise ValueError("answer fields do not match questions")
        for field, q in row["questions"].items():
            if set(q) != {"type", "instructions", "criteria"} or q["type"] != "choice":
                raise ValueError("question projection contains labels or unsupported fields")
            if not isinstance(q["instructions"], str) or not 2 <= len(q["criteria"]) <= 255:
                raise ValueError("malformed choice question")
            if row["expected"][field] not in q["criteria"]:
                raise ValueError("gold is not a supplied choice")
            order_slots[family][list(q["criteria"]).index(row["expected"][field])] += 1
        state = row["state"]
        if family == "incident_triage":
            org = row["metadata"]["organization"]
            if legacy_guide_group(org) in protected["groups"]:
                raise ValueError("known Sev organization admitted")
            if source_owners.setdefault((family, org), split) != split:
                raise ValueError("GUIDE organization crosses splits")
            if set(state) != {"scope", "alerts", "evidence_rows", "alert_evidence", "counts"}:
                raise ValueError("unexpected GUIDE visible fields")
            if len(state["alert_evidence"]) != state["evidence_rows"]:
                raise ValueError("incomplete selected GUIDE incident")
            if set(state["counts"]) != set(GUIDE_FIELDS) or any(
                    set(e) - {*GUIDE_FIELDS, "alert", "Timestamp"} for e in state["alert_evidence"]):
                raise ValueError("GUIDE outcome or identity field exposed")
        elif family == "network_defense":
            if source_owners.setdefault((family, row["metadata"]["capture_id"]), split) != split:
                raise ValueError("network capture crosses splits")
            if set(state) != {"scope", "flow"} or set(state["flow"]) != set(FLOW_FIELDS):
                raise ValueError("flow label or unexpected field exposed")
            gold = row["expected"]["flow_class"]
            if row["metadata"]["binary_disposition_judged"] != (gold != "Background"):
                raise ValueError("unjudged background relabeled")
        elif family == "endpoint_investigation":
            if set(state) != {"scope", "events"} or any(set(e) - set(VISIBLE) for e in state["events"]):
                raise ValueError("endpoint projection exposes unavailable fields")
            for ref in row["provenance"]["native_refs"]:
                if source_owners.setdefault((family, ref["sha256"]), split) != split:
                    raise ValueError("native capture crosses splits")
                if ref["sha256"] in protected["archives"] or ref["event_identity"] in protected["native_events"]:
                    raise ValueError("protected native capture admitted")
                if native_owners.setdefault(ref["event_identity"], split) != split:
                    raise ValueError("native event identity crosses splits")
        else:
            p, m = row["provenance"], row["metadata"]
            if source_owners.setdefault((family, m["template_sha256"]), split) != split:
                raise ValueError("policy AST template crosses splits")
            if (m["source_record_id"] in protected["program_ids"] or
                    m["template_sha256"] in protected["templates"] or
                    p["source_text_sha256"] in protected["program_texts"]):
                raise ValueError("previously used policy program admitted")
            if not isinstance(state, str) or not state.startswith("Recovered complete program. No execution is observed.\n"):
                raise ValueError("unexpected policy source projection")
    partitions = {split: {track: summary([r for r in values if r["split"] == split and r["family"] == track])
                          for track in TRACKS} for split in SPLITS}
    if any(not part["records"] for tracks in partitions.values() for part in tracks.values()):
        raise ValueError("empty track partition")
    return {"checks": {"question_gold_isolation": True, "state_allowlists": True,
                       "group_split_disjointness": True, "exact_evidence_split_disjointness": True,
                       "known_Sev_overlap_exclusion": True, "payload_hashes": True,
                       "native_event_split_disjointness": True},
            "partitions": partitions, "answer_position_counts": {k: {str(i): n for i, n in sorted(v.items())} for k, v in order_slots.items()},
            "total": summary(values)}


def replay(values, collection, sev_root, *, typescript=None, log=print):
    """Reopen original sources, rather than trusting cached answer sidecars."""
    collection, sev_root = Path(collection), Path(sev_root)
    by_family, result = defaultdict(list), {}
    for row in values:
        by_family[row["family"]].append(row)
    # GUIDE: a second EOF pass checks selected incidents are complete and labels unanimous.
    incidents = {(r["metadata"]["organization"], r["metadata"]["incident"]): r for r in by_family["incident_triage"]}
    native = defaultdict(list)
    path = collection / next(iter(incidents.values()))["provenance"]["path"]
    with path.open(encoding="utf-8", newline="") as f:
        for ordinal, row in enumerate(csv.DictReader(f), 1):
            key = row["OrgId"], row["IncidentId"]
            if key in incidents:
                native[key].append((ordinal, row))
            if ordinal % 2000000 == 0:
                log(f"Source replay: checked {ordinal:,} GUIDE rows")
    for key, benchmark in incidents.items():
        found = native[key]
        if ([i for i, _ in found] != benchmark["provenance"]["row_ordinals"] or
                visible_incident([r for _, r in found]) != benchmark["state"] or
                {r["IncidentGrade"] for _, r in found} != {benchmark["expected"]["incident_grade"]}):
            raise ValueError("GUIDE complete incident/grade replay mismatch")
    result["incident_triage"] = {"records_replayed": len(incidents), "source_eof_rows": ordinal,
                                  "complete_incidents_and_unanimous_provider_grades": True}
    flows = defaultdict(dict)
    for r in by_family["network_defense"]:
        flows[r["provenance"]["path"]][r["provenance"]["csv_record"]] = r
    count = 0
    for rel, selected in flows.items():
        with (collection / rel).open(encoding="utf-8", newline="") as f:
            for ordinal, row in enumerate(csv.DictReader(f), 1):
                if ordinal not in selected:
                    continue
                b = selected[ordinal]
                if ({k: row[k] for k in FLOW_FIELDS} != b["state"]["flow"] or
                        row["Label"] != b["metadata"]["publisher_label"] or
                        label_class(row["Label"]) != b["expected"]["flow_class"]):
                    raise ValueError("flow native annotation replay mismatch")
                count += 1
    if count != len(by_family["network_defense"]):
        raise ValueError("missing selected native flows")
    result["network_defense"] = {"records_replayed": count, "native_labels_verified": True}
    # Endpoint: recover each complete event by native line range and hash.
    refs, recovered = defaultdict(dict), {}
    for r in by_family["endpoint_investigation"]:
        for ref in r["provenance"]["native_refs"]:
            refs[ref["path"], ref.get("member")][ref["line"]] = ref
    for (rel, member), selected in refs.items():
        archive = zipfile.ZipFile(collection / rel) if member else None
        with (archive.open(member) if archive else (collection / rel).open("rb")) as f:
            for start, end, raw, fmt in native_records(f):
                if start not in selected:
                    continue
                ref = selected[start]
                if end != ref["end_line"] or hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest() != ref["native_line_sha256"]:
                    raise ValueError("native endpoint record receipt mismatch")
                event = xml_event(raw) if fmt == "xml" else json.loads(raw)
                event["EventID"] = int(event["EventID"])
                recovered[rel, member, start] = {k: event[k] for k in VISIBLE if k in event}
        if archive:
            archive.close()
    for r in by_family["endpoint_investigation"]:
        events = [recovered[ref["path"], ref.get("member"), ref["line"]] for ref in r["provenance"]["native_refs"]]
        if any(any(e.get(k) != v for k, v in projected.items()) for e, projected in zip(events, r["state"]["events"])):
            raise ValueError("native endpoint projection mismatch")
        if r["metadata"]["task"].startswith("join_"):
            instruction = r["questions"]["application"]["instructions"]
            target = int(re.search(r"linked to event (\d+)", instruction)[1]) - 1
            # Derive the answer from the visible projection only, not hidden Image/User.
            shown = r["state"]["events"]
            matches = [e for i, e in enumerate(shown) if i != target and joined(e, shown[target])]
            gold = {field: matches[0][key] if len(matches) == 1 else UNKNOWN
                    for field, key in (("application", "Image"), ("credential_account", "User"))}
        else:
            event = events[0]
            gold = {"observed_effect": EFFECTS.get(event["EventID"], UNKNOWN)}
            for field, key in (("application", "Image"), ("credential_account", "User")):
                if field in r["questions"]:
                    gold[field] = event.get(key) or UNKNOWN
        if gold != r["expected"]:
            raise ValueError("endpoint projected-evidence answer replay mismatch")
    result["endpoint_investigation"] = {"records_replayed": len(by_family["endpoint_investigation"]),
                                         "native_events_replayed": len(recovered), "visible_evidence_labels_verified": True}
    # Policy: archive binding + closed spans; optional fresh parser invocation never executes source.
    programs = {r["metadata"]["source_record_id"]: r for r in by_family["authorization_policy"]}
    policy_input = []
    path = collection / next(iter(programs.values()))["provenance"]["path"]
    with gzip.open(path, "rb") as f:
        for ordinal, raw in enumerate(f, 1):
            native = json.loads(raw)
            if native["id"] not in programs:
                continue
            b = programs[native["id"]]
            p, text = b["provenance"], native["text"]
            if (ordinal != p["source_line"] or hashlib.sha256(raw).hexdigest() != p["source_line_sha256"] or
                    hashlib.sha256(text.encode()).hexdigest() != p["source_text_sha256"] or
                    b["state"].split("\n", 1)[1] != text):
                raise ValueError("policy native program replay mismatch")
            contract = b["metadata"]["behavior_evidence"]
            if not valid_contract(contract, text):
                raise ValueError("policy behavior evidence contract mismatch")
            observed = {o["operation"] for o in contract["operations"]}
            if b["expected"] != {op: "requires_approval" if op in observed else "permitted" for op in OPERATIONS}:
                raise ValueError("policy answer does not follow explicit authored rule")
            policy_input.append({"id": native["id"], "text": text})
    if len(policy_input) != len(programs):
        raise ValueError("selected policy source missing")
    reparse = False
    if typescript:
        completed = subprocess.run(["node", str(sev_root / "scripts/swarmtraces_ast.mjs"), str(typescript)],
                                   input="".join(json.dumps(r) + "\n" for r in policy_input), text=True,
                                   capture_output=True, check=True)
        analyses = [json.loads(line) for line in completed.stdout.splitlines()]
        if {a["id"] for a in analyses} != set(programs):
            raise ValueError("fresh AST analysis coverage mismatch")
        for a in analyses:
            b = programs[a["id"]]
            focal = int(re.search(r"For fetch call (\d+)", b["questions"]["plain_text_display"]["instructions"])[1])
            calls = [c for c in a.get("calls", []) if c["ordinal"] == focal]
            if len(calls) != 1 or calls[0].get("response_behaviors") != b["metadata"]["behavior_evidence"]:
                raise ValueError("fresh AST behavior analysis differs from cached annotation")
        reparse = True
    result["authorization_policy"] = {"records_replayed": len(programs), "source_programs_and_spans_verified": True,
                                       "fresh_static_AST_reparse": reparse,
                                       "independent_human_adjudication": False}
    return result
