"""Projection, matched-pair, provenance and source-replay gates for v2."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..common import SPLITS, TRACKS, digest, file_hash, summary
from ..endpoint import IDENTITY, VISIBLE, native_records, xml_event
from ..network import FIELDS, label_class
from .endpoint import FIELDS as ENDPOINT_FIELDS, open_source, solve
from .network import IOT_FIELDS, iot_rows
from .network_rules import FIELDS as RULE_FIELDS, decide as network_decide
from .policy import decide
from .policy import dependency
from .common import payload_hash
from .language import normalize
from . import cic, workplace


def validate(values, originals, protected):
    by_old = {r["id"]: r for r in originals}
    groups, states, ids, events, templates, captures = {}, {}, set(), {}, {}, {}
    pairs = defaultdict(list)
    decision_pairs = defaultdict(list)
    for row in values:
        family, split, m = row["family"], row["split"], row["metadata"]
        if family not in TRACKS or split not in SPLITS or row["id"] in ids:
            raise ValueError("Invalid or duplicate v2 record")
        ids.add(row["id"])
        group = m["split_group"]
        if groups.setdefault(group, split) != split:
            raise ValueError("Independent source component crosses folds")
        if group in protected["groups"] or digest(row["state"]) in protected["states"]:
            raise ValueError("Known Sev training evidence admitted")
        if states.setdefault(digest(row["state"]), split) != split:
            raise ValueError("Identical model evidence crosses folds")
        if set(row["questions"]) != set(row["expected"]):
            raise ValueError("Mismatched answer fields")
        if payload_hash(row["state"], row["questions"]) != row["_evaluation"]["payload_sha256"]:
            raise ValueError("Modified model payload")
        for key, q in row["questions"].items():
            if set(q) != {"type", "instructions", "criteria"} or q["type"] != "choice" or row["expected"][key] not in q["criteria"]:
                raise ValueError("Gold or annotation exposed in question schema")
        state = row["state"]
        if m.get("match_pair"):
            pairs[(family, m["match_pair"])].append(row)
        for pair in m.get("decision_pairs", []):
            decision_pairs[(pair["axis"], pair["id"])].append(row)
        if family == "authorization_policy":
            if m["task"].startswith("workplace_"):
                old = by_old[row["provenance"]["fold_anchor_v01_id"]]
                if old["split"] != split or old["metadata"]["split_group"] != group or workplace.decide(state) != row["expected"]["authorization"]:
                    raise ValueError("Workplace evidence, policy or source fold disagrees")
                if "data_flow" in state and state["data_flow"]["classification"] != row["expected"]["data_classification"]:
                    raise ValueError("Workplace data classification disagrees")
                if "access_request" in state and ("covered" if state["access_request"]["role"] in state["policy"]["approval_roles"] else "outside") != row["expected"]["role_scope"]:
                    raise ValueError("Workplace role scope disagrees")
                continue
            old = by_old[row["provenance"]["v01_id"]]
            if old["split"] != split or old["metadata"]["split_group"] != group:
                raise ValueError("Recovered policy source component moved")
            template = m["template_sha256"]
            if templates.setdefault(template, split) != split or template in protected["templates"]:
                raise ValueError("Policy source template leakage")
            if set(state) != {"scope", "program", "target_call", "assessment_callback", "assessed_operation", "destination", "grants"}:
                raise ValueError("Unexpected policy visible fields")
            text = old["state"].split("\n", 1)[1]
            edit = row["provenance"]["intervention"]
            edits = [{"span": edit["original_span"], "replacement": edit["replacement"]}]
            if edit["reader_edit"]:
                edits.append(edit["reader_edit"])
            for change in sorted(edits, key=lambda e: e["span"][0], reverse=True):
                a, b = change["span"]
                text = text[:a] + change["replacement"] + text[b:]
            if text != state["program"] or digest(text) != edit["modified_program_sha256"]:
                raise ValueError("Policy variant cannot be replayed from native ancestor")
            if decide(state, row["questions"]["authorization"]["instructions"]) != row["expected"]["authorization"]:
                raise ValueError("Visible policy evidence cannot justify answer")
            if dependency(state) != row["expected"]["data_dependency"]:
                raise ValueError("Program dependency answer disagrees")
        elif family == "endpoint_investigation":
            if m["diagnostic_only"]:
                old = by_old[m["v01_id"]]
                old = normalize({**old, "metadata": {**old["metadata"], "diagnostic_only": True}})
                if any(row[k] != old[k] for k in ("state", "questions", "expected", "split")):
                    raise ValueError("Carried field diagnostic changed")
            else:
                if set(state) != {"scope", "target_event", "events", "request"} or any(set(e) - {*ENDPOINT_FIELDS, "ref"} for e in state["events"]):
                    raise ValueError("Endpoint state includes unsupported fields")
                answer, support = solve(state, m["task"])
                if row["expected"] != {"application": answer, "supporting_events": support}:
                    raise ValueError("Endpoint answer or citation lacks a visible chain")
                if not 7 <= len(state["events"]) <= 11 or "supporting_events" not in row["questions"]:
                    raise ValueError("Hard endpoint is missing distractors or citation")
            for ref in row["provenance"]["native_refs"]:
                if ref["sha256"] in protected["archives"] or ref["event_identity"] in protected["native_events"]:
                    raise ValueError("Known Sev native capture included")
                if captures.setdefault(ref["sha256"], split) != split or events.setdefault(ref["event_identity"], split) != split:
                    raise ValueError("Native endpoint capture or event crosses splits")
        elif family == "incident_triage":
            old = by_old[m["v01_id"]]
            old = normalize(old)
            if any(row[k] != old[k] for k in ("state", "questions", "expected", "split")):
                raise ValueError("GUIDE incident or organization partition changed")
        else:
            if set(state) != {"scope", "flow"}:
                raise ValueError("Network outcomes exposed in state")
            allowed = RULE_FIELDS if m["task"] == "native_network_boundary_rule" else cic.FIELDS if m["source_dataset"] == "CSE-CIC-IDS2018" else IOT_FIELDS if m["source_dataset"] == "IoT-23" else FIELDS
            if set(state["flow"]) - set(allowed):
                raise ValueError("Network label/UID exposed")
            capture = m["capture_id"]
            if captures.setdefault(capture, split) != split:
                raise ValueError("Network capture crosses folds")
            family_key = (m["source_dataset"], m["capture_family"].casefold())
            if captures.setdefault(family_key, split) != split:
                raise ValueError("Related network capture family crosses folds")
            if m["task"] == "native_network_boundary_rule" and network_decide(state, m["rule"]) != row["expected"]["boundary_action"]:
                raise ValueError("Network boundary rule disagrees")
    policy_pairs, endpoint_pairs = 0, 0
    for (axis, _), pair in decision_pairs.items():
        if len(pair) != 2 or {r["expected"]["authorization"] for r in pair} != {"permitted", "requires_approval"}:
            raise ValueError("Policy, approval or destination intervention does not reverse the decision")
        if axis.startswith("policy"):
            if pair[0]["state"] != pair[1]["state"]:
                raise ValueError("Policy-only pair changed non-policy evidence")
            questions = [{key: {k: v for k, v in q.items() if not (key == "authorization" and k == "instructions")} for key, q in r["questions"].items()} for r in pair]
            if questions[0] != questions[1]:
                raise ValueError("Option ordering exposes the policy-only intervention")
        elif pair[0]["questions"] != pair[1]["questions"]:
            raise ValueError("Evidence-only decision intervention changed policy or options")
    for (family, _), pair in pairs.items():
        if family == "authorization_policy":
            if len(pair) != 2 or pair[0]["expected"] == pair[1]["expected"]:
                raise ValueError("Policy pair does not change the required answer")
            left, right = sorted(pair, key=lambda r: r["metadata"]["variant"])
            axis = left["metadata"]["intervention"]
            if axis == "policy" and left["metadata"]["task"] == "program_approval" and left["state"] != right["state"]:
                raise ValueError("Policy-only intervention changed evidence")
            if axis != "policy" and left["questions"]["authorization"]["instructions"] != right["questions"]["authorization"]["instructions"]:
                raise ValueError("Evidence-only intervention changed policy")
            if left["metadata"]["task"] == "program_approval":
                left_blind = {k: v for k, v in left["state"].items() if k != "program"}
                right_blind = {k: v for k, v in right["state"].items() if k != "program"}
                if left_blind != right_blind or left["questions"] != right["questions"]:
                    raise ValueError("Program-blind payloads differ within a dependency pair")
            policy_pairs += 1
        elif family == "endpoint_investigation":
            if len(pair) != 2 or sum(r["metadata"]["insufficient_evidence"] for r in pair) != 1:
                raise ValueError("Missing-link endpoint pair is incomplete")
            def shape(r):
                return (len(r["state"]["events"]), r["state"]["target_event"],
                        [(e["EventID"], tuple(sorted(e))) for e in r["state"]["events"]],
                        sorted(e.get("Image", "") for e in r["state"]["events"]),
                        r["questions"])
            if shape(pair[0]) != shape(pair[1]):
                raise ValueError("Endpoint count, field schema or option count exposes missing links")
            endpoint_pairs += 1
    return {"checks": {"group_disjoint": True, "evidence_disjoint": True, "known_Sev_excluded": True,
                       "labels_isolated": True, "payload_receipts": True, "matched_policy_and_evidence_necessary": True,
                       "endpoint_citations_and_missing_links": True}, "policy_pairs": policy_pairs, "endpoint_pairs": endpoint_pairs,
            "policy_decision_pairs_by_axis": dict(Counter(axis for axis, _ in decision_pairs)),
            "counts": {track: summary([r for r in values if r["family"] == track]) for track in TRACKS},
            "partitions": {s: {t: summary([r for r in values if r["split"] == s and r["family"] == t]) for t in TRACKS} for s in SPLITS}}


def replay(values, originals, reports, collection, log=print):
    inherited = {r["id"]: r for r in originals}
    checked_files, refs_checked, native_selected = set(), 0, 0
    # Previously replayed native rows are reused exactly; pin their source bytes
    # again, with the v1 manifest and audit included in the v2 ancestry receipts.
    paths = {r["provenance"]["path"]: r["provenance"]["sha256"] for r in originals
             if r["family"] in ("incident_triage", "authorization_policy", "network_defense")}
    for path, pin in paths.items():
        if file_hash(Path(collection) / path) != pin:
            raise ValueError("Native ancestor source changed")
        checked_files.add(path)
    endpoint_rows = [r for r in values if r["family"] == "endpoint_investigation" and not r["metadata"]["diagnostic_only"]]
    by_path = defaultdict(list)
    for row in endpoint_rows:
        for event, ref in zip(row["state"]["events"], row["provenance"]["native_refs"]):
            by_path[ref["path"]].append((row, event, ref))
    sources = {r["path"]: r for r in reports["endpoint_investigation"]["sources"]}
    for path, wanted in by_path.items():
        wanted_by_line = defaultdict(list)
        for row, event, ref in wanted:
            wanted_by_line[ref["line"]].append((row, event, ref))
        seen = set()
        with open_source(collection, sources[path]) as f:
            for line, end, raw, fmt in native_records(f):
                if line not in wanted_by_line:
                    continue
                native = xml_event(raw) if fmt == "xml" else json.loads(raw)
                native["EventID"] = int(native["EventID"])
                for row, event, ref in wanted_by_line[line]:
                    if end != ref["end_line"] or hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest() != ref["native_line_sha256"]:
                        raise ValueError("Native endpoint event receipt mismatch")
                    fields = set(ENDPOINT_FIELDS)
                    if event["ref"] == row["state"]["target_event"]:
                        fields -= {"Image", "User", "CommandLine"}
                    projection = {k: native[k] for k in ENDPOINT_FIELDS if k in fields and k in native}
                    if projection != {k: v for k, v in event.items() if k != "ref"}:
                        raise ValueError("Endpoint event fields were fabricated")
                    refs_checked += 1
                seen.add(line)
        if seen != set(wanted_by_line):
            raise ValueError("Endpoint event missing from source EOF")
        checked_files.add(path)
    network = [r for r in values if r["family"] == "network_defense" and not r["metadata"].get("v01_id")]
    by_path = defaultdict(list)
    for row in network:
        by_path[row["provenance"]["path"]].append(row)
    for path, wanted in by_path.items():
        source = Path(collection) / path
        if file_hash(source) != wanted[0]["provenance"]["sha256"]:
            raise ValueError("Network source receipt changed")
        by_line = defaultdict(list)
        for row in wanted:
            by_line[row["provenance"].get("csv_record", row["provenance"].get("line"))].append(row)
        dataset = wanted[0]["metadata"]["source_dataset"]
        with source.open(newline="") as f:
            iterator = enumerate(csv.DictReader(f), 1) if dataset in ("CTU-13", "CSE-CIC-IDS2018") else iot_rows(source) if dataset == "IoT-23" else ((i, json.loads(line)) for i, line in enumerate(f, 1))
            seen = set()
            for ordinal, native in iterator:
                if ordinal not in by_line:
                    continue
                for row in by_line[ordinal]:
                    flow = row["state"]["flow"]
                    if any(native[k] != v for k, v in flow.items()):
                        raise ValueError("Network fields differ from native row")
                    if dataset != "Splunk attack_data":
                        y = label_class(native["Label"]) if dataset == "CTU-13" else cic.binary(native["Label"]) if dataset == "CSE-CIC-IDS2018" else native["label"]
                        if row["expected"]["flow_class"] != y:
                            raise ValueError("Native network annotation differs")
                    native_selected += 1
                seen.add(ordinal)
            if seen != set(by_line):
                raise ValueError("Selected flow not found at complete EOF")
        checked_files.add(path)
        log(f"Source replay: {dataset}, {len(wanted)} selected cases", flush=True)
    return {"source_files_pinned": len(checked_files), "hard_endpoint_native_refs_replayed": refs_checked,
            "new_network_native_rows_replayed": native_selected, "inherited_v01_native_replay": True,
            "policy_modified_program_replay": "performed by structural audit and complete static reparse",
            "human_adjudication": False}
