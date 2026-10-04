"""Source-backed Sysmon effects and host-scoped process joins, stdlib only."""
import hashlib
import heapq
import json
import re
import subprocess
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .common import assign_groups, choice, digest, file_hash, rank, receipt, record

COMMIT = "d9d40ef123d2c87d5d3df28c96bcab4f0faccc87"
CHANNEL = "Microsoft-Windows-Sysmon/Operational"
GUIDS = ("ProcessGuid", "ParentProcessGuid", "SourceProcessGUID", "TargetProcessGUID",
         "SourceProcessGuid", "TargetProcessGuid")
IDENTITY = ("Hostname", "EventID", "UtcTime", *GUIDS, "ProcessId", "Image", "TargetFilename", "TargetObject", "Details")
VISIBLE = ("EventID", "UtcTime", "Hostname", "ProcessGuid", "ProcessId", "Image", "User",
           "ParentImage", "TargetFilename", "TargetObject", "ImageLoaded")
UNKNOWN = "The supplied events do not establish this"
EFFECTS = {1: "A process was created", 11: "A file was created or overwritten", 13: "A registry value was set"}
NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"


def lifetime(event, field="ProcessGuid"):
    host, value = event.get("Hostname"), event.get(field)
    if not isinstance(host, str) or not host or not isinstance(value, str):
        return None
    try:
        guid = uuid.UUID(value)
    except ValueError:
        return None
    return (host.casefold(), str(guid)) if guid.int else None


def utc(value):
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z?", value or "")
    if not match:
        raise ValueError("unrecognized native UTC time")
    return datetime.fromisoformat(f"{match[1]}T{match[2]}"), int((match[3] or "").ljust(9, "0"))


def joined(creation, activity):
    return (creation.get("EventID") == 1 and lifetime(creation) is not None
            and lifetime(creation) == lifetime(activity) and utc(creation["UtcTime"]) <= utc(activity["UtcTime"]))


def xml_event(line):
    node = ET.fromstring(line)
    ns = NS if node.tag == NS + "Event" else ""
    system = node.find(ns + "System")
    if system is None:
        raise ValueError("Sysmon XML lacks System")
    provider = system.find(ns + "Provider")
    if provider is None or provider.get("Name") != "Microsoft-Windows-Sysmon":
        return None
    event = {child.get("Name"): child.text or "" for child in node.findall(f"{ns}EventData/{ns}Data")}
    event["EventID"] = int(system.findtext(ns + "EventID"))
    event["Hostname"] = system.findtext(ns + "Computer") or ""
    event["Channel"] = system.findtext(ns + "Channel") or CHANNEL
    return event


def native_records(lines):
    """Yield complete records with original line ranges, including multiline XML."""
    mode, buffer, start = None, b"", 0
    for ordinal, raw in enumerate(lines, 1):
        if mode is None and raw.strip():
            mode = "xml" if raw.lstrip().startswith(b"<Event") else "json"
        if mode != "xml":
            if raw.strip():
                yield ordinal, ordinal, raw, "json"
            continue
        if not buffer:
            raw = raw.lstrip()
            start = ordinal
        buffer += raw
        while b"</Event>" in buffer:
            end = buffer.index(b"</Event>") + len(b"</Event>")
            record_bytes, tail = buffer[:end], buffer[end:]
            if not re.match(rb"<Event(?:\s|>)", record_bytes):
                raise ValueError("unexpected data outside complete XML Event")
            end_line = start + record_bytes.count(b"\n")
            yield start, end_line, record_bytes, "xml"
            trimmed = tail.lstrip()
            start = end_line + tail[:len(tail) - len(trimmed)].count(b"\n")
            buffer = trimmed
        if len(buffer) > 2000000:
            raise ValueError("XML event exceeds complete-record byte bound")
    if buffer.strip():
        raise ValueError("incomplete XML event at source EOF")


def retain(heap, item, limit):
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def scan_capture(lines, source, *, scenario=None, native_limit=60):
    identities, lifetimes, types = set(), set(), Counter()
    direct, activities, creations = defaultdict(list), [], defaultdict(list)
    native_count, raw_lines = 0, 0
    for ordinal, end_line, raw, format_name in native_records(lines):
        raw_lines = end_line
        if format_name == "xml":
            event = xml_event(raw)
        else:
            event = json.loads(raw)
        if event is None:
            continue
        if not isinstance(event, dict):
            raise ValueError("native capture line must be an object")
        if event.get("Channel") != CHANNEL:
            continue
        if isinstance(event.get("EventID"), str):
            event["EventID"] = int(event["EventID"])
        native_count += 1
        identity = digest({key: event[key] for key in IDENTITY if key in event})
        identities.add(identity)
        for field in GUIDS:
            key = lifetime(event, field)
            if key:
                lifetimes.add(key)
        types[str(event.get("EventID"))] += 1
        try:
            utc(event.get("UtcTime"))
        except (ValueError, TypeError):
            continue
        if not event.get("Hostname"):
            continue
        event = {k: event[k] for k in VISIBLE if k in event}
        ref = {"path": source["path"], "sha256": source["sha256"], "line": ordinal, "end_line": end_line,
               "native_line_sha256": hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(), "event_identity": identity}
        if "member" in source:
            ref["member"] = source["member"]
        item = (-int(rank("endpoint-event", identity), 16), f"{source['sha256']}:{ordinal}", event, ref)
        family = {1: "process", 11: "file", 13: "registry", 5: "other_effect", 7: "other_effect"}.get(event.get("EventID"))
        if family:
            retain(direct[family], item, native_limit)
        key = lifetime(event)
        if event.get("EventID") == 1 and key and event.get("Image") and event.get("User"):
            creations[key].append(item)
        elif key:
            retain(activities, item, native_limit * 4)
    return {"source": source, "scenario": scenario, "identities": identities, "lifetimes": lifetimes,
            "direct": direct, "activities": activities, "creations": creations,
            "native_events": native_count, "source_lines": raw_lines, "event_types": dict(types)}


def components(captures, protected):
    parents = list(range(len(captures)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, cap in enumerate(captures):
        keys = [("event", v) for v in cap["identities"]] + [("lifetime", v) for v in cap["lifetimes"]]
        if cap["scenario"]:
            keys.append(("scenario", cap["scenario"]))
        for key in keys:
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i, cap in enumerate(captures):
        groups[find(i)].append(cap)
    result = []
    for caps in groups.values():
        hashes = sorted(c["source"]["sha256"] for c in caps)
        blocked = bool(set(hashes) & protected["archives"])
        blocked |= any(c["identities"] & protected["native_events"] for c in caps)
        result.append({"id": "endpoint-component/" + digest(hashes), "captures": caps, "blocked": blocked})
    return sorted(result, key=lambda g: g["id"])


def make_case(group, split, task, items, evidence, specifications):
    refs = [x[3] for x in items]
    case = [group, task, refs]
    qs, golds = {}, {}
    for field, instructions, options, answer in specifications:
        qs[field], golds[field] = choice(case, field, instructions, {v: v for v in options}, answer)
    return record("endpoint_investigation", case, split, group,
                  {"scope": "Selected native Sysmon fields in source order. Omitted values are unavailable. "
                            "Accounts are recorded credentials; final command outcomes and human owners are unestablished.",
                   "events": evidence}, qs, golds,
                  {"task": task, "source_kind": "captured_security_lab",
                   "label_basis": "visible native fields, documented event effects, and host-scoped process joins"},
                  {"path": refs[0]["path"], "sha256": refs[0]["sha256"], "native_refs": refs})


def direct_cases(group, split, cap):
    pools = {key: {x[2][key] for heap in cap["direct"].values() for x in heap if x[2].get(key)} for key in ("Image", "User")}
    for family, heap in cap["direct"].items():
        for item in heap:
            event = item[2]
            specs = [("observed_effect", "Which listed narrow effect is established by this event? "
                      "Do not infer a final command outcome.", [*EFFECTS.values(), UNKNOWN], EFFECTS.get(event["EventID"], UNKNOWN))]
            for field, key, wording in (("application", "Image", "Which executable is recorded in this event's Image field?"),
                                        ("credential_account", "User", "Which credential account is recorded in this event's User field?")):
                answer = event.get(key) or UNKNOWN
                alternatives = sorted(pools[key] - {answer}, key=lambda v: rank("endpoint-alternative", item[1], v))[:2]
                options = sorted({answer, *alternatives, UNKNOWN})
                if len(options) >= 2:
                    specs.append((field, wording, options, answer))
            yield make_case(group, split, family, [item], [event], specs)


def join_cases(group, split, cap):
    pool = [min(v, key=lambda x: x[1]) for v in cap["creations"].values()]
    pool.sort(key=lambda x: rank("creation-pool", x[1]))
    for activity in cap["activities"]:
        event = activity[2]
        matches = [x for x in cap["creations"].get(lifetime(event), []) if joined(x[2], event)]
        # Ambiguous process-creation attributes cannot yield a single answer.
        if len({(x[2]["Image"], x[2]["User"]) for x in matches}) > 1:
            continue
        matches.sort(key=lambda x: x[3]["line"])
        same_host = [x for x in pool if x[2]["Hostname"] == event["Hostname"] and not joined(x[2], event)]
        if matches:
            first = matches[0]
            other = next((x for x in same_host if x[2]["Image"] != first[2]["Image"] and x[2]["User"] != first[2]["User"]), None)
            if other:
                yield join_case(group, split, [first, other], activity, "join_supported")
        pair = None
        for i, first in enumerate(same_host):
            other = next((x for x in same_host[i + 1:] if x[2]["Image"] != first[2]["Image"] and x[2]["User"] != first[2]["User"]), None)
            if other:
                pair = [first, other]
                break
        if pair:
            yield join_case(group, split, pair, activity, "join_unsupported")


def join_case(group, split, pair, activity, task):
    items = sorted([*pair, activity], key=lambda x: x[3]["line"])
    target = next(i + 1 for i, x in enumerate(items) if x is activity)
    evidence = []
    for x in items:
        fields = ("EventID", "UtcTime", "Hostname", "ProcessGuid", "ProcessId") if x is activity else VISIBLE
        evidence.append({key: x[2][key] for key in fields if key in x[2]})
    matches = [x for x in pair if joined(x[2], activity[2])]
    if (task == "join_supported") != (len(matches) == 1):
        raise ValueError("native join labels disagree with projected visible evidence")
    specs = []
    for field, key, description in (("application", "Image", "executable"), ("credential_account", "User", "credential account")):
        answer = matches[0][2][key] if matches else UNKNOWN
        specs.append((field, f"Which {description} is linked to event {target} by a supplied process-creation event? "
                      "Require the same host and valid ProcessGuid, with a creation time no later than the target activity. "
                      "A path or process ID alone does not establish a lifetime.", [pair[0][2][key], pair[1][2][key], UNKNOWN], answer))
    return make_case(group, split, task, items, evidence, specs)


def build(collection, protected, *, max_zip_member=100000000, splunk_files=60, per_group_task=24, log=print):
    collection = Path(collection)
    otrf = collection / "datasets/mordor_security_datasets"
    tree = subprocess.check_output(["git", "-C", str(otrf), "ls-tree", "-r", COMMIT], text=True)
    blobs = {line.split("\t", 1)[1]: line.split("\t", 1)[0].split()[2] for line in tree.splitlines() if " blob " in line}
    captures, excluded = [], []
    for rel, blob in sorted(blobs.items()):
        if not rel.startswith("datasets/") or not rel.endswith(".zip"):
            continue
        path = otrf / rel
        raw = path.read_bytes()
        if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != blob:
            raise ValueError("OTRF source differs from pinned Git revision")
        sha = hashlib.sha256(raw).hexdigest()
        with zipfile.ZipFile(path) as z:
            members = z.infolist()
            if len(members) != 1 or not members[0].filename.endswith(".json") or members[0].file_size > max_zip_member:
                excluded.append({"path": str(path.relative_to(collection)), "sha256": sha,
                                 "reason": "not_one_complete_json_member_within_byte_bound"})
                continue
            member = members[0]
            source = {"path": str(path.relative_to(collection)), "sha256": sha, "bytes": path.stat().st_size,
                      "member": member.filename, "member_bytes": member.file_size,
                      "git_revision": COMMIT, "git_blob": blob, "license": "MIT"}
            with z.open(member) as f:
                cap = scan_capture(f, source)
        captures.append(cap)
        if len(captures) % 20 == 0:
            log(f"Endpoint: scanned {len(captures)} complete OTRF capture archives")
    splunk = collection / "datasets/splunk_attack_data"
    candidates = [p for p in (splunk / "datasets").rglob("*sysmon*.log") if 1000 <= p.stat().st_size <= 20000000]
    # Entire scenario families remain together even when multiple logs exist.
    chosen = sorted(candidates, key=lambda p: rank("splunk-capture", str(p.relative_to(splunk))))[:splunk_files]
    for path in chosen:
        source = {**receipt(path, collection), "license": "Apache-2.0 (Splunk attack_data source)"}
        stat = path.stat()
        with path.open("rb") as f:
            first = f.readline()
            if not first.lstrip().startswith(b"<Event"):
                excluded.append({**source, "reason": "not_native_XML_Event_stream"})
                continue
        with path.open("rb") as f:
            cap = scan_capture(f, source, scenario="splunk/" + "/".join(path.relative_to(splunk).parts[:3]))
        if (path.stat().st_size, path.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("Splunk capture changed during scan")
        captures.append(cap)
    groups = components(captures, protected)
    fresh = [g for g in groups if not g["blocked"] and any(c["native_events"] for c in g["captures"])]
    assignments = assign_groups([g["id"] for g in fresh], "endpoint-capture-components")
    output, rejections = [], Counter()
    for group in fresh:
        by_task = defaultdict(list)
        for cap in group["captures"]:
            for row in [*direct_cases(group["id"], assignments[group["id"]], cap),
                        *join_cases(group["id"], assignments[group["id"]], cap)]:
                if digest(row["state"]) in protected["states"]:
                    rejections["known_sev_state"] += 1
                    continue
                by_task[row["metadata"]["task"]].append(row)
        for task, candidates in by_task.items():
            output.extend(sorted(candidates, key=lambda r: rank("endpoint-final", r["id"]))[:per_group_task])
    present = {c["source"]["sha256"] for c in captures}
    if protected["archives"] - present:
        raise ValueError("previously used native captures were not scanned for overlap closure")
    log(f"Endpoint: {len(fresh)} fresh components, {len(output)} bounded evidence cases")
    return output, {"sources": [{**c["source"], "native_events": c["native_events"], "source_lines": c["source_lines"],
                                 "event_types": c["event_types"], "eof_verified": True} for c in captures],
                    "components": [{"id": g["id"], "blocked": g["blocked"],
                                    "sources": [c["source"]["path"] for c in g["captures"]]} for g in groups],
                    "excluded_sources": excluded, "rejections": dict(rejections),
                    "selection": {"max_complete_zip_member_bytes": max_zip_member, "splunk_files": splunk_files,
                                  "per_component_task_cap": per_group_task},
                    "limitations": ["Captured lab exercises, not field incident prevalence or actor-origin labels.",
                                    "Exact native events, process lifetimes, and Splunk scenario families connect split groups.",
                                    "Visible-field tasks admit exact deterministic parser baselines; joins remain bounded to supplied evidence."]}
