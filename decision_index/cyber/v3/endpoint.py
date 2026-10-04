"""Native multi-event lifetime and ancestry questions with required citations."""
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from ..common import choice, digest, file_hash, rank
from ..endpoint import IDENTITY, VISIBLE, joined, lifetime, native_records, utc, xml_event
from .common import carry, make

FIELDS = (*VISIBLE, "ParentProcessGuid", "CommandLine")
UNKNOWN = "insufficient_evidence"


@contextmanager
def open_source(collection, source):
    path = Path(collection) / source["path"]
    if file_hash(path) != source["sha256"]:
        raise ValueError("endpoint source differs from pinned v0.1 replay")
    if source.get("member"):
        with zipfile.ZipFile(path) as z, z.open(source["member"]) as f:
            yield f
    else:
        with path.open("rb") as f:
            yield f


def scan(collection, source):
    creations, activities, distractors = defaultdict(list), [], []
    count = 0
    with open_source(collection, source) as f:
        for line, end, raw, fmt in native_records(f):
            native = xml_event(raw) if fmt == "xml" else json.loads(raw)
            if native is None or native.get("Channel") != "Microsoft-Windows-Sysmon/Operational":
                continue
            native["EventID"] = int(native["EventID"])
            count += 1
            try:
                utc(native.get("UtcTime"))
            except (ValueError, TypeError):
                continue
            if not lifetime(native):
                continue
            identity = digest({k: native[k] for k in IDENTITY if k in native})
            event = {k: native[k] for k in FIELDS if k in native}
            ref = {k: source[k] for k in ("path", "sha256", "member") if k in source}
            ref.update(line=line, end_line=end, native_line_sha256=hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(), event_identity=identity)
            item = {"event": event, "ref": ref}
            if event["EventID"] == 1 and event.get("Image"):
                creations[lifetime(event)].append(item)
            if event["EventID"] in (3, 5, 7, 10, 11, 12, 13, 22, 23, 26):
                activities.append(item)
                if len(activities) > 256:
                    activities = sorted(activities, key=lambda x: rank("hard-activity", x["ref"]))[:128]
            distractors.append(item)
            if len(distractors) > 1024:
                distractors = sorted(distractors, key=lambda x: rank("hard-distractor", x["ref"]))[:512]
    return creations, sorted(activities, key=lambda x: rank("hard-activity", x["ref"]))[:128], distractors, count


def matches(creations, event, parent=False):
    key = lifetime(event, "ParentProcessGuid" if parent else "ProcessGuid")
    return [x for x in creations.get(key, []) if key and utc(x["event"]["UtcTime"]) <= utc(event["UtcTime"])]


def solve(state, task):
    """Visible-event graph baseline, independent of generated answer metadata."""
    events = state["events"]
    target = next(e for e in events if e["ref"] == state["target_event"])
    creations = [e for e in events if e["EventID"] == 1 and joined(e, target)]
    chains = []
    for creation in creations:
        if task == "process_lifetime":
            chains.append((creation.get("Image"), [creation["ref"], target["ref"]]))
        else:
            key = lifetime(creation, "ParentProcessGuid")
            for parent in events:
                if (parent["EventID"] == 1 and key and lifetime(parent) == key
                        and utc(parent["UtcTime"]) <= utc(creation["UtcTime"])):
                    chains.append((parent.get("Image"), [parent["ref"], creation["ref"], target["ref"]]))
    names = {name for name, _ in chains if name}
    if len(names) != 1:
        return UNKNOWN, f"{target['ref']}:insufficient"
    answer = next(iter(names))
    # Duplicate event observations admit alternative minimal support sets. The
    # builder excludes those cases rather than marking one equivalent set wrong.
    supports = {"+".join(sorted(refs, key=lambda s: int(s[1:]))) for name, refs in chains if name == answer}
    if len(supports) != 1:
        raise ValueError("ambiguous equivalent citation sets")
    return answer, supports.pop()


def case(native, source_group, split, items, chain, target, task, missing, option_images=None):
    ordered = sorted(items, key=lambda x: (x["ref"]["line"], x["ref"]["native_line_sha256"]))
    events, refs = [], []
    local = {}
    for i, item in enumerate(ordered, 1):
        event = dict(item["event"])
        event["ref"] = f"E{i}"
        local[item["ref"]["native_line_sha256"]] = event["ref"]
        if item is target:
            event.pop("Image", None)
            event.pop("User", None)
            event.pop("CommandLine", None)
        events.append(event)
        refs.append(item["ref"])
    target_ref = local[target["ref"]["native_line_sha256"]]
    state = {"scope": "Event fields from one capture, in source order. Unlisted records and fields are unavailable. "
                      "Image, User and CommandLine are omitted from the target event.",
             "target_event": target_ref, "events": events}
    answer, support = solve(state, task)
    options = {name: name for name in (option_images or {e["Image"] for e in events if e.get("Image")})}
    options[UNKNOWN] = "Insufficient evidence"
    if not 3 <= len(options) <= 12:
        return None
    evidence_options = {f"{target_ref}:insufficient": f"{target_ref}; insufficient evidence"}
    # Plausible support chains must cite creation records and the target, never
    # expose a hidden source ID or ordinal as an answer cue.
    candidates = [e for e in events if e["EventID"] == 1]
    for event in candidates[:8]:
        refs2 = [event["ref"], target_ref]
        if task == "ancestor_lifetime":
            others = [e for e in candidates if e is not event]
            for other in others[:4]:
                key = "+".join(sorted({other["ref"], *refs2}, key=lambda s: int(s[1:])))
                evidence_options[key] = key
        else:
            key = "+".join(sorted(refs2, key=lambda s: int(s[1:])))
            evidence_options[key] = key
    if support not in evidence_options:
        return None
    # Limit distractor support alternatives by a label-independent hash; the gold
    # set is admitted as a candidate graph chain, not appended after selection.
    if len(evidence_options) > 33:
        return None
    identity = [source_group, task, refs, target["ref"], missing]
    instr = (f"For target {target_ref}, identify the {'parent of the process that produced the activity' if task == 'ancestor_lifetime' else 'process that produced the activity'}. "
             "Require process-creation records with the same host, valid matching ProcessGuid (ParentProcessGuid for "
             "the ancestry edge), and creation times no later than their child activity or creation. Return insufficient "
             "evidence if a link is missing or multiple executable answers remain. Do not use PID or path similarity as a join.")
    option_identity = [source_group, task, [x["ref"] for x in chain], target["ref"]]
    q, y = choice(option_identity, "application", instr, options, answer)
    eq, ey = choice(option_identity, "supporting_events", "Choose the related events that support the process answer, "
                    "including the target. For insufficient evidence choose the target reference with no complete chain.", evidence_options, support)
    return make("endpoint_investigation", identity, split, source_group, state,
                {"application": q, "supporting_events": eq}, {"application": y, "supporting_events": ey},
                {"task": task, "diagnostic_only": False, "source_kind": "captured_security_lab_with_explicit_evidence_omission",
                 "match_pair": digest([source_group, task, [x["ref"] for x in chain], target["ref"]]),
                 "variant": int(missing), "review_status": "pending_independent_security_review",
                 "evidence_count": len(events), "insufficient_evidence": answer == UNKNOWN},
                {"path": refs[0]["path"], "sha256": refs[0]["sha256"], "native_refs": refs,
                 "omission": "Required creation link deliberately omitted" if missing else None})


def build(originals, report, collection, per_group_task=8, log=print):
    groups = {r["metadata"]["split_group"]: r["split"] for r in originals}
    by_path = {s["path"]: s for s in report["sources"]}
    outputs, scanned = [], []
    for group in report["components"]:
        if group["id"] not in groups or group["blocked"]:
            continue
        candidates = defaultdict(list)
        for path in group["sources"]:
            source = by_path[path]
            creations, activities, pool, count = scan(collection, source)
            scanned.append({**source, "native_events_replayed": count, "eof_verified": True})
            for target in activities:
                process = matches(creations, target["event"])
                if not process or len({x["event"].get("Image") for x in process}) != 1:
                    continue
                process = min(process, key=lambda x: x["ref"]["line"])
                for task in ("process_lifetime", "ancestor_lifetime"):
                    chain = [process]
                    if task == "ancestor_lifetime":
                        parent = matches(creations, process["event"], parent=True)
                        if not parent or len({x["event"].get("Image") for x in parent}) != 1:
                            continue
                        parent = min(parent, key=lambda x: x["ref"]["line"])
                        if parent is process:
                            continue
                        chain = [parent, process]
                    used = {x["ref"]["native_line_sha256"] for x in [*chain, target]}
                    chain_keys = {lifetime(x["event"]) for x in chain}
                    distractions = [x for x in pool if x["ref"]["native_line_sha256"] not in used
                                    and not (x["event"]["EventID"] == 1 and lifetime(x["event"]) in chain_keys)]
                    # Favor same host, same PID/path, and nearby native events.
                    distractions.sort(key=lambda x: (x["event"].get("Hostname") != target["event"]["Hostname"],
                        x["event"].get("ProcessId") != target["event"].get("ProcessId"),
                        abs(x["ref"]["line"] - target["ref"]["line"]), rank("endpoint-v2-distractors", x["ref"])))
                    diverse, seen_images = [], {x["event"].get("Image") for x in chain}
                    for item in distractions:
                        image = item["event"].get("Image")
                        if image and image not in seen_images:
                            diverse.append(item)
                            seen_images.add(image)
                            if len(diverse) == 2:
                                break
                    extras = diverse + [x for x in distractions if x not in diverse][:8 - len(diverse)]
                    if len(extras) < 5:
                        continue
                    # Replace a missing creation with a native creation of the
                    # same event type. Event count and type count stay identical.
                    extra_hashes = {x["ref"]["native_line_sha256"] for x in extras}
                    fixed_items = [*chain[1:], target, *extras]
                    original_slot = sum(x["ref"]["line"] < chain[0]["ref"]["line"] for x in fixed_items)
                    replacements = sorted((x for items in creations.values() for x in items), key=lambda x: rank("v3-replacement", x["ref"]))
                    replacement = next((x for x in replacements if x["event"]["EventID"] == chain[0]["event"]["EventID"]
                                        and set(x["event"]) == set(chain[0]["event"])
                                        and x["event"].get("Image") == chain[0]["event"].get("Image")
                                        and x["event"]["Hostname"].casefold() == chain[0]["event"]["Hostname"].casefold()
                                        and lifetime(x["event"]) not in chain_keys
                                        and sum(y["ref"]["line"] < x["ref"]["line"] for y in fixed_items) == original_slot
                                        and (x["ref"]["line"] < target["ref"]["line"]) == (chain[0]["ref"]["line"] < target["ref"]["line"])
                                        and x["ref"]["native_line_sha256"] not in extra_hashes | used), None)
                    if replacement is None:
                        continue
                    option_images = {x["event"]["Image"] for x in [*chain, target, *extras, replacement] if x["event"].get("Image")}
                    pair = []
                    for missing in (False, True):
                        supplied = [*(chain[1:] if missing else chain), target, *extras, *([replacement] if missing else [])]
                        try:
                            row = case(None, group["id"], groups[group["id"]], supplied, chain, target, task, missing, option_images)
                        except ValueError:
                            row = None
                        if row:
                            pair.append(row)
                    if len(pair) == 2 and pair[0]["questions"] == pair[1]["questions"] and pair[0]["expected"]["application"] != UNKNOWN and pair[1]["expected"]["application"] == UNKNOWN:
                        candidates[task].append(pair)
        for task, pairs in candidates.items():
            selected = sorted(pairs, key=lambda p: rank("endpoint-v2-final", p[0]["id"]))[:per_group_task]
            outputs.extend(r for pair in selected for r in pair)
        log(f"Endpoint v2: {len(outputs)} cases after {len(scanned)} complete capture replays", flush=True)
    # Preserve extraction as an explicitly excluded diagnostic, with a bounded
    # equal-group sample rather than allowing easy fields to dominate ranking.
    diagnostic = defaultdict(list)
    for row in originals:
        if not row["metadata"]["task"].startswith("join_"):
            diagnostic[row["metadata"]["split_group"]].append(row)
    for group, values in diagnostic.items():
        outputs.extend(carry(r, True) for r in sorted(values, key=lambda r: rank("diagnostic-v2", r["id"]))[:6])
    return outputs, {"sources": scanned, "cases": len(outputs), "tasks": dict(Counter(r["metadata"]["task"] for r in outputs)),
        "diagnostic_cases": sum(r["metadata"]["diagnostic_only"] for r in outputs),
        "independent_groups": len({r["metadata"]["split_group"] for r in outputs}),
        "support_required_for_case_credit": True, "human_review": "pending"}
