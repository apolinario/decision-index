"""Independent capture-family sampling; native publisher outcomes stay separate."""
import csv
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..common import assign_groups, choice, digest, file_hash, rank, receipt, read_json
from ..network import DESCRIPTIONS, FIELDS, label_class
from .common import carry, make

IOT_FIELDS = ("ts", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service", "duration",
              "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
              "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes")
IOT_DESCRIPTIONS = {"Benign": "Publisher-annotated benign flow", "Malicious": "Publisher-annotated malicious flow"}


def iot_rows(path):
    fields = None
    with Path(path).open() as f:
        for line_number, line in enumerate(f, 1):
            if line.startswith("#fields"):
                fields = line.split()[1:]
                if set(fields) != {*IOT_FIELDS, "uid", "tunnel_parents", "label", "detailed-label"}:
                    raise ValueError("Unexpected complete IoT-23 Zeek annotation schema")
            elif not line.startswith("#") and line.strip():
                parts = line.split()
                if not fields or len(parts) != len(fields):
                    raise ValueError("Malformed complete IoT-23 flow")
                yield line_number, dict(zip(fields, parts))


def select(iterator, labels, fields, namespace, cap):
    heaps, counts, scanned = {label: [] for label in labels}, Counter(), 0
    retained = {label: set() for label in labels}
    for ordinal, native in iterator:
        scanned += 1
        label = label_class(native["Label"]) if "Label" in native else native["label"]
        if label not in labels:
            raise ValueError(f"Unknown native publisher annotation: {label}")
        counts[label] += 1
        visible = {k: native[k] for k in fields}
        identity = digest(visible)
        if identity in retained[label]:
            continue
        item = (-int(rank("network-v2", namespace, identity), 16), identity, ordinal, visible, native)
        heap = heaps[label]
        if len(heap) < cap:
            heapq.heappush(heap, item)
            retained[label].add(identity)
        elif item[:2] > heap[0][:2]:
            old = heapq.heapreplace(heap, item)
            retained[label].remove(old[1])
            retained[label].add(identity)
    return heaps, counts, scanned


def build(originals, collection, cap=100, log=print):
    root = Path(collection).resolve()
    index = read_json(root / "external/esdb-network-v2/sources.json")
    # Correct descriptions from the complete publisher README, not the local
    # downloader's old inaccurate malware-name comments.
    import re
    for entry in index["entries"]:
        for role in ("source", "readme"):
            if file_hash(root / entry[role]["path"]) != entry[role]["sha256"]:
                raise ValueError("Downloaded network source changed")
        text = (root / entry["readme"]["path"]).read_text()
        if entry["dataset"] == "CTU-13":
            entry["family"] = re.search(r"Probable Name:\s*([^\n]+)", text)[1].strip()
    legacy_families = {}
    for capture in {r["metadata"]["capture_id"] for r in originals}:
        text = (root / f"datasets/stratosphere_ips/CTU-Malware-Capture-Botnet-{capture}/README.md").read_text()
        legacy_families[capture] = re.search(r"Probable Name:\s*([^\n]+)", text)[1].strip()
    # Related malware families and exact samples stay together. Network folds
    # are strengthened in v2; the immutable v1 release retains its old folds.
    new_families = {entry["dataset"] + "/" + entry["family"].casefold() for entry in index["entries"]}
    assignments = assign_groups(new_families | {"CTU-13/" + f.casefold() for f in legacy_families.values()}, "network-v2-capture-families")
    output, sources = [], []
    by_capture_class = defaultdict(list)
    for row in originals:
        by_capture_class[(row["metadata"]["capture_id"], row["expected"]["flow_class"])].append(row)
    for (capture, label), candidates in by_capture_class.items():
        for row in sorted(candidates, key=lambda r: rank("network-v2-legacy", r["id"]))[:cap]:
            item = carry(row)
            item["split"] = assignments["CTU-13/" + legacy_families[capture].casefold()]
            item["metadata"]["split_group"] = "network-family/CTU-13/" + legacy_families[capture].casefold()
            item["metadata"]["capture_id"] = "CTU-13/" + capture
            item["metadata"].update(source_dataset="CTU-13", source_publisher="Stratosphere Laboratory", capture_family=legacy_families[capture].casefold())
            output.append(item)
    for entry in index["entries"]:
        source = entry["source"]
        dataset, capture, family = entry["dataset"], entry["capture"], entry["family"]
        group = f"network-family/{dataset}/{family.casefold()}"
        split = assignments[dataset + "/" + family.casefold()]
        path = root / source["path"]
        if dataset == "CTU-13":
            with path.open(newline="") as f:
                heaps, counts, scanned = select(enumerate(csv.DictReader(f), 1), DESCRIPTIONS, FIELDS, capture, cap)
            descriptions, fields, task = DESCRIPTIONS, FIELDS, "publisher_flow_annotation"
        else:
            heaps, counts, scanned = select(iot_rows(path), IOT_DESCRIPTIONS, IOT_FIELDS, capture, cap)
            descriptions, fields, task = IOT_DESCRIPTIONS, IOT_FIELDS, "iot_publisher_flow_annotation"
        for label, heap in heaps.items():
            for _, identity, ordinal, visible, native in heap:
                state = {"scope": "One complete published flow projection. Source annotation, record UID and detailed outcome labels are withheld.", "flow": visible}
                q, y = choice([dataset, capture, identity], "flow_class", "Predict the publisher's annotation category for this flow. "
                    "This is a flow annotation task, not evidence of a named human actor or of successful payload execution.", descriptions, label)
                output.append(make("network_defense", [source["sha256"], ordinal], split, group, state, {"flow_class": q}, {"flow_class": y},
                    {"task": task, "diagnostic_only": False, "source_dataset": dataset, "source_publisher": "Stratosphere Laboratory",
                     "capture_id": dataset + "/" + capture, "capture_family": family.casefold(), "binary_disposition_judged": label != "Background",
                     "publisher_label": native.get("Label", native.get("detailed-label")), "label_basis": "native publisher annotation"},
                    {**source, "csv_record" if dataset == "CTU-13" else "line": ordinal, "flow_sha256": identity,
                     "upstream": "https://www.stratosphereips.org/datasets-ctu13" if dataset == "CTU-13" else "https://www.stratosphereips.org/datasets-iot23"}))
        sources.append({**entry, "split": split, "rows_scanned": scanned, "classes": dict(counts), "eof_verified": True})
        log(f"Network v2: {dataset} {capture} {family}, {scanned:,} complete flow records", flush=True)
    return output, {"sources": sources, "legacy_capture_subsample_per_class": cap,
        "new_capture_subsample_per_class": cap, "capture_count": len({r["metadata"]["capture_id"] for r in output}),
        "families": sorted({r["metadata"]["capture_family"] for r in output}),
        "source_datasets": ["CTU-13", "IoT-23"], "publisher_count": 1,
        "limitations": ["Two collections share one publisher/lab; this does not establish independent-lab transfer.",
                        "Network folds are reassigned by malware family. Shared IP/device fingerprints can recur; report an address-blind classifier.",
                        "Balanced subsampling changes prevalence. CTU Background is unjudged and excluded from binary detection."]}
