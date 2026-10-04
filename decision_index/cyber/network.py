"""Capture-disjoint, source-labeled flow cases; Background remains unjudged."""
import csv
import heapq
import re
from collections import Counter
from pathlib import Path

from .common import assign_groups, choice, digest, rank, receipt, record

CAPTURES = {"42": "capture20110810.binetflow", "45": "capture20110815.binetflow", "51": "capture20110818.binetflow"}
FIELDS = ("StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr", "Dport", "State",
          "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes")
DESCRIPTIONS = {"Botnet": "Publisher-labeled botnet-related flow",
                "Normal": "Publisher-verified normal flow",
                "Background": "Background traffic; normal versus botnet activity is unjudged"}


def label_class(value):
    matches = set(re.findall(r"(?:^|[-=])(Botnet|Normal|Background)(?=[-]|$)", value))
    if len(matches) != 1:
        raise ValueError("unrecognized or ambiguous CTU publisher flow label")
    return matches.pop()


def build(collection, protected, *, per_class=300, log=print):
    root = Path(collection)
    assignments = assign_groups(CAPTURES, "ctu13-captures")
    output, reports = [], []
    for capture, name in CAPTURES.items():
        path = root / f"datasets/stratosphere_ips/CTU-Malware-Capture-Botnet-{capture}/detailed-bidirectional-flow-labels/{name}"
        source = receipt(path, root)
        stat = path.stat()
        heaps = {label: [] for label in DESCRIPTIONS}
        labels, scan = Counter(), 0
        seen_selected = {label: set() for label in DESCRIPTIONS}
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, strict=True)
            if not reader.fieldnames or not {*FIELDS, "Label"} <= set(reader.fieldnames):
                raise ValueError("CTU flow schema missing required fields")
            for scan, native in enumerate(reader, 1):
                if scan % 1000000 == 0:
                    log(f"CTU {capture}: scanned {scan:,} labeled flows")
                if None in native or any(v is None for v in native.values()):
                    raise ValueError(f"malformed flow at CSV record {scan}")
                label = label_class(native["Label"])
                labels[label] += 1
                flow = {key: native[key] for key in FIELDS}
                flow_id = digest(flow)
                if flow_id in seen_selected[label]:
                    continue
                priority = int(rank("network-flow", capture, flow_id), 16)
                item = (-priority, flow_id, scan, flow, native["Label"])
                heap = heaps[label]
                if len(heap) < per_class:
                    heapq.heappush(heap, item)
                    seen_selected[label].add(flow_id)
                elif item[:2] > heap[0][:2]:
                    removed = heapq.heapreplace(heap, item)
                    seen_selected[label].discard(removed[1])
                    seen_selected[label].add(flow_id)
        if (path.stat().st_size, path.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("network source changed during full scan")
        for label, heap in heaps.items():
            for _, flow_id, ordinal, flow, native_label in heap:
                state = {"scope": "One complete published bidirectional flow record. No packet payload or label supplied.",
                         "flow": flow}
                if digest(state) in protected["states"]:
                    continue
                q, gold = choice(flow_id, "flow_class",
                                 "Predict the published annotation category for this flow. Background is unjudged traffic; "
                                 "it does not establish normal activity, maliciousness, actor identity, or execution success.",
                                 DESCRIPTIONS, label)
                output.append(record("network_defense", [source["sha256"], ordinal], assignments[capture],
                                     "ctu13-capture/" + capture, state, {"flow_class": q}, {"flow_class": gold},
                                     {"task": "publisher_flow_annotation", "source_kind": "captured_malware_lab_network",
                                      "capture_id": capture, "publisher_label": native_label,
                                      "label_basis": "native Label column", "binary_disposition_judged": label != "Background"},
                                     {"path": source["path"], "sha256": source["sha256"], "csv_record": ordinal,
                                      "flow_sha256": flow_id, "upstream": "https://www.stratosphereips.org/datasets-ctu13"}))
        reports.append({"source": source, "capture_id": capture, "split": assignments[capture],
                        "source_rows_scanned": scan, "eof_verified": True, "native_classes": dict(labels),
                        "selected_per_class": {k: len(h) for k, h in heaps.items()}})
    return output, {"sources": reports, "selection": {"per_capture_class_cap": per_class, "unit": "complete flow"},
                    "limitations": ["Three captured lab scenarios; capture-disjoint splitting does not establish field generalization.",
                                    "Background is unjudged, never relabeled as benign. Binary detection metrics exclude it.",
                                    "Equal class sampling changes prevalence. Source labels apply to flows, not operator origin."]}
