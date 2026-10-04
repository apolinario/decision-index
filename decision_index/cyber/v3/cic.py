"""Public publisher files, bounded complete-source sampling, one campaign fold."""
import csv
import hashlib
import heapq
import math
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from ..common import choice, digest, file_hash, rank, read_json, receipt, write_json
from .common import make

SOURCE_PAGE = "https://www.unb.ca/cic/datasets/ids-2018.html"
REGISTRY = "https://registry.opendata.aws/cse-cic-ids2018/"
PREFIX = "external/esdb-network-v3/cse-cic-ids2018"
FILES = {
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv": (358223333, '"46c0f45f8e6fe1edf1f08487448c102f-22"'),
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv": (333723605, '"e68ef27c09c98ba91fe3c6b1108a18b9-20"'),
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv": (107842858, '"5889e4b7e0a421070747a2441a2772d7-7"'),
}
FIELDS = ("Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
          "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean",
          "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Pkt Len Mean", "Pkt Len Std",
          "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt")
DESCRIPTIONS = {"Benign": "Benign", "Malicious": "Malicious"}
LABELS = {"Benign", "FTP-BruteForce", "SSH-Bruteforce", "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest", "Infilteration"}


def acquire(collection, log=print):
    root = Path(collection).resolve()
    directory = root / PREFIX
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / "sources.json"
    if index.exists():
        data = read_json(index)
        if all(file_hash(root / e["source"]["path"]) == e["source"]["sha256"] for e in data["entries"]):
            return data
        raise ValueError("Previously acquired publisher file changed")
    entries = []
    for name, (size, etag) in FILES.items():
        url = "https://cse-cic-ids2018.s3.amazonaws.com/" + urllib.parse.quote("Processed Traffic Data for ML Algorithms/" + name)
        path, partial = directory / name, directory / (name + ".part")
        if not path.exists():
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise ValueError("Partial source exceeds pinned byte bound")
            while offset < size:
                end = min(size - 1, offset + 16 * 1024 * 1024 - 1)
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}", "If-Match": etag})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers["ETag"] != etag or response.headers["Content-Range"] != f"bytes {offset}-{end}/{size}":
                        raise ValueError("Publisher range or source version changed")
                    data = response.read()
                if len(data) != end - offset + 1:
                    raise ValueError("Incomplete source range")
                with partial.open("ab") as output:
                    output.write(data)
                offset = end + 1
                log(f"CSE-CIC-IDS2018: {name}: {offset:,}/{size:,} bytes", flush=True)
            partial.rename(path)
        if path.stat().st_size != size:
            raise ValueError("Complete source size differs from publisher receipt")
        entries.append({"capture": name, "url": url, "etag": etag, "source": receipt(path, root)})
    data = {"dataset": "CSE-CIC-IDS2018", "source_page": SOURCE_PAGE, "registry": REGISTRY,
            "selection": "Predeclared complete days: brute force, DoS and infiltration. No head sampling.",
            "grouping": "One campaign; test-only publisher transfer. No model is fitted on this publisher.",
            "terms": "Publisher permits redistribution with dataset citation and registry link; lab traffic, not field incidents.",
            "entries": entries}
    write_json(index, data)
    return data


def binary(label):
    return "Benign" if label == "Benign" else "Malicious"


def build(collection, cap=100, log=print):
    root = Path(collection).resolve()
    index = read_json(root / PREFIX / "sources.json")
    output, reports = [], []
    for entry in index["entries"]:
        source = root / entry["source"]["path"]
        if file_hash(source) != entry["source"]["sha256"]:
            raise ValueError("CIC source changed")
        heaps, counts, rejected, scanned = {}, Counter(), Counter(), 0
        with source.open(newline="") as f:
            reader = csv.DictReader(f, strict=True)
            if not {*FIELDS, "Label"} <= set(reader.fieldnames):
                raise ValueError("Unexpected CICFlowMeter schema")
            for ordinal, native in enumerate(reader, 1):
                scanned = ordinal
                if None in native or any(v is None for v in native.values()):
                    raise ValueError("Incomplete native CSV record")
                label = native["Label"]
                counts[label] += 1
                if label == "Label" and all(native[k] == k for k in FIELDS):
                    rejected["repeated_header"] += 1
                    continue
                if label not in LABELS:
                    raise ValueError("Unknown native CIC outcome; no inferred label admitted")
                try:
                    if not all(math.isfinite(float(native[k])) and float(native[k]) >= 0 for k in FIELDS):
                        raise ValueError()
                except ValueError:
                    rejected[label] += 1
                    continue
                visible = {k: native[k] for k in FIELDS}
                identity = digest(visible)
                heap = heaps.setdefault(label, [])
                # Stable ordinal breaks ties without allowing identical selected
                # requests; equal visible rows are retained once per native class.
                if any(x[1] == identity for x in heap):
                    continue
                item = (-int(rank("cic-v3", entry["capture"], identity), 16), identity, ordinal, visible)
                if len(heap) < cap:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
        for label, heap in heaps.items():
            for _, identity, ordinal, flow in heap:
                state = {"scope": "Selected CICFlowMeter event fields. Timestamp and source labels are unavailable. Flow Duration is in microseconds.", "flow": flow}
                q, y = choice([entry["capture"], identity], "flow_class", "Classify the network flow as benign or malicious.", DESCRIPTIONS, binary(label))
                output.append(make("network_defense", [entry["source"]["sha256"], ordinal], "test",
                    "network-family/CSE-CIC-IDS2018/campaign2018", state, {"flow_class": q}, {"flow_class": y},
                    {"task": "cic_publisher_flow_annotation", "diagnostic_only": False, "source_dataset": "CSE-CIC-IDS2018",
                     "source_publisher": "CSE and Canadian Institute for Cybersecurity", "capture_id": "CSE-CIC-IDS2018/" + entry["capture"],
                     "capture_family": "campaign2018", "binary_disposition_judged": True, "publisher_label": label,
                     "label_basis": "native publisher annotation", "transfer_panel": True},
                    {**entry["source"], "csv_record": ordinal, "flow_sha256": identity, "upstream": SOURCE_PAGE,
                     "license": index["terms"], "registry": REGISTRY}))
        reports.append({**entry, "rows_scanned": scanned, "classes": dict(counts), "invalid_projection_rows": dict(rejected), "eof_verified": True})
        log(f"CIC v3: {entry['capture']}: {scanned:,} native records", flush=True)
    return output, {**index, "sources": reports, "cases": len(output), "cap_per_capture_native_class": cap,
                    "independent_campaigns": 1, "split": "test", "model_fitting_on_publisher": False}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Acquire bounded, byte-pinned CSE-CIC-IDS2018 publisher CSV files")
    parser.add_argument("--collection", default="../jevalin-collect")
    acquire(parser.parse_args().collection)
