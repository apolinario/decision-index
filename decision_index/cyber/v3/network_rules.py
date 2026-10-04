"""Native Splunk network evidence under explicit authored boundary rules.

Capture narratives never become per-flow maliciousness labels. This independent
publisher panel tests rule application and reports separately from CTU/IoT.
"""
import ipaddress
import json
import re
from collections import Counter
from pathlib import Path

from ..common import assign_groups, choice, digest, rank, receipt
from .common import make

PATHS = ["attack_techniques/T1572/cobalt_strike/stream_events_zeek.log",
         "attack_techniques/T1021.001/remote_desktop_connection/zeek_conn.log",
         "attack_techniques/T1048/ftp_connection/zeek_conn.log"]
FIELDS = ("ts", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service", "duration",
          "orig_bytes", "resp_bytes", "conn_state", "missed_bytes", "history", "orig_pkts", "resp_pkts")
DESCRIPTIONS = {"block": "Block under the supplied boundary rule", "permit": "Permit under the supplied boundary rule",
                "insufficient_evidence": "Insufficient evidence: a required rule field is unavailable"}


def visible_rule(instructions):
    match = re.search(r"outside ([^,]+),.*protocol is ([^,]+),.*port is in (\[[^]]*\])", instructions)
    if not match:
        raise ValueError("Unsupported visible boundary rule")
    return {"protected_network": match[1], "protocol": match[2], "blocked_destination_ports": json.loads(match[3])}


def decide(state, rule):
    flow = state["flow"]
    if not {"id.orig_h", "id.resp_h", "id.resp_p", "proto"} <= set(flow):
        return "insufficient_evidence"
    source = ipaddress.ip_address(flow["id.orig_h"])
    destination = ipaddress.ip_address(flow["id.resp_h"])
    zone = ipaddress.ip_network(rule["protected_network"])
    matches = (source not in zone and destination in zone and flow["proto"] == rule["protocol"]
               and int(flow["id.resp_p"]) in rule["blocked_destination_ports"])
    return "block" if matches else "permit"


def build(collection, component_report, group_splits, cap=40):
    root = Path(collection).resolve()
    prefix = "datasets/splunk_attack_data/datasets/"
    scenario_owners = {}
    for component in component_report["components"]:
        for source in component["sources"]:
            if source.startswith(prefix):
                scenario_owners["/".join(source[len(prefix):].split("/")[:2])] = component
    assignments = assign_groups(PATHS, "splunk-network-rule-captures")
    output, reports = [], []
    for rel in PATHS:
        path = root / prefix / rel
        source = receipt(path, root)
        scenario = "/".join(rel.split("/")[:2])
        owner = scenario_owners.get(scenario)
        if owner and owner["blocked"]:
            reports.append({**source, "excluded": "Known Sev capture scenario overlap"})
            continue
        group = owner["id"] if owner and owner["id"] in group_splits else "splunk-network-capture/" + scenario
        split = group_splits.get(group, assignments[rel])
        candidates, total = [], 0
        with path.open() as f:
            for ordinal, line in enumerate(f, 1):
                native = json.loads(line)
                total += 1
                if not {"id.orig_h", "id.resp_h", "id.resp_p", "proto"} <= native.keys():
                    continue
                if ipaddress.ip_address(native["id.orig_h"]).version != 4 or ipaddress.ip_address(native["id.resp_h"]).version != 4:
                    continue
                candidates.append((rank("splunk-network-rule", source["sha256"], ordinal), ordinal, native))
        selected = sorted(candidates)[:cap]
        for _, ordinal, native in selected:
            # The protected zone is an authored evaluation context; it need not
            # equal the capture operator's historical firewall configuration.
            dest = ipaddress.ip_address(native["id.resp_h"])
            zone = str(ipaddress.ip_network(f"{dest}/32"))
            port = int(native["id.resp_p"])
            for variant in ("matched_rule", "nonmatching_rule", "missing_port"):
                flow = {k: native[k] for k in FIELDS if k in native}
                if variant == "missing_port":
                    flow.pop("id.resp_p")
                # Source == destination flows cannot exercise the boundary pair.
                if native["id.orig_h"] == native["id.resp_h"]:
                    continue
                rule = {"protected_network": zone, "protocol": native["proto"],
                        "blocked_destination_ports": [port if variant != "nonmatching_rule" else (port + 1) % 65536]}
                state = {"scope": "Native network flow projection under an authored firewall rule. This is a conditional "
                                  "rule decision, not a maliciousness or historical firewall-action label. Missing fields are unavailable.", "flow": flow}
                y = decide(state, rule)
                q, gold = choice([source["sha256"], ordinal, variant], "boundary_action",
                    "Boundary rule: block a flow if its origin is outside " + zone + ", its responder is inside that network, "
                    f"its protocol is {rule['protocol']}, and its responder port is in {rule['blocked_destination_ports']}. "
                    "Permit other flows. Return insufficient evidence if a required field is unavailable. Decide from this rule and supplied flow only.", DESCRIPTIONS, y)
                output.append(make("network_defense", [source["sha256"], ordinal, variant], split, group, state,
                    {"boundary_action": q}, {"boundary_action": gold}, {"task": "native_network_boundary_rule", "diagnostic_only": False,
                    "source_dataset": "Splunk attack_data", "source_publisher": "Splunk", "capture_id": "Splunk/" + scenario,
                    "capture_family": scenario, "rule": rule, "match_pair": digest([source["sha256"], ordinal]),
                    "variant": variant, "label_basis": "authored rule evaluated against native flow fields"},
                    {**source, "line": ordinal, "native_record_sha256": digest(native), "license": "Apache-2.0"}))
        reports.append({**source, "rows_scanned": total, "eof_verified": True, "selected_native_rows": len(selected), "split": split})
    return output, {"sources": reports, "cases": len(output), "classes": dict(Counter(r["expected"]["boundary_action"] for r in output)),
                    "scope": "Independent publisher native evidence, authored boundary rules, separate metric from malware annotation."}
