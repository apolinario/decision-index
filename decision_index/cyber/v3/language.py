"""Plain authored wrappers; native evidence is never rewritten."""
import copy
import re

from ..common import digest
from .common import refresh

SOURCES = [
    {"url": "https://developers.google.com/style/tone", "use": "Conversational, direct instructions"},
    {"url": "https://learn.microsoft.com/en-us/style-guide/word-choice/use-simple-words-concise-sentences", "use": "Simple words and concise sentences"},
    {"url": "https://docs.cloud.google.com/chronicle/docs/secops/investigate/investigation-management/use-events-viewer", "terms": ["event details", "event fields", "raw logs"]},
    {"url": "https://docs.cloud.google.com/chronicle/docs/investigation/udm-search", "terms": ["related events", "user account"]},
    {"url": "https://learn.microsoft.com/en-us/defender-xdr/security-copilot-m365d-create-incident-report", "terms": ["incident classification", "incident report", "analyst comments and notes"]},
    {"url": "https://learn.microsoft.com/en-us/sysinternals/downloads/sysmon", "terms": ["process creation", "process GUID", "network connections"]},
    {"url": "https://docs.cloud.google.com/deploy/docs/subscribe-deploy-notifications", "terms": ["Approval required"]},
    {"url": "https://docs.cloud.google.com/assured-workloads/access-approval/docs/reference/rest/v1/AccessApprovalSettings", "terms": ["approval requests"]},
    {"url": "https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tabledata/list", "terms": ["response body"]},
    {"url": "https://docs.cloud.google.com/iam/docs/overview", "terms": ["principal", "role", "resource", "approval"]},
    {"url": "https://learn.microsoft.com/en-us/microsoft-365-app-certification/docs/seg2_overview", "terms": ["insufficient evidence"]},
    {"url": "https://learn.microsoft.com/en-us/biztalk/technical-guides/test-scenario-overview", "terms": ["test scenario"]},
]
FORMS = ("ticket", "chat", "email", "handoff")
FLAGGED = ("supplied evidence", "provider-adjudicated", "credential account", "published flow projection",
           "narrow effect", "unestablished", "authored callback intervention", "selected complete native")


def request(form, target):
    body = f"Please review event {target}. Identify the process and include the related events."
    if form == "ticket":
        return {"format": form, "subject": "Endpoint alert review", "description": body}
    if form == "chat":
        return {"format": form, "messages": [{"from": "SOC analyst", "text": body},
                {"from": "IT operations", "text": "The event details are attached."}]}
    if form == "email":
        return {"format": form, "from": "SOC analyst", "to": "IT operations", "subject": "Endpoint alert review", "body": body}
    return {"format": form, "analyst_notes": body, "next_action": "Review the event details."}


def normalize(row):
    row = copy.deepcopy(row)
    state, m = row["state"], row["metadata"]
    if row["family"] == "endpoint_investigation":
        if m["diagnostic_only"]:
            state["scope"] = "Event fields from one capture. The User field identifies a user account; the events do not identify a human owner."
            replacements = {"observed_effect": "What happened in this event?", "application": "Which process produced this event?",
                            "credential_account": "Which user account is recorded in this event?"}
            for field, q in list(row["questions"].items()):
                if field in replacements:
                    q["instructions"] = replacements[field]
                old_unknown = "The supplied events do not establish this"
                q["criteria"] = {"insufficient_evidence" if k == old_unknown else k: "Insufficient evidence" if k == old_unknown else v for k, v in q["criteria"].items()}
                if row["expected"][field] == old_unknown:
                    row["expected"][field] = "insufficient_evidence"
                if field == "credential_account":
                    row["questions"]["user_account"] = row["questions"].pop(field)
                    row["expected"]["user_account"] = row["expected"].pop(field)
        else:
            form = FORMS[int(digest(m["match_pair"])[:8], 16) % len(FORMS)]
            state["request"] = request(form, state["target_event"])
            m.update(workplace_format=form, request_authored=True)
    elif row["family"] == "incident_triage":
        state["scope"] = ("Incident metadata and event timestamps in source order. Alert references are local to this incident. "
                          "Response actions and source classifications are unavailable. Detector IDs and alert titles are anonymized categories.")
        q = row["questions"]["incident_grade"]
        q["instructions"] = ("Predict the incident classification recorded by the source organization. True positive means an actual threat; "
                              "benign positive means expected activity that triggered an alert; false positive means an incorrect alert. "
                              "Classify the whole incident.")
        q["criteria"].update(TruePositive="True positive: actual threat", BenignPositive="Benign positive: expected activity",
                             FalsePositive="False positive: incorrect alert")
    elif row["family"] == "network_defense" and m["task"] == "native_network_boundary_rule":
        state["scope"] = "Network flow fields. Apply the firewall rule below. Missing fields are unavailable."
        q = row["questions"]["boundary_action"]
        q["instructions"] = q["instructions"].replace("supplied flow", "network flow")
        q["criteria"].update(block="Block", permit="Permit", insufficient_evidence="Insufficient evidence")
    elif row["family"] == "network_defense" and m["source_dataset"] != "CSE-CIC-IDS2018":
        state["scope"] = "Selected network flow fields. Source classifications and record identifiers are unavailable."
        row["questions"]["flow_class"]["instructions"] = "Predict the classification recorded by the dataset publisher for this network flow."
        for key in row["questions"]["flow_class"]["criteria"]:
            row["questions"]["flow_class"]["criteria"][key] = {
                "Botnet": "Botnet traffic", "Normal": "Normal traffic", "Background": "Background traffic; maliciousness not determined",
                "Benign": "Benign", "Malicious": "Malicious"}.get(key, key)
    return refresh(row)


def audit(values):
    hits = []
    for r in values:
        # Audit wrappers and options only. Native programs/logs can contain
        # arbitrary language and must retain their original source bytes.
        texts = [r["state"].get("scope", "")]
        texts += [q["instructions"] for q in r["questions"].values()]
        texts += [str(v) for q in r["questions"].values() for v in q["criteria"].values()]
        texts += [str(k) for q in r["questions"].values() for k in q["criteria"]]
        for phrase in FLAGGED:
            if any(phrase in t.casefold() for t in texts):
                hits.append({"id": r["id"], "phrase": phrase})
    return {"purpose": "Editorial terminology check, not an AI authorship detector or proof of natural workplace distribution.",
            "sources": SOURCES, "checked_cases": len(values), "flagged_authored_phrases": hits,
            "native_evidence_rewritten": False, "workplace_wrappers": "Authored; not recovered private Slack or email."}
