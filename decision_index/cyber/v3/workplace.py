"""Authored corporate access/DLP requests, paired by policy or evidence."""
from .common import make
from .language import FORMS
from .policy import DESCRIPTIONS
from ..common import choice, digest, rank

CLASSES = {"restricted": "Restricted data", "public": "Public data"}
ROLE_SCOPE = {"covered": "Role covered by the approval policy", "outside": "Role outside the approval policy"}


def decide(state):
    if "access_request" in state:
        access, policy, grant = state["access_request"], state["policy"], state["approval"]
        approved = grant["status"] == "approved" and all(grant[k] == access[k] for k in ("principal", "role", "resource")) and grant["request_id"] == state["request_id"]
        return "requires_approval" if access["role"] in policy["approval_roles"] and access["environment"] == "production" and not approved else "permitted"
    flow, policy = state["data_flow"], state["policy"]
    covered = policy["data_scope"] == "all" or flow["classification"] == "restricted"
    grant = state["approval"]
    approved = (grant["status"] == "approved" and grant["request_id"] == state["request_id"]
                and grant["destination"] == flow["destination"] and grant["operation"] == flow["operation"])
    return "requires_approval" if covered and flow["zone"] == "external" and not approved else "permitted"


def build(originals):
    values = []
    for index, original in enumerate(sorted(originals, key=lambda r: rank("workplace-v3", r["id"]))):
        for pair_index in range(2):
            axis = ("data", "approval", "destination", "policy")[(index + pair_index) % 4]
            form = FORMS[(index + pair_index) % 4]
            pair = digest([original["id"], "workplace", pair_index])
            for variant in (0, 1):
                request_id = "REQ-" + digest([original["id"], pair_index])[:8]
                destination = "service-" + digest([original["id"], "destination"])[:8] + ".example"
                flow = {"operation": "upload", "classification": "restricted", "destination": destination, "zone": "external",
                        "document": "Quarterly account report", "fields": ["account number", "contact email"]}
                policy = {"data_scope": "restricted", "rule": "External uploads in the data scope require approval for this request, operation and destination. Internal uploads are permitted."}
                grant = {"status": "pending", "request_id": request_id, "operation": "upload", "destination": destination}
                if axis == "data" and variant:
                    flow.update(classification="public", document="Published product guide", fields=["product name", "public specifications"])
                elif axis == "approval" and variant:
                    grant["status"] = "approved"
                elif axis == "destination" and variant:
                    flow["zone"] = "internal"
                elif axis == "policy":
                    flow.update(classification="public", document="Published product guide", fields=["product name", "public specifications"])
                    policy["data_scope"] = "all" if not variant else "restricted"
                body = "Please review this upload request. Check the data flow, destination and approval before we proceed."
                communication = {"format": form, "subject": "Upload approval", "body": body}
                if form == "chat":
                    communication = {"format": form, "messages": [{"from": "IT operations", "text": body}]}
                elif form == "handoff":
                    communication = {"format": form, "analyst_notes": body, "next_action": "Check the approval request."}
                state = {"scope": "Test scenario. The policy and request details below apply to this upload.", "request_id": request_id,
                         "request": communication, "policy": policy, "data_flow": flow, "approval": grant}
                gold = "requires_approval" if variant == 0 else "permitted"
                if decide(state) != gold:
                    raise ValueError("Workplace policy reference disagrees")
                identity = [original["id"], "workplace", pair_index, variant]
                q, y = choice(identity, "authorization", "Can this upload proceed under the policy?", DESCRIPTIONS, gold)
                bq, by = choice(identity, "data_classification", "Which data classification is recorded in the request?", CLASSES, flow["classification"])
                questions, expected, task, operation = {"authorization": q, "data_classification": bq}, {"authorization": y, "data_classification": by}, "workplace_upload_approval", "upload"
                if pair_index == 1:
                    access = {"principal": "operations-service@example.com", "role": "project editor", "resource": "project-" + digest(original["id"])[:8], "environment": "production"}
                    policy = {"approval_roles": ["project editor", "project administrator"], "rule": "Assigning a role in this list on a production resource requires approval for the request, principal, role and resource. Other assignments are permitted."}
                    grant = {"request_id": request_id, **{k: access[k] for k in ("principal", "role", "resource")}, "status": "pending"}
                    if variant:
                        if axis == "data":
                            access["role"] = "project viewer"
                        elif axis == "approval":
                            grant["status"] = "approved"
                        elif axis == "destination":
                            access["environment"] = "development"
                        else:
                            policy["approval_roles"] = ["project administrator"]
                    communication = {"format": form, "subject": "Access approval", "body": "Please review this role assignment. Check the principal, role, resource and approval before we proceed."}
                    if form == "chat":
                        communication = {"format": form, "messages": [{"from": "IT operations", "text": communication["body"]}]}
                    elif form == "handoff":
                        communication = {"format": form, "analyst_notes": communication["body"], "next_action": "Check the approval request."}
                    state = {"scope": "Test scenario. Apply the organization's approval policy to this role assignment.", "request_id": request_id,
                             "request": communication, "access_request": access, "policy": policy, "approval": grant}
                    if decide(state) != gold:
                        raise ValueError("IAM policy reference disagrees")
                    q, y = choice(identity, "authorization", "Can this role assignment proceed under the policy?", DESCRIPTIONS, gold)
                    bq, by = choice(identity, "role_scope", "Is the requested role covered by the approval policy?", ROLE_SCOPE, "covered" if access["role"] in policy["approval_roles"] else "outside")
                    questions, expected, task, operation = {"authorization": q, "role_scope": bq}, {"authorization": y, "role_scope": by}, "workplace_access_approval", "grant_role"
                values.append(make("authorization_policy", identity, original["split"], original["metadata"]["split_group"], state,
                    questions, expected,
                    {"task": task, "source_kind": "authored_workplace_scenario", "diagnostic_only": False,
                     "workplace_format": form, "request_authored": True, "match_pair": pair, "variant": variant,
                     "intervention": "role" if task == "workplace_access_approval" and axis == "data" else axis, "operation": operation, "template_sha256": original["metadata"]["template_sha256"],
                     "review_status": "pending_independent_security_review"},
                    {"authored": True, "fold_anchor_v01_id": original["id"], "note": "New scenario; no claim of native Slack/email or a real incident."}))
    return values, {"cases": len(values), "independent_real_incidents_added": 0,
                    "folds": "Inherited source-component folds; these scenarios add no independent incident count.",
                    "formats": list(FORMS), "axes": ["data", "approval", "destination", "policy"], "human_review": "pending"}
