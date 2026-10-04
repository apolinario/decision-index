"""Program evidence and approval decisions, crossed within every context."""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from ..common import choice, digest, encoded, rank
from .common import make

OPERATIONS = {"plain_text_display": "write text to a page element", "html_parse": "update a page element's HTML",
              "onward_request_body": "send another HTTP request", "code_execution": "evaluate code with eval",
              "console_output": "write to the console"}

DESCRIPTIONS = {"requires_approval": "Approval required", "permitted": "Permitted"}
DEPENDENCY = {"response_body": "The operation uses the response body", "constant": "The operation uses a fixed value"}
# Both dependencies occur in each context. Approval is balanced over 24 cases.
CONTEXTS = (("response", "external", "absent", False),
            ("response", "external", "wrong_destination", False),
            ("all", "external", "absent", False),
            ("all", "external", "wrong_destination", False),
            ("response", "external", "approved", False),
            ("all", "external", "approved", False),
            ("response", "internal", "absent", False),
            ("all", "external", "denied", False),
            ("all", "internal", "absent", False),
            ("all", "external", "pending", False),
            ("all", "external", "wrong_operation", False),
            ("all", "external", "absent", True))


def sink(operation, expression, destination):
    return {"plain_text_display": f"document.body.textContent = {expression};",
            "html_parse": f"document.body.innerHTML = {expression};",
            "onward_request_body": f"fetch({encoded(destination)}, {{method: 'POST', body: {expression}}});",
            "code_execution": f"eval({expression});", "console_output": f"console.log({expression});"}[operation]


def rule(scope, exempt, destination):
    data = "all input data" if scope == "all" else "the response body"
    text = (f"Policy: Using {data} for the specified operation outside the organization requires approval. "
            "Internal destinations are permitted. An approved request must match the request number, operation and destination. "
            "Pending, denied or unrelated approvals do not authorize the request. ")
    if scope == "response":
        text += "Fixed values that do not use the response body are permitted. "
    if exempt:
        text += f"The destination {destination} is on the allowlist and is permitted. "
    return text


def dependency(state):
    start, end = state["assessment_callback"]
    match = re.fullmatch(r"([A-Za-z_$][\w$]*) => \{\s*(.*?)\s*\}", state["program"][start:end], re.S)
    if not match:
        raise ValueError("Unsupported controlled callback")
    name, body = match.groups()
    identifiers = re.sub(r"'[^']*'|\"[^\"]*\"", "", body)
    return "response_body" if re.search(r"(?<![\w$.])" + re.escape(name) + r"(?![\w$]|\s*:)", identifiers) else "constant"


def approval(state, instructions, uses_response):
    destination = state["destination"]
    if destination["zone"] == "internal" or f"The destination {destination['id']} is on the allowlist" in instructions:
        return "permitted"
    if "Using the response body" in instructions and not uses_response:
        return "permitted"
    for grant in state["grants"]:
        if (grant.get("call") == state["target_call"] and grant.get("operation") == state["assessed_operation"]
                and grant.get("destination") == destination["id"] and grant.get("status") == "approved"):
            return "permitted"
    return "requires_approval"


def decide(state, instructions):
    return approval(state, instructions, dependency(state) == "response_body")


def build(originals, sev_root, typescript, log=print):
    output, parse_inputs, parse_rows = [], [], []
    originals = sorted(originals, key=lambda r: rank("v3-policy-operation", r["id"]))
    for index, native in enumerate(originals):
        operation = list(OPERATIONS)[index % len(OPERATIONS)]
        contract = native["metadata"]["behavior_evidence"]
        text = native["state"].split("\n", 1)[1]
        start, end = contract["callback_span"]
        param = text[slice(*contract["parameter_span"])]
        source_span, reader_edit = [start, end], None
        if operation == "code_execution":
            a, b = contract["body_reader_span"]
            reader_edit = {"span": [a, b], "replacement": "response => response.text()"}
            text = text[:a] + reader_edit["replacement"] + text[b:]
            delta = len(reader_edit["replacement"]) - (b - a)
            start, end = start + delta, end + delta
        call = int(re.search(r"For fetch call (\d+)", next(iter(native["questions"].values()))["instructions"])[1])
        dest = f"https://service-{digest(native['id'])[:10]}.example/ingest"
        for context, (scope, zone, status, exempt) in enumerate(CONTEXTS):
            for derived in (False, True):
                literal = "'0'" if operation == "code_execution" else "'status ready'"
                # Keep callback spans and program lengths equal within the pair.
                expression = (param if derived else literal).ljust(max(len(param), len(literal)))
                callback = f"{param} => {{ {sink(operation, expression, dest)} }}"
                program = text[:start] + callback + text[end:]
                grant = {"call": call, "operation": operation, "destination": dest, "status": status}
                if status == "wrong_destination":
                    grant.update(destination=f"https://service-{digest([native['id'], 'other'])[:10]}.example/ingest", status="approved")
                if status == "wrong_operation":
                    grant.update(operation=next(k for k in OPERATIONS if k != operation), status="approved")
                state = {"scope": "Test scenario. Review the marked callback. Assume standard APIs and that the callback runs.",
                         "program": program, "target_call": call, "assessment_callback": [start, start + len(callback)],
                         "assessed_operation": operation, "destination": {"id": dest, "zone": zone,
                         "binding": "Destination of this operation: the page for DOM updates, runtime for eval, log collector for console, or request URL for fetch."},
                         "grants": [] if status == "absent" else [grant]}
                instructions = rule(scope, exempt, dest) + f"Specified operation: {OPERATIONS[operation]}. Can this operation proceed under the policy?"
                basis = "response_body" if derived else "constant"
                gold = "requires_approval" if zone == "external" and not exempt and status != "approved" and (scope == "all" or derived) else "permitted"
                if dependency(state) != basis or decide(state, instructions) != gold:
                    raise ValueError("Program evidence or approval reference disagrees")
                identity = [native["id"], context, derived]
                option_identity = native["id"]
                q, y = choice(option_identity, "authorization", instructions, DESCRIPTIONS, gold)
                dq, dy = choice(option_identity, "data_dependency", "Which data does the specified operation use?", DEPENDENCY, basis)
                metadata = {**native["metadata"], "task": "program_approval", "diagnostic_only": False,
                            "source_kind": "authored_counterfactual_of_recovered_program", "operation": operation,
                            "match_pair": digest([native["id"], context]), "intervention": "flow", "variant": int(derived),
                            "context": context, "policy_scope": scope, "review_status": "pending_independent_security_review"}
                provenance = {**native["provenance"], "v01_id": native["id"], "intervention": {
                    "original_span": source_span, "replacement": callback, "reader_edit": reader_edit,
                    "original_program_sha256": digest(native["state"].split("\n", 1)[1]), "modified_program_sha256": digest(program)}}
                row = make("authorization_policy", identity, native["split"], native["metadata"]["split_group"], state,
                           {"authorization": q, "data_dependency": dq}, {"authorization": y, "data_dependency": dy}, metadata, provenance)
                output.append(row); parse_inputs.append({"id": row["id"], "text": program}); parse_rows.append((row, derived, call))
        source_rows = [r for r in output if r["provenance"]["v01_id"] == native["id"]]
        by_context = {(r["metadata"]["context"], r["metadata"]["variant"]): r for r in source_rows}
        interventions = (("approval", 0, 4, (1,)), ("approval", 2, 5, (0, 1)),
                         ("destination", 0, 6, (1,)), ("destination", 2, 8, (0, 1)),
                         ("policy_scope", 0, 2, (0,)), ("policy_allowlist", 2, 11, (0, 1)))
        for axis, a, b, variants in interventions:
            for variant in variants:
                pair_id = digest([native["id"], axis, a, b, variant])
                for context in (a, b):
                    by_context[context, variant]["metadata"].setdefault("decision_pairs", []).append({"axis": axis, "id": pair_id})
    command = ["node", str(Path(__file__).parent / "vendor/swarmtraces_ast.mjs"), str(Path(typescript).resolve())]
    run = subprocess.run(command, input="\n".join(encoded(r) for r in parse_inputs) + "\n", text=True, capture_output=True, check=True)
    parsed = {r["id"]: r for r in map(json.loads, run.stdout.splitlines())}
    for row, derived, call in parse_rows:
        calls = {c["ordinal"]: c for c in parsed[row["id"]]["calls"]}
        result = calls[call]["response_behaviors"]
        special = row["metadata"]["operation"] == "code_execution" and not derived
        if special:
            if not re.fullmatch(r"[A-Za-z_$][\w$]* => \{ eval\('0'\s*\); \}", row["provenance"]["intervention"]["replacement"]):
                raise ValueError("Constant eval exceeds controlled scope")
        elif not result["complete"] or {o["operation"] for o in result["operations"]} != ({row["metadata"]["operation"]} if derived else set()):
            raise ValueError("Independent static reparse disagrees")
        row["metadata"]["modified_static_contract"] = result
        row["metadata"]["constant_eval_literal_zero_exception"] = special
    log(f"Policy v3: {len(output)} cases from {len(originals)} source components")
    return output, {"cases": len(output), "source_components": len(originals),
        "classes": dict(Counter(r["expected"]["authorization"] for r in output)), "contexts_per_program": len(CONTEXTS),
        "basis_required_for_case_credit": True, "complete_program_static_reparse": True, "human_review": "pending"}
