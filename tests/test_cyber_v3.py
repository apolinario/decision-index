import copy
from pathlib import Path

import pytest

from decision_index.cyber.common import choice, write_rows
from decision_index.cyber.v3.common import make, payload_hash
from decision_index.cyber.v3.inference import preflight
from decision_index.cyber.v3.policy import CONTEXTS, approval, decide, dependency, rule, sink
from decision_index.cyber.v3.scoring import score
from decision_index.cyber.v3 import workplace


def policy_state(derived=True, operation="onward_request_body"):
    expression = "body" if derived else "'ok'"
    program = "body => { " + sink(operation, expression, "https://service.example") + " }"
    return {"program": program, "assessment_callback": [0, len(program)], "target_call": 2,
            "assessed_operation": operation, "destination": {"id": "https://service.example", "zone": "external"}, "grants": []}


def fixture(split="calibration"):
    q, y = choice("fixture", "authorization", "Approval required?", {"requires_approval": "Approval required", "permitted": "Permitted"}, "requires_approval")
    return make("authorization_policy", "fixture", split, "fixture", policy_state(), {"authorization": q}, {"authorization": y},
                {"task": "program_approval", "diagnostic_only": False}, {})


def test_program_and_policy_both_change_decision():
    for operation in ("plain_text_display", "html_parse", "onward_request_body", "code_execution", "console_output"):
        state = policy_state(operation=operation)
        instructions = rule("response", False, state["destination"]["id"])
        assert dependency(state) == "response_body"
        assert decide(state, instructions) == "requires_approval"
        fixed = policy_state(False, operation)
        assert dependency(fixed) == "constant"
        assert decide(fixed, instructions) == "permitted"
        assert decide(fixed, rule("all", False, state["destination"]["id"])) == "requires_approval"
        state["grants"] = [{"call": 2, "operation": operation, "destination": "https://other.example", "status": "approved"}]
        assert decide(state, instructions) == "requires_approval"
        state["grants"][0]["destination"] = state["destination"]["id"]
        assert decide(state, instructions) == "permitted"


def test_ordered_request_receipt_and_preflight(tmp_path):
    row = fixture()
    path = tmp_path / "input.jsonl"
    write_rows(path, [{k: row[k] for k in ("id", "family", "split", "state", "questions", "_evaluation")}])
    preflight(path)
    row["questions"]["authorization"]["criteria"] = dict(reversed(list(row["questions"]["authorization"]["criteria"].items())))
    write_rows(path, [row])
    with pytest.raises(ValueError, match="label-sidecar"):
        preflight(path)
    write_rows(path, [{k: row[k] for k in ("id", "family", "split", "state", "questions", "_evaluation")}])
    with pytest.raises(ValueError, match="receipt"):
        preflight(path)


def test_all_public_evaluation_paths_reject_ungated_test(tmp_path, monkeypatch):
    from decision_index.cyber.scoring import baseline
    from decision_index import runner
    row = fixture("test")
    path = tmp_path / "test.jsonl"
    write_rows(path, [{k: row[k] for k in ("id", "family", "split", "state", "questions", "_evaluation")}])
    for call in (lambda: preflight(path), lambda: score([row], []), lambda: baseline([row], [], "uniform"),
                 lambda: runner.run("never-load", {}, path, tmp_path / "out")):
        with pytest.raises(ValueError, match="sealed.*protocol"):
            call()
    assert not (tmp_path / "out").exists()


def test_workplace_pairs_flip_one_decision_axis_and_remain_in_fold():
    native = {"id": "anchor", "split": "development", "metadata": {"split_group": "source-one", "template_sha256": "t"}}
    values, report = workplace.build([native])
    assert len(values) == 4 and report["independent_real_incidents_added"] == 0
    pairs = {}
    for row in values:
        assert workplace.decide(row["state"]) == row["expected"]["authorization"]
        assert row["metadata"]["split_group"] == "source-one"
        pairs.setdefault(row["metadata"]["match_pair"], []).append(row)
    for pair in pairs.values():
        assert {r["expected"]["authorization"] for r in pair} == {"requires_approval", "permitted"}
        assert pair[0]["state"]["request"] == pair[1]["state"]["request"]


def test_finished_duplicate_predictions_are_rejected():
    row = fixture()
    prediction = {**row["_evaluation"], "status": "context_overflow"}
    with pytest.raises(ValueError, match="Repeated"):
        score([row], [prediction, prediction])


def test_same_path_and_pid_distractor_cannot_replace_guid_link():
    from decision_index.cyber.v3.endpoint import case, solve, UNKNOWN
    from decision_index.cyber.v3.language import normalize
    from decision_index.cyber.v3.shortcuts import endpoint_shape
    def event(line, guid, image, event_id=1):
        return {"event": {"EventID": event_id, "UtcTime": f"2024-01-01 00:00:{line:02}", "Hostname": "host-one",
                "ProcessGuid": "{00000000-0000-0000-0000-" + f"{guid:012}" + "}", "ProcessId": "7", "Image": image},
                "ref": {"line": line, "path": "native-fixture", "sha256": "source", "native_line_sha256": str(line)}}
    creation = event(1, 1, "app.exe")
    decoy = event(2, 2, "app.exe")
    target = event(10, 1, "app.exe", 3)
    extras = [event(i, i, "other.exe" if i % 2 else "another.exe") for i in range(3, 8)]
    known = normalize(case(None, "group", "calibration", [creation, target, *extras], [creation], target, "process_lifetime", False))
    missing = normalize(case(None, "group", "calibration", [decoy, target, *extras], [creation], target, "process_lifetime", True))
    assert known["questions"] == missing["questions"]
    assert endpoint_shape(known) == endpoint_shape(missing)
    assert solve(known["state"], "process_lifetime")[0] == "app.exe"
    assert solve(missing["state"], "process_lifetime")[0] == UNKNOWN
