import copy
import io
import uuid
from collections import defaultdict

import pytest

from decision_index.cyber.audit import deduplicate, validate
from decision_index.cyber.build import build, verify, verify_sources
from decision_index.cyber.common import SPLITS, TRACKS, choice, digest, file_hash, read_json, record, rows, summary, write_json, write_rows
from decision_index.cyber.endpoint import CHANNEL, joined, native_records, xml_event
from decision_index.cyber.network import DESCRIPTIONS, label_class
from decision_index.cyber.network import FIELDS as FLOW_FIELDS
from decision_index.cyber.guide import FIELDS as GUIDE_FIELDS, visible_incident
from decision_index.cyber.policy import valid_contract
from decision_index.cyber.protection import collect
from decision_index.cyber.scoring import baseline, score


def network_case(identifier, label, split="calibration", group=None):
    q, gold = choice(identifier, "flow_class", "Predict annotation", DESCRIPTIONS, label)
    return record("network_defense", identifier, split, group or identifier, {"flow": identifier},
                  {"flow_class": q}, {"flow_class": gold}, {"task": "flow"}, {"path": "source.csv"})


def result(row, prediction=None):
    label = prediction or row["expected"]["flow_class"]
    return {**row["_evaluation"], "status": "ok", "response": {"answers": {"flow_class": {
        "type": "choice", "choice": label, "probabilities": {k: float(k == label) for k in DESCRIPTIONS}}}}}


def test_missing_and_invalid_answers_remain_in_denominator():
    a, b, c = [network_case(str(i), "Botnet") for i in range(3)]
    bad = result(b)
    del bad["response"]["answers"]["flow_class"]["probabilities"]["Normal"]
    report = score([a, b, c], [result(a), bad])
    track = report["tracks"]["network_defense"]
    assert track["accuracy"] == pytest.approx(1 / 3)
    assert track["coverage"] == pytest.approx(1 / 3)
    assert track["failures"] == {"invalid": 1, "missing": 1}
    assert track["binary_detection_errors"]["miss_rate_including_failed_answers"] == pytest.approx(2 / 3)
    assert not report["complete"]


def test_background_never_counts_as_a_binary_negative():
    cases = [network_case("a", "Normal"), network_case("b", "Botnet"), network_case("c", "Background")]
    report = score(cases, [result(r, "Botnet") for r in cases])["tracks"]["network_defense"]
    binary = report["binary_detection_errors"]
    assert binary["negative_support"] == 1
    assert binary["excluded_unjudged"] == 1
    assert binary["false_positive_rate"] == 1


def test_scorer_rejects_wrong_partition_or_changed_payload():
    r = network_case("a", "Normal")
    changed = result(r)
    changed["payload_sha256"] = "wrong"
    with pytest.raises(ValueError, match="payload hash"):
        score([r], [changed])
    with pytest.raises(ValueError, match="outside"):
        score([r], [result(network_case("other", "Botnet"))])


def test_zero_probability_gold_has_finite_penalized_loss():
    r = network_case("a", "Normal")
    metrics = score([r], [result(r, "Botnet")])["tracks"]["network_defense"]
    assert metrics["multiclass_brier_answered_only"] == 2
    assert metrics["log_loss_answered_only"] > 30


def test_majority_uses_development_not_target_answers():
    development = [network_case(str(i), "Normal", "development") for i in range(4)]
    target = [network_case("target", "Botnet")]
    prediction = baseline(target, development, "development_majority")[0]
    assert prediction["response"]["answers"]["flow_class"]["choice"] == "Normal"
    changed = copy.deepcopy(target)
    changed[0]["expected"]["flow_class"] = "Background"
    assert baseline(changed, development, "development_majority") == [prediction]


def test_duplicate_owner_does_not_depend_on_labels_or_input_order():
    a, b = network_case("a", "Normal", "test"), network_case("b", "Botnet", "development")
    b["state"] = copy.deepcopy(a["state"])
    first, report = deduplicate([a, b])
    a["expected"]["flow_class"], b["expected"]["flow_class"] = "Botnet", "Normal"
    second, _ = deduplicate([b, a])
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert report["removed_identical_evidence"]["network_defense"] == 1
    assert report["duplicate_evidence_with_different_answers_before_removal"]["network_defense"] == 1


def test_gzip_is_deterministic_and_keeps_permuted_choice_order(tmp_path):
    r = network_case("a", "Botnet")
    r["questions"]["flow_class"]["criteria"] = {"Normal": "n", "Botnet": "b", "Background": "u"}
    a, b = tmp_path / "one.gz", tmp_path / "two.gz"
    write_rows(a, [r])
    write_rows(b, [r])
    assert a.read_bytes() == b.read_bytes()
    assert list(next(rows(a))["questions"]["flow_class"]["criteria"]) == ["Normal", "Botnet", "Background"]


def test_join_requires_host_valid_nonzero_guid_and_creation_time():
    guid = str(uuid.uuid4())
    creation = {"EventID": 1, "Hostname": "HOST", "ProcessGuid": "{" + guid.upper() + "}", "UtcTime": "2024-01-01 00:00:00.123456789"}
    activity = {"EventID": 11, "Hostname": "host", "ProcessGuid": guid, "UtcTime": "2024-01-01 00:00:00.123456790"}
    assert joined(creation, activity)
    assert not joined(creation, {**activity, "Hostname": "other"})
    assert not joined(creation, {**activity, "ProcessGuid": str(uuid.UUID(int=0))})
    assert not joined(creation, {**activity, "ProcessGuid": "bad"})
    assert not joined(creation, {**activity, "UtcTime": "2024-01-01 00:00:00.123456788"})


def test_multiline_xml_and_adjacent_roots_keep_complete_records():
    event = (b'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">\n'
             b'<System><Provider Name="Microsoft-Windows-Sysmon"/><EventID>11</EventID>'
             b'<Computer>host</Computer></System>\n<EventData><Data Name="Image">app</Data></EventData></Event>')
    parsed = list(native_records(io.BytesIO(event + b"\n" + event)))
    assert [(a, b) for a, b, _, _ in parsed] == [(1, 3), (4, 6)]
    assert all(xml_event(raw)["Channel"] == CHANNEL for _, _, raw, _ in parsed)
    adjacent = list(native_records(io.BytesIO(event.replace(b"\n", b"") * 2)))
    assert len(adjacent) == 2
    with pytest.raises(ValueError, match="incomplete"):
        list(native_records(io.BytesIO(event[:-10])))


def test_linux_sysmon_is_not_reinterpreted_as_windows():
    raw = b'<Event><System><Provider Name="Linux-Sysmon"/></System></Event>'
    assert xml_event(raw) is None


def test_policy_offsets_are_unicode_codepoints():
    contract = {"complete": True, "version": 1, "scope": "immediate_body_callback", "reason": None,
                **{k: [0, 1] for k in ("request_span", "body_reader_span", "callback_span", "parameter_span")},
                "operations": [{"operation": "html_parse", "sink_span": [0, 1], "value_span": [0, 1], "dependency_spans": [[0, 1]]}]}
    assert valid_contract(contract, "😀abc")
    contract["callback_span"] = [0, 5]
    with pytest.raises(ValueError, match="outside"):
        valid_contract(contract, "😀abc")


def test_protection_collector_never_opens_existing_test_payload(tmp_path):
    collection, sev = tmp_path / "collection", tmp_path / "sev"
    directory = collection / "datasets/sev_fixture"
    directory.mkdir(parents=True)
    (collection / "external").mkdir()
    (sev / "evals/sev").mkdir(parents=True)
    write_rows(directory / "train.jsonl", [{"state": "known", "_meta": {"group_id": "protected"}}])
    (directory / "test.jsonl").write_text("THIS IS NOT JSON; MUST NEVER BE OPENED")
    write_json(directory / "manifest.json", {"files": {"train.jsonl": {"sha256": file_hash(directory / "train.jsonl")}}})
    protected = collect(collection, sev)
    assert protected["groups"] == {"protected"}
    assert protected["states"] == {digest("known")}


def test_changed_source_fails_receipt_verification(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("one\n")
    receipts = [{"root": "collection", "path": source.name, "sha256": file_hash(source)}]
    source.write_text("two\n")
    with pytest.raises(ValueError, match="receipt mismatch"):
        verify_sources(receipts, tmp_path, tmp_path)


@pytest.mark.parametrize("native,gold", [("flow=Background-TCP", "Background"), ("From-Botnet-V42-TCP", "Botnet"), ("From-Normal-V42", "Normal")])
def test_native_network_categories(native, gold):
    assert label_class(native) == gold


def test_ambiguous_network_labels_are_rejected():
    with pytest.raises(ValueError):
        label_class("From-Botnet-Normal-TCP")


def small_suite():
    values = []
    for split in SPLITS:
        for track in TRACKS:
            identifier = f"{split}:{track}"
            q, gold = choice(identifier, "q", "Assess shown evidence", {"yes": "yes", "no": "no"}, "yes")
            meta, provenance = {"task": "test"}, {"path": "source"}
            if track == "network_defense":
                r = network_case(identifier, "Normal", split)
                r["state"] = {"scope": "native", "flow": {k: identifier for k in FLOW_FIELDS}}
                r["metadata"].update(capture_id=identifier, binary_disposition_judged=True)
                r["_evaluation"]["payload_sha256"] = digest({"state": r["state"], "questions": r["questions"]})
                values.append(r)
                continue
            if track == "incident_triage":
                state = visible_incident([{**{k: "" for k in GUIDE_FIELDS}, "AlertId": identifier, "Timestamp": identifier}])
                meta["organization"] = identifier
            elif track == "endpoint_investigation":
                state = {"scope": "native", "events": [{"EventID": 1, "Image": identifier}]}
                provenance["native_refs"] = [{"sha256": identifier, "event_identity": identifier}]
            else:
                state = "Recovered complete program. No execution is observed.\n" + identifier
                meta.update(source_record_id=identifier, template_sha256=identifier)
                provenance["source_text_sha256"] = identifier
            values.append(record(track, identifier, split, identifier, state, {"q": q}, {"q": gold}, meta, provenance))
    return values


def test_frozen_artifact_roundtrip_and_tamper_detection(tmp_path):
    values = small_suite()
    structural = validate(values)
    for split in SPLITS:
        selected = [r for r in values if r["split"] == split]
        write_rows(tmp_path / f"inputs/{split}.jsonl.gz", [
            {k: r[k] for k in ("id", "family", "split", "state", "questions", "_evaluation")} for r in selected])
        write_rows(tmp_path / f"labels/{split}.jsonl.gz", [
            {k: r[k] for k in ("id", "expected", "metadata", "provenance")} for r in selected])
    empty_protection = {k: [] for k in ("groups", "states", "archives", "native_events", "program_ids", "templates", "program_texts", "receipts")}
    write_json(tmp_path / "reports/protection.json", empty_protection)
    write_json(tmp_path / "reports/audit.json", {"structural": structural})
    assert read_json(tmp_path / "reports/audit.json")["structural"] == structural
    files = {str(p.relative_to(tmp_path)): {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in tmp_path.rglob("*") if p.is_file()}
    write_json(tmp_path / "manifest.json", {"version": "test", "files": files,
        "counts": {track: summary([r for r in values if r["family"] == track]) for track in TRACKS}})
    assert verify(tmp_path)["records"] == 12
    with (tmp_path / "inputs/test.jsonl.gz").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify(tmp_path)


@pytest.mark.parametrize("track,key", [("incident_triage", "organization"), ("network_defense", "capture_id"), ("authorization_policy", "template_sha256")])
def test_hidden_source_identity_cannot_cross_splits(track, key):
    values = small_suite()
    selected = [r for r in values if r["family"] == track]
    selected[1]["metadata"][key] = selected[0]["metadata"][key]
    with pytest.raises(ValueError, match="crosses splits"):
        validate(values)


def test_freeze_refuses_existing_destination_before_reading_sources(tmp_path):
    out = tmp_path / "frozen"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError, match="already exists"):
        build(tmp_path / "missing-source", tmp_path / "missing-sev", out, tmp_path / "work")
    assert marker.read_text() == "preserve"
