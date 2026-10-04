"""Controls for the editorial checker, rather than assertions about its own output."""
import importlib.util
from pathlib import Path
import shutil

import pytest

spec = importlib.util.spec_from_file_location("esdb_language_audit", Path(__file__).parents[1] / "scripts/esdb_language_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(scope="module")
def nlp():
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_sm"):
        pytest.skip("install tools/esdb/language-audit-requirements.txt for NLP controls")
    return spacy.load("en_core_web_sm", disable=["ner"])


@pytest.fixture(scope="module")
def grammar():
    pytest.importorskip("language_tool_python")
    if not shutil.which("java"):
        pytest.skip("local LanguageTool requires java")
    with audit.local_grammar() as tool:
        yield tool


@pytest.mark.parametrize("text", ["Please review the approval request.", "Which process produced this event?", "The request is approved."])
def test_accepts_commands_questions_and_statements(nlp, text):
    _, flags = audit.sentence_features(nlp(text), "instruction")
    assert not flags


def test_catches_fragment_but_allows_label(nlp):
    text = "The external destination."
    _, flags = audit.sentence_features(nlp(text), "instruction")
    assert any(f["rule"] == "possible_sentence_fragment" for f in flags)
    _, flags = audit.sentence_features(nlp(text), "answer_label")
    assert not flags


def test_technical_command_and_labeled_field_are_not_false_fragments(nlp):
    for text in ("Require process-creation records with matching process identifiers.",
                 "Specified operation: write text to a page element."):
        _, flags = audit.sentence_features(nlp(text), "instruction")
        assert not any(f["rule"] == "possible_sentence_fragment" for f in flags)


def test_real_grammar_engine_catches_agreement_and_article_errors(grammar):
    flags = audit.grammar_findings(grammar, "This are a error.", "message")
    assert any(f["grammar_rule"] == "EN_A_VS_AN" for f in flags)
    assert any(f["category"] == "GRAMMAR" for f in flags)
    assert not audit.grammar_findings(grammar, "Please review the approval request.", "message")


def test_native_evidence_and_gold_are_not_language_inputs():
    row = {"state": {"scope": "Review the event.", "program": "native code: I has a error.",
                     "events": [{"CommandLine": "native log: I has a error."}],
                     "request": {"body": "Please review the request."}},
           "questions": {"application": {"instructions": "Which process produced this event?",
                                          "criteria": {"path": r"C:\\Windows\\process.exe", "unknown": "Insufficient evidence"}}},
           "expected": {"application": "SECRET GOLD"}}
    values = list(audit.segments(row))
    assert any(role == "message" for _, role, _ in values)
    assert not any("native" in text or "SECRET GOLD" in text or "process.exe" in text for _, _, text in values)


def test_unknown_references_and_contradictory_omission_are_detected():
    row = {"state": {"target_event": "E1", "scope": "Image, User and CommandLine are omitted from the target event.",
                     "events": [{"ref": "E1", "Image": "process.exe"}]},
           "questions": {"application": {"instructions": "Review event E9.", "criteria": {"E1+E8": "E1+E8"}}}}
    flags = audit.evidence_findings(row)
    assert {f["rule"] for f in flags} == {"unknown_event_reference", "unknown_option_reference", "omitted_field_is_present"}


def test_plain_prose_differences_are_not_collapsed_by_template_normalization():
    assert audit.normalize("Review event E1.") == audit.normalize("Review event E9.")
    assert audit.normalize("The request is approved.") != audit.normalize("The request is denied.")


def test_repeated_messages_are_grouped_without_flagging_shared_instructions():
    rows = [{"id": str(i), "state": {"data_flow": {}, "request": {"body": "Please review this upload request."}},
             "questions": {"authorization": {"instructions": "Can this upload proceed?", "criteria": {"yes": "Permitted"}}}}
            for i in range(20)]
    summary, flags = audit.repetition(rows)
    assert summary["upload_policy"]["unique_message_templates"] == 1
    assert len(flags) == 1
    assert len(flags[0]["case_ids"]) == 20
    for row in rows:
        row["state"].pop("request")
    assert audit.repetition(rows) == ({}, [])


def test_reference_phrase_absence_is_informational(nlp):
    _, flags = audit.phrase_findings(nlp("The executable answers remain."), {"https://example.com": "approval request"})
    assert flags
    assert all(f["severity"] == "info" for f in flags)


def test_reference_lookup_ignores_determiners(nlp):
    support, flags = audit.phrase_findings(nlp("Review the related events."), {"https://example.com": "use related events to investigate"})
    assert any(p["source_urls"] for p in support)
    assert not flags


def test_placeholder_options_are_flagged_without_claiming_label_error():
    row = {"state": {}, "questions": {"application": {"criteria": {"unknown": "<unknown process>"}}}}
    flags = audit.evidence_findings(row)
    assert flags[0]["rule"] == "placeholder_answer_option"
    assert flags[0]["severity"] == "review"
