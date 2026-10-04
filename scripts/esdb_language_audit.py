"""Local editorial audit of blinded ESDB cases; never rewrites frozen evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TECHNICAL = re.compile(r"https?://\S+|\b(?:E\d+|ProcessGuid|ParentProcessGuid|CommandLine|GUID|PID|DOM|APIs?|Sysmon|eval|fetch)\b")
REFERENCES = re.compile(r"\bE\d+\b")
STRICT_ROLES = {"instruction", "message", "policy"}
COMMANDS = {"review", "identify", "require", "return", "choose", "check", "apply", "assume", "predict", "classify"}
EXTRA_SOURCES = [
    "https://platform.claude.com/docs/en/test-and-evaluate/develop-tests",
    "https://claude.com/blog/claude-for-enterprise",
    "https://www.gartner.com/en/articles/cybersecurity-roadmap",
    "https://privacypolicy.kochinc.com/",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def verify_files(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text())
    for name, pin in manifest["files"].items():
        path = dataset / name
        if sha(path) != pin["sha256"] or path.stat().st_size != pin["bytes"]:
            raise ValueError(f"frozen file changed: {name}")
    return {"manifest_sha256": sha(dataset / "manifest.json"), "files_verified": len(manifest["files"])}


def normalize(text):
    """Group variable identifiers, not grammatical or semantic differences."""
    text = re.sub(r"https?://[^\s]+", "https://service.example/ingest", text)
    text = REFERENCES.sub("E0", text)
    return re.sub(r"\s+", " ", text).strip()


def technical_label(text):
    return bool(re.match(r"^(?:[A-Za-z]:\\|\\\\|/|https?://|E\d+(?:\+|;|:|$))", text))


def task(row):
    state = row["state"]
    return ("program_policy" if "program" in state else "upload_policy" if "data_flow" in state
            else "access_policy" if "access_request" in state else "endpoint_reasoning")


def segments(row):
    """Explicit allowlist keeps programs, raw events and identifiers out of NLP."""
    state = row["state"]
    if isinstance(state.get("scope"), str):
        yield "state.scope", "context", state["scope"]
    if isinstance(state.get("policy", {}).get("rule"), str):
        yield "state.policy.rule", "policy", state["policy"]["rule"]
    if isinstance(state.get("destination", {}).get("binding"), str):
        yield "state.destination.binding", "description", state["destination"]["binding"]
    request = state.get("request", {})
    if isinstance(request, dict):
        for key in ("body", "description", "analyst_notes", "next_action", "subject"):
            if isinstance(request.get(key), str):
                yield f"state.request.{key}", "heading" if key == "subject" else "message", request[key]
        for index, message in enumerate(request.get("messages", [])):
            if isinstance(message.get("text"), str):
                yield f"state.request.messages[{index}].text", "message", message["text"]
    for key, question in row["questions"].items():
        yield f"questions.{key}.instructions", "instruction", question["instructions"]
        for option, text in question["criteria"].items():
            if isinstance(text, str) and not technical_label(text):
                yield f"questions.{key}.criteria.{option}", "answer_label", text


def evidence_findings(row):
    findings = []
    state = row["state"]
    for key, question in row["questions"].items():
        for option, value in question["criteria"].items():
            if re.fullmatch(r"[<\[]\s*(?:unknown|missing|placeholder)[^>\]]*[>\]]", str(value), re.I):
                findings.append({"rule": "placeholder_answer_option", "severity": "review",
                                 "path": f"questions.{key}.criteria.{option}", "text": value,
                                 "detail": "an option is a placeholder rather than a named value; inspect answerability without rewriting native evidence"})
    if "events" not in state:
        return findings
    refs = [e.get("ref") for e in state["events"]]
    known = set(refs)
    if len(refs) != len(known):
        findings.append({"rule": "duplicate_event_reference", "severity": "error", "detail": "event references are not unique"})
    if state.get("target_event") not in known:
        findings.append({"rule": "missing_target_event", "severity": "error", "detail": "the target event is absent"})
    for path, _, text in segments(row):
        missing = sorted(set(REFERENCES.findall(text)) - known)
        if missing:
            findings.append({"rule": "unknown_event_reference", "severity": "error", "path": path,
                             "detail": "text names unavailable event references", "references": missing})
    for key, question in row["questions"].items():
        for option in question["criteria"]:
            missing = sorted(set(REFERENCES.findall(option)) - known)
            if missing:
                findings.append({"rule": "unknown_option_reference", "severity": "error",
                                 "path": f"questions.{key}.criteria", "references": missing})
    target = next((e for e in state["events"] if e.get("ref") == state.get("target_event")), {})
    if "omitted from the target event" in state.get("scope", ""):
        for field in ("Image", "User", "CommandLine"):
            if field in target:
                findings.append({"rule": "omitted_field_is_present", "severity": "error", "detail": field})
    return findings


def sentence_features(doc, role, max_words=25):
    results, findings = [], []
    for sentence in doc.sents:
        words = [t for t in sentence if not t.is_space and not t.is_punct]
        predicates = [t for t in words if t.pos_ in {"VERB", "AUX"}]
        finite = [t for t in predicates if t.tag_ in {"VBD", "VBP", "VBZ", "MD"}]
        imperative = sentence.root.pos_ == "VERB" and sentence.root.tag_ == "VB"
        # The small model sometimes tags a leading command such as "Require"
        # as a proper noun. Record the fallback rather than pretending the
        # statistical parse itself identified a verb.
        command_fallback = bool(words and words[0].text.casefold() in COMMANDS and not finite and not imperative)
        field_label = bool(re.match(r"^[A-Za-z][A-Za-z -]{0,35}:\s+\S", sentence.text))
        subjects = [t.text for t in words if t.dep_ in {"nsubj", "nsubjpass", "csubj", "expl"}]
        feature = {"text": sentence.text, "words": len(words), "root": sentence.root.text,
                   "root_pos": sentence.root.pos_, "finite_predicate": bool(finite),
                   "imperative": imperative, "command_fallback": command_fallback,
                   "field_label": field_label, "subjects": subjects,
                   "tokens": [{"text": t.text, "pos": t.pos_, "tag": t.tag_, "dependency": t.dep_,
                               "head": t.head.text} for t in sentence if not t.is_space]}
        results.append(feature)
        if role in STRICT_ROLES and not finite and not imperative and not command_fallback and not field_label:
            findings.append({"rule": "possible_sentence_fragment", "severity": "review", "sentence": sentence.text,
                             "detail": "parser found neither a finite predicate nor an imperative; inspect the parse"})
        if role in STRICT_ROLES and len(words) > max_words:
            findings.append({"rule": "long_sentence", "severity": "review", "sentence": sentence.text,
                             "detail": f"{len(words)} tokens exceeds the configured {max_words}-token editorial threshold"})
    return results, findings


def grammar_findings(tool, text, role):
    results = []
    for match in tool.check(text):
        span = text[match.offset:match.offset + match.error_length]
        # Keep technical names and valid label fragments separate from typos.
        if match.rule_issue_type == "misspelling" and TECHNICAL.fullmatch(span):
            continue
        if role in {"heading", "answer_label", "context", "description"} and match.category in {"CASING", "PUNCTUATION"}:
            continue
        results.append({"rule": "grammar_checker", "severity": "review", "grammar_rule": match.rule_id,
                        "category": match.category, "issue_type": match.rule_issue_type,
                        "detail": match.message, "span": span, "offset": match.offset,
                        "suggestions": match.replacements[:4]})
    return results


def local_grammar(languagetool_version="6.6"):
    import language_tool_python
    # Keep local prose checks on loopback even in shells configured with an
    # HTTP proxy. This changes this process only, not the user's shell settings.
    for name in ("NO_PROXY", "no_proxy"):
        entries = [p for p in os.environ.get(name, "").split(",") if p]
        os.environ[name] = ",".join(dict.fromkeys([*entries, "127.0.0.1", "localhost", "::1"]))
    os.environ.setdefault("LTP_PATH", str(ROOT / "work/esdb-language-audit/languagetool"))
    return language_tool_python.LanguageTool("en-US", host="127.0.0.1",
                                            language_tool_download_version=languagetool_version,
                                            config={"cacheSize": 1000, "pipelineCaching": True})


def extract_page(raw):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup.select("script,style,nav,footer,aside,header,noscript"):
        node.decompose()
    main = soup.select_one("main,article,[role=main]") or soup.body or soup
    return re.sub(r"\s+", " ", main.get_text(" ", strip=True)).strip()


def corpus_sources(dataset, cache, offline=False):
    import requests
    cache.mkdir(parents=True, exist_ok=True)
    existing = json.loads((dataset / "reports/language.json").read_text())["sources"]
    urls = list(dict.fromkeys([s["url"] for s in existing] + EXTRA_SOURCES))

    def fetch(url):
        name = hashlib.sha256(url.encode()).hexdigest()
        receipt, html_path, text_path = cache / f"{name}.json", cache / f"{name}.html", cache / f"{name}.txt"
        if receipt.exists():
            data = json.loads(receipt.read_text())
            if data.get("status") == "available" and html_path.exists() and text_path.exists():
                if sha(html_path) == data["html_sha256"] and sha(text_path) == data["text_sha256"]:
                    return data, text_path.read_text()
        if offline:
            return {"url": url, "status": "unavailable", "reason": "no verified cached copy"}, ""
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            response.raise_for_status()
            text = extract_page(response.content)
            if len(text.split()) < 100:
                raise ValueError("too little article text; source does not count as phrase support")
            html_path.write_bytes(response.content)
            text_path.write_text(text)
            data = {"url": url, "resolved_url": response.url, "status": "available",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(), "words": len(text.split()),
                    "html_sha256": sha(html_path), "text_sha256": sha(text_path)}
            write_json(receipt, data)
            return data, text
        except Exception as error:
            data = {"url": url, "status": "unavailable", "reason": str(error)}
            write_json(receipt, data)
            return data, ""

    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(fetch, urls))
    receipts = [r for r, _ in values]
    texts = {r["url"]: re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip() for r, text in values if text}
    if not texts:
        raise ValueError("no enterprise reference pages were available")
    return receipts, texts


def phrase_findings(doc, corpus):
    support, findings = [], []
    seen = set()
    for chunk in doc.noun_chunks:
        tokens = list(chunk)
        while tokens and (tokens[0].pos_ in {"DET", "PRON"} or tokens[0].text.casefold() in COMMANDS):
            tokens.pop(0)
        content = [t for t in tokens if not t.is_stop and t.pos_ in {"ADJ", "NOUN", "PROPN"}]
        if len(content) < 2 or len(tokens) > 7 or any(TECHNICAL.fullmatch(t.text) for t in tokens):
            continue
        phrase_text = doc[tokens[0].i:tokens[-1].i + 1].text
        phrase = re.sub(r"[^a-z0-9]+", " ", phrase_text.casefold()).strip()
        if phrase in seen:
            continue
        seen.add(phrase)
        sources = [url for url, text in corpus.items() if f" {phrase} " in f" {text} "]
        support.append({"phrase": phrase_text, "source_urls": sources})
        if not sources:
            findings.append({"rule": "phrase_not_found_in_reference_pages", "severity": "info", "phrase": phrase_text,
                             "detail": "absent from the retrieved enterprise pages; absence alone does not mean bad English"})
    return support, findings


def repetition(rows):
    groups = defaultdict(lambda: {"cases": set(), "templates": defaultdict(set)})
    for row in rows:
        for path, role, text in segments(row):
            if role == "message" and not path.endswith("next_action"):
                g = groups[task(row)]
                g["cases"].add(row["id"])
                g["templates"][normalize(text)].add(row["id"])
    findings, summary = [], {}
    for name, group in groups.items():
        counts = sorted(group["templates"].items(), key=lambda item: (-len(item[1]), item[0]))
        summary[name] = {"cases": len(group["cases"]), "unique_message_templates": len(counts),
                         "largest_template_cases": len(counts[0][1]), "largest_template_share": len(counts[0][1]) / len(group["cases"])}
        if len(group["cases"]) >= 20 and len(counts) <= 2:
            findings.append({"rule": "repeated_workplace_template", "severity": "review", "task": name,
                             "case_ids": sorted(group["cases"]), "detail": "workplace message wording has very little variety",
                             "templates": [{"text": text, "cases": len(ids)} for text, ids in counts]})
    return summary, findings


def render_report(summary, findings):
    lines = ["# esdb language audit", "", f"checked {summary['cases']} blinded cases and {summary['questions']} questions.", "",
             "all language processing ran locally. frozen files are unchanged. this report lists candidates for editing, not automatic proof of question correctness.", "",
             f"grammar-checker findings: {summary['grammar_checker_findings']}. event-reference or omission contradictions: {summary['reference_errors']}. possible sentence-fragment groups: {summary['by_rule'].get('possible_sentence_fragment', {}).get('groups', 0)}.", "",
             "| check | finding groups | affected cases |", "| --- | ---: | ---: |"]
    for rule, count in sorted(summary["by_rule"].items()):
        lines.append(f"| {rule} | {count['groups']} | {count['cases']} |")
    lines += ["", "## message variety", "", "| task | cases | message templates |", "| --- | ---: | ---: |"]
    for name, data in summary["message_templates"].items():
        lines.append(f"| {name} | {data['cases']} | {data['unique_message_templates']} |")
    lines += ["", "## findings to inspect", ""]
    for finding in findings:
        if finding["severity"] == "info":
            continue
        text = finding.get("sentence", finding.get("text", ""))
        lines += [f"### {finding['rule']} — {len(finding['case_ids'])} cases", "", finding["detail"], ""]
        if text:
            lines += ["> " + text.replace("\n", " "), ""]
        if finding.get("span"):
            lines += [f"flagged span: `{finding['span']}`", ""]
        if finding.get("suggestions"):
            lines += ["checker suggestions: " + ", ".join(finding["suggestions"]), ""]
        for template in finding.get("templates", []):
            lines += [f"> {template['text']}", ""]
        lines += [f"example case: `{finding['case_ids'][0]}`", ""]
    lines += ["## scope", "", "noun-phrase matches use only the pinned enterprise pages in summary.json. unavailable pages provide no support. a missing phrase is an informational finding, not a grammar failure or an ai-authorship claim. leading determiners are removed before phrase lookup.", "",
              "answer labels and headings may be fragments. imperatives do not require an explicit subject. native code, logs, paths and reference-only options are excluded from language checking. reference checks still inspect event references.", "",
              "variable event ids and destination urls are normalized for repeated-template analysis; annotations describe that normalized text. known leading commands have an explicit fallback when the parser tags them as nouns. colon-labeled fields are recorded as fields, not certified as complete sentences.", "",
              "the script does not read gold labels, model outputs or reviewer answers; it does not certify security correctness. it performs no inference, public write, data rewrite or reviewer outreach.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/esdb-v0.3")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/esdb-v03-language-audit")
    parser.add_argument("--corpus-cache", type=Path, default=ROOT / "work/esdb-language-audit/enterprise-pages")
    parser.add_argument("--offline", action="store_true", help="use verified cached enterprise pages only")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--languagetool-version", default="6.6")
    parser.add_argument("--max-sentence-words", type=int, default=25)
    args = parser.parse_args(argv)
    dataset, out = args.dataset.resolve(), args.out.resolve()
    if out.is_relative_to(dataset) or args.corpus_cache.resolve().is_relative_to(dataset):
        parser.error("audit output and cache must be outside the frozen dataset")
    before = verify_files(dataset)
    with gzip.open(dataset / "review/blinded-cases.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in review packet")
    import spacy
    nlp = spacy.load(args.model, disable=["ner"])
    sources, corpus = corpus_sources(dataset, args.corpus_cache, args.offline)
    print(f"enterprise pages: {len(corpus)} available, {len(sources) - len(corpus)} unavailable", flush=True)
    groups = defaultdict(list)
    findings = []
    for row in rows:
        for path, role, text in segments(row):
            groups[normalize(text), role].append({"id": row["id"], "path": path, "text": text})
        for finding in evidence_findings(row):
            findings.append({**finding, "case_ids": [row["id"]]})
    annotations = []
    with local_grammar(args.languagetool_version) as tool:
        for index, ((text, role), doc) in enumerate(zip(groups, nlp.pipe((t for t, _ in groups), batch_size=64)), 1):
            occurrences = groups[text, role]
            features, language_flags = sentence_features(doc, role, args.max_sentence_words)
            grammar_flags = grammar_findings(tool, text, role)
            phrases, phrase_flags = phrase_findings(doc, corpus)
            case_ids = sorted({o["id"] for o in occurrences})
            for flag in [*language_flags, *grammar_flags, *phrase_flags]:
                findings.append({**flag, "text": text, "role": role, "case_ids": case_ids,
                                 "paths": sorted({o["path"] for o in occurrences})})
            annotations.append({"text": text, "role": role, "occurrences": len(occurrences),
                                "case_ids": case_ids, "sentences": features, "phrase_support": phrases})
            if index % 50 == 0:
                print(f"parsed and checked {index}/{len(groups)} text groups", flush=True)
    message_summary, repeats = repetition(rows)
    findings.extend(repeats)
    after = verify_files(dataset)
    if before != after:
        raise ValueError("dataset changed while audit ran")
    by_rule = {}
    for rule in sorted({f["rule"] for f in findings}):
        matches = [f for f in findings if f["rule"] == rule]
        by_rule[rule] = {"groups": len(matches), "cases": len({i for f in matches for i in f["case_ids"]})}
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "dataset": str(dataset), **before,
               "review_packet_sha256": sha(dataset / "review/blinded-cases.jsonl.gz"), "script_sha256": sha(__file__),
               "cases": len(rows), "questions": sum(len(r["questions"]) for r in rows),
               "text_occurrences": sum(len(v) for v in groups.values()), "unique_text_role_groups": len(groups),
               "cases_with_review_findings": len({i for f in findings if f["severity"] != "info" for i in f["case_ids"]}),
               "by_rule": by_rule, "message_templates": message_summary, "sources": sources,
               "grammar_checker_findings": sum(f["rule"] == "grammar_checker" for f in findings),
               "reference_errors": sum(f["severity"] == "error" for f in findings),
               "runtime": {"python": sys.version.split()[0], "spacy": version("spacy"), "model": nlp.meta["name"],
                           "model_version": nlp.meta["version"], "language_tool_python": version("language-tool-python"),
                           "languagetool": args.languagetool_version, "grammar_server": "local", "language": "en-US"},
               "thresholds": {"max_sentence_tokens": args.max_sentence_words, "repeated_message_min_cases": 20,
                              "repeated_message_max_templates": 2}, "frozen_files_unchanged": True,
               "gold_labels_read": False, "model_predictions_read": False, "native_evidence_rewritten": False,
               "limitations": ["pos and dependency predictions can be wrong, especially on technical wording",
                               "grammar candidates need inspection before editing",
                               "an exact phrase missing from a small reference corpus is not proof of unnatural language",
                               "public documentation is not a sample of private enterprise chat",
                               "sentence grammar does not establish security correctness or benchmark usefulness"]}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    for name, data in (("findings.jsonl", findings), ("annotations.jsonl", annotations)):
        with (out / name).open("w") as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(render_report(summary, findings))
    print(json.dumps({k: summary[k] for k in ("cases", "questions", "unique_text_role_groups", "by_rule", "frozen_files_unchanged")}, indent=2))
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
