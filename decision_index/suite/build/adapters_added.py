import ast
import csv
import hashlib
import json
import random
import re
import sqlite3
import unicodedata

from decision_index.suite.build import acquire as acq
from decision_index.suite.build.layout import sha256

MODEL = "jev-1.13.0"
SPECS = {
    57: "MMLU-Pro",
    58: "BBH fixed-option tasks",
    59: "RAGTruth response-level hallucination",
    61: "HoVer claim verification",
    62: "When2Call MCQ",
    64: "New Yorker caption matching",
    56: "PhishNChips phishing decisions",
}
BATCH = {57: 1, 58: 1, 62: 1, 64: 1, 59: 2, 61: 3}
ORDER = (57, 58, 62, 64, 59, 61, 56)
GIT = {
    "cand-BIG-Bench-Hard": ("https://github.com/suzgunmirac/BIG-Bench-Hard", "9ee07bd481feebf959a6b59d61ea57bdcf30964d", ["bbh"]),
    "cand-RAGTruth": ("https://github.com/ParticleMedia/RAGTruth", "c103204b9ce28d6bbad859304bf30de72b8ed8fe", ["dataset"]),
    "cand-hover": ("https://github.com/hover-nlp/hover", "39b84697f196308f398a251a7aea9b82ae0f0562", ["data/hover"]),
    "jev-phishing-bench": ("https://github.com/anisselbd/jev-phishing-bench", "1d56e8c64d029a9554a0874e2ef2901ed196e230", ["run_jev.py"]),
}
HF = {
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "data/test-00000-of-00001.parquet", "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"),
    "when2call": ("nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53", "test/when2call_test_mcq.jsonl", "8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c"),
    "newyorker": ("jmhessel/newyorker_caption_contest", "d81cbab7d0392708d5371d3a4960e69261824db4", "matching/test-00000-of-00001.parquet", "bd2ced3906f63fe8ab36b48b10d291436f8ed6ca4233d639069d04521d83a054"),
    "phishnchips": ("AreLit/PhishNChips", "89afcc39610084298c4679159cb2e27d9ffffa46", "core_emails.csv", "cebb407ff8630491a97400e37464b8db8dfc4299164fca51fcb4ac7eec8204ef"),
}
HOVER_DB = ("hover/wiki_wo_links.db", "https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db", "c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f")
FILE_SHA = {
    "cand-RAGTruth/dataset/response.jsonl": "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    "cand-RAGTruth/dataset/source_info.jsonl": "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    "cand-hover/data/hover/hover_dev_release_v1.1.json": "67c14858f2d7fcdb96b6fe3d538ffcd6f76e3ba594aa2c0cd4359f601101e89d",
}
NEEDS = {57: ["hf:mmlu_pro"], 58: ["git:cand-BIG-Bench-Hard"], 59: ["git:cand-RAGTruth"], 61: ["git:cand-hover", "db:hover"], 62: ["hf:when2call"], 64: ["hf:newyorker"], 56: ["hf:phishnchips", "git:jev-phishing-bench"]}

CHOICE = "Which option is the correct answer?"
BBH_EXCLUDED = ["dyck_languages", "multistep_arithmetic_two", "object_counting", "word_sorting"]
BBH_UNLISTED = {"boolean_expressions": ["True", "False"], "sports_understanding": ["yes", "no"], "web_of_lies": ["Yes", "No"]}
RAGTRUTH_INSTRUCTIONS = "The response contains content that is not supported by the context in the prompt."
RAGTRUTH_CRITERIA = {"true": "The response contains content that is not supported by the context in the prompt.", "false": "All content of the response is supported by the context in the prompt."}
HOVER_INSTRUCTIONS = "Is the claim supported by the evidence?"
HOVER_CRITERIA = {"SUPPORTED": "The evidence supports the claim.", "NOT_SUPPORTED": "The evidence does not support the claim."}
WHEN2CALL_INSTRUCTIONS = "Which response should the assistant give to the user's question, given the available tools?"
NEWYORKER_INSTRUCTIONS = "Which caption was written for this cartoon?"


def dumps(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def letters(n):
    return [chr(65 + i) for i in range(n)]


def parquet(path, drop=()):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.drop_columns([c for c in drop if c in table.column_names]).to_pylist()


def candidate_row(n, source_id, state, questions, expected, track=None, **metadata):
    name = SPECS[n]
    return dict(id=f"{n}:{source_id}", benchmark=name, family=name, split="evaluation", state=state, questions=questions, expected=expected, metadata=dict(source_id=source_id, group_id=f"{n}:{source_id}", track=track or name, **metadata), scoring=dict(type="candidate_decisions", primary_field="q"))


def mmlu_pro(layout):
    path = layout.raw / "mmlu_pro" / HF["mmlu_pro"][2]
    for r in parquet(path):
        keys = letters(len(r["options"]))
        assert r["answer"] == keys[r["answer_index"]]
        yield candidate_row(57, r["question_id"], r["question"], {"q": dict(type="choice", instructions=CHOICE, criteria=dict(zip(keys, r["options"])))}, {"q": r["answer"]}, track=r["category"], src=r["src"], source_file=layout.rel(path), source_split="test")


def bbh(layout):
    folder = layout.repos / "cand-BIG-Bench-Hard/bbh"
    tasks = sorted(p.stem for p in folder.glob("*.json"))
    assert len(tasks) == 27
    for task in tasks:
        if task in BBH_EXCLUDED:
            continue
        for i, x in enumerate(json.loads((folder / f"{task}.json").read_text())["examples"]):
            text, target = x["input"], x["target"]
            if task in BBH_UNLISTED:
                assert "Options:" not in text
                criteria = {k: k for k in BBH_UNLISTED[task]}
            else:
                block = text.rsplit("\nOptions:\n", 1)[1].split("\n")
                if all(line.startswith("- ") for line in block):
                    criteria = {line[2:].strip(): line[2:].strip() for line in block}
                else:
                    parsed = [re.fullmatch(r"(\([A-Z]\)) (.*)", line, re.S) for line in block]
                    assert all(parsed), (task, i)
                    criteria = {m.group(1): m.group(2) for m in parsed}
                    assert list(criteria) == [f"({c})" for c in letters(len(parsed))], (task, i)
            if target not in criteria or len(criteria) < 2:
                continue
            family = re.sub(r"_(three|five|seven)_objects$", "", task)
            yield candidate_row(58, f"{task}:{i}", text, {"q": dict(type="choice", instructions=CHOICE, criteria=criteria)}, {"q": target}, track=task, task_family=family, source_file=layout.rel(folder / f"{task}.json"))


def ragtruth(layout):
    folder = layout.repos / "cand-RAGTruth/dataset"
    sources = {x["source_id"]: x for x in map(json.loads, (folder / "source_info.jsonl").open(encoding="utf-8"))}
    for r in map(json.loads, (folder / "response.jsonl").open(encoding="utf-8")):
        if r["split"] != "test":
            continue
        s = sources[r["source_id"]]
        yield candidate_row(59, r["id"], {"prompt": s["prompt"], "response": r["response"]}, {"q": dict(type="noul", instructions=RAGTRUTH_INSTRUCTIONS, criteria=RAGTRUTH_CRITERIA)}, {"q": len(r["labels"]) > 0}, track=s["task_type"], source_task_id=r["source_id"], responder_model=r["model"], quality=r["quality"], label_types=sorted({x["label_type"] for x in r["labels"]}))


def hover(layout):
    path = layout.repos / "cand-hover/data/hover/hover_dev_release_v1.1.json"
    db = sqlite3.connect(f"file:{layout.raw / HOVER_DB[0]}?mode=ro", uri=True)
    for r in json.loads(path.read_text(encoding="utf-8")):
        evidence = []
        for t in dict.fromkeys(t for t, _ in r["supporting_facts"]):
            found = db.execute("SELECT id, text FROM documents WHERE id=?", (unicodedata.normalize("NFD", t),)).fetchall()
            assert len(found) == 1, (r["uid"], t)
            evidence.append({"title": t, "text": found[0][1]})
        yield candidate_row(61, r["uid"], {"claim": r["claim"], "evidence": evidence}, {"q": dict(type="choice", instructions=HOVER_INSTRUCTIONS, criteria=HOVER_CRITERIA)}, {"q": r["label"]}, track=f"{r['num_hops']}-hop", num_hops=r["num_hops"], supporting_facts=r["supporting_facts"], hpqa_id=r["hpqa_id"])
    db.close()


def when2call(layout):
    path = layout.raw / "when2call" / HF["when2call"][2]
    order = ["direct", "tool_call", "request_for_info", "cannot_answer"]
    keys = letters(4)
    for r in map(json.loads, path.open(encoding="utf-8")):
        assert list(r["answers"]) == order
        yield candidate_row(62, r["uuid"], {"tools": [json.loads(t) for t in r["tools"]], "question": r["question"]}, {"q": dict(type="choice", instructions=WHEN2CALL_INSTRUCTIONS, criteria=dict(zip(keys, (r["answers"][k] for k in order))))}, {"q": keys[order.index(r["correct_answer"])]}, track=r["correct_answer"], answer_key_map=dict(zip(keys, order)), source=r["source"], bfcl_source_id=r["source_id"])


def newyorker(layout):
    path = layout.raw / "newyorker" / HF["newyorker"][2]
    for r in parquet(path, drop=("image",)):
        keys = letters(len(r["caption_choices"]))
        yield candidate_row(64, r["instance_id"], {"scene": r["image_location"], "description": r["image_description"], "uncanny_description": r["image_uncanny_description"], "entities": r["entities"]}, {"q": dict(type="choice", instructions=NEWYORKER_INSTRUCTIONS, criteria=dict(zip(keys, r["caption_choices"])))}, {"q": r["label"]}, track="matching-fold-0", contest_number=r["contest_number"])


def assignments(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {t.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name) and t.id in names}


def phishnchips(layout):
    source = layout.raw / "phishnchips/core_emails.csv"
    assert sha256(source) == HF["phishnchips"][3]
    questions = assignments(layout.repos / "jev-phishing-bench/run_jev.py", {"QUESTIONS"})["QUESTIONS"]
    for q in questions.values():
        q.setdefault("criteria", {})
    records = list(csv.DictReader(source.open(encoding="utf-8")))
    random.Random(20260916).shuffle(records)
    name = SPECS[56]
    for r in records:
        gold = bool(int(r["phish_label"]))
        email_data = json.loads(r["email_content"])
        state = {k: email_data[k] for k in ["sender", "from", "subject", "body", "link_display_text", "link_url"]}
        expected = {k: None for k in questions}
        expected.update(verdict="phishing" if gold else "legitimate", is_phishing=gold, verdict_alt_click="do_not_click" if gold else "click", verdict_alt_minimal="phishing" if gold else "legitimate")
        yield dict(id=f"56:{r['id']}", benchmark=name, family=name, split="evaluation", state=state, questions=questions, expected=expected, metadata=dict(source_id=r["id"], group_id=f"56:{r['id']}", track="native-nine-question", datasource=r["datasource"], url_category=r["url_category"], strategy=r["strategy"]), scoring=dict(type="calibration_decisions", primary_field="verdict", unscored_fields=[k for k in questions if k.startswith("sig_")]))


BUILDERS = {57: mmlu_pro, 58: bbh, 59: ragtruth, 61: hover, 62: when2call, 64: newyorker, 56: phishnchips}


def evaluation(n, row, encoder):
    payload = {"model": MODEL, "state": row["state"], "questions": row["questions"]}
    digest = hashlib.sha256(dumps(payload).encode()).hexdigest()
    if n == 56:
        return dict(run_id="calibration-v1:" + row["id"], catalog_id=n, dataset=row["benchmark"], track=row["metadata"]["track"], group_id=row["metadata"]["group_id"], payload_sha256=digest, benchmark_origin="Original-task benchmarks", proxy_tokens=max(1, len(dumps(payload)) // 4))
    proxy = len(encoder.encode(dumps({"state": row["state"], "questions": row["questions"]}), disallowed_special=()))
    return dict(run_id="candidates-v3:" + row["id"], catalog_id=n, dataset=row["benchmark"], track=row["metadata"]["track"], group_id=row["metadata"].pop("group", None) or row["metadata"]["group_id"], payload_sha256=digest, benchmark_origin="Original-task benchmarks", proxy_tokens=proxy, batch=BATCH[n])


def acquire(layout, numbers=ORDER, log=print):
    done = set()
    for n in numbers:
        for spec in NEEDS[n]:
            if spec in done:
                continue
            done.add(spec)
            kind, _, key = spec.partition(":")
            if kind == "git":
                url, rev, sparse = GIT[key]
                acq.git_fetch(layout.repos / key, url, rev, sparse, log)
            elif kind == "hf":
                repo, rev, rel, digest = HF[key]
                target = layout.raw / key / rel
                if not (target.exists() and sha256(target) == digest):
                    acq.hf_fetch(layout.raw / key, repo, rev, [rel], log)
                if sha256(target) != digest:
                    raise ValueError(f"{repo}/{rel}: sha256 differs from the pinned {digest}")
            elif kind == "db":
                rel, url, digest = HOVER_DB
                acq.http_fetch(layout.raw / rel, url, digest, log)
    for rel, digest in FILE_SHA.items():
        path = layout.repos / rel
        if path.exists() and sha256(path) != digest:
            raise ValueError(f"{rel}: sha256 differs from the pinned {digest}")


def build(layout, out_path, numbers=ORDER, log=print):
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    counts = {}
    digest = hashlib.sha256()
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for n in ORDER:
            if n not in numbers:
                continue
            counts[n] = 0
            for row in BUILDERS[n](layout):
                row["_evaluation"] = evaluation(n, row, encoder)
                line = dumps(row) + "\n"
                f.write(line)
                digest.update(line.encode())
                counts[n] += 1
            log(json.dumps({"event": "added_benchmark", "catalog_id": n, "dataset": SPECS[n], "requests": counts[n]}))
    return {"requests": sum(counts.values()), "counts": counts, "sha256": digest.hexdigest()}
