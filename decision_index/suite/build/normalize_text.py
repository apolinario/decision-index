import csv
import hashlib
import io
import json
import random
import re
import zipfile
from collections import Counter

from decision_index.suite.build.layout import dump_ascii, git_revision, sha256


def write(layout, name, rows):
    p = layout.normalized / (name + ".jsonl")
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(dump_ascii(row) + "\n")
    return dict(rows=len(rows), path=str(p.relative_to(layout.suite)), sha256=sha256(p))


def isarcasm(layout):
    repo = layout.repos / "isarcasm"
    revision = git_revision(repo)
    stats = {}
    for path in sorted((repo / "test").glob("*.csv")):
        task, language = path.stem.split("_")[1:3]
        family = f"iSarcasmEval-{task}-{language}"
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        target = layout.normalized / (family + ".jsonl")
        count = 0
        with path.open(encoding="utf-8-sig", newline="") as src, target.open("w", encoding="utf-8") as dst:
            for i, row in enumerate(csv.DictReader(src)):
                state, questions, expected = "", {}, {}
                if task == "C":
                    questions["answer"] = dict(type="choice", instructions="Which of these two texts is the sarcastic one?", criteria={"text_0": row["text_0"], "text_1": row["text_1"]})
                    assert row["sarcastic_id"] in {"0", "1"}
                    expected["answer"] = "text_" + row["sarcastic_id"]
                else:
                    state = row["text"]
                    labels = ["sarcastic"] if task == "A" else ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]
                    for label in labels:
                        assert row[label] in {"0", "1"}
                        instruction = "Is this text intended to be sarcastic?" if task == "A" else f'Does this text exhibit {label.replace("_", " ")}? Evaluate this category independently.'
                        questions[label] = dict(type="choice", instructions=instruction, criteria={"no": "No", "yes": "Yes"})
                        expected[label] = "yes" if row[label] == "1" else "no"
                obj = dict(id=f"{family}:test:{i}", family=family, split="test", state=state, questions=questions, expected=expected, metadata=dict(benchmark_number=40, source_file=layout.rel(path), source_sha256=source_hash, source_revision=revision, source_id=i, source="https://github.com/iabufarha/iSarcasmEval", task=task, language=language, mapping="native binary/pairwise decisions; B uses one binary question per label", primary_metric={"A": "sarcastic-class F1", "B": "macro-F1 across labels", "C": "accuracy"}[task]))
                dst.write(dump_ascii(obj) + "\n")
                count += 1
        stats[family] = dict(rows=count, path=str(target.relative_to(layout.suite)), sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (layout.suite / "isarcasm-normalization.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def sata(layout):
    repo = layout.repos / "sata"
    source = repo / "src/satabench/methods/data/sata_bench_final_2025.json"
    revision = git_revision(repo)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    target = layout.normalized / "SATA-Bench.jsonl"
    n = fields = 0
    with source.open() as src, target.open("w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            row = json.loads(line)
            choices = list(enumerate(row["choices"]))
            assert choices and all(type(pair[1]) is bool for _, pair in choices)
            random.Random(f"20260918:SATA:{i}").shuffle(choices)
            questions, expected, mapping = {}, {}, {}
            for j, (original_index, (text, label)) in enumerate(choices):
                key = f"option_{j}"
                questions[key] = dict(type="choice", instructions=f'Question: {row["question"]}\nDoes this candidate correctly answer the question?\nCandidate: {text}', criteria={"no": "No", "yes": "Yes"})
                expected[key] = "yes" if label else "no"
                mapping[key] = original_index
            obj = dict(id=f"SATA-Bench:test:{i}", family="SATA-Bench", split="test", state={"paragraph": row["paragraph"], "question": row["question"], "options": [p[0] for _, p in choices]}, questions=questions, expected=expected, metadata=dict(benchmark_number=33, source_file=layout.rel(source), source_sha256=source_hash, source_revision=revision, source_id=i, source_subset=row.get("dataset"), source_option_indices=mapping, source="https://github.com/sata-bench/sata-bench", mapping="select-all to independent binary choices; exact set is scoring unit", primary_metric="request exact-set accuracy", shuffle_seed="20260918:SATA:<source-index>", gold_rule="choices[i][1], matching upstream sata_prompt_provider.py; legacy answer letters refer to a different permutation"))
            dst.write(dump_ascii(obj) + "\n")
            n += 1
            fields += len(questions)
    stats = dict(rows=n, fields=fields, path=str(target.relative_to(layout.suite)), sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (layout.suite / "sata-normalization.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def contracts(layout):
    repo = layout.repos / "contractnli"
    source = repo / "resources/contract-nli.zip"
    with zipfile.ZipFile(source) as z:
        data = json.loads(z.read("contract-nli/test.json"))
    provenance = dict(benchmark_number=11, source="https://stanfordnlp.github.io/contract-nli/", source_revision=git_revision(repo), source_file=layout.rel(source), source_sha256=sha256(source), member="contract-nli/test.json", track="native NLI classification; evidence extraction not scored", primary_metric="label macro-F1 and hypothesis accuracy; document exact-match separately")
    criteria = {"Entailment": "The contract entails the hypothesis.", "Contradiction": "The contract contradicts the hypothesis.", "NotMentioned": "The hypothesis is neither entailed nor contradicted by the contract."}
    for doc in data["documents"]:
        assert len(doc["annotation_sets"]) == 1
        annotations = doc["annotation_sets"][0]["annotations"]
        assert set(annotations) == set(data["labels"])
        questions = {key: dict(type="choice", instructions="Classify the relationship between the contract and this hypothesis:\n" + label["hypothesis"], criteria=criteria) for key, label in data["labels"].items()}
        expected = {key: a["choice"] for key, a in annotations.items()}
        assert all(x in criteria for x in expected.values())
        yield dict(id=f'ContractNLI:test:{doc["id"]}', family="ContractNLI", split="test", state=doc["text"], questions=questions, expected=expected, metadata={**provenance, "source_id": doc["id"]})


def clinical(layout):
    repo = layout.repos / "nli4ct"
    source = repo / "test.json"
    gold_source = repo / "gold_test.json"
    rows = json.loads(source.read_text())
    gold = json.loads(gold_source.read_text())
    assert set(rows) == set(gold)
    archive = repo / "training_data.zip"
    provenance = dict(benchmark_number=42, source="https://github.com/ai-systems/Task-2-SemEval-2024", source_revision=git_revision(repo), source_file=layout.rel(source), source_sha256=sha256(source), gold_sha256=sha256(gold_source), trial_archive_sha256=sha256(archive), primary_metric="entailment F1; original consistency/faithfulness metrics require upstream grouping", context_rule="complete indicated clinical-trial section, no gold evidence selection", archive_note="training_data.zip also contains the trial documents used by test; no training statements emitted")
    criteria = {"Entailment": "The clinical trial evidence entails the statement.", "Contradiction": "The clinical trial evidence contradicts the statement."}
    with zipfile.ZipFile(archive) as z:
        for uid, row in rows.items():
            section = row["Section_id"]
            state = {}
            for key, role in [("Primary_id", "primary_trial"), ("Secondary_id", "secondary_trial")]:
                if key in row:
                    trial = json.loads(z.read("CT json/" + row[key] + ".json"))
                    state[role] = {"id": row[key], "section": section, "text": trial[section]}
            for key in ["Statement", "Section_id", "Primary_id", "Secondary_id"]:
                assert row.get(key) == gold[uid].get(key), (uid, key)
            label = gold[uid]["Label"]
            assert label in criteria
            yield dict(id=f"NLI4CT-2024:test:{uid}", family="NLI4CT-2024", split="test", state=state, questions={"answer": dict(type="choice", instructions="Classify the statement against the supplied clinical trial evidence:\n" + row["Statement"], criteria=criteria)}, expected={"answer": label}, metadata={**provenance, "source_id": uid, "trial_ids": [row[k] for k in ["Primary_id", "Secondary_id"] if k in row]})


def nli_documents(layout):
    def write_fields(name, rows):
        target = layout.normalized / (name + ".jsonl")
        n = fields = 0
        with target.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(dump_ascii(row) + "\n")
                n += 1
                fields += len(row["questions"])
        return dict(rows=n, fields=fields, path=str(target.relative_to(layout.suite)), sha256=sha256(target))

    stats = {"ContractNLI": write_fields("ContractNLI", contracts(layout)), "NLI4CT-2024": write_fields("NLI4CT-2024", clinical(layout))}
    (layout.suite / "nli-documents-normalization.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def humor(layout):
    path = layout.downloads / "humicroedit-full.zip"
    member = "semeval-2020-task-7-dataset/subtask-2/test.csv"
    with zipfile.ZipFile(path) as z:
        source = list(csv.DictReader(io.StringIO(z.read(member).decode("utf-8-sig"))))
    rows, ties = [], []
    for r in source:
        assert r["label"] in {"0", "1", "2"}
        if r["label"] == "0":
            ties.append(r["id"])
            continue
        criteria = {}
        for n in (1, 2):
            rendered, count = re.subn(r"<[^<>]+/>", lambda _: r[f"edit{n}"], r[f"original{n}"])
            assert count == 1, r["id"]
            criteria[f"headline_{n}"] = rendered
        means = [float(r["meanGrade1"]), float(r["meanGrade2"])]
        assert means[int(r["label"]) - 1] > means[2 - int(r["label"])]
        rows.append(dict(id=f'Humicroedit:test:{r["id"]}', family="Humicroedit", split="test", state="", questions={"answer": dict(type="choice", instructions="Which edited news headline is funnier?", criteria=criteria)}, expected={"answer": "headline_" + r["label"]}, metadata=dict(benchmark_number=21, source="https://cs.rochester.edu/u/nhossain/semeval-2020-task-7-dataset.zip", source_file=layout.rel(path), source_sha256=sha256(path), member=member, source_id=r["id"], primary_metric="pairwise accuracy on non-ties, matching official score_task_2.py", human_mean_grades=means, mapping="replace exactly the marked token; no grades supplied to model")))
    result = write(layout, "Humicroedit", rows)
    result.update(original_rows=len(source), tie_rows=len(ties), excluded_tie_ids=ties, tie_policy="Official scorer excludes label0 ties; retained in raw source, no forced winner.")
    return result


def gpqa(layout):
    path = layout.sources / "gpqa/dataset.zip"
    with zipfile.ZipFile(path) as z:
        source = list(csv.DictReader(io.StringIO(z.read("dataset/gpqa_diamond.csv", pwd=b"deserted-untie-orchid").decode("utf-8-sig"))))
    rows = []
    source_hash = sha256(path)
    for i, r in enumerate(source):
        options = [r["Correct Answer"]] + [r[f"Incorrect Answer {n}"] for n in (1, 2, 3)]
        order = list(range(4))
        random.Random(f"20260918:GPQA:{i}").shuffle(order)
        criteria = {chr(65 + k): options[j] for k, j in enumerate(order)}
        rows.append(dict(id=f"GPQA-Diamond:test:{i}", family="GPQA-Diamond", split="test", state="", questions={"answer": dict(type="choice", instructions=r["Question"], criteria=criteria)}, expected={"answer": chr(65 + order.index(0))}, metadata=dict(benchmark_number=25, source_file=layout.rel(path), source_sha256=source_hash, source_id=i, member="dataset/gpqa_diamond.csv", source="https://github.com/idavidrein/gpqa", shuffle_seed="20260918:GPQA:<source-index>", source_option_indices=order, handling="LOCAL EVALUATION ONLY; do not upload benchmark text or tokens to public artifacts", primary_metric="accuracy")))
    assert len(rows) == 198
    return write(layout, "GPQA-Diamond", rows)


def humor_gpqa(layout):
    stats = {"Humicroedit": humor(layout), "GPQA-Diamond": gpqa(layout)}
    (layout.suite / "humor-gpqa-normalization.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def _mc_row(benchmark, split, sid, instructions, options, gold, **meta):
    keys = [chr(65 + i) for i in range(len(options))]
    assert 2 <= len(options) <= 255 and 0 <= gold < len(options)
    return {"id": f"{benchmark}:{split}:{sid}", "benchmark": benchmark, "family": benchmark, "split": split, "state": {}, "questions": {"q1": {"type": "choice", "instructions": instructions, "criteria": dict(zip(keys, options))}}, "gold": {"q1": keys[gold]}, "expected": {"q1": keys[gold]}, "provenance": meta}


def _write_mc(layout, name, rows):
    p = layout.normalized / f"{name}.jsonl"
    p.write_text("".join(dump_ascii(x) + "\n" for x in rows))
    return {"path": layout.rel(p), "count": len(rows), "sha256": sha256(p)}


def vast(layout):
    p = layout.raw / "vast/data/VAST/vast_test.csv"
    sh = sha256(p)
    labels = ["against", "favor", "neutral"]
    rows = []
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = int(r["label"])
            assert 0 <= label < 3
            prompt = f"Topic: {r['topic_str']}\nPost: {r['post']}\nDetermine the stance of the post toward the topic."
            rows.append(_mc_row("VAST", "test", r["new_id"], prompt, labels, label, source_id=r["new_id"], source_file=layout.rel(p), source_sha256=sh, source_label=r["label"]))
    return _write_mc(layout, "VAST", rows), {"source_rows": len(rows), "label_counts": {x: sum(1 for r in rows if r["gold"]["q1"] == chr(65 + i)) for i, x in enumerate(labels)}, "label_mapping": dict(enumerate(labels)), "source_sha256": sh}


def cladder(layout):
    zp = layout.raw / "cladder/data/cladder-v1.zip"
    sh = sha256(zp)
    with zipfile.ZipFile(zp) as z:
        qs = json.loads(z.read("cladder-v1-q-balanced.json"))
        models = {x["model_id"]: x for x in json.loads(z.read("cladder-v1-meta-models.json"))}
    rows = []
    counts = {}
    for q in qs:
        meta = q["meta"]
        model = models[meta["model_id"]]
        prompt = "\n\n".join([model["background"], q["given_info"], q["question"]])
        ans = q["answer"]
        assert ans in ("yes", "no")
        rows.append(_mc_row("CLadder", "balanced", q["question_id"], prompt, ["yes", "no"], 0 if ans == "yes" else 1, source_file=layout.rel(zp), source_sha256=sh, source_question_id=q["question_id"], story_id=meta.get("story_id"), graph_id=meta.get("graph_id"), query_type=meta.get("query_type"), rung=meta.get("rung")))
        counts[ans] = counts.get(ans, 0) + 1
    return _write_mc(layout, "CLadder", rows), {"source_rows": len(rows), "answer_counts": counts, "source_sha256": sh, "variant": "cladder-v1-q-balanced.json", "prompt_recipe": "meta.background + given_info + question"}


def vast_cladder(layout):
    v, va = vast(layout)
    c, ca = cladder(layout)
    audit = {"schema": "jev-eval-row-v1", "sources": {"VAST": v, "CLadder": c}, "mapping": {"VAST": va, "CLadder": ca}, "request_exclusion": ["gold", "expected", "reasoning", "CLadder meta groundtruth/model parameters"]}
    (layout.suite / "vast-cladder-mapping.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def _parse_hle_choices(question):
    marker = re.search(r"(?im)^\s*Answer Choices:\s*$", question)
    if not marker:
        return None, "missing_answer_choices_marker"
    text = question[marker.end():]
    matches = list(re.finditer(r"(?m)^\s*([A-Z])\.\s*(.*)$", text))
    if not matches:
        return None, "no_option_headers"
    opts = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        first = m.group(2).strip()
        rest = text[m.end():end].strip()
        value = "\n".join(x.strip() for x in ([first] if first else []) + [rest] if x.strip()).strip()
        if not value:
            return None, f"empty_option_{m.group(1)}"
        opts.append((m.group(1), value))
    keys = [k for k, _ in opts]
    expected = [chr(ord("A") + i) for i in range(len(keys))]
    if keys != expected:
        return None, "noncontiguous_option_labels"
    if len(opts) < 2:
        return None, "fewer_than_two_options"
    return opts, None


def hle(layout):
    import pyarrow.parquet as pq

    source = layout.raw / "hle/data/test-00000-of-00001.parquet"
    source_meta = layout.raw / "hle/.cache/huggingface/download/data/test-00000-of-00001.parquet.metadata"
    out_path = layout.normalized / "HLE-text-MC.jsonl"
    rows = pq.read_table(source).to_pylist()
    out = []
    reasons = Counter()
    excluded = {}
    source_hash = sha256(source)
    cache_meta = source_meta.read_text().splitlines() if source_meta.exists() else []
    cache_signature = cache_meta[0] if cache_meta else None
    cache_etag = cache_meta[1] if len(cache_meta) > 1 else None
    for r in rows:
        rid = str(r["id"])

        def reject(reason):
            reasons[reason] += 1
            excluded.setdefault(reason, []).append(rid)

        if r.get("image") or r.get("image_preview"):
            reject("image_present")
            continue
        if r.get("answer_type") != "multipleChoice":
            reject("not_multipleChoice")
            continue
        opts, reason = _parse_hle_choices(r.get("question") or "")
        if reason:
            reject(reason)
            continue
        answer = str(r.get("answer", "")).strip().upper()
        keys = [k for k, _ in opts]
        if answer not in keys:
            reject("gold_not_in_options")
            continue
        out.append({"id": f"HLE-text-MC:test:{rid}", "benchmark": "HLE-text-MC", "family": "HLE-text-MC", "split": "test", "state": {}, "questions": {"q1": {"type": "choice", "instructions": r["question"].split("Answer Choices:", 1)[0].strip(), "criteria": dict(opts)}}, "gold": {"q1": answer}, "expected": {"q1": answer}, "provenance": {"source_id": rid, "source_file": layout.rel(source), "source_sha256": source_hash, "source_cache_signature": cache_signature, "source_cache_etag": cache_etag, "answer_type": "multipleChoice", "raw_subject": r.get("raw_subject"), "category": r.get("category")}})
    out_path.write_text("".join(dump_ascii(x) + "\n" for x in out))
    audit = {"schema": "jev-eval-row-v1", "source_file": layout.rel(source), "source_sha256": source_hash, "source_rows": len(rows), "emitted_rows": len(out), "excluded_rows": len(rows) - len(out), "exclusion_counts": dict(reasons), "excluded_ids": excluded, "rules": ["answer_type must equal multipleChoice", "image and image_preview must both be empty", "Answer Choices marker and contiguous A..Z options required", "gold answer must match an emitted option", "rationale/rationale_image/author/canary omitted"]}
    (layout.suite / "hle-normalization.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def esci(layout):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    repo = layout.repos / "esci"
    source = repo / "shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
    products_path = repo / "shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    revision = git_revision(repo)
    examples = pq.read_table(source, filters=[("split", "=", "test")])
    products = pq.read_table(products_path)
    keys = ["product_locale", "product_id"]
    joined = examples.join(products, keys=keys, join_type="left outer")
    assert joined.num_rows == examples.num_rows, "Nonunique product join keys"
    assert pc.sum(pc.is_null(joined["product_title"])).as_py() == 0, "Missing product text"
    criteria = {"E": "Exact: the product satisfies the search query.", "S": "Substitute: a product that could substitute for the requested product.", "C": "Complement: a product that complements the requested product.", "I": "Irrelevant: the product does not address the requested product need."}
    provenance = dict(benchmark_number=37, source="https://github.com/amazon-science/esci-data", source_revision=revision, source_sha256=sha256(source), product_source_sha256=sha256(products_path), source_file=layout.rel(source), product_source_file=layout.rel(products_path), primary_metric="micro-F1 and macro-F1 for native 4-class ESCI classification", mapping="all released test query-product pairs; no candidate sampling")
    target = layout.normalized / "Amazon-ESCI.jsonl"
    tmp = target.with_suffix(".jsonl.partial")
    count = 0
    labels, locales = {}, {}
    joined = joined.sort_by([("example_id", "ascending")])
    with tmp.open("w", encoding="utf-8") as f:
        for batch in joined.to_batches(max_chunksize=4096):
            for row in batch.to_pylist():
                label = row["esci_label"]
                assert label in criteria
                product = {key.removeprefix("product_"): row[key] for key in ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"] if row.get(key) is not None}
                obj = dict(id=f'Amazon-ESCI:test:{row["example_id"]}', family="Amazon-ESCI", split="test", state={"search_query": row["query"], "product": product}, questions={"answer": dict(type="choice", instructions="Classify the relevance of this product to the search query using the ESCI categories.", criteria=criteria)}, expected={"answer": label}, metadata={**provenance, "source_id": row["example_id"], "query_id": row["query_id"], "product_id": row["product_id"], "locale": row["product_locale"], "small_version": row["small_version"], "large_version": row["large_version"]})
                f.write(dump_ascii(obj) + "\n")
                count += 1
                labels[label] = labels.get(label, 0) + 1
                locales[row["product_locale"]] = locales.get(row["product_locale"], 0) + 1
    assert count == examples.num_rows
    tmp.replace(target)
    stats = dict(rows=count, path=str(target.relative_to(layout.suite)), sha256=sha256(target), source_revision=revision, label_counts=labels, locale_counts=locales, source_test_rows=examples.num_rows)
    (layout.suite / "esci-normalization.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


BUILDERS = {40: isarcasm, 33: sata, 11: nli_documents, 42: nli_documents, 21: humor_gpqa, 25: humor_gpqa, 41: vast_cladder, 44: vast_cladder, 45: hle, 37: esci}
