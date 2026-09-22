import csv
import json
import re

from decision_index.suite.build.layout import choice_row, dump_ascii, sha256


def write(layout, name, rows):
    p = layout.normalized / f"{name}.jsonl"
    p.write_text("".join(dump_ascii(r) + "\n" for r in rows), encoding="utf-8")
    return {"path": str(p.relative_to(layout.suite)), "count": len(rows), "sha256": sha256(p)}


def knowledge(layout):
    import pyarrow.parquet as pq

    src = layout.sources
    out = {}

    def meta(path):
        return {"source_file": layout.rel(path), "source_sha256": sha256(path)}

    for family in ("ARC-Easy", "ARC-Challenge"):
        sp = src / "arc" / family / "test-00000-of-00001.parquet"
        rs = pq.read_table(sp).to_pylist()
        out[family] = write(layout, family, [choice_row(family, "test", r["id"], r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]), **meta(sp)) for r in rs])
    sp = src / "mmlu/all/test-00000-of-00001.parquet"
    rs = pq.read_table(sp).to_pylist()
    out["MMLU"] = write(layout, "MMLU", [choice_row("MMLU", "test", i, r["question"], r["choices"], r["answer"], subject=r.get("subject"), **meta(sp)) for i, r in enumerate(rs)])
    sp = src / "winogrande/winogrande_xl/validation-00000-of-00001.parquet"
    rs = pq.read_table(sp).to_pylist()
    out["WinoGrande"] = write(layout, "WinoGrande", [choice_row("WinoGrande", "validation", i, "Which option correctly fills the blank?\n" + r["sentence"], [r["option1"], r["option2"]], int(r["answer"]) - 1, **meta(sp)) for i, r in enumerate(rs)])
    sp = src / "hellaswag/data/validation-00000-of-00001.parquet"
    rs = pq.read_table(sp).to_pylist()
    clean = lambda s: " ".join(str(s).split())
    out["HellaSwag"] = write(layout, "HellaSwag", [choice_row("HellaSwag", "validation", f"{i}:{r['ind']}", "Which continuation is most plausible?\n" + clean(r["ctx"]), [clean(x) for x in r["endings"]], int(r["label"]), **meta(sp)) for i, r in enumerate(rs)])
    return out


def anli(layout):
    import pyarrow.parquet as pq

    rows = []
    mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for p in sorted((layout.raw / "anli" / "plain_text").glob("test_*.parquet")):
        split = p.stem.split("-")[0]
        for r in pq.read_table(p).to_pylist():
            rows.append(choice_row("ANLI", split, r["uid"], "Classify the relationship between the premise and hypothesis.\nPremise: " + r["premise"] + "\nHypothesis: " + r["hypothesis"], list(mapping.values()), int(r["label"]), source_file=str(p.relative_to(layout.suite)), label_names=mapping))
    return write(layout, "ANLI", rows)


def banking77(layout):
    bp = layout.raw / "banking77" / "test.csv"
    with bp.open(newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    labels = json.loads((layout.raw / "banking77/categories.json").read_text())
    assert len(labels) == 77 and set(r["category"] for r in data) <= set(labels)
    sh = sha256(bp)
    rows = [choice_row("BANKING77", "test", i, "Classify the banking intent of this user request:\n" + r["text"], labels, labels.index(r["category"]), **{"source_file": str(bp.relative_to(layout.suite)), "source_sha256": sh}) for i, r in enumerate(data)]
    return write(layout, "BANKING77", rows)


def clinc150(layout):
    cp = layout.raw / "clinc150" / "data_full.json"
    obj = json.loads(cp.read_text())
    rows = []
    labels = sorted({pair[1] for pair in obj["train"]} | {"oos"})
    assert len(labels) == 151
    descriptions = ["out of scope: none of the listed intents" if label == "oos" else label.replace("_", " ") for label in labels]
    sh = sha256(cp)
    for split in ("test", "oos_test"):
        for i, (text, label) in enumerate(obj.get(split, [])):
            rows.append(choice_row("CLINC150+OOS", split, i, "Classify the intent of this user request, or choose out of scope if none applies:\n" + text, descriptions, labels.index(label), **{"source_file": str(cp.relative_to(layout.suite)), "source_sha256": sh, "label_names": labels}))
    return write(layout, "CLINC150+OOS", rows)


def simplebench(layout):
    sp = layout.raw / "simplebench" / "simple_bench_public.json"
    items = json.loads(sp.read_text())["eval_data"]
    rows = []
    for x in items:
        lines = x["prompt"].splitlines()
        opts, body = [], []
        for line in lines:
            m = re.match(r"^([A-Z])\.\s*(.*)$", line)
            if m:
                opts.append((m.group(1), m.group(2)))
            else:
                body.append(line)
        assert len(opts) >= 2 and x.get("answer") in {k for k, _ in opts}, x["question_id"]
        rows.append(choice_row("SimpleBench", "public", x["question_id"], "\n".join(body).strip(), [v for _, v in opts], [k for k, _ in opts].index(x["answer"]), **{"source_file": str(sp.relative_to(layout.suite)), "source_sha256": sha256(sp)}))
    return write(layout, "SimpleBench", rows)


def musr(layout):
    mp = layout.raw / "musr" / "datasets"
    rows = []
    for fp in sorted(mp.glob("*.json")):
        for i, item in enumerate(json.loads(fp.read_text())):
            for j, q in enumerate(item.get("questions", [])):
                opts = q.get("choices", [])
                assert len(opts) >= 2 and isinstance(q.get("answer"), int) and 0 <= q["answer"] < len(opts)
                rows.append(choice_row("MuSR", "test", f"{fp.stem}:{i}:{j}", item.get("context", "") + "\n\n" + q["question"], opts, q["answer"], source_file=str(fp.relative_to(layout.suite)), source_sha256=sha256(fp)))
    return write(layout, "MuSR", rows)


BUILDERS = {24: knowledge, 26: knowledge, 27: knowledge, 28: knowledge, 29: knowledge, 12: anli, 4: banking77, 5: clinc150, 34: simplebench, 32: musr}
