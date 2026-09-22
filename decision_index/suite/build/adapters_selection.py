import ast
import hashlib
import json
import random
from collections import Counter
from decimal import Decimal

from decision_index.suite.build.layout import REFERENCE_MODEL, dump_ascii, sha256, write_jsonl, write_requests

SEED = 20260917


def num(n):
    return format(n.normalize(), "f")


def gsm(layout):
    import pyarrow.parquet as pq

    p = layout.sources / "gsm8k/main/test-00000-of-00001.parquet"
    rows = []
    for i, r in enumerate(pq.read_table(p).to_pylist()):
        gold = Decimal(r["answer"].split("####")[-1].strip().replace(",", ""))
        pool = sorted(({gold + Decimal(d) for d in [-10, -5, -2, -1, 1, 2, 5, 10]} | {gold * 2, gold * 10, gold // 2}) - {gold})
        random.Random(f"{SEED}:GSM:{i}").shuffle(pool)
        for count in (4, 10):
            opts = [num(gold)] + [num(x) for x in pool[:count - 1]]
            order = list(range(count))
            random.Random(f"{SEED}:GSM:{i}:{count}").shuffle(order)
            ordered = [opts[j] for j in order]
            correct = ordered.index(num(gold))
            rows.append({"id": f"GSM8K-{count}:test:{i}", "family": f"GSM8K-{count}", "split": "test", "state": {"question": r["question"], "task": f"GSM8K deterministic {count}-choice numeric selection"}, "questions": {"answer": {"type": "choice", "instructions": "Choose the numeric answer to the problem in state. Do not provide reasoning. This is a named multiple-choice adaptation of GSM8K.", "criteria": {f"option_{j}": v for j, v in enumerate(ordered)}}}, "expected": {"answer": f"option_{correct}"}, "metadata": {"group_id": f"GSM8K:{i}", "provenance": {"source": layout.rel(p), "source_sha256": sha256(p), "source_id": i, "seed": SEED}, "adaptation_scope": "Deterministic numeric multiple-choice selection; native GSM8K exact-answer score is not claimed.", "gold_numeric": num(gold)}})
    write_jsonl(layout.normalized / "GSM8K-4choice.jsonl", [r for r in rows if r["family"] == "GSM8K-4"])
    write_jsonl(layout.normalized / "GSM8K-10choice.jsonl", [r for r in rows if r["family"] == "GSM8K-10"])
    return {"rows": len(rows), "source_cases": len(rows) // 2, "source_sha256": sha256(p), "excluded": []}


def crux(layout):
    base = layout.repos / "cruxeval"
    src = base / "data/cruxeval.jsonl"
    gen = base / "samples/model_generations/sample_codellama-7b_temp0.2_output/generations.json"
    ev = base / "samples/evaluation_results/sample_scored_codellama-7b_temp0.2_output.json"
    g = json.loads(gen.read_text())
    flags = json.loads(ev.read_text())["raw_scored_generations"]
    source = {json.loads(x)["id"]: json.loads(x) for x in src.open()}
    rows = []
    excluded = []
    for sid, item in sorted(source.items()):
        vals = []
        gold = ast.literal_eval(item["output"])

        def add(v):
            try:
                key = (type(v).__name__, repr(v))
            except Exception:
                return
            if not any(v == prior for _, prior in vals):
                vals.append((key, v))

        add(gold)
        assert len(g.get(sid, [])) == len(flags.get(sid, [])), sid
        for raw, ok in zip(g.get(sid, []), flags.get(sid, [])):
            assert type(ok) is bool, (sid, ok)
            if ok:
                continue
            try:
                add(ast.literal_eval(raw))
            except (ValueError, SyntaxError, MemoryError, TypeError):
                continue
        if len(vals) < 2:
            excluded.append({"id": sid, "reason": "fewer than two distinct valid choices"})
            continue
        vals = vals[:10]
        random.Random(f"{SEED}:CRUX:{sid}").shuffle(vals)
        choices = [repr(v) for _, v in vals]
        correct = next(i for i, (_, v) in enumerate(vals) if v == gold)
        assert sum(v == gold for _, v in vals) == 1, sid
        rows.append({"id": f"CRUXEval-output-choice:test:{sid}", "family": "CRUXEval-output-choice", "split": "test", "state": {"code": item["code"], "input": item["input"], "task": "CRUXEval output selection"}, "questions": {"answer": {"type": "choice", "instructions": "Choose the correct output of f called with the supplied input arguments. Candidates are Python literal values.", "criteria": {f"option_{i}": v for i, v in enumerate(choices)}}}, "expected": {"answer": f"option_{correct}"}, "metadata": {"group_id": sid, "provenance": {"source": layout.rel(src), "source_sha256": sha256(src), "generations_source": layout.rel(gen), "generations_sha256": sha256(gen), "evaluation_source": layout.rel(ev), "evaluation_sha256": sha256(ev), "seed": SEED}, "adaptation_scope": "Frozen output selection; not pass@1 or code execution.", "candidate_count": len(choices)}})
    write_jsonl(layout.normalized / "CRUXEval-output-choice.jsonl", rows)
    return {"rows": len(rows), "source_cases": len(source), "excluded": excluded}


def reasoning_selection(layout):
    a = {"GSM8K": gsm(layout), "CRUXEval": crux(layout)}
    (layout.suite / "reasoning-selection-audit.json").write_text(json.dumps(a, indent=2, ensure_ascii=True) + "\n")
    return a


APIBANK_SEED = 20260918


def apibank_catalog(base):
    out = {}
    for p in sorted((base / "apis").glob("*.py")):
        if p.name in ("api.py", "__init__.py"):
            continue
        tree = ast.parse(p.read_text())
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
        if not cls:
            continue
        vals = {}
        for n in cls.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("description", "input_parameters", "output_parameters"):
                try:
                    vals[n.targets[0].id] = ast.literal_eval(n.value)
                except (ValueError, SyntaxError):
                    vals[n.targets[0].id] = None
        out[cls.name] = {"name": cls.name, "description": vals.get("description", ""), "input_parameters": vals.get("input_parameters", {}), "output_parameters": vals.get("output_parameters", {})}
    return out


def apibank(layout):
    base = layout.repos / "apibank/api-bank"
    tools = apibank_catalog(base)
    rows = []
    excluded = []
    files = sorted((base / "lv1-lv2-samples").rglob("*.jsonl"))
    for p in files:
        turns = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        api_positions = [i for i, t in enumerate(turns) if t.get("role") == "API"]
        if not api_positions:
            excluded.append({"source": layout.rel(p), "reason": "no API decision"})
            continue
        for pos in api_positions:
            current = turns[pos]
            gold = current.get("api_name")
            if gold not in tools:
                excluded.append({"source": layout.rel(p), "position": pos, "reason": "gold tool absent from catalog", "gold_tool": gold})
                continue
            names = sorted(tools)
            order = list(range(len(names)))
            random.Random(f"{APIBANK_SEED}:{p.name}:{pos}").shuffle(order)
            labels = {f"option_{i}": names[j] for i, j in enumerate(order)}
            goldkey = next(k for k, v in labels.items() if v == gold)
            qs = {"tool": {"type": "choice", "instructions": "Select the single API tool to invoke for the dialogue prefix in state. Use the published catalog in state. Do not infer arguments or use the current/future API result.", "criteria": {k: v for k, v in labels.items()}}}
            rows.append({"id": f"API-Bank-tool-selection:{p.stem}:api-{pos}", "family": "API-Bank-tool-selection", "split": "evaluation-only", "state": {"dialogue": turns[:pos], "available_tools": [{k: tools[k] for k in names}[k] for k in names], "task": "API-Bank current API tool selection"}, "questions": qs, "expected": {"tool": goldkey}, "metadata": {"provenance": {"source": layout.rel(p), "source_sha256": sha256(p), "turn_index": pos}, "gold_tool": gold, "candidate_policy": "full published API catalog; source available subset is not documented in this sample format", "adaptation_scope": "Tool-name selection only; arguments and execution are not claimed."}})
    out = layout.normalized / "API-Bank-tool-selection.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(dump_ascii(r) + "\n")
    write_requests(layout, "API-Bank-tool-selection", rows)
    audit = {"rows": len(rows), "files": len(files), "catalog_tools": len(tools), "excluded": excluded}
    (layout.suite / "apibank-selection-audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n")
    return audit


def bpomp(layout):
    rows = []
    exclusions = []
    originals = {}
    files = []
    for p in sorted(layout.downloads.glob("BPoMP_p*.json")):
        files.append(p)
        file_hash = sha256(p)
        for variant, pairs in json.loads(p.read_text()).items():
            for i, pair in enumerate(pairs):
                if not isinstance(pair, list) or len(pair) < 2:
                    exclusions.append({"source": str(p), "variant": variant, "index": i, "reason": "malformed"})
                    continue
                a, b = pair[:2]
                if a == b:
                    exclusions.append({"source": str(p), "variant": variant, "index": i, "reason": "identical pair"})
                    continue
                key = hashlib.sha256(a.encode()).hexdigest()
                originals.setdefault(key, a)
                opts = [a, b]
                order = [0, 1]
                random.Random(f"{APIBANK_SEED}:{key}:{variant}:{i}").shuffle(order)
                ordered = [opts[j] for j in order]
                rows.append({"id": f"BPoMP:original-limerick:{p.stem}:{variant}:{i}", "family": "BPoMP-original-limerick", "split": "evaluation-only", "state": {"task": "BPoMP original-versus-perturbed discrimination"}, "questions": {"answer": {"type": "choice", "instructions": "Which candidate is the original released limerick? Judge rhyme, meter, and linguistic coherence. The original is not defined by subjective human preference.", "criteria": {f"option_{j}": ordered[j] for j in range(2)}}}, "expected": {"answer": f"option_{ordered.index(a)}"}, "metadata": {"group_id": key, "variant": variant, "provenance": {"source": layout.rel(p), "source_sha256": file_hash, "source_index": i}, "adaptation_scope": "Original-limerick discrimination; does not claim human preference labels."}})
    audit = {"rows": len(rows), "source_files": [{"path": layout.rel(p), "sha256": sha256(p)} for p in files], "variants": Counter(r["metadata"]["variant"] for r in rows), "excluded": exclusions, "scoring": "accuracy and macro accuracy by perturbation variant; grouped resampling by underlying poem hash"}
    with (layout.normalized / "BPoMP-original-limerick.jsonl").open("w") as f:
        for r in rows:
            f.write(dump_ascii(r) + "\n")
    write_requests(layout, "BPoMP-original-limerick", rows)
    (layout.suite / "bpomp-adaptation-audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n")
    return {"rows": len(rows), "excluded": len(exclusions)}


BUILDERS = {30: reasoning_selection, 43: reasoning_selection, 3: apibank, 20: bpomp}
