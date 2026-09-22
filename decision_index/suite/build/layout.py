import hashlib
import json
import subprocess
from pathlib import Path

REFERENCE_MODEL = "jev-1.13.0"


class Layout:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.suite = self.root / "artifacts/benchmark-suite"
        self.raw = self.suite / "raw"
        self.repos = self.raw / "repos"
        self.downloads = self.raw / "downloads"
        self.normalized = self.suite / "normalized"
        self.requests = self.suite / "requests"
        self.sources = self.root / "data/sources"
        self.home = self.suite / "home"
        for p in (self.raw, self.repos, self.downloads, self.normalized, self.requests, self.sources):
            p.mkdir(parents=True, exist_ok=True)

    def rel(self, path):
        return str(Path(path).relative_to(self.root))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision(repo):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def dump_ascii(x):
    return json.dumps(x, ensure_ascii=True, separators=(",", ":"))


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(dump_ascii(row) + "\n")
            n += 1
    return n


def write_requests(layout, name, rows):
    path = layout.requests / (name + ".jsonl")
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(dump_ascii({"model": REFERENCE_MODEL, "state": r["state"], "questions": r["questions"]}) + "\n")


def choice_row(benchmark, split, source_id, instructions, options, gold, **extra):
    assert 2 <= len(options) <= 255 and 0 <= gold < len(options)
    keys = [chr(65 + i) if len(options) <= 26 else f"option_{i}" for i in range(len(options))]
    labels = dict(zip(keys, options))
    gold_key = keys[gold]
    return {"id": f"{benchmark}:{split}:{source_id}", "benchmark": benchmark, "family": benchmark, "split": split, "state": {}, "questions": {"q1": {"type": "choice", "instructions": instructions, "criteria": labels}}, "gold": {"q1": gold_key}, "expected": {"q1": gold_key}, "provenance": {"source_id": str(source_id), **extra}}


NAMES = {
    11: ["ContractNLI"], 12: ["ANLI"], 21: ["Humicroedit"], 24: ["MMLU"], 25: ["GPQA-Diamond"], 26: ["ARC-Easy"],
    27: ["ARC-Challenge"], 28: ["WinoGrande"], 29: ["HellaSwag"], 4: ["BANKING77"], 5: ["CLINC150+OOS"], 32: ["MuSR"],
    33: ["SATA-Bench"], 34: ["SimpleBench"], 37: ["Amazon-ESCI"], 40: ["iSarcasmEval-A-Ar", "iSarcasmEval-A-En", "iSarcasmEval-B-En", "iSarcasmEval-C-Ar", "iSarcasmEval-C-En"],
    41: ["VAST"], 42: ["NLI4CT-2024"], 44: ["CLadder"], 45: ["HLE-text-MC"],
    1: ["BFCL-tool-selection"], 9: ["Home-Appliance"], 10: ["SGD-service-given-intent"],
    31: ["ChessBench-legal-move"], 38: ["ACOS-category-sentiment"], 39: ["FinEntity-entity-given"],
    30: ["GSM8K-4choice", "GSM8K-10choice"], 43: ["CRUXEval-output-choice"], 50: ["Habermas-consensus"],
    2: ["ToolRet-retrieval"], 3: ["API-Bank-tool-selection"], 6: ["RouterBench-0shot", "RouterBench-5shot"],
    20: ["BPoMP-original-limerick"], 22: ["POP909-chord-pitch-class"], 23: ["CFColor-preference"],
    36: ["BRIGHT-retrieval"], 48: ["ForecastBench-binary"],
}

DATASETS = {
    1: "BFCL", 2: "ToolRet", 3: "API-Bank", 4: "BANKING77", 5: "CLINC150+OOS", 6: "RouterBench", 9: "Home appliance simulator",
    10: "SGD/SGD-X", 11: "ContractNLI", 12: "ANLI", 20: "BPoMP", 21: "Humicroedit", 22: "POP909-CL", 23: "cfcolor", 24: "MMLU",
    25: "GPQA Diamond", 26: "ARC-Easy", 27: "ARC-Challenge", 28: "WinoGrande", 29: "HellaSwag", 30: "GSM8K", 31: "ChessBench",
    32: "MuSR", 33: "SATA-Bench", 34: "SimpleBench", 36: "BRIGHT", 37: "Amazon ESCI", 38: "ACOS", 39: "FinEntity", 40: "iSarcasmEval",
    41: "VAST", 42: "NLI4CT", 43: "CRUXEval", 44: "CLadder", 45: "HLE", 48: "ForecastBench", 50: "Habermas Machine",
}

STATIC_ORDER = sorted(DATASETS)
