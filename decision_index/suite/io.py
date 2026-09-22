import gzip
import hashlib
import json
from pathlib import Path


def dumps(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def sha256_file(path, gunzip=False):
    path = Path(path)
    opener = gzip.open if gunzip else open
    with opener(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def open_text(path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def read_jsonl(path, complete_lines_only=False):
    with open_text(path) as f:
        for line in f:
            if complete_lines_only and not line.endswith("\n"):
                break
            if line.strip():
                yield json.loads(line)


def atomic_json(path, data, indent=2):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent, ensure_ascii=False) + "\n")
    tmp.replace(path)


class Suite:
    def __init__(self, directory):
        from decision_index import constants as C

        self.directory = Path(directory)
        self.rows_path = self.directory / C.SUITE_ROWS_FILE
        self.exclusions_path = self.directory / C.SUITE_EXCLUSIONS_FILE
        self.manifest_path = self.directory / C.SUITE_MANIFEST_FILE
        if not self.rows_path.exists():
            raise FileNotFoundError(f"missing {self.rows_path}; run `decision-index suite download` first")

    def manifest(self):
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    def excluded(self):
        if not self.exclusions_path.exists():
            return set()
        return set(json.loads(self.exclusions_path.read_text())["rows"])

    def rows(self, apply_exclusions=False):
        excluded = self.excluded() if apply_exclusions else set()
        for r in read_jsonl(self.rows_path):
            if r["_evaluation"]["run_id"] in excluded:
                continue
            yield r

    def verify(self, strict=True):
        from decision_index import constants as C

        actual = sha256_file(self.rows_path)
        report = {"rows_file": str(self.rows_path), "sha256": actual, "expected_sha256": C.SUITE_ROWS_GZ_SHA256, "match": actual == C.SUITE_ROWS_GZ_SHA256}
        if not report["match"]:
            inner = sha256_file(self.rows_path, gunzip=True)
            report.update(uncompressed_sha256=inner, uncompressed_match=inner == C.SUITE_ROWS_SHA256)
            report["match"] = report["uncompressed_match"]
        if self.exclusions_path.exists():
            ex = sha256_file(self.exclusions_path)
            report.update(exclusions_sha256=ex, exclusions_match=ex == C.SUITE_EXCLUSIONS_SHA256)
        if strict and not report["match"]:
            raise ValueError(f"frozen suite hash mismatch: {report}")
        return report
