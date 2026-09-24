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


def detect_edition(directory):
    from decision_index import editions

    directory = Path(directory)
    manifest = directory / editions.MANIFEST_FILE
    if manifest.exists():
        name = json.loads(manifest.read_text()).get("edition")
        if name:
            return editions.get(name)["id"]
    return "0.2" if (directory / editions.ADDED_FILE).exists() else "0.1"


class Suite:
    def __init__(self, directory, edition=None):
        from decision_index import editions

        self.directory = Path(directory)
        self.edition = editions.get(edition or detect_edition(self.directory))
        self.rows_path = self.directory / editions.ROWS_FILE
        self.added_path = self.directory / editions.ADDED_FILE if self.edition["added_sha256"] else None
        self.exclusions_path = self.directory / editions.EXCLUSIONS_FILE
        self.manifest_path = self.directory / editions.MANIFEST_FILE
        self.in_edition = editions.in_edition(self.edition["id"])
        missing = [p for p in (self.rows_path, self.added_path) if p is not None and not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing {', '.join(map(str, missing))}; build it with `decision-index suite rebuild --edition {self.edition['id']}` and `suite import`")
        found = self.manifest().get("edition")
        if found and editions.get(found)["id"] != self.edition["id"]:
            raise ValueError(f"{self.directory} holds edition {found}, not {self.edition['name']}")

    @property
    def row_paths(self):
        return [self.rows_path] + ([self.added_path] if self.added_path else [])

    def manifest(self):
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    def excluded(self):
        if not self.exclusions_path.exists():
            return set()
        return set(json.loads(self.exclusions_path.read_text())["rows"])

    def rows(self, apply_exclusions=False):
        excluded = self.excluded() if apply_exclusions else set()
        for path in self.row_paths:
            for r in read_jsonl(path):
                e = r["_evaluation"]
                if e["run_id"] in excluded or not self.in_edition(e):
                    continue
                yield r

    def verify(self, strict=True):
        e = self.edition
        actual = sha256_file(self.rows_path)
        report = {"edition": e["id"], "rows_file": str(self.rows_path), "sha256": actual, "expected_sha256": e["rows_gz_sha256"], "match": actual == e["rows_gz_sha256"]}
        if not report["match"]:
            inner = sha256_file(self.rows_path, gunzip=True)
            report.update(uncompressed_sha256=inner, uncompressed_match=inner == e["rows_sha256"])
            report["match"] = report["uncompressed_match"]
        if self.added_path:
            added = sha256_file(self.added_path, gunzip=self.added_path.suffix == ".gz")
            report.update(added_file=str(self.added_path), added_sha256=added, added_match=added == e["added_sha256"])
            report["match"] = report["match"] and report["added_match"]
        if self.exclusions_path.exists():
            ex = sha256_file(self.exclusions_path)
            report.update(exclusions_sha256=ex, exclusions_match=ex == e["exclusions_sha256"])
        if strict and not report["match"]:
            raise ValueError(f"frozen suite hash mismatch: {report}")
        return report
