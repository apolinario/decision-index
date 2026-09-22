import argparse
import hashlib
import json
import shutil
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_index import constants as C


def sha(p):
    return hashlib.file_digest(Path(p).open("rb"), "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Stage the three frozen-suite files for upload to the Hub dataset.")
    ap.add_argument("--rows", required=True, help="path to selected-rows.jsonl.gz")
    ap.add_argument("--exclusions", default=str(Path(__file__).resolve().parents[1] / "hub/excluded-questions.json"))
    ap.add_argument("--manifest", default=str(Path(__file__).resolve().parents[1] / "hub/manifest.json"))
    ap.add_argument("--out", default="hub-upload")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    for src, name, expected in ((a.rows, C.SUITE_ROWS_FILE, C.SUITE_ROWS_GZ_SHA256), (a.exclusions, C.SUITE_EXCLUSIONS_FILE, C.SUITE_EXCLUSIONS_SHA256), (a.manifest, C.SUITE_MANIFEST_FILE, None)):
        shutil.copyfile(src, out / name)
        digest = sha(out / name)
        checks[name] = {"sha256": digest, "expected": expected, "ok": expected is None or digest == expected}
    print(json.dumps({"staged": str(out), "files": checks, "upload": f"hf upload {C.SUITE_DATASET} {out} . --repo-type dataset"}, indent=2))
    if not all(v["ok"] for v in checks.values()):
        raise SystemExit("hash mismatch; do not upload")


if __name__ == "__main__":
    main()
