import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_index import editions
from decision_index.suite.download import copy_rows, hub_file
from decision_index.suite.io import sha256_file


def main():
    ap = argparse.ArgumentParser(description="Stage the frozen-suite files of one edition for upload to a private Hub dataset.")
    ap.add_argument("--edition", default=editions.DEFAULT, choices=sorted(editions.EDITIONS))
    ap.add_argument("--rows", required=True, help="path to selected-rows.jsonl(.gz)")
    ap.add_argument("--added-rows", help="path to added-rows.jsonl(.gz) (0.2)")
    ap.add_argument("--exclusions")
    ap.add_argument("--manifest")
    ap.add_argument("--out", default="hub-upload")
    a = ap.parse_args()
    e = editions.get(a.edition)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    copy_rows(a.rows, out / editions.ROWS_FILE)
    digest = sha256_file(out / editions.ROWS_FILE, gunzip=True)
    checks[editions.ROWS_FILE] = {"uncompressed_sha256": digest, "expected": e["rows_sha256"], "ok": digest == e["rows_sha256"]}
    if e["added_sha256"]:
        if not a.added_rows:
            raise SystemExit(f"edition {e['id']} needs --added-rows")
        copy_rows(a.added_rows, out / editions.ADDED_FILE)
        digest = sha256_file(out / editions.ADDED_FILE, gunzip=True)
        checks[editions.ADDED_FILE] = {"uncompressed_sha256": digest, "expected": e["added_sha256"], "ok": digest == e["added_sha256"]}
    for src, name, expected in ((a.exclusions or hub_file(e["id"], editions.EXCLUSIONS_FILE), editions.EXCLUSIONS_FILE, e["exclusions_sha256"]), (a.manifest or hub_file(e["id"], editions.MANIFEST_FILE), editions.MANIFEST_FILE, None)):
        shutil.copyfile(src, out / name)
        digest = sha256_file(out / name)
        checks[name] = {"sha256": digest, "expected": expected, "ok": expected is None or digest == expected}
    print(json.dumps({"edition": e["id"], "staged": str(out), "files": checks, "upload": f"hf upload <you>/<private-dataset> {out} . --repo-type dataset"}, indent=2))
    if not all(v["ok"] for v in checks.values()):
        raise SystemExit("hash mismatch; do not upload")


if __name__ == "__main__":
    main()
