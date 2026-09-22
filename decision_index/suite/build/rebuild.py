import gzip
import json
import shutil
from pathlib import Path

from decision_index import constants as C
from decision_index.suite.build import acquire as acq
from decision_index.suite.build import adapters_creative, adapters_mechanical, adapters_retrieval, adapters_scored, adapters_selection, freeze, home_appliance, normalize_direct, normalize_text
from decision_index.suite.build.layout import DATASETS, Layout
from decision_index.suite.io import read_jsonl

BUILDERS = {}
for module in (normalize_direct, normalize_text, adapters_mechanical, adapters_selection, adapters_scored, adapters_retrieval, adapters_creative, home_appliance):
    BUILDERS.update(module.BUILDERS)


def normalize(layout, numbers, log=print):
    done = set()
    reports = {}
    for n in numbers:
        fn = BUILDERS[n]
        if fn in done:
            continue
        done.add(fn)
        log(json.dumps({"event": "normalize", "builder": fn.__name__, "catalog_id": n, "dataset": DATASETS[n]}))
        reports[fn.__name__] = fn(layout)
    return reports


def compare(rebuilt, reference, log=print):
    ref = {}
    for r in read_jsonl(reference):
        e = r["_evaluation"]
        ref[e["run_id"]] = e["payload_sha256"]
    new = {}
    for r in read_jsonl(rebuilt):
        e = r["_evaluation"]
        new[e["run_id"]] = e["payload_sha256"]
    missing = sorted(set(ref) - set(new))
    extra = sorted(set(new) - set(ref))
    changed = sorted(k for k in set(ref) & set(new) if ref[k] != new[k])
    report = {"reference_rows": len(ref), "rebuilt_rows": len(new), "missing": len(missing), "extra": len(extra), "payload_changed": len(changed), "identical": not (missing or extra or changed), "examples": {"missing": missing[:10], "extra": extra[:10], "payload_changed": changed[:10]}}
    log(json.dumps(report))
    return report


def main(work, only=None, skip_download=False, skip_normalize=False, compare_path=None, log=print, compare=None):
    layout = Layout(work)
    numbers = sorted(only) if only else sorted(BUILDERS)
    if not skip_download:
        acq.acquire(layout, numbers, log=log)
    if not skip_normalize:
        normalize(layout, numbers, log=log)
    out = layout.suite / "release-v1-rebuilt"
    manifest = freeze.freeze(layout, out, log=log)
    rows = out / "selected-rows.jsonl"
    with rows.open("rb") as src, gzip.open(out / C.SUITE_ROWS_FILE, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    result = {"work": str(layout.root), "out": str(out), "requests": manifest["requests"], "selected_cases": manifest["selected_cases"], "fields": manifest["fields"], "selected_rows_sha256": manifest["selected_rows_sha256"], "expected_sha256": C.SUITE_ROWS_SHA256, "byte_identical": manifest["selected_rows_sha256"] == C.SUITE_ROWS_SHA256}
    reference = compare or compare_path
    if reference:
        result["comparison"] = globals()["compare"](rows, Path(reference), log=log)
    return result
