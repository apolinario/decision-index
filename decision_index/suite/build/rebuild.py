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


def main(work, only=None, skip_download=False, skip_normalize=False, compare_path=None, log=print, compare=None, edition="0.1", exclusions=None):
    from decision_index import editions

    if editions.get(edition)["id"] == "0.2":
        return main_v2(work, only, skip_download, skip_normalize, compare or compare_path, log, exclusions)
    return main_v1(work, only, skip_download, skip_normalize, compare or compare_path, log)


def main_v2(work, only, skip_download, skip_normalize, reference, log, exclusions):
    from decision_index.suite.build import adapters_added, release_v2
    from decision_index.suite.download import hub_file

    added = [n for n in (only or []) if n in adapters_added.SPECS]
    base = [n for n in (only or []) if n not in adapters_added.SPECS]
    result = {"edition": "0.2"}
    v1_rows = None
    if not only or base:
        v1 = main_v1(work, base or None, skip_download, skip_normalize, None, log)
        result["release_v1"] = v1
        if v1["byte_identical"]:
            v1_rows = Path(v1["out"]) / "selected-rows.jsonl"
        else:
            log(json.dumps({"event": "release_v2_skipped", "reason": "release-v1 rows are not byte-identical, so the v2 cut would not be either"}))
    layout = Layout(work)
    v2 = release_v2.main(layout, v1_rows, exclusions or hub_file("0.2", C.SUITE_EXCLUSIONS_FILE), added=added if only else None, skip_download=skip_download, log=log)
    result.update(v2)
    if reference and v1_rows is not None:
        result["comparison"] = compare(Path(v2["out"]) / "selected-rows.jsonl", Path(reference), log=log)
    return result


def main_v1(work, only=None, skip_download=False, skip_normalize=False, reference=None, log=print):
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
    if reference:
        result["comparison"] = compare(rows, Path(reference), log=log)
    return result
