import json
from importlib import resources

DEFAULT = "0.2"

EDITIONS = {
    "0.1": dict(
        id="0.1",
        name="release-v1",
        label="Decision Index 0.1",
        panel_id="core25-observed-protocol-v1",
        headline="balanced_raw",
        suite_dir="suite",
        dataset="multimodalart/decision-index-suite",
        rows_gz_sha256="750d353a3a83af615c67cfe9752e005bf09e6c28d9c4ba28d3a9f57ba8536cfd",
        rows_sha256="288d37207a9581187bdf83eada1983aa63de6fc50b0108e2badb229547a57f99",
        added_sha256=None,
        exclusions_sha256="331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81",
        acos_subset_sha256=None,
        rows_file_requests=132422,
        requests=132422,
        added_requests=0,
        excluded=442,
        scoreable=131980,
        benchmarks=37,
    ),
    "0.2": dict(
        id="0.2",
        name="release-v2",
        label="Decision Index 0.2",
        panel_id="decision-index-0.2",
        headline="balanced_skill",
        suite_dir="suite-0.2",
        dataset="multimodalart/decision-index-suite-0.2",
        rows_gz_sha256="25aac5e890a54a3172c7a0c184b4cc8b9a43f10b6ee89bbad8da923be423c656",
        rows_sha256="b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5",
        added_sha256="7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89",
        exclusions_sha256="331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81",
        acos_subset_sha256="3b20eea1613ae3e1644339e6f89a4287b307a5750a86239bb8bca6f8238b8e45",
        retrieval_subsets_sha256="9228a9492ea40c499d8bb024417f566d0c0ced62c71ed4c1bf908e87ed50d8df",
        rows_file_requests=124971,
        requests=121057,
        added_requests=30419,
        excluded=442,
        scoreable=120615,
        benchmarks=44,
    ),
}

ROWS_FILE = "selected-rows.jsonl.gz"
ADDED_FILE = "added-rows.jsonl.gz"
EXCLUSIONS_FILE = "excluded-questions.json"
MANIFEST_FILE = "manifest.json"


def get(edition=None):
    key = str(edition or DEFAULT)
    key = {"release-v1": "0.1", "release-v2": "0.2", "v0.1": "0.1", "v0.2": "0.2"}.get(key, key)
    if key not in EDITIONS:
        raise ValueError(f"unknown edition {edition!r}; choose from {sorted(EDITIONS)}")
    return EDITIONS[key]


def files(edition):
    e = get(edition)
    return (ROWS_FILE, EXCLUSIONS_FILE, MANIFEST_FILE) + ((ADDED_FILE,) if e["added_sha256"] else ())


def data(name):
    return json.loads(resources.files("decision_index").joinpath("data", name).read_text())


def acos_subset():
    return data("release-v2/acos-subset.json")


def retrieval_subsets():
    return data("release-v2/retrieval-subsets.json")


def scoring_subset(edition):
    if get(edition)["id"] != "0.2":
        return None
    subset = acos_subset()
    return subset["catalog_id"], frozenset(subset["run_ids"])


def in_edition(edition):
    subset = scoring_subset(edition)
    if subset is None:
        return lambda e: True
    catalog, keep = subset
    return lambda e: e["catalog_id"] != catalog or e["run_id"] in keep
