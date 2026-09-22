SUITE_DATASET = "multimodalart/decision-index-suite"
SUITE_ROWS_FILE = "selected-rows.jsonl.gz"
SUITE_EXCLUSIONS_FILE = "excluded-questions.json"
SUITE_MANIFEST_FILE = "manifest.json"
SUITE_FILES = (SUITE_ROWS_FILE, SUITE_EXCLUSIONS_FILE, SUITE_MANIFEST_FILE)
SUITE_ROWS_GZ_SHA256 = "750d353a3a83af615c67cfe9752e005bf09e6c28d9c4ba28d3a9f57ba8536cfd"
SUITE_ROWS_SHA256 = "288d37207a9581187bdf83eada1983aa63de6fc50b0108e2badb229547a57f99"
SUITE_EXCLUSIONS_SHA256 = "331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81"
SUITE_REQUESTS = 132422
SUITE_CASES = 117764
SUITE_FIELDS = 775202
SUITE_BENCHMARKS = 37
SUITE_EXCLUDED_REQUESTS = 442
SUITE_SCOREABLE_REQUESTS = SUITE_REQUESTS - SUITE_EXCLUDED_REQUESTS
FREEZE_SEED = 20260919
REFERENCE_MODEL = "jev-1.13.0"
RUN_SEED = 20260919
PANEL_ID = "core25-observed-protocol-v1"
INTERACTIVE = {7: "MiniWoB++", 16: "ScienceWorld", 13: "Boxoban", 15: "RTFM", 17: "Hanabi", 19: "Codenames"}
FOLD = {"games": "knowledge"}
HEADLINE = {40: "iSarcasmEval-A-En"}
AREAS = [
    {"id": "knowledge", "label": "Knowledge & Reasoning", "panel": [24, 25, 30, 43, 44]},
    {"id": "language", "label": "Language Understanding", "panel": [11, 40, 41]},
    {"id": "retrieval", "label": "Retrieval & Classification", "panel": [36, 37]},
    {"id": "tools", "label": "Tools & Automation", "panel": [1, 2, 6, 7, 16]},
    {"id": "arts", "label": "Arts & Human Judgment", "panel": [20, 21, 22, 23, 50]},
]
FOLDED_PANEL = {"games": [31, 13, 15, 17, 19]}
TRACK_LABEL = {
    "iSarcasmEval-A-En": "A · English",
    "iSarcasmEval-A-Ar": "A · Arabic",
    "iSarcasmEval-C-En": "C · English pairs",
    "iSarcasmEval-C-Ar": "C · Arabic pairs",
    "iSarcasmEval-B-En": "B · English figurative",
}
HEADLINE_METRIC = "Sarcasm F1 · track A, English"
