# Jet v6.2 evaluation harness

This engine evaluates the published **michaljach/jet** full BF16 checkpoint at revision `fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a`. It does not evaluate an unreleased LoRA experiment. It loads the checkpoint's own pinned format and runtime files, verifies inference asset hashes against its published manifest, and uses its native label-token probabilities and released calibration.

Declared limits: complete prompts up to16,384tokens and2–255 choice options. Oversized requests are `unsupported`; no truncation, candidate filtering, benchmark-specific prompt changes, generation, or access to gold labels. Structured descriptions are rendered mechanically as JSON. Boolean true/false descriptions become yes/no instruction lines because Jet's native Boolean primitive has no criteria field. Every supplied question is answered or the entire request is unsupported.

Local hardware: one NVIDIA RTX4080SUPER16GB, single process/request, Linux/Python3.12, Torch2.11.0+cu128, Transformers5.17.0, flash-linear-attention0.5.2. The published runtime uses SDPA, FLA gated delta attention, and native PyTorch convolution. Maintainer hardware latency must be measured separately under the repository's rules.

Reproduction after installing the repository and the model's runtime dependencies:

```bash
python -m unittest discover -s submissions/jet -p 'test_*.py'
python -m decision_index suite rebuild --work work --edition 0.2.1
python -m decision_index suite import --edition 0.2.1 \
  --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
  --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz
python -m decision_index pipeline --edition 0.2.1 \
  --engine submissions.jet.engine:JetEngine --compact --out runs/jet-v6.2
```

`local_model=/path/to/release` is an optional engine option for reusing local weights. All runtime assets are still checked against the exact Hub release. `--compact` preserves untouched runner responses and metadata while excluding source prompts and raw outputs from public result artifacts. Raw suite sources stay local/private.

A submission line is added only after strict suite hash verification and a full run with `scores.json` reporting `complete: true`. Public upload contains the official runner's compact results and generated score/environment/status files, with an engine commit reference. No subset score is presented as a complete Decision Index result.
