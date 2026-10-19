# Submissions

Each row is an entrant submitted for the Decision Index board. The run must be
complete (`scores.json` has `"complete": true`) and untouched; the results file
is re-scored on review. See [Submitting a model to the leaderboard](../README.md#submitting-a-model-to-the-leaderboard)
for the full process.

| Model | Hub model | Results | Engine / commit | Hardware | Inference settings |
| --- | --- | --- | --- | --- | --- |
| Kieve-JeV (base, 21.9M) | [scairesearch/kieve-jev](https://huggingface.co/scairesearch/kieve-jev) | [scairesearch/kieve-jev-decision-index](https://huggingface.co/datasets/scairesearch/kieve-jev-decision-index) | `decision_index.engines.kieve:KieveEngine` @ scairesearch/decision-index | NVIDIA L4 24 GB, CUDA 12.4, torch 2.5.1, in-process | `variant=jev`, LoRA rank 16, ModernBERT-base tokenizer, `max_state=160`, `max_question=112`, `max_pair=64`, `max_options=12`, calibrated softmax temperature from the checkpoint; one forward pass per request; no options filtered or truncated |
