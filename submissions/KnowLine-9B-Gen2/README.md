# KnowLine-9B-Gen2

This adds a complete self-scored Decision Index 0.3 public run for [KnowLine-9B-Gen2](https://huggingface.co/PelaAI/KnowLine-9B-Gen2/tree/8ec8eb609bc10aa6050ae3754aee419d458fae42), served with read twice as in our earlier KnowLine submissions.

- **Model:** Qwen/Qwen3.5-9B, language-model LoRA rank 128 / alpha 256, one epoch in the latest fine-tuning stage, merged into bf16 weights. Readout temperature T=1.63 is folded into the final language-model norm. Front-end temperature is 1; `temperatures.json` divides score-question logits by 0.6135 (approximately 1/1.63), so score questions are read at an effective temperature of 1.0. Choice and yes/no questions retain the folded temperature.
- **Readout:** one prefill per question, option-label probabilities at the answer position, no generated reasoning. Short text-only requests are read twice when their one-pass request length is at most 3,000 tokens; longer requests are read once. The model's tokenizer and chat template are used with thinking off.

| Measurement | Result |
|---|---|
| Decision Index 0.3 public index | **69.00** (raw 76.40) |
| Coverage | 140,620 requests, 140,178 scoreable; all scoreable requests answered, zero errors or unsupported requests; no truncation |
| Completion | `scores.json` reports `complete: true` |

- **Results:** [PelaAI/KnowLine-9B-Gen2-decision-index](https://huggingface.co/datasets/PelaAI/KnowLine-9B-Gen2-decision-index/tree/1b6bf62bd31427936221f374cff35d3d23145cab/runs/KnowLine-9B-Gen2): `results.jsonl.gz`, `scores.json`, `index.json`, `benchmark-summary.json`, `environment.json`.
- **Model/code:** [KnowLine-9B-Gen2 release files](https://huggingface.co/PelaAI/KnowLine-9B-Gen2/tree/8ec8eb609bc10aa6050ae3754aee419d458fae42). The package's `SHA256SUMS` has SHA256 `766b585457237bde9374b36f0903296166a095f53914e0b099493a0d7fa1f684`.
- **Runner/scorer:** official kit `62d2f51de34a2de64906345b6bc3e98e27ff55c7`, edition 0.3, `http` engine; runner SHA256 `f4c9f36c4c557f537f1d8184db7bf00c5a84d9f4f0aa67f8c06198cead25818d`.
- **Hardware/runtime:** two NVIDIA H20 96 GB GPUs, one identical bf16 SGLang 0.5.21 server per GPU; Transformers 5.12.1 and Torch 2.13.0. Sixteen round-robin client shards total, eight per server; front-end workers 16, mem fraction 0.90, fp32 LM head, `mamba-radix-cache-strategy=extra_buffer`. Question headers and referenced-fields layout are disabled.
- **Retries:** None. Every request was answered in the single fresh pass; nothing was retried or selected by correctness.
- **Consistency:** the packaged server differs from the evaluated server in comments only; its executable Python AST is unchanged. The compressed result file decompresses byte-for-byte to the final merged result file and is re-scored with the official kit.
- **Limits:** 64 questions per request, 255 options per question; no truncation or benchmark-specific routing.

## Training disclosure

The language-model component was fine-tuned with LoRA using text decision tasks, including tasks written and labelled by LLM teachers. The labels were filtered. Standard and strict decontamination checks were applied during training-data assembly, removing 63,444 rows. The vision component was not trained. See the model card's Disclosures section.

## Reproduce

```sh
pip install "sglang==0.5.21" "transformers==5.12.1" "requests==2.34.2"
hf download PelaAI/KnowLine-9B-Gen2 --revision 8ec8eb609bc10aa6050ae3754aee419d458fae42 --local-dir KnowLine-9B-Gen2
(cd KnowLine-9B-Gen2 && sha256sum -c SHA256SUMS)
MEM=0.90 bash KnowLine-9B-Gen2/serve_knowline.sh ./KnowLine-9B-Gen2 0 8080
# In another shell, with the public suite already rebuilt:
git clone https://github.com/apolinario/decision-index
cd decision-index
git checkout 62d2f51de34a2de64906345b6bc3e98e27ff55c7
pip install -e .
python -m decision_index pipeline --edition 0.3 --engine http \
  --option base_url=http://127.0.0.1:8080 --option model=m --out runs/KnowLine-9B-Gen2
```

The public index is separate from the board's private Full score, which is evaluated by the maintainers.
