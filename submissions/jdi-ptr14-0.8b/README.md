# jdi-ptr14-0.8b

[jdi-ptr14-0.8b](https://huggingface.co/AnamK/jdi-ptr14-0.8b) is a 0.8B typed decision model built on Qwen3.5-0.8B. It runs only the
first 14 of the 24 decoder layers. A small bilinear pointer head then scores every option: it compares the hidden state of the last prompt
token with the hidden state at the end of each option line, both taken at layer 14. Probabilities are the softmax over all options. It does
not generate text and has no LM head, so it handles any number of options. CC BY-NC 4.0 (the training data includes non-commercial sets).

| | jdi-ptr14-0.8b |
|---|---:|
| Decision Index 0.3, public index | **35.79** |
| raw | 51.74 |
| Knowledge & Reasoning | 13.5 |
| Language Understanding | 40.3 |
| Retrieval & Classification | 52.6 |
| Tools & Automation | 53.0 |
| Arts & Human Taste | 16.8 |
| Requests | 140,620 `ok` (140,178 scoreable), 0 unsupported, 0 errors |
| Request time, 1× RTX 3090 (shared), one request at a time (scoring run) | median 31 · mean 52 · p80 45 ms |
| Model revision | [`5bb55cb177d4f0d344f891af146775be30dbb3c5`](https://huggingface.co/AnamK/jdi-ptr14-0.8b/tree/5bb55cb177d4f0d344f891af146775be30dbb3c5) |
| Results | [runs/ptr14nosyn-soup3](https://huggingface.co/datasets/AnamK/decision-index-results/tree/20844ea936c62259276fba4502abac7bcdf3ebd3/runs/ptr14nosyn-soup3) (compact: no suite text) |

## Reproduce

```sh
huggingface-cli download AnamK/jdi-ptr14-0.8b --local-dir jdi-ptr14-0.8b
export PYTHONPATH=$PWD/jdi-ptr14-0.8b/engine:$PYTHONPATH
python -m decision_index pipeline --edition 0.3 --suite-dir suite-0.3-full \
    --engine jdi_ptr14.engine:Ptr14Engine \
    --option model=AnamK/jdi-ptr14-0.8b --option mem_gb=4.5 --option batch_tokens=8192 \
    --out runs/ptr14nosyn-soup3
```

Exact settings: bf16 weights, sdpa attention, layers 0-13 only, `ptr_head_L14.pt`, fixed prompt (chat template with thinking off,
numbered options, assistant prefill `Best answer: [`), no per-benchmark prompts, no option filtering, no truncation. For requests with
two or more questions the shared prefix (state + question header) runs once with a cache and each question's remainder runs in batches of
at most 8,192 tokens; a single question that does not fit in `mem_gb` runs alone (11 ToolRet rows took this path; answers unchanged).
Software: transformers 5.8.0, torch 2.9.1+cu128, no flash-linear-attention (torch fallback). The engine needs only torch and transformers.

## Training data

LoRA (r16) on layers 0-13 plus the pointer head, 1 epoch, 3-seed weight soup. 178,783 rows: general tasksource and typed-decision tasks,
plus the **train splits** of public datasets that several suite benchmarks are drawn from. Suite rows were removed by exact-segment and
13-gram matching; details and remaining gaps are in the model card. No generator-made data in benchmark formats is included.
Because of these train splits, the public index is partly in-distribution; on our own held-out new-domain tasks the train splits gave no
measurable gain.

## Results file

`results.jsonl.gz` is the kit's compact form (`payload` and `raw_output` dropped). Every other field is as the runner wrote it, and
`python -m decision_index score --edition 0.3` on it reproduces `index.json` byte for byte. In `environment.json` the local checkpoint
path was replaced by the Hub id (same weights).
