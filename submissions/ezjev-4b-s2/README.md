# ezjev-4b-s2

[everettjf/ezjev-4b-s2](https://huggingface.co/everettjf/ezjev-4b-s2) (revision `3d77eb1`) is `Qwen/Qwen3.5-4B` with two LoRA stages
merged into full weights. Stage 1 is LoRA r=16 over all language-model linear layers, including Gated-DeltaNet, trained on about 85k
rows. Stage 2 is a low-LR LoRA on weak task families, with replay. It uses the llm2jev chat prompt with thinking off and reads the
option-label logits after `Answer:`, at temperature 1.26.

| | |
|---|---|
| Decision Index 0.2.1 | **51.15** |
| Area skill | knowledge 33.5 · language 60.2 · retrieval 56.3 · tools 69.9 · arts 28.8 |
| Requests | 150,759, all `ok`, none unsupported, no errors |
| Median request time | 35.5 ms (RTX PRO 6000, one request at a time) |
| Results | [everettjf/decision-index-results-ezjev-4b-s2](https://huggingface.co/datasets/everettjf/decision-index-results-ezjev-4b-s2/tree/603d2ffc15af8bb47c183ac586a141ef6f4c5263/runs/ezjev-4b-s2) (`--compact`, no suite text) |

## Serving

```sh
pip install "vllm==0.30.0" "llm2jev>=0.6.1"
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve everettjf/ezjev-4b-s2 --revision 3d77eb1565de5b04e77fa3b599b273656fd22a62 --port 8000 \
    --max-logprobs 256 --return-tokens-as-token-ids --max-model-len 131072 --gpu-memory-utilization 0.90 \
    --additional-config '{"gdn_prefill_backend": "triton"}'
llm2jev --model everettjf/ezjev-4b-s2 --backend vllm --url http://127.0.0.1:8000 --port 8080 --temperature 1.26
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8080 --option model=ezjev-4b-s2 ...
```

The FlashInfer-sampler and Triton GDN-prefill settings are there only because the job image had no CUDA toolkit for JIT, and you
can drop them where nvcc is available.

## Training data

All data comes from train or dev splits of public datasets, plus code-generated items whose labels are computed by the
generating program. No suite rows were used. Every training row was also checked against the frozen suite with exact-segment
and 13-gram overlap checks and dropped on any match. The source list and the decontamination code are in
[everettjf/ezjev](https://github.com/everettjf/ezjev/tree/ceeec6f) (`parts/data_v2.py`, `parts/gen.py`, `parts/decontam.py`).
