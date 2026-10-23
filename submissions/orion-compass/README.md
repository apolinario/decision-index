# Orion Compass

[Orion Compass](https://huggingface.co/SubstrateAILab/orion-compass/tree/46f8d15dd15d25c397a5f102c4f60598a94b95aa) is a decision model by Substrate AI, fine-tuned from google/gemma-4-31B-it and released under Apache-2.0. It returns one probability for every option of a question about a state, read in a single forward pass. It does not generate text.

| | Orion Compass |
|---|---:|
| Decision Index 0.3, public index | **65.90** |
| Requests | 140,620 `ok` (140,178 scoreable), 0 unsupported, 0 errors |
| Request time, one request at a time per GPU (scoring run) | median 102 · mean 429 · p80 406 ms |
| Hardware | 3× NVIDIA RTX PRO 6000 Blackwell Server Edition, one shard each |
| Model | [`SubstrateAILab/orion-compass`](https://huggingface.co/SubstrateAILab/orion-compass/tree/46f8d15dd15d25c397a5f102c4f60598a94b95aa): private, access on request |
| Results | [runs/orion-compass-flexdm](https://huggingface.co/datasets/SubstrateAILab/orion-compass-decision-index/tree/55605b9f29c6de5d79a0c842ac391b9bd219738a/runs/orion-compass-flexdm) (compact: no suite text) |

## Engine

FlexDM's engine, `jevlab.vllm_reader:VllmReader`, from [FlexDM](https://github.com/1000luca/FlexDM/tree/d147cd7e) v1.0 at `d147cd7e` (MIT). It is the engine of the board's FlexDM entry for the base model.
- It runs in-process on vLLM 0.30.0, one request at a time per process.
- The weights are BF16, quantised online to FP8 per block.
- It reads the log-probability of each shown option label after `Answer: (`, and takes a softmax at FlexDM's temperatures.

Options, exactly as run:

```text
--engine jevlab.vllm_reader:VllmReader
--option model=SubstrateAILab/orion-compass
--option layout=header --option read_twice=true --option twice_max_tokens=3000
--option temperature_choice=10.9757 --option temperature_choice_log_slope=-1.7016
--option temperature_noul=10.732 --option noul_bias=0
--compact
```

vLLM settings:
- `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90` (0.30.0);
- BF16 with online `fp8_per_block`;
- `max_model_len` 131,072, `gpu_memory_utilization` 0.70;
- prefix caching, `max_num_batched_tokens` 8,192, `max_num_seqs` 256;
- `TRITON_ATTN`, `language_model_only`, seed 0, raw log-probabilities.

One 96 GB GPU holds the model.

## Declared limits

Up to 131,072 tokens per request and 255 options per question. A longer request is refused, never truncated. No request of the public suite was refused.

## Reproduce

The kit at `9eb2dbe`, FlexDM v1.0 on the Python path, and the options above. Each process reads one shard of the suite:

```bash
python3 -m decision_index run --engine jevlab.vllm_reader:VllmReader <options above> \
  --compact --rows <shard>.jsonl.gz --out runs/orion-compass-flexdm/shard-N
python -m decision_index score --edition 0.3 --results runs/orion-compass-flexdm/results.jsonl \
  --engine orion-compass-flexdm --out runs/orion-compass-flexdm/score
```

The model card has the details: [How to run it for the Decision Index](https://huggingface.co/SubstrateAILab/orion-compass#how-to-run-it-for-the-decision-index).
