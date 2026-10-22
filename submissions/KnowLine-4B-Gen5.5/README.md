# KnowLine-4B-Gen5.5

[PelaAI/KnowLine-4B-Gen5.5](https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5) is [KnowLine-4B-Gen5](https://huggingface.co/PelaAI/KnowLine-4B-Gen5) with new inference code. Every weight, config and tokenizer file is byte-identical to Gen5 (a weight average (0.5 / 0.5) of two LoRA (rank 32) fine-tunes of Qwen3.5-4B, merged into bf16 weights, Apache-2.0, with a readout temperature of 1.50 folded into the final norm weight). Only the front end changed: it reads short requests twice.

- It reads the state and one question with all its options, and returns the probability of each option's label token at the answer position: one prefill per question, nothing generated.
- **Read twice:** for a text-only request of at most 3,000 tokens (counted with the model's tokenizer over the whole request read once), every turn after the system turn (state, instruction, question and options) is written a second time, introduced by "Read the same input again before answering:", before the answer is read. Longer requests and requests with images are sent exactly as in Gen5. In this run, 133,004 of the 140,620 requests (94.6%) were read twice.
- Architecture, config, tokenizer and chat template are the same as KnowLine-4B-Gen1-Gen5. The vision tower and MTP head are the base model's, unchanged; the text suite does not use them.

| | |
|---|---|
| Decision Index 0.3 public index | 66.75 (raw 74.86); KnowLine-4B-Gen5, same weights: 65.67 |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Request time, one request at a time | median 42.1 ms, mean 82.9 ms, p80 99.1 ms, over 300 rows of the public suite on one NVIDIA H20 (Gen5 on the same rows: median 31.6 ms, mean 58.8 ms); not measured on an RTX PRO 6000 |
| Request time in `scores.json` | median 392.2 ms, mean 1,028.7 ms. Measured with 16 client shards in parallel against each of two identical servers (one per NVIDIA H20), so it includes queueing and is not a single-request latency; not measured on an RTX PRO 6000 |

## Run

```sh
pip install "sglang==0.5.21" "transformers==5.12.1" requests
git clone https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5 && cd KnowLine-4B-Gen5.5      # @ 779944d
bash serve_knowline.sh PelaAI/KnowLine-4B-Gen5.5 0 8080     # SGLang (FP8 at load) on :9080, /v1/systemone on :8080
# equivalent manual launch:
#   CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path PelaAI/KnowLine-4B-Gen5.5 --served-model-name m --tp 1 \
#     --quantization fp8 --mem-fraction-static 0.72 --mamba-radix-cache-strategy extra_buffer --enable-fp32-lm-head --port 9080 &
#   python knowline_server.py --model PelaAI/KnowLine-4B-Gen5.5 --backend sglang --url http://127.0.0.1:9080 \
#     --served-model-name m --temperature 1 --workers 16 --port 8080          # read twice is on by default

git clone https://github.com/apolinario/decision-index && cd decision-index    # 62d2f51
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8080 \
    --option model=m --out runs/KnowLine-4B-Gen5.5
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

- **Front end:** `knowline_server.py` with `knowline_layout.py` next to it (both in the model repo) is the `/v1/systemone` front end of this run. It is the front end of KnowLine-35B-A3B-Gen1.5; only the model name in its docstring and help text differs. Apart from read twice, it is the Gen5 front end:
  - It renders the model's chat template with thinking off: the state, then one user turn with the instruction, the question and the options labelled A, B, ...; with read twice, those turns are then written again as described above.
  - It reads the label-token logprobs right after `Answer:` and returns a softmax over the option labels, temperature 1, with no calibration file; the readout temperature is in the weights.
  - Multi-question requests are scored one prefill per question, from 16 threads, after one warm-up of the shared prefix.
  - The run passed `--read-twice --twice-max-tokens 3000` explicitly to a copy of `knowline_server.py` in which read twice was still off by default; the released file has it on by default and otherwise differs from that copy only in its docstring and model name. `knowline_layout.py` is the same file. `--no-read-twice` gives exactly the Gen5 prompts. `knowline_engine.py` in the model repo runs the same front end in process (`--option read_twice=1` by default).
- **Credit:** the read-twice layout follows FlexDM ([github.com/1000luca/FlexDM](https://github.com/1000luca/FlexDM), MIT); repeating the input goes back to RE2 (Xu et al., 2023, arXiv:2309.06275).
- **Settings:** FP8 at load, as for Gen5. `--mem-fraction-static` only sizes SGLang's KV cache; scores do not depend on it.
- **Weights:** the run (2026-10-22) served the local copy of the released weights before the upload. All files in the model repo @ `779944d` other than the model cards match `SHA256SUMS` there; the weight, config and tokenizer files have the same SHA-256 as in the KnowLine-4B-Gen5 repo.
- **How the run was made:** a fresh 0.3 run with the kit at `62d2f51`, no rows carried over. The 140,620 rows that `decision-index run --edition 0.3` sends were split round-robin into 32 shards, each run as `decision-index run --edition 0.3 --rows <shard>` in parallel, 16 against each of two identical servers (one per GPU), then concatenated and scored with `score --edition 0.3`. Re-scoring the published `results.jsonl.gz` with `62d2f51` gives 66.75.
- **Training data and decontamination:** the same weights as KnowLine-4B-Gen5, so its description applies unchanged: it includes the train splits of Decision Index benchmarks in their request formats. In the data rounds added for this model, about 24% of the training rows target Decision Index benchmarks in their request format or a reworded version of it (train splits of public datasets and synthetic items written in the same format), plus 1.4% copies of the same items rendered in other formats; the rest is other public datasets and synthetic decision tasks. The earlier KnowLine-4B rounds are described in the KnowLine-4B-Gen4 submission. For BPoMP and cfcolor, training items come from the same source datasets as the benchmarks, with every benchmark item excluded.
- **Decontamination:** all training data was checked against every public suite row with the same check as our earlier releases: near-duplicates, repeated questions or claims and whole states were removed; rows that only share a source passage (for example a Wikipedia paragraph or a tool description) were kept.
