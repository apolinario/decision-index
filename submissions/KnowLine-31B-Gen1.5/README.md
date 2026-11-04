# KnowLine-31B-Gen1.5

[PelaAI/KnowLine-31B-Gen1.5](https://huggingface.co/PelaAI/KnowLine-31B-Gen1.5) is [KnowLine-31B-Gen1](https://huggingface.co/PelaAI/KnowLine-31B-Gen1) with new inference code. Every weight, config and tokenizer file is byte-identical to Gen1 (google/gemma-4-31B-it with one LoRA (rank 32) fine-tune merged into bf16 weights, then averaged with the base model tensor by tensor: 0.75 × fine-tuned + 0.25 × gemma-4-31B-it, Apache-2.0; a readout temperature of 1.44 is folded in exactly: the final norm weight is divided by 1.44 and `final_logit_softcapping` goes from 30 to 20.83). Only the front end changed: short requests are still read twice, and longer text-only requests get a question header.

- It reads the state and one question with all its options, and returns the probability of each option's label token at the answer position: one prefill per question, nothing generated.
- **Read twice (as in Gen1):** for a text-only request of at most 3,000 tokens (counted with the model's tokenizer over the whole request read once), every turn after the system turn (state, instruction, question and options) is written a second time, introduced by "Read the same input again before answering:", before the answer is read. In this run, 132,376 of the 140,620 requests (94.1%) were read twice, the same requests as in the Gen1 run.
- **Question header (new):** for a text-only request of more than 3,000 tokens, a header goes before the state: the line "Questions that will be asked about the conversation or state below:", then one line "- <question>" per question of the request, in request order and without the options. The request is then read once; the instruction, the question, the options and `Answer:` are as in Gen1. The header is used only if the longest of the request's prompts with it is at most 32,768 tokens; otherwise the request keeps the Gen1 prompts. In this run, 7,845 of the 8,244 longer requests (5.6% of all requests) got the header, and 399 (0.3%) were over the limit and were sent once, as in Gen1. Requests with images are sent exactly as in Gen1.
- Dense, about 31B parameters (31.3B counting the vision tower). The tokenizer and chat template are Gemma 4's. The vision tower is the base model's, unchanged; the text suite does not use it.

| | |
|---|---|
| Decision Index 0.3 public index | 70.50 (raw 77.41); KnowLine-31B-Gen1, same weights: 70.13 |
| Requests | complete: 140,178 of 140,178 scoreable requests, all answered, none unsupported, nothing truncated, no errors |
| Request time, one request at a time | median 330.3 ms, mean 1,135.1 ms, p80 1,417.7 ms, over 700 rows of the public suite (sampled in the style of the board's latency-v1 method, after 10 warm-up rows) on one NVIDIA H20. Gen1's 340.3 ms median and 910.0 ms mean were measured on a different sample of 300 rows, so they are not directly comparable; not measured on an RTX PRO 6000 |
| Request time in `scores.json` | median 1,317.2 ms, mean 3,565.9 ms. 132,376 of these times come from the Gen1 run (4 client shards in parallel against each of four identical servers, one per NVIDIA H20) and 8,244 from the re-run (1 to 3 client shards in parallel against each of four identical servers, one of which also served other requests for part of the time), so they include queueing and are not a single-request latency; not measured on an RTX PRO 6000 |

## Run

```sh
pip install "sglang==0.5.21" "transformers==5.12.1" requests
git clone https://huggingface.co/PelaAI/KnowLine-31B-Gen1.5 && cd KnowLine-31B-Gen1.5      # @ 8e56aa1
bash serve_knowline.sh PelaAI/KnowLine-31B-Gen1.5 0 8080     # SGLang (bf16) on :9080, /v1/systemone on :8080 (read twice + question header)
# equivalent manual launch:
#   CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path PelaAI/KnowLine-31B-Gen1.5 --served-model-name m \
#     --tp 1 --mem-fraction-static 0.85 --enable-fp32-lm-head --port 9080 &
#   python knowline_server.py --model PelaAI/KnowLine-31B-Gen1.5 --backend sglang --url http://127.0.0.1:9080 \
#     --served-model-name m --temperature 1 --workers 16 --port 8080 --temperatures temperatures.json \
#     --header-when-long 32768 --box-prompt gemma --box-coords yxyx          # read twice is on by default

git clone https://github.com/apolinario/decision-index && cd decision-index    # 62d2f51
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8080 \
    --option model=m --out runs/KnowLine-31B-Gen1.5
python -m decision_index score --edition 0.3 --results results.jsonl.gz --out rescore   # the published file
```

- **Front end:** `knowline_server.py` with `knowline_layout.py` next to it (both in the model repo) is the `/v1/systemone` front end of this run; the re-run used these exact files. It is the Gen1 front end plus the question header (`--header-when-long`): requests that read twice covers, and requests with images, are rendered exactly as by the Gen1 front end, prompt for prompt.
  - It renders the model's chat template with thinking off: the state, then one user turn with the instruction, the question and the options labelled A, B, ...; with read twice, those turns are then written again as described above; with the question header, the list of questions goes before the state.
  - It reads the label-token logprobs right after `Answer:` and returns a softmax over the option labels, temperature 1, with no calibration file; the readout temperature is in the weights. With `--temperatures temperatures.json` (`{"score": 0.6944}`, in the model repo; `serve_knowline.sh` passes it and this run used it), the logits of score (rubric) questions are divided by a further 0.6944, an effective temperature of 1.0 for them; this changes their probabilities, not which option ranks first.
  - Multi-question requests are scored one prefill per question, from 16 threads, after one warm-up of the shared prefix.
  - `--header-when-long 0`, the default of `knowline_server.py` started on its own, gives exactly the Gen1 prompts; `--no-read-twice` reads every request once, without the header. `knowline_engine.py` in the model repo runs the same front end in process (`header_when_long=32768` by default).
- **Credit:** the question header follows FlexDM ([github.com/1000luca/FlexDM](https://github.com/1000luca/FlexDM), MIT), with "conversation or state" in place of its "context", and Aleria Judge ([github.com/ALERIA-ML/aleria-judge](https://github.com/ALERIA-ML/aleria-judge), Apache-2.0) uses the same header structure; the read-twice layout also follows FlexDM; repeating the input goes back to RE2 (Xu et al., 2023, arXiv:2309.06275).
- **Settings:** the weights are served in bf16, as for Gen1. `--mem-fraction-static` only sizes SGLang's KV cache; scores do not depend on it (0.90 in both parts of this run, 0.85 by default). One 96 GB GPU per server (the bf16 weights are about 63 GB).
- **Weights:** the carried rows come from the Gen1 run (2026-10-24), which served the local copy of the KnowLine-31B-Gen1 release files; the re-run (2026-10-28) served the local copy of the KnowLine-31B-Gen1.5 release files before the upload. All files in the model repo @ `8e56aa1` match `SHA256SUMS` there; the weight, config and tokenizer files have the same SHA-256 as in the KnowLine-31B-Gen1 repo.
- **How the run was made:** composed, not fresh, with the kit at `62d2f51`.
  - 132,376 rows were carried over unchanged from the Gen1 release run (#283): the requests read twice, whose prompts are byte-identical under the Gen1.5 front end (checked through the code path and a sample).
  - 8,244 rows, the requests of more than 3,000 tokens and the only ones whose prompts change, were re-run with the Gen1.5 release server (`serve_knowline.sh` of the model repo): split into 12 shards, each run as `decision-index run --edition 0.3 --rows <shard>` in parallel against four identical servers (one per GPU). Nothing failed and nothing was retried.
  - The re-run rows replace the Gen1 rows with the same `run_id`, in the Gen1 row order; the composed file was scored with `score --edition 0.3`, which gives 70.50, and re-scoring the published `results.jsonl.gz` with `62d2f51` gives the same.
  - An 830-row consistency re-run agrees on 99.75%: 830 requests (2,808 questions), sampled by benchmark from both parts and sent again through the Gen1.5 release server, gave the same top answer as the scored results on 2,801 of the 2,808 questions (carried rows 1,283 of 1,289, re-run rows 1,518 of 1,519).
  - `environment.json` in the results dataset describes both parts.
- **Training data and decontamination:** the same weights as KnowLine-31B-Gen1, so its description applies unchanged: the training data of KnowLine-26B-A4B-Gen1 in one pass (906,122 rows). About 33% of the training rows target Decision Index benchmarks in their request format or a reworded version of it (train splits of public datasets and synthetic items written in the same format), plus 0.5% copies of the same items rendered in other formats, and 2.6% injection-invariance items; for BPoMP and cfcolor, training items come from the same source datasets as the benchmarks, with every benchmark item excluded. All training data was checked against every public suite row: near-duplicates, repeated questions or claims and whole states were removed (491 rows when this data was built for KnowLine-26B-A4B-Gen1); rows that only share a source passage (for example a Wikipedia paragraph or a tool description) were kept.
- **Images:** requests with images are sent exactly as in Gen1, so the public vision run of KnowLine-31B-Gen1 (69.30, #283) applies to these weights and this front end unchanged; it was not rerun.
