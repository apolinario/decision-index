# Darwin-27B-RSI-Decision

[FINAL-Bench/Darwin-27B-RSI-Decision](https://huggingface.co/FINAL-Bench/Darwin-27B-RSI-Decision/tree/b125c60b853c133c622d706f214c3b2db4df8f96) is a full-weight decision engine on [FINAL-Bench/Darwin-27B-RSI](https://huggingface.co/FINAL-Bench/Darwin-27B-RSI), BF16, Apache-2.0 (inference code in `autojev/`, MIT). It reads the state, the question and its options in one forward pass and returns a probability for every option. It generates nothing and has no thinking mode. One model answers every request; there is no hand-off to another model.

| | |
|---|---|
| Decision Index 0.3 public index | **68.77** |
| Areas | knowledge 54.00, language 71.74, retrieval 73.06, tools 85.59, arts 59.99 |
| Requests | 140,178 of 140,178 scoreable answered (140,620 rows sent); 0 unsupported, 0 errors |
| Request time in `scores.json` | median 320.2 ms, mean 1,079.1 ms, p95 4,138.7 ms (8 runner processes against one server on one B200, so requests queue behind each other) |

We have no RTX PRO 6000, so we have not measured single-process latency on it.

## Run

```sh
huggingface-cli download FINAL-Bench/Darwin-27B-RSI-Decision --revision b125c60b853c133c622d706f214c3b2db4df8f96 --local-dir Darwin-27B-RSI-Decision
cd Darwin-27B-RSI-Decision && pip install torch transformers fastapi uvicorn httpx pillow
PORT=8721 python decision_server.py                      # /v1/systemone
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8721 --option model=jev-latest --out runs/darwin-27b-rsi-decision
```

- **Server settings (the defaults of `decision_server.py`, as run):** one forward pass per question, no generated tokens, batch 8, token budget 24,000, max length 32,768, `TAU=0` (the optional hand-off path in the file is off and was not used).
- **Prompt attention:** the prompt is read with bidirectional attention (`prompt_attention: bidirectional` in `decision_config.json`); `autojev/` applies it when the checkpoint loads.
- **Readout:** options are listed with short codes; a readout head maps the final hidden state to one score per option; a softmax with a fitted temperature gives the distribution.

**Declared limits:** none. Inputs over 32,768 tokens would be refused, never cut; none occurred in the suite.

## How the submitted run was made

- The kit at `f8552a1` (runner and scorer), `http` engine, 8 runner processes (`decision_index run`, one per shard) against one `decision_server.py` process on 1x NVIDIA B200, then `decision_index pipeline` to score. The run directory is uploaded untouched.
- The weights in the model repo are the exact files that were served (sha256 checked after upload).

## Training data and decontamination

Training and development splits of public decision benchmarks, plus synthetic decision data written for this model (multi-step reasoning stories, code-output questions, financial-entity sentiment, review aspect and sentiment pairs) and schema-variant dialogues built from the SGD and SGD-X training and development splits (CC-BY-SA-4.0). No test split of any benchmark was used for training. Synthetic rows were checked against the public 0.3 suite for 13-gram overlap. No prompt, setting or checkpoint was chosen per benchmark.

Previous entry from the same team: Darwin-27B-ZTC.
