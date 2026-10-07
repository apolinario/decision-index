# CoCo-Decision-4B-Ko

[corners-ai/CoCo-Decision-4B-Ko](https://huggingface.co/corners-ai/CoCo-Decision-4B-Ko) is a LoRA adapter on `Qwen/Qwen3.5-4B` (Apache-2.0). It is read out in one forward pass per question from the option-label logits, with no generation. It was trained for Korean and English decisions.

| | |
|---|---|
| Decision Index 0.2.1 | **41.03** (raw 55.83, breadth skill 39.93) |
| Area skill | knowledge 30.2 · language 41.3 · retrieval 46.0 · tools 58.7 · arts 26.0 |
| Requests | 150,759 run (150,317 scoreable), all `ok`; none unsupported, no errors |
| Results | [corners-ai/CoCo-Decision-4B-Ko-decision-index](https://huggingface.co/datasets/corners-ai/CoCo-Decision-4B-Ko-decision-index/tree/485635190dd8ab36a0cba4b7f4e8203cca07cf6a/runs/CoCo-Decision-4B-Ko) (`--compact`, no suite text) |
| Latency on the run hardware | median 29.8 ms, p95 329 ms, mean 97.2 ms (RTX 5090, one sequential client) |

## Running it

The server is `omj serve` from [iamupd/oh-my-jev](https://github.com/iamupd/oh-my-jev) at commit `f0b8c78`. The run used the semif backend in bf16, with the adapter at tag `v1.0.0` merged into the base weights at load and no calibration file.

```sh
git clone https://github.com/iamupd/oh-my-jev && cd oh-my-jev && git checkout f0b8c78
uv sync --extra semif
hf download corners-ai/CoCo-Decision-4B-Ko --revision v1.0.0 --local-dir coco-v1.0.0
cat > di.toml <<'EOF'
[backend]
name = "semif"
model = "Qwen/Qwen3.5-4B"
revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
quant = "bf16"
adapter = "coco-v1.0.0"
max_state_tokens = 32768
EOF
OMJ_CONFIG=di.toml uv run omj serve          # http://127.0.0.1:8799/v1/systemone
python -m decision_index run --engine http \
    --option base_url=http://127.0.0.1:8799 --option model=CoCo-Decision-4B-Ko ...
```

That is the fastest path the code supports: one sequential client, no order ensembling, no prefix cache.

## Declared limits

- The server accepts prompts up to 32,768 tokens (`max_state_tokens`), and up to 237 options per question, the option labels that are single tokens in the Qwen3.5 tokenizer.
- Larger requests are refused with HTTP 422 naming the limit.
- In this suite the longest prompt is 22,656 tokens and no question has more than 237 options, so nothing was refused or truncated.

## Suite

The suite was rebuilt with `suite rebuild` and imported with verification:
- base rows: uncompressed sha256 `b2b56d6f…`
- added rows: sha256 `7429f3c9…`
- exclusions and subsets: matching

## Training data disclosure

The training mixture included the public train splits of three suite datasets: GSM8K (3,000 problems, reworded into "is this answer correct" and "pick the answer" questions), BANKING77 (1,750 examples) and CLINC150 (`clinc_oos`, plus config, 2,000 examples). No tool-calling data was used.
