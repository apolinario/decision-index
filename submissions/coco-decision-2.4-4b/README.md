# CoCo-Decision-2.4-4B 2.4.0

[corners-ai/CoCo-Decision-2.4-4B](https://huggingface.co/corners-ai/CoCo-Decision-2.4-4B) is a decision model built on `Qwen/Qwen3.5-4B` with LoRA stages merged into the published text-only bf16 weights. It is a separate repository from [corners-ai/CoCo-Decision-4B](https://huggingface.co/corners-ai/CoCo-Decision-4B) (versions 1.0.0 through 2.3.0), which stays available unchanged. The later stages were trained on public train splits of Decision Index source datasets; version 2.4.0 additionally trained on same-format decision data drawn from domains disjoint from the Decision Index benchmarks. It is read out in one forward pass per question from a readout head over the option-letter tokens, with bidirectional full attention and no generation.

| | |
|---|---|
| Decision Index 0.3 (public) | **55.19** (raw 66.25, breadth skill 53.38) |
| Area skill | knowledge 35.6 · language 62.6 · retrieval 58.7 · tools 77.3 · arts 39.1 |
| Requests | 140,620 requests (140,178 scoreable, 442 excluded), all scoreable `ok`; none unsupported, no errors |
| Results | [corners-ai/CoCo-Decision-2.4-4B-decision-index](https://huggingface.co/datasets/corners-ai/CoCo-Decision-2.4-4B-decision-index/tree/318daa4c0815607ff76fafd22d4a3419d67e2993/runs/CoCo-Decision-2.4-4B-2.4.0) (`--compact`, no suite text) |

## Running it

The server is `omj serve` from [iamupd/oh-my-jev](https://github.com/iamupd/oh-my-jev) at commit `292dc1b`, semif backend in bf16, no calibration file. The submitted run applied the 2.4.0 LoRA adapter to the public 1.0.0 weights at load time; the merged weights on the Hub gave the same top choice on all 24 inputs we checked, with a maximum readout-logit difference of 0.18, and the full suite was not re-run on them.

```sh
git clone https://github.com/iamupd/oh-my-jev && cd oh-my-jev && git checkout 292dc1b
uv sync --extra semif
cat > di.toml <<'TOML'
[backend]
name = "semif"
model = "corners-ai/CoCo-Decision-2.4-4B"
revision = "58e9b7445760834a3ad9c704985102b0b08418d9"
quant = "bf16"
max_state_tokens = 32768
TOML
OMJ_CONFIG=di.toml uv run omj serve          # http://127.0.0.1:8799/v1/systemone
python -m decision_index pipeline --edition 0.3 --engine http --compact \
    --option base_url=http://127.0.0.1:8799 --option model=CoCo-Decision-2.4-4B --out runs/CoCo-Decision-2.4-4B
```

That is the fastest path the code supports: one sequential client, no order ensembling, no prefix cache.

## Declared limits

- The server accepts prompts up to 32,768 tokens (`max_state_tokens`), and up to 237 options per question, the option labels that are single tokens in the Qwen3.5 tokenizer.
- Larger requests are refused with HTTP 422 naming the limit.
- In this suite nothing was refused or truncated (0 unsupported, no errors over all 140,178 scoreable requests).

## Suite

Edition 0.3, kit `62d2f51`. Row files used by this run: selected rows `d3cd4a0c…`, added rows `b0c5865f…`, GSM8K rows `8cef105d…` (`frozen_corpus_sha256` `d3cd4a0c1128d34af25d719f0c2db9187e1f8f2e0dc38dfacba9312ec92b3907`).

## Training data disclosure

- The second stage trained on the public train splits of the datasets behind 14 suite benchmarks: VAST, iSarcasmEval, When2Call, HellaSwag, WinoGrande, RAGTruth, ContractNLI, New Yorker caption contest, Humicroedit, ANLI, GSM8K, BANKING77, CLINC150 and Amazon ESCI. The first stage also included the GSM8K, BANKING77 and CLINC150 train splits.
- Version 2.4.0 additionally trained on same-format decision data drawn from different datasets/domains than the Decision Index benchmarks (new-domain items in the API-Bank, CLINC, GSM8K, HellaSwag, NLI4CT, VAST and POP909 formats).
- Rows overlapping Decision Index 0.3 evaluation items (13-gram and exact-line matching) were removed before training. No evaluation items were used for training or model selection.
