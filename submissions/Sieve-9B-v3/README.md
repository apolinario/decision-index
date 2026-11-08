# Sieve-9B-v3 evaluation harness

Sieve-9B-v3 is a proprietary decision model by Sthānika AI, a full fine-tune of `Qwen/Qwen3.5-9B`, stored as a bf16
model plus `head.pt`. Its weights are not public:
they are in the private repository `sthanika-ai/Sieve-9B-v3-eval`, to which we have invited the maintainers'
Hugging Face account (`multimodalart`), under an evaluation licence. Customers use the model through the Sieve platform
API.

The engine is `sieve.decision_index_engine:SieveEngine` from the open-source `sieve` package 0.5.1
([github.com/sthanika-ai/Sieve](https://github.com/sthanika-ai/Sieve) at
`223b92a5e5b5b1e1def38e1d53625a7244f6b569`).

## Run

Python 3.12:

```sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r https://raw.githubusercontent.com/sthanika-ai/Sieve/223b92a5e5b5b1e1def38e1d53625a7244f6b569/requirements.txt
pip install ninja packaging wheel setuptools && pip install --no-build-isolation causal-conv1d==1.7.0   # recommended
pip install --no-deps "sieve-decisions @ git+https://github.com/sthanika-ai/Sieve@223b92a5e5b5b1e1def38e1d53625a7244f6b569"
pip install --no-deps "decision-index @ git+https://github.com/apolinario/decision-index@25c6ec7"
hf download sthanika-ai/Sieve-9B-v3-eval --revision d372c05aa5a628b42ebca109de77f003bfaf80b5 --local-dir Sieve-9B-v3
python -m decision_index run --engine sieve.decision_index_engine:SieveEngine --option model=Sieve-9B-v3 --out runs/Sieve-9B-v3
```

The engine reads everything from `head.pt`; no other option is needed. The weights take about 17 GiB, so one RTX PRO
6000 runs every public request (our run used one 40 GB A100 per engine).

**Default and exact mode.** By default, a long shared state (2,048 tokens or more) is computed once for all the
questions of a request; this is the fastest path. Requests below that size are bit-identical to the exact mode. On the
others probabilities move by at most 0.022: on the kit's 400-request sample and the 12 longest requests, 3,504 of 3,505
answers were the same as in the submitted run, and the one change was a 50/50 tie (0.504 / 0.496). With
`SIEVE_LETTERS_PREFIX_MIN=0` the engine runs exactly as the submitted run did and reproduces its answers exactly (337
requests over all 43 benchmarks, 1,606 questions, probability difference 0.0).

## The submitted run

The complete public run (`runs/Sieve-9B-v3/` in
[sthanika-ai/Sieve-9B-v3-eval-results](https://huggingface.co/datasets/sthanika-ai/Sieve-9B-v3-eval-results/tree/089745f78f8733615304148bbbf0841b124fc090)) used the same files (checksums in the model
repository's `SHA256SUMS`; `head.pt` sha256 `67753ed7…`) with our in-process letter engine
(`sieve.letters_engine:LettersEngine`), which computes what the exact mode above computes: bf16, eager, one request at a
time per process, the suite in ten shards on 10× NVIDIA A100-SXM4-40GB. Public index **67.60**; all 140,178 scoreable requests `ok`, nothing unsupported or truncated.

## Inference settings

All settings are fixed in `head.pt`; the engine needs no option besides `model`. The model answers from its own
output-layer logits over the option labels and generates nothing. Each question is read once, with its options as
given. Nothing is truncated, no option is dropped, and no prompt is changed per benchmark. The calibrated temperatures
change no answer.

## Limits

Up to 255 choice options per question; a request beyond that is `Unsupported`, and none was. No token limit: the model
was trained on prompts of up to 16,384 tokens, and every public request (up to about 54,000 tokens) ran on one 40 GB
A100. Running out of GPU memory raises an ordinary error, which a resumed run retries; none occurred.

## Latency

Measured by us with `sieve` 0.5.1 in its default mode on one NVIDIA A100-SXM4-40GB, one request at a time, in-process
(the kit's `model_request_wall_ms`): on the kit's 400-request sample (`decision_index suite sample --n 400`, all 43
benchmarks) median 69 ms, mean 450 ms, 80th percentile 676 ms (95th 2,012 ms); the 12 longest public requests (50,000
to 54,000 tokens, 10 to 29 questions each) took 25 to 34 s each. The submitted run (exact mode, one engine per
A100-SXM4-40GB, all 140,620 requests) had median 59 ms, mean 1,099 ms, 80th percentile 139 ms.
