# AnyJev L0 · Gemma-4-31B

An inference technique on a stock checkpoint: [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)
@ `842da37`, quantized to FP8 when vLLM loads it, read through the L0 readout of [anyjev](https://pypi.org/project/anyjev/)
0.3.0. No weights are trained or published.

- Each question is rendered with AnyJev's prompt and read as the next-token distribution restricted to the option labels
  (letters A..Z), one prefill per rotation; nothing is generated.
- L0: the options are read in 2 cyclic rotations and the per-option log-probabilities are averaged; a label-free running
  prior corrects label bias once a question has been seen 8 times. No labelled data is used.
- More than 26 options: each option gets a letter and a digit (A0..A9, B0..B9, ...); p(option) = p(letter) · p(digit | letter).
- Yes/no questions are shown with their criteria.

| | |
|---|---|
| Decision Index 0.3 public index | 59.79 (raw 69.37, breadth 59.06) |
| Areas | knowledge 47.8 · language 61.7 · retrieval 63.5 · tools 76.4 · arts 48.0 |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Request time in `scores.json` | median 506 ms, mean 706 ms, measured with several client processes in parallel per server, so it includes queueing |
| Single-process latency | on a stratified 750-row sample, one request at a time, one H100 NVL, FP8: median 123 ms, mean 460 ms, p80 801 ms, p95 1,854 ms; not measured on an RTX PRO 6000 |

## Run

Code, settings and the exact commands: [MorrisZJ/anyjev-l0-gemma](https://github.com/MorrisZJ/anyjev-l0-gemma/tree/50476e9915944d26bba628aca24d9bb46b78ea1e).

```sh
pip install anyjev==0.3.0 vllm==0.29.0           # plus this kit at 9eb2dbe
vllm serve google/gemma-4-31B-it --quantization fp8 --logprobs-mode processed_logprobs --max-logprobs 32 \
  --enable-prefix-caching --max-model-len 131072 --limit-mm-per-prompt '{"image": 0, "video": 0}' --port 8000
python -m decision_index run --engine anyjev_engine:AnyJevEngine --option base_url=http://127.0.0.1:8000 \
  --option model=google/gemma-4-31B-it --option level=L0 --option max_rotations=2 --option big_rotations=2 --out runs/anyjev-l0
```

Results: [morriszjm/decision-index-results](https://huggingface.co/datasets/morriszjm/decision-index-results/tree/7f4ce4f187d2bf212466c9283acbfd2dd8c6a12a/runs/anyjev-l0-gemma-4-31b)
(written with `--compact`). The run was split over group-whole shards and identical servers; every shard used the
settings above.
