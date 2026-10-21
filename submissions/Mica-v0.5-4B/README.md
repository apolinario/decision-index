# Mica v0.5 4B

[sky7350/Mica-v0.5-4B](https://huggingface.co/sky7350/Mica-v0.5-4B) is a fine-tune of Qwen3.5-4B for typed decisions. It reads
the state and the questions in one prefill and returns a probability for every option; it generates nothing. The run
used the pre-converted INT8 (W8A8) weights in `int8/` with the CUDA runtime in [akivet/Mica](https://github.com/akivet/Mica).

| | |
|---|---|
| Decision Index 0.3 public index | 57.56 (raw 68.47) |
| Run | complete: 140,178 of 140,178 scoreable requests, all answered, none unsupported, nothing truncated, no errors |
| Request time in `scores.json` | median 23.9 ms, mean 63.5 ms, p95 175 ms on one RTX 6000 Ada, kit defaults; not measured on an RTX PRO 6000 |

Results: [`sky7350/Mica-v0.5-4B-decision-index`](https://huggingface.co/datasets/sky7350/Mica-v0.5-4B-decision-index/tree/152d4e8b147715b4e495d91cc4194570b2527a0a/runs/mica-v0.5-4b)
@ `152d4e8` (`results.jsonl.gz`, `scores.json`, `index.json`, `benchmark-summary.json`, `environment.json`, `status.json`).

## Run

Linux, CUDA 12, Python 3.10 to 3.13, an NVIDIA GPU with compute capability 8.0 or newer:

```sh
git clone https://github.com/akivet/Mica && cd Mica
python3 -m venv .venv && . .venv/bin/activate
pip install -r v0.5/serving/requirements.txt
hf download sky7350/Mica-v0.5-4B --include "int8/*" --local-dir weights
python v0.5/serving/serve.py --backend hf --model weights/int8 --name mica-v0.5-4b --port 8010 --state-dir .mica-state
# wait for the start line ("serving": ..., "precision": "int8"); the first start on a GPU takes 1-2 minutes

git clone https://github.com/apolinario/decision-index && cd decision-index    # 25c6ec7
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8010 \
  --option model=mica-v0.5-4b --out runs/mica-v0.5-4b
```

The runtime checks that the INT8 kernels run on the GPU at start and falls back to bfloat16 if they do not; the start
line reports which precision is in use. The run above used the runtime packaged as Python 3.11 bytecode; the repository
now ships the same runtime as compiled modules for Python 3.10 to 3.13, which gave identical answers on a 500-request
check.
