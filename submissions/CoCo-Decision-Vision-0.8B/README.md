# CoCo-Decision-Vision-0.8B

[corners-ai/CoCo-Decision-Vision-0.8B](https://huggingface.co/corners-ai/CoCo-Decision-Vision-0.8B/tree/36e9b8ff61a9a1ea7f672f7ce4d079798a3bb8dc) is a vision-language decision model: a full fine-tune of [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (revision `2fc06364715b967f1860aea9cf38778875588b17`) read out in one forward pass per question by a 237-way head over the option-label tokens, with bidirectional full attention and no generation (design adapted from [oh-my-jev](https://github.com/iamupd/oh-my-jev), Apache-2.0). It was trained on train splits of public image datasets and internal safety-camera images; no benchmark evaluation split was used.

| | |
|---|---|
| Public vision index (vision 0.3) | **55.30** |
| Benchmark skill | CV-Bench 67.7 · BLINK 48.6 · RealWorldQA 38.4 · CharXiv 55.9 · InfographicVQA 77.6 · Mind2Web 75.9 · Winoground 48.8 · KIE 85.0 · Hateful Memes 34.2 · R-Bench-M 10.7 · MMMU-Pro vision 7.1 |
| Requests | 12,210, all `ok`; none unsupported, no errors; `"complete": true`, `missing_rows` 0 |
| Results | [corners-ai/CoCo-Decision-Vision-0.8B-decision-index](https://huggingface.co/datasets/corners-ai/CoCo-Decision-Vision-0.8B-decision-index/tree/238d3f92d1823c57c7518dd032b2414579c02507/runs/CoCo-Decision-Vision-0.8B-vision-0.3): the run directory exactly as `vision pipeline --compact` wrote it (`results.jsonl`, `vision-scores.json` and an identical `scores.json`, `environment.json`, `status.json`); no suite text |

## Running it

The run downloaded the model repository at revision `36e9b8ff61a9a1ea7f672f7ce4d079798a3bb8dc` from the Hub and served it with its own `q1/server.py`: BF16 backbone, FP32 readout and softmax, one forward pass per question, one request at a time, no calibration file. The current head `b2a2e2e2f7d27a13dff81137f1e6bdf31b6b864f` differs only in its README.

```sh
pip install "torch==2.13.0" torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "transformers==5.17.0" pillow safetensors
pip install "decision-index[vision] @ git+https://github.com/apolinario/decision-index.git@25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d"
decision-index suite rebuild --vision --work work --font "/path/to/Arial Bold.ttf"   # then: decision-index vision verify --dir work/vision
hf download corners-ai/CoCo-Decision-Vision-0.8B --revision 36e9b8ff61a9a1ea7f672f7ce4d079798a3bb8dc --local-dir coco-vision
python coco-vision/q1/server.py --model coco-vision --port 8852      # POST /v1/systemone, GET /health
decision-index vision pipeline --dir work/vision --engine decide     --option base_url=http://127.0.0.1:8852 --option path=/v1/systemone --option protocol=decisions     --option model=CoCo-Decision-Vision-0.8B --workers 1 --compact --out runs/CoCo-Decision-Vision-0.8B-vision-0.3
```

These are the commands of the submitted run. It is the fastest path the code supports: one sequential client, no batching across requests, no prefix cache.

## Declared limits

- Choice and yes/no questions with up to 237 options (the single-token option labels of the Qwen3.5 tokenizer); more are refused with HTTP 422 `too many options per choice`, which the kit records as unsupported.
- No prompt-length limit is set. The longest prompt in this suite is 3,419 tokens and no question has more than 10 options, so nothing was refused or truncated.
- Images at native resolution up to 1,638,400 pixels; `effective_image_budget`: max_image_pixels 1,638,400.

## Suite

Vision 0.3, kit `25c6ec7`, rebuilt with `suite rebuild --vision` and checked with `vision verify` (all sets and benchmarks match). Rows file `vision-rows.jsonl` sha256 `f5f1cfb5cd92ab48d4d14919dff9c56963f4bbbed483bdd45a4f102dba87c133`.

## Timing

HTTP wall time per request in this run: median 113 ms, mean 173 ms, p80 250 ms (one sequential client, one GB10 shared with another service). This is not the single-request RTX PRO 6000 measurement.

## Training data

Train splits only, listed with their licences in the model card; every training image was compared by SHA-256 with every image of the public vision suite, with no match. The model card asks for research use only because some training sets are non-commercial.
