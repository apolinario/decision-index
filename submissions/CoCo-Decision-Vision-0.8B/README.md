# CoCo-Decision-Vision-0.8B

Board: **vision** (Decision Index vision 0.3, public suite).

- Model: [corners-ai/CoCo-Decision-Vision-0.8B](https://huggingface.co/corners-ai/CoCo-Decision-Vision-0.8B/tree/fba7ec35d1ef48c925f13e0173122a045f3cac68), pinned revision `fba7ec35d1ef48c925f13e0173122a045f3cac68` (Apache-2.0 code and weights; the model card asks for research use only because of the licences of some training data). A fine-tune of [vllm-sr/d3-lite](https://huggingface.co/vllm-sr/d3-lite) (revision `03560c26f5199c3c04129a2ccb38960c913635e0`), not an official vLLM Semantic Router release. Runtime code, prompt format and server are d3-lite's, unchanged.
- Results: [`runs/CoCo-Decision-Vision-0.8B-vision-0.3`](https://huggingface.co/corners-ai/CoCo-Decision-Vision-0.8B/tree/fba7ec35d1ef48c925f13e0173122a045f3cac68/runs/CoCo-Decision-Vision-0.8B-vision-0.3) in the model repository at the same revision: `results.jsonl`, `vision-scores.json`, `environment.json`, `verify.json` (`decision-index vision verify` of the suite used: ok), and each part's `environment` and `status` under `parts/`.
- Public vision index (kit scorer): **54.96**, complete, all 12,210 rows answered.
- The run was made in two parts with the same model files, server, engine and settings: the Mind2Web rows (973) were run again after the suite's Mind2Web set was rebuilt with Arial Bold (our first build used Pillow's default font and failed `vision verify` on Mind2Web payloads); the other 11,237 rows are from the first part, whose payloads are identical to the verified suite's.
- Engine and options: the kit's `decide` engine with `--option protocol=decisions --option path=/v1/systemone --option base_url=http://127.0.0.1:8851`, `--workers 4`, against the package's own `d3_server.py` (`python d3_server.py --model <package dir> --port 8851 --host 127.0.0.1`, default settings).
- Kit commit: `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`.
- Hardware and software: 1 x NVIDIA GB10 (DGX Spark, unified memory); torch 2.13.0+cu130, transformers 5.17.0; backbone BF16, readout FP32.
- Resolution: native up to 1,638,400 pixels per image; the kit downscales larger images and the package's own image cap is 1,638,400. `effective_image_budget`: max_image_pixels 1,638,400 (first part), 1,637,748 (Mind2Web part).
- Declared limits: none beyond d3-lite's (choice questions with up to 255 options; no image-count cap; no truncation).
- Latency in our run (HTTP wall time per request, four concurrent workers on a shared GB10, not the board's single-request RTX PRO 6000 setting): median 391 ms, mean 565 ms, 80th percentile 785 ms. The architecture and serving path are d3-lite's, which is already on the board.
- Training data: train splits only (listed with licences in the model card); every training image was compared by SHA-256 with every image of the public vision suite, with no match. No benchmark evaluation split was used for training.
