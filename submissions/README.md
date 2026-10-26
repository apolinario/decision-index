# Submissions

| Board | Model | Hub repo | Results | Engine and options | Kit commit | Hardware | Resolution | Declared limits |
|---|---|---|---|---|---|---|---|---|
| **vision** | Trio-Spark DI (9B) | [MachineFi/Trio-Spark-DI](https://huggingface.co/MachineFi/Trio-Spark-DI/tree/35cdf449ab84f8705a46b6e0bce18c5f83862f76) | [runs/trio-spark-di-vision](https://huggingface.co/datasets/MachineFi/Trio-Spark-DI-decision-index/tree/44296c5a238afafbb36bf9539e6f43d5073dc06d/runs/trio-spark-di-vision) | `vllm-vl`, `model=MachineFi/Trio-Spark-DI`, `llm_options={"revision": "35cdf449ab84f8705a46b6e0bce18c5f83862f76"}` | `25c6ec7` | 1× RTX PRO 6000 96 GB, vLLM 0.30.0, bf16 | up to 1.6 MP per image (board cap) | `vllm-vl` defaults: ≤8 images, 32,768-token context; no public row hit either |
