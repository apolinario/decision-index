# Submissions

| Model | Hub | Results | Engine | Hardware | Settings | Public index |
|---|---|---|---|---|---|---:|
| Standard One 3B SH (v3) | [StandardThinking/StandardOne-3B-SH](https://huggingface.co/StandardThinking/StandardOne-3B-SH) @ `v3` | [runs/standardone-3b-sh-v3](https://huggingface.co/datasets/StandardThinking/StandardOne-SH-decision-index/tree/main/runs/standardone-3b-sh-v3) | Standard One engine + [`di_proxy.py`](StandardOne-3B-SH/di_proxy.py) | up to 4x MI350X | bf16, fp32 head, no system prompt, 131,072 ctx | 51.53 |
| Standard One 8B SH (v4) | [StandardThinking/StandardOne-8B-SH](https://huggingface.co/StandardThinking/StandardOne-8B-SH) @ `v4` | [runs/standardone-8b-sh-v4](https://huggingface.co/datasets/StandardThinking/StandardOne-SH-decision-index/tree/main/runs/standardone-8b-sh-v4) | Standard One engine + [`di_proxy.py`](StandardOne-8B-SH/di_proxy.py) | up to 4x MI350X | bf16, fp32 head, no system prompt, 131,072 ctx | 57.40 |
