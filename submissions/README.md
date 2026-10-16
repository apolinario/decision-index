# Submissions

## Valen

Complete public text-suite results for edition 0.3, submitted for maintainer review.

| Model | Model revision | Public index | Results | Inference code |
| --- | --- | ---: | --- | --- |
| Valen-0.8B | [c2c481fc](https://huggingface.co/Valen-Team/Valen-0.8B/tree/c2c481fcb319bd9ae834abc7c6373271710cf24e) | 20.71 | [scores.json](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/resolve/257d351884cc1c90bb1681199863e72a8dbc289c/runs/Valen-0.8B/scores.json) | [Valen 7b432422](https://github.com/Liuziyu77/Valen/tree/7b432422c18428f519ba1970f716df8a5feb5f89/evaluation/decision_index) |
| Valen-2B | [fb9205c2](https://huggingface.co/Valen-Team/Valen-2B/tree/fb9205c2541abc97ce276670e66f79b78d9068fc) | 27.20 | [scores.json](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/resolve/257d351884cc1c90bb1681199863e72a8dbc289c/runs/Valen-2B/scores.json) | [Valen 7b432422](https://github.com/Liuziyu77/Valen/tree/7b432422c18428f519ba1970f716df8a5feb5f89/evaluation/decision_index) |
| Valen-4B | [22366059](https://huggingface.co/Valen-Team/Valen-4B/tree/22366059194719ad8c4270567569e1d559491fa3) | 38.54 | [scores.json](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/resolve/257d351884cc1c90bb1681199863e72a8dbc289c/runs/Valen-4B/scores.json) | [Valen 7b432422](https://github.com/Liuziyu77/Valen/tree/7b432422c18428f519ba1970f716df8a5feb5f89/evaluation/decision_index) |

Settings: native decision head, shared-state inference, SDPA, temperature 1.0 and saved context limit 32,768. Each model completed all 140,178 requests with 140,177 successful predictions, one Unsupported request and no remaining runtime errors. The same BRIGHT request exceeds the context window; nothing was truncated or option-filtered.

Hardware: eight independent NVIDIA H200 single-GPU workers per model, serial requests within each worker. 0.8B/2B retain FP32 backbone parameters and use BF16 autocast; 4B retains the merged release's BF16 backbone parameters and uses BF16 autocast. Mixers remain FP32. Request timing excludes loading and warmup. Reference-hardware latency and Full scores await maintainer evaluation.

The [pinned results dataset](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/tree/257d351884cc1c90bb1681199863e72a8dbc289c) contains predictions, upstream scoring outputs, exact model/file hashes, sanitized environments, the original engine source and untouched union logs. The [reproduction guide](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/blob/257d351884cc1c90bb1681199863e72a8dbc289c/README.md#reproduce) gives the exact inference commands.
