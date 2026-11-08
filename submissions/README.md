# Jevora 9B V1.6

Jevora is a 9B model that chooses between answers and assigns a probability to each option.

## Public Decision Index result

| Metric | Result |
|---|---:|
| Public score, edition 0.3 | 49.83 |
| Total requests | 140,620 |
| Unsupported requests | 321 |
| Errors | 0 |

The run is complete. The original result files are available below and have not been changed.

## Model and results

| Item | Link |
|---|---|
| Model weights | [Hugging Face](https://huggingface.co/kadirbekovvv/jevora-9b-v1/tree/545549b8e8104f1780966ef1c13136abb32712b4) |
| Inference code | [GitHub](https://github.com/doxanocap/jevora-9b-v1/tree/84864f9a8e8a6642a7b7bc1ab57ee0d5f5297cd2) |
| Full run results | [Hugging Face dataset](https://huggingface.co/datasets/kadirbekovvv/jevora-decision-index/tree/bf27829ed031ac12a9050e2e2c6d45c71c6e3f36/runs/jevora-v16) |
| Decision Index runner | [Pinned revision](https://github.com/apolinario/decision-index/tree/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d) |

## Run settings

| Setting | Value |
|---|---|
| Hardware | NVIDIA H100 PCIe 80 GB |
| Precision | BF16 |
| Temperature | 1 |
| Maximum input | 8192 tokens |
| Maximum options | 255 |
| Truncation | None |

Requests exceeding the supported limits are marked as unsupported.

## Reproduction

The published score was produced with `JevoraResearchCheckpointEngine`. The public code provides `JevoraPackageEngine` for reproduction.

All package tensors were checked for exact equality with the original checkpoint. The research and public loaders were compared on three requests. A full-suite comparison between the two engines has not been run.
