# Solon 1.0 (31B)

[Solon 1.0](https://huggingface.co/Sciforium/Solon-1.0) is [Sciforium](https://www.sciforium.ai/)'s highly optimized decision model, based on Gemma 4 31B using a modified attention implementation and trained with supervised fine-tuning. It returns one probability for every option of a question about a state. It does not generate free text.

| | Solon 1.0 (31B) |
|---|---:|
| Decision Index 0.3, public index | **75.47** (raw 81.40) |
| Requests | 140,620: 140,620 `ok` (140,178 scoreable), 0 unsupported, 0 errors |
| Request time, scoring run | median 171 · mean 828 · p80 628 · p95 3,254 ms |
| Hardware | AMD MI355X (288 GB) |
| Model | [`Sciforium/Solon-1.0`](https://huggingface.co/sciforium/Solon-1.0) |
| Results | [solon-1.0](https://huggingface.co/datasets/sciforium/solon-1.0-decision-index-run/tree/main/solon) |

## Decision Index 0.3

| Area | Skill | Raw |
|---|---:|---:|
| Knowledge & Reasoning | 66.15 | 72.19 |
| Language Understanding | 77.87 | 84.80 |
| Retrieval & Classification | 76.70 | 81.57 |
| Tools & Automation | 90.03 | 91.32 |
| Arts & Human Taste | 64.33 | 77.98 |
| **Public index** | **75.47** | **81.40** |

## Serving

SGLang 0.5.21 (custom). Serving details are available in `solon_server/` with the model weights on [HuggingFace](https://huggingface.co/sciforium/Solon-1.0).

## Declared limits

None. Every request in the public suite was answered, including the 64-question and 151-option requests. Nothing was truncated.

Please refer to the model card: [Sciforium/Solon-1.0](https://huggingface.co/Sciforium/Solon-1.0).
