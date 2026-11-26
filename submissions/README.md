# Model submissions

Please evaluate mentat-Spice-4B on the private Text and Vision suites.

| Model | Public Text 0.3 | Public Vision | Results |
|---|---:|---:|---|
| [mentat-Spice-4B](https://huggingface.co/yunqu/mentat-Spice-4B/tree/90d4f76971ebd07c11054ada274767458b615ad5) | 72.83 | 73.09 | [Complete results](https://huggingface.co/datasets/yunqu/mentat-Spice-4B-decision-index-results/tree/2ac22b522b23db84861a1601265ec07ed0334631/runs) |

- Engine: [`spice_di:SpiceEngine`](https://github.com/mentat-asi/mentat-sys1/tree/5e8cbdd58f408cd5ea05e26df49bfb22e59ddea8/decision-index), commit `5e8cbdd58f408cd5ea05e26df49bfb22e59ddea8`; kit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`, edition 0.3.
- Fixed options: `runtime=turbo`, `device=cuda:0`, model revision `90d4f76971ebd07c11054ada274767458b615ad5`. Backbone and installation are pinned in the engine guide.
- Hardware: five disjoint single-A800 text runs concatenated without changing records; one A800 vision run. Both panels complete, zero errors/unsupported; 26 text abstentions counted as wrong.
- Limits: 32,768 processed tokens, 255 choices, four images, 1,638,400 pixels and fixed 2,048 visual tokens per image; state JSON depth 64 and 8 MiB; groups of at most eight questions, all options retained.
- Private Full scores and RTX PRO 6000 latency qualification are requested from the maintainers using the same engine/settings.
