# Crivo-Prumo-68M

[tardellirs/crivo-prumo-68m](https://huggingface.co/tardellirs/crivo-prumo-68m/tree/be95db2dc4c210e70b3e841655a287f6b7a0116f) is a typed-decision model with two parts: an encoder
that scores every `choice` / `noul` question in one pass, and exact rule capabilities (code execution, chess search,
Boolean logic, puzzles, dates, arithmetic, …) that claim the questions they can solve exactly. A fixed table
(`route.json`), chosen on development data and never on the suite, decides per claiming capability whether the rule's
answer, the encoder's, or their normalised product is kept. Unclaimed questions keep the encoder's answer.

Components: the encoder weights are those of [Crivo-68M](https://huggingface.co/tardellirs/crivo-68m), unchanged; the rule capabilities are
[Prumo](https://huggingface.co/tardellirs/prumo) v1's code and 527 constants, unchanged. Nothing was retrained.

| | Crivo-Prumo-68M |
|---|---:|
| Decision Index 0.3, public index | **38.75** (raw 54.67) |
| Requests | 140,178 `ok`, 0 `unsupported`, 0 errors (`"complete": true`) |
| Parameters | 68,410,881 learned (Crivo-68M) + 527 hand-set constants (Prumo) |
| Results | [runs/crivo-prumo-68m](https://huggingface.co/datasets/tardellirs/crivo-prumo-results/tree/1acb239df9a6913f6098767715062c7f4cf5e88a/runs/crivo-prumo-68m) (compact; `provenance.json` explains the run) |

```sh
hf download tardellirs/crivo-prumo-68m --revision be95db2dc4c210e70b3e841655a287f6b7a0116f --local-dir crivo-prumo-68m
cd crivo-prumo-68m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-prumo-68m
```

The reported run is a complete run of this package with the kit on one A100 40 GB (shared with other jobs, so the request times are upper bounds).
