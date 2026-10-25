# Crivo-Prumo-17M

[tardellirs/crivo-prumo-17m](https://huggingface.co/tardellirs/crivo-prumo-17m/tree/cdb8caa9c2eec184795d685cb79b9be7b3bd2c3b) is a typed-decision model with two parts: an encoder
that scores every `choice` / `noul` question in one pass, and exact rule capabilities (code execution, chess search,
Boolean logic, puzzles, dates, arithmetic, …) that claim the questions they can solve exactly. A fixed table
(`route.json`), chosen on development data and never on the suite, decides per claiming capability whether the rule's
answer, the encoder's, or their normalised product is kept. Unclaimed questions keep the encoder's answer.

Components: the encoder weights are those of [Crivo-17M](https://huggingface.co/tardellirs/crivo-17m), unchanged; the rule capabilities are
[Prumo](https://huggingface.co/tardellirs/prumo) v1's code and 527 constants, unchanged. Nothing was retrained.

| | Crivo-Prumo-17M |
|---|---:|
| Decision Index 0.3, public index | **29.79** (raw 47.08) |
| Requests | 140,178 `ok`, 0 `unsupported`, 0 errors (`"complete": true`) |
| Parameters | 16,865,025 learned (Crivo-17M) + 527 hand-set constants (Prumo) |
| Results | [runs/crivo-prumo-17m](https://huggingface.co/datasets/tardellirs/crivo-prumo-results/tree/1acb239df9a6913f6098767715062c7f4cf5e88a/runs/crivo-prumo-17m) (compact; `provenance.json` explains the run) |

```sh
hf download tardellirs/crivo-prumo-17m --revision cdb8caa9c2eec184795d685cb79b9be7b3bd2c3b --local-dir crivo-prumo-17m
cd crivo-prumo-17m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-prumo-17m
```

**How the run was produced (please read).** To save GPU time, this package's results were composed offline from Crivo-17M's official complete run (A100 40 GB, already on the board) and Prumo v1 run on the same requests, combined exactly as the engine does. On Crivo-Prumo-68M, where the package itself ran the whole suite, this composition agrees with the package's run on 99.998% of 323,259 questions (38.78 vs 38.75). Running the package with the command above reproduces it up to GPU numerics.
