# Submissions

| Model | Decision Index | Results | Engine / code | Hardware | Declared capacity limits |
|---|---|---|---|---|---|
| Darwin-27B-JEV | 61.17 (kit 0.2.1; 54.63 under 0.2) | [FINAL-Bench/Darwin-27B-JEV-decision-index](https://huggingface.co/datasets/FINAL-Bench/Darwin-27B-JEV-decision-index) · `runs/darwin-27b-jev/scores.json` (complete: true; 0.2.1 scores in `edition-0.2.1/`) | `http` engine, kit `19ad28e` · weights [FINAL-Bench/Darwin-27B-RSI](https://huggingface.co/FINAL-Bench/Darwin-27B-RSI) + AutoJev-27B · code [seawolf2357/darwin-27b-jev](https://github.com/seawolf2357/darwin-27b-jev) · one-GPU reasoner [Darwin-27B-RSI-GGUF](https://huggingface.co/FINAL-Bench/Darwin-27B-RSI-GGUF) | 8× B200, 1× H100 | none; every request answered, nothing truncated |