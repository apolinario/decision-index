# Model submissions

| Model | Model repository | Complete public results | Engine / hardware / settings |
|---|---|---|---|
| 𒋻𒇥 (kudhab) | [plantwaterer/kudhab](https://huggingface.co/plantwaterer/kudhab/tree/957137ea436f79d23ce6db6b10ba7707bf33ef58) | [Decision Index 0.3 scores.json](https://huggingface.co/datasets/plantwaterer/kudhab-decision-index/blob/e12413e1e55f19c4733a091447c429d8c40ea70f/runs/kudhab/scores.json) | In-process engine `karpo_decision.serve.engine:DecisionIndexEngine` (karpo-decision @ a6fa38bf, copy in the model repo), `--option max_length=32768`; kit 9eb2dbe2a358004c8782c66e40a83ac07b953fec; NVIDIA H200; [settings](kudhab/README.md). |
