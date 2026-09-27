# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Hopper (G) 1.2 (`HopitAI/hopper-g` @ `d60a1d6`) | **40.77** | [full run and scores](https://huggingface.co/datasets/HopitAI/hopper-g-decision-index-results/blob/610f97d39e509fa135086011bd8bd842c91508fd/runs/hopper-g-1.2/scores.json) | `hopper_decisions.Decider` (one pass, option-letter readout, per-kind calibration map; disclosed two-stage shortlist above 26 options) in a kit-format engine; [hopper](https://github.com/hopit-ai/hopper) `g-1.2.0` (`0204f92`); kit `87d4650` (0.2.1 scoring) | NVIDIA A10G (Modal), 9 jobs, bf16 eager | None: 151,034 of 151,034 requests answered; nothing truncated. |
