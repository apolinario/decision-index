## PR title
submissions: add Kodiak-v0.2-1B accuracy mode, Decision Index 0.2.1 (18.69)

## File: submissions/README.md  (create it if it doesn't exist on main)

# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Kodiak-v0.2-1B accuracy mode (`cortex-agent-llc/kodiak-v0.2-1b-accuracy`: three Kodiak-v0.2-1B encoders on Ettin-encoder-1B, 1.04B each, calibrated outputs averaged; Apache-2.0) | **18.69** | [full run and scores](https://huggingface.co/datasets/cortex-agent-llc/decision-index-results/blob/7cd5f3b1e51e09ed649639fc58ffa9938540b340/runs/kodiak-v0.2-1b-accuracy/scores.json) | native in-process engine `kodiak_s1.decision_index_engine:KodiakEngine` from [grizzlypeaksoftware/kodiak](https://github.com/grizzlypeaksoftware/kodiak) @ `85d2f87`; kit `87d4650` (0.2.1 scoring) | 1 x NVIDIA GB10 (DGX Spark), PyTorch, one process, one request at a time | Position limit 7,999 per segment and 12,288 packed tokens, nothing truncated: 577 requests that do not fit are refused and recorded unsupported (174 scoreable; by dataset: ToolRet 383, BRIGHT 187, HLE 4, ContractNLI 2, POP909-CL 1). |

## PR description

Adds Kodiak-v0.2-1B in accuracy mode (Cortex Agent LLC, Apache-2.0) to Decision Index 0.2.1.

Weights: https://huggingface.co/cortex-agent-llc/kodiak-v0.2-1b-accuracy (three members of https://huggingface.co/cortex-agent-llc/kodiak-v0.2-1b).
Code: https://github.com/grizzlypeaksoftware/kodiak @ `85d2f87`, engine `kodiak_s1.decision_index_engine:KodiakEngine` (in-process; the run's `environment.json` records the policy).

Complete results: https://huggingface.co/datasets/cortex-agent-llc/decision-index-results/blob/7cd5f3b1e51e09ed649639fc58ffa9938540b340/runs/kodiak-v0.2-1b-accuracy/scores.json

Engine policy: all questions of a request are answered in one forward pass per member (structured attention). Every question is sent with Kodiak's "must answer" setting (its "can't tell" output is not used). In accuracy mode the three members' calibrated outputs are averaged. When the state is empty, the question text is also given as the state (one rule for every benchmark). Option texts are the kit's criteria verbatim; no per-benchmark prompts, no option filtering, no retries.

Decision Index: **18.69** (raw 37.96). All 150,317 scoreable requests have results, zero errors. Nothing is truncated: requests beyond the 7,999-position limit or the 12,288 packed-token limit are refused and recorded unsupported (577 total, 174 scoreable; ToolRet 383, BRIGHT 187, HLE 4, ContractNLI 2, POP909-CL 1).

The dataset holds compact runner results (`payload` and `raw_output` stripped after the run so no suite inputs are redistributed; the file was re-scored with the kit's `score` before upload and matches 18.69 / 37.96, complete), plus official scores, index, benchmark summary and environment.

Measured local latency over the run (three members per request): median 142 ms/request, p95 589 ms, mean 236 ms (GB10). This is not the maintainers' RTX PRO 6000 measurement.
