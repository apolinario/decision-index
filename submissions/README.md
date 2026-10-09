# Submissions

| Model | Decision Index | Results | Engine and exact code | Hardware | Declared capacity limits |
|---|---:|---|---|---|---|
| Bespoke-Nimble-27B-v3 ([`bespokelabs/Bespoke-Nimble-27B-v3`](https://huggingface.co/bespokelabs/Bespoke-Nimble-27B-v3); weights not released, served as a hosted `/v1/systemone` API, temperature 1.58) | **65.33** | [full run and scores](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-27b-v3-decision-index/blob/f2a8e72d11d49ad9f1d6bc070c1799ef1ba5da38/runs/bespoke-nimble-27b-v3/scores.json) | the kit's `http` engine (`model=nimble-27b`), kit `9eb2dbe` (edition 0.3), unmodified runner | Bespoke Labs hosted endpoint, NVIDIA B200 per replica, BF16; 96 client processes | Context 32,768 tokens. Nothing was truncated and no options were removed. 12 scored requests (3 ToolRet, 9 BRIGHT) have prompts over 32,768 tokens and are `unsupported`. |
