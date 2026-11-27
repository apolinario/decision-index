# nexus_systemone

The inference code for Nexus SmallDecision 26B A4B (soup M at LoRA strength
0.75). It turns a System One request into the prompts the model was trained on,
reads the answer tokens from a running vLLM server, and serves
`POST /v1/systemone` and `GET /v1/models`.

It is a small standalone package. It imports nothing from the training
repository and needs only the seven libraries in `requirements.txt`.

## What it covers

| Supported | Not supported (the request fails with HTTP 422) |
|---|---|
| Text state, `noul`, `choice`, `score` and `x_count` questions | Images in state, instructions or criteria |
| Layouts `state_first` and `read_twice` | `x_nexus.image_token_budget` |
| One option view, temperature 1.0, reserved answer tokens | Any other layout |
| `x_nexus.deterministic` and `x_nexus.now` | |
| vLLM over `/v1/completions` | Any engine other than vLLM |

This runtime does not serve GGUF files. It reads answers through vLLM's
`/v1/completions` log-probability output and is built for the safetensors
release.

## 1. Start vLLM

Point vLLM at the model repo's weights. These are the flags the answers depend
on; the release runs used vLLM 0.30.0 on one RTX 5090 (32 GB).

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_BATCH_INVARIANT=1

vllm serve <path-or-repo-of-the-model> \
  --served-model-name nexus-smalldecision \
  --host 127.0.0.1 --port 8001 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 96 \
  --logprobs-mode raw_logits \
  --max-logprobs 1 \
  --enable-prefix-caching
```

- `--logprobs-mode raw_logits` makes vLLM return the model's raw scores for the
  answer tokens the runtime asks for (`logprob_token_ids`). The runtime was
  written and tested against that mode; other modes were not tested.
- `--enable-prefix-caching` lets the question branches of one request share
  the state's prefill.
- `VLLM_BATCH_INVARIANT=1` is what the release measurements ran with. It does
  not make every answer repeatable: on a 704-row sample, running the same
  sweep twice against one server changed the top answer on 5 rows (0.7%).
  Rows whose answers are exactly tied change more often.
- Answers also depend on the state of vLLM's compile caches. Against the
  recorded release run, starts on the release host's existing caches changed the
  top answer on 4 to 6 of those 704 rows. Starts with empty vLLM, Triton and
  inductor cache directories changed it on 48 to 49 rows (about 7%), and a start
  that reused the cache an empty start had built stayed at 49. Three empty-cache
  runs agreed with each other within 4 rows. The cause is not identified, so
  expect a fresh machine to differ from the recorded run on a few percent of
  top answers.
- Do not pass `--quantization`. The experts are NVFP4 and vLLM reads that from
  the checkpoint.
- If vLLM asks for `nvcc` on first start, set `CUDA_HOME` to a CUDA 13.0
  toolkit. The release host did.

## 2. Start this server

Run these from the folder that holds the `nexus_systemone/` directory.

```bash
pip install -r nexus_systemone/requirements.txt

python -m nexus_systemone serve \
  --tokenizer <folder holding the model repo's tokenizer files> \
  --vllm-url http://127.0.0.1:8001 \
  --served-model nexus-smalldecision \
  --release-date <YYYY-MM-DD> \
  --layout read_twice
```

`--tokenizer` is a folder with the model repo's `tokenizer.json` and
`chat_template.jinja`. The runtime checks the template's SHA-256 and the
reserved answer tokens and refuses a folder that does not match the release.
`--served-model` must equal vLLM's `--served-model-name`. The context window is
read from vLLM's model list, not guessed.

Other flags: `--model-id` (the name `/v1/models` reports, default the served
name), `--host`, `--port` (default 127.0.0.1:8000), `--api-key-env NAME` (require
the Bearer key held in that environment variable), `--rate-limit` (requests per
minute per caller), `--max-pending` (requests admitted at once, default 8) and
`--concurrency` (vLLM requests per System One request, default 16).

## 3. Ask a question

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"ticket": "My card was charged twice for the same order."},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {"billing": "Charges, refunds, invoices",
                   "shipping": "Delivery and tracking",
                   "security": "Account takeover and fraud"}
    }
  }
}'
```

The response carries one answer per question, `usage` (input and output token
counts) and `x_nexus.fingerprint`, which changes whenever anything that can
change an answer changes (prompt layout, chat template, served model, context
window, runtime version).

## How the prompt is built

- Every request is one shared prefix (system message and state) plus one short
  branch per question, each ending in the reserved `<ans>` token. The answer is
  the model's score over that question's own answer tokens at that position.
- `read_twice` writes the state and the question twice before the answer slot.
  It applies when the shared prefix plus the longest once-read branch is at most
  3,000 tokens. Longer requests are read once, in the `state_first` layout. The
  twice-read prompt itself can be longer than 3,000 tokens. On the Decision
  Index 0.3 public suite about 97.55% of requests read twice.
- Reserved token strings found in the state are escaped, so a state cannot forge
  an answer slot.
- Relative dates are resolved only when the request gives `x_nexus.now`;
  without it they are left as written.
- A request with `x_nexus.deterministic` runs by itself on the server and its
  question branches go to vLLM one at a time. In determinism runs with the
  training repository's runtime on this vLLM setup, repeated deterministic
  requests gave identical answers; this package has not been through those
  runs. Reproducibility holds for one serving configuration, not across
  different flags or hardware.

## Layout of the package

`render.py` builds the prompts, `vllm_backend.py` reads the answer tokens,
`answers.py` turns scores into the answer types, `engine.py` ties them
together and `server.py` is the HTTP layer. `__main__.py` is the command line.
