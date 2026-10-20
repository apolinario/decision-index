# Nadia-1 Small

Developed by Bernardo B. Perobeli at Artinos. Public text suite 0.3 submission: **3.61/100**, with 140,178 requests and `complete: true`. The public score is adjusted for chance and coverage; it is neither overall accuracy nor the Full score.

## Pinned artifacts

- [Public model overview](https://huggingface.co/artinos/Nadia-1-Small).
- [Weights and inference code](https://huggingface.co/artinos/Nadia-1-140M/tree/47bf26321b31b84b00507c9fddcecdd3822bab41), revision `47bf26321b31b84b00507c9fddcecdd3822bab41`. The repository is private; maintainer access still needs to be arranged for private evaluation.
- [Results](https://huggingface.co/datasets/artinos/Nadia-1-Decision-Index/tree/cf8b253e14ddb39177776a8b65f72ca78b04e8cd), revision `cf8b253e14ddb39177776a8b65f72ca78b04e8cd`.
- [scores.json](https://huggingface.co/datasets/artinos/Nadia-1-Decision-Index/blob/cf8b253e14ddb39177776a8b65f72ca78b04e8cd/runs/nadia-1-small/scores.json), [predictions](https://huggingface.co/datasets/artinos/Nadia-1-Decision-Index/resolve/cf8b253e14ddb39177776a8b65f72ca78b04e8cd/runs/nadia-1-small/results.jsonl.gz), [environment](https://huggingface.co/datasets/artinos/Nadia-1-Decision-Index/blob/cf8b253e14ddb39177776a8b65f72ca78b04e8cd/runs/nadia-1-small/environment.json) and [hash manifest](https://huggingface.co/datasets/artinos/Nadia-1-Decision-Index/blob/cf8b253e14ddb39177776a8b65f72ca78b04e8cd/manifest.json).
- GitHub code: [betitanx/Nadia-1](https://github.com/betitanx/Nadia-1/tree/06e4319cd8354f3da2d245dfb604dcc430ce7f57), revision `06e4319cd8354f3da2d245dfb604dcc430ce7f57`, also private.
- Kit revision: `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`.
- Engine: `avaliacao.engine_decision_index:NadiaEngine`, included in the model repository.
- Checkpoint SHA-256: `7ea3e7b6a582a09fcf23439575f8150404c8cf71ac087ddd3dbaab06153dcff1`.

## Inference configuration

- NVIDIA L40S; one process; four CPU threads.
- PyTorch `2.11.0+cu128`, Transformers `5.3.0`; FP32, MATH attention and TF32 disabled.
- 2,048 tokens per field, including state, instructions and all options. Up to 255 options per field.
- No truncation, option filtering or option reduction. If any field exceeds the context limit, the entire request is refused.
- The seven additional tasks (MMLU-Pro, BBH, When2Call, New Yorker, RAGTruth, HoVer and PhishNChips) used one field per batch.
- Other tasks used up to eight fields from the same request per batch, subject to an 8,192-token padding budget. Requests were not batched together.
- 130,316 answered requests and 9,862 context-limit refusals, with no inference errors or pending requests.

The model uses a shared head to score explicit options. It does not generate chain-of-thought text, call external tools or make multiple attempts. For Boolean decisions it returns the probability of `true`; for choices it returns a distribution over all supplied options.

## Reproduction

Once model access has been granted, download the pinned revision and install `modelo/requirements.txt` in an environment with the CUDA version listed above. Install the evaluation kit at its pinned revision. From the model repository root, run:

```bash
python -m decision_index pipeline --edition 0.3 --engine avaliacao.engine_decision_index:NadiaEngine --option repositorio=. --option dispositivo=cuda --option threads=4 --option lote=8 --suite-dir /path/to/suite-0.3 --out runs/nadia-1-small
```

This command starts a new run with a batch limit of 8. To reproduce the historical configuration exactly, run the seven additional tasks separately with `--option lote=1`, and use batch size 8 for the remaining tasks. The environment file records the configuration of the four runs. The engine was checked on CPU against the portable evaluator; the submitted quality measurements were obtained on GPU. Official latency must be measured by the maintainers on their hardware.

The option names and Python module paths above are the actual interface identifiers and must not be translated. Model access is not required to rescore existing predictions:

```bash
python -m decision_index score --edition 0.3 --suite-dir /path/to/suite-0.3 --results runs/nadia-1-small/results.jsonl.gz --out rescored
```

## Integrity and limitations

The three suite files and the exclusion list passed the official hash checks. Results combine four runs of the same checkpoint without subsequent training. A total of 109,905 predictions were reused where inputs were identical; the complementary run processed 30,273 requests. It replaced 182 iSarcasm predictions after restoring canonical line endings. Original records remain in the local archive.

Predictions and the three official scoring files are unchanged from the scored local consolidation. Documentation and descriptive metadata were translated into English. Original identifiers and verbatim runtime messages inside prediction records remain unchanged; the dataset card explains the recorded context-limit refusal message.

Reconstruction on Windows required restoring API-Bank's POSIX file order, source metadata and the complete SGD dialogue history. The frozen SGD file contains future turns; the kit also documents leakage in RouterBench. Both tasks are excluded from the index. Raw benchmark data, questions and answer keys are not redistributed in the predictions dataset. This submission does not provide an automated end-to-end suite reconstruction command.

The result shows broad generalization limitations. On the text-only multiple-choice HLE subset, the model answered 74/501 correctly, below the kit's chance baseline. Earlier textual exposure screening did not cover HLE or the six subsequently added display-only tasks. Pretraining and teacher exposure are not fully known. Private tests have not been run and no official leaderboard placement is claimed.
