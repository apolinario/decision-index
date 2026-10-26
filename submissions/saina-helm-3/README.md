# Saina Helm 3

[run-saina/saina-helm-3-0.8b @ `bcc9d791ff361b034da2a2b616c1440f90300505`](https://huggingface.co/run-saina/saina-helm-3-0.8b/tree/bcc9d791ff361b034da2a2b616c1440f90300505), Apache-2.0.
0.8B natively multimodal decision model: reads text, images and audio; returns a probability per option.

## Running it

Engine: [run-saina/saina-decision-index @ `6b267af`](https://github.com/run-saina/saina-decision-index/tree/6b267afb9d4d5da91c7238a8b4721d283c1557c1).

```sh
git clone https://github.com/run-saina/saina-decision-index && cd saina-decision-index
git checkout 6b267afb9d4d5da91c7238a8b4721d283c1557c1
python -m venv .venv && . .venv/bin/activate
pip install -e ".[fast]"
```

Text (public 0.3 suite built and verified in `suite-0.3`):

```sh
python -m decision_index pipeline --engine saina_decision_index:SainaEngine \
  --option model=run-saina/saina-helm-3-0.8b --option 'revision="bcc9d791ff361b034da2a2b616c1440f90300505"' \
  --option device=cuda --option 'repo_commit="6b267afb9d4d5da91c7238a8b4721d283c1557c1"' \
  --option 'conv_kernel="20b2508ad12ae40260291539bf45183000451850"' \
  --out runs/helm-3-text
```

Vision (public vision suite rebuilt and verified in `work/vision`):

```sh
decision-index vision pipeline --dir work/vision \
  --engine saina_decision_index.engine:SainaVisionEngine \
  --option model=run-saina/saina-helm-3-0.8b --option revision=bcc9d791ff361b034da2a2b616c1440f90300505 \
  --option device=cuda --option max_pixels=1638400 --option repo_commit=6b267afb9d4d5da91c7238a8b4721d283c1557c1 \
  --workers 1 --out runs/helm-3-vision
```

Both load the weights from the Hub at the revision above and run one request at a time.

## Serving

The same adapter serves `/v1/systemone` from the `saina` SDK:

```sh
pip install "saina[local]==0.9.3"
saina serve --checkpoint run-saina/saina-helm-3-0.8b --revision bcc9d791ff361b034da2a2b616c1440f90300505
```

The engine above calls that adapter in process (`saina.jev.evaluate`), so its answers are the server's.
