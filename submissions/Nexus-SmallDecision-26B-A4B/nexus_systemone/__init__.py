"""Nexus SmallDecision System One runtime: renders requests into the prompts the
model was trained on, reads the answer tokens from a running vLLM server, and
serves ``POST /v1/systemone``. Run it with ``python -m nexus_systemone serve``."""

RUNTIME_VERSION = "1.0.0"
