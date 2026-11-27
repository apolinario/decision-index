"""Serve the System One API over a running vLLM server.

Usage: python -m nexus_systemone serve --tokenizer DIR --vllm-url URL
       --served-model NAME --release-date YYYY-MM-DD [--layout read_twice]
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from datetime import date

import httpx
import uvicorn
from fastapi import FastAPI

from .engine import QUESTION_TYPES, DecisionEngine, EngineConfig
from .fingerprint import fingerprint_of, serving_components
from .render import LAYOUTS
from .server import create_app
from .tokenizer import ServingTokenizer
from .vllm_backend import VllmBackend, default_client


def discover_max_model_len(
    client: httpx.Client, base_url: str, served_model: str
) -> int:
    """The context window vLLM serves ``served_model`` with, from its model list."""
    response = client.get(f"{base_url.rstrip('/')}/v1/models")
    if response.status_code != 200:
        raise RuntimeError(
            f"vLLM model list failed ({response.status_code}): {response.text[:500]}"
        )
    listed = response.json().get("data", [])
    for entry in listed:
        if entry.get("id") == served_model:
            length = entry.get("max_model_len")
            if not isinstance(length, int) or length < 1:
                raise RuntimeError(
                    f"vLLM lists {served_model!r} without a usable max_model_len "
                    f"(got {length!r})"
                )
            return length
    raise RuntimeError(
        f"vLLM does not serve {served_model!r}; it serves "
        f"{[entry.get('id') for entry in listed]}"
    )


def build_app(args: argparse.Namespace, client: httpx.Client | None = None) -> FastAPI:
    client = client or default_client()
    tokenizer = ServingTokenizer(args.tokenizer)
    max_context = discover_max_model_len(client, args.vllm_url, args.served_model)
    backend = VllmBackend(
        tokenizer,
        base_url=args.vllm_url,
        served_model=args.served_model,
        max_context=max_context,
        client=client,
        concurrency=args.concurrency,
    )
    temperatures = dict.fromkeys(QUESTION_TYPES, 1.0)
    config = EngineConfig(
        model_id=args.model_id or args.served_model,
        fingerprint=fingerprint_of(
            serving_components(
                backend_name=backend.name,
                served_model=args.served_model,
                max_context=max_context,
                layout=args.layout,
                temperatures=temperatures,
            )
        ),
        layout=args.layout,
        temperatures=temperatures,
    )
    api_keys = [os.environ[args.api_key_env]] if args.api_key_env else []
    return create_app(
        DecisionEngine(tokenizer, backend, config),
        release_date=args.release_date,
        api_keys=api_keys,
        rate_limit_per_minute=args.rate_limit,
        max_pending=args.max_pending,
    )


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="nexus_systemone", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="serve POST /v1/systemone")
    serve.add_argument(
        "--tokenizer",
        required=True,
        help="folder holding the model repo's tokenizer.json and chat_template.jinja",
    )
    serve.add_argument("--vllm-url", required=True, help="base URL of the vLLM server")
    serve.add_argument(
        "--served-model",
        required=True,
        help="the model name vLLM serves (--served-model-name, listed by /v1/models)",
    )
    serve.add_argument(
        "--model-id",
        help="the model name this API reports (default: the served model name)",
    )
    serve.add_argument(
        "--release-date",
        required=True,
        type=date.fromisoformat,
        help="YYYY-MM-DD, reported by GET /v1/models",
    )
    serve.add_argument("--layout", choices=LAYOUTS, default="read_twice")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--api-key-env",
        help="name of an environment variable holding the Bearer key to require",
    )
    serve.add_argument(
        "--rate-limit", type=_positive, help="requests per minute per caller"
    )
    serve.add_argument("--max-pending", type=_positive, default=8)
    serve.add_argument(
        "--concurrency",
        type=_positive,
        default=16,
        help="concurrent vLLM requests per System One request",
    )
    args = parser.parse_args(argv)
    if args.api_key_env and not os.environ.get(args.api_key_env):
        parser.error(f"--api-key-env {args.api_key_env}: that variable is not set")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
