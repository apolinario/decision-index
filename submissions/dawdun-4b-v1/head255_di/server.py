"""Serve the Decision Index text and vision request formats."""

import argparse
import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging

from decision_index.engines.base import Unsupported

from .engine import Head255Engine, VISION_MAX_PIXELS


MODEL_NAME = "DawDun-4B-v1"
ROUTES = {"/v1/systemone", "/v1/decisions"}


def decode_image(value):
    if not isinstance(value, str):
        raise ValueError("images must be data URLs or base64 strings")
    if value.startswith("data:"):
        media_type, separator, value = value.partition(",")
        if not separator or not media_type.startswith("data:image/") or not media_type.endswith(";base64"):
            raise ValueError("image data URLs must contain base64-encoded images")
    return base64.b64decode(value, validate=True)


def request_arguments(body):
    if not isinstance(body, dict) or "state" not in body:
        raise ValueError("the request needs a state and a questions object")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions or not all(isinstance(spec, dict) for spec in questions.values()):
        raise ValueError("questions must contain question objects")
    images = body.get("images", [])
    if not isinstance(images, list):
        raise ValueError("images must be an array")
    prompt = body.get("prompt")
    if prompt is not None and not isinstance(prompt, dict):
        raise ValueError("prompt must be an object")
    return body["state"], questions, [decode_image(image) for image in images], prompt


class DecisionHandler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path not in ROUTES:
            self.reply(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0:
                raise ValueError("expected a JSON request body")
            arguments = request_arguments(json.loads(self.rfile.read(size)))
        except ValueError as error:
            self.reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        try:
            result, metadata = self.server.engine(*arguments)
            self.reply(HTTPStatus.OK, {**(metadata or {}), **result, "model": MODEL_NAME})
        except Unsupported as error:
            self.reply(HTTPStatus.UNPROCESSABLE_ENTITY, {
                "error": f"context window, options per choice or image limit: {error}"
            })
        except Exception:
            logging.exception("Inference request failed")
            self.reply(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "inference failed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--context-tokens", type=int, default=262144)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=VISION_MAX_PIXELS)
    args = parser.parse_args()
    engine = Head255Engine(model=args.model, revision=args.revision, device=args.device,
                           cpu_threads=args.cpu_threads, context_tokens=args.context_tokens,
                           min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    try:
        with ThreadingHTTPServer((args.host, args.port), DecisionHandler) as server:
            server.engine = engine
            print(f"{MODEL_NAME} listening at http://{args.host}:{server.server_port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()


if __name__ == "__main__":
    main()
