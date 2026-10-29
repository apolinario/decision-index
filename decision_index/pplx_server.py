"""Loopback text/JSON subset of Perplexity's /v1/decisions, using its native checkpoint code.

No model downloads, chat prompting, sampling, or temperature overrides. Install
the checkpoint's pinned dependencies in a separate serving environment first.
"""
import argparse
import hashlib
import json
import math
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


MODEL = "pplx-decider-v1.2"
BODY_LIMIT = 32 * 1024 * 1024


def reject_nonfinite(value):
    raise ValueError("Nonfinite numbers are not valid JSON")


def check_text_json(value):
    if isinstance(value, dict):
        if value.get("type") in ("image_url", "input_image", "image"):
            raise ValueError("This local server supports text/JSON only; image input is not implemented")
        for item in value.values():
            check_text_json(item)
    elif isinstance(value, list):
        for item in value:
            check_text_json(item)


def validate_request(body):
    if not isinstance(body, dict) or set(body) - {"model", "state", "questions"}:
        raise ValueError("Expected model, state and questions; unknown top-level fields are not allowed")
    if body.get("model") != MODEL:
        raise ValueError(f"This checkpoint serves only {MODEL}; aliases and other models are not accepted")
    if not isinstance(body.get("state"), (str, dict, list)):
        raise ValueError("state must be text, an object or an array")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Expected a nonempty map of named questions")
    if len(questions) > 128:
        raise ValueError("questions per request exceeds maximum (128)")
    for name, q in questions.items():
        if not isinstance(name, str) or not name or not isinstance(q, dict):
            raise ValueError("Question names must be nonempty and questions must be objects")
        if set(q) - {"type", "instructions", "criteria"}:
            raise ValueError("Unknown question field")
        kind, criteria = q.get("type"), q.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("Choice criteria must be a nonempty option map")
            if len(criteria) > 255:
                raise ValueError("options per choice exceeds maximum (255)")
        elif kind == "noul":
            if not q.get("instructions") and not criteria:
                raise ValueError("Noul question must have criteria or instructions")
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
                raise ValueError("Noul criteria must contain true/false descriptions")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria or any(not isinstance(v, (str, dict, list)) for v in criteria):
                raise ValueError("Score criteria must be a nonempty list of text/JSON levels")
            if not 2 <= len(criteria) <= 10:
                raise ValueError("a score takes 2 to 10 levels in this native runtime (hosted API also accepts 1)")
        else:
            raise ValueError("Unknown question type; expected choice, noul or score")
    check_text_json(body)


def api_response(body, native):
    """Preserve native probabilities; expose the hosted API's documented summaries."""
    answers = {}
    if set(native["answers"]) != set(body["questions"]):
        raise RuntimeError("Native runtime returned different question IDs")
    for key, q in body["questions"].items():
        a = dict(native["answers"][key])
        if q["type"] == "noul":
            p = a["noul"]
            if not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
                raise RuntimeError("Invalid native probability")
            answers[key] = {"type": "noul", "noul": p}
            continue
        keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
        probs = a["probabilities"]
        if set(probs) != set(keys) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()) or abs(sum(probs.values()) - 1) > 0.01:
            raise RuntimeError("Invalid native probability distribution")
        top = max(range(len(keys)), key=lambda i: probs[keys[i]])
        n = len(keys)
        if q["type"] == "choice":
            a.update(choice=keys[top], confidence=1.0 if n == 1 else (probs[keys[top]] - 1 / n) / (1 - 1 / n))
        else:
            spread = sum(probs[k] * abs(i - top) for i, k in enumerate(keys))
            baseline = sum(abs(i - top) for i in range(n)) / n
            a.update(score=sum(i * probs[k] for i, k in enumerate(keys)),
                     confidence=max(0.0, 1 - spread / baseline),
                     legend=dict(zip(keys, q["criteria"])))
        answers[key] = a
    return {"model": body["model"], "answers": answers,
            "usage": {"input_tokens": native["usage"]["input_tokens"], "output_tokens": len(answers)}}


class DeciderService:
    def __init__(self, decider, request_errors=(), length_errors=(), metadata=None):
        self.decider = decider
        self.request_errors = request_errors
        self.length_errors = length_errors
        self.metadata = metadata or {}

    def decide(self, body):
        validate_request(body)
        try:
            # All questions stay together. The author's runtime owns prompt
            # construction, saved calibration and any context-driven splitting.
            native = self.decider.decide({"state": body["state"], "questions": body["questions"]})
        except self.request_errors as exc:
            raise ValueError(str(exc)) from exc
        except self.length_errors as exc:
            raise ValueError("maximum context length exceeded: " + str(exc)) from exc
        return api_response(body, native)


def create_server(service, port=8011):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # no request contents in logs

        def send_json(self, status, data, request_id=None):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if request_id:
                self.send_header("x-request-id", request_id)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"status": "ok", "model": MODEL, **service.metadata})
            else:
                self.send_json(404, {"error": {"type": "not_found", "message": "Use POST /v1/decisions"}})

        def do_POST(self):
            if self.path != "/v1/decisions":
                self.send_json(404, {"error": {"type": "not_found", "message": "Use /v1/decisions without a trailing slash"}})
                return
            request_id = str(uuid.uuid4())
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Chunked request bodies are not supported")
                length = int(self.headers.get("Content-Length", "0"))
                if length > BODY_LIMIT:
                    self.send_json(413, {"error": {"type": "invalid_request", "message": "request body exceeds maximum (32 MiB)"}}, request_id)
                    return
                if length <= 0:
                    raise ValueError("Expected a JSON body")
                body = json.loads(self.rfile.read(length), parse_constant=reject_nonfinite)
                response = service.decide(body)
            except (ValueError, UnicodeError, RecursionError) as exc:
                self.send_json(400, {"error": {"type": "invalid_request", "message": str(exc)}}, request_id)
            except Exception:
                self.send_json(500, {"error": {"type": "server_error", "message": "Native inference failed; inspect the local runtime"}}, request_id)
            else:
                self.send_json(200, response, request_id)

    # Sequential GPU execution, bound only to loopback. No authentication or TLS:
    # this is not an Internet-facing production server.
    return HTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Existing local checkpoint snapshot; never a Hub ID")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--fast", action="store_true", help="Author's optional compiled fast path; probabilities may differ slightly")
    args = parser.parse_args(argv)
    path = Path(args.checkpoint).resolve()
    source = path / "source" / "src"
    if not path.is_dir() or not (source / "pplx_decider").is_dir():
        parser.error("Download the complete pinned checkpoint with its source package first")
    if not 1 <= args.max_length <= 262144:
        parser.error("max-length must be between 1 and 262144")
    sys.path.insert(0, str(source))
    from pplx_decider import Decider, LengthError, RequestError

    decider = Decider.from_pretrained(path, device=args.device, fast=args.fast, max_length=args.max_length)
    metadata = {"checkpoint": str(path), "max_length": args.max_length, "fast": args.fast,
                "protocol_scope": "text-json", "source_sha256": hashlib.sha256((source / "pplx_decider/model.py").read_bytes()).hexdigest(),
                "decision_config_sha256": hashlib.sha256((path / "decision_config.json").read_bytes()).hexdigest()}
    service = DeciderService(decider, (RequestError,), (LengthError,), metadata)
    server = create_server(service, args.port)
    print(json.dumps({"listening": f"http://127.0.0.1:{server.server_port}", **metadata}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
