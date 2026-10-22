#!/usr/bin/env python3
"""Decision Index capacity wording in front of a Standard One engine (/v1/systemone), nothing else.

The kit (decision_index.engines.http) counts an HTTP 400/413/422 as `unsupported` only when the body contains one of
its CAPACITY_MARKERS; any other failure is an `error`. The engine already refuses requests it cannot take with 422,
but in its own words (pydantic "Dictionary should have at most 255 items ...", "the model accepts at most N" tokens).
This proxy forwards every request unchanged and, on those three capacity refusals only, appends the matching kit
marker to the error message. Status codes, successful responses and every other error pass through byte for byte.

  python3 di_proxy.py --engine http://127.0.0.1:30251 --port 30261
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

MARK_CONTEXT = "request exceeds the maximum context length"
MARK_OPTIONS = "a choice takes 2 to 255 options per choice"
MARK_FEW = "a choice needs at least two options"
MARK_SCORE = "a score takes 2 to 10 levels"


def marker(err):
    code, field, msg = err.get("code", ""), err.get("field") or "", err.get("message", "")
    if code == "context_length_exceeded":
        return MARK_CONTEXT
    if code == "invalid_request" and field.endswith(".choice.criteria"):
        return MARK_FEW if "at least 2 items" in msg else MARK_OPTIONS if "at most 255 items" in msg else None
    if code == "invalid_request" and field.endswith(".score.criteria") and ("at most 10 items" in msg or "at least 2 items" in msg):
        return MARK_SCORE
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30261)
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    client = httpx.Client(base_url=a.engine, timeout=a.timeout, limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status, body, ctype):
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            r = client.get(self.path)
            self._send(r.status_code, r.content, r.headers.get("content-type", "application/json"))

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            r = client.post(self.path, content=body, headers={"content-type": self.headers.get("content-type", "application/json")})
            out = r.content
            if r.status_code in (400, 413, 422):
                try:
                    j = r.json()
                    m = marker(j.get("error") or {})
                    if m:
                        j["error"]["message"] = f"{j['error']['message']} ({m})"
                        out = json.dumps(j).encode()
                except ValueError:
                    pass
            self._send(r.status_code, out, r.headers.get("content-type", "application/json"))

        def log_message(self, *args):
            pass

    ThreadingHTTPServer.request_queue_size = 256
    srv = ThreadingHTTPServer((a.host, a.port), H)
    srv.daemon_threads = True
    print(json.dumps({"proxy": f"http://{a.host}:{a.port}", "engine": a.engine}), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
