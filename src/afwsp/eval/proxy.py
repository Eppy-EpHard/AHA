"""A logging proxy for an OpenAI-compatible model server.

The agentic evals send every model call through it, so each call is recorded the same way whatever the harness
(BFCL, tau2-bench). Standard library only.

Per call, calls.jsonl gets one line with:
  - the time and latency;
  - a conversation key (a hash of the messages up to the first user message), so calls group into episodes;
  - the number of messages and tools in the request, and its sampling parameters;
  - the response: content, reasoning_content, tool calls, finish_reason ("length" means it hit the token limit)
    and token usage.
Streamed (SSE) responses are relayed chunk by chunk and recorded once they finish. Other paths (/v1/models,
/health) pass straight through, unlogged.
"""
import hashlib
import http.client
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

PARAMS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "max_tokens",
          "max_completion_tokens", "seed", "stream", "chat_template_kwargs")
HOP = ("host", "content-length", "connection", "accept-encoding", "transfer-encoding", "keep-alive")


class CallLog:
    def __init__(self, path):
        self.path, self.lock, self.n = Path(path), threading.Lock(), 0

    def set_path(self, path):
        with self.lock:
            self.path = Path(path)

    def write(self, rec):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.n += 1


def conv_key(messages) -> str:
    """A hash of the messages up to the first user message: the same for every call of one conversation, and
    different between conversations whose system prompt and opening greeting are shared (tau2)."""
    upto = next((i + 1 for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"),
                min(len(messages), 2))
    return hashlib.sha1(json.dumps(messages[:upto], sort_keys=True, default=str).encode()).hexdigest()[:12]


class Stream:
    """Rebuilds the final message from streamed deltas."""

    def __init__(self):
        self.content, self.reasoning, self.tools, self.finish, self.usage = [], [], {}, None, None

    def add(self, chunk: dict):
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                self.content.append(d["content"])
            r = d.get("reasoning_content") or d.get("reasoning")
            if r:
                self.reasoning.append(r)
            for tc in d.get("tool_calls") or []:
                t = self.tools.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                f = tc.get("function") or {}
                t["name"] += f.get("name") or ""
                t["arguments"] += f.get("arguments") or ""
            if ch.get("finish_reason"):
                self.finish = ch["finish_reason"]

    def result(self) -> dict:
        return {"content": "".join(self.content), "reasoning_content": "".join(self.reasoning),
                "tool_calls": [self.tools[i] for i in sorted(self.tools)], "finish_reason": self.finish,
                "usage": self.usage}


def from_body(body: dict) -> dict:
    ch = (body.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    return {"content": m.get("content") or "", "reasoning_content": m.get("reasoning_content") or m.get("reasoning") or "",
            "tool_calls": [{"name": (t.get("function") or {}).get("name"), "arguments": (t.get("function") or {}).get("arguments")}
                           for t in m.get("tool_calls") or []],
            "finish_reason": ch.get("finish_reason"), "usage": body.get("usage")}


def handler_for(upstream: str, log: CallLog):
    up = urlsplit(upstream)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):     # keep the console for the harness
            pass

        def do_GET(self):
            self.forward("GET")

        def do_POST(self):
            self.forward("POST")

        def forward(self, method):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            req = {}
            if body and self.path.rstrip("/").endswith("completions"):
                try:
                    req = json.loads(body)
                except ValueError:
                    req = {}
            t0 = time.time()
            rec = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "path": self.path}
            if req:
                msgs = req.get("messages") or []
                rec.update({"conv": conv_key(msgs), "n_messages": len(msgs), "n_tools": len(req.get("tools") or []),
                            "params": {k: req[k] for k in PARAMS if k in req}})
            try:
                conn = http.client.HTTPConnection(up.hostname, up.port, timeout=7200)
                conn.request(method, self.path, body=body,
                             headers={k: v for k, v in self.headers.items() if k.lower() not in HOP})
                resp = conn.getresponse()
            except OSError as e:
                self.send_json(502, {"error": {"message": f"proxy could not reach the model server: {e}"}})
                if req:
                    log.write({**rec, "status": 502, "error": str(e), "latency_s": round(time.time() - t0, 3)})
                return
            streaming = "text/event-stream" in (resp.getheader("Content-Type") or "")
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            if streaming:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                acc = Stream()
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
                    self.wfile.flush()
                    s = line.strip()
                    if s.startswith(b"data:") and s[5:].strip() not in (b"[DONE]", b""):
                        try:
                            acc.add(json.loads(s[5:]))
                        except ValueError:
                            pass
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                out = acc.result()
            else:
                data = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                try:
                    out = from_body(json.loads(data)) if data else {}
                except ValueError:
                    out = {"raw": data[:2000].decode(errors="replace")}
            conn.close()
            if req:
                log.write({**rec, "status": resp.status, "stream": streaming, "latency_s": round(time.time() - t0, 3),
                           **out})

        def send_json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return                      # a client hanging up on a kept-alive connection: not an error
        super().handle_error(request, client_address)


class Proxy:
    """Proxy(upstream, log_path).start() serves on a free local port; .url is the OpenAI base URL (".../v1")."""

    def __init__(self, upstream: str, log_path, port: int = 0):
        self.log = CallLog(log_path)
        self.server = _Server(("127.0.0.1", port), handler_for(upstream, self.log))
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
