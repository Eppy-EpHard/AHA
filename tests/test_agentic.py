"""CPU tests of the agentic-eval plumbing: the logging proxy, episode statistics and both harness parsers."""
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from afwsp.eval import bfcl, record, tau2
from afwsp.eval.proxy import Proxy, conv_key


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not req.get("stream"):
            data = json.dumps({"choices": [{"message": {"content": "hi", "reasoning_content": "think",
                                                        "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]},
                                            "finish_reason": "tool_calls"}],
                               "usage": {"completion_tokens": 7}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for obj in ({"choices": [{"delta": {"reasoning_content": "th"}}]}, {"choices": [{"delta": {"reasoning_content": "ink"}}]},
                    {"choices": [{"delta": {"content": "hi"}, "finish_reason": "length"}]},
                    {"choices": [], "usage": {"completion_tokens": 9}}):
            line = b"data: " + json.dumps(obj).encode() + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
        line = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(line), line))


def post(url, body):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def test_the_proxy_relays_and_logs_plain_and_streamed_calls(tmp_path):
    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    with Proxy(f"http://127.0.0.1:{up.server_address[1]}", tmp_path / "calls.jsonl") as px:
        plain = json.loads(post(px.url, {"model": "m", "messages": msgs, "temperature": 1.0}))
        streamed = post(px.url, {"model": "m", "messages": msgs, "stream": True, "top_k": 20})
    up.shutdown()
    assert plain["choices"][0]["message"]["content"] == "hi" and b"[DONE]" in streamed
    a, b = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    assert a["tool_calls"] == [{"name": "f", "arguments": "{}"}] and a["params"] == {"temperature": 1.0}
    assert b["stream"] and b["reasoning_content"] == "think" and b["content"] == "hi"
    assert b["finish_reason"] == "length" and b["usage"] == {"completion_tokens": 9} and a["conv"] == b["conv"]


def test_conversations_key_on_the_first_user_message():
    greet = [{"role": "system", "content": "policy"}, {"role": "assistant", "content": "Hi! How can I help?"}]
    one = greet + [{"role": "user", "content": "cancel my flight"}]
    two = greet + [{"role": "user", "content": "change my seat"}]
    assert conv_key(one) != conv_key(two)
    assert conv_key(one) == conv_key(one + [{"role": "assistant", "content": "sure"}, {"role": "user", "content": "x"}])


def test_episode_stats_count_conversations_with_a_loop():
    rows = [{"conv": "a", "tokens": 10, "cut": False, "loop_onset": None, "dup_frac": 0.0},
            {"conv": "a", "tokens": 900, "cut": True, "loop_onset": 40, "dup_frac": 0.9},
            {"conv": "b", "tokens": 10, "cut": False, "loop_onset": None, "dup_frac": 0.0}]
    s = record.call_stats(rows)
    assert s["calls_n"] == 3 and s["episodes"] == 2 and s["episodes_with_loop_share"] == 0.5
    assert s["loop_share"] == round(1 / 3, 4) and s["call_errors"] == 0


def test_tau2_counts_an_empty_reply_crash_as_a_failure(tmp_path):
    sims = [{"task_id": "t1", "trial": 0, "termination_reason": "user_stop", "reward_info": {"reward": 1.0},
             "messages": [{"role": "assistant"}, {"role": "user"}]},
            {"task_id": "t2", "trial": 0, "termination_reason": "infrastructure_error", "reward_info": None,
             "info": {"error": "AssistantMessage must have either content or tool_calls. Got AssistantMessage"}}]
    (tmp_path / "results.json").write_text(json.dumps({"simulations": sims}))
    a, b = tau2.conversations(tau2.find_results(tmp_path), "airline")
    assert a["success"] and a["id"] == "airline/t1/0" and a["agent_messages"] == 1
    assert not b["success"] and b["termination"] == "empty_agent_reply"


def test_bfcl_marks_entries_missing_from_the_failure_list_as_correct(tmp_path, monkeypatch):
    res = tmp_path / "bfcl/result" / bfcl.MODEL / "non_live"
    score = tmp_path / "bfcl/score" / bfcl.MODEL / "non_live"
    res.mkdir(parents=True)
    score.mkdir(parents=True)
    (res / "BFCL_v4_simple_python_result.json").write_text(
        "\n".join(json.dumps({"id": f"simple_python_{i}"}) for i in range(3)))
    (score / "BFCL_v4_simple_python_score.json").write_text(
        json.dumps({"accuracy": 2 / 3}) + "\n" + json.dumps({"id": "simple_python_1", "valid": False}))
    per, cats = bfcl.per_entry(tmp_path)
    assert [p["correct"][0] for p in per] == [1, 0, 1] and cats["simple_python"] == {"n": 3, "acc": 2 / 3}
