"""Agentic evaluation: BFCL-V4 and tau2-bench v1.0.1, with every model call logged.

  python -m afwsp.agentic bfcl <checkpoint> <out_dir> --bfcl-python <bfcl venv python> [--categories all_scoring]
  python -m afwsp.agentic tau2 <checkpoint> <out_dir> --tau2 <tau2 executable> [--domains airline,retail,telecom]

vLLM serves the checkpoint with Qwen3.5's tool-call and reasoning parsers. A logging proxy in front of it records
every call (thinking, answer, tool calls, tokens, whether it hit the limit), and each harness runs unmodified in its
own environment. Sampling is the setting Qwen reports Qwen3.5's agent scores under: thinking on, temperature 1.0,
top_p 0.95, top_k 20, presence_penalty 1.5, up to 32,768 tokens per call. docs/AGENTIC.md has the setup.
"""
import argparse
import collections
import csv
import hashlib
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .evals import loop_stats, mean, summarize, write_summary

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768}
LOGGED_PARAMS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "max_tokens",
                 "seed", "stream", "chat_template_kwargs")
HOP_HEADERS = ("host", "content-length", "connection", "accept-encoding", "transfer-encoding", "keep-alive")
BFCL_ENTRY = Path(__file__).with_name("bfcl_entry.py")
BFCL_MODEL = "afwsp-qwen3.5-think-FC"                  # the name bfcl_entry.py registers
EMPTY_REPLY = "must have either content or tool_calls"   # tau2's error for an agent turn with nothing in it


# ------------------------------------------------------------------------------------------------- serving

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def vllm_server(checkpoint, log_path, reasoning_parser=None, max_model_len=131072, timeout=1200):
    """`vllm serve` the checkpoint; yield its base URL once healthy. On exit kill the whole process group: vLLM's
    engine is a child process that outlives its parent and keeps the GPU (docs/REPRODUCE.md)."""
    port = free_port()
    command = [str(Path(sys.executable).parent / "vllm"), "serve", str(checkpoint), "--served-model-name", "afwsp",
               "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16", "--max-model-len",
               str(max_model_len), "--gpu-memory-utilization", "0.85", "--seed", "0",
               "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
    if reasoning_parser:
        command += ["--reasoning-parser", reasoning_parser]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    url = f"http://127.0.0.1:{port}"
    try:
        began = time.time()
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {server.returncode}; see {log_path}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.time() - began > timeout:
                raise TimeoutError(f"vLLM not healthy after {timeout}s; see {log_path}")
            time.sleep(5)
        yield url
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=120)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
        log.close()


@contextmanager
def model_endpoint(checkpoint, out_dir, reasoning_parser, model_url=None):
    """A running server's URL if given, else a fresh vLLM server for the checkpoint."""
    if model_url:
        yield model_url.rstrip("/").removesuffix("/v1")
    else:
        with vllm_server(checkpoint, Path(out_dir) / "server.log", reasoning_parser) as url:
            yield url


# ------------------------------------------------------------------------------------------------- logging proxy

def conversation_key(messages):
    """A hash of the messages up to the first user message: the same for every call of one conversation, and
    different between conversations that share a system prompt and opening greeting (tau2)."""
    end = next((i + 1 for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"),
               min(len(messages), 2))
    return hashlib.sha1(json.dumps(messages[:end], sort_keys=True, default=str).encode()).hexdigest()[:12]


def merge_stream(chunks):
    """The final message, rebuilt from streamed chunks."""
    content, reasoning, tools, finish, usage = [], [], {}, None, None
    for chunk in chunks:
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            for call in delta.get("tool_calls") or []:
                tool = tools.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                tool["name"] += (call.get("function") or {}).get("name") or ""
                tool["arguments"] += (call.get("function") or {}).get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    return {"content": "".join(content), "reasoning_content": "".join(reasoning),
            "tool_calls": [tools[i] for i in sorted(tools)], "finish_reason": finish, "usage": usage}


def read_message(body):
    """The message of a non-streamed response."""
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {"content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or message.get("reasoning") or "",
            "tool_calls": [{"name": (t.get("function") or {}).get("name"),
                            "arguments": (t.get("function") or {}).get("arguments")}
                           for t in message.get("tool_calls") or []],
            "finish_reason": choice.get("finish_reason"), "usage": body.get("usage")}


def start_proxy(upstream, log_path):
    """Relay OpenAI-style requests to upstream on a free local port, and append one line per completion call to
    log_path. Returns the server and its base URL; stop it with server.shutdown()."""
    upstream = urlsplit(upstream)
    lock = threading.Lock()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    class Relay(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.relay("GET")

        def do_POST(self):
            self.relay("POST")

        def relay(self, method):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or None
            request = json.loads(body) if body and self.path.rstrip("/").endswith("completions") else {}
            began = time.time()
            record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if request:
                messages = request.get("messages") or []
                record.update({"conv": conversation_key(messages), "n_messages": len(messages),
                               "n_tools": len(request.get("tools") or []),
                               "params": {k: request[k] for k in LOGGED_PARAMS if k in request}})
            connection = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=7200)
            connection.request(method, self.path, body=body,
                               headers={k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS})
            response = connection.getresponse()
            streamed = "text/event-stream" in (response.getheader("Content-Type") or "")
            self.send_response(response.status)
            for k, v in response.getheaders():
                if k.lower() not in HOP_HEADERS:
                    self.send_header(k, v)
            if streamed:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                chunks = []
                while line := response.readline():
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
                    self.wfile.flush()
                    data = line.strip()[5:].strip() if line.startswith(b"data:") else b""
                    if data and data != b"[DONE]":
                        chunks.append(json.loads(data))
                self.wfile.write(b"0\r\n\r\n")
                message = merge_stream(chunks)
            else:
                data = response.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                message = read_message(json.loads(data)) if data else {}
            connection.close()
            if request:
                record.update({"status": response.status, "stream": streamed,
                               "latency_s": round(time.time() - began, 3), **message})
                with lock, open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    server.daemon_threads = True
    report_error = server.handle_error

    def ignore_hangups(request, address):              # a client closing a kept-alive connection is not an error
        if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            report_error(request, address)

    server.handle_error = ignore_hangups
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


@contextmanager
def proxy(upstream, log_path):
    server, url = start_proxy(upstream, log_path)
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------------------------------------- call statistics

def read_calls(log_path, checkpoint):
    """One row per logged call, with loop stats from re-tokenizing its thinking and answer text."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    rows = []
    for line in open(log_path):
        call = json.loads(line)
        text = (call.get("reasoning_content") or "") + (call.get("content") or "")
        ids = tokenizer(text, add_special_tokens=False).input_ids
        rows.append({"conv": call.get("conv"), "tokens": (call.get("usage") or {}).get("completion_tokens") or len(ids),
                     "cut": call.get("finish_reason") == "length", "failed": call.get("status") != 200,
                     **loop_stats(ids)})
    return rows


def call_stats(rows):
    """summarize() over calls, plus how loops and cut-offs spread over episodes (calls grouped by conversation)."""
    stats = summarize(rows)
    stats["calls"] = stats.pop("n")
    episodes = collections.defaultdict(list)
    for r in rows:
        episodes[r["conv"]].append(r)
    stats.update({"episodes": len(episodes), "calls_per_episode": mean([len(e) for e in episodes.values()]),
                  "episodes_with_loop_share": mean([any(r["loop_onset"] is not None for r in e)
                                                    for e in episodes.values()]),
                  "episodes_with_cut_share": mean([any(r["cut"] for r in e) for e in episodes.values()]),
                  "failed_calls": sum(r["failed"] for r in rows)})
    return stats


def run(command, env=None, cwd=None):
    print("$ " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(c) for c in command], env=env, cwd=cwd, check=True)


# ------------------------------------------------------------------------------------------------- BFCL

def bfcl(checkpoint, out_dir, bfcl_python, categories="all_scoring", threads=16, mode="raw", model_url=None):
    """BFCL-V4: BFCL generates and scores; writes bfcl_v4_per_problem.json, scores.json and summary.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    expand = ("import json, sys\nfrom bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING as M\n"
              "names = []\nfor w in sys.argv[1].split(','):\n    names += [c for c in M.get(w, [w]) if c not in names]\n"
              "print(json.dumps(names))")
    names = json.loads(subprocess.run([bfcl_python, "-c", expand, categories], capture_output=True, text=True,
                                      check=True).stdout.splitlines()[-1])
    left_out = [] if os.environ.get("SERPAPI_API_KEY") else [c for c in names if c.startswith("web_search")]
    names = [c for c in names if c not in left_out]
    if left_out:
        print(f"no SERPAPI_API_KEY: leaving out {left_out}, so the overall is not BFCL V4's", flush=True)
    env = {**os.environ, "BFCL_PROJECT_ROOT": str(out / "bfcl"), "AFWSP_SERVED_NAME": "afwsp",
           "AFWSP_BFCL_MODE": mode, "AFWSP_SAMPLING": json.dumps({**SAMPLING, "seed": 0})}
    began = time.time()
    with model_endpoint(checkpoint, out, "qwen3" if mode == "raw" else None, model_url) as url, \
            proxy(url, out / "calls.jsonl") as proxy_url:
        env["AFWSP_BASE_URL"] = proxy_url
        # no --allow-overwrite: a rerun resumes, skipping entries that already have a result
        run([bfcl_python, BFCL_ENTRY, "generate", "--test-category", ",".join(names), "--num-threads", threads,
             "--result-dir", out / "bfcl/result"], env)
    run([bfcl_python, BFCL_ENTRY, "evaluate", "--test-category", ",".join(names), "--result-dir", out / "bfcl/result",
         "--score-dir", out / "bfcl/score"] + (["--partial-eval"] if left_out else []), env)

    per_entry, per_category = bfcl_results(out)
    table = next(csv.DictReader(open(out / "bfcl/score/data_overall.csv")), {})
    overall = table.get("Overall Acc", "").rstrip("%")
    overall = float(overall) / 100 if overall.replace(".", "", 1).isdigit() else None
    calls = read_calls(out / "calls.jsonl", checkpoint) if (out / "calls.jsonl").exists() else []
    stats = {"acc": overall, "acc_entries": mean([p["correct"][0] for p in per_entry]), "entries": len(per_entry),
             "per_category": per_category, "web_search_included": not left_out, **(call_stats(calls) if calls else {})}
    json.dump(per_entry, open(out / "bfcl_v4_per_problem.json", "w"))
    json.dump({"bfcl_v4": {"acc": overall, "n_problems": len(per_entry), "cut_share": stats.get("cut_share"),
                           "loop_share": stats.get("loop_share"), "mean_tokens": stats.get("tokens_mean")}},
              open(out / "scores.json", "w"), indent=1)
    write_summary(out, {"bfcl_v4": stats}, checkpoint, SAMPLING, began, sum(c["tokens"] for c in calls), mode=mode,
                  categories=names, left_out=left_out, bfcl_table=table)


def bfcl_results(out):
    """Per entry {id, category, correct} and per category {n, acc}. BFCL's score files list only the failures."""
    result_dir, score_dir = out / "bfcl/result" / BFCL_MODEL, out / "bfcl/score" / BFCL_MODEL
    per_entry, per_category = [], {}
    for result_file in sorted(result_dir.rglob("*_result.json")):
        category = result_file.name.removeprefix("BFCL_v4_").removesuffix("_result.json")
        ids = [json.loads(line)["id"] for line in open(result_file) if line.strip()]
        score_file = score_dir / result_file.relative_to(result_dir).parent / result_file.name.replace("_result", "_score")
        lines = [json.loads(line) for line in open(score_file) if line.strip()] if score_file.exists() else []
        failed = {d["id"] for d in lines[1:] if "id" in d}
        per_entry += [{"id": i, "category": category, "correct": [int(i not in failed)]} for i in ids]
        per_category[category] = {"n": len(ids), "acc": lines[0].get("accuracy") if lines else None}
    return per_entry, per_category


# ------------------------------------------------------------------------------------------------- tau2

def tau2(checkpoint, out_dir, tau2_bin, domains="airline,retail,telecom", trials=1, concurrency=8, num_tasks=None,
         user_llm="gpt-4.1-2025-04-14", user_llm_args='{"temperature": 0.0}', model_url=None):
    """tau2-bench: the model is the agent, tau2's default model the customer. Writes tau2_per_problem.json,
    scores.json and summary.json (pass^1 per domain and averaged, how conversations ended, loop statistics).

    A looping agent turn comes back empty, and tau2 v1.0.1 ends that conversation as an infrastructure error and
    by default retries it. Retrying would hand a looping model extra attempts, so every task runs once
    (--max-retries 0) and such a conversation counts as a failure, reported as empty_agent_reply."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    agent_args = {"temperature": SAMPLING["temperature"], "top_p": SAMPLING["top_p"],
                  "presence_penalty": SAMPLING["presence_penalty"], "max_tokens": SAMPLING["max_tokens"],
                  "api_key": "EMPTY", "extra_body": {"top_k": SAMPLING["top_k"],
                                                     "chat_template_kwargs": {"enable_thinking": True}}}
    began = time.time()
    with model_endpoint(checkpoint, out, "qwen3", model_url) as url:
        for domain in domains.split(","):
            with proxy(url, out / f"calls_{domain}.jsonl") as proxy_url:
                command = [tau2_bin, "run", "--domain", domain, "--agent-llm", "openai/afwsp",
                           "--agent-llm-args", json.dumps({**agent_args, "api_base": proxy_url}),
                           "--user-llm", user_llm, "--user-llm-args", user_llm_args, "--num-trials", trials,
                           "--max-concurrency", concurrency, "--max-retries", 0, "--save-to", out / "tau2" / domain]
                run(command + (["--num-tasks", num_tasks] if num_tasks else []), cwd=Path(tau2_bin).resolve().parents[2])

    per_conversation, sets, all_calls = [], {}, []
    for domain in domains.split(","):
        conversations = tau2_conversations(out / "tau2" / domain / "results.json", domain)
        calls = read_calls(out / f"calls_{domain}.jsonl", checkpoint) if (out / f"calls_{domain}.jsonl").exists() else []
        all_calls += calls
        per_conversation += conversations
        sets[domain] = conversation_stats(conversations) | (call_stats(calls) if calls else {})
    accs = [s["acc"] for s in sets.values() if s["acc"] is not None]
    overall = sum(accs) / len(accs) if accs else None
    sets["tau2"] = conversation_stats(per_conversation) | {"acc": overall} | (call_stats(all_calls) if all_calls else {})
    json.dump([{"id": c["id"], "correct": [int(c["success"])], "termination": c["termination"]}
               for c in per_conversation], open(out / "tau2_per_problem.json", "w"))
    json.dump({"tau2": {"acc": overall, "n_problems": len(per_conversation), "cut_share": sets["tau2"].get("cut_share"),
                        "loop_share": sets["tau2"].get("loop_share"), "mean_tokens": sets["tau2"].get("tokens_mean")}},
              open(out / "scores.json", "w"), indent=1)
    write_summary(out, sets, checkpoint, SAMPLING, began, sum(c["tokens"] for c in all_calls), domains=domains,
                  trials=trials, user_llm=user_llm, tau2_version="1.0.1")


def tau2_conversations(results_file, domain):
    rows = []
    for s in json.load(open(results_file))["simulations"]:
        reward = (s.get("reward_info") or {}).get("reward")
        termination = str(s.get("termination_reason"))
        if termination.endswith("infrastructure_error") and EMPTY_REPLY in str((s.get("info") or {}).get("error")):
            termination = "empty_agent_reply"            # the agent thought until its token limit and said nothing
        messages = s.get("messages") or []
        rows.append({"id": f"{domain}/{s['task_id']}/{s.get('trial', 0)}", "reward": reward,
                     "success": reward is not None and abs(reward - 1.0) < 1e-6, "termination": termination,
                     "agent_messages": sum(m.get("role") == "assistant" for m in messages)})
    return rows


def conversation_stats(conversations):
    n = len(conversations)
    return {"acc": mean([c["success"] for c in conversations]), "conversations": n,
            "terminations": dict(collections.Counter(c["termination"] for c in conversations)),
            "empty_reply_share": mean([c["termination"] == "empty_agent_reply" for c in conversations]),
            "agent_messages": mean([c["agent_messages"] for c in conversations])}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bfcl")
    b.add_argument("checkpoint")
    b.add_argument("out_dir")
    b.add_argument("--bfcl-python", required=True, help="python of a venv with bfcl-eval and soundfile")
    b.add_argument("--categories", default="all_scoring")
    b.add_argument("--threads", type=int, default=16)
    b.add_argument("--mode", choices=["raw", "nous"], default="raw",
                   help="raw: tools go to the server natively (Qwen3.5 card); nous: BFCL's Qwen3 prompt recipe")
    b.add_argument("--model-url", help="use a running server instead of starting vLLM")
    t = sub.add_parser("tau2")
    t.add_argument("checkpoint")
    t.add_argument("out_dir")
    t.add_argument("--tau2", required=True, help="the tau2 executable of a tau2-bench v1.0.1 checkout")
    t.add_argument("--domains", default="airline,retail,telecom")
    t.add_argument("--trials", type=int, default=1)
    t.add_argument("--concurrency", type=int, default=8)
    t.add_argument("--num-tasks", type=int, help="first N tasks per domain, for smoke tests")
    t.add_argument("--user-llm", default="gpt-4.1-2025-04-14")
    t.add_argument("--user-llm-args", default='{"temperature": 0.0}')
    t.add_argument("--model-url", help="use a running server instead of starting vLLM")
    args = parser.parse_args()
    if args.command == "bfcl":
        bfcl(args.checkpoint, args.out_dir, args.bfcl_python, args.categories, args.threads, args.mode, args.model_url)
    else:
        tau2(args.checkpoint, args.out_dir, args.tau2, args.domains, args.trials, args.concurrency, args.num_tasks,
             args.user_llm, args.user_llm_args, args.model_url)


if __name__ == "__main__":
    main()
