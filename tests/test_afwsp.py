"""CPU tests of everything that needs neither a GPU nor a model download:  uv run pytest"""
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from afwsp import agentic, edit, evals, model, report
from afwsp.loops import dup_mask, label, onset

# Toy: a 3-number stream, 4 neurons. Neuron 3 reads 3 on looping tokens and 0 on productive ones.
H_LOOP = np.array([[1, 0, 3, 2], [2, 1, 3, 2]], np.float32)
H_PROD = np.array([[1, 1, 0, 2], [2, 0, 0, 2]], np.float32)
U = np.array([0.6, 0.0, 0.8])
W = torch.tensor([[1, 0, .3, 0], [0, 1, 0, 0], [0, 0, .4, 1]], dtype=torch.bfloat16)


def test_loop_detector_finds_a_planted_loop():
    rng = np.random.default_rng(0)
    ids = np.concatenate([rng.integers(1000, 100000, 600), np.tile(np.arange(7), 200)]).astype(np.int64)
    start = onset(dup_mask(ids))
    assert 600 <= start < 630
    labels = label(ids, cut=True)
    assert labels.onset == start and len(labels.P) == start - 256 and len(labels.R_during) > 0
    assert label(np.arange(800, dtype=np.int64), cut=False).onset is None


def test_the_lp_picks_the_loop_neuron():
    v = edit.pdhg(H_LOOP, H_PROD, delta=6.0, eps=1.0, cost=np.ones(4))
    assert np.allclose(v, [0, 0, -2, 0], atol=0.05)


def test_the_edit_is_rank_one_on_the_chosen_column(tmp_path):
    save_file({model.down_proj(0): W}, str(tmp_path / "model.safetensors"))
    np.savez(tmp_path / "solutions.npz", u0=U, v0=np.array([0, 0, -2.0, 0]))
    edit.write_edit(tmp_path, tmp_path, [0], tmp_path / "edits/toy.safetensors")
    new = load_file(str(tmp_path / "edits/toy.safetensors"))[model.down_proj(0)]
    info = json.load(open(tmp_path / "edits/toy.json"))["layers"]["0"]
    assert info["neurons"] == [2] and info["changed_weights"] == 2
    assert np.allclose(new.float().numpy(), W.double().numpy() + np.outer(U, [0, 0, -2, 0]), atol=1e-2)


def test_the_text_config_is_the_text_part_plus_the_causal_lm_keys(tmp_path):
    hf = {"architectures": ["Qwen3_5ForConditionalGeneration"],
          "text_config": {"num_hidden_layers": 24, "eos_token_id": 248044,
                          "rope_parameters": {"partial_rotary_factor": 0.25}}}
    (tmp_path / "config.json").write_text(json.dumps(hf))
    config, generation = model.text_config(tmp_path)
    assert config["architectures"] == ["Qwen3_5ForCausalLM"] and config["partial_rotary_factor"] == 0.25
    assert config["bos_token_id"] is None and generation["eos_token_id"] == 248044


def test_answers_are_read_from_the_last_box_after_thinking():
    assert evals.extract_answer("<think>\\boxed{1}</think> so \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert evals.extract_answer("<think>looping...") is None


def test_answer_statistics_count_loops_cutoffs_and_terminated_accuracy():
    looping = list(range(600)) + [1, 2, 3, 4, 5, 6, 7] * 200
    answers = [{"tokens": len(looping), "cut": True, "correct": 0, "answered": False, **evals.loop_stats(looping)},
               {"tokens": 300, "cut": False, "correct": 1, "answered": True, **evals.loop_stats(range(300))}]
    s = evals.summarize(answers)
    assert s["loop_share"] == 0.5 and s["cut_share"] == 0.5 and s["loop_share_of_cut"] == 1.0
    assert s["acc"] == 0.5 and s["acc_terminated"] == 1.0 and s["answered_share"] == 0.5


def test_sigma_matches_the_published_math500_number():
    r = report.paired({i: float(i < 49) for i in range(500)}, {i: float(i < 259) for i in range(500)})
    assert r["delta"] == 42.0 and r["sigma_pooled"] == 14.38 and r["fixed"] == 210 and r["broken"] == 0


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not request.get("stream"):
            data = json.dumps({"choices": [{"message": {"content": "hi", "reasoning_content": "think",
                                                        "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]},
                                            "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 7}}).encode()
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
        for chunk in ({"choices": [{"delta": {"reasoning_content": "th"}}]},
                      {"choices": [{"delta": {"reasoning_content": "ink"}}]},
                      {"choices": [{"delta": {"content": "hi"}, "finish_reason": "length"}]},
                      {"choices": [], "usage": {"completion_tokens": 9}}):
            line = b"data: " + json.dumps(chunk).encode() + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
        line = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(line), line))


def post(url, body):
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as r:
        return r.read()


def test_the_proxy_relays_and_logs_plain_and_streamed_calls(tmp_path):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    with agentic.proxy(f"http://127.0.0.1:{upstream.server_address[1]}", tmp_path / "calls.jsonl") as url:
        plain = json.loads(post(url, {"model": "m", "messages": messages, "temperature": 1.0}))
        streamed = post(url, {"model": "m", "messages": messages, "stream": True, "top_k": 20})
    upstream.shutdown()
    assert plain["choices"][0]["message"]["content"] == "hi" and b"[DONE]" in streamed
    a, b = [json.loads(line) for line in open(tmp_path / "calls.jsonl")]
    assert a["tool_calls"] == [{"name": "f", "arguments": "{}"}] and a["params"] == {"temperature": 1.0}
    assert b["reasoning_content"] == "think" and b["content"] == "hi" and b["finish_reason"] == "length"
    assert b["usage"] == {"completion_tokens": 9} and a["conv"] == b["conv"]


def test_conversations_are_keyed_on_their_first_user_message():
    greeting = [{"role": "system", "content": "policy"}, {"role": "assistant", "content": "Hi! How can I help?"}]
    one = greeting + [{"role": "user", "content": "cancel my flight"}]
    two = greeting + [{"role": "user", "content": "change my seat"}]
    assert agentic.conversation_key(one) != agentic.conversation_key(two)
    assert agentic.conversation_key(one) == agentic.conversation_key(one + [{"role": "assistant", "content": "ok"}])


def test_episode_statistics_count_conversations_with_a_loop():
    rows = [{"conv": "a", "tokens": 10, "cut": False, "failed": False, "loop_onset": None, "dup_frac": 0.0},
            {"conv": "a", "tokens": 900, "cut": True, "failed": False, "loop_onset": 40, "dup_frac": 0.9},
            {"conv": "b", "tokens": 10, "cut": False, "failed": False, "loop_onset": None, "dup_frac": 0.0}]
    s = agentic.call_stats(rows)
    assert s["calls"] == 3 and s["episodes"] == 2 and s["episodes_with_loop_share"] == 0.5 and s["failed_calls"] == 0


def test_tau2_counts_an_empty_reply_crash_as_a_failure(tmp_path):
    simulations = [{"task_id": "t1", "trial": 0, "termination_reason": "user_stop", "reward_info": {"reward": 1.0},
                    "messages": [{"role": "assistant"}, {"role": "user"}]},
                   {"task_id": "t2", "trial": 0, "termination_reason": "infrastructure_error", "reward_info": None,
                    "info": {"error": "AssistantMessage must have either content or tool_calls. Got ..."}}]
    (tmp_path / "results.json").write_text(json.dumps({"simulations": simulations}))
    ok, crashed = agentic.tau2_conversations(tmp_path / "results.json", "airline")
    assert ok["success"] and ok["id"] == "airline/t1/0" and ok["agent_messages"] == 1
    assert not crashed["success"] and crashed["termination"] == "empty_agent_reply"


def test_bfcl_entries_missing_from_the_failure_list_are_correct(tmp_path):
    results = tmp_path / "bfcl/result" / agentic.BFCL_MODEL / "non_live"
    scores = tmp_path / "bfcl/score" / agentic.BFCL_MODEL / "non_live"
    results.mkdir(parents=True)
    scores.mkdir(parents=True)
    (results / "BFCL_v4_simple_python_result.json").write_text(
        "\n".join(json.dumps({"id": f"simple_python_{i}"}) for i in range(3)))
    (scores / "BFCL_v4_simple_python_score.json").write_text(
        json.dumps({"accuracy": 2 / 3}) + "\n" + json.dumps({"id": "simple_python_1", "valid": False}))
    entries, categories = agentic.bfcl_results(tmp_path)
    assert [e["correct"][0] for e in entries] == [1, 0, 1] and categories["simple_python"] == {"n": 3, "acc": 2 / 3}
