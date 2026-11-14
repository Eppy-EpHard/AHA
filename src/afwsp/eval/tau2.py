"""tau2-bench (v1.0.1) for one checkpoint, on the airline, retail and telecom domains. The model is the agent: vLLM
serves it with Qwen3.5's tool-call and reasoning parsers, and every call goes through the logging proxy. tau2's
default user model simulates the customer.

  python -m afwsp.eval.tau2 <ckpt> <out> [--domains airline,retail,telecom] [--trials 1] [--concurrency 8]

Needs:
  AFWSP_VLLM_PYTHON  the vLLM virtualenv's python (or AFWSP_MODEL_URL: a running server)
  AFWSP_TAU2         the tau2 executable of a tau2-bench v1.0.1 checkout (docs/AGENTIC.md)
  OPENAI_API_KEY     for the user simulator and tau2's judge, gpt-4.1-2025-04-14 by default (tau2 1.x)
Sampling defaults to the setting Qwen reports Qwen3.5's agent scores under: thinking on, temperature 1.0,
top_p 0.95, top_k 20, presence_penalty 1.5, up to 32,768 tokens per call. AFWSP_SAMPLING (JSON) overrides it.

Writes into <out>:
  tau2/<domain>/                     tau2's own results, one per domain
  calls_<domain>.jsonl, server.log   every agent call (logging proxy); the vLLM server's log
  tau2_per_problem.json              {id: domain/task/trial, correct} per conversation, for paired comparisons
  scores.json, summary.json          pass^1 per domain and averaged; how the conversations ended (agent or user
                                     stop, step limit, too many errors); loop and cut-off statistics per call and
                                     per conversation
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..serve import model_endpoint
from . import record
from .proxy import Proxy

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768, "seed": 0}


EMPTY_REPLY = "must have either content or tool_calls"   # tau2's error for an agent turn with nothing in it


def agent_args(url: str, s: dict) -> dict:
    # no seed: tau2 sets its own per task from --seed (300)
    return {"api_base": url, "api_key": "EMPTY", "temperature": s["temperature"], "top_p": s["top_p"],
            "presence_penalty": s["presence_penalty"], "max_tokens": s["max_tokens"],
            "extra_body": {"top_k": s["top_k"], "chat_template_kwargs": {"enable_thinking": True}}}


def find_results(save_to: Path) -> Path:
    for p in (save_to / "results.json", save_to.with_suffix(".json"), save_to):
        if p.is_file():
            return p
    found = sorted(save_to.rglob("results.json")) if save_to.is_dir() else []
    if found:
        return found[0]
    raise FileNotFoundError(f"no tau2 results under {save_to}")


def conversations(path: Path, domain: str) -> list[dict]:
    rows = []
    for s in json.load(open(path))["simulations"]:
        reward = (s.get("reward_info") or {}).get("reward")
        msgs = s.get("messages") or []
        term = str(s.get("termination_reason"))
        if term.endswith("infrastructure_error") and EMPTY_REPLY in str((s.get("info") or {}).get("error")):
            term = "empty_agent_reply"      # the agent thought until its token limit and said nothing
        rows.append({"id": f"{domain}/{s['task_id']}/{s.get('trial', 0)}", "domain": domain, "task_id": s["task_id"],
                     "trial": s.get("trial", 0), "reward": reward,
                     "success": reward is not None and abs(reward - 1.0) < 1e-6,
                     "termination": term, "duration": s.get("duration"),
                     "agent_messages": sum(m.get("role") == "assistant" for m in msgs),
                     "tool_messages": sum(m.get("role") == "tool" for m in msgs)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--domains", default="airline,retail,telecom")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--num-tasks", type=int, help="first N tasks per domain (smoke tests)")
    ap.add_argument("--retries", type=int, default=0, help="tau2's retries of a failed conversation (its default is 3)")
    ap.add_argument("--user-llm", default="gpt-4.1-2025-04-14")
    ap.add_argument("--user-llm-args", default='{"temperature": 0.0}')
    ap.add_argument("--max-model-len", type=int, default=131072)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tau2 = os.environ["AFWSP_TAU2"]
    sampling = {**SAMPLING, **json.loads(os.environ.get("AFWSP_SAMPLING") or "{}")}
    domains = a.domains.split(",")
    t0 = time.time()
    with model_endpoint(a.ckpt, out / "server.log", reasoning_parser="qwen3", max_model_len=a.max_model_len) as base, \
            Proxy(base, out / f"calls_{domains[0]}.jsonl") as px:
        for d in domains:
            px.log.set_path(out / f"calls_{d}.jsonl")
            cmd = [tau2, "run", "--domain", d, "--agent-llm", "openai/afwsp",
                   "--agent-llm-args", json.dumps(agent_args(px.url, sampling)), "--user-llm", a.user_llm,
                   "--user-llm-args", a.user_llm_args, "--num-trials", str(a.trials),
                   "--max-concurrency", str(a.concurrency), "--save-to", str(out / "tau2" / d),
                   # A looping agent turn comes back empty and tau2 ends the conversation with an infrastructure
                   # error; retrying would hand a looping model extra attempts, so every task runs once.
                   "--max-retries", str(a.retries)]
            if a.num_tasks:
                cmd += ["--num-tasks", str(a.num_tasks)]
            print("$ " + " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=Path(tau2).resolve().parents[2])   # tau2 finds data/ from its checkout

    per, sets, all_calls = [], {}, []
    for d in domains:
        convs = conversations(find_results(out / "tau2" / d), d)
        f = out / f"calls_{d}.jsonl"
        calls = record.call_rows(f, a.ckpt) if f.exists() else []
        all_calls += calls
        per += [{"id": c["id"], "correct": [int(c["success"])], "termination": c["termination"]} for c in convs]
        sets[d] = {"acc": sum(c["success"] for c in convs) / len(convs) if convs else None, "n": len(convs),
                   "reward_mean": sum(c["reward"] or 0 for c in convs) / len(convs) if convs else None,
                   "terminations": dict(collections.Counter(c["termination"] for c in convs)),
                   "empty_reply_share": (sum(c["termination"] == "empty_agent_reply" for c in convs) / len(convs)
                                         if convs else None),
                   "agent_messages_mean": sum(c["agent_messages"] for c in convs) / len(convs) if convs else None,
                   **(record.call_stats(calls) if calls else {})}
    accs = [s["acc"] for s in sets.values() if s["acc"] is not None]
    overall = sum(accs) / len(accs) if accs else None
    cs = record.call_stats(all_calls) if all_calls else {}
    sets["tau2"] = {"acc": overall, "acc_pooled": sum(p["correct"][0] for p in per) / len(per) if per else None,
                    "n": len(per), "terminations": dict(collections.Counter(p["termination"] for p in per)),
                    "empty_reply_share": (sum(p["termination"] == "empty_agent_reply" for p in per) / len(per)
                                          if per else None), **cs}
    json.dump(per, open(out / "tau2_per_problem.json", "w"))
    json.dump({"tau2": {"acc": overall, "n_problems": len(per), "cut_share": cs.get("cut_share"),
                        "loop_share": cs.get("loop_share"), "mean_tokens": cs.get("tokens_mean")}},
              open(out / "scores.json", "w"), indent=1)
    record.write_summary(out, sets, a.ckpt, sampling, t0, sum(c["tokens"] for c in all_calls), domains=domains,
                         trials=a.trials, user_llm=a.user_llm, tau2_version="1.0.1")
    print(json.dumps({d: {k: sets[d].get(k) for k in ("acc", "n", "loop_share", "terminations")} for d in sets}),
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
