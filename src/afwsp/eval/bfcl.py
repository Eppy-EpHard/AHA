"""BFCL V4 for one checkpoint. vLLM serves the model, every call goes through the logging proxy, and BFCL does the
generation and scoring itself.

  python -m afwsp.eval.bfcl <ckpt> <out> [--categories all_scoring] [--threads 16] [--mode raw|nous]

Needs:
  AFWSP_VLLM_PYTHON   the vLLM virtualenv's python, which serves the checkpoint (or AFWSP_MODEL_URL: a running server)
  AFWSP_BFCL_PYTHON   a virtualenv with bfcl-eval and soundfile (docs/AGENTIC.md)
  SERPAPI_API_KEY     for the two web-search categories. Without it they are left out: the per-category scores
                      still stand, but the overall is not BFCL V4's.
Sampling defaults to the setting Qwen reports Qwen3.5's agent scores under: thinking on, temperature 1.0,
top_p 0.95, top_k 20, presence_penalty 1.5, up to 32,768 tokens per call. AFWSP_SAMPLING (JSON) overrides it.

Writes into <out>:
  bfcl/                       BFCL's own result and score files (data_overall.csv is the V4 table)
  calls.jsonl, server.log     every model call (logging proxy); the vLLM server's log
  bfcl_v4_per_problem.json    {id, category, correct} per test entry, for paired comparisons
  scores.json, summary.json   BFCL's overall and per-category accuracy, with loop and cut-off statistics per call
                              and per episode
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..serve import model_endpoint
from . import record
from .proxy import Proxy

ENTRY = Path(__file__).with_name("bfcl_entry.py")
MODEL = "afwsp-qwen3.5-think-FC"          # must match bfcl_entry.MODEL
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768, "seed": 0}


def expand(bfcl_python, wanted: str) -> list[str]:
    code = ("import json, sys\nfrom bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING as M\n"
            "out = []\nfor w in sys.argv[1].split(','):\n    out += [c for c in M.get(w, [w]) if c not in out]\n"
            "print(json.dumps(out))")
    r = subprocess.run([bfcl_python, "-c", code, wanted], capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def run(cmd, env):
    print("$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], env=env, check=True)


def per_entry(out: Path) -> tuple[list[dict], dict]:
    res_dir, score_dir = out / "bfcl/result" / MODEL, out / "bfcl/score" / MODEL
    per, cats = [], {}
    for rf in sorted(res_dir.rglob("*_result.json")):
        cat = rf.name.removeprefix("BFCL_v4_").removesuffix("_result.json")
        ids = [json.loads(l)["id"] for l in open(rf) if l.strip()]
        sf = score_dir / rf.relative_to(res_dir).parent / rf.name.replace("_result.json", "_score.json")
        lines = [json.loads(l) for l in open(sf) if l.strip()] if sf.exists() else []
        failed = {d["id"] for d in lines[1:] if "id" in d}      # the score file lists only the failures
        per += [{"id": i, "category": cat, "correct": [0 if i in failed else 1]} for i in ids]
        cats[cat] = {"n": len(ids), "acc": lines[0].get("accuracy") if lines else None}
    return per, cats


def overall_table(out: Path) -> dict:
    f = out / "bfcl/score/data_overall.csv"
    if not f.exists():
        return {}
    rows = list(csv.DictReader(open(f)))
    return rows[0] if rows else {}


def pct(v):
    try:
        return float(str(v).rstrip("%")) / 100
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--categories", default="all_scoring", help="BFCL categories or groups, comma-separated")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--mode", default=os.environ.get("AFWSP_BFCL_MODE", "raw"), choices=["raw", "nous"])
    ap.add_argument("--max-model-len", type=int, default=131072)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bfcl_py = os.environ["AFWSP_BFCL_PYTHON"]
    sampling = {**SAMPLING, **json.loads(os.environ.get("AFWSP_SAMPLING") or "{}")}
    cats = expand(bfcl_py, a.categories)
    skipped = [] if os.environ.get("SERPAPI_API_KEY") else [c for c in cats if c.startswith("web_search")]
    cats = [c for c in cats if c not in skipped]
    if skipped:
        print(f"SERPAPI_API_KEY not set: leaving out {skipped}; the overall is then not BFCL V4's", flush=True)
    env = {**os.environ, "BFCL_PROJECT_ROOT": str(out / "bfcl"), "AFWSP_SERVED_NAME": "afwsp",
           "AFWSP_BFCL_MODE": a.mode, "AFWSP_SAMPLING": json.dumps(sampling)}
    t0 = time.time()
    with model_endpoint(a.ckpt, out / "server.log", reasoning_parser="qwen3" if a.mode == "raw" else None,
                        max_model_len=a.max_model_len) as base, Proxy(base, out / "calls.jsonl") as px:
        env["AFWSP_BASE_URL"] = px.url
        # no --allow-overwrite: a rerun resumes, skipping entries that already have a result
        run([bfcl_py, ENTRY, "generate", "--test-category", ",".join(cats), "--num-threads", a.threads,
             "--result-dir", out / "bfcl/result"], env)
    run([bfcl_py, ENTRY, "evaluate", "--test-category", ",".join(cats), "--result-dir", out / "bfcl/result",
         "--score-dir", out / "bfcl/score"] + (["--partial-eval"] if skipped else []), env)

    per, by_cat = per_entry(out)
    table = overall_table(out)
    calls = record.call_rows(out / "calls.jsonl", a.ckpt) if (out / "calls.jsonl").exists() else []
    cs = record.call_stats(calls) if calls else {}
    overall = pct(table.get("Overall Acc"))
    entry_acc = sum(p["correct"][0] for p in per) / len(per) if per else None
    sets = {"bfcl_v4": {"acc": overall if overall is not None else entry_acc, "acc_overall_bfcl": overall,
                        "acc_entries": entry_acc, "n_entries": len(per), "per_category": by_cat,
                        "web_search_included": not skipped, **cs}}
    json.dump(per, open(out / "bfcl_v4_per_problem.json", "w"))
    json.dump({"bfcl_v4": {"acc": sets["bfcl_v4"]["acc"], "n_problems": len(per), "cut_share": cs.get("cut_share"),
                           "loop_share": cs.get("loop_share"), "mean_tokens": cs.get("tokens_mean")}},
              open(out / "scores.json", "w"), indent=1)
    record.write_summary(out, sets, a.ckpt, sampling, t0, sum(c["tokens"] for c in calls), mode=a.mode,
                         categories=cats, left_out=skipped, bfcl_table=table)
    print(json.dumps({k: sets["bfcl_v4"].get(k) for k in ("acc", "n_entries", "cut_share", "loop_share",
                                                           "episodes_with_loop_share")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
