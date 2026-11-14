"""Generation eval with the authors' think35 protocol (exp8_math/code/evaluate.py):
chat template with thinking on; user = problem + "\\n" + INSTRUCTION; T=0.6, top_p 0.95, top_k 20,
presence_penalty 1.5, max 32,768 new tokens, seed 0; last \\boxed{} graded with math-verify.

  python -m afwsp.eval.maths <ckpt> <out> --sets math500 [--n 1] [--limit 128] [--ids file.json | split:<name>]
Sets: gsm8k_train, math_train (the practice files), math500 and gsm8k (the test split).
Writes <out>/scores.json, <out>/<set>_per_problem.json and <out>/<set>_samples.jsonl.gz (all sample texts).
Adapted from weight-space-lp (MIT).
"""
import argparse
import gzip
import json
import os
import time

from . import grade as G
from . import record
from .datasets import load as load_set

SAMPLES = {"math500": 4, "gsm8k": 2, "gsm8k_train": 1, "math_train": 1}
THINK35 = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768, "seed": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--sets", default="math500")
    ap.add_argument("--n", type=int, help="samples per problem (default: protocol's)")
    ap.add_argument("--limit", type=int, help="first k problems of each set (after --ids filter)")
    ap.add_argument("--ids", help="json file of problem ids to keep, or split:<name> from data/splits.json")
    ap.add_argument("--max-tokens", type=int, default=THINK35["max_tokens"])
    ap.add_argument("--sampling", default="{}", help="json overrides of the sampling params (baselines)")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.makedirs(a.out, exist_ok=True)
    sampling = dict(THINK35, max_tokens=a.max_tokens, **json.loads(a.sampling))
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    llm = LLM(model=a.ckpt, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.88)),
              max_model_len=sampling["max_tokens"] + 2048, seed=0)
    if a.ids and a.ids.startswith("split:"):
        from ..paths import ROOT
        keep = set(json.load(open(ROOT / "data/splits.json"))[a.ids[6:]])
    else:
        keep = set(json.load(open(a.ids))) if a.ids else None
    res = {"_protocol": {"sampling": sampling, "ckpt": a.ckpt}}
    t_start, summary, n_gen = time.time(), {}, 0
    for name in a.sets.split(","):
        t0 = time.time()
        rows = load_set(name)
        if keep is not None:
            rows = [r for r in rows if r["id"] in keep]
        if a.limit:
            rows = rows[:a.limit]
        n = a.n or SAMPLES[name]
        prompts = [tok.apply_chat_template([{"role": "user", "content": G.build_prompt(r["problem"])}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) for r in rows]
        outs = llm.generate(prompts, SamplingParams(n=n, **sampling))
        per, ok, tot, cut, toks, noans, recs = [], 0, 0, 0, 0, 0, []
        with gzip.open(os.path.join(a.out, f"{name}_samples.jsonl.gz"), "wt") as f:
            for r, o in zip(rows, outs):
                c = []
                for s in o.outputs:
                    assert s.finish_reason in ("stop", "length"), s.finish_reason
                    pred, _ = G.extract(s.text)
                    good = G.is_correct(pred, r["answer"])
                    is_cut = s.finish_reason == "length"
                    c.append(int(good))
                    cut += is_cut
                    noans += pred is None and not is_cut
                    toks += len(s.token_ids)
                    ls = record.loop_stats(s.token_ids)
                    recs.append({"tokens": len(s.token_ids), "cut": is_cut, "correct": int(good),
                                 "answered": pred is not None, **ls})
                    f.write(json.dumps({"id": r["id"], "answer": r["answer"], "pred": pred, "correct": good,
                                        "cut": is_cut, "tokens": len(s.token_ids), **ls, "text": s.text}) + "\n")
                per.append({"id": r["id"], "correct": c})
                ok += sum(c)
                tot += len(c)
        json.dump(per, open(os.path.join(a.out, f"{name}_per_problem.json"), "w"))
        res[name] = {"acc": ok / tot, "cut_share": cut / tot, "mean_tokens": toks / tot, "n_problems": len(rows),
                     "samples": n, "ok": ok, "total": tot, "cut": cut, "tokens_sum": toks,
                     "no_answer_finished": noans, "s": round(time.time() - t0)}
        summary[name] = record.stats(recs)
        res[name]["loop_share"] = summary[name]["loop_share"]
        n_gen += toks
        print(name, json.dumps(res[name]), flush=True)
        json.dump(res, open(os.path.join(a.out, "scores.json"), "w"), indent=1)
        record.write_summary(a.out, summary, a.ckpt, sampling, t_start, n_gen)   # after every set, so a crash keeps what ran


if __name__ == "__main__":
    main()
