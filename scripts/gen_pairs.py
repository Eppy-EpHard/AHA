"""Phase 1 of the LLM-free direction: generate matched looping/productive continuations from shared prefixes.

The agent's direction beats the heuristic one because it uses MATCHED PAIRS -- a productive and a looping
version of the same text -- so the difference isolates looping instead of also picking up topic, position and
length. The heuristic compares looping tokens to productive tokens from unrelated contexts, which is why it is
contaminated.

Matched pairs can be made automatically: take a trace that loops, cut it just before its loop onset, and
resample several continuations. Some loop, some do not. Those share an identical prefix by construction, so
the difference between them is the behaviour and nothing else. Same mechanism as the agent's hand-written
pairs, but from the model itself, and thousands instead of four.

Writes cache/pairs/continuations.json: per prefix, the prefix token ids and each continuation's ids, loop flag,
onset and the sampler that drew it; and cache/pairs/gen_summary.json: counts and loop yield per sampler.
The base traces are read from BASE_TRACES, which scripts/run_experiment.py sets per run.

Everything lives under a main guard: vLLM 0.33 forces the `spawn` start method once CUDA is initialised, and
spawn re-imports this module in the child process. Without the guard the child builds a second engine and the
run dies in multiprocessing bootstrap.
"""
import gzip
import json
import os
import sys
import time

import numpy as np
from transformers import AutoTokenizer

from afwsp.eval import grade as G
from afwsp.eval.datasets import load as load_set
from afwsp.loops import dup_mask, onset
from afwsp.paths import CACHE, RESULTS, env_path

CKPT = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.5-0.8B"
MAX_PREFIX = int(os.environ.get("PAIR_MAX_PREFIX", "300"))   # looping traces to branch from
MAX_NEW = int(os.environ.get("PAIR_MAX_NEW", "3072"))        # must exceed loops.onset min_tail (512)
MARGIN = int(os.environ.get("PAIR_MARGIN", "8"))              # cut this many tokens before the onset
MIN_KEEP = int(os.environ.get("PAIR_MIN_KEEP", "150"))        # fewest prefixes worth running
KEEP_FRAC = float(os.environ.get("PAIR_KEEP_FRAC", "0.85"))   # share of candidates the window must retain
# The first attempt resampled every continuation under the repo's THINK35 sampling, whose presence_penalty of
# 1.5 exists precisely to suppress repetition: 12 of 1800 continuations looped (1%), giving 10 usable pairs.
# So draw the two classes under different penalties -- low penalty continuations loop readily, the protocol's
# own penalty keeps producing productive ones. This does NOT contaminate the direction: the residuals are
# taken by teacher-forcing the resulting token sequences, and sampling parameters play no part in a forward
# pass. Sampling is only the means of obtaining both behaviours from one prefix, and every continuation is
# still LABELLED by its own detected loop onset, never by which sampler drew it.
N_LOOP = int(os.environ.get("PAIR_N_LOOP", "4"))              # drawn at low presence_penalty
N_PROD = int(os.environ.get("PAIR_N_PROD", "4"))              # drawn at the protocol's presence_penalty
PP_LOOP = float(os.environ.get("PAIR_PP_LOOP", "0.0"))
PP_PROD = float(os.environ.get("PAIR_PP_PROD", "1.5"))
TRACES = env_path("BASE_TRACES", RESULTS / "gpu/base_fitgen")   # base-model traces to cut prefixes from


def build_prefixes(tok):
    probs = {r["id"]: r["problem"] for s in ("gsm8k_train", "math_train") for r in load_set(s)}
    cands = []
    for s in ("gsm8k_train", "math_train"):
        for r in map(json.loads, gzip.open(TRACES / f"{s}_samples.jsonl.gz")):
            if not r["cut"]:
                continue
            rids = np.array(tok(r["text"], add_special_tokens=False).input_ids, np.int64)
            o = onset(dup_mask(rids), min_tail=512)
            if o is None or o < 64:
                continue
            cands.append({"id": r["id"], "onset": int(o), "rids": rids})
    print(f"{len(cands)} looping traces with a detected onset", flush=True)
    rng = np.random.default_rng(0)
    rng.shuffle(cands)
    # build every candidate's prefix first: the context cap below is chosen from the full length
    # distribution, and sampling down to MAX_PREFIX happens only after that filter.
    prefixes = []
    for c in cands:
        p = tok.apply_chat_template([{"role": "user", "content": G.build_prompt(probs[c["id"]])}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pids = tok(p, add_special_tokens=False).input_ids
        cut = max(c["onset"] - MARGIN, 16)
        prefixes.append({"id": c["id"], "onset": c["onset"],
                         "ids": list(pids) + [int(x) for x in c["rids"][:cut]], "n_prompt": len(pids)})

    lens = sorted(len(p["ids"]) for p in prefixes)
    pct = {q: lens[min(int(q / 100 * len(lens)), len(lens) - 1)] for q in (10, 25, 50, 75, 90, 100)}
    print(f"prefix token lengths: {pct}", flush=True)

    # Smallest window that still retains most of the prefixes.
    #
    # An earlier version required an ABSOLUTE count (min(MAX_PREFIX, MIN_KEEP) = 150 prefixes). When a model
    # yields fewer candidates than that the condition can never be met, so it fell through to the largest
    # window -- on Qwen3.5-2B, 76 candidates meant a 32,768-token context when almost every prefix was far
    # shorter, costing throughput and memory for nothing. Target a FRACTION of what is actually available
    # instead, and fall back to the longest prefix rather than a fixed constant.
    target = max(1, min(MAX_PREFIX, int(round(KEEP_FRAC * len(lens)))))
    cap = next((C for C in (2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768)
                if sum(1 for L in lens if L <= C) >= target), max(lens))
    kept = [p for p in prefixes if len(p["ids"]) <= cap]
    print(f"context cap {cap} keeps {len(kept)}/{len(prefixes)} prefixes "
          f"(target {target}, longest prefix {max(lens)})", flush=True)
    rng2 = np.random.default_rng(1)
    if len(kept) > MAX_PREFIX:
        kept = [kept[i] for i in rng2.permutation(len(kept))[:MAX_PREFIX]]
    print(f"branching from {len(kept)} prefixes x {N_LOOP + N_PROD} continuations", flush=True)
    return kept, cap


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(CKPT)
    prefixes, cap = build_prefixes(tok)
    if not prefixes:
        raise SystemExit("NO-PREFIXES")
    max_len = cap + MAX_NEW + 512

    from vllm import LLM, SamplingParams

    print(f"max_model_len={max_len}", flush=True)
    llm = LLM(model=CKPT, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)),
              max_model_len=max_len, seed=0)
    # the repo's think35 sampling, minus the fixed seed so the continuations differ, and in two penalty groups
    sp = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=pp, max_tokens=MAX_NEW, seed=i)
          for i, pp in enumerate([PP_LOOP] * N_LOOP + [PP_PROD] * N_PROD)]

    all_ids = [p["ids"] for p in prefixes]
    assert max(len(i) for i in all_ids) + MAX_NEW <= max_len, "prefix still exceeds the window"
    outs = [[] for _ in prefixes]
    for j in range(len(sp)):
        try:                                            # newer vLLM
            from vllm import TokensPrompt
            res = llm.generate([TokensPrompt(prompt_token_ids=i) for i in all_ids], sp[j])
        except ImportError:                             # older vLLM
            res = llm.generate(prompt_token_ids=all_ids, sampling_params=sp[j])
        for i, r in enumerate(res):
            outs[i].append([int(t) for t in r.outputs[0].token_ids])
        print(f"continuation set {j + 1}/{len(sp)} (pp={sp[j].presence_penalty}) done", flush=True)

    out = []
    n_both = 0
    for p, conts in zip(prefixes, outs):
        rows = []
        for j, ids in enumerate(conts):
            a = np.array(ids, np.int64)
            o = onset(dup_mask(a), min_tail=512) if len(a) >= 512 else None
            rows.append({"ids": ids, "loops": o is not None, "onset": (int(o) if o is not None else None),
                         "hit_cap": len(ids) >= MAX_NEW,     # so "not looping" can be told from "never finished"
                         "pp": float(sp[j].presence_penalty), "seed": j})   # the sampler that drew it
        has_loop = any(r["loops"] for r in rows)
        has_prod = any(not r["loops"] for r in rows)
        if has_loop and has_prod:
            n_both += 1
        out.append({**{k: p[k] for k in ("id", "onset", "n_prompt")}, "prefix_ids": p["ids"], "conts": rows})

    (CACHE / "pairs").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(CACHE / "pairs/continuations.json", "w"))
    tot = sum(len(o["conts"]) for o in out)
    nl = sum(sum(c["loops"] for c in o["conts"]) for o in out)
    print(f"\n{len(out)} prefixes, {tot} continuations, {nl} loop ({100 * nl / max(tot,1):.0f}%)")
    print(f"MATCHED PAIRS AVAILABLE: {n_both} prefixes produced both a looping and a productive continuation")
    by_pp = {}
    for o in out:
        for c in o["conts"]:
            d = by_pp.setdefault(str(c["pp"]), {"continuations": 0, "loops": 0, "hit_cap_without_loop": 0})
            d["continuations"] += 1
            d["loops"] += int(c["loops"])
            d["hit_cap_without_loop"] += int(c["hit_cap"] and not c["loops"])
    json.dump({"model": CKPT, "base_traces": str(TRACES), "prefixes": len(out), "continuations": tot, "loops": nl,
               "matched_prefixes": n_both, "context_cap": cap, "max_new": MAX_NEW, "margin": MARGIN,
               "by_presence_penalty": by_pp, "seconds": round(time.time() - t0)},
              open(CACHE / "pairs/gen_summary.json", "w"), indent=1)
    print("wrote cache/pairs/continuations.json and gen_summary.json")


if __name__ == "__main__":
    main()
