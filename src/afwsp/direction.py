"""The loop direction u, from matched pairs of the model's own continuations.

  python -m afwsp.direction pairs <model_dir> <traces_dir> <pairs_dir> [--n-loop 8] [--n-prod 0] [--max-prefix 300]
  python -m afwsp.direction build <model_dir> <pairs_dir>

pairs: cut each base answer that loops 8 tokens before its loop starts, and sample continuations from that exact
prefix. Each continuation is labelled by its own loop onset. A prefix with both a looping and a non-looping
continuation is a matched pair. All continuations are drawn at presence_penalty 0, so the two classes come from
one sampler; the penalty-1.5 protocol barely loops at all (12 of 1,800 continuations).

build: teacher-force both continuations of each pair through the unedited model and average the residual stream
over the same window in each: 160 tokens, starting 64 past the looping one's onset. Subtract.
u (per layer) = normalise(mean over pairs of looping − non-looping), fitted on 80% of the pairs; the other 20%
score how well u separates the two.
"""
import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .evals import chat_prompt, load_problems
from .loops import dup_mask, onset
from .model import capture, decoder_layers, load

CONTEXT_CAPS = (2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768)


def looping_prefixes(tokenizer, traces_dir, margin=8, max_prefix=300, keep_share=0.85):
    """Prefixes cut `margin` tokens before each looping base answer's onset, and the context length that keeps
    `keep_share` of them."""
    problems = {p["id"]: p["problem"] for s in ("gsm8k_train", "math_train") for p in load_problems(s)}
    candidates = []
    for s in ("gsm8k_train", "math_train"):
        for record in map(json.loads, gzip.open(Path(traces_dir) / f"{s}_samples.jsonl.gz")):
            if not record["cut"]:
                continue
            ids = np.array(tokenizer(record["text"], add_special_tokens=False).input_ids, np.int64)
            start = onset(dup_mask(ids))
            if start is None or start < 64:
                continue
            candidates.append({"id": record["id"], "onset": int(start), "ids": ids})
    print(f"{len(candidates)} looping base answers with a detected onset", flush=True)
    if not candidates:
        return [], 0
    np.random.default_rng(0).shuffle(candidates)
    prefixes = []
    for c in candidates:
        prompt = tokenizer(chat_prompt(tokenizer, problems[c["id"]]), add_special_tokens=False).input_ids
        cut = max(c["onset"] - margin, 16)
        prefixes.append({"id": c["id"], "onset": c["onset"], "n_prompt": len(prompt),
                         "ids": list(prompt) + [int(t) for t in c["ids"][:cut]]})
    lengths = sorted(len(p["ids"]) for p in prefixes)
    target = max(1, min(max_prefix, round(keep_share * len(lengths))))
    cap = next((c for c in CONTEXT_CAPS if sum(n <= c for n in lengths) >= target), max(lengths))
    kept = [p for p in prefixes if len(p["ids"]) <= cap]
    if len(kept) > max_prefix:
        kept = [kept[i] for i in np.random.default_rng(1).permutation(len(kept))[:max_prefix]]
    print(f"context cap {cap}: {len(kept)} prefixes kept of {len(prefixes)}", flush=True)
    return kept, cap


def make_pairs(model_dir, traces_dir, pairs_dir, n_loop=8, n_prod=0, max_prefix=300, max_new=3072, margin=8):
    """Sample continuations of every looping prefix and write continuations.json and pairs_summary.json."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    began = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prefixes, cap = looping_prefixes(tokenizer, traces_dir, margin, max_prefix)
    if not prefixes:
        raise SystemExit("no looping base answers to branch from")
    llm = LLM(model=str(model_dir), dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=cap + max_new + 512,
              seed=0)
    penalties = [0.0] * n_loop + [1.5] * n_prod
    samplers = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=p, max_tokens=max_new, seed=i)
                for i, p in enumerate(penalties)]
    continuations = [[] for _ in prefixes]
    for j, sampler in enumerate(samplers):
        results = llm.generate([TokensPrompt(prompt_token_ids=p["ids"]) for p in prefixes], sampler)
        for i, r in enumerate(results):
            continuations[i].append([int(t) for t in r.outputs[0].token_ids])
        print(f"continuation set {j + 1}/{len(samplers)} done", flush=True)

    records = []
    for prefix, conts in zip(prefixes, continuations):
        rows = []
        for j, ids in enumerate(conts):
            loop_start = onset(dup_mask(np.array(ids, np.int64)))
            rows.append({"ids": ids, "loops": loop_start is not None, "onset": loop_start,
                         "hit_cap": len(ids) >= max_new, "presence_penalty": penalties[j], "seed": j})
        records.append({"id": prefix["id"], "onset": prefix["onset"], "n_prompt": prefix["n_prompt"],
                        "prefix_ids": prefix["ids"], "conts": rows})
    out = Path(pairs_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(out / "continuations.json", "w"))
    conts = [c for r in records for c in r["conts"]]
    matched = sum(any(c["loops"] for c in r["conts"]) and any(not c["loops"] for c in r["conts"]) for r in records)
    summary = {"prefixes": len(records), "continuations": len(conts), "loops": sum(c["loops"] for c in conts),
               "matched_prefixes": matched, "context_cap": cap, "max_new": max_new, "margin": margin,
               "n_loop": n_loop, "n_prod": n_prod, "seconds": round(time.time() - began)}
    json.dump(summary, open(out / "pairs_summary.json", "w"), indent=1)
    print(json.dumps(summary), flush=True)


def matched_pairs(records):
    """Per prefix with both kinds: the earliest-looping continuation and the longest non-looping one."""
    pairs = []
    for r in records:
        loops = [c for c in r["conts"] if c["loops"]]
        others = [c for c in r["conts"] if not c["loops"]]      # not looping here, which is not the same as answered
        if loops and others:
            pairs.append((r, min(loops, key=lambda c: c["onset"]), max(others, key=lambda c: len(c["ids"]))))
    return pairs


@torch.no_grad()
def build_direction(model_dir, pairs_dir, window=160, offset=64, max_context=32768, min_pairs=20):
    """Write directions.npz (u per layer), separation.json (held-out separation per layer), pair_diffs.npz (every
    pair's difference per layer, with the fit / held-out split) and pairs_used.json (what each pair was made of)."""
    pairs_dir = Path(pairs_dir)
    pairs = matched_pairs(json.load(open(pairs_dir / "continuations.json")))
    print(f"{len(pairs)} matched pairs", flush=True)
    _, model = load(model_dir)
    layers = list(range(len(decoder_layers(model))))
    diffs = {l: [] for l in layers}
    means = {l: {"loop": [], "other": []} for l in layers}
    used = []
    for i, (record, looping, other) in enumerate(pairs, 1):
        if i % 25 == 0:
            print(f"  pair {i}/{len(pairs)}", flush=True)
        lo, hi = looping["onset"] + offset, looping["onset"] + offset + window
        if hi > len(looping["ids"]) or hi > len(other["ids"]):
            continue                                            # the window has to exist in both continuations
        prefix = record["prefix_ids"]
        if len(prefix) + hi > max_context:
            prefix = prefix[len(prefix) - max(max_context - hi, 0):]
        window_means = {}
        for key, cont in (("loop", looping), ("other", other)):
            ids = torch.tensor(prefix + cont["ids"][:hi], device=model.device)[None]
            positions = torch.arange(len(prefix) + lo, len(prefix) + hi, device=model.device)
            with capture(model, layers, what=("resid",)) as store:
                model.model(ids)
            window_means[key] = {l: store[l]["resid"][positions].float().mean(0).cpu().numpy() for l in layers}
        for l in layers:
            diffs[l].append(window_means["loop"][l] - window_means["other"][l])
            means[l]["loop"].append(window_means["loop"][l])
            means[l]["other"].append(window_means["other"][l])
        used.append({"id": record["id"], "base_onset": record["onset"], "loop_onset": looping["onset"],
                     "window": [lo, hi], "prefix_tokens": len(record["prefix_ids"]), "prefix_used": len(prefix),
                     "loop_tokens": len(looping["ids"]), "other_tokens": len(other["ids"]),
                     "other_hit_cap": other["hit_cap"]})
    n = len(used)
    print(f"used {n} pairs ({len(pairs) - n} dropped: the window did not fit both continuations)", flush=True)
    if n < min_pairs:
        raise SystemExit(f"too few pairs ({n} < {min_pairs})")

    heldout = np.random.default_rng(0).permutation(n)[: max(n // 5, 5)]
    fit = np.setdiff1d(np.arange(n), heldout)
    directions, separation = {}, []
    for l in layers:
        u = np.stack(diffs[l])[fit].mean(0)
        u = u / (np.linalg.norm(u) + 1e-9)
        a = np.stack(means[l]["loop"])[heldout] @ u
        b = np.stack(means[l]["other"])[heldout] @ u
        cohens_d = (a.mean() - b.mean()) / (np.sqrt((a.var() + b.var()) / 2) + 1e-9)
        directions[f"u{l}"] = u
        separation.append({"layer": l, "cohens_d": round(float(cohens_d), 3),
                           "pair_agreement": round(float((a > b).mean()), 3)})
    np.savez(pairs_dir / "directions.npz", **directions, n_pairs=np.array([n]))
    np.savez_compressed(pairs_dir / "pair_diffs.npz", fit=fit, heldout=heldout,
                        **{f"d{l}": np.stack(diffs[l]).astype(np.float16) for l in layers})
    json.dump(separation, open(pairs_dir / "separation.json", "w"), indent=1)
    json.dump(used, open(pairs_dir / "pairs_used.json", "w"), indent=1)
    best = sorted(separation, key=lambda r: -r["cohens_d"])[:5]
    print("best held-out separation:", ", ".join(f"L{r['layer']} d={r['cohens_d']}" for r in best), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pairs = sub.add_parser("pairs", help="sample continuations of looping prefixes")
    pairs.add_argument("model_dir")
    pairs.add_argument("traces_dir")
    pairs.add_argument("pairs_dir")
    pairs.add_argument("--n-loop", type=int, default=8, help="continuations at presence_penalty 0")
    pairs.add_argument("--n-prod", type=int, default=0, help="continuations at presence_penalty 1.5")
    pairs.add_argument("--max-prefix", type=int, default=300)
    build = sub.add_parser("build", help="the direction u at every layer")
    build.add_argument("model_dir")
    build.add_argument("pairs_dir")
    args = parser.parse_args()
    if args.command == "pairs":
        make_pairs(args.model_dir, args.traces_dir, args.pairs_dir, args.n_loop, args.n_prod, args.max_prefix)
    else:
        build_direction(args.model_dir, args.pairs_dir)


if __name__ == "__main__":
    main()
