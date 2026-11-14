"""Phase 2 of the LLM-free direction: a loop direction from automatically generated matched pairs.

Input: cache/pairs/continuations.json (from gen_pairs.py) -- prefixes, each with several continuations that
share that prefix exactly, some looping and some not.

For each prefix with both kinds, teacher-force one looping and one productive continuation through the base
model and take the difference of mean residuals over a MATCHED window. Because the prefix is identical and the
window offsets are identical, the difference isolates looping from content, position and length -- which is
what the agent got from four hand-written pairs and what the heuristic difference-of-means lacks.

    u_l = normalise( mean over pairs [ mean resid_l(looping, window) - mean resid_l(productive, window) ] )

Writes cache/pairs/directions.npz (u per layer) and prints held-out separability per layer, which also gives
an LLM-free way to choose WHICH layers to edit -- the author's other stated open problem. For later analysis it
also keeps every pair's per-layer difference (cache/pairs/pair_diffs.npz, with the fit / held-out split) and
what each pair was made of (cache/pairs/pairs_meta.json).
"""
import json
import os

import numpy as np
import torch

from afwsp.model import capture, load
from afwsp.paths import CACHE, TEXT_CONFIG

W_WIN = int(os.environ.get("PAIR_WINDOW", "160"))    # tokens of each continuation to average over
OFF = int(os.environ.get("PAIR_OFFSET", "64"))       # start this far past the looping continuation's onset
MAX_CTX = int(os.environ.get("PAIR_MAX_CTX", "32768"))   # prefixes run long; see gen_pairs context cap
# Depth, from the text-only config of the model being run (TEXT_CONFIG, set per model with AFWSP_TEXT_CONFIG).
# The default, data/model_text, is the 0.8B's; for any other model set AFWSP_TEXT_CONFIG or PAIR_LAYERS.
def _n_layers():
    c = TEXT_CONFIG / "config.json"
    if c.exists():
        d = json.load(open(c))
        if "num_hidden_layers" in d:
            return int(d["num_hidden_layers"])
    raise SystemExit(f"cannot determine depth from {c}: set AFWSP_TEXT_CONFIG or PAIR_LAYERS")

LAYERS = list(range(int(os.environ.get("PAIR_LAYERS", 0)) or _n_layers()))

pairs = json.load(open(CACHE / "pairs/continuations.json"))
usable = []
for p in pairs:
    loops = [c for c in p["conts"] if c["loops"] and c["onset"] is not None]
    prods = [c for c in p["conts"] if not c["loops"]]   # "not looping HERE", not "answered"
    if loops and prods:
        # longest productive continuation, earliest-onset looping one
        usable.append((p, min(loops, key=lambda c: c["onset"]), max(prods, key=lambda c: len(c["ids"]))))
print(f"{len(usable)} usable matched pairs of {len(pairs)} prefixes", flush=True)
if not usable:
    raise SystemExit("NO-MATCHED-PAIRS")

tok, model = load()
dev = model.device
dropped = 0
pair_meta = []                         # what each used pair was made of, in the order of acc
acc = {l: [] for l in LAYERS}          # per-pair paired differences
hold = {l: {"R": [], "P": []} for l in LAYERS}   # per-pair window means, for separability

with torch.no_grad():
    for i, (p, lc, pc) in enumerate(usable):
        o = lc["onset"]
        lo, hi = o + OFF, o + OFF + W_WIN
        if hi > len(lc["ids"]) or hi > len(pc["ids"]):
            dropped += 1
            continue                      # need the same window to exist in both continuations
        pre = p["prefix_ids"]
        if len(pre) + hi > MAX_CTX:
            keep = max(MAX_CTX - hi, 0)
            pre = pre[len(pre) - keep:] if keep else []   # same prefix for both branches
        means = {}
        for key, c in (("R", lc), ("P", pc)):
            ids = torch.tensor(pre + c["ids"][:hi], device=dev)[None]
            pos = torch.arange(len(pre) + lo, len(pre) + hi, device=dev)
            with capture(model, LAYERS, what=("resid",)) as st:
                model.model(ids)
            means[key] = {l: st[l]["resid"][pos].float().mean(0).cpu().numpy() for l in LAYERS}
        for l in LAYERS:
            acc[l].append(means["R"][l] - means["P"][l])
            hold[l]["R"].append(means["R"][l]); hold[l]["P"].append(means["P"][l])
        pair_meta.append({"id": p["id"], "base_onset": p["onset"], "prefix_len": len(p["prefix_ids"]),
                          "prefix_used": len(pre), "loop_onset": o, "window": [lo, hi],
                          "loop_len": len(lc["ids"]), "prod_len": len(pc["ids"]), "prod_hit_cap": pc.get("hit_cap"),
                          "loop_pp": lc.get("pp"), "prod_pp": pc.get("pp")})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(usable)} pairs", flush=True)

n = len(acc[0])
print(f"\nused {n} pairs ({dropped} dropped: the matched window did not fit both branches)\n")
MIN_PAIRS = int(os.environ.get("PAIR_MIN_PAIRS", "20"))
if n < MIN_PAIRS:
    raise SystemExit(f"TOO-FEW-PAIRS ({n} < {MIN_PAIRS})")

# held-out split over pairs, so the direction is never scored on the pairs that built it
rng = np.random.default_rng(0)
ho = rng.permutation(n)[: max(n // 5, 5)]
fit = np.setdiff1d(np.arange(n), ho)

out, rows = {}, []
for l in LAYERS:
    D = np.stack(acc[l])
    u = D[fit].mean(0)
    u = u / (np.linalg.norm(u) + 1e-9)
    R = np.stack(hold[l]["R"])[ho] @ u
    P = np.stack(hold[l]["P"])[ho] @ u
    d = (R.mean() - P.mean()) / (np.sqrt((R.var() + P.var()) / 2) + 1e-9)
    # paired: the sign of the per-pair difference on held-out pairs
    agree = float((R > P).mean())
    out[f"u{l}"] = u
    rows.append({"layer": l, "cohens_d": round(float(d), 3), "pair_agreement": round(agree, 3)})

np.savez(CACHE / "pairs/directions.npz", **out, n_pairs=np.array([n]))
np.savez_compressed(CACHE / "pairs/pair_diffs.npz", fit=fit, heldout=ho,
                    **{f"d{l}": np.stack(acc[l]).astype(np.float16) for l in LAYERS})
json.dump(pair_meta, open(CACHE / "pairs/pairs_meta.json", "w"), indent=1)
rows.sort(key=lambda r: -r["cohens_d"])
print("layers ranked by held-out separation of the matched-pair direction:")
for r in rows[:8]:
    print(f"  L{r['layer']:2d}  d={r['cohens_d']:+.3f}  pair_agreement={r['pair_agreement']:.3f}")
if all(any(r["layer"] == L for r in rows) for L in (15, 19)):
    print("\n(the author's edit uses layers 15 and 19; their ranks here are "
          f"{[i for i, r in enumerate(rows) if r['layer'] == 15][0] + 1} and "
          f"{[i for i, r in enumerate(rows) if r['layer'] == 19][0] + 1} of {len(rows)})")
json.dump(rows, open(CACHE / "pairs/layer_ranking.json", "w"), indent=1)
print("wrote cache/pairs/directions.npz, layer_ranking.json, pair_diffs.npz and pairs_meta.json")
