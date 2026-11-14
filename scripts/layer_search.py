"""Derive WHICH layers to edit, instead of inheriting 15 and 19 from the LLM agent's analysis.

Our earlier attempt ranked layers by how well the matched-pair direction SEPARATES looping from productive
tokens (held-out Cohen's d). That ranking was flat and, worse, ranks by a criterion we showed does not predict
edit quality: the whitened direction has the best separability of the three and gives by far the worst edit.

This ranks by a FUNCTIONAL criterion instead -- whether the linear program can actually do its job at that
layer. The LP is asked to push looping tokens down by delta while leaving productive tokens alone; the
diagnostics it already reports say how well that succeeded on held-out positions:

    R_during_hit  share of held-out looping tokens actually pushed past delta   (want high)
    P             mean shift on held-out productive tokens                      (want near 0)
    P_abs_p90     90th percentile |shift| on productive tokens                  (want small)

A layer where the LP satisfies the loop constraint without disturbing productive tokens is a layer where the
behaviour is cleanly linearly separable in that layer's down_proj input -- which is the property the edit
needs. No evaluation is involved, so all 24 layers cost about a minute each.

  ACTS=acts_all EPS=1.0 K=8 python scripts/layer_search.py
Writes cache/layer_search.json (and a copy tagged with the settings) and prints the ranking. Each row also
records the neurons the LP chose, their v, how sparse the dense solve was, the sample sizes and the time taken.
SEED != 0 redraws the LP's token subsamples, to test whether the layer choice is stable.
"""
import json
import os
import time

import numpy as np

from afwsp import acts, lp
from afwsp.paths import CACHE, down_proj
from afwsp.edits import base_tensor

ACTS = os.environ.get("ACTS", "acts_all")
EPS = float(os.environ.get("EPS", "1.0"))
K = int(os.environ.get("K", "8"))
# Tag solutions with the settings that produced them. Without this, two runs at different EPS/MULT
# write the same filename and make_edit_layers silently picks up whichever ran last.
MULT = float(os.environ.get("MULT", "4"))
N_LAYERS = int(os.environ.get("N_LAYERS", "24"))
SEED = int(os.environ.get("SEED", "0"))   # 0 reproduces the original subsamples
TAG = os.environ.get("LS_TAG", f"{ACTS}_eps{EPS}_m{MULT:g}" + (f"_s{SEED}" if SEED else ""))
LAYERS = list(range(N_LAYERS))

# CAVEAT: `u` comes from pairs cut out of the same base traces that `acts.load` splits into fit and
# held-out. The two splits are uncoordinated, so held-out positions here can sit inside prefixes `u`
# was fitted on. The LP solve is honestly held out; the DIRECTION is not. Treat the diagnostics below
# as partly in-sample, and judge layers by task accuracy instead.
U = np.load(CACHE / "pairs/directions.npz")        # the matched-pair direction, all layers, no LLM
H, RES, meta = acts.load(ACTS, layers=LAYERS)
lab, ho = meta["label"], meta["heldout"]
rows = []
for l in LAYERS:
    h = np.asarray(H[l], np.float32)
    fitR, fitP = (lab == 2) & ~ho, (lab == 0) & ~ho
    if fitR.sum() < 50 or fitP.sum() < 50:
        print(f"L{l:2d} skipped (too few positions)", flush=True)
        continue
    t0 = time.time()
    rng = np.random.default_rng(l if SEED == 0 else [SEED, l])   # per-layer, so every layer is fit on its own subsample
    u = U[f"u{l}"].astype(np.float32)
    wu = u @ base_tensor(down_proj(l))
    s = h @ wu
    delta = float(MULT * (s[fitR].mean() - s[fitP].mean()))
    if not np.isfinite(delta) or delta <= 0:
        print(f"L{l:2d} skipped (delta={delta:.3g})", flush=True)
        continue

    iR = rng.choice(np.flatnonzero(fitR), min(4000, int(fitR.sum())), replace=False)
    iP = rng.choice(np.flatnonzero(fitP), min(8000, int(fitP.sum())), replace=False)
    spec = lp.Spec(delta_R=np.full(len(iR), delta), eps=EPS, C=10.0)
    r_full = lp.pdhg(h[iR], h[iP], lp.Spec(**{**spec.__dict__, "cost": np.full(h.shape[1], 1.0)}), omega=100.0)
    scale = h[fitP | fitR].std(0)
    sup = np.sort(np.argsort(-np.abs(r_full.v) * scale)[:K])
    r = lp.pdhg(h[iR], h[iP], lp.Spec(**{**spec.__dict__, "support": sup, "cost": np.full(h.shape[1], 1e-3)}),
                omega=100.0)

    z = h[ho] @ r.v
    hit = float((z[lab[ho] == 2] <= -delta).mean())
    P = float(z[lab[ho] == 0].mean())
    p90 = float(np.percentile(np.abs(z[lab[ho] == 0]), 90))
    # collateral damage per unit of loop suppression: small is good
    ratio = abs(P) / delta if delta else float("inf")
    rows.append({"layer": l, "delta": round(delta, 3), "R_during_hit": round(hit, 3),
                 "P": round(P, 3), "P_abs_p90": round(p90, 3), "P_over_delta": round(ratio, 3),
                 "R_during": round(float(z[lab[ho] == 2].mean()), 3),
                 "support": sup.tolist(), "v_support": [round(float(x), 5) for x in r.v[sup]],
                 "nnz_dense": int((r_full.v != 0).sum()), "n_R": len(iR), "n_P": len(iP),
                 "seconds": round(time.time() - t0, 1)})
    np.save(CACHE / f"layersearch_{TAG}_L{l}_k{K}.npy", np.stack([np.pad(u, (0, len(r.v) - len(u))), r.v]))
    print(f"L{l:2d} {json.dumps(rows[-1])}", flush=True)

# Rank: satisfy the loop constraint first, then least collateral damage on productive tokens.
rows.sort(key=lambda r: (-r["R_during_hit"], r["P_over_delta"]))
json.dump(rows, open(CACHE / "layer_search.json", "w"), indent=1)
json.dump(rows, open(CACHE / f"layer_search_{TAG}_k{K}.json", "w"), indent=1)   # rank order, never overwritten by other settings

print("\nlayers ranked by LP feasibility (hit rate first, then collateral damage)")
print(f"{'rank':>4s} {'layer':>5s} {'hit':>6s} {'P':>7s} {'|P|/delta':>10s}")
for i, r in enumerate(rows):
    mark = "   <-- Astra's choice" if r["layer"] in (15, 19) else ""
    print(f"{i+1:4d} {r['layer']:5d} {r['R_during_hit']:6.3f} {r['P']:7.3f} {r['P_over_delta']:10.3f}{mark}")

ranks = {r["layer"]: i + 1 for i, r in enumerate(rows)}
print(f"\nAstra's layers 15 and 19 rank {ranks.get(15, '-')} and {ranks.get(19, '-')} of {len(rows)}")
print("top pair by this criterion:", [r["layer"] for r in rows[:2]])
