"""Cache activations for an arbitrary RANGE of layers, so layer choice can be searched rather than inherited.

The repo caches only layers 15 and 19 (`cache/acts_fitgen`), because those are the layers Astra picked for this
model. Deriving the layers ourselves needs h and resid at every layer.

  LAYERS=0-11 TRACES=300 OUT=acts_all python scripts/collect_layers.py

Writes into cache/<OUT>/ alongside whatever earlier passes wrote, so the 24 layers can be done in two passes
and keep peak RAM near half. meta.npz is written on the pass that produces it and must match across passes,
so the trace cap and seed are fixed by the caller and checked here.
"""
import gzip
import json
import os

import numpy as np

from afwsp.collect import collect
from afwsp.eval.datasets import load as load_set
from afwsp.model import load
from afwsp.paths import CACHE, RESULTS, env_path

spec = os.environ.get("LAYERS", "0-23")
lo, hi = (int(x) for x in spec.split("-")) if "-" in spec else (int(spec), int(spec))
layers = list(range(lo, hi + 1))
n_traces = int(os.environ.get("TRACES", "300"))
out = CACHE / os.environ.get("OUT", "acts_all")
TRACE_DIR = env_path("BASE_TRACES", RESULTS / "gpu/base_fitgen")   # the same base traces the pairs came from

probs = {r["id"]: r["problem"] for s in ("gsm8k_train", "math_train") for r in load_set(s)}
traces = []
for s in ("gsm8k_train", "math_train"):
    for r in map(json.loads, gzip.open(TRACE_DIR / f"{s}_samples.jsonl.gz")):
        traces.append({"id": r["id"], "problem": probs[r["id"]], "text": r["text"],
                       "cut": r["cut"], "correct": r["correct"]})
# fixed order and cap, so every pass sees exactly the same traces in the same order
traces = traces[:n_traces]
print(f"{len(traces)} traces ({sum(t['cut'] for t in traces)} cut); layers {lo}-{hi}", flush=True)

tok, model = load()
H, RES, meta = collect(model, tok, traces, layers)

out.mkdir(parents=True, exist_ok=True)
for l in layers:
    np.save(out / f"h_L{l}.npy", H[l])
    np.save(out / f"resid_L{l}.npy", RES[l])

mp = out / "meta.npz"
if mp.exists():       # a previous pass wrote it: the two passes must line up position for position
    old = dict(np.load(mp))
    assert len(old["label"]) == len(meta["label"]), "pass mismatch: different number of positions"
    assert (old["label"] == meta["label"]).all() and (old["trace"] == meta["trace"]).all(), "pass mismatch"
    print("meta matches the earlier pass", flush=True)
else:
    np.savez(mp, label=meta["label"], trace=meta["trace"], pos=meta["pos"])
    json.dump({"ids": [t["id"] for t in traces], "onset": meta["onset"]}, open(out / "traces.json", "w"))

n = len(meta["label"])
print(f"wrote {out} : {n} positions, "
      f"{ {k: int((meta['label'] == k).sum()) for k in (0, 1, 2)} } per label (P, R_onset, R_during)")
print(f"h is [{n},{H[layers[0]].shape[1]}] fp16 per layer "
      f"= {n * H[layers[0]].shape[1] * 2 / 1e9:.2f} GB x {len(layers)} layers")
