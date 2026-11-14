"""Summarise a run into small JSON files for analysis and the write-up. CPU only; reads what the pipeline wrote.

  python scripts/observe.py runs/<name>

Writes runs/<name>/observations/:
  traces.json      the base-model traces the pairs were cut from: accuracy, cut_share, tokens
  pairs.json       prefixes, continuations, loop yield per sampler, matched pairs, onsets, lengths
  directions.json  per layer: held-out separation, how consistently individual pairs agree with u, how much of
                   the pairs' difference survives averaging, how u turns from layer to layer, and its cosine
                   with a reference direction when the config names one
  layers.json      the LP at every layer in rank order, each layer's rank, and the neurons it chose
  edits.json       per arm: layers, their ranks, neurons, changed weights
A section whose inputs do not exist yet is recorded as missing rather than failing the run.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def q(xs):
    xs = np.asarray([x for x in xs if x is not None], float)
    if not len(xs):
        return {"n": 0}
    p = np.percentile(xs, [10, 50, 90])
    return {"n": int(len(xs)), "mean": round(float(xs.mean()), 2), "min": float(xs.min()), "p10": float(p[0]),
            "median": float(p[1]), "p90": float(p[2]), "max": float(xs.max())}


def traces(paths):
    s = json.load(open(ROOT / paths["traces"] / "scores.json"))
    return {k: {f: v.get(f) for f in ("n_problems", "acc", "cut_share", "mean_tokens")}
            for k, v in s.items() if not k.startswith("_")}


def pairs(paths):
    d = ROOT / paths["cache"] / "pairs"
    data = json.load(open(d / "continuations.json"))
    conts = [c for p in data for c in p["conts"]]
    loops = [c for c in conts if c["loops"]]
    prod = [c for c in conts if not c["loops"]]
    by_pp = {}
    for c in conts:
        b = by_pp.setdefault(str(c.get("pp")), {"continuations": 0, "loops": 0})
        b["continuations"] += 1
        b["loops"] += int(c["loops"])
    for b in by_pp.values():
        b["loop_rate"] = round(b["loops"] / b["continuations"], 3)
    out = {"prefixes": len(data), "continuations": len(conts), "loop_rate": round(len(loops) / max(len(conts), 1), 3),
           "matched_prefixes": sum(any(c["loops"] for c in p["conts"]) and any(not c["loops"] for c in p["conts"])
                                   for p in data),
           "by_presence_penalty": by_pp,
           "base_onset": q(p["onset"] for p in data), "prefix_tokens": q(len(p["prefix_ids"]) for p in data),
           "continuation_onset": q(c["onset"] for c in loops),
           "non_looping_tokens": q(len(c["ids"]) for c in prod),
           "non_looping_hit_cap": round(sum(bool(c.get("hit_cap")) for c in prod) / max(len(prod), 1), 3)}
    if (d / "gen_summary.json").exists():
        out["generation"] = json.load(open(d / "gen_summary.json"))
    if (d / "pairs_meta.json").exists():
        meta = json.load(open(d / "pairs_meta.json"))
        out["used_pairs"] = {"n": len(meta), "loop_onset": q(m["loop_onset"] for m in meta),
                             "same_sampler_share": round(np.mean([m["loop_pp"] == m["prod_pp"] for m in meta]), 3)
                             if meta else None,
                             "prefix_truncated": sum(m["prefix_used"] < m["prefix_len"] for m in meta)}
    return out


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def directions(paths, cfg):
    d = ROOT / paths["cache"] / "pairs"
    U = np.load(d / "directions.npz")
    rank = {r["layer"]: r for r in json.load(open(d / "layer_ranking.json"))}
    D = np.load(d / "pair_diffs.npz") if (d / "pair_diffs.npz").exists() else None
    ref = None
    if cfg.get("reference_directions") and (ROOT / cfg["reference_directions"]).exists():
        ref = np.load(ROOT / cfg["reference_directions"])
    layers = sorted(int(k[1:]) for k in U.files if k.startswith("u") and k[1:].isdigit())
    rows = []
    for l in layers:
        u = U[f"u{l}"].astype(np.float64)
        row = {"layer": l, "cohens_d_heldout": rank.get(l, {}).get("cohens_d"),
               "pair_agreement_heldout": rank.get(l, {}).get("pair_agreement")}
        if D is not None:
            X = D[f"d{l}"].astype(np.float64)
            ho = D["heldout"]
            cos = unit(X) @ u
            row.update({"mean_pair_diff_norm": round(float(np.linalg.norm(X, axis=1).mean()), 3),
                        # 1 when every pair points the same way; small when averaging is cancelling noise
                        "coherence": round(float(np.linalg.norm(X.mean(0)) / np.linalg.norm(X, axis=1).mean()), 3),
                        "cos_pair_u_heldout": round(float(cos[ho].mean()), 3),
                        "share_pairs_along_u_heldout": round(float((X[ho] @ u > 0).mean()), 3)})
        if f"u{l + 1}" in U.files:
            row["cos_next_layer"] = round(float(u @ U[f"u{l + 1}"].astype(np.float64)), 3)
        if ref is not None and f"u{l}" in ref.files and ref[f"u{l}"].shape == u.shape:
            row["cos_reference"] = round(float(u @ ref[f"u{l}"].astype(np.float64)), 3)
        rows.append(row)
    return {"n_pairs": int(U["n_pairs"][0]) if "n_pairs" in U.files else None,
            "reference": cfg.get("reference_directions"), "layers": rows}


def layers(paths, manifest):
    rows = json.load(open(ROOT / paths["cache"] / f"layer_search_{paths['layer_search_tag']}.json"))
    ranks = {r["layer"]: i + 1 for i, r in enumerate(rows)}
    arms = manifest.get("arms", {})
    return {"criterion": "share of held-out looping tokens pushed past delta, then collateral damage per delta",
            "ranking": [r["layer"] for r in rows], "rank_of": ranks,
            "arms": {a: {"layers": ls, "ranks": [ranks.get(l) for l in ls or []]} for a, ls in arms.items()},
            "rows": rows}


def edits(paths, manifest, k):
    out = {}
    for arm, ls in manifest.get("arms", {}).items():
        f = ROOT / paths["edits"] / f"{arm}_k{k}.json"
        if f.exists():
            info = json.load(open(f))
            out[arm] = {"layers": ls, "per_layer": info["layers"],
                        "changed_weights_total": sum(v["changed_weights_bf16"] for v in info["layers"].values())}
    return out


def main():
    run = Path(sys.argv[1])
    manifest = json.load(open(run / "manifest.json"))
    paths, cfg = manifest["paths"], manifest["config"]
    k = int(cfg.get("lp", {}).get("K", 8))
    obs = run / "observations"
    obs.mkdir(parents=True, exist_ok=True)
    for name, fn in (("traces", lambda: traces(paths)), ("pairs", lambda: pairs(paths)),
                     ("directions", lambda: directions(paths, cfg)), ("layers", lambda: layers(paths, manifest)),
                     ("edits", lambda: edits(paths, manifest, k))):
        try:
            res = fn()
        except FileNotFoundError as e:
            res = {"missing": str(e)}
        json.dump(res, open(obs / f"{name}.json", "w"), indent=1)
        print(f"{name}: {'missing ' + res['missing'] if 'missing' in res else 'ok'}")


if __name__ == "__main__":
    main()
