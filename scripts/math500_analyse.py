"""Analyse the MATH-500 arms: headline effect, the conditional-accuracy control, and the per-level curve.

Reads the author's own harness output (<out>/math500_samples.jsonl.gz), so nothing here re-grades anything.
The three questions, fixed in PREREG_math500.md before the run:

  1. the effect, edited minus base, with pooled-SE sigma, claimed only at >= 2 sigma;
  2. accuracy among responses that TERMINATED -- MedQA showed this FALLS, because the problems a model
     loops on are the harder ones, and it has never been measured where this method wins;
  3. accuracy by difficulty level, which maps where looping and ability actually overlap.

  python scripts/math500_analyse.py <base_dir> <edited_dir> <out.json>
"""
import gzip
import json
import math
import os
import sys
from collections import defaultdict


def load(d):
    """-> {id: record}. One sample per problem (the protocol), so id is unique."""
    p = os.path.join(d, "math500_samples.jsonl.gz")
    rows = {}
    with gzip.open(p, "rt") as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = {"correct": bool(r["correct"]), "cut": bool(r["cut"]),
                             "pred": r.get("pred"), "tokens": r.get("tokens", 0)}
    return rows


def levels():
    """unique_id -> level, joined AFTER the run; plays no part in generation."""
    try:
        from datasets import load_dataset
        return {r["unique_id"]: r.get("level") for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    except Exception as e:
        print(f"  (levels unavailable: {type(e).__name__}) ", flush=True)
        return {}


def summarise(rows):
    n = len(rows)
    cut = sum(r["cut"] for r in rows.values())
    ok = sum(r["correct"] for r in rows.values())
    term = [r for r in rows.values() if not r["cut"]]
    term_ok = sum(r["correct"] for r in term)
    return {"n": n, "acc": ok / n, "cut_share": cut / n,
            "terminated": len(term),
            "acc_given_terminated": (term_ok / len(term)) if term else None,
            "mean_tokens": sum(r["tokens"] for r in rows.values()) / n}


def main():
    bd, ed, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    b, e = load(bd), load(ed)
    common = sorted(set(b) & set(e))
    if len(common) != len(b) or len(common) != len(e):
        print(f"  WARNING: base {len(b)}, edited {len(e)}, common {len(common)} -- comparing the common set")
    b = {k: b[k] for k in common}
    e = {k: e[k] for k in common}

    B, E = summarise(b), summarise(e)
    n = len(common)
    d = 100 * (E["acc"] - B["acc"])
    p = (B["acc"] + E["acc"]) / 2
    se = math.sqrt(2 * p * (1 - p) / n) * 100 if 0 < p < 1 else 0.0
    sig = abs(d) / se if se else 0.0

    print("\nMATH-500, Qwen3.5-0.8B -- held out; the direction was fitted on gsm8k_train/math_train")
    print("the author's own harness and grader, so these are comparable to every dev_a number\n")
    print(f"  {'arm':28s} {'acc':>6s} {'cut':>6s} {'termd':>7s} {'acc|term':>9s} {'tokens':>8s}")
    for lab, S in (("base (unedited)", B), ("edited (no LLM, no training)", E)):
        agt = S["acc_given_terminated"]
        print(f"  {lab:28s} {100*S['acc']:6.1f} {S['cut_share']:6.2f} {S['terminated']:7d} "
              f"{(100*agt if agt is not None else float('nan')):9.1f} {S['mean_tokens']:8.0f}")

    print(f"\n  effect: {d:+.1f} points, pooled SE {se:.1f} -> {sig:.2f} sigma  (n={n}, same problems)")
    print(f"  {'CLEARS' if sig >= 2 else 'BELOW'} the pre-registered 2 sigma bar"
          f"{'' if sig >= 2 else ' -> no detectable effect'}")

    # the control this run exists for
    ab, ae = B["acc_given_terminated"], E["acc_given_terminated"]
    res = {"base": B, "edited": E, "delta": d, "sigma": sig, "n": n}
    if ab is not None and ae is not None:
        print(f"\n  CONTROL -- accuracy among responses that terminated: "
              f"base {100*ab:.1f} vs edited {100*ae:.1f} ({100*(ae-ab):+.1f})")
        print("  MedQA fell 11.7 -> 9.3 here, because the problems a model loops on are the harder ones.")
        print("  A fall is EXPECTED and is not a defect; what matters is whether the gain survives it.")
        # how good are the problems the edit newly rescued?
        new = [k for k in common if b[k]["cut"] and not e[k]["cut"]]
        if new:
            rescued_ok = sum(e[k]["correct"] for k in new)
            print(f"\n  rescued: {len(new)} problems the base never finished; edited solves "
                  f"{rescued_ok} of them ({100*rescued_ok/len(new):.1f}%)")
            print(f"  against {100*ab:.1f}% on the ones the base finished unaided"
                  f" -- the gap is the difficulty of what was hidden")
            res["rescued"] = {"n": len(new), "correct": rescued_ok, "acc": rescued_ok / len(new)}
        # did the edit BREAK anything the base already had?
        broke = [k for k in common if b[k]["correct"] and not e[k]["correct"]]
        fixed = [k for k in common if e[k]["correct"] and not b[k]["correct"]]
        print(f"\n  churn: {len(fixed)} newly correct, {len(broke)} newly wrong "
              f"(net {len(fixed)-len(broke):+d})")
        res["churn"] = {"fixed": len(fixed), "broke": len(broke)}

    lv = levels()
    if lv:
        print(f"\n  by difficulty level -- where looping and ability overlap\n")
        print(f"  {'level':8s} {'n':>4s} {'base':>7s} {'edited':>7s} {'delta':>7s} "
              f"{'base cut':>9s} {'edit cut':>9s}")
        per = {}
        buckets = defaultdict(list)
        for k in common:
            buckets[lv.get(k, "?")].append(k)
        for level in sorted(buckets, key=lambda x: (x is None, str(x))):
            ks = buckets[level]
            ba = sum(b[k]["correct"] for k in ks) / len(ks)
            ea = sum(e[k]["correct"] for k in ks) / len(ks)
            bc = sum(b[k]["cut"] for k in ks) / len(ks)
            ec = sum(e[k]["cut"] for k in ks) / len(ks)
            print(f"  {str(level):8s} {len(ks):4d} {100*ba:7.1f} {100*ea:7.1f} {100*(ea-ba):+7.1f} "
                  f"{bc:9.2f} {ec:9.2f}")
            per[str(level)] = {"n": len(ks), "base": 100 * ba, "edited": 100 * ea,
                               "delta": 100 * (ea - ba), "base_cut": bc, "edited_cut": ec}
        res["by_level"] = per

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
