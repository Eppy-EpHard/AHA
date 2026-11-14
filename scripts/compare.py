"""Paired comparisons for every benchmark: each arm against base, and the arms against each other.

  python scripts/compare.py runs/<name>

Questions are paired across arms by id; MedQA rows carry no id but come in a fixed order, so they pair by
position, with the gold answer as part of the key so a mismatch drops out instead of being compared. Reported:
accuracy, the difference in points, sigma by the pooled two-proportion SE (the convention of docs/RESULTS.md,
e.g. MATH-500's 9.8 -> 51.8 at n=500 is 14.38 sigma) and by the paired SE, questions fixed and broken, cut_share
and mean tokens; where the eval wrote a summary.json, also the loop detector's loop share and the accuracy among samples
that terminated. Writes runs/<name>/observations/evals.json, observations/evals.csv (one row per benchmark and
arm, for charts) and runs/<name>/report.md.
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATHS = ("dev_a", "dev_b", "final_practice", "math500", "gsm8k_test")


def load(bench, d: Path):
    """_scores plus, when the eval wrote one, what its summary.json adds (pooled over its sets)."""
    per, meta = _scores(bench, d)
    if (d / "summary.json").exists():
        sets = json.load(open(d / "summary.json"))["sets"]
        if bench in sets:                   # agentic evals write their pooled set under the benchmark's name
            s = sets[bench]
            meta.update({"loop_share": s.get("loop_share"), "acc_terminated": s.get("acc_terminated"),
                         "episodes_with_loop_share": s.get("episodes_with_loop_share"),
                         "empty_reply_share": s.get("empty_reply_share")})
            return per, meta
        sets = [s for s in sets.values() if s.get("loop_share") is not None]
        n = sum(s.get("n") or 0 for s in sets)
        nt = sum(s.get("n_terminated") or 0 for s in sets)
        meta["loop_share"] = sum(s["loop_share"] * (s.get("n") or 0) for s in sets) / n if n else None
        meta["acc_terminated"] = (sum(s["acc_terminated"] * s["n_terminated"] for s in sets if s.get("n_terminated"))
                                  / nt) if nt else None
    return per, meta


def _scores(bench, d: Path):
    """{question key: score in [0, 1]} and {cut_share, mean_tokens} for one benchmark run."""
    s = json.load(open(d / "scores.json"))
    if bench in MATHS:
        per = {r["id"]: float(np.mean(r["correct"]))
               for f in sorted(d.glob("*_per_problem.json")) for r in json.load(open(f))}
        sets = [v for k, v in s.items() if not k.startswith("_")]
        tot = sum(v["total"] for v in sets)
        return per, {"cut_share": sum(v["cut_share"] * v["total"] for v in sets) / tot,
                     "mean_tokens": sum(v["mean_tokens"] * v["total"] for v in sets) / tot}
    if bench == "ifeval":
        per = {r["id"]: float(r["correct"][0]) for r in json.load(open(d / "ifeval_per_problem.json"))}
        return per, {k: s["ifeval"][k] for k in ("cut_share", "mean_tokens")}
    if bench in ("bfcl_v4", "tau2"):
        per = {r["id"]: float(r["correct"][0]) for r in json.load(open(d / f"{bench}_per_problem.json"))}
        return per, {k: s[bench].get(k) for k in ("cut_share", "mean_tokens")}
    if bench in ("medqa", "medqa_open"):
        recs = json.load(open(d / ("per_problem.json" if bench == "medqa" else "records.json")))
        per = {f"{i}:{r['gold']}": float(r["correct"]) for i, r in enumerate(recs)}
        return per, {k: s[bench][k] for k in ("cut_share", "mean_tokens")}
    raise KeyError(bench)


def paired(a: dict, b: dict) -> dict:
    """b against a, on the questions both answered."""
    ids = sorted(set(a) & set(b))
    x, y = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
    n = len(ids)
    if not n:
        return {"n": 0}
    pa, pb, d = x.mean(), y.mean(), y - x
    pbar = (pa + pb) / 2
    se_pooled = np.sqrt(2 * pbar * (1 - pbar) / n)
    se_paired = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    sig = lambda se: round(float((pb - pa) / se), 2) if se > 0 else None
    return {"n": n, "acc_a": round(100 * pa, 1), "acc_b": round(100 * pb, 1), "delta": round(100 * (pb - pa), 1),
            "sigma_pooled": sig(se_pooled), "sigma_paired": sig(se_paired),
            "fixed": int((d > 0).sum()), "broken": int((d < 0).sum()), "unpaired": len(set(a) ^ set(b))}


def main():
    run = Path(sys.argv[1])
    manifest = json.load(open(run / "manifest.json"))
    cfg, paths = manifest["config"], manifest["paths"]
    arms = list(cfg["arms"])
    reuse = cfg.get("base_runs", {})
    results, missing = {}, []
    for bench in cfg["benchmarks"]:
        runs = {}
        for arm in ["base"] + arms:
            d = ROOT / reuse[bench] if (arm == "base" and bench in reuse) else ROOT / paths["results"] / arm / bench
            try:
                runs[arm] = load(bench, d)
            except FileNotFoundError:
                missing.append(f"{bench}/{arm}")
        if "base" not in runs:
            continue
        res = {"base_dir": reuse.get(bench, f"{paths['results']}/base/{bench}"),
               "runs": {arm: {"acc": round(100 * np.mean(list(p.values())), 1), "n": len(p),
                              **{k: (round(v, 3) if v is not None else None) for k, v in m.items()}}
                        for arm, (p, m) in runs.items()},
               "vs_base": {arm: paired(runs["base"][0], runs[arm][0]) for arm in arms if arm in runs},
               "arm_vs_arm": {f"{b} - {a}": paired(runs[a][0], runs[b][0])
                              for a, b in itertools.combinations([x for x in arms if x in runs], 2)}}
        results[bench] = res
    obs = run / "observations"
    obs.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "missing": missing}, open(obs / "evals.json", "w"), indent=1)
    cols = ["benchmark", "arm", "n", "acc", "delta_vs_base", "sigma_pooled", "sigma_paired", "fixed", "broken",
            "cut_share", "loop_share", "mean_tokens", "acc_terminated", "episodes_with_loop_share", "empty_reply_share"]
    with open(obs / "evals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for bench, r in results.items():
            for arm, run_ in r["runs"].items():
                v = r["vs_base"].get(arm, {})
                w.writerow({"benchmark": bench, "arm": arm, "n": run_["n"], "acc": run_["acc"],
                            "delta_vs_base": v.get("delta"), "sigma_pooled": v.get("sigma_pooled"),
                            "sigma_paired": v.get("sigma_paired"), "fixed": v.get("fixed"), "broken": v.get("broken"),
                            **{k: run_.get(k) for k in ("cut_share", "loop_share", "mean_tokens", "acc_terminated",
                                                        "episodes_with_loop_share", "empty_reply_share")}})
    (run / "report.md").write_text(report(manifest, results, missing, obs))
    print(f"wrote {obs / 'evals.json'} and {run / 'report.md'}" + (f"; missing: {', '.join(missing)}" if missing else ""))


def report(manifest, results, missing, obs):
    cfg = manifest["config"]
    lines = [f"# {cfg['name']}: {cfg['model']}", ""]
    lay = json.load(open(obs / "layers.json")) if (obs / "layers.json").exists() else {}
    for arm, a in lay.get("arms", {}).items():
        spec = cfg["arms"][arm]
        how = "top pair of the LP layer search" if spec == "top2" else "fixed in the config"
        lines.append(f"- **{arm}**: layers {a['layers']} ({how}; LP ranks {a['ranks']} of {len(lay['ranking'])})")
    pr = json.load(open(obs / "pairs.json")) if (obs / "pairs.json").exists() else {}
    if "matched_prefixes" in pr:
        used = pr.get("used_pairs", {}).get("n", pr["matched_prefixes"])
        lines.append(f"- pairs: {used} used, from {pr['matched_prefixes']} matched prefixes of {pr['prefixes']}; "
                     f"loop rate {pr['loop_rate']:.0%} of {pr['continuations']} continuations")
    arms = list(cfg["arms"])
    lines += ["", "## Against the unedited model", "",
              "| benchmark | n | base | " + " | ".join(f"{a} (Δ, σ pooled / paired, fixed/broken)" for a in arms)
              + " | cut_share base → " + " / ".join(arms) + " | loop share base → " + " / ".join(arms) + " |",
              "|---|---|---|" + "---|" * len(arms) + "---|---|"]
    for bench, r in results.items():
        cells = []
        for a in arms:
            v = r["vs_base"].get(a)
            cells.append(f"{v['acc_b']} ({v['delta']:+}, {v['sigma_pooled']} / {v['sigma_paired']}, {v['fixed']}/{v['broken']})"
                         if v and v.get("n") else "—")
        n = max((v.get("n", 0) for v in r["vs_base"].values()), default=0)
        cuts = " / ".join(str(r["runs"].get(a, {}).get("cut_share", "—")) for a in arms)
        loops = " / ".join(str(r["runs"].get(a, {}).get("loop_share") or "—") for a in arms)
        lines.append(f"| {bench} | {n} | {r['runs']['base']['acc']} | " + " | ".join(cells)
                     + f" | {r['runs']['base']['cut_share']} → {cuts} | {r['runs']['base'].get('loop_share') or '—'} → {loops} |")
    pairs_ = [k for r in results.values() for k in r["arm_vs_arm"]]
    if pairs_:
        names = sorted(set(pairs_))
        lines += ["", "## Arms against each other", "", "| benchmark | " + " | ".join(f"{k} (Δ, σ pooled / paired)" for k in names) + " |",
                  "|---|" + "---|" * len(names)]
        for bench, r in results.items():
            cells = [f"{v['delta']:+} ({v['sigma_pooled']} / {v['sigma_paired']})" if (v := r["arm_vs_arm"].get(k)) and v.get("n")
                     else "—" for k in names]
            lines.append(f"| {bench} | " + " | ".join(cells) + " |")
    steps = manifest.get("steps", {})
    if steps:
        lines += ["", "## Compute", "", "| step | minutes |", "|---|---|"]
        lines += [f"| {k} | {v['seconds'] / 60:.1f} |" for k, v in steps.items() if "seconds" in v]
        lines.append(f"| **total** | **{sum(v.get('seconds', 0) for v in steps.values()) / 60:.1f}** |")
        gpu = manifest.get("environment", {}).get("gpu")
        if gpu:
            lines.append(f"\nGPU: {gpu}")
    if missing:
        lines += ["", "Not run yet: " + ", ".join(missing)]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
