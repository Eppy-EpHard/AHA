"""What a run built and how each arm did, on one page.

  python -m afwsp.report runs/<name>

Writes observations.json, evals.csv (one row per benchmark and arm) and report.md into the run folder. The
observations cover the pairs, the direction's quality at every layer, the layer ranking with the neurons it chose,
and the edits. Sigma is given two ways: by the pooled two-proportion SE, the convention of docs/RESULTS.md
(MATH-500's 9.8 -> 51.8 at n=500 is 14.38), and by the paired SE over the same questions.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


def quantiles(values):
    values = np.asarray([v for v in values if v is not None], float)
    if not len(values):
        return None
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {"n": len(values), "mean": round(float(values.mean()), 1), "p10": float(p10), "median": float(p50),
            "p90": float(p90)}


def direction_quality(pairs_dir):
    """Per layer: held-out separation, how much of the pairs' difference survives averaging (coherence: 1 when every
    pair points the same way), how well held-out pairs agree with u, and how u turns from one layer to the next."""
    directions = np.load(pairs_dir / "directions.npz")
    diffs = np.load(pairs_dir / "pair_diffs.npz")
    separation = {r["layer"]: r for r in json.load(open(pairs_dir / "separation.json"))}
    heldout = diffs["heldout"]
    layers = sorted(int(k[1:]) for k in directions.files if k[0] == "u" and k[1:].isdigit())
    rows = []
    for l in layers:
        u = directions[f"u{l}"].astype(np.float64)
        d = diffs[f"d{l}"].astype(np.float64)
        norms = np.linalg.norm(d, axis=1)
        row = {"layer": l, "cohens_d": separation[l]["cohens_d"], "pair_agreement": separation[l]["pair_agreement"],
               "coherence": round(float(np.linalg.norm(d.mean(0)) / norms.mean()), 3),
               "cos_pair_u_heldout": round(float((d[heldout] @ u / (norms[heldout] + 1e-9)).mean()), 3),
               "share_pairs_along_u_heldout": round(float((d[heldout] @ u > 0).mean()), 3)}
        if f"u{l + 1}" in directions.files:
            row["cos_next_layer"] = round(float(u @ directions[f"u{l + 1}"]), 3)
        rows.append(row)
    return rows


def observe(run):
    obs = {}
    if (run / "traces/scores.json").exists():
        obs["traces"] = {k: v for k, v in json.load(open(run / "traces/scores.json")).items() if k != "protocol"}
    pairs = run / "pairs"
    if (pairs / "pairs_summary.json").exists():
        obs["pairs"] = json.load(open(pairs / "pairs_summary.json"))
        if (pairs / "pairs_used.json").exists():
            used = json.load(open(pairs / "pairs_used.json"))
            obs["pairs"].update({"used": len(used), "loop_onset": quantiles(p["loop_onset"] for p in used)})
    if (pairs / "directions.npz").exists():
        obs["direction"] = direction_quality(pairs)
    if (run / "layers/layers.json").exists():
        rows = json.load(open(run / "layers/layers.json"))
        obs["layers"] = {"ranking": [r["layer"] for r in rows], "rows": rows}
    edits = {f.stem: json.load(open(f))["layers"] for f in sorted((run / "edits").glob("*.json"))}
    if edits:
        obs["edits"] = edits
    return obs


def load_results(folder):
    """Per question {key: score in [0, 1]}, and the run's cut-off, loop and token statistics."""
    scores = json.load(open(folder / "scores.json"))
    per_question = {r["id"]: float(np.mean(r["correct"]))
                    for f in sorted(folder.glob("*_per_problem.json")) for r in json.load(open(f))}
    sets = [v for k, v in scores.items() if k != "protocol"]
    total = sum(s["total"] for s in sets)
    return per_question, {k: round(sum(s[k] * s["total"] for s in sets) / total, 4)
                          for k in ("cut_share", "loop_share", "mean_tokens")}


def paired(a, b):
    """b against a, on the questions both answered."""
    ids = sorted(set(a) & set(b))
    n = len(ids)
    if not n:
        return None
    x, y = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
    pa, pb, diff = x.mean(), y.mean(), y - x
    p = (pa + pb) / 2
    pooled = np.sqrt(2 * p * (1 - p) / n)
    paired_se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return {"n": n, "acc_a": round(100 * pa, 1), "acc_b": round(100 * pb, 1), "delta": round(100 * (pb - pa), 1),
            "sigma_pooled": round(float((pb - pa) / pooled), 2) if pooled > 0 else None,
            "sigma_paired": round(float((pb - pa) / paired_se), 2) if paired_se > 0 else None,
            "fixed": int((diff > 0).sum()), "broken": int((diff < 0).sum())}


def compare(run, config):
    arms = list(config["arms"])
    results = {}
    for benchmark in config["benchmarks"]:
        runs = {}
        for arm in ["base"] + arms:
            folder = run / "results" / arm / benchmark
            if (folder / "scores.json").exists():
                runs[arm] = load_results(folder)
        if "base" not in runs:
            continue
        results[benchmark] = {
            "runs": {arm: {"acc": round(100 * np.mean(list(per.values())), 1), "n": len(per), **stats}
                     for arm, (per, stats) in runs.items()},
            "vs_base": {arm: paired(runs["base"][0], runs[arm][0]) for arm in arms if arm in runs},
            "arm_vs_arm": {f"{b} - {a}": paired(runs[a][0], runs[b][0])
                           for a, b in itertools.combinations([x for x in arms if x in runs], 2)}}
    return results


def markdown(manifest, obs, results):
    config = manifest["config"]
    arms = list(config["arms"])
    ranks = {l: i + 1 for i, l in enumerate(obs.get("layers", {}).get("ranking", []))}
    lines = [f"# {config['name']}: {Path(config['model']).name}", ""]
    for arm, layers in manifest.get("arms", {}).items():
        how = "top pair of the layer search" if config["arms"][arm] == "top2" else "given"
        lines.append(f"- **{arm}**: layers {layers} ({how}; LP ranks {[ranks.get(l) for l in layers or []]} "
                     f"of {len(ranks)})")
    if "pairs" in obs:
        p = obs["pairs"]
        lines.append(f"- pairs: {p.get('used', '?')} used of {p['matched_prefixes']} matched prefixes "
                     f"({p['prefixes']} prefixes, {p['loops']} of {p['continuations']} continuations looped)")
    lines += ["", "## Against the unedited model", "",
              "| benchmark | n | base | " + " | ".join(f"{a} (Δ, σ pooled / paired, fixed/broken)" for a in arms)
              + " | cut-offs base → arms | loops base → arms |", "|---|---|---|" + "---|" * len(arms) + "---|---|"]
    for benchmark, r in results.items():
        cells = []
        for arm in arms:
            v = r["vs_base"].get(arm)
            cells.append(f"{v['acc_b']} ({v['delta']:+}, {v['sigma_pooled']} / {v['sigma_paired']}, "
                         f"{v['fixed']}/{v['broken']})" if v else "—")
        base = r["runs"]["base"]
        cut = " / ".join(str(r["runs"].get(a, {}).get("cut_share", "—")) for a in arms)
        loop = " / ".join(str(r["runs"].get(a, {}).get("loop_share", "—")) for a in arms)
        lines.append(f"| {benchmark} | {base['n']} | {base['acc']} | " + " | ".join(cells)
                     + f" | {base['cut_share']} → {cut} | {base['loop_share']} → {loop} |")
    contrasts = sorted({k for r in results.values() for k in r["arm_vs_arm"]})
    if contrasts:
        lines += ["", "## Arms against each other", "",
                  "| benchmark | " + " | ".join(f"{k} (Δ, σ pooled / paired)" for k in contrasts) + " |",
                  "|---|" + "---|" * len(contrasts)]
        for benchmark, r in results.items():
            cells = [f"{v['delta']:+} ({v['sigma_pooled']} / {v['sigma_paired']})" if (v := r["arm_vs_arm"].get(k))
                     else "—" for k in contrasts]
            lines.append(f"| {benchmark} | " + " | ".join(cells) + " |")
    steps = manifest.get("steps", {})
    if steps:
        lines += ["", "## Compute", "", "| step | minutes |", "|---|---|"]
        lines += [f"| {name} | {s['seconds'] / 60:.1f} |" for name, s in steps.items() if "seconds" in s]
        lines.append(f"| **total** | **{sum(s.get('seconds', 0) for s in steps.values()) / 60:.1f}** |")
        if manifest.get("gpu"):
            lines.append(f"\nGPU: {manifest['gpu']}")
    return "\n".join(lines) + "\n"


def write_report(run):
    run = Path(run)
    manifest = json.load(open(run / "manifest.json"))
    obs = observe(run)
    results = compare(run, manifest["config"])
    json.dump({**obs, "evals": results}, open(run / "observations.json", "w"), indent=1)
    with open(run / "evals.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "arm", "n", "acc", "delta_vs_base", "sigma_pooled", "sigma_paired", "fixed",
                         "broken", "cut_share", "loop_share", "mean_tokens"])
        for benchmark, r in results.items():
            for arm, stats in r["runs"].items():
                v = r["vs_base"].get(arm) or {}
                writer.writerow([benchmark, arm, stats["n"], stats["acc"], v.get("delta"), v.get("sigma_pooled"),
                                 v.get("sigma_paired"), v.get("fixed"), v.get("broken"), stats["cut_share"],
                                 stats["loop_share"], stats["mean_tokens"]])
    (run / "report.md").write_text(markdown(manifest, obs, results))
    print(f"wrote {run / 'report.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    write_report(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
