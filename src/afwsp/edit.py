"""From activations to the edit ΔW = u vᵀ on an MLP's down_proj.

  python -m afwsp.edit collect <model_dir> <traces_dir> <acts_dir> [--traces 300]
  python -m afwsp.edit search <model_dir> <pairs_dir> <acts_dir> <layers_dir> [--eps 1] [--k 8] [--mult 4] [--seed 0]
  python -m afwsp.edit write <model_dir> <layers_dir> 9,21 <edit.safetensors>

collect: teacher-force base answers and save, at every layer, the MLP input h on tokens inside loops and on
productive tokens.
search: at every layer, solve the linear program for v. Looping tokens must be pushed down along u by at least
delta, productive tokens moved by no more than eps, with as few neurons as possible: solve once over all neurons,
keep the k that matter most, and solve again on those. Layers are ranked by how well that holds on held-out tokens.
write: W = bf16(W0 + u vᵀ) on the chosen layers.

The loop labels, the collection and the LP solver are adapted from weight-space-lp (MIT).
"""
import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from .evals import chat_prompt, load_problems
from .loops import label
from .model import capture, decoder_layers, device, down_proj, load, read_weight

# collection: context kept per trace, and positions kept per trace and label
TAIL, MAX_FINISHED, MAX_CONTEXT = 1024, 8192, 16384
PER_TRACE = {0: 256, 1: 64, 2: 64}            # labels: 0 productive, 1 loop onset, 2 inside a loop


# ------------------------------------------------------------------------------------------------- activations

@torch.no_grad()
def collect(model, tokenizer, traces, layers, seed=0):
    """h and resid at sampled labelled positions of each trace, and which trace, position and label each row is."""
    rng = np.random.default_rng(seed)
    h = {l: [] for l in layers}
    resid = {l: [] for l in layers}
    meta = {"label": [], "trace": [], "pos": []}
    for t, trace in enumerate(tqdm(traces)):
        prompt = tokenizer(chat_prompt(tokenizer, trace["problem"]), add_special_tokens=False).input_ids
        answer = np.array(tokenizer(trace["text"], add_special_tokens=False).input_ids, np.int64)
        labels = label(answer, trace["cut"])
        if labels.onset is None and trace["cut"]:
            continue
        end = labels.onset + TAIL if labels.onset is not None else MAX_FINISHED
        end = min(end, len(answer), MAX_CONTEXT - len(prompt))
        picked = []
        for lab, positions in ((0, labels.P), (1, labels.R_onset), (2, labels.R_during)):
            positions = positions[positions < end]
            if len(positions) > PER_TRACE[lab]:
                positions = np.sort(rng.choice(positions, PER_TRACE[lab], replace=False))
            picked += [(lab, int(p)) for p in positions]
        if not picked:
            continue
        ids = torch.tensor(prompt + answer[:end].tolist())[None].to(model.device)
        rows = torch.tensor([len(prompt) + p for _, p in picked], device=model.device)
        with capture(model, layers) as store:
            model.model(ids)
        for l in layers:
            h[l].append(store[l]["h"][rows].to(torch.float16).cpu().numpy())
            resid[l].append(store[l]["resid"][rows].to(torch.float16).cpu().numpy())
        meta["label"] += [lab for lab, _ in picked]
        meta["trace"] += [t] * len(picked)
        meta["pos"] += [p for _, p in picked]
    return ({l: np.concatenate(h[l]) for l in layers}, {l: np.concatenate(resid[l]) for l in layers},
            {k: np.array(v) for k, v in meta.items()})


def read_traces(traces_dir, n_traces):
    problems = {p["id"]: p["problem"] for s in ("gsm8k_train", "math_train") for p in load_problems(s)}
    traces = []
    for s in ("gsm8k_train", "math_train"):
        for r in map(json.loads, gzip.open(Path(traces_dir) / f"{s}_samples.jsonl.gz")):
            traces.append({"id": r["id"], "problem": problems[r["id"]], "text": r["text"], "cut": r["cut"]})
    return traces[:n_traces]


def collect_activations(model_dir, traces_dir, acts_dir, n_traces=300, group=12):
    """Save h_L*.npy and resid_L*.npy for every layer, and meta.npz. Layers go `group` at a time to bound memory."""
    out = Path(acts_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces = read_traces(traces_dir, n_traces)
    tokenizer, model = load(model_dir)
    n_layers = len(decoder_layers(model))
    for first in range(0, n_layers, group):
        layers = list(range(first, min(first + group, n_layers)))
        print(f"layers {layers[0]}-{layers[-1]}", flush=True)
        h, resid, meta = collect(model, tokenizer, traces, layers)
        for l in layers:
            np.save(out / f"h_L{l}.npy", h[l])
            np.save(out / f"resid_L{l}.npy", resid[l])
    np.savez(out / "meta.npz", **meta)
    print(f"{len(meta['label'])} positions: " + ", ".join(
        f"{name} {int((meta['label'] == k).sum())}" for k, name in ((0, "productive"), (1, "loop onset"), (2, "in loop"))))


def load_activations(acts_dir, layers, heldout_share=0.2, seed=0):
    """h and resid per layer (memory-mapped) and meta, with whole traces held out at random."""
    acts_dir = Path(acts_dir)
    meta = dict(np.load(acts_dir / "meta.npz"))
    traces = np.unique(meta["trace"])
    held = set(np.random.default_rng(seed).choice(traces, int(len(traces) * heldout_share), replace=False).tolist())
    meta["heldout"] = np.array([t in held for t in meta["trace"]])
    h = {l: np.load(acts_dir / f"h_L{l}.npy", mmap_mode="r") for l in layers}
    resid = {l: np.load(acts_dir / f"resid_L{l}.npy", mmap_mode="r") for l in layers}
    return h, resid, meta


# ------------------------------------------------------------------------------------------------- the LP

def pdhg(h_loop, h_prod, delta, eps, C=10.0, cost=None, support=None, v_max=10.0, iters=20000, tol=1e-5, omega=100.0):
    """Solve  min Σ c_j|v_j| + C Σ max(0, h·v + delta) [looping] + C Σ max(0, |h·v| − eps) [productive]
    by primal-dual hybrid gradient (Chambolle-Pock, PDLP's base algorithm). Only products with K = [h_loop; h_prod]
    are needed, so every token fits at once, and the L1 step returns an exactly sparse v."""
    dev = device()
    cols = np.arange(h_loop.shape[1]) if support is None else np.asarray(support)
    K = torch.tensor(np.concatenate([h_loop[:, cols], h_prod[:, cols]]), dtype=torch.float32, device=dev)
    n_loop = len(h_loop)
    delta = torch.full((n_loop,), float(delta), device=dev)
    cost = torch.ones(len(cols), device=dev) if cost is None else torch.tensor(cost[cols], dtype=torch.float32, device=dev)
    x = torch.randn(len(cols), device=dev)
    for _ in range(50):                                  # ||K|| by power iteration
        x = K.T @ (K @ x)
        x /= x.norm()
    norm = float((K @ x).norm())
    tau, sigma = 0.95 / (norm * omega), 0.95 * omega / norm
    v = torch.zeros(len(cols), device=dev)
    v_bar, y = v.clone(), torch.zeros(K.shape[0], device=dev)

    def objective(v):
        z = K @ v
        return float((cost * v.abs()).sum() + C * torch.relu(z[:n_loop] + delta).sum()
                     + C * torch.relu(z[n_loop:].abs() - eps).sum())

    best, history = (np.inf, v.clone()), []
    for it in range(iters):
        w = y + sigma * (K @ v_bar)
        y = torch.cat([(w[:n_loop] + sigma * delta).clamp(0, C),
                       torch.sign(w[n_loop:]) * (w[n_loop:].abs() - sigma * eps).clamp(min=0).clamp(max=C)])
        u = v - tau * (K.T @ y)
        v_new = (torch.sign(u) * (u.abs() - tau * cost).clamp(min=0)).clamp(-v_max, v_max)
        v_bar = 2 * v_new - v
        v = v_new
        if it % 200 == 0 or it == iters - 1:
            value = objective(v)
            if value < best[0]:
                best = (value, v.clone())
            history.append(value)
            if it > 2000 and len(history) > 5 and abs(history[-5] - value) < tol * max(1.0, abs(value)):
                break
    solution = np.zeros(h_loop.shape[1])
    solution[cols] = best[1].cpu().numpy()
    return solution


def search_layers(model_dir, pairs_dir, acts_dir, layers_dir, eps=1.0, k=8, mult=4.0, seed=0):
    """Solve for v at every layer and rank the layers: most held-out looping tokens pushed past delta first, then
    least disturbance of productive tokens per unit of delta. Writes layers.json (in rank order) and solutions.npz
    (u and v per layer)."""
    directions = np.load(Path(pairs_dir) / "directions.npz")
    layers = sorted(int(key[1:]) for key in directions.files if key[0] == "u" and key[1:].isdigit())
    h_all, _, meta = load_activations(acts_dir, layers)
    labels, heldout = meta["label"], meta["heldout"]
    rows, solutions = [], {}
    for l in layers:
        began = time.time()
        h = np.asarray(h_all[l], np.float32)
        fit_loop, fit_prod = (labels == 2) & ~heldout, (labels == 0) & ~heldout
        if fit_loop.sum() < 50 or fit_prod.sum() < 50:
            print(f"L{l}: skipped, too few positions", flush=True)
            continue
        u = directions[f"u{l}"].astype(np.float32)
        writes = h @ (u @ read_weight(model_dir, down_proj(l)))       # how much u the layer writes at each token
        delta = float(mult * (writes[fit_loop].mean() - writes[fit_prod].mean()))
        if not np.isfinite(delta) or delta <= 0:
            print(f"L{l}: skipped, delta={delta:.3g}", flush=True)
            continue
        rng = np.random.default_rng(l if seed == 0 else [seed, l])
        loop_rows = rng.choice(np.flatnonzero(fit_loop), min(4000, int(fit_loop.sum())), replace=False)
        prod_rows = rng.choice(np.flatnonzero(fit_prod), min(8000, int(fit_prod.sum())), replace=False)
        dense = pdhg(h[loop_rows], h[prod_rows], delta, eps, cost=np.ones(h.shape[1]))
        neurons = np.sort(np.argsort(-np.abs(dense) * h[fit_loop | fit_prod].std(0))[:k])
        v = pdhg(h[loop_rows], h[prod_rows], delta, eps, cost=np.full(h.shape[1], 1e-3), support=neurons)

        shift = h[heldout] @ v
        on_loops, on_productive = shift[labels[heldout] == 2], shift[labels[heldout] == 0]
        rows.append({"layer": l, "delta": round(delta, 3),
                     "hit_rate": round(float((on_loops <= -delta).mean()), 3),
                     "loop_shift": round(float(on_loops.mean()), 3),
                     "productive_shift": round(float(on_productive.mean()), 3),
                     "productive_shift_p90": round(float(np.percentile(np.abs(on_productive), 90)), 3),
                     "damage_per_delta": round(abs(float(on_productive.mean())) / delta, 3),
                     "neurons": neurons.tolist(), "v": [round(float(x), 5) for x in v[neurons]],
                     "dense_nonzeros": int((dense != 0).sum()), "loop_tokens": len(loop_rows),
                     "productive_tokens": len(prod_rows), "seconds": round(time.time() - began, 1)})
        solutions[f"u{l}"], solutions[f"v{l}"] = u, v
        print(f"L{l}: " + json.dumps({key: rows[-1][key] for key in ("delta", "hit_rate", "damage_per_delta", "neurons")}),
              flush=True)
    rows.sort(key=lambda r: (-r["hit_rate"], r["damage_per_delta"]))
    out = Path(layers_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "layers.json", "w"), indent=1)
    np.savez(out / "solutions.npz", **solutions)
    ranks = {r["layer"]: i + 1 for i, r in enumerate(rows)}
    print(f"ranking: {[r['layer'] for r in rows]}; top pair {sorted(r['layer'] for r in rows[:2])}; "
          f"Astra's 15 and 19 rank {ranks.get(15)} and {ranks.get(19)} of {len(rows)}", flush=True)


# ------------------------------------------------------------------------------------------------- the edit

def write_edit(model_dir, layers_dir, layers, edit_file):
    """W = bf16(W0 + u vᵀ) on each layer, saved to edit_file, with a .json beside it: neurons and changed weights."""
    solutions = np.load(Path(layers_dir) / "solutions.npz")
    tensors, info = {}, {"layers": {}}
    for l in layers:
        u, v = solutions[f"u{l}"].astype(np.float64), solutions[f"v{l}"].astype(np.float64)
        w0 = torch.from_numpy(read_weight(model_dir, down_proj(l)))
        w = (w0.double() + torch.outer(torch.from_numpy(u), torch.from_numpy(v))).to(torch.bfloat16)
        w0 = w0.to(torch.bfloat16)
        changed = w != w0
        tensors[down_proj(l)] = w
        info["layers"][l] = {"neurons": np.flatnonzero(v).tolist(), "changed_weights": int(changed.sum()),
                             "changed_columns": int(changed.any(0).sum()),
                             "relative_change": float((w.float() - w0.float()).norm() / w0.float().norm())}
    edit_file = Path(edit_file)
    edit_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(edit_file))
    json.dump(info, open(edit_file.with_suffix(".json"), "w"), indent=1)
    print(edit_file.name, {l: d["changed_weights"] for l, d in info["layers"].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="activations at every layer")
    c.add_argument("model_dir")
    c.add_argument("traces_dir")
    c.add_argument("acts_dir")
    c.add_argument("--traces", type=int, default=300)
    s = sub.add_parser("search", help="the LP at every layer, ranked")
    s.add_argument("model_dir")
    s.add_argument("pairs_dir")
    s.add_argument("acts_dir")
    s.add_argument("layers_dir")
    s.add_argument("--eps", type=float, default=1.0)
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--mult", type=float, default=4.0)
    s.add_argument("--seed", type=int, default=0)
    w = sub.add_parser("write", help="write the edit for a set of layers")
    w.add_argument("model_dir")
    w.add_argument("layers_dir")
    w.add_argument("layers", help="comma-separated, e.g. 9,21")
    w.add_argument("edit_file")
    args = parser.parse_args()
    if args.command == "collect":
        collect_activations(args.model_dir, args.traces_dir, args.acts_dir, args.traces)
    elif args.command == "search":
        search_layers(args.model_dir, args.pairs_dir, args.acts_dir, args.layers_dir, args.eps, args.k, args.mult,
                      args.seed)
    else:
        write_edit(args.model_dir, args.layers_dir, [int(x) for x in args.layers.split(",")], args.edit_file)


if __name__ == "__main__":
    main()
