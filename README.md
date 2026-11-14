# agent-free-wsp

Agent-free weight-space programming. Finding the loop direction for a weight-space edit **without an LLM**, from the model's own samples.

Small reasoning models fall into degenerate repetition: they repeat themselves until they hit the token
limit and never answer. A rank-1 edit `ΔW = u·vᵀ` on the MLP `down_proj` of two layers removes it — no
fine-tuning, no gradients. The hard part is finding the direction `u`.

[Takehi, Teoh, Zhang and Khattab](https://rikiyat.github.io/blog/weight-space-programming/) found it with a
frontier LLM reading the weights. [Nathan Barry](https://nathan.rs/posts/weight-space-lp/) formalised the
rest of the edit as a linear program and supplied an LLM-free fallback for `u` that works but lands several
points lower. **This repository replaces `u` with an automatic construction that needs no LLM, no labels and
no gradients, and matches what the frontier model found.**

---

## Results

| | base | ours (no LLM) | effect |
|---|---|---|---|
| **Qwen3.5-0.8B**, MATH-500 (500, held out) | 9.8 | **51.8** | +42.0 pts, **14.4 σ** |
| **Qwen3.5-0.8B**, dev_a (128 problems) | 10.9 | **69.5** | — |
| **Qwen3.5-2B**, dev_a (128 problems) | 72.7 | **86.7** | +14.1 pts, **2.80 σ** |
| **Qwen3.5-0.8B**, IFEval (541 prompts) | 22.9 | **62.5** | +39.6 pts, **13.2 σ** |
| **Qwen3.5-2B**, IFEval (541 prompts) | *see RESULTS.md* | **77.1** | control pending at time of writing |

Against the LLM-derived directions on 0.8B: ours 69.5, the agent's direction 62.5, the published
`astra_p50c` edit 59.4 on the same split. Those margins are **under 1.2 σ** and the honest word for them is
*parity*, not a win — see [docs/RESULTS.md](docs/RESULTS.md), which carries every number with its sigma and
every negative result.

The unambiguous numbers are the ones against the **unedited** model: +14.1 points at 2.80 σ on 2B, and
+39.6 points at 13.2 σ on IFEval for 0.8B.

---

## The method

1. Take a generated trace that loops. Find the loop onset with a duplicate-run detector over token ids.
2. Cut the trace a few tokens **before** the onset.
3. Resample several continuations from that **identical prefix**.
4. Keep prefixes that yield **both** a looping and a non-looping continuation — these are matched pairs.
   ("Non-looping" means the detector does not fire at that point, **not** that the branch answered
   correctly; see [docs/REVIEW.md](docs/REVIEW.md).)
5. Teacher-force both branches and difference the mean residual over a matched window:

   `u_l = normalise( mean over pairs [ mean resid_l(looping, window) − mean resid_l(non-looping, window) ] )`

6. Hand `u` to Barry's linear program, which picks ~8 neurons whose `down_proj` columns suppress it.

The prefix is identical by construction, so the difference isolates the **behaviour** rather than the topic,
the position or the length. That is the whole idea, and it is why it beats a difference of means taken over
unrelated contexts.

**Why matched rather than better statistics.** A whitened (Fisher/LDA) direction separates looping from
productive tokens *best* of anything tried — held-out AUC 0.953 against 0.895 — and produces by far the
*worst* edit, 21.9 against 64.1. A dedicated magnitude control rules out mis-scaling. Separability does not
predict edit quality; sharing context does. This is the most interesting result here and it is documented in
[docs/RESULTS.md](docs/RESULTS.md).

### The sampling detail that decides whether it works

The reference protocol sets `presence_penalty=1.5` — a parameter whose job is to suppress repetition.
Resampling every continuation under it yielded **12 loops in 1800 continuations (1%)** and the method simply
failed. Drawing all continuations at `presence_penalty=0` takes the yield to **772/2400**, and matched pairs
then exist *within a single sampler*, so class and sampler are not confounded. That is the default here
(`"pairs": {"n_loop": 8, "n_prod": 0}`).

The residuals come from teacher-forcing, where sampling parameters play no part in the forward pass, and
every continuation is labelled by **its own detected loop onset**, never by which sampler drew it.

---

## Layout

```
src/afwsp/
  loops.py        the loop detector
  model.py        load the model, hook its layers, read its weights, build vLLM checkpoints
  direction.py    matched pairs → u
  edit.py         activations → the LP for v → layer ranking → W + u·vᵀ
  evals.py        MATH-500 and GSM8K, with loop statistics for every answer
  report.py       what a run built and how each arm did → report.md
  agentic.py      BFCL-V4 and tau2-bench (docs/AGENTIC.md); bfcl_entry.py runs inside BFCL's own venv
scripts/
  run.py          the whole method from one config, every step logged
  setup.sh        practice problems and the base models, once
configs/          rerun_0.8b.json, rerun_2b.json (build the edit, evaluate on MATH-500); agentic_*.json (parked)
data/splits.json  which practice problems are fit_gen, the set an edit is built from
tests/            CPU tests: uv run pytest
docs/             RESULTS (every number), REVIEW, REPRODUCE (the traps), AGENTIC, pre-registrations
```

## Running it

On a Linux machine with a GPU. One environment holds everything, vLLM included:

```bash
uv sync
bash scripts/setup.sh                                           # practice problems, Qwen3.5-0.8B and 2B
uv run python scripts/run.py configs/rerun_0.8b.json --dry-run  # every step, nothing run
uv run python scripts/run.py configs/rerun_0.8b.json
```

`run.py` runs, in order: a checkpoint of the base model → its answers to the 512 fit_gen problems → matched
pairs → u → activations → the LP at every layer → an edit per arm → a checkpoint per arm → every benchmark for
every arm and for base → the report. A step that finished is skipped next time, so a crashed run resumes. Any
step can be run by hand: `--dry-run` prints the exact commands.

Everything goes in `runs/<name>/`:

| | |
|---|---|
| `report.md`, `evals.csv`, `observations.json` | the results; what the method built (pairs, direction quality per layer, the layer ranking and its neurons, the edits) |
| `manifest.json`, `events.jsonl`, `gpu.csv`, `logs/` | every step's command, timing and exit code; machine and versions; GPU use every 30 s; each step's full output |
| `results/<arm>/<benchmark>/` | every answer with its loop statistics, scores, `summary.json` |
| `traces/`, `pairs/`, `activations/`, `layers/`, `edits/`, `checkpoints/` | the intermediate products, stage by stage |

Building the direction takes about 22 minutes on one RTX 4090. For reference, Barry reports the frontier
model took 15 minutes for the same step, so **this is not a speed-up** — it removes the dependency, not the
time.

---

## Credits

The idea is Takehi, Teoh, Zhang and Khattab's
([Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), Nov 2026).
The linear program, the evaluation harness and every reference number here are Nathan Barry's
([Can a Linear Program Do Weight-Space Programming?](https://nathan.rs/posts/weight-space-lp/),
[code](https://github.com/nathanrs/weight-space-lp)). Parts of `src/afwsp/` are adapted from his code under its
MIT licence (THIRD_PARTY_NOTICES.md). This repository contributes the matched-pair construction for `u`
and the measurements in `docs/RESULTS.md`.
