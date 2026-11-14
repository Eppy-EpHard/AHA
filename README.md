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
then exist *within a single sampler*, so class and sampler are not confounded. `gen_pairs.py` still defaults
to the older 4/4 split, so set `PAIR_N_LOOP=8 PAIR_N_PROD=0` to draw everything at `pp=0`.

The residuals come from teacher-forcing, where sampling parameters play no part in the forward pass, and
every continuation is labelled by **its own detected loop onset**, never by which sampler drew it.

---

## Layout

```
src/afwsp/                  the package
  loops.py                  loop detector and token labels
  model.py, collect.py      load the model; capture and cache MLP inputs and layer outputs
  acts.py                   cached activations, split by trace into fit and held-out
  lp.py                     the linear program for v, solved by PDHG
  edits.py, ckpt.py         write rank-1 edits; build text-only checkpoints for vLLM
  serve.py                  serve a checkpoint with vLLM's OpenAI-compatible server (agentic evals)
  paths.py                  every path, overridable per run (AFWSP_MODEL, _TEXT_CONFIG, _CACHE, _EDITS)
  eval/                     maths (GSM8K, MATH, MATH-500), ifeval, medqa, medqa_open, cut_survey;
                            bfcl and tau2 (agentic, docs/AGENTIC.md); proxy (logs every agentic model call);
                            record (per-sample loop stats and summary.json for every eval)
scripts/
  run_experiment.py         the whole pipeline from one config, every step logged (docs/RERUN.md)
  gen_pairs.py         1-3  matched continuations from shared prefixes      (vLLM)
  pair_directions.py   4-5  the direction per layer, and every pair's difference
  collect_layers.py         activations at every layer
  layer_search.py      6    the LP at every layer, ranked (as a predictor, a NULL result - see RESULTS)
  make_edit_layers.py       build an edit for a set of layers
  observe.py, compare.py    observation summaries; paired comparisons and report.md
  math500_analyse.py        MATH-500: effect, the conditional-accuracy control, per-level curve
  make_text_config.py       the text-only config for any Qwen3.5 size
  fetch_exp8.sh             practice problems and the 0.8B text-only config
  fetch_model.sh            a base model at a pinned revision
configs/                    rerun_0.8b/2b (agent-free reruns), math500_0.8b, agentic_0.8b/2b (BFCL-V4, tau2)
tests/                      CPU tests: uv run pytest
docs/
  RESULTS.md                every number, every sigma, every negative result
  REVIEW.md                 independent correctness review: what was wrong, what is still limited
  REPRODUCE.md              step by step, including the traps
  PREREG_medqa_open.md      pre-registration, written before the open-ended run
  PREREG_math500.md         pre-registration + the reference bars, read before the result
  RERUN.md                  the agent-free reruns: what, why, how, what gets logged
  AGENTIC.md                BFCL-V4 and tau2-bench: setup, protocol, what was found wiring them up
data/splits.json            which practice problems (GSM8K / MATH train) are in fit_gen, the set an edit is built
                            from, and in the dev_a, dev_b and final_practice evaluation slices
```

Each run writes its own `runs/`, `cache/`, `edits/`, `ckpts/` and `results/` folders. Nothing is shipped
pre-computed: every run generates its own base traces and base runs.

---

## Running it

On a CUDA machine (docs/RERUN.md has the details and the compute):

```bash
uv sync
bash scripts/fetch_exp8.sh                 # practice problems + the 0.8B text-only config
uv run bash scripts/fetch_model.sh         # Qwen3.5-0.8B at the pinned revision, into models/
uv run pytest                              # CPU tests
```

vLLM needs its own virtualenv: the lockfile pins a vllm that cannot resolve against its torch.

```bash
uv venv /workspace/vllmenv --python 3.12
uv pip install --python /workspace/vllmenv/bin/python vllm transformers datasets safetensors math-verify lm-eval
uv pip install --python /workspace/vllmenv/bin/python -e . --no-deps
```

Then the whole pipeline from one config — traces, pairs, direction, layer search, edits, checkpoints, every
benchmark, observations and a paired report — with each step logged and skipped once done:

```bash
uv run python scripts/run_experiment.py configs/rerun_0.8b.json --dry-run   # the plan, exact commands
uv run python scripts/run_experiment.py configs/rerun_0.8b.json
```

Any stage can also be run by hand: `--dry-run` prints the exact command and environment for each. The pair
generation draws every continuation at `presence_penalty` 0 (`PAIR_N_LOOP=8 PAIR_N_PROD=0`; the script's own
default is the older 4/4 split).

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
MIT licence (THIRD_PARTY_NOTICES.md). This
repository contributes the matched-pair
construction for `u` and the measurements in `docs/RESULTS.md`.
