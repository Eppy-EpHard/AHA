# The agent-free reruns

Four of the five headline numbers used Astra's layers 15/19 (RESULTS.md, *Which arm produced which number*);
only dev_a 69.5 chose its own. These configs build the edit end to end with nothing borrowed, and evaluate it on
held-out MATH-500. The search picks the layers. A control arm puts the same direction on 15/19, so the write-up
can say both "no agent anywhere" and whether the agent's layer choice mattered.

| config | model | arms | benchmark |
|---|---|---|---|
| `configs/rerun_0.8b.json` | Qwen3.5-0.8B | `derived`: the search's top pair; `inherited`: 15/19 | MATH-500 (500, held out) |
| `configs/rerun_2b.json` | Qwen3.5-2B | the same two | MATH-500 (500, held out) |

- **Nothing is reused.** Each run generates its own base traces on the 512 fit_gen problems, and runs the base
  model on MATH-500 itself, on the same machine as the edits.
- **To add benchmarks,** list them in `benchmarks`. The registry in `scripts/run_experiment.py` has dev_a, dev_b,
  final_practice, gsm8k_test, ifeval, medqa, medqa_open, bfcl_v4 and tau2.
- **To save compute,** drop the `inherited` arm.

## Setup, once, on a CUDA machine

```bash
uv sync
bash scripts/fetch_exp8.sh                       # practice problems + the 0.8B text-only config
uv run bash scripts/fetch_model.sh               # Qwen3.5-0.8B, pinned
uv run bash scripts/fetch_model.sh Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc   # pinned 2026-11-14
uv venv /workspace/vllmenv --python 3.12
uv pip install --python /workspace/vllmenv/bin/python vllm transformers datasets safetensors math-verify lm-eval
uv pip install --python /workspace/vllmenv/bin/python -e . --no-deps
uv run pytest
```

The 2B's text-only config is derived on its first run by `scripts/make_text_config.py`. The same rule reproduces
the authors' 0.8B files exactly: `uv run python scripts/make_text_config.py --check models/Qwen3.5-0.8B`.

## Run

```bash
uv run python scripts/run_experiment.py configs/rerun_0.8b.json --dry-run    # every command, nothing run
uv run python scripts/run_experiment.py configs/rerun_0.8b.json
uv run python scripts/run_experiment.py configs/rerun_2b.json
```

Re-invoking resumes where it stopped. `--only STAGE[,STAGE]`, `--from STAGE` and `--force` rerun parts.

Optional, for the write-up: is the layer choice stable? Redraw the LP's token subsamples on the same activations
and see whether the top pair moves. It takes minutes:

```bash
AFWSP_CACHE=cache/rerun-0.8b AFWSP_MODEL=models/Qwen3.5-0.8B ACTS=acts_all SEED=1 \
  uv run python scripts/layer_search.py          # writes cache/rerun-0.8b/layer_search_acts_all_eps1.0_m4_s1_k8.json
```

## What gets logged

`runs/<name>/`:

| file | contents |
|---|---|
| `manifest.json` | the config; every step's exact command, environment, start, end, seconds, exit code, mean GPU utilisation, peak GPU memory and mean power; GPU; package versions in both environments; model revision; a hash of the code; the layers each arm used |
| `events.jsonl` | one line as each step starts and ends, with its status and time: `tail -f` it while a run is going |
| `gpu.csv` | nvidia-smi every 30 s: utilisation, memory, power, tagged with the running step |
| `logs/<step>.log` | full output of each step (not tracked by git) |
| `observations/traces.json` | the base traces: accuracy, cut_share, tokens |
| `observations/pairs.json` | prefixes, continuations, loop rate per sampler, matched prefixes, onsets, lengths, truncation |
| `observations/directions.json` | per layer: held-out Cohen's d and pair agreement; how coherent the pairs are (how much survives averaging); cosine of held-out pairs with u; share of pairs along u; cosine with the next layer; cosine with a reference direction, if the config names one |
| `observations/layers.json` | the LP at every layer in rank order: δ, hit rate, collateral damage, the 8 neurons and their v, dense sparsity, time; each layer's rank; the ranks of each arm's layers (where 15/19 land) |
| `observations/edits.json` | per arm: neurons and changed weights per layer |
| `observations/evals.json` | per benchmark, each arm against base and the arms against each other: accuracy, Δ, σ by the pooled convention of RESULTS.md and by paired SE, questions fixed and broken, cut_share, loop share, tokens, accuracy among answers that terminated |
| `observations/evals.csv` | the same, one row per benchmark and arm, for charts |
| `report.md` | all of it on one page |

Every eval folder (`results/gpu/<name>/<arm>/<benchmark>/`) holds, besides the scores the analysis reads:

- **A samples file,** one line per generated answer: the full text, tokens, whether it hit the token limit, the extracted answer and grade, and the loop detector's verdict on the generated tokens (`loop_onset`, `dup_frac`).
- **`summary.json`:** accuracy, and accuracy among answers that terminated; cut_share, loop share, and loops among cut answers; tokens (mean, median, p90); wall time and tokens per second; the checkpoint and the edit it carries; the sampling parameters, GPU and package versions.

The cut survey (`python -m afwsp.eval.cut_survey`) now reports loop share next to cut_share too.

Large intermediates go to `cache/<name>/`, which includes every pair's per-layer difference
(`pairs/pair_diffs.npz`) and what each pair was made of (`pairs/pairs_meta.json`). Edits go to `edits/<name>/`,
checkpoints to `ckpts/<name>/` and per-question results to `results/gpu/<name>/<arm>/<benchmark>/`.

## Compute

These are estimates, scaled from the timings recorded in `results/gpu/*/scores.json` and Haitham's 8.4-hour,
$7.50 day on one RTX 4090.

- **Building an edit:** under an hour and a half. Pairs take about 20 minutes, activations plus the 24-layer
  search 30–60.
- **Evaluation** dominates. Edited arms take 5–30 minutes per benchmark. Base runs are slower, because the base
  model loops to 32,768 tokens; MATH-500 base is the longest.
- **Total:** the 0.8B config takes roughly 5–8 GPU-hours and the 2B config roughly 3–5. That is about $10 on a
  rented 4090.
