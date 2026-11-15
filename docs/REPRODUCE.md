# Reproducing this, including the traps

Everything below cost real time or money to find on 2026-11-13. They are recorded so nobody pays twice.

---

## Environment

**vLLM 0.33 + FlashInfer 0.7.2 cannot start on a CUDA 12.4 image.** FlashInfer JIT-compiles sampling kernels
and passes nvcc `--compress-mode=size`, which needs CUDA ≥ 12.8. The failure surfaces as
`RuntimeError: Engine core initialization failed` with the real cause (`nvcc fatal: Unknown option`) buried
dozens of lines up.

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

**Put the venv's `bin/` on PATH.** Launching by absolute interpreter path (`.venv/bin/python`)
hides `ninja` from the compiler and you get `FileNotFoundError: 'ninja'` from inside the engine.

**Force spawn.** vLLM picks `fork` when the parent has not touched CUDA, and the forked engine core then
cannot initialise it (`Cannot re-initialize CUDA in forked subprocess`). Our scripts are main-guarded, so
spawn is safe. Relying on CUDA already being live is luck, not design.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

**Check the GPU actually works before spending setup time.** A cheap cloud instance gave
`torch.cuda.is_available() == False` while `nvidia-smi` happily listed the GPU. Twenty seconds of checking
saves twenty minutes of confusing downstream errors.

```bash
python3 -c "import torch; assert torch.cuda.is_available(); print((torch.ones(2000,2000,device='cuda')@torch.ones(2000,2000,device='cuda')).sum())"
```

**Killing a run does not kill its vLLM engine.** `pkill` on the launcher leaves `VLLM::EngineCore` holding
the GPU — ours sat on 37.5 GB and the next run died at startup. Kill children by PID and verify:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

---

## Data traps

**Base traces belong to one model.** Upstream shipped the 0.8B's traces in git, and a bare "does this file exist"
check silently reused them for a different model. The prefixes and loop onsets then belong to the wrong model
entirely, and the run produces plausible numbers that mean nothing. Upstream shipped the 0.8B's base *scores*
too (`dev_a_base`), with no check at all, and a 2B run compared its edits against that 0.8B baseline; it was
caught only because the logs were read by hand. Here every run generates its own traces and scores its own base
model inside its own folder, `runs/<name>/`, so two models never share them or any other intermediate.

**`presence_penalty` suppresses the behaviour you are collecting.** See the README. 1% yield versus 32%.

**The vLLM engine is not bitwise deterministic across hardware.** The same model, problems and seed gave
`cut_share` 0.234 on one GPU and 0.312 on another. Gate thresholds and any claim about loop frequency carry
a few points of noise for this reason alone.

---

## Analysis traps

**Compare matched splits and matched conditions.** Three errors were made and caught on the same day:

- a `cut_share` for one set compared against a two-set mean, which invented a loop-suppression advantage
  that did not exist;
- a one-set accuracy (70.3, gsm8k only) compared against a two-set mean (64.1), which inverted a model
  comparison;
- a derived-layer arm compared against a run that used a **different direction**, attributing a direction
  improvement (+4 points) to the layer change. The matched control put the layer contribution at +1.5
  points, 0.27 σ.

Always state n, and change one factor at a time.

**The LP diagnostics do not predict the outcome.** Three times the intermediate metric pointed the opposite
way to the task result:

| diagnostic | what it said | what happened |
|---|---|---|
| held-out AUC | whitened direction separates best (0.953) | worst edit (21.9 vs 64.1) |
| layer ranking | Astra's layers rank 16th and 19th of 19 | within noise of the top pair |
| `R_during_hit` | 0.267 / 0.075 on 2B, so the edit must be weak | +14.1 points, 2.80 σ |

Partial suppression is evidently enough to stop a loop. Judge arms by task outcome only.

**Do not "improve" the LP with per-token targets.** `delta_R` is a per-token array, and setting
`delta_t = MULT · max(0, s_t − mean(s_P))` instead of a global constant improves every intermediate metric —
hit rate 0.267 → 0.573 at L15 and 0.075 → 0.453 at L19, collateral damage down at both layers — and makes
the model **worse**: 78.1 against 86.7. Overshooting appears to be a feature. The constant pushes the whole
distribution clear while per-token targets leave each token barely fixed.

---

## Things that did not work

- **Layer selection.** Ranking layers by LP feasibility picked 9 and 21 over the inherited 15 and 19, worth
  +1.5 points (0.27 σ). The ranking placed 15 and 19 *last* of 19 scored layers and they still performed
  within noise of the top pair. Layer choice is simply not where the result lives, which is convenient:
  transferring to a new model does not require getting it right.
- **Push multiplier.** 10× instead of 4× costs 4.7 points on maths and gains nothing.
- **Tighter LP tolerance.** At `eps=0.25` the solver satisfies its constraint on *zero* held-out looping
  tokens — a failed solve, not a weak edit. 32.0 on dev_a.
- **More neurons.** k=64 is worse than k=8 (58.6 vs 64.1).

## Scope

Looping diminishes sharply with scale in single-turn QA: ~89% of traces at 0.8B, ~27% at 2B. The method
removes an obstacle; it cannot add an ability the model lacks, which is also what Takehi et al. found for
Gemma 3 270M. Check how much accuracy a model actually loses to looping before running the full pipeline —
`cut_share` is an upper bound on what the edit could recover, and anything under ~8 points is not worth it.
Haitham's `cut_survey.py` (haithamb/matched-pairs) measures it in one generation pass; a survey of eight
benchmarks is in RESULTS.md.

**Beware the answer format, not just the task.** MedQA scores `cut_share` 0.17 as multiple choice and 0.52
on the *same questions* with the options hidden. A visible option list gives the model somewhere to stop, so
multiple-choice benchmarks understate how much looping is costing. Survey the format you intend to score.
