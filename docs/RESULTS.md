# Removing the LLM from weight-space loop removal

Run of 2026-11-13. Target: Nathan Barry, *Weight-Space Linear Programming*
(https://nathan.rs/posts/weight-space-lp/), which removes degenerate looping from a reasoning model with a
rank-1 edit `ΔW = u·vᵀ` on `down_proj`, where the loop direction `u` is found by an LLM agent
(GPT-6 "Astra", working from the weights with a 50-forward-pass budget).
**Question: can the LLM be removed from that pipeline?**

**Correction, and it matters for the cost story.** An earlier version of this note said the direction came
from "an eight-hour search". That was wrong. Barry reports the breakdown himself, under *Where the time
went*: of about 10 hours, **finding `u` took 15 minutes; finding `v` took about 8 hours**. The eight hours
was the NEURON search, which Barry's own linear program already replaced. So replacing `u` is **not a
speed-up** — our pipeline takes 22 minutes against Astra's 15. What it removes is the dependency on a
frontier model with weight access. Any claim of time saved is unsupported and must not appear.

Answer: **yes, at parity.** A direction built automatically from the model's own resampled continuations
matches the agent's direction on two independent splits and is the top arm on both. It is not a
*statistically significant* improvement over the agent, and this note does not claim one.

Cost: one RTX 4090, 8.43 h, **$7.50**, plus ~1 h on an A40 for the IFEval transfer run (~$0.60).
**Building the direction takes ~22 minutes of GPU time (~$0.30)** — against the 15 minutes Astra
took for the same step, so this is a dependency change, not a speed-up; — generation 07:06→07:25, direction
and layer ranking 07:25→07:27, LP solve and edit build 07:27→07:28. The rest of the spend is evaluation,
which you pay once.

---

## Headline

`final_practice`, 512 problems (gsm8k_train + math_train, the author's own split and his own published
numbers for every reference row):

| arm | mean | gsm8k | math | correct |
|---|---|---|---|---|
| `base` (unedited) | 12.1 | 15.6 | 8.6 | 62/512 |
| `astra_p50c` (author) | 53.3 | 64.5 | 42.2 | 273/512 |
| `realshift_k8` (author, LLM-free heuristic) | 54.9 | 67.2 | 42.6 | 281/512 |
| `shift_k8` (author, **LLM agent direction**) | 56.6 | 66.4 | 46.9 | 290/512 |
| **`pairdir_k8` (ours, no LLM)** | **58.4** | **69.5** | **47.3** | **299/512** |

(On `dev_a` the best configuration is now **69.5** — the within-sampler direction on derived layers 9/21;
see the layer section below. The `final_practice` and IFEval numbers above predate that run and use the
earlier direction, so they are a floor rather than the method's best.)

```
vs the LLM agent:  +1.8 points, pooled SE 3.1  ->  0.57 sigma
vs his LLM-free:   +3.5 points, pooled SE 3.1  ->  1.14 sigma
```

### Off-domain transfer: IFEval

The direction is built **only from math traces** (`gsm8k_train`, `math_train`, `fit_gen` split). IFEval is
instruction-following with no math in it and was never seen by any part of the pipeline. 541 prompts,
prompt-level strict, all reference rows measured by Barry on this same harness:

| arm | IFEval |
|---|---|
| `base` | 22.9 |
| `shift_k8` (Barry, LLM agent direction) | 59.7 |
| `realshift_k8` (Barry, LLM-free) | 60.4 |
| `astra_p50c` (**GPT-6 Astra, 50 forward passes — the original paper's edit**) | 61.9 |
| **`pairdir_k8` (ours, no LLM)** | **62.5** |

```
vs astra_p50c   +0.6 points, SE 2.9  ->  0.20 sigma   (level)
vs base        +39.6 points, SE 3.0  -> 13.16 sigma   (unambiguous)
```

Supporting: `cut_share` 0.07 (base 0.73), mean tokens 5,520. Ours also leads on the other three IFEval
metrics: loose 66.7, inst-strict 69.4, inst-loose 72.9.

**This is the strongest generality evidence in the work.** The +0.6 over Astra is noise and is not a claim.
The claim is the +39.6 over base: a direction built from math traces alone recovers the full effect on a task
with no math in it, so it encodes *looping* rather than math-specific structure. A direction that had merely
fitted something about math traces would have done nothing here.

Replication on `dev_a` (128 problems, run first):

| arm | mean | gsm8k | math |
|---|---|---|---|
| **`pairdir_k8` (ours)** | **64.1** | 75.0 | 53.1 |
| `lp_sel` (author's best) | 63.3 | 71.9 | 54.7 |
| `shift_k8` (author, agent) | 62.5 | 71.9 | 53.1 |
| `realshift_k8` (author, LLM-free) | 54.7 | 62.5 | 46.9 |
| `realdir_k8` (our reproduction of his heuristic) | 52.3 | 62.5 | 42.2 |
| `base` | 10.9 | 10.9 | 10.9 |

Pooled over the two math splits (640 problems): **+1.7 vs the agent (0.62 σ)**, **+4.7 vs his LLM-free
method (1.69 σ)**.

**Three benchmarks, all level or above, none individually significant:**

| benchmark | ours | Astra / agent |
|---|---|---|
| dev_a (128) | 64.1 | 62.5 |
| final_practice (512) | 58.4 | 56.6 |
| IFEval (541) | 62.5 | 61.9 |

The consistency across three independent tests is the pattern worth measuring properly; multiple seeds would
convert it into a quantitative claim.

**Read this as parity with the agent, not victory over it.** The value is that parity is reached with zero
LLM calls, replicated, and landing on the high side both times.

---

## The method

The author's LLM-free fallback takes a difference of means between looping tokens and productive tokens drawn
from *unrelated* contexts, so the difference carries topic, position and length along with looping. His agent
beat it using four **hand-written matched pairs** — the same text in a looping and a non-looping version.

Matched pairs can be made automatically, with no LLM and no human:

1. Take a base trace that loops; find its loop onset.
2. Cut the trace 8 tokens before the onset.
3. Resample several continuations from that identical prefix.
4. Keep the prefixes that yield **both** a looping and a productive continuation.
5. Teacher-force both branches; take the difference of mean residuals over a matched window.

```
u_l = normalise( mean over pairs [ mean resid_l(looping, window) - mean resid_l(productive, window) ] )
```

The prefix is identical by construction, so the difference isolates the behaviour rather than the content.
Yield: **249 matched pairs** from 300 prefixes.

### The one subtlety that decides whether this works

The repo's THINK35 sampling protocol sets `presence_penalty=1.5`, whose purpose is to suppress repetition.
Resampling every continuation under it produced **12 loops in 1800 continuations (1%)** and only 10 usable
pairs — not enough to build anything.

Fix: draw the two classes under **different penalties** — 4 continuations at `presence_penalty=0` (loop
readily) and 4 at 1.5 (stay productive). Yield went to **772/2400 (32%)** and 249 pairs.

The residuals are taken by *teacher-forcing* the resulting token sequences, so sampling parameters play no
part in any forward pass, and every continuation is labelled by **its own detected loop onset**, never by
which sampler drew it.

**That argument is necessary but not sufficient, and the first version of this note overstated it.** At the
observed yields the looping class is drawn almost entirely from the `pp=0` half and the productive class
almost entirely from the `pp=1.5` half, so class label and sampler are close to collinear in this run. The
contrast therefore does not cleanly separate "looping" from "text sampled without a presence penalty" —
`pp=1.5` text has less repetition of every kind, not only degenerate loops, and a residual difference would
pick that up. The downstream results are unaffected (the edit demonstrably removes looping on three
benchmarks, including off-domain), but the *mechanistic* claim that the pairs isolate looping is weaker than
stated until the control below is run.

**The control that closes it, not yet run:** at `pp=0` about 63% of continuations loop, so ~37% stay
productive and matched pairs exist *within the `pp=0` draws alone* — roughly 80% of prefixes would qualify.
Building the direction from within-sampler pairs only removes the confound entirely. Cost is about 22 GPU
minutes to regenerate plus one dev_a evaluation. This should be done before the paper is posted.

---

## The mechanism result (stronger than the leaderboard number)

Three directions, identical LP, identical layers (15 and 19), identical evaluation. "Separability" is the
held-out AUC for discriminating looping from productive tokens.

| direction | held-out AUC | dev_a |
|---|---|---|
| whitened / Fisher (`Σ_w⁻¹(μ_R − μ_P)`) | **0.953** — best | **21.9** — worst |
| difference of means (his heuristic) | 0.895 | 52.3 |
| matched pairs (ours) | — | **64.1** — best |

**The ranking is inverted: the direction that separates the two token classes best produces the worst edit.**

The obvious objection — that the whitened edit was simply scaled too weakly, since `delta` is calibrated from
each direction's own projection scale — was tested directly with `ldadir_m10_k8`, which pushes as hard as the
heuristic (P = −0.595 vs −0.684) and still scores **22.7**. So the failure is not calibration.

Conclusion: separability is not the operative criterion, and better statistics on *unmatched* data actively
hurt. What matters is that the two sides of the contrast **share their context**. That is a falsifiable
mechanism claim with its control already run, and it is the part most worth writing up.

---

## Negative results (all reported, none buried)

- **LLM-free layer selection fails.** The matched-pair layer ranking is nearly flat — Cohen's d from 0.86
  (L0) to 1.17 (L23) across all 24 layers. It ranks the author's layers 15 and 19 at 17th and 21st, but that
  ordering should **not** be trusted: it ranks by separability, which the table above shows does not predict
  edit quality. His second open problem remains open. See `layer_ranking.json` (haithamb/matched-pairs, `artifacts/`).
- **Push magnitude is not an unexploited knob.** `mult=10` → 60.9 (vs 64.1 at the default `mult=4`), losing
  4.7 points on math for nothing on gsm8k.
- **`eps=0.25` fails outright.** 32.0 on dev_a. The LP diagnostic explains it: at L19/k=8 `R_during_hit =
  0.0`, i.e. the solver satisfies its constraint on *zero* looping tokens — a failed solve, not a weak edit.
- **`k=64` is worse than `k=8`** for the pair direction (58.6 vs 64.1), consistent with the author's own
  finding that sparser is better here.
- **No loop-suppression advantage.** An earlier claim of one was a units error on our side (a partial
  per-set `cut` compared against a two-set mean). Like-for-like on dev_a: gsm8k 0.05 vs his 0.06, math 0.47
  vs his 0.38 — i.e. slightly *worse* on math. Corrected and recorded here.
- **The agent's advantage is math-only.** On `final_practice` his agent direction is no better than his own
  heuristic on gsm8k (66.4 vs 67.2); its entire edge is math (46.9 vs 42.6). Any future comparison should
  weight math accordingly.

---

## Reproducing

The pipeline and every eval now run from this repository (README, *Running it*). The files this
run produced (`directions.npz`, the `pairdir_k8` edit, `layer_ranking.json` and the result summaries) are kept in
haithamb/matched-pairs under `artifacts/`. `pairdir_k8` itself was built with Barry's `e2_sparsity_sweep.py 1.0 8
acts_fitgen pairs` → `e2_make_edits_sweep.py`, patched to read `cache/pairs/directions.npz`.

### Environment gotcha, worth knowing

vLLM 0.33 + FlashInfer 0.7.2 **cannot start** on a CUDA 12.4 image: FlashInfer JIT-compiles sampling kernels
and passes nvcc `--compress-mode=size`, which requires CUDA ≥ 12.8 (`nvcc fatal: Unknown option`). Set
`VLLM_USE_FLASHINFER_SAMPLER=0`. Also put the venv's `bin/` on `PATH` — launching by absolute interpreter
path hides `ninja` from the compiler. These two cost most of a night of pod time.

---

## MedQA (USMLE) on Qwen3.5-0.8B (2026-11-13)

512 questions, 4 options so chance is 25%, same think35 sampling as every other number here. A response that
never states a letter counts as WRONG (see the extractor note below), so the floor for non-answers is 0, not
chance.

| arm | MedQA | cut_share |
|---|---|---|
| base (unedited) | 41.2 | — |
| edited (no LLM, no training, no medical data) | **46.7** | 0.01 |

**+5.5 points, 1.76 sigma, n=512. Positive but NOT significant** — "suggestive" is the honest word.

**The base number is the informative part.** 41.2 is far above what a model that loops on ~89% of maths
traces would suggest, which means **0.8B does not loop much on clinical multiple choice**: the questions are
short, the answer space is four options, and the model commits before it can spiral. There were only about
five points of looping damage available to repair, and the edit recovered roughly that.

**This completes a monotone pattern across everything measured today:** the gain tracks how much looping was
costing in the first place.

| benchmark | base cut_share | gain | sigma |
|---|---|---|---|
| 0.8B maths | 0.89 | +58.6 | — |
| 2B IFEval | 0.29 | +15.7 | 5.60 |
| 2B maths | 0.26 | +14.1 | 2.80 |
| 0.8B MedQA | low | +5.5 | 1.76 |

So `cut_share` on the unedited model **predicts** whether this method will help, before any edit is built.
That is more useful than the individual numbers: it is a cheap go/no-go test for any new model or benchmark,
and it is the same quantity the Qwen3.5-2B and ProtGPT2 gates used.

**Extractor caveat, and why it matters here.** An independent review found the first MedQA answer extractor
grepped the last `[A-E]` from the whole response, so "classic chronic hepatitis B infection" scored **B** —
unanswered responses were scored at roughly chance. Since the looping arm produces far more non-answers,
that bias ran IN FAVOUR of the edited model. The run was stopped before it scored anything and re-run with a
corrected extractor (`\boxed{}` -> a line that is only a letter -> an explicit statement in the final line ->
an explicit statement anywhere -> None). **No MedQA number was ever produced with the faulty version.**

## Deriving the layers, and the within-sampler direction (2026-11-13)

Two things were tested in one run, both on `dev_a` (128 problems). The pairs were regenerated with ALL
continuations drawn at `presence_penalty=0`, so the looping and productive halves come from the same sampler
and the confound noted above is gone: **228 matched pairs from 300 prefixes**, against 249 with mixed
penalties. The yield barely moved, so matched pairs are plentiful within one sampler.

Layers were then chosen by a **pre-registered** criterion: solve the LP at every layer and rank by whether it
satisfies its constraint on held-out looping tokens, tie-broken by collateral damage per unit of suppression.
No evaluation, about a minute per layer.

| arm | layers | dev_a | gsm8k | math | correct |
|---|---|---|---|---|---|
| `derived_k8` | 9, 21 (derived) | **69.5** | 75.0 | 64.1 | 89/128 |
| `inherited_k8` | 15, 19 (Astra's) | 68.0 | 75.0 | 60.9 | 87/128 |
| `pairdir_k8` (earlier run, mixed-penalty pairs) | 15, 19 | 64.1 | 75.0 | 53.1 | 82/128 |
| `shift_k8` (author, agent direction) | 15, 19 | 62.5 | 71.9 | 53.1 | 80/128 |

**1. The within-sampler direction is BETTER, not merely cleaner.** On identical layers 15/19 it scores 68.0
against 64.1 for the confounded version — 87 vs 82 correct. Removing the confound cost nothing and gained
about four points.

**2. Layer choice barely matters, and the ranking has little predictive power.** Derived minus inherited is
+1.5 points (89 vs 87 of 128), **0.27 sigma** — noise. The ranking placed Astra's layers 16th and 19th of 19,
yet they perform within noise of the top pair. The criterion is not shown to work; what the experiment does
show is that the outcome is insensitive to layer choice over this range. That is the more useful fact: it
means transferring to a new model does not require getting the layers right.

**3. A correction to how this was first read.** The derived arm was initially compared against `pairdir_k8`
(64.1) and the +5.4 difference attributed to layers. That was wrong — `pairdir_k8` used a DIFFERENT
direction. Against the matched control the layer contribution is +1.5. A direction improvement had been
mis-attributed to layer choice; only the control separated them.

Best arm overall: **69.5 against the agent direction's 62.5** — 89 vs 80 of 128, +7.0 points, pooled SE 5.9,
**1.18 sigma**. The largest margin in this work and still not significant.

## Scope: Qwen3.5-2B (2026-11-13)

Run against Qwen3.5-2B, halted at a deliberate gate before any edit was built:

```
gsm8k_train, fit_gen split, 64 problems, base model, no edit
acc 70.3   cut_share 0.234   mean_tokens 12568
```

2B hits the token limit on 23% of problems against ~89% for 0.8B, so **looping diminishes sharply with
scale**. This is the same direction as Takehi et al.'s Gemma 3 270M result from the other side: editing
removes an obstacle, it cannot add an ability.

**Two corrections to how this was first written up.**

1. It is NOT true that 2B unedited beats our edited 0.8B. That compared 70.3 (gsm8k only) against 64.1 (a
   two-set mean) — the same units error as the `cut_share` one above. Like for like on gsm8k, the edited
   0.8B scores **75.0** against 2B base's 70.3.
2. **The gate threshold (cut >= 0.40) was the wrong test and stopped a run worth finishing.** Of 64
   problems, 45 are correct and 15 are cut; since a cut trace never answers, ~15 of the 19 failures are
   looping and only ~4 are genuine errors. The ceiling if looping were fully removed is therefore ~94%, and
   at the ~85% loop-suppression rate the edit achieves on 0.8B, 2B would be expected around **85-90% from
   70.3**. The right gate is accuracy lost to looping, not loop frequency.

### The rerun, with the corrected gate — the edit WAS built and it works

The paragraphs above describe the halted first attempt and are kept for the record. **The run was repeated
with the gate set on accuracy lost to looping, and an edit was built.** Everything here is the 2B's own:

- its own base traces on 192 `fit_gen` problems, stamped `base traces belong to: Qwen3.5-2B` (the upstream
  repo ships the author's **0.8B** traces, and a bare file-exists check silently reuses them — see
  REPRODUCE.md);
- its own matched continuations, within-sampler at `presence_penalty=0`: **35 of 76 prefixes** produced both
  a looping and a non-looping continuation;
- its own per-layer directions and its own LP solve (`L15 delta=1.284`, `L19 delta=1.353`).

**Only the two layer indices (15, 19) were inherited from the 0.8B work.** Nothing else was reused.

| arm `fast2b_k8` | gsm8k | math | mean | `cut_share` |
|---|---|---|---|---|
| base | — | — | 72.7 | 0.31 |
| **edited** | **90.6** | **82.8** | **86.7** | 0.06 / 0.14 |

**dev_a maths: 72.7 -> 86.7, +14.1 points, 2.80 sigma.**
**IFEval (541, prompt-level strict): 61.4 -> 77.1, +15.7 points, 5.60 sigma.** Loose 79.7, inst-strict 81.1,
inst-loose 83.0, `cut_share` 0.10.

**This is the better generality evidence in this work, not the 0.8B result.** A different model, its own
traces, its own pairs, its own direction — and the method still works. That supports the claim that the
*construction* transfers, rather than that it fitted one checkpoint.

The scale observation above still stands and is the boundary: looping diminishes sharply with size
(~89% of traces at 0.8B, ~31% at 2B), so the headroom shrinks as the model improves.

## What would make this publishable

The parity claim is replicated but each split is individually under-powered (0.57 σ and 0.62 σ). To convert
"indistinguishable from the agent" into a quantitative equivalence claim:

1. Several seeds per arm (currently n=1 sample per problem, seed 0) — the cheapest variance reduction.
2. The held-out `dev_b` split, untouched so far, as a clean third replication.
3. The author's other benchmarks (ifeval, math500) where he has published numbers for the same arms.
4. More pairs — 249 came from 300 prefixes of a 451-trace pool; the pool itself can be grown.

## Where is this worth running? A `cut_share` survey (2026-11-13)

Every result above is consistent with one rule: **the gain tracks how much accuracy the unedited model was
losing to looping in the first place.** The edit removes an obstacle, so its ceiling is however much the
obstacle was costing.

| | base `cut_share` | gain from the edit |
|---|---|---|
| 0.8B, maths | 0.89 | +58.6 |
| 2B, IFEval | 0.29 | +15.7 (5.60 σ) |
| 2B, maths | 0.26 | +14.1 (2.80 σ) |
| 0.8B, MedQA (multiple choice) | 0.17 | +5.5 (1.76 σ) |

That is monotone across everything measured. It makes `cut_share` a **cheap go/no-go test**: it needs no
edit, no pairs and no linear program, only one generation pass on the base model, and it is an upper bound
on what any loop edit could recover. `matchedpairs/cut_survey.py` in haithamb/matched-pairs runs it.

Measured on **unedited Qwen3.5-0.8B**, 64 problems per benchmark, the same sampling protocol as every other
number here (`temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768`). Nothing is
graded — `cut_share` is the share of responses that hit the token limit, so it needs no answer extraction and
cannot be biased by a faulty one. 72 minutes on one RTX 4090.

| benchmark | `cut_share` | no answer | mean tokens |
|---|---|---|---|
| MATH-500 | **0.95** | 0.95 | 31,436 |
| gsm8k | **0.91** | 0.91 | 29,952 |
| IFEval | **0.73** | 0.73 | 25,138 |
| **MedQA (USMLE), options hidden** | **0.52** | 0.48 | 19,539 |
| PubMedQA | 0.28 | 0.28 | 11,586 |
| **MedQA (USMLE), multiple choice** | **0.17** | 0.16 | 10,187 |
| MMLU-Pro (health) | 0.11 | 0.11 | 8,619 |
| MedMCQA | 0.08 | 0.05 | 6,567 |

### Multiple choice hides the problem

The two MedQA rows are the **same questions** under the same sampler. The only difference is whether the four
options are shown. Hiding them takes `cut_share` from 0.17 to 0.52 and mean length from 10,187 tokens to
19,539.

Multiple choice gives the model an exit: it can commit to a visible option and stop. Remove that and it has
to reason open-endedly, which is where the looping lives. So the +5.5 points at 1.76 σ we measured on
MedQA-MCQ was never the method running out of room — **the format capped what was recoverable at about 17
points.** On the open-ended version there is roughly three times as much available.

This cuts both ways and the second direction matters more. Multiple-choice benchmarks are the standard way
medical LMs are scored, and they systematically *understate* how much of a small model's failure is
degenerate repetition rather than missing knowledge. A method that fixes looping will look weak on them for
reasons that have nothing to do with the method.

**Caveat on acting on this.** Open-ended MedQA needs a grader that is not a letter match. A sloppy one biases
toward whichever arm answers more often — exactly the failure mode that nearly contaminated the
multiple-choice numbers (see REVIEW.md). That grader is real work, not a free swap.

**What not to run.** MedMCQA at 0.08 and MMLU-Pro health at 0.11 are below the ~8-point threshold where there
is anything to recover. A null result there would say nothing about the method.

## MedQA open-ended: the mechanism works, the score does not move (2026-11-13)

Pre-registered before any number existed (`PREREG_medqa_open.md`): two arms, same 512 rows, one factor (the
edit), graded by token-F1 against the four option texts with unanswered responses scored wrong and kept in
the denominator. A gain claimed only at >= 2 sigma.

| arm | acc | contained | answered | `cut_share` | acc given answered |
|---|---|---|---|---|---|
| base (unedited) | 6.4 | 4.5 | 55.3 | 0.46 | 11.7 |
| edited (no LLM, no training) | 9.2 | 5.5 | **99.0** | **0.01** | 9.3 |

**Effect: +2.7 points, pooled SE 1.7, 1.63 sigma, n=512. Below the pre-registered bar, so the result is
"no detectable effect".** Not a gain, by our own rule, written down in advance.

**The mechanism worked almost perfectly.** `cut_share` 0.46 -> 0.01 and the share of questions answered at
all went 55.3% -> 99.0%. The edit did exactly what it is supposed to do. The score still did not move,
and that gap is the finding.

### Why the gain was half what the cut_share rule predicted

The pre-registered control -- accuracy among responses that actually answered -- **fell**, 11.7 -> 9.3.
Decomposing (assuming the edited arm answered everything the base arm did):

```
base    answered 283 of 512, 33 correct   (11.7%)
edited  answered 507 of 512, 47 correct   ( 9.3%)
newly answered 224, of which 14 correct   ( 6.2%)
```

**The questions the model loops on are the hard ones.** It answers them at 6.2% against 11.7% for the ones
it reaches unaided. Predicting the gain from base `cut_share` and base conditional accuracy gives +5.1; the
actual is +2.7, and the difference is entirely this.

This is **selection, not damage** -- the edit is not making the model reason worse, it is making it answer
questions it was previously avoiding, and those are harder. But that distinction rests on an assumption we
cannot now check: `records.json` was not pulled before the pod was terminated, so the per-question
comparison is unavailable. A rerun would settle it.

### This refines the central claim of this work

Every earlier result supported "the gain tracks base `cut_share`". That rule is still directionally right
but it is an **upper bound, and a loose one**, because the looped questions are systematically harder than
the rest. A sharper predictor is `cut_share` weighted by the accuracy the model achieves on the recovered
subset -- which is not knowable before running the edit, but whose *sign* is predictable: always lower.

**The control was never measured on the benchmarks where this method wins.** Maths and IFEval report overall
accuracy only. Conditional accuracy may have fallen there too, with a larger gain hiding it. That is now the
most important open question about this method, and it is cheap to answer.

### MedQA is closed, both formats

| | base | edited | effect |
|---|---|---|---|
| multiple choice | 41.2 | 46.7 | +5.5, 1.76 sigma |
| options hidden | 6.4 | 9.2 | +2.7, 1.63 sigma |

Neither clears 2 sigma. The binding constraint is **not looping, it is knowledge**: the 0.8B model scores
11.7% on open-ended clinical questions even when it answers, so recovering a response is worth very little.
The obvious fix -- a bigger model -- runs into the pattern that closed ALFWorld: **loops and ability do not
overlap**. A 2B knows more medicine and loops less.

**An earlier claim here was wrong and is withdrawn.** On finding `cut_share` 0.17 (MCQ) against 0.52 (open),
this work stated the open-ended format had "roughly three times as much available". That measured headroom in
`cut_share` units without weighting by what a recovered response is worth. Open-ended has three times the
looping and roughly a third the value per fix, so the two nearly cancel: the ceilings are about +5 either
way, and the measured results (+5.5 and +2.7) are consistent with that.

## Which arm produced which number — read this before quoting any of them

Four of the five headline rows use **Astra's inherited layers 15/19**. Only one has both the direction and
the layers derived. Stated plainly because "no LLM anywhere in the pipeline" is true of that one row alone.

| result | arm | layers | direction |
|---|---|---|---|
| MATH-500 51.8 | `pairdir_k8` | 15/19 **inherited** | matched-pair, mixed-penalty (early build) |
| IFEval 62.5 (0.8B) | `pairdir_k8` | 15/19 **inherited** | same |
| **dev_a 69.5** | `derived_k8` | **9/21 derived** | matched-pair, within-sampler |
| IFEval 77.1 (2B) | `fast2b_k8` | 15/19 **inherited** | matched-pair, 2B's own traces and pairs |
| dev_a 86.7 (2B) | `fast2b_k8` | 15/19 **inherited** | same |

**Two measured facts belong next to that, and neither is an excuse.**

1. **The agent's layer choice contributes nothing detectable.** On identical directions, derived 9/21 scores
   69.5 against inherited 15/19's 68.0 — **+1.5 points, 0.27 sigma** — and the pre-registered ranking placed
   15 and 19 *last of 19 scored layers* while they performed within noise of the top pair. What was inherited
   is two integers worth 1.5 points at 0.27 sigma.
2. **The two biggest held-out numbers came from the weakest arm.** `pairdir_k8` is the early
   mixed-presence-penalty build; on the same layers the clean within-sampler direction scores 68.0 against
   its 64.1 on dev_a. So MATH-500's +42.0 and IFEval's +39.6 are **conservative** — the better direction has
   never been run on either.

## MATH-500: +42.0 points at 14.4 sigma, and the MedQA caveat does NOT replicate (2026-11-14)

Held out — the direction was fitted on `gsm8k_train`/`math_train` and never saw these problems. Barry's own
harness and grader (`wslp.eval.run_vllm --sets math500 --n 1`), unmodified, so these numbers are directly
comparable to every `dev_a` number here. Pre-registered in `docs/PREREG_math500.md`, including the reference
bars, which were read from the upstream repository **before our result existed**.

| arm | acc | `cut_share` | terminated | acc given terminated | mean tokens |
|---|---|---|---|---|---|
| base (unedited) | 9.8 | 0.91 | 45 | 100.0 | 30,454 |
| **edited (no LLM, no training)** | **51.8** | 0.43 | 286 | 90.6 | 21,762 |

**+42.0 points, pooled SE 2.9, 14.38 sigma, n=500.** The clearest result in this work. Our base of 9.8
replicates the author's shipped 7.0 within the cross-machine drift already documented here.

### Against the reference arms: parity, again

| shipped arm | what it is | acc | vs ours |
|---|---|---|---|
| `math500_realshift_k8` | the author's **heuristic, no LLM** | 47.8 (k=4) | **+4.0**, 1.26 sigma |
| `e0_astra_math500` | GPT-6 Astra's edit, **uses an LLM** | 50.0 | +1.8, 0.57 sigma |
| `e0_shift_k8_math500` | LP on **the agent's direction**, uses an LLM | 53.2 | −1.4, 0.44 sigma |

We are **ahead of the no-LLM arm** and at **parity with both LLM-derived arms**. The claim this supports is
*matches what the frontier model found without needing one, and beats the existing LLM-free alternative*.

**CORRECTION (2026-11-14).** An earlier version of this section labelled `shift_k8` (53.2) as "the author's
LLM-free fallback" and concluded that our construction loses to a simpler LLM-free method on MATH-500. That
was a misreading of the arm names and the conclusion was false. The upstream README (line 68) defines the LP
sweep's direction argument as `astra = agent's, real = heuristic`, and `scripts/e3_overlap.py` maps
`realshift_k8` to `acts_fitgen_real_eps1.0`. So `shift_k8` is the **agent-direction** arm and needs an LLM;
the LLM-free arm is `realshift_k8` at **47.8**, which we beat. The mislabelling also reached
`PREREG_math500.md` and commit `eeb75d5`; both are corrected.

Caveats that stand: `realshift_k8` is k=4 against our k=1, the +4.0 margin is 1.26 sigma rather than
decisive, and the shipped numbers come from the author's hardware — our own base arm is the controlled
comparison.

### The control this run existed for

MedQA found accuracy among answering responses **falling** under the edit, 11.7 -> 9.3, because the questions
a model loops on are the harder ones — which made "gain tracks `cut_share`" an upper bound rather than an
estimate. **That does not replicate here, and the difference is stark:**

| | rescued problems solved at | vs self-completed |
|---|---|---|
| MedQA (open) | **6.2%** | 11.7% |
| MATH-500 | **89.6%** | 100.0% |

Of 241 problems the base model never finished, the edited model solves **216**. The problems hidden behind
looping were nearly as solvable as the ones the model reached unaided — the opposite of MedQA, where they
were effectively unsolvable.

The conditional accuracy does fall, 100.0 -> 90.6, but **that is a ceiling artifact and should not be read as
damage**: the base terminated on only 45 of 500 problems and got every one right, because a 0.8B only
finishes a maths problem it has actually solved. A 100% rate on n=45 of the easiest problems cannot be
maintained while answering 286.

**Churn: 212 newly correct, 2 newly wrong, net +210.** The edit essentially never breaks a problem the base
already had.

### Where looping and ability overlap

| level | n | base | edited | delta | base cut | edited cut |
|---|---|---|---|---|---|---|
| 1 | 43 | 27.9 | 88.4 | **+60.5** | 0.74 | 0.09 |
| 2 | 90 | 15.6 | 73.3 | **+57.8** | 0.87 | 0.20 |
| 3 | 105 | 9.5 | 61.9 | **+52.4** | 0.90 | 0.35 |
| 4 | 128 | 8.6 | 47.7 | **+39.1** | 0.92 | 0.48 |
| 5 | 134 | 1.5 | 21.6 | **+20.1** | 0.99 | 0.70 |

Three candidates died today because looping and ability do not overlap (ALFWorld model sizes, MedQA formats,
AIME). **On MATH-500 they overlap at every level**, and the curve shows how the overlap decays: the gain
falls monotonically with difficulty while residual looping rises, until at level 5 the model still cuts off
on 70% of problems after the edit. Level 5 is where MATH-500 starts to look like AIME.

This also explains why the edit suppresses looping less here (0.91 -> 0.43) than on MedQA (0.46 -> 0.01) or
IFEval (0.73 -> 0.07): the residual is concentrated in problems that are genuinely too hard, where the model
has nothing to terminate with.

## GSM8K test split: +57.2 points, and ahead of the LLM-free alternative (2026-11-14)

The cell this work had never filled. Every GSM8K figure above is the 64-problem `dev_a` slice of
`gsm8k_train` — the split the method was built on. This is the **test** split, 500 problems, 1 sample,
seed 0, the author's own harness and grader. Arm: `pairdir_k8`, our earliest and weakest build.

| arm | | acc | `cut_share` |
|---|---|---|---|
| base, unedited (ours) | | 14.0 | 0.86 |
| base, unedited (his, n=1319) | | 14.1 | 0.88 |
| `realshift_k8` — his heuristic | **no LLM** | 65.4 (k=2) | 0.23 |
| `astra_p50c` — GPT-6 Astra | uses an LLM | 68.2 (k=2) | 0.07 |
| `shift_k8` — LP on the agent's direction | uses an LLM | 69.6 (k=2) | 0.05 |
| **`pairdir_k8` (ours)** | **no LLM** | **71.2** | 0.14 |

**+57.2 points over our own base, pooled SE 3.1, 18.29 sigma, n=500.**

**Our base reproduces his to 0.1 points (14.0 vs 14.1), which is what makes the cross-comparison sound** —
on MATH-500 the same comparison carried a 2.8-point base gap. Against his arms: +1.6 over `shift_k8`
(0.55 sigma), +3.0 over `astra_p50c` (1.03 sigma), and **+5.8 over the LLM-free `realshift_k8`
(1.97 sigma)** — the clearest evidence in this work that the matched-pair construction earns its complexity
against a simpler LLM-free method.

Caveats: his edited arms are k=2 and n=1319 against our k=1 and n=500 — same expectation, less noise on his
side — and 1.97 sigma is at the threshold, not past it.

### The pattern across all three benchmarks, every arm correctly labelled

| | ours (no LLM) | theirs, no LLM | their best, uses an LLM |
|---|---|---|---|
| GSM8K test (held out) | **71.2** | 65.4 | 69.6 |
| MATH-500 (held out) | 51.8 | 47.8 | **53.2** |
| IFEval (541) | **62.5** | 60.4 | 59.7 |

Ahead of the LLM-free alternative on all three; at parity with the LLM-derived arms on all three. **Every
one of these numbers comes from `pairdir_k8`, the weakest of our three builds** — the within-sampler
direction and the derived-layer arm have only ever been evaluated on the development split.

## Qwen3.5-2B, held out on both benchmarks: the LLM-free arm is never behind (2026-11-14)

The 2B edit was rebuilt from scratch because the original was never exported and its pod was terminated
(see REPRODUCE.md). The rebuild produced **both** arms, which is the first chance this work has had to score
the fully-LLM-free configuration on held-out data.

Everything below is the 2B's own: its own base traces (192 `fit_gen` problems, stamped), its own matched
continuations (**35 usable pairs of 80 prefixes** — the original run gave 35 of 76, so the pair yield
replicates), its own per-layer directions and its own LP solve. **Only the two layer indices 15/19 were ever
inherited from the agent**, and `derived_k8` does not use them: our own search chose **layers 2 and 4**.

All six cells measured on one pod, 500 problems each, 1 sample, seed 0, the author's harness and grader.

| | base | `inherited_k8` (15/19) | `derived_k8` (2/4, **no LLM anywhere**) |
|---|---|---|---|
| **GSM8K test** | 75.8 (cut 0.25) | 87.8 (cut 0.06) | **88.4** (cut 0.06) |
| **MATH-500** | 69.6 (cut 0.31) | 79.2 (cut 0.19) | **79.6** (cut 0.19) |

| effect | GSM8K test | MATH-500 |
|---|---|---|
| `inherited_k8` vs base | +12.0, **4.92 sigma** | +9.6, **3.48 sigma** |
| `derived_k8` vs base | +12.6, **5.20 sigma** | +10.0, **3.63 sigma** |
| `derived_k8` vs `inherited_k8` | +0.6, 0.29 sigma | +0.4, 0.16 sigma |

**The layer-choice question is now settled across four independent tests, all null:**

| | comparison | effect |
|---|---|---|
| 0.8B `dev_a` | 9/21 vs 15/19 | +1.5, 0.27 sigma |
| 2B `dev_a` | 2/4 vs 15/19 | +0.8, 0.17 sigma |
| 2B GSM8K test | 2/4 vs 15/19 | +0.6, 0.29 sigma |
| 2B MATH-500 | 2/4 vs 15/19 | +0.4, 0.16 sigma |

The last component taken from a language model was two integers, and deriving them costs nothing measurable.
That is what puts "no language model anywhere in the pipeline" on held-out numbers rather than on the split
the method was fitted to. Note the search chose layers **2 and 4** for the 2B against **9 and 21** for the
0.8B and the agent's **15 and 19** — structurally very different placements, all within noise of each other.

### Headroom recovered falls as the task gets harder

| | base `cut_share` | available | recovered | share | residual cut |
|---|---|---|---|---|---|
| 0.8B GSM8K test | 0.86 | ~72 pts | +57.2 | **79%** | 0.14 |
| 2B GSM8K test | 0.25 | ~25 pts | +12.6 | **50%** | 0.06 |
| 2B MATH-500 | 0.31 | ~31 pts | +10.0 | **32%** | 0.19 |

The edit removes a *smaller fraction* of truncation on harder material, not merely a smaller absolute amount,
and leaves the largest residual (0.19) on the hardest benchmark. That is consistent with the measured
composition of residual truncation: on the barely-drained `math_train` set only **10.8%** of cut samples are
looping, the rest being verbose-and-lost, which a repetition direction cannot address (see PASSK_RESULT.md
in the research notes).

### Scale beats editing, and the writeup should say so

On MATH-500 the **unedited 2B scores 69.6**, above every edited 0.8B — ours at 51.8 and the best published
0.8B arm at 53.2. Going from 0.8B to 2B is worth more on this benchmark than any weight edit at 0.8B. This
method recovers accuracy a given model is losing to a specific failure; it is not a substitute for a larger
model, and it is not distillation.

### Artefacts

`artifacts/2b/` in haithamb/matched-pairs holds `directions_2b.npz` (all 24 layers) and `layer_search_2b.json`.
The two edit safetensors are 50 MB each and are **not** committed; they rebuild from the directions file in
about 40 minutes (activation cache + LP, no traces or pairs needed).

**No external comparison exists for the 2B.** Barry's repository is Qwen3.5-0.8B only — 14 references to it,
no 2B anywhere, and Takehi et al. used the same 0.8B plus a smaller Gemma 3 270M. These are the only numbers
for this method at a second model size, which also means the harness is unvalidated here; the 0.8B is where
it reproduces someone else's baseline (14.0 against his 14.1 on GSM8K test).

## IFEval measured end to end here: 19.6 -> 62.7, and the drift quantified (2026-11-14)

The last provenance gap. Every IFEval number in this work used **the author's** base (22.9) from his
hardware, so the published +39.6 was a cross-machine comparison. Both arms were re-run on one pod.

| arm | | strict | loose | `cut_share` |
|---|---|---|---|---|
| base, unedited | **ours** | **19.6** | 20.3 | 0.76 |
| `pairdir_k8` | **ours, no LLM** | **62.7** | 65.2 | 0.07 |

**+43.1 points, pooled SE 3.0, 14.40 sigma, n=541, same prompts, same pod.**

**The edit replicates across machines; the base does not.** Our edited arm scores 62.7 here against 62.5
measured on a different pod — 0.2 points apart. Our base scores 19.6 against the author's 22.9, a drift of
**-3.3 points (1.33 sigma, within noise)**. So the effect was slightly *understated* by using his baseline:
+43.1 rather than +39.6.

### Cross-machine drift, now measured on all three benchmarks

| | ours | author's | drift |
|---|---|---|---|
| GSM8K test | 14.0 | 14.1 | −0.1 |
| MATH-500 | 9.8 | 7.0 | **+2.8** |
| IFEval | 19.6 | 22.9 | **−3.3** |

Not systematic in either direction, which is what vLLM's documented non-determinism across hardware predicts
rather than any harness difference. The practical rule: **run your own base arm.** Two of the three drifts
exceed 2.5 points, enough to matter for any claim smaller than about 8 points — which is exactly the size of
the MedQA and residual-truncation effects this work has had to argue about.

### Every cell, both models, measured here

| | GSM8K train | GSM8K test | MATH train | MATH-500 | IFEval |
|---|---|---|---|---|---|
| **0.8B** base | 10.9* | **14.0** | 10.9* | **9.8** | **19.6** |
| **0.8B** edited | 75.0* | **71.2** | 64.1* | **51.8** | **62.7** |
| **2B** base | 70.3 | **75.8** | 75.0 | **69.6** | **61.4** |
| **2B** edited | 89.1 | **88.4** | 76.6 | **79.6** | **77.1** |

Bold = measured by us; `*` = the author's shipped `dev_a` numbers, which nobody's method holds out anyway.
Train columns are n=64 (the `dev_a` slice); test columns are n=500, IFEval n=541. The 2B edited row is
`derived_k8` on the held-out columns — our direction **and** our layers, no language model anywhere.
