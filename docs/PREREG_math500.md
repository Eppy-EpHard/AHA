# Pre-registration: MATH-500, Qwen3.5-0.8B

Written 2026-11-13 **before any number exists**.

## Why this benchmark

Every headline number in this work is on `dev_a`: 128 problems drawn from `gsm8k_train + math_train`, the
distribution the direction was **fitted** on. MATH-500 is a standard public benchmark the direction has
never seen, from the same family as `math_train`, with `cut_share` **0.95** measured on the unedited model.

Unlike AIME, headroom is already demonstrated: on `dev_a` the edit takes this model from **10.9 to 69.5**.
The ability exists and looping is what hides it. AIME failed because a 0.8B cannot do olympiad problems at
all (360 of 360 generations never terminated).

## Arms

Two, differing in **one factor**: the edit.

- `base` — unedited Qwen3.5-0.8B.
- `pairdir_k8` — the saved matched-pair edit, built only from this model's `fit_gen` maths traces. Byte
  identical to the one that scored 69.5 on `dev_a` and 46.7 on MedQA.

Same checkpoint path, sampler, prompts, seed, problems.

## Protocol

**The author's own harness, unmodified**: `wslp.eval.run_vllm --sets math500 --n 1`. That is the same code
path, prompt and grader behind every `dev_a` number here, so the results are directly comparable.

- 500 problems, 1 sample each, `temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5,
  max_tokens 32768, seed 0`.
- Grading: last `\boxed{}` **after the last `</think>`**, math-verify equivalence. A response with no such
  box is wrong and stays in the denominator.

## Primary endpoint

**Accuracy difference, edited minus base, on all 500, with pooled-SE sigma.** Claimed only at **>= 2 sigma**;
below that the words are "no detectable effect", whatever the sign. n stated with every number.

## Secondary, reported whether or not they help

1. **`cut_share` per arm** — the mechanism check.
2. **Accuracy among responses that terminated** (`acc | terminated`). **This is the point of the run.**
   MedQA showed the edit took answering from 55% to 99% while conditional accuracy FELL, 11.7 -> 9.3,
   because the questions a model loops on are the harder ones. That control has **never been measured on a
   benchmark where this method wins**; maths and IFEval report overall accuracy only. If it falls here too,
   every headline number in this work needs a caveat, and that is a more important finding than the
   headline.
3. **Per difficulty level (1-5).** MATH-500 ships levels. Three times today a candidate died because
   looping and ability do not overlap (ALFWorld sizes, MedQA formats, AIME). Scoring by level maps exactly
   where they do overlap for this model, turning one run into a curve. Levels are joined onto the results
   by `unique_id` after the run; they play no part in generation.

## Predictions, recorded in advance

- Base accuracy on MATH-500 will be **low**, in the 5-15% range, because `cut_share` is 0.95.
- The edit will cut `cut_share` to **under 0.10**, as it did on MedQA (0.46 -> 0.01) and IFEval (0.73 -> 0.07).
- The gain will be **large and clear the 2 sigma bar**, unlike both MedQA formats.
- `acc | terminated` will **fall**, as it did on MedQA, because recovered problems are harder. The gain
  should survive it, which is the difference from MedQA.
- The gain will be **concentrated in levels 1-3** and near zero at level 5.

## Known limitations, stated in advance

- One sample per problem, one seed.
- Qwen3.5-0.8B only.
- MATH-500 is from the same family as `math_train`, so it is held out but not out of distribution. It tests
  generalisation across problems, not across domains; IFEval already covers the latter.

## Reference numbers, read BEFORE our result existed (added 2026-11-13, pre-result)

The upstream repository ships the author's own MATH-500 results for this model, same harness, n=500. They
were read while our run was still in setup and are recorded here so the comparison cannot be chosen after
the fact.

| shipped arm | acc | `cut_share` | what it is |
|---|---|---|---|
| `e0_base_math500` | **7.0** | 0.93 | unedited |
| `e0_astra_math500` | **50.0** | 0.31 | the LLM-found direction |
| `e0_shift_k8_math500` | **53.2** | 0.37 | the author's **LLM-free** fallback |
| `math500s1_astra_p50c` | 51.3 | 0.31 | same, 3 samples |
| `math500s1_shift_k8` | 51.2 | 0.40 | same, 3 samples |

**Two things this changes, both stated in advance.**

1. **The prediction of a 5-15% base accuracy is confirmed at 7.0**, and his `cut_share` 0.93 matches the
   0.95 we measured independently. The setup replicates.

2. **The bar here is 53.2, not 50.0.** On `dev_a` the ordering is ours 69.5 > the agent's 62.5 > published
   `astra_p50c` 59.4, so the matched-pair direction beats the simple LLM-free fallback. On MATH-500 the
   author's **LLM-free fallback (53.2) BEATS his LLM direction (50.0)**. If our edit lands below 53.2, then
   on this benchmark our construction is worse than a much simpler LLM-free method, and that is a result we
   report plainly rather than comparing ourselves only to the 50.0 that flatters us.

So the honest targets, fixed now:

- vs **base 7.0** — a large gain is expected and is the weakest claim available.
- vs **astra 50.0** — parity would match the headline claim of this work.
- vs **shift_k8 53.2** — the real test. Below it, the matched-pair construction is not earning its
  complexity on this benchmark.

Comparability caveat: these shipped numbers come from the author's runs on his hardware. The vLLM engine is
not bitwise deterministic across GPUs, and we measured `cut_share` drift of several points between machines
on identical settings. Our own base arm is run fresh in this experiment and is the controlled comparison;
the shipped numbers are context, not a substitute for it.

---

## CORRECTION, 2026-11-14 (after the result)

**The arm labels below are wrong and the conclusion drawn from them is false.** `e0_shift_k8_math500`
(53.2) is NOT the author's LLM-free fallback — it is the LP applied to **the agent's direction**, which
needs an LLM. The upstream README line 68 defines the sweep argument as `astra = agent's, real = heuristic`.

The author's actual LLM-free arm on MATH-500 is `math500_realshift_k8` = **47.8** (k=4).

So "the bar here is 53.2, not 50.0" and "his LLM-free fallback BEATS his LLM direction" are both wrong.
Corrected: our 51.8 is **+4.0 ahead of the no-LLM arm** (1.26 sigma) and at parity with both LLM arms
(−1.4 and +1.8, neither significant). Everything else in this pre-registration — the protocol, the
endpoints, the 2 sigma bar, the predictions — stands as written and was not affected.
