# Pre-registration: open-ended MedQA, Qwen3.5-0.8B

Written 2026-11-13 **before any number exists**. The survey found MedQA `cut_share` 0.17 as multiple
choice and 0.52 on the same questions with the options hidden, so the multiple-choice format capped the
earlier +5.5 result at ~17 points of headroom. This tests the open-ended version, where ~52 points are
nominally available.

Three units errors and one biased answer extractor were made and caught on 2026-11-13. This document fixes
the grader and the decision rule in advance so neither can be chosen after seeing the outcome.

## Arms

Exactly two, differing in **one factor**: the rank-1 edit.

- `base` — unedited Qwen3.5-0.8B.
- `pairdir_k8` — the SAVED matched-pair edit, built only from this model's gsm8k/math traces. No medical
  data, no gradients, no LLM. Byte-identical to the one that scored 46.7 on MedQA-MCQ.

Same checkpoint build path, same sampler, same prompts, same seed, same n. Nothing else changes.

## Protocol

- **Dataset**: `GBaker/MedQA-USMLE-4-options`, test split, first **512** rows (same rows as the MCQ run).
- **Prompt**: the question only. The four options are **not shown**. Instruction: give the final answer as a
  short phrase at the end.
- **Sampling**: `temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768, seed=0` —
  the protocol every other number in this work used, unchanged.

## Grader (fixed in advance)

The dataset supplies the correct answer's **text** and the three distractors' texts. Grading is by which
option the free-text answer is closest to, never by a letter.

1. If the response contains no `</think>`, it never finished reasoning and produced no answer → **incorrect**.
2. Otherwise take the text after the last `</think>`. Within it, prefer an explicit
   "the answer is X" / "Answer: X" / "Final answer: X"; else the last non-empty line; else the whole span.
3. Normalise: lowercase, strip punctuation, drop articles, collapse whitespace (SQuAD normalisation).
4. Score token-F1 of the answer span against each of the four option texts.
5. **Correct** iff the gold option is the unique argmax **and** its F1 ≥ 0.40. Ties → incorrect.

**Unanswered responses are scored incorrect and stay in the denominator.** They are not dropped. A cut
response has genuinely produced nothing, and dropping them is the exact bias that corrupted the first MedQA
extractor — it discards the failures of whichever arm loops more, which flatters this method.

Secondary, stricter metric reported alongside: **containment** — the normalised gold option text appears in
the normalised answer span and no distractor's does.

### Grader self-test (must pass before the evaluation runs)

Run on the real dataset rows, on the pod, before generation:

- feeding the **gold answer text** as the response must score **≥ 0.95**;
- feeding each **distractor text** must score **≤ 0.05**;
- feeding an **empty / unterminated-think** response must score **0.00**.

If any of these fail the run aborts and no accuracy number is produced.

## What counts as a result

Primary endpoint: **accuracy difference, edited minus base**, on all 512, with the pooled-SE sigma.

- A gain is claimed only at **≥ 2 σ**. Below that the word is "no detectable effect", whatever the sign.
- n is stated with every number. Both arms are scored on the same 512 rows, so no two-set / one-set
  comparison is possible.

Secondary, reported whether or not it helps the story:

- **`cut_share` per arm** — the mechanism check. If the edit does not cut looping, any accuracy change is
  not this method working.
- **Conditional accuracy among responses that actually answered.** This is the honest control. If the edit
  only recovers lost responses, conditional accuracy should be roughly **unchanged**; if it falls, the edit
  is making the model answer *worse*, and that must be reported even if overall accuracy rises.

## Known limitations, stated in advance

- Free-text answers that are correct but phrased unlike the option text are marked wrong. This penalises
  both arms identically, so it biases the absolute level down, not the difference.
- 512 rows, one sample per row, one seed.
- Qwen3.5-0.8B only. Nothing here transfers to another model without rerunning it.
- Not a clinical claim of any kind. MedQA is a licensing-exam benchmark.
