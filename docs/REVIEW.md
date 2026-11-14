# Independent correctness review, 2026-11-13

An adversarial review was run against the `wslp` source before this code was shared. What follows is what
it found, what was fixed, and what is still a known limitation. Nothing here is hidden from the results.

## Verified correct

- **Shapes, slices, pads and transposes.** `down_proj` is `[d_model, d_ff]`; `u` is built from `resid`
  (d_model) and `v` from `h` (d_ff); the zero-pad/slice round-trip in `layer_search.py` →
  `make_edit_layers.py` is exact; `torch.outer(u, v)` matches `w0`.
- **The `delta_R` length contract.** `lp.pdhg` adds `delta` to `z[:nR]` where `nR = len(H_R)`, so `delta_R`
  must match the rows passed, not the full pool. Our calls do. (`lp.solve` has the opposite convention —
  it indexes `spec.delta_R[iR]` — but nothing here calls it.)
- **Sign conventions** end to end: `u` points productive→looping, the LP enforces `v·h ≤ −delta`, the edit
  adds `u(v·h)`, so the write is pushed against `u`.
- **The held-out split inside `pair_directions.py`** is clean: `u` is fitted on `fit` only and scored on
  `ho` only.
- **The matched window** is computed once from the looping branch and required to fit *both* branches, and
  the prefix is built once outside the per-branch loop, so both branches see byte-identical context.

## Fixed

1. **MedQA answer extraction** (`med_eval.py`) was badly wrong and biased *toward* our claim. The old
   version grepped the last `[A-E]` from the whole response, so "classic chronic hepatitis B infection"
   scored **B** — i.e. unanswered responses were scored as answers, and a looping model produces many of
   them. It also lost to markdown (`**D**`) and to post-hoc elimination ("the answer is C… D is wrong" →
   **D**). Rewritten: `\boxed{}` first, then a line that is only a letter, then an explicit statement in the
   final line, and **no bare-letter fallback** — a response that never states a letter returns `None` and is
   marked wrong. 10/10 on the reviewer's failing cases. *No MedQA numbers were ever produced with the old
   extractor; the run was stopped before it scored.*
2. **Context cap** (`gen_pairs.py`) required an absolute number of surviving prefixes, which no model with
   few candidates can meet, so it fell through to the largest window — a 32,768-token context on
   Qwen3.5-2B where almost every prefix was far shorter. Now targets a fraction of what is available and
   falls back to the longest prefix.
3. **Solution filenames** (`layer_search.py`, `make_edit_layers.py`) carried only layer and k, so two runs
   at different `EPS`/`MULT` overwrote each other and the edit builder could not tell them apart. Now
   tagged with the settings, and `LS_TAG` is required rather than defaulted.
4. **Per-layer RNG** (`layer_search.py`): one generator was shared across layers, so each layer was fitted
   on a different subsample and the ranking mixed layer effect with subsample noise. Now seeded per layer.
5. **Depth detection** (`pair_directions.py`): a comment claimed the depth was read "from the model's own
   config so this ports to other sizes". It is read from `data/model_text/config.json`, which
   `fetch_exp8.sh` copies from the authors' 0.8B directory and is therefore pinned at 24 layers. The
   comment was false; it now says so and fails loudly rather than silently returning 24.
6. **`MAX_CTX` truncation** (`pair_directions.py`) was a no-op at the boundary (`pre[-0:]` is the whole
   list) and inverted past it. Unreachable at defaults; fixed anyway.
7. **Wasted capture** (`pair_directions.py`): `h` was captured for all layers and never used — several GB
   of dead tensors per forward pass. Now captures `resid` only.
8. **Dead knob** (`gen_pairs.py`): `PAIR_N_CONT` was read and ignored; the real count is
   `PAIR_N_LOOP + PAIR_N_PROD`. Removed.
9. **Dropped pairs** are now counted and reported instead of silently skipped.

## Known limitations, not fixed

- **"Productive" means "not looping here", not "answered".** A continuation is labelled productive when the
  loop detector does not fire. With `PAIR_MAX_NEW=3072` and models that need 12,000–17,000 tokens to answer,
  **100% of the branches actually chosen on the 2B run hit the token cap** (518 productive continuations,
  509 of them capped). The contrast the direction measures is therefore *looping vs not-yet-looping at the
  same point from the same prefix* — which is the behavioural contrast the method needs, and the results
  confirm it works. But it is NOT "looping vs successful reasoning", and the documentation now says so.
  A branch that is degenerate in some way the 16-gram detector misses cannot be ruled out.
- **`u` leaks into `layer_search.py`'s held-out set.** The direction is built from pairs cut out of the same
  base traces that `acts.load` splits by trace, and the two splits are uncoordinated. The LP solve is
  honestly held out; the direction is not. The layer-search diagnostics are therefore partly in-sample —
  which matters little, because that experiment is reported as a null result and its conclusion rests on
  task accuracy, not on those diagnostics. Flagged in the code.
- **Multi-pass activation caching** (`collect_layers.py`) asserts that a second pass sees identical traces,
  and the assertion does catch the realistic desync. It can still be bypassed by a pass killed between
  saving arrays and writing `meta.npz`, and `acts.load` does not check row counts. `traces.json` is written
  but never compared.
- **`e1_base_fitgen` is not stamped in code.** The upstream repo ships the authors' 0.8B traces; reusing
  them for another model is a silent, serious error. `docs/REPRODUCE.md` gives the stamp recipe, but none of
  these six scripts enforce it.
