# AHA

**Who needs an LLM to program an LLM?** AHA is self-supervised weight-space programming: weight-space programming
with no external language model in it at all.

Small reasoning models often repeat themselves until they run out of tokens and never answer. A rank-one change to two
weight matrices stops this. Until now, finding that change has needed a frontier AI agent with read and write access
to the weights. AHA builds the same kind of edit from the model's own samples and a linear program: no agent, no
gradient step, no training run.

**Blog post:** [*Who Needs an LLM to Program an LLM? Introducing AHA!*](https://haithamb.github.io/aha/) by Ahmed
Khamis, Rasul Tutunov and Haitham Bou Ammar, November 2026.

---

## Results

Every number here is held out:
- The edit is built only from GSM8K-train and MATH-train traces.
- IFEval (instruction following, no maths) never enters the pipeline.
- One sample per problem at seed 0, with Barry's evaluation harness and grader.
- The unedited and edited models were measured on the same machine.

Each effect is given in standard errors (s.e.): the gap divided by its pooled standard error.

| unedited → edited | GSM8K test (500) | MATH-500 | IFEval (541) |
|---|---|---|---|
| **Qwen3.5-0.8B** | 14.0 → **71.2** (+57.2, 18.3 s.e.) | 9.8 → **51.8** (+42.0, 14.4 s.e.) | 19.6 → **62.7** (+43.1, 14.4 s.e.) |
| **Qwen3.5-2B** | 75.8 → **88.4** (+12.6, 5.2 s.e.) | 69.6 → **79.6** (+10.0, 3.6 s.e.) | 61.4 → **77.1** (+15.7, 5.6 s.e.) |

Where each number comes from:
- **Direction:** every cell uses the matched-pair direction. The 0.8B cells use the first version of it, built from
  continuations sampled at mixed presence penalties.
- **Layers:** the 2B's GSM8K and MATH-500 edits also use the layers our search chose (2 and 4), so no language model
  is involved in them at all. The other four cells edit layers 15 and 19, inherited from the agent.

On the 0.8B:
- AHA's direction is level with the agent's own edit: the largest gap is 3.0 points, about 1 s.e.
- It is ahead of Barry's LLM-free difference of means on all three benchmarks, by 2.3 to 5.8 points.

That is parity with the agent, not a win. [docs/RESULTS.md](docs/RESULTS.md) has every number with its run, its
standard error and its caveats, including everything that did not work.

---

## How it works

The edit adds a rank-one term to the MLP's `down_proj` in two layers:

```
W' = W + u vᵀ,   so that   W' h_t = W h_t + (v · h_t) u
```

At every token the layer now writes one extra term along a fixed direction `u`. Its strength, `v · h_t`, depends on
what the hidden units are doing at that token:
- `u` decides where the push points.
- `v` decides which units trigger it and how hard.

With 8 non-zero entries in `v`, 8 columns of `down_proj` change: at most 8,192 weights per layer in the 0.8B. The edit
is Takehi et al.'s and the linear program is Barry's. What is new is that all three inputs (the direction, the
neurons and the layers) now come from the model's own output.

**1. The direction `u`: matched pairs.**
1. Take a base answer that loops, and find its loop onset with Barry's detector.
2. Cut the answer 8 tokens before the onset. Sample 8 continuations of up to 3,072 tokens from that identical prefix.
3. Keep the prefix if at least one continuation loops and at least one does not. Pair the branch that starts looping
   earliest with the productive branch that runs longest.
4. Run both branches through the unedited model. At every layer, average the residual stream over the same 160
   tokens of each branch, starting 64 tokens after the looping branch's onset, and subtract.
5. `u` is the normalised mean of those differences over 80% of the pairs. The other 20% are held out to check it.

Both branches follow the same prefix, so the problem, the position and the length cancel and the behaviour is what is
left. Nobody labels anything: the detector finds the onsets, and the model writes both branches itself.

**2. The neurons `v`: the linear program** (Barry's, unchanged).
- Minimise Σ|v_j| subject to two constraints, each with a penalised slack:
  - every looping token is pushed back along `u` by at least δ (`v · h_t ≤ −δ`);
  - every productive token moves by at most ε (`|v · h_t| ≤ ε`).
- Keep the 8 columns with the largest |v_j| · std(h_j), and solve again on those alone.
- ε is 1. δ is four times the gap between how much of `u` the layer writes on looping tokens and on productive ones.

**3. The layers: the layer search.**
- Solve the linear program at every layer.
- Rank the layers by hit rate: the share of held-out looping tokens pushed past −δ.
- Break ties by how far held-out productive tokens move, relative to the push.
- Edit the top two layers.

The rule was written down before any accuracy was seen, and it needs no evaluation run. Hit rates are close to 1 on
most layers, so the top pair is nearly a tie and can change between runs: one 0.8B run picked 9 and 21, another
picked 7 and 13. Four comparisons of derived layers against the agent's layers found no detectable difference (at
most +1.5 points, 0.29 s.e.).

### The sampling detail that decides whether it works

The evaluation protocol sets `presence_penalty=1.5`, which suppresses repetition. Resampling under it gave **12 loops
in 1,800 continuations**, far too few to make pairs. AHA therefore draws every continuation at
`presence_penalty=0`:
- About 60% of continuations loop, and **228 of 300 prefixes** yield a matched pair.
- Both branches of a pair come from the same sampler, so the contrast cannot pick up the sampler.

The penalty only shapes which tokens get sampled. The residuals are read by running finished text back through the
unedited model.

### Why matched pairs rather than better statistics

A whitened (Fisher) direction separates looping tokens from productive ones better than any other direction we scored:
a held-out AUC of 0.953, against 0.895 for the plain difference of means. Yet it makes by far the worst edit: 21.9 on
the development split, against 64.1 for matched pairs. Pushing it 2.5× harder still gives 22.7.

So separability does not predict edit quality. What helps is a contrast whose two sides share everything except the
behaviour.

---

## Limits

- **It removes an obstacle and adds nothing.**
  - A trace that hits the token limit is almost always scored wrong. So the most an anti-loop edit can recover is the
    share of problems the unedited model loses to the limit.
  - Measure that cut-off rate first. It takes one generation pass, with nothing graded.
  - The 2B hits the limit on 25–31% of problems and gains 10 to 16 points.
  - On clinical multiple choice, the 0.8B barely loops and gains nothing detectable.
- **Harder problems recover less.** On MATH-500 the 0.8B gains 60.5 points at level 1 and 20.1 at level 5.
- **Scale beats editing.** The unedited 2B scores 69.6 on MATH-500, more than any edited 0.8B.
- **The 0.8B held-out numbers come from our earliest build:** the mixed-penalty direction on the agent's layers.
- **Machines disagree.** vLLM is not bitwise deterministic across GPUs, so every edit is compared against an unedited
  model run on the same machine.

---

## Running it

You need a Linux machine with a GPU. One environment holds everything, vLLM included:

```bash
uv sync
bash scripts/setup.sh                                              # practice problems, Qwen3.5-0.8B and 2B
uv run python scripts/run.py configs/qwen3.5-0.8b.json --dry-run   # every step, nothing run
uv run python scripts/run.py configs/qwen3.5-0.8b.json
```

On a GPU that other jobs are using, add `"gpu_memory"` to the config. It is vLLM's share of the whole card, so set it
a little under free / total: for example, 0.22 for 20 GB free on an 80 GB H100.

`run.py` runs these steps in order:
1. A checkpoint of the base model.
2. The base model's answers to the 512 practice problems the edit is built from.
3. Matched pairs, then `u`.
4. Activations, then the linear program at every layer.
5. The edit on the top two layers, and a checkpoint of the edited model.
6. Every benchmark, for the edited model and for the base model.
7. The report.

A step that finished is skipped next time, so a crashed run resumes. `--dry-run` prints the exact command of every
step.

Everything goes in `runs/<name>/`:

| | |
|---|---|
| `report.md`, `evals.csv`, `observations.json` | the results, and what the method built: the pairs, the direction's quality per layer, the layer ranking and its neurons, the edit |
| `manifest.json`, `events.jsonl`, `gpu.csv`, `logs/` | every step's command, timing and exit code; the machine and package versions; GPU use every 30 s; each step's full output |
| `results/<arm>/<benchmark>/` | every answer with its loop statistics, the scores, `summary.json` |
| `traces/`, `pairs/`, `activations/`, `layers/`, `edits/`, `checkpoints/` | the intermediate products, step by step |

The configs score MATH-500. To score GSM8K test as well, add `"gsm8k_test"` to `"benchmarks"`. The IFEval numbers
were measured with the IFEval runner in Barry's code (`run_ifeval.py`), which this repository does not include.

Building the direction takes about 22 minutes on one RTX 4090, once the base answers exist. Generating those 512
answers comes first and takes longer: about 1¾ hours on a shared H100. Barry reports the agent took 15 minutes to find
its direction, so AHA is not faster. It removes the agent, not the time.

## Layout

```
src/aha/
  loops.py        the loop detector
  model.py        load the model, hook its layers, read its weights, build vLLM checkpoints
  direction.py    matched pairs → u
  edit.py         activations → the linear program for v → layer ranking → W + u vᵀ
  evals.py        MATH-500 and GSM8K, with loop statistics for every answer
  report.py       what a run built and how the edit did → report.md
scripts/
  run.py          the whole method from one config, every step logged
  setup.sh        the practice problems and the base models, once
configs/          qwen3.5-0.8b.json, qwen3.5-2b.json
data/splits.json  which practice problems the edit is built from
docs/             RESULTS (every number), REVIEW, REPRODUCE (the traps), pre-registrations
```

## Citation

```bibtex
@misc{khamis2026aha,
  title  = {Who Needs an LLM to Program an LLM? Introducing AHA!},
  author = {Khamis, Ahmed and Tutunov, Rasul and Bou Ammar, Haitham},
  year   = {2026},
  month  = nov,
  url    = {https://haithamb.github.io/aha/},
  note   = {Blog post}
}
```

## Credits

**Takehi, Teoh, Zhang and Khattab.** The idea and the edit are theirs:
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). They released their
agent's complete logs, which made every claim about the agent checkable. The grading in `evals.py` comes from their
code, and `scripts/setup.sh` downloads the practice problems from their logs (CC BY 4.0).

**Nathan Barry.** The linear program and the evaluation harness are his:
[Can a Linear Program Do Weight-Space Programming?](https://nathan.rs/posts/weight-space-lp/)
([code](https://github.com/nathanrs/weight-space-lp)). Several parts of `src/aha/` are adapted from his code:
- the loop detector;
- the model hooks and the checkpoint builder;
- the activation collection;
- the linear-program solver;
- the maths harness.

`data/splits.json` is his split of the practice problems.

**Timothy Chen** pointed us to the linear-program work.

**AHA's contribution** is the matched-pair direction and the layer search.

## License

MIT; see [LICENSE](LICENSE). It also carries the MIT notices of the code adapted from Barry and from Takehi et al.
