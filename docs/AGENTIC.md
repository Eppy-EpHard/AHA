# Agentic evaluation: BFCL-V4 and tau2-bench

The question is whether removing degenerate loops helps a small model act as an agent: calling tools and
holding a multi-turn conversation. Neither the original paper nor Barry's follow-up evaluated agentic tasks.
These two benchmarks come from Qwen's own model cards, so the unedited baselines are published:

| unedited, thinking mode | BFCL-V4 | TAU2-Bench |
|---|---|---|
| Qwen3.5-0.8B | 25.3 | 11.6 |
| Qwen3.5-2B | 43.6 | 48.8 |

## How it runs

```
BFCL / tau2 (their own venvs) ──> logging proxy ──> vLLM server ──> checkpoint        (all in afwsp/agentic.py)
```

- **The model.** vLLM serves the base or edited checkpoint with Qwen3.5's own tool-call parser (`qwen3_coder`)
  and reasoning parser (`qwen3`), thinking on. Sampling is the setting Qwen reports these scores under:
  temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5, up to 32,768 tokens per call. Set
  `SAMPLING` in `agentic.py` to change it.
- **The proxy.** It records every model call: the thinking, the answer, any tool calls, tokens, and whether the
  call hit the limit. From that come per-call and per-episode loop statistics, measured by the same detector as
  every other eval here.
- **BFCL** runs unmodified (bfcl-eval). Our model is registered through BFCL's own handler for self-hosted Qwen
  thinking models, in the Qwen3.5 model card's Qwen-Agent setup (`use_raw_api`: tools go to the server natively).
  `--mode nous` switches to BFCL's older Qwen3 recipe.
- **tau2-bench v1.0.1** runs unmodified on airline, retail and telecom. The customer is simulated by tau2's 1.x
  default, `gpt-4.1-2025-04-14` at temperature 0. v1.0.1 is the release comparable with the 1.x leaderboard;
  the current 2.0 release is not.

## Two things found while wiring it up, and what was done about them

**In tau2, a looping agent turn kills the conversation.** When the model thinks until its token limit, its visible
reply is empty. tau2 v1.0.1 rejects that (`AssistantMessage must have either content or tool_calls`), ends the
conversation as an *infrastructure error*, and by default retries it three times. Retrying would give a looping
model extra attempts and flatter the base. So `--max-retries 0`: every task runs once. A conversation ended this
way counts as a failure and is reported as `empty_agent_reply`. Its share is the most direct agentic measure of
what looping costs.

**BFCL's Qwen-Agent dependency imports `soundfile` without declaring it**, so BFCL does not start without it. It is
in the install line below.

## Setup, once, on the GPU machine

```bash
# BFCL
uv venv /workspace/bfclenv --python 3.11
uv pip install --python /workspace/bfclenv/bin/python bfcl-eval soundfile

# tau2-bench, the 1.x-comparable release (Python 3.12)
git clone --branch v1.0.1 https://github.com/sierra-research/tau2-bench.git /workspace/tau2-bench
cd /workspace/tau2-bench && uv sync --python 3.12 && cd -

# keys, from the environment only: never written to configs, manifests or logs
export OPENAI_API_KEY=...        # tau2's user simulator and judge
export SERPAPI_API_KEY=...       # BFCL's two web-search categories; without it they are left out
```

vLLM comes from the project's own environment (`uv sync`). The paths above match `configs/agentic_*.json`.

## Run

```bash
uv run python scripts/run.py configs/agentic_0.8b.json --dry-run
uv run python scripts/run.py configs/agentic_0.8b.json
uv run python scripts/run.py configs/agentic_2b.json      # once Haitham's 2B edit is in place
```

A benchmark can also be run alone on any checkpoint:

```bash
uv run python -m afwsp.agentic bfcl runs/agentic-0.8b/checkpoints/base out/bfcl \
  --bfcl-python /workspace/bfclenv/bin/python
uv run python -m afwsp.agentic tau2 runs/agentic-0.8b/checkpoints/base out/tau2 \
  --tau2 /workspace/tau2-bench/.venv/bin/tau2 --num-tasks 5
```

`--num-tasks 5` (tau2) and `--categories simple_python` (BFCL) make quick smoke runs. `--model-url` uses a server
that is already running instead of starting one.

## What gets saved, per arm and benchmark

`runs/<run>/results/<arm>/<benchmark>/`:

| file | contents |
|---|---|
| `summary.json` | accuracy (BFCL's official overall and per category; tau2's pass^1 per domain and averaged); how the tau2 conversations ended; per-call cut share, loop share, loops among cut calls, tokens; per-episode share with a loop or a cut-off; call errors; sampling, checkpoint and edit, GPU, versions |
| `scores.json`, `<benchmark>_per_problem.json` | the score, and per test entry / conversation correctness, for paired comparisons in `report.md` |
| `calls*.jsonl`, `server.log` | every model call and the server's log (local only, not tracked) |
| `bfcl/`, `tau2/` | each harness's own outputs (BFCL's score tables are tracked; raw results stay local) |

## Cost

Both benchmarks generate with thinking on, up to 32,768 tokens per call, so a looping base model is the slow arm.
tau2 also pays OpenAI for the simulated customer. That is roughly 280 conversations per model and arm across the
three domains, on the order of $10–20 per arm at gpt-4.1 prices. That figure is an estimate, not a measurement.
