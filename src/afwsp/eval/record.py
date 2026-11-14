"""What every eval saves besides its score, so a run can still be analysed once the GPU is gone.

Per sample, each eval adds loop_stats(token_ids) to its samples file, next to the text, token count, cut flag,
answer and grade: loop_onset is the loop detector's onset on the generated tokens (None when it does not loop),
dup_frac the share of 16-token windows that repeat an earlier one.

Per run, write_summary() leaves <out>/summary.json. For each set it records:
  - accuracy, and accuracy among samples that terminated;
  - cut_share, the loop detector's loop share, and loops among cut samples;
  - median loop onset, and tokens (mean, median, p90);
  - answered share, where the eval knows it.
It also records the wall time, generated tokens per second, the checkpoint and the edit it carries (its
edits.json), the sampling parameters, the GPU and package versions.
"""
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..loops import dup_mask, onset


def loop_stats(token_ids) -> dict:
    ids = np.asarray(list(token_ids), np.int64)
    if not len(ids):
        return {"loop_onset": None, "dup_frac": 0.0}
    dup = dup_mask(ids)
    return {"loop_onset": onset(dup), "dup_frac": round(float(dup.mean()), 4)}


def _mean(x):
    x = np.asarray(x, float)
    return round(float(x.mean()), 4) if len(x) else None


def stats(rows: list[dict]) -> dict:
    """rows: one dict per sample with tokens, cut, loop_onset, dup_frac and, when the eval grades, correct (0/1)
    and answered (bool)."""
    tok = np.array([r["tokens"] for r in rows], float)
    cut = np.array([bool(r["cut"]) for r in rows], bool)
    loop = np.array([r["loop_onset"] is not None for r in rows], bool)
    onsets = [r["loop_onset"] for r in rows if r["loop_onset"] is not None]
    out = {"n": len(rows), "cut_share": _mean(cut), "loop_share": _mean(loop), "loop_share_of_cut": _mean(loop[cut]),
           "loop_onset_median": float(np.median(onsets)) if onsets else None,
           "dup_frac_mean": _mean([r["dup_frac"] for r in rows]),
           "tokens_mean": round(float(tok.mean()), 1) if len(tok) else None,
           "tokens_median": float(np.median(tok)) if len(tok) else None,
           "tokens_p90": float(np.percentile(tok, 90)) if len(tok) else None}
    if rows and "correct" in rows[0]:
        cor = np.array([float(r["correct"]) for r in rows])
        out.update({"acc": _mean(cor), "acc_terminated": _mean(cor[~cut]), "n_terminated": int((~cut).sum())})
    if rows and "answered" in rows[0]:
        out["answered_share"] = _mean([bool(r["answered"]) for r in rows])
    return out


def call_rows(calls_path, tokenizer_dir) -> list[dict]:
    """One row per model call in an agentic eval, read from the logging proxy's calls.jsonl. Loop stats come from
    re-tokenizing the call's thinking and answer text; tool calls are parsed out by the server, so they are not in it."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    rows = []
    for line in open(calls_path):
        c = json.loads(line)
        if c.get("status") != 200:
            rows.append({"conv": c.get("conv"), "error": c.get("error") or c.get("status"), "tokens": 0,
                         "cut": False, "loop_onset": None, "dup_frac": 0.0, "n_tool_calls": 0})
            continue
        text = (c.get("reasoning_content") or "") + (c.get("content") or "")
        ids = tok(text, add_special_tokens=False).input_ids
        usage = c.get("usage") or {}
        rows.append({"conv": c.get("conv"), "tokens": usage.get("completion_tokens") or len(ids),
                     "cut": c.get("finish_reason") == "length", "n_tool_calls": len(c.get("tool_calls") or []),
                     "latency_s": c.get("latency_s"), **loop_stats(ids)})
    return rows


def call_stats(rows: list[dict]) -> dict:
    """stats() over calls, plus how loops and cut-offs spread over episodes (calls grouped by conversation)."""
    out = {f"calls_{k}" if k == "n" else k: v for k, v in stats(rows).items()}
    convs = {}
    for r in rows:
        convs.setdefault(r["conv"], []).append(r)
    out.update({"episodes": len(convs), "calls_per_episode_mean": _mean([len(v) for v in convs.values()]),
                "episodes_with_loop_share": _mean([any(r["loop_onset"] is not None for r in v) for v in convs.values()]),
                "episodes_with_cut_share": _mean([any(r["cut"] for r in v) for v in convs.values()]),
                "call_errors": sum("error" in r for r in rows)})
    return out


def _version(p):
    try:
        return importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_summary(out, sets: dict, ckpt, sampling: dict, t0: float, n_tokens: int, **extra) -> dict:
    seconds = time.time() - t0
    edits = Path(ckpt) / "edits.json"
    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    s = {"sets": sets, "ckpt": str(ckpt), "edit": json.load(open(edits)) if edits.exists() else None,
         "sampling": sampling, "seconds": round(seconds), "generated_tokens": int(n_tokens),
         "tokens_per_second": round(n_tokens / seconds, 1) if seconds > 0 else None,
         "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "gpu": gpu,
         "python": platform.python_version(),
         "packages": {p: _version(p) for p in ("vllm", "torch", "transformers", "math-verify", "lm-eval")}, **extra}
    json.dump(s, open(Path(out) / "summary.json", "w"), indent=1)
    return s
