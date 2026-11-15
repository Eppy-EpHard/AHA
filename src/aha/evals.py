"""Maths evaluation under the original paper's protocol: thinking on, temperature 0.6, top_p 0.95, top_k 20,
presence_penalty 1.5, up to 32,768 tokens, seed 0. The answer is the last \\boxed{} after </think>, graded with
math-verify.

  python -m aha.evals <checkpoint> <out_dir> --sets math500 [--n 1] [--split dev_a] [--limit 500]

Sets: math500, gsm8k (the test split), and the practice sets gsm8k_train and math_train; data/splits.json names
their subsets. Every answer is saved with the loop detector's verdict on its tokens. summary.json records accuracy,
cut-offs, loops and tokens per set.

Grading is ported from the original paper's mathqa.py (MIT), via weight-space-lp (MIT).
"""
import argparse
import functools
import gzip
import importlib.metadata
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .loops import dup_mask, onset

ROOT = Path(__file__).resolve().parents[2]
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768, "seed": 0}
SAMPLES = {"math500": 4, "gsm8k": 2, "gsm8k_train": 1, "math_train": 1}     # per problem, unless --n says otherwise
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


# ------------------------------------------------------------------------------------------------- problems

def load_problems(name):
    """A set as a list of {id, problem, answer}, matching the original paper's evaluate.py."""
    if name in ("gsm8k_train", "math_train"):
        return [json.loads(line) for line in open(ROOT / "data/practice" / f"{name}.jsonl")]
    from datasets import load_dataset
    if name == "math500":
        return [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
                for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    if name == "gsm8k":
        return [{"id": f"gsm8k_test_{i}", "problem": r["question"],
                 "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))]
    raise KeyError(name)


def split_ids(name):
    return set(json.load(open(ROOT / "data/splits.json"))[name])


def chat_prompt(tokenizer, problem):
    return tokenizer.apply_chat_template([{"role": "user", "content": problem + "\n" + INSTRUCTION}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)


# ------------------------------------------------------------------------------------------------- grading

def answer_text(text):
    """What follows the thinking: everything after the last </think>."""
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def boxed_content(text, start):
    key = "\\boxed" if text.startswith("\\boxed", start) else "\\fbox"
    rest = text[start + len(key):]
    stripped = rest.lstrip()
    if stripped.startswith("{"):
        depth = 0
        for i, ch in enumerate(stripped):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return stripped[1:i].strip() or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def extract_answer(text):
    """The last \\boxed{} (or \\fbox{}) after the thinking, or None."""
    text = answer_text(text)
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
        if start < 0:
            return None
    return boxed_content(text, start)


@functools.lru_cache(maxsize=1 << 16)
def parse_math(expression, timeout):
    import math_verify
    return math_verify.parse("\\boxed{" + expression + "}", parsing_timeout=timeout)


def is_correct(prediction, gold):
    if prediction is None or gold is None:
        return False
    import math_verify
    timeout = 5 if threading.current_thread() is threading.main_thread() else None
    gold_parsed, prediction_parsed = parse_math(gold, timeout), parse_math(prediction, timeout)
    if not gold_parsed or not prediction_parsed:
        return False
    try:
        return bool(math_verify.verify(gold_parsed, prediction_parsed, timeout_seconds=timeout))
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------- statistics

def loop_stats(token_ids):
    """The loop detector on generated tokens: loop_onset (None when the answer does not loop) and dup_frac, the
    share of 16-token windows that repeat an earlier one."""
    ids = np.asarray(list(token_ids), np.int64)
    if not len(ids):
        return {"loop_onset": None, "dup_frac": 0.0}
    dup = dup_mask(ids)
    return {"loop_onset": onset(dup), "dup_frac": round(float(dup.mean()), 4)}


def mean(values):
    values = np.asarray(values, float)
    return round(float(values.mean()), 4) if len(values) else None


def summarize(answers):
    """Statistics over answers, each a dict with tokens, cut, loop_onset, dup_frac and, when graded, correct and
    answered."""
    tokens = np.array([a["tokens"] for a in answers], float)
    cut = np.array([bool(a["cut"]) for a in answers], bool)
    loops = np.array([a["loop_onset"] is not None for a in answers], bool)
    onsets = [a["loop_onset"] for a in answers if a["loop_onset"] is not None]
    stats = {"n": len(answers), "cut_share": mean(cut), "loop_share": mean(loops), "loop_share_of_cut": mean(loops[cut]),
             "loop_onset_median": float(np.median(onsets)) if onsets else None,
             "dup_frac_mean": mean([a["dup_frac"] for a in answers]),
             "tokens_mean": round(float(tokens.mean()), 1) if len(tokens) else None,
             "tokens_median": float(np.median(tokens)) if len(tokens) else None,
             "tokens_p90": float(np.percentile(tokens, 90)) if len(tokens) else None}
    if answers and "correct" in answers[0]:
        correct = np.array([float(a["correct"]) for a in answers])
        stats.update({"acc": mean(correct), "acc_terminated": mean(correct[~cut]), "n_terminated": int((~cut).sum())})
    if answers and "answered" in answers[0]:
        stats["answered_share"] = mean([bool(a["answered"]) for a in answers])
    return stats


def write_summary(out_dir, sets, checkpoint, sampling, start, generated_tokens, **extra):
    """summary.json: the statistics per set, plus what produced them."""
    import torch
    seconds = time.time() - start
    edit = Path(checkpoint) / "edits.json"
    summary = {"sets": sets, "checkpoint": str(checkpoint), "edit": json.load(open(edit)) if edit.exists() else None,
               "sampling": sampling, "seconds": round(seconds), "generated_tokens": int(generated_tokens),
               "tokens_per_second": round(generated_tokens / seconds, 1) if seconds else None,
               "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "versions": {p: version(p) for p in ("vllm", "torch", "transformers", "math-verify")}, **extra}
    json.dump(summary, open(Path(out_dir) / "summary.json", "w"), indent=1)


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


# ------------------------------------------------------------------------------------------------- the run

def evaluate(checkpoint, out_dir, sets, n=None, split=None, limit=None, max_tokens=32768, gpu_memory=0.88):
    """Generate with vLLM, grade, and write per set <set>_samples.jsonl.gz (every answer) and
    <set>_per_problem.json, plus scores.json and summary.json."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sampling = dict(SAMPLING, max_tokens=max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    llm = LLM(model=str(checkpoint), dtype="bfloat16", gpu_memory_utilization=gpu_memory,
              max_model_len=max_tokens + 2048, seed=0)
    keep = split_ids(split) if split else None
    scores, summary, start, generated = {"protocol": {"sampling": sampling, "checkpoint": str(checkpoint)}}, {}, time.time(), 0
    for name in sets:
        set_start = time.time()
        problems = load_problems(name)
        if keep is not None:
            problems = [p for p in problems if p["id"] in keep]
        if limit:
            problems = problems[:limit]
        samples = n or SAMPLES[name]
        outputs = llm.generate([chat_prompt(tokenizer, p["problem"]) for p in problems],
                               SamplingParams(n=samples, **sampling))
        per_problem, answers = [], []
        with gzip.open(out / f"{name}_samples.jsonl.gz", "wt") as f:
            for problem, output in zip(problems, outputs):
                correct = []
                for sample in output.outputs:
                    prediction = extract_answer(sample.text)
                    good = is_correct(prediction, problem["answer"])
                    cut = sample.finish_reason == "length"
                    loop = loop_stats(sample.token_ids)
                    correct.append(int(good))
                    answers.append({"tokens": len(sample.token_ids), "cut": cut, "correct": int(good),
                                    "answered": prediction is not None, **loop})
                    f.write(json.dumps({"id": problem["id"], "answer": problem["answer"], "pred": prediction,
                                        "correct": good, "cut": cut, "tokens": len(sample.token_ids), **loop,
                                        "text": sample.text}) + "\n")
                per_problem.append({"id": problem["id"], "correct": correct})
        json.dump(per_problem, open(out / f"{name}_per_problem.json", "w"))
        stats = summarize(answers)
        generated += sum(a["tokens"] for a in answers)
        summary[name] = stats
        scores[name] = {"acc": stats["acc"], "cut_share": stats["cut_share"], "loop_share": stats["loop_share"],
                        "mean_tokens": stats["tokens_mean"], "n_problems": len(problems), "samples": samples,
                        "total": len(answers), "seconds": round(time.time() - set_start)}
        print(name, json.dumps(scores[name]), flush=True)
        json.dump(scores, open(out / "scores.json", "w"), indent=1)
        write_summary(out, summary, checkpoint, sampling, start, generated)    # after every set: a crash keeps what ran


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument("--sets", default="math500", help="comma-separated: math500, gsm8k, gsm8k_train, math_train")
    parser.add_argument("--n", type=int, help="samples per problem (default: the protocol's)")
    parser.add_argument("--split", help="keep only this split of data/splits.json (e.g. fit_gen, dev_a)")
    parser.add_argument("--limit", type=int, help="first N problems of each set")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory", type=float, default=0.88,
                        help="share of the GPU's total memory vLLM may take; on a shared GPU, below free / total")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.out_dir, args.sets.split(","), args.n, args.split, args.limit, args.max_tokens,
             args.gpu_memory)


if __name__ == "__main__":
    main()
