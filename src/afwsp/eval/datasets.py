"""The maths sets (practice splits, MATH-500, GSM8K test) as lists of {id, problem, answer}, matching the
original paper's evaluate.py prepare().

Adapted from weight-space-lp (MIT). gsm8k_train and math_train are the original paper's practice files,
fetched into data/practice/ by scripts/fetch_exp8.sh; data/splits.json names the fit_gen, dev_a, dev_b and
final_practice subsets of them.
"""
import json

from ..paths import ROOT


def load(name: str) -> list[dict]:
    if name in ("gsm8k_train", "math_train"):
        return [json.loads(l) for l in open(ROOT / "data/practice" / f"{name}.jsonl")]
    from datasets import load_dataset
    if name == "math500":
        return [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
                for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    if name == "gsm8k":       # the test split; the headline runs use its first 500 (--limit 500)
        return [{"id": f"gsm8k_test_{i}", "problem": r["question"],
                 "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))]
    raise KeyError(name)
