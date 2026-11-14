"""Measure cut_share across benchmarks on the UNEDITED model: a go/no-go test before building any edit.

Across everything measured in this work the gain from the loop edit tracks how much accuracy looping was
costing in the first place -- 0.89 -> +58.6, 0.29 -> +15.7, 0.26 -> +14.1, 0.17 -> +5.5. So cut_share
PREDICTS whether the method is worth running, and it needs no edit, no pairs and no LP, just one generation
pass on the base model. It is an upper bound on what any loop edit could recover.

Nothing here is graded. cut_share is the share of responses that hit the token limit, which needs no answer
extraction and so cannot be biased by a faulty one.

Measured on Qwen3.5-0.8B, 64 problems each, 72 minutes on one RTX 4090 (see docs/RESULTS.md):

    math500 0.95   gsm8k 0.91   ifeval 0.73   medqa_open 0.52
    pubmedqa 0.28  medqa_mcq 0.17   mmlu_pro_health 0.11   medmcqa 0.08

medqa_mcq and medqa_open are the SAME questions; the only difference is whether the four options are shown.
0.17 against 0.52 is why multiple-choice benchmarks understate how much of a small model's failure is
degenerate repetition rather than missing knowledge.

  CUT_N=64 python -m afwsp.eval.cut_survey <ckpt> <out.json> [bench1,bench2,...]
"""
import json
import os
import sys
import time

from .record import loop_stats

INSTR_MCQ = "Answer the following multiple choice question. End your response with the letter of the correct option."
INSTR_OPEN = "Answer the following question. Give your final answer as a short phrase at the end."
INSTR_MATH = "Please reason step by step, and put your final answer within \\boxed{}."


def _opts(r):
    o = r.get("options") or r.get("choices")
    if isinstance(o, dict):
        return [(k, v) for k, v in sorted(o.items())]
    if isinstance(o, list) and o and isinstance(o[0], dict):
        return [(d.get("key") or d.get("label"), d.get("value") or d.get("text")) for d in o]
    return list(zip("ABCDEFGHIJ", o or []))


def _load(name, n):
    """Return a list of user-message strings, or raise."""
    from datasets import load_dataset
    if name == "medqa_mcq":
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        return [f"{r['question']}\n\n" + "\n".join(f"{k}. {v}" for k, v in _opts(r)) + "\n" + INSTR_MCQ
                for r in list(ds)[:n]]
    if name == "medqa_open":
        # the SAME questions with the options hidden. Multiple choice lets a model commit early; removing it
        # forces open-ended reasoning, which is where looping lives. This is the cheapest high-value probe.
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        return [f"{r['question']}\n{INSTR_OPEN}" for r in list(ds)[:n]]
    if name == "pubmedqa":
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        out = []
        for r in list(ds)[:n]:
            ctx = " ".join(r["context"]["contexts"]) if isinstance(r.get("context"), dict) else ""
            out.append(f"{ctx}\n\n{r['question']}\nAnswer yes, no, or maybe, with brief reasoning.")
        return out
    if name == "medmcqa":
        ds = load_dataset("openlifescienceai/medmcqa", split="validation")
        out = []
        for r in list(ds)[:n]:
            opts = [("A", r.get("opa")), ("B", r.get("opb")), ("C", r.get("opc")), ("D", r.get("opd"))]
            out.append(f"{r['question']}\n\n" + "\n".join(f"{k}. {v}" for k, v in opts) + "\n" + INSTR_MCQ)
        return out
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [f"{r['question']}\n{INSTR_MATH}" for r in list(ds)[:n]]
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [f"{r['problem']}\n{INSTR_MATH}" for r in list(ds)[:n]]
    if name == "ifeval":
        ds = load_dataset("google/IFEval", split="train")
        return [r["prompt"] for r in list(ds)[:n]]
    if name == "mmlu_pro_health":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        rows = [r for r in ds if r.get("category") == "health"][:n]
        return [f"{r['question']}\n\n" + "\n".join(f"{k}. {v}" for k, v in zip("ABCDEFGHIJ", r["options"]))
                + "\n" + INSTR_MCQ for r in rows]
    raise ValueError(f"unknown benchmark {name}")


def main():
    ckpt, out_path = sys.argv[1], sys.argv[2]
    names = (sys.argv[3].split(",") if len(sys.argv) > 3 else
             ["gsm8k", "math500", "ifeval", "medqa_mcq", "medqa_open", "pubmedqa", "medmcqa", "mmlu_pro_health"])
    n = int(os.environ.get("CUT_N", "64"))
    max_tok = int(os.environ.get("CUT_MAX_TOKENS", "32768"))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sets = {}
    for b in names:
        try:
            p = _load(b, n)
            if p:
                sets[b] = p
                print(f"loaded {b}: {len(p)} prompts", flush=True)
            else:
                print(f"skip {b}: no rows", flush=True)
        except Exception as e:
            print(f"skip {b}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    if not sets:
        raise SystemExit("NO-BENCHMARKS-LOADED")

    tok = AutoTokenizer.from_pretrained(ckpt)
    llm = LLM(model=ckpt, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)),
              max_model_len=max_tok + 2048, seed=0)
    # the protocol every other number in this work used
    sp = SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                        max_tokens=max_tok, seed=0)

    results = {}
    for b, raw in sets.items():
        prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) for q in raw]
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        cut = sum(o.outputs[0].finish_reason == "length" for o in outs)
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
        # a response that never closes its thinking block has produced no answer at all
        noans = sum("</think>" not in o.outputs[0].text for o in outs)
        # the loop detector on the generated tokens: how much of the cut-off is degenerate repetition
        loops = sum(loop_stats(o.outputs[0].token_ids)["loop_onset"] is not None for o in outs)
        results[b] = {"n": len(outs), "cut_share": cut / len(outs), "loop_share": loops / len(outs),
                      "mean_tokens": toks / len(outs), "no_answer_share": noans / len(outs),
                      "s": round(time.time() - t0)}
        print(f"  {b:18s} cut={results[b]['cut_share']:.2f}  loop={results[b]['loop_share']:.2f}  "
              f"no_answer={results[b]['no_answer_share']:.2f}  tokens={results[b]['mean_tokens']:.0f}  "
              f"({results[b]['s']}s)", flush=True)
        json.dump(results, open(out_path, "w"), indent=1)   # write as we go, so a timeout keeps what ran

    print("\nranked by cut_share — highest is where the loop edit has most to recover")
    print(f"{'benchmark':20s} {'cut':>6s} {'no_ans':>7s} {'tokens':>8s}   expected payoff")
    for b, r in sorted(results.items(), key=lambda kv: -kv[1]["cut_share"]):
        c = r["cut_share"]
        verdict = ("large, worth a full experiment" if c >= 0.5 else
                   "moderate, probably worth it" if c >= 0.25 else
                   "small, expect a few points" if c >= 0.08 else
                   "not worth running")
        print(f"{b:20s} {c:6.2f} {r['no_answer_share']:7.2f} {r['mean_tokens']:8.0f}   {verdict}")
    json.dump(results, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
