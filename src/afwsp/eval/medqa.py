"""MedQA (USMLE) under the repo's think35 sampling, scored by the letter after the last </think>.

  MED_N=512 python -m afwsp.eval.medqa <ckpt> <out>

The repo's grader is maths-specific (last \\boxed{}), so this uses a small multiple-choice extractor instead.
Everything else — chat template with thinking on, temperature 0.6, top_p 0.95, top_k 20, presence_penalty
1.5, 32,768 max tokens — matches the protocol used for every other number in this work, so a MedQA score is
comparable to the gsm8k and IFEval ones in how it was produced.
"""
import gzip
import json
import os
import re
import sys
import time

from . import record

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768, "seed": 0}

# Answer extraction. The naive version of this (grep the last \b[A-E]\b out of the whole tail) is badly
# wrong on MedQA, because clinical prose is full of bare capitals that are not answers: "hepatitis B",
# "vitamin D", "type A", "class IV", "group B strep". Measured failures of that version:
#   "The answer is C. Option A is wrong..., D is wrong."   -> D   (post-hoc elimination wins)
#   "The correct choice is **D** (vitamin A toxicity)."     -> A   (markdown defeats the keyword pattern)
#   "This is classic chronic hepatitis B infection."        -> B   (a NON-ANSWER scored as an answer)
# The last one matters most: it scores unanswered responses, and a looping model produces many of them, so
# the bias runs in favour of whichever arm loops more. There is NO bare-letter fallback here: a response
# that never states a letter returns None and is marked wrong, which is the truthful outcome.

EMPH = re.compile(r"[*_`]+")
BOXED = re.compile(r"\\boxed\{\s*\(?\s*([A-E])\s*\)?\s*\}")
ONLY_LETTER = re.compile(r"^\s*\(?\s*([A-E])\s*\)?\s*[.):]?\s*$")
STATED = re.compile(r"(?:answer|option|choice)\s*(?:is|:|=)\s*\(?\s*([A-E])\b", re.I)


def pick(text: str) -> str | None:
    """The model's chosen letter, or None if it never stated one.

    The prompt asks the model to END with the letter, so later evidence outranks earlier evidence and the
    final line is consulted before the body.
    """
    if "</think>" not in text:
        return None                       # cut inside the thinking: no answer at all
    tail = EMPH.sub("", text.rsplit("</think>", 1)[1]).strip()
    if not tail:
        return None

    m = BOXED.findall(tail)               # an explicit \boxed{} wins outright
    if m:
        return m[-1].upper()

    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    for l in reversed(lines):             # a line that is nothing but a letter
        g = ONLY_LETTER.match(l)
        if g:
            return g.group(1).upper()

    if lines:                             # an answer stated in the final line
        g = STATED.findall(lines[-1])
        if g:
            return g[-1].upper()

    g = STATED.findall(tail)              # last resort: an explicit statement anywhere
    return g[-1].upper() if g else None


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    n_max = int(os.environ.get("MED_N", "512"))
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.makedirs(out, exist_ok=True)
    ds = None
    for name, cfg, split in (("GBaker/MedQA-USMLE-4-options", None, "test"),
                             ("bigbio/med_qa", "med_qa_en_source", "test")):
        try:
            ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
            print(f"loaded {name} ({len(ds)} rows)", flush=True)
            break
        except Exception as e:
            print(f"could not load {name}: {e}", flush=True)
    if ds is None:
        raise SystemExit("MEDQA-NO-DATASET")

    rows = []
    for r in list(ds)[:n_max]:
        q = r.get("question") or r.get("sent1") or ""
        opts = r.get("options") or r.get("choices")
        if isinstance(opts, dict):
            items = [(k, v) for k, v in sorted(opts.items())]
        elif isinstance(opts, list) and opts and isinstance(opts[0], dict):
            items = [(o.get("key") or o.get("label"), o.get("value") or o.get("text")) for o in opts]
        else:
            items = list(zip("ABCD", opts or []))
        gold = r.get("answer_idx") or r.get("answer")
        if gold and len(str(gold)) > 1:                       # the answer text, not a letter
            gold = next((k for k, v in items if str(v).strip() == str(gold).strip()), None)
        if not items or not gold:
            continue
        body = "\n".join(f"{k}. {v}" for k, v in items)
        rows.append({"q": f"{q}\n\n{body}", "gold": str(gold).strip().upper()[:1]})
    print(f"{len(rows)} usable questions", flush=True)

    tok = AutoTokenizer.from_pretrained(ckpt)
    INSTR = "Answer the following multiple choice question. End your response with the letter of the correct option."
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["q"] + "\n" + INSTR}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True)
               for r in rows]
    llm = LLM(model=ckpt, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)),
              max_model_len=32768 + 2048, seed=0)
    sp = SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768, seed=0)
    t0 = time.time()
    outs = llm.generate(prompts, sp)

    ok = cut = toks = 0
    per, texts = [], []
    for r, o in zip(rows, outs):
        s = o.outputs[0]
        got = pick(s.text)
        hit = int(got is not None and got == r["gold"])
        ok += hit
        cut += s.finish_reason == "length"
        toks += len(s.token_ids)
        per.append({"gold": r["gold"], "got": got, "correct": hit, "cut": s.finish_reason == "length",
                    "tokens": len(s.token_ids), "answered": got is not None, **record.loop_stats(s.token_ids)})
        texts.append(s.text)
    n = len(rows)
    res = {"medqa": {"acc": ok / n, "ok": ok, "n": n, "cut_share": cut / n,
                     "mean_tokens": toks / n, "s": round(time.time() - t0)}}
    print(json.dumps(res["medqa"]), flush=True)
    summary = {"medqa": record.stats(per)}
    res["medqa"]["loop_share"] = summary["medqa"]["loop_share"]
    json.dump(res, open(os.path.join(out, "scores.json"), "w"), indent=1)
    json.dump(per, open(os.path.join(out, "per_problem.json"), "w"))
    with gzip.open(os.path.join(out, "medqa_samples.jsonl.gz"), "wt") as f:
        for i, (p, t) in enumerate(zip(per, texts)):
            f.write(json.dumps({"row": i, **p, "text": t}) + "\n")
    record.write_summary(out, summary, ckpt, SAMPLING, t0, toks)


if __name__ == "__main__":
    main()
