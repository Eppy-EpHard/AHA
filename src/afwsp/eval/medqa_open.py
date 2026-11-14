"""Open-ended MedQA (USMLE): the same questions as the multiple-choice run, with the options HIDDEN.

The cut_share survey found MedQA cuts off on 17% of responses as multiple choice and 52% on the same
questions with the options removed. A visible option list gives the model somewhere to stop; without it the
model has to reason open-endedly, which is where the looping lives. So the +5.5 points measured on
MedQA-MCQ was capped by the FORMAT at ~17 points of headroom, not by the method. This measures the version
where ~52 points are nominally available.

Grading is fixed in advance in PREREG_medqa_open.md and must not be altered after seeing a number.

  - no `</think>`           -> no answer was produced -> INCORRECT, kept in the denominator
  - else take the text after the last `</think>`, prefer an explicit "answer is X", else the last line
  - normalise (lowercase, strip punctuation, drop articles), token-F1 against all four option TEXTS
  - correct iff the gold option is the unique argmax and its F1 >= 0.40

Unanswered responses are scored incorrect and NOT dropped. Dropping them is the exact bias that corrupted
the first MedQA extractor: it discards the failures of whichever arm loops more, which flatters this method.

The grader self-tests on the real rows before any generation happens -- gold text must score >= 0.95,
distractor text <= 0.05, an empty response exactly 0 -- and the run aborts if it does not.

  MED_N=512 python -m afwsp.eval.medqa_open <ckpt> <outdir>
"""
import gzip
import json
import os
import re
import string
import sys
import time

from . import record

F1_FLOOR = 0.40

INSTR = ("Answer the following question. "
         "Give your final answer as a short phrase at the end, as 'The answer is ...'.")

_ART = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_STATED = re.compile(r"(?:final\s+answer|answer|diagnosis)\s*(?:is|:|=)\s*(.+)", re.I | re.S)


def norm(s):
    """SQuAD-style normalisation: lowercase, drop punctuation and articles, collapse whitespace."""
    s = s.lower().translate(_PUNCT)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def f1(pred, gold):
    """Token-overlap F1 between two normalised strings."""
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return 0.0
    common = 0
    gc = list(g)
    for t in p:
        if t in gc:
            gc.remove(t)
            common += 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_span(text):
    """The model's actual answer, or None if it never produced one.

    A response with no `</think>` never left its reasoning block, so it has produced no answer at all. That
    is a failure, not a missing datum -- the caller scores it incorrect rather than dropping it.
    """
    if "</think>" not in text:
        return None
    tail = text.rsplit("</think>", 1)[1].strip()
    if not tail:
        return None
    m = _STATED.search(tail)
    if m:
        cand = m.group(1).strip().split("\n")[0].strip()
        if cand:
            return cand[:300]
    for line in reversed(tail.split("\n")):
        if line.strip():
            return line.strip()[:300]
    return None


def grade(text, gold, distractors):
    """-> (correct, contained, answered). Fixed in advance; see PREREG_medqa_open.md."""
    span = answer_span(text)
    if span is None:
        return False, False, False
    scores = [f1(span, o) for o in [gold] + list(distractors)]
    best = max(scores)
    correct = bool(scores[0] == best and scores.count(best) == 1 and best >= F1_FLOOR)
    ns = norm(span)
    contained = bool(norm(gold) and norm(gold) in ns
                     and not any(norm(d) and norm(d) in ns for d in distractors))
    return correct, contained, True


def selftest(rows):
    """The grader must recover gold text, reject distractor text, and score a non-answer zero.

    Run on the REAL rows before any generation. An evaluation whose grader cannot do this cannot be trusted
    to score a model, and the caller aborts rather than produce a number.
    """
    wrap = lambda s: f"reasoning</think>\nThe answer is {s}."
    g = sum(grade(wrap(r["gold"]), r["gold"], r["distractors"])[0] for r in rows) / len(rows)
    d = sum(grade(wrap(r["distractors"][0]), r["gold"], r["distractors"])[0] for r in rows) / len(rows)
    e = sum(grade("reasoning with no closing tag", r["gold"], r["distractors"])[0] for r in rows) / len(rows)
    print(f"  grader self-test: gold={g:.3f} (need >=0.95)  distractor={d:.3f} (need <=0.05)  "
          f"unterminated={e:.3f} (need 0.00)", flush=True)
    return g >= 0.95 and d <= 0.05 and e == 0.0


def load_rows(n):
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    rows = []
    for r in list(ds)[:n]:
        opts = r["options"]
        texts = [opts[k] for k in sorted(opts)] if isinstance(opts, dict) else list(opts)
        gold = r["answer"]
        distractors = [t for t in texts if t != gold]
        if not gold or len(distractors) != len(texts) - 1:
            continue                      # malformed row: gold absent from options, or duplicated
        rows.append({"q": r["question"], "gold": gold, "distractors": distractors})
    return rows


def main():
    ckpt, outdir = sys.argv[1], sys.argv[2]
    n = int(os.environ.get("MED_N", "512"))
    max_tok = int(os.environ.get("MED_MAX_TOKENS", "32768"))
    os.makedirs(outdir, exist_ok=True)

    rows = load_rows(n)
    print(f"loaded {len(rows)} MedQA rows (options hidden)", flush=True)
    if not selftest(rows):
        raise SystemExit("GRADER-SELFTEST-FAILED")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(ckpt)
    llm = LLM(model=ckpt, dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)),
              max_model_len=max_tok + 2048, seed=0)
    sp = SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                        max_tokens=max_tok, seed=0)

    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{r['q']}\n{INSTR}"}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True)
               for r in rows]
    t0 = time.time()
    outs = llm.generate(prompts, sp)

    nc = ncont = nans = ncut = 0
    toks = 0
    recs, texts = [], []
    for r, o in zip(rows, outs):
        t = o.outputs[0].text
        c, cont, ans = grade(t, r["gold"], r["distractors"])
        cut = o.outputs[0].finish_reason == "length"
        nc += c; ncont += cont; nans += ans; ncut += cut
        toks += len(o.outputs[0].token_ids)
        recs.append({"gold": r["gold"], "span": answer_span(t), "correct": c, "answered": ans, "cut": cut,
                     "tokens": len(o.outputs[0].token_ids), **record.loop_stats(o.outputs[0].token_ids)})
        texts.append(t)

    N = len(rows)
    res = {"n": N, "acc": nc / N, "contained": ncont / N, "answered_share": nans / N,
           "cut_share": ncut / N, "mean_tokens": toks / N,
           # the control: among responses that DID answer, is the reasoning as good?
           "acc_given_answered": (nc / nans) if nans else None}
    summary = {"medqa_open": record.stats(recs)}
    res["loop_share"] = summary["medqa_open"]["loop_share"]
    json.dump({"medqa_open": res}, open(os.path.join(outdir, "scores.json"), "w"), indent=1)
    json.dump(recs, open(os.path.join(outdir, "records.json"), "w"), indent=1)
    with gzip.open(os.path.join(outdir, "medqa_open_samples.jsonl.gz"), "wt") as f:
        for i, (rec, t) in enumerate(zip(recs, texts)):
            f.write(json.dumps({"row": i, **rec, "text": t}) + "\n")
    record.write_summary(outdir, summary, ckpt, {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                                                 "presence_penalty": 1.5, "max_tokens": max_tok, "seed": 0}, t0, toks)
    print(f"  acc={100*res['acc']:.1f}  contained={100*res['contained']:.1f}  "
          f"answered={100*res['answered_share']:.1f}  cut={res['cut_share']:.2f}  "
          f"acc|answered={100*res['acc_given_answered']:.1f}  n={N}", flush=True)


if __name__ == "__main__":
    main()
