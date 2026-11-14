"""Evaluations, all under the original paper's think35 sampling (thinking on, T 0.6, top_p 0.95, top_k 20,
presence_penalty 1.5, 32,768 new tokens, seed 0), each runnable as `python -m afwsp.eval.<name>`:

  maths        GSM8K / MATH practice splits and MATH-500, last \\boxed{} graded with math-verify
  ifeval       IFEval, scored by lm-eval's port of the official checkers
  medqa        MedQA (USMLE) multiple choice
  medqa_open   MedQA with the options hidden, graded against the option texts
  cut_survey   cut_share across benchmarks on an unedited model: the go/no-go test before building an edit
"""
