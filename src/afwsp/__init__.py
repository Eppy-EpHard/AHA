"""Agent-free weight-space programming.

Removes a reasoning model's degenerate loops with a rank-1 edit to the MLP down_proj, ΔW = u vᵀ, with no LLM
agent anywhere: u comes from matched pairs of the model's own continuations, v from a linear program, and the
layers from a search over all of them.

  paths     every path, overridable per run
  loops     loop detector and token labels
  model     load the model; capture MLP inputs (h) and layer outputs (resid)
  collect   teacher-forced activation collection over labelled traces
  acts      cached activations, split by trace into fit and held-out
  lp        the linear program for v, solved by PDHG
  edits     read base weights; write rank-1 edits as full bf16 down_proj tensors
  ckpt      text-only checkpoints for vLLM, with an edit applied
  eval/     maths (GSM8K, MATH, MATH-500), IFEval, MedQA, MedQA with the options hidden, cut_share survey

Parts are adapted from Nathan Barry's weight-space-lp (MIT); see THIRD_PARTY_NOTICES.md.
"""
