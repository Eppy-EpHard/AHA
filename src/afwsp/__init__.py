"""AHA: self-supervised weight-space programming.

Stops a reasoning model's degenerate loops with a rank-1 edit ΔW = u vᵀ to an MLP's down_proj, with no LLM anywhere:
u comes from matched pairs of the model's own continuations, v from a linear program, the layers from a search over
all of them.

  loops       the loop detector
  model       load the model, hook its layers, read its weights, build vLLM checkpoints
  direction   matched pairs -> u
  edit        activations -> the LP for v -> layer ranking -> W + u vᵀ
  evals       MATH-500 and GSM8K, with loop statistics for every answer
  report      what a run built and how each arm did

Parts are adapted from Nathan Barry's weight-space-lp (MIT); see LICENSE.
"""
