"""Load the model and capture each layer's MLP down_proj input and output with hooks.

Adapted from weight-space-lp (MIT).
"""
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import BASE_MODEL


def device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load(dtype=torch.bfloat16, dev: str | None = None):
    dev = dev or device()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(dev).eval()
    return tok, model


def layers(model):
    m = model.model
    return (m.language_model if hasattr(m, "language_model") else m).layers


@contextmanager
def capture(model, layer_ids, what=("h", "resid")):
    """Record per-layer down_proj input ('h', [T, d_mlp]) and layer output ('resid', [T, d_model]) for batch 1."""
    store = {l: {} for l in layer_ids}
    hooks = []
    L = layers(model)
    for l in layer_ids:
        if "h" in what:
            hooks.append(L[l].mlp.down_proj.register_forward_pre_hook(
                lambda mod, args, l=l: store[l].__setitem__("h", args[0][0].detach())))
        if "resid" in what:
            hooks.append(L[l].register_forward_hook(
                lambda mod, args, out, l=l: store[l].__setitem__("resid", (out[0] if isinstance(out, tuple) else out)[0].detach())))
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()
