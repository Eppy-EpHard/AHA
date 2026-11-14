"""Read base weights, and turn rank-1 specs {layer: (u, v)} into full bf16 down_proj tensors.

Adapted from weight-space-lp (MIT).
"""
import json

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .paths import BASE_ST, EDITS, down_proj


def base_tensor(name: str) -> np.ndarray:
    """One tensor of the unedited checkpoint, as float32."""
    with safe_open(str(BASE_ST), "pt") as f:
        return f.get_tensor(name).float().numpy()


def write(name: str, rank1: dict[int, tuple[np.ndarray, np.ndarray]], note: str = "") -> dict:
    """W = bf16(W0 + u vᵀ) per layer, saved to EDITS/<name>.safetensors with bookkeeping in EDITS/<name>.json."""
    tensors, info = {}, {"name": name, "note": note, "layers": {}}
    for l, (u, v) in rank1.items():
        w0 = torch.from_numpy(base_tensor(down_proj(l)))
        w = (w0.double() + torch.outer(torch.from_numpy(u).double(), torch.from_numpy(v).double())).to(torch.bfloat16)
        w0b = w0.to(torch.bfloat16)
        changed = (w != w0b)
        tensors[down_proj(l)] = w
        info["layers"][l] = {"nnz_v": int((v != 0).sum()), "neurons": np.flatnonzero(v).tolist(),
                             "changed_weights_bf16": int(changed.sum()),
                             "changed_columns": int(changed.any(0).sum()),
                             "rel_frobenius": float((w.float() - w0b.float()).norm() / w0b.float().norm())}
    EDITS.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(EDITS / f"{name}.safetensors"))
    json.dump(info, open(EDITS / f"{name}.json", "w"), indent=1)
    return info
