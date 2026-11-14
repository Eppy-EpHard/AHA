"""Build an edit from an ARBITRARY set of layers, so a derived layer choice can be evaluated.

The repo's e2_make_edits_sweep.py hardcodes layers (15, 19) and a hidden size of 1024. This takes the layers
on the command line and reads the hidden size from the solution file instead.

  python scripts/make_edit_layers.py <name> <k> <layer,layer,...>
reading cache/layersearch_<tag>_L{l}_k{k}.npy written by layer_search.py; pass the same LS_TAG.
"""
import sys

import os

import numpy as np

from afwsp import edits
from afwsp.paths import CACHE, down_proj
from afwsp.edits import base_tensor

TAG = os.environ.get("LS_TAG")
if not TAG:
    raise SystemExit("set LS_TAG to the same value layer_search.py used")
name, k, layers = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
sp = {}
for l in layers:
    a = np.load(CACHE / f"layersearch_{TAG}_L{l}_k{k}.npy")
    d_model = base_tensor(down_proj(l)).shape[0]      # u lives in the residual stream, v in the MLP hidden
    sp[l] = (a[0][:d_model], a[1])
info = edits.write(name, sp, note=f"layer search k={k} layers={layers}")
print(name, {l: d["changed_weights_bf16"] for l, d in info["layers"].items()})
