"""Every path the pipeline reads or writes.

Each one can be overridden per run, so a second model or a second run never reads another one's files (the data
trap in docs/REPRODUCE.md). A relative override resolves against the repository root.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    if not v:
        return default
    return Path(v) if os.path.isabs(v) else ROOT / v


BASE_MODEL = env_path("AFWSP_MODEL", ROOT / "models" / "Qwen3.5-0.8B")       # Hugging Face checkpoint folder
BASE_ST = BASE_MODEL / "model.safetensors-00001-of-00001.safetensors"
TEXT_CONFIG = env_path("AFWSP_TEXT_CONFIG", ROOT / "data" / "model_text")   # text-only config for vLLM checkpoints
CACHE = env_path("AFWSP_CACHE", ROOT / "cache")                              # pairs, activations, LP solutions
EDITS = env_path("AFWSP_EDITS", ROOT / "edits")                              # rank-1 edits as bf16 tensors
RESULTS = ROOT / "results"


def down_proj(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.down_proj.weight"
