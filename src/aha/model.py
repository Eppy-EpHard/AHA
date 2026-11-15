"""The model: load it, capture what its MLPs read and write, read its weights, and build the text-only checkpoint
that vLLM evaluates (Qwen3_5ForCausalLM, as the original paper did).

  python -m aha.model checkpoint <model_dir> <out_dir> [--edit edit.safetensors]

Adapted in part from weight-space-lp (MIT).
"""
import argparse
import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model_dir, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device()).eval()
    return tokenizer, model


def decoder_layers(model):
    inner = model.model
    return (inner.language_model if hasattr(inner, "language_model") else inner).layers


@contextmanager
def capture(model, layer_ids, what=("h", "resid")):
    """Record, per layer, the MLP's down_proj input ("h", [T, d_mlp]) and the layer's output ("resid", [T, d_model]),
    for a batch of one."""
    store = {l: {} for l in layer_ids}
    hooks = []
    for l in layer_ids:
        layer = decoder_layers(model)[l]
        if "h" in what:
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
                lambda module, args, l=l: store[l].__setitem__("h", args[0][0].detach())))
        if "resid" in what:
            hooks.append(layer.register_forward_hook(
                lambda module, args, out, l=l: store[l].__setitem__(
                    "resid", (out[0] if isinstance(out, tuple) else out)[0].detach())))
    try:
        yield store
    finally:
        for hook in hooks:
            hook.remove()


def down_proj(layer):
    return f"model.language_model.layers.{layer}.mlp.down_proj.weight"


def weights_file(model_dir):
    files = sorted(Path(model_dir).glob("model*.safetensors"))
    if len(files) != 1:
        raise SystemExit(f"expected one weights file in {model_dir}, found {len(files)}")
    return files[0]


def read_weight(model_dir, name):
    with safe_open(str(weights_file(model_dir)), "pt") as f:
        return f.get_tensor(name).float().numpy()


def text_config(model_dir):
    """The text-only config the checkpoint is served with: the Hugging Face config's text_config plus four keys.
    For the 0.8B this reproduces the original paper's Qwen3.5-0.8B-text config."""
    hf = json.load(open(Path(model_dir) / "config.json"))
    config = dict(hf["text_config"])
    config["architectures"] = ["Qwen3_5ForCausalLM"]
    config.setdefault("bos_token_id", None)
    config.setdefault("pad_token_id", None)
    config.setdefault("partial_rotary_factor", config.get("rope_parameters", {}).get("partial_rotary_factor"))
    generation = {"_from_model_config": True, "eos_token_id": config["eos_token_id"], "use_cache": True}
    return config, generation


def build_checkpoint(model_dir, out_dir, edit_file=None):
    """Write a checkpoint vLLM can serve: the text-only config, the tokenizer and the language-model weights, with
    the tensors in edit_file in place of the originals."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config, generation = text_config(model_dir)
    json.dump(config, open(out / "config.json", "w"), indent=2)
    json.dump(generation, open(out / "generation_config.json", "w"), indent=2)
    for name in TOKENIZER_FILES:
        shutil.copyfile(Path(model_dir) / name, out / name)
    weights = {}
    with safe_open(str(weights_file(model_dir)), "pt") as f:
        for name in f.keys():
            if name.startswith("model.language_model."):
                weights[name] = f.get_tensor(name)
    changed = {}
    if edit_file:
        for name, tensor in load_file(str(edit_file)).items():
            assert name in weights and weights[name].shape == tensor.shape, name
            tensor = tensor.to(torch.bfloat16)
            changed[name] = {"changed_entries": int((tensor != weights[name]).sum()),
                             "relative_change": float((tensor.float() - weights[name].float()).norm()
                                                      / weights[name].float().norm())}
            weights[name] = tensor.contiguous()
    save_file(weights, str(out / "model.safetensors"), metadata={"format": "pt"})
    json.dump({"base": str(model_dir), "edit": str(edit_file) if edit_file else None, "changed": changed},
              open(out / "edits.json", "w"), indent=1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ckpt = sub.add_parser("checkpoint", help="build a vLLM checkpoint, optionally with an edit applied")
    ckpt.add_argument("model_dir")
    ckpt.add_argument("out_dir")
    ckpt.add_argument("--edit")
    args = parser.parse_args()
    build_checkpoint(args.model_dir, args.out_dir, args.edit)


if __name__ == "__main__":
    main()
