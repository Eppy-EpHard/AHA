"""Write the text-only config a Qwen3.5 checkpoint is evaluated with, for any size.

The original paper evaluates the text model alone (Qwen3_5ForCausalLM) and ships that config only for the 0.8B,
which scripts/fetch_exp8.sh copies to data/model_text/. It is the Hugging Face checkpoint's text_config plus a few
keys, so it can be derived for any size; --check confirms the rule reproduces the shipped 0.8B files exactly.

  python scripts/make_text_config.py models/Qwen3.5-2B data/model_text_2b
  python scripts/make_text_config.py --check models/Qwen3.5-0.8B          # against data/model_text
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def derive(model_dir: Path, ref: Path | None = None) -> tuple[dict, dict]:
    hf = json.load(open(model_dir / "config.json"))
    ref_cfg = json.load(open(ref / "config.json")) if ref and (ref / "config.json").exists() else {}
    version = ref_cfg.get("transformers_version", hf.get("transformers_version"))
    text = dict(hf["text_config"])
    text["architectures"] = ["Qwen3_5ForCausalLM"]
    text.setdefault("bos_token_id", None)
    text.setdefault("pad_token_id", None)
    text.setdefault("partial_rotary_factor", text.get("rope_parameters", {}).get("partial_rotary_factor"))
    text.setdefault("transformers_version", version)
    gen = {"_from_model_config": True, "eos_token_id": text["eos_token_id"], "transformers_version": version,
           "use_cache": True}
    return text, gen


def diff(a: dict, b: dict) -> dict:
    return {k: (a.get(k, "<missing>"), b.get(k, "<missing>")) for k in sorted(set(a) | set(b))
            if a.get(k, "<missing>") != b.get(k, "<missing>")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("out_dir", type=Path, nargs="?")
    ap.add_argument("--check", action="store_true", help="compare the derivation with --ref instead of writing")
    ap.add_argument("--ref", type=Path, default=ROOT / "data" / "model_text")
    a = ap.parse_args()
    text, gen = derive(a.model_dir, a.ref)
    if a.check:
        if not (a.ref / "config.json").exists():
            raise SystemExit(f"no reference at {a.ref}: run scripts/fetch_exp8.sh first")
        bad = {f: d for f, d in (("config.json", diff(text, json.load(open(a.ref / "config.json")))),
                                 ("generation_config.json", diff(gen, json.load(open(a.ref / "generation_config.json")))))
               if d}
        print("OK: the derivation reproduces the reference exactly" if not bad else f"MISMATCH: {bad}")
        raise SystemExit(1 if bad else 0)
    if a.out_dir is None:
        raise SystemExit("give an output folder, or --check")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(text, open(a.out_dir / "config.json", "w"), indent=2)
    json.dump(gen, open(a.out_dir / "generation_config.json", "w"), indent=2)
    print(f"wrote {a.out_dir}/config.json ({text['num_hidden_layers']} layers, width {text['hidden_size']}, "
          f"MLP {text['intermediate_size']}) and generation_config.json")


if __name__ == "__main__":
    main()
