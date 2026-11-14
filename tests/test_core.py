"""CPU tests of everything that needs neither a GPU nor a model download:  uv run pytest"""
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from afwsp import edits, lp
from afwsp.loops import dup_mask, label, onset
from afwsp.paths import ROOT, down_proj


def test_loop_detector_finds_a_planted_loop():
    rng = np.random.default_rng(0)
    ids = np.concatenate([rng.integers(1000, 100000, 600), np.tile(np.arange(7), 200)]).astype(np.int64)
    o = onset(dup_mask(ids))
    assert 600 <= o < 630
    L = label(ids, cut=True)
    assert L.onset == o and len(L.P) == o - 256 and len(L.R_during) > 0


def test_a_finished_trace_is_all_productive():
    L = label(np.arange(800, dtype=np.int64), cut=False)
    assert L.onset is None and len(L.P) == 800


# Toy: 4 neurons, a 3-number stream. Neuron 3 reads 3 on looping tokens and 0 on productive ones.
HR = np.array([[1, 0, 3, 2], [2, 1, 3, 2]], np.float32)
HP = np.array([[1, 1, 0, 2], [2, 0, 0, 2]], np.float32)
U = np.array([0.6, 0.0, 0.8])
W = torch.tensor([[1, 0, .3, 0], [0, 1, 0, 0], [0, 0, .4, 1]], dtype=torch.bfloat16)


def test_the_lp_picks_the_loop_neuron():
    spec = lp.Spec(delta_R=np.full(2, 6.0), eps=1.0, C=10.0, cost=np.ones(4))
    assert np.allclose(lp.pdhg(HR, HP, spec, omega=100.0, device="cpu").v, [0, 0, -2, 0], atol=0.05)


def test_the_edit_is_rank_one_on_the_chosen_column(tmp_path, monkeypatch):
    save_file({down_proj(0): W}, str(tmp_path / "base.safetensors"))
    monkeypatch.setattr(edits, "BASE_ST", tmp_path / "base.safetensors")
    monkeypatch.setattr(edits, "EDITS", tmp_path / "edits")
    v = np.array([0, 0, -2.0, 0])
    info = edits.write("toy", {0: (U, v)})["layers"][0]
    Wn = load_file(str(tmp_path / "edits/toy.safetensors"))[down_proj(0)]
    assert info["neurons"] == [2] and info["changed_weights_bf16"] == 2
    assert np.allclose(Wn.float().numpy(), W.double().numpy() + np.outer(U, v), atol=1e-2)


def test_eval_stats_count_loops_cutoffs_and_terminated_accuracy():
    from afwsp.eval import record
    looping = list(range(600)) + [1, 2, 3, 4, 5, 6, 7] * 200     # 600 fresh tokens, then a 7-token loop
    rows = [{"tokens": len(looping), "cut": True, "correct": 0, "answered": False, **record.loop_stats(looping)},
            {"tokens": 300, "cut": False, "correct": 1, "answered": True, **record.loop_stats(range(300))}]
    s = record.stats(rows)
    assert rows[0]["loop_onset"] is not None and rows[1]["loop_onset"] is None
    assert s["loop_share"] == 0.5 and s["cut_share"] == 0.5 and s["loop_share_of_cut"] == 1.0
    assert s["acc"] == 0.5 and s["acc_terminated"] == 1.0 and s["n_terminated"] == 1 and s["answered_share"] == 0.5


def test_sigma_matches_the_published_math500_number():
    spec = importlib.util.spec_from_file_location("compare", ROOT / "scripts/compare.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    base = {i: float(i < 49) for i in range(500)}      # 9.8%
    edited = {i: float(i < 259) for i in range(500)}   # 51.8%
    r = compare.paired(base, edited)
    assert r["delta"] == 42.0 and r["sigma_pooled"] == 14.38 and r["fixed"] == 210 and r["broken"] == 0


@pytest.mark.skipif(not ((ROOT / "models/Qwen3.5-0.8B/config.json").exists()
                         and (ROOT / "data/model_text/config.json").exists()),
                    reason="needs scripts/fetch_model.sh and scripts/fetch_exp8.sh")
def test_the_text_config_derivation_reproduces_the_authors_file():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_text_config.py"), "--check",
                        str(ROOT / "models/Qwen3.5-0.8B")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
