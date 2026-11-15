"""CPU tests of everything that needs neither a GPU nor a model download:  uv run pytest"""
import json

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from afwsp import edit, evals, model, report
from afwsp.loops import dup_mask, label, onset

# Toy: a 3-number stream, 4 neurons. Neuron 3 reads 3 on looping tokens and 0 on productive ones.
H_LOOP = np.array([[1, 0, 3, 2], [2, 1, 3, 2]], np.float32)
H_PROD = np.array([[1, 1, 0, 2], [2, 0, 0, 2]], np.float32)
U = np.array([0.6, 0.0, 0.8])
W = torch.tensor([[1, 0, .3, 0], [0, 1, 0, 0], [0, 0, .4, 1]], dtype=torch.bfloat16)


def test_loop_detector_finds_a_planted_loop():
    rng = np.random.default_rng(0)
    ids = np.concatenate([rng.integers(1000, 100000, 600), np.tile(np.arange(7), 200)]).astype(np.int64)
    start = onset(dup_mask(ids))
    assert 600 <= start < 630
    labels = label(ids, cut=True)
    assert labels.onset == start and len(labels.P) == start - 256 and len(labels.R_during) > 0
    assert label(np.arange(800, dtype=np.int64), cut=False).onset is None


def test_the_lp_picks_the_loop_neuron():
    v = edit.pdhg(H_LOOP, H_PROD, delta=6.0, eps=1.0, cost=np.ones(4))
    assert np.allclose(v, [0, 0, -2, 0], atol=0.05)


def test_the_edit_is_rank_one_on_the_chosen_column(tmp_path):
    save_file({model.down_proj(0): W}, str(tmp_path / "model.safetensors"))
    np.savez(tmp_path / "solutions.npz", u0=U, v0=np.array([0, 0, -2.0, 0]))
    edit.write_edit(tmp_path, tmp_path, [0], tmp_path / "edits/toy.safetensors")
    new = load_file(str(tmp_path / "edits/toy.safetensors"))[model.down_proj(0)]
    info = json.load(open(tmp_path / "edits/toy.json"))["layers"]["0"]
    assert info["neurons"] == [2] and info["changed_weights"] == 2
    assert np.allclose(new.float().numpy(), W.double().numpy() + np.outer(U, [0, 0, -2, 0]), atol=1e-2)


def test_the_text_config_is_the_text_part_plus_the_causal_lm_keys(tmp_path):
    hf = {"architectures": ["Qwen3_5ForConditionalGeneration"],
          "text_config": {"num_hidden_layers": 24, "eos_token_id": 248044,
                          "rope_parameters": {"partial_rotary_factor": 0.25}}}
    (tmp_path / "config.json").write_text(json.dumps(hf))
    config, generation = model.text_config(tmp_path)
    assert config["architectures"] == ["Qwen3_5ForCausalLM"] and config["partial_rotary_factor"] == 0.25
    assert config["bos_token_id"] is None and generation["eos_token_id"] == 248044


def test_answers_are_read_from_the_last_box_after_thinking():
    assert evals.extract_answer("<think>\\boxed{1}</think> so \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert evals.extract_answer("<think>looping...") is None


def test_answer_statistics_count_loops_cutoffs_and_terminated_accuracy():
    looping = list(range(600)) + [1, 2, 3, 4, 5, 6, 7] * 200
    answers = [{"tokens": len(looping), "cut": True, "correct": 0, "answered": False, **evals.loop_stats(looping)},
               {"tokens": 300, "cut": False, "correct": 1, "answered": True, **evals.loop_stats(range(300))}]
    s = evals.summarize(answers)
    assert s["loop_share"] == 0.5 and s["cut_share"] == 0.5 and s["loop_share_of_cut"] == 1.0
    assert s["acc"] == 0.5 and s["acc_terminated"] == 1.0 and s["answered_share"] == 0.5


def test_sigma_matches_the_published_math500_number():
    r = report.paired({i: float(i < 49) for i in range(500)}, {i: float(i < 259) for i in range(500)})
    assert r["delta"] == 42.0 and r["sigma_pooled"] == 14.38 and r["fixed"] == 210 and r["broken"] == 0
