"""Tests for the pure PLM-bias refresh orchestrator (no models/containers)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from protein_chisel.sampling.mpnn_with_refresh import (
    choose_inband_representative, refuse_esmc_only, run_with_refresh,
)


def _rand_lp(L, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(L, 20))
    return x - np.log(np.exp(x).sum(1, keepdims=True))


# ---- WS-F: representative selection (pure; host-testable) ------------------
_BAND = dict(gravy_min=-0.8, gravy_max=0.3, net_charge_min=-18.0, net_charge_max=-4.0)


def test_choose_inband_representative_picks_median_fitness():
    df = pd.DataFrame({
        "sequence": ["LO", "MED", "HI", "OUT"],
        "gravy": [-0.3, -0.2, -0.1, 1.2],            # OUT is above gravy_max=0.3
        "net_charge_full_HH": [-10, -9, -8, -10],
        "fitness__logp_fused_mean": [-3.0, -2.0, -1.0, -2.0],
        "passed_seq_filter": [True, True, True, True],
    })
    assert choose_inband_representative(df, **_BAND) == "MED"   # median of the 3 in-band


def test_choose_inband_representative_none_when_none_in_band():
    df = pd.DataFrame({"sequence": ["X"], "gravy": [1.5],
                       "net_charge_full_HH": [-10], "fitness__logp_fused_mean": [-1.0]})
    assert choose_inband_representative(df, **_BAND) is None
    assert choose_inband_representative(None, **_BAND) is None
    assert choose_inband_representative(pd.DataFrame(), **_BAND) is None


def test_choose_inband_representative_robust_passed_seq_filter_strings():
    """codex: a string 'False' / NaN in passed_seq_filter must be treated as FAILED
    (a naive .astype(bool) would read 'False' as True)."""
    df = pd.DataFrame({
        "sequence": ["A", "B"],
        "gravy": [-0.2, -0.2], "net_charge_full_HH": [-9, -9],
        "fitness__logp_fused_mean": [-1.0, -2.0],
        "passed_seq_filter": ["False", "True"],       # strings, not bools
    })
    assert choose_inband_representative(df, **_BAND) == "B"   # 'False' row excluded


def test_choose_inband_representative_skips_null_sequences():
    df = pd.DataFrame({
        "sequence": [None, "GOOD"],
        "gravy": [-0.2, -0.2], "net_charge_full_HH": [-9, -9],
        "fitness__logp_fused_mean": [-1.0, -2.0],
        "passed_seq_filter": [True, True],
    })
    assert choose_inband_representative(df, **_BAND) == "GOOD"


def test_choose_inband_representative_excludes_seq_filter_failures():
    df = pd.DataFrame({
        "sequence": ["A", "B", "C"],
        "gravy": [-0.2, -0.2, -0.2], "net_charge_full_HH": [-9, -9, -9],
        "fitness__logp_fused_mean": [-3.0, -2.0, -1.0],
        "passed_seq_filter": [True, False, True],     # B (the median) failed -> excluded
    })
    # qualified {A:-3, C:-1}; median -2; ties on distance -> higher fitness -> C.
    assert choose_inband_representative(df, **_BAND) == "C"


def test_refuse_esmc_only_equals_two_expert_fusion():
    from protein_chisel.sampling.plm_fusion import FusionConfig, fuse_experts
    L = 12
    new_esmc, seed_saprot = _rand_lp(L, 1), _rand_lp(L, 2)
    classes = ["distal_surface"] * L
    cfg = FusionConfig()
    bias = refuse_esmc_only(new_esmc_lp=new_esmc, seed_saprot_lp=seed_saprot,
                            position_classes=classes, fusion_cfg=cfg)
    expected = fuse_experts([new_esmc, seed_saprot], classes, config=cfg,
                            expert_names=["esmc", "saprot"]).bias
    assert np.array_equal(bias, expected)


def test_rounds_zero_samples_once():
    calls = []
    out = run_with_refresh(
        "bias0",
        rounds=0,
        sample_fn=lambda b: calls.append(b) or f"result({b})",
        choose_representative_fn=lambda r: pytest.fail("should not be called"),
        recompute_bias_fn=lambda s, b: pytest.fail("should not be called"),
    )
    assert calls == ["bias0"]                  # exactly one sample, with initial bias
    assert out == "result(bias0)"


def test_two_rounds_refresh_chain():
    sampled, recomputed = [], []

    def sample_fn(b):
        sampled.append(b)
        return {"bias": b, "rep": f"seq_after_{b}"}

    def choose(result):
        return result["rep"]

    def recompute(rep, prev):
        recomputed.append((rep, prev))
        return f"bias_from_{rep}"

    out = run_with_refresh("b0", rounds=2, sample_fn=sample_fn,
                           choose_representative_fn=choose,
                           recompute_bias_fn=recompute)
    # initial sample + 2 refresh samples = 3 sample calls; bias chains forward
    assert sampled == ["b0", "bias_from_seq_after_b0",
                       "bias_from_seq_after_bias_from_seq_after_b0"]
    assert len(recomputed) == 2
    assert out["bias"] == sampled[-1]


def test_stop_when_no_representative():
    sampled = []
    out = run_with_refresh(
        "b0", rounds=5,
        sample_fn=lambda b: sampled.append(b) or "r",
        choose_representative_fn=lambda r: None,     # nothing survived
        recompute_bias_fn=lambda s, b: pytest.fail("should not recompute"),
    )
    assert sampled == ["b0"]                  # only the initial sample ran
    assert out == "r"


def test_stop_when_recompute_unavailable():
    sampled = []
    out = run_with_refresh(
        "b0", rounds=5,
        sample_fn=lambda b: sampled.append(b) or "r",
        choose_representative_fn=lambda r: "drifted",
        recompute_bias_fn=lambda s, b: None,         # recompute failed/unavailable
    )
    assert sampled == ["b0"]                  # no resample after failed recompute
    assert out == "r"


def test_negative_rounds_raises():
    with pytest.raises(ValueError):
        run_with_refresh("b0", rounds=-1, sample_fn=lambda b: "r",
                         choose_representative_fn=lambda r: None,
                         recompute_bias_fn=lambda s, b: None)


def test_non_int_rounds_raises():
    with pytest.raises(ValueError):
        run_with_refresh("b0", rounds=2.0, sample_fn=lambda b: "r",
                         choose_representative_fn=lambda r: None,
                         recompute_bias_fn=lambda s, b: None)
