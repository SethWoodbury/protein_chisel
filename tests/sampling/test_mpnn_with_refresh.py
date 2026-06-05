"""Tests for the pure PLM-bias refresh orchestrator (no models/containers)."""
from __future__ import annotations

import pytest

from protein_chisel.sampling.mpnn_with_refresh import run_with_refresh


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
