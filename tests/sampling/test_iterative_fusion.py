"""Tests for iterative_fusion: cross-cycle bias refinement."""

from __future__ import annotations

import numpy as np
import pytest

from protein_chisel.sampling.iterative_fusion import (
    IterationBiasConfig,
    build_iteration_bias,
    consensus_aa_frequencies,
)
from protein_chisel.sampling.plm_fusion import AA_ORDER

_AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


# ----------------------------------------------------------------------
# consensus_aa_frequencies
# ----------------------------------------------------------------------


def test_consensus_freq_empty_input():
    out = consensus_aa_frequencies([], L=5)
    assert out.shape == (5, 20)
    assert (out == 0).all()


def test_consensus_freq_all_identical():
    seqs = ["MKVLA"] * 10
    L = 5
    out = consensus_aa_frequencies(seqs, L)
    # Each position should have 1.0 freq for the seed AA
    for i, c in enumerate("MKVLA"):
        j = _AA_TO_IDX[c]
        assert out[i, j] == pytest.approx(1.0)
        # All other AAs should be 0
        mask = np.ones(20, dtype=bool); mask[j] = False
        assert (out[i, mask] == 0.0).all()


def test_consensus_freq_half_half():
    seqs = ["A" * 5] * 5 + ["G" * 5] * 5
    L = 5
    out = consensus_aa_frequencies(seqs, L)
    a_idx = _AA_TO_IDX["A"]
    g_idx = _AA_TO_IDX["G"]
    for i in range(L):
        assert out[i, a_idx] == pytest.approx(0.5)
        assert out[i, g_idx] == pytest.approx(0.5)


def test_consensus_freq_wrong_length_raises():
    with pytest.raises(ValueError):
        consensus_aa_frequencies(["MKVL", "AGD"], L=5)


def test_consensus_freq_non_canonical_skipped():
    seqs = ["MXV", "MAV"]
    out = consensus_aa_frequencies(seqs, L=3)
    # Position 0: 'M' x 2 → freq 1.0
    assert out[0, _AA_TO_IDX["M"]] == pytest.approx(1.0)
    # Position 1: only 'A' counted (X skipped) → freq = 1/1
    assert out[1, _AA_TO_IDX["A"]] == pytest.approx(1.0)
    # Position 2: 'V' x 2 → freq 1.0
    assert out[2, _AA_TO_IDX["V"]] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# build_iteration_bias
# ----------------------------------------------------------------------


def _zero_bias(L: int) -> np.ndarray:
    return np.zeros((L, 20), dtype=np.float64)


def test_build_bias_no_survivors_returns_base_unchanged():
    L = 6
    base = _zero_bias(L)
    out, telem = build_iteration_bias(
        base_bias=base,
        survivor_sequences=[],
        position_classes=["surface"] * L,
    )
    assert np.array_equal(out, base)
    assert telem.n_survivors == 0
    assert telem.n_positions_augmented == 0


def test_build_bias_consensus_above_threshold_augments():
    L = 4
    base = _zero_bias(L)
    # 90% of survivors agree on 'A' at all positions
    seqs = ["AAAA"] * 9 + ["KKKK"] * 1
    cfg = IterationBiasConfig(
        consensus_threshold=0.85, consensus_strength=2.0,
        only_at_classes=("surface",), max_augmented_fraction=1.0,
    )
    out, telem = build_iteration_bias(
        base_bias=base,
        survivor_sequences=seqs,
        position_classes=["surface"] * L,
        config=cfg,
    )
    # All 4 positions augmented at 'A' with +2.0
    a_idx = _AA_TO_IDX["A"]
    for i in range(L):
        assert out[i, a_idx] == pytest.approx(2.0)
        # other AAs unchanged
        mask = np.ones(20, dtype=bool); mask[a_idx] = False
        assert (out[i, mask] == 0).all()
    assert telem.n_positions_augmented == 4
    assert telem.augmented_resnos == [1, 2, 3, 4]


def test_build_bias_below_threshold_does_not_augment():
    L = 3
    base = _zero_bias(L)
    seqs = ["AAA"] * 8 + ["KKK"] * 2  # 80% A agreement
    cfg = IterationBiasConfig(
        consensus_threshold=0.85, consensus_strength=2.0,
        only_at_classes=("surface",), max_augmented_fraction=1.0,
    )
    out, telem = build_iteration_bias(base, seqs, ["surface"] * L, config=cfg)
    assert (out == base).all()
    assert telem.n_positions_augmented == 0


def test_build_bias_active_site_class_never_augmented():
    L = 5
    base = _zero_bias(L)
    seqs = ["AAAAA"] * 10
    classes = ["active_site", "surface", "surface", "active_site", "ligand"]
    cfg = IterationBiasConfig(
        consensus_threshold=0.5, consensus_strength=2.0,
        only_at_classes=("surface", "buried"), max_augmented_fraction=1.0,
    )
    out, telem = build_iteration_bias(base, seqs, classes, config=cfg)
    a_idx = _AA_TO_IDX["A"]
    # only positions 1, 2 (surface) augmented
    assert out[0, a_idx] == 0  # active_site
    assert out[1, a_idx] == pytest.approx(2.0)
    assert out[2, a_idx] == pytest.approx(2.0)
    assert out[3, a_idx] == 0  # active_site
    assert out[4, a_idx] == 0  # ligand (not in only_at_classes)
    assert telem.n_positions_augmented == 2


def test_build_bias_fixed_resnos_skipped():
    L = 4
    base = _zero_bias(L)
    seqs = ["AAAA"] * 10
    # Index 1 is fixed (e.g. catalytic)
    cfg = IterationBiasConfig(
        consensus_threshold=0.5, consensus_strength=2.0,
        only_at_classes=("surface",), max_augmented_fraction=1.0,
    )
    out, telem = build_iteration_bias(
        base_bias=base,
        survivor_sequences=seqs,
        position_classes=["surface"] * L,
        fixed_resnos_zero_indexed=[1],
        config=cfg,
    )
    a_idx = _AA_TO_IDX["A"]
    assert out[0, a_idx] == pytest.approx(2.0)
    assert out[1, a_idx] == 0   # fixed → skipped
    assert out[2, a_idx] == pytest.approx(2.0)
    assert out[3, a_idx] == pytest.approx(2.0)
    assert telem.n_positions_augmented == 3


def test_build_bias_max_augmented_fraction_caps():
    L = 10
    base = _zero_bias(L)
    seqs = ["A" * L] * 10
    cfg = IterationBiasConfig(
        consensus_threshold=0.5, consensus_strength=2.0,
        only_at_classes=("surface",),
        max_augmented_fraction=0.30,  # cap = 3
    )
    out, telem = build_iteration_bias(base, seqs, ["surface"] * L, config=cfg)
    a_idx = _AA_TO_IDX["A"]
    n_aug = int((out[:, a_idx] == 2.0).sum())
    assert n_aug == 3
    assert telem.capped is True
    assert telem.n_positions_augmented == 3


def test_build_bias_preserves_base_at_non_augmented_positions():
    L = 5
    base = np.random.RandomState(0).normal(0, 0.5, size=(L, 20))
    seqs = ["AAAAA"] * 10
    classes = ["active_site"] * L  # nothing eligible
    out, telem = build_iteration_bias(base, seqs, classes)
    assert np.array_equal(out, base)


def test_build_bias_wrong_shape_raises():
    base = np.zeros((5, 19))  # 19 cols, bad
    with pytest.raises(ValueError):
        build_iteration_bias(base, ["AAAAA"], ["surface"] * 5)


def test_build_bias_wrong_classes_length_raises():
    base = np.zeros((5, 20))
    with pytest.raises(ValueError):
        build_iteration_bias(base, ["AAAAA"], ["surface"] * 6)
