"""Tests for the survivor-pool covariation diagnostic (experimental)."""
from __future__ import annotations

import numpy as np

from protein_chisel.scoring.covariation import (
    CovariationResult,
    covariation_diagnostic,
    mutual_information_matrix,
    anticorrelated_pairs,
)


def test_empty_and_single():
    assert mutual_information_matrix([]).size == 0
    r = covariation_diagnostic([])
    assert r.length == 0 and r.top_pairs == []


def test_independent_positions_low_mip():
    rng = np.random.default_rng(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    # 3 positions, each independently uniform -> couplings ~ 0 after APC
    seqs = ["".join(rng.choice(list(aa), size=3)) for _ in range(300)]
    mip = mutual_information_matrix(seqs)
    assert mip.shape == (3, 3)
    assert np.allclose(np.diag(mip), 0.0)
    assert np.abs(mip).max() < 0.25        # no strong spurious coupling


def test_perfectly_coupled_pair_is_top():
    # positions 0 and 2 perfectly co-vary; position 1 is constant.
    rng = np.random.default_rng(1)
    seqs = []
    for _ in range(200):
        x = rng.choice(list("ACDE"))
        partner = {"A": "K", "C": "L", "D": "M", "E": "F"}[x]
        seqs.append(f"{x}G{partner}")
    res = covariation_diagnostic(seqs, top_n=3)
    assert res.n_sequences == 200 and res.length == 3
    top_i, top_j, top_v = res.top_pairs[0]
    assert {top_i, top_j} == {0, 2}        # the coupled pair ranks first
    assert top_v > 0.05                     # positive coupling (APC + pseudocount shrink small-L)


def test_apc_reduces_conservation_inflation():
    # A highly conserved column should not dominate after APC.
    rng = np.random.default_rng(2)
    seqs = []
    for _ in range(200):
        a = rng.choice(list("ACDEFGHIKL"))
        b = rng.choice(list("ACDEFGHIKL"))
        seqs.append(f"{a}W{b}")            # middle col constant 'W'
    raw = mutual_information_matrix(seqs, apc=False)
    mip = mutual_information_matrix(seqs, apc=True)
    # APC should not increase the coupling of the conserved column with others
    assert mip[0, 1] <= raw[0, 1] + 1e-9


def test_ragged_sequences_dropped():
    seqs = ["ACD", "AED", "AC"]            # last is ragged -> dropped
    res = covariation_diagnostic(seqs)
    assert res.n_sequences == 2 and res.length == 3
    assert res.meta["dropped_ragged"] == 1


def test_raw_mi_is_nonnegative():
    # marginals derived from the smoothed joint -> MI is a proper >= 0 quantity.
    rng = np.random.default_rng(7)
    seqs = ["".join(rng.choice(list("ACDEFG"), size=5)) for _ in range(80)]
    raw = mutual_information_matrix(seqs, apc=False)
    assert (raw >= -1e-12).all()


def test_invalid_pseudocount_raises():
    import pytest
    with pytest.raises(ValueError):
        mutual_information_matrix(["AC", "AD"], pseudocount=-1.0)


def test_anticorrelated_pairs_threshold():
    rng = np.random.default_rng(3)
    seqs = []
    for _ in range(200):
        x = rng.choice(list("ACDE"))
        partner = {"A": "K", "C": "L", "D": "M", "E": "F"}[x]
        seqs.append(f"{x}G{partner}")
    res = covariation_diagnostic(seqs)
    hi = anticorrelated_pairs(res, threshold=0.05)
    assert any({i, j} == {0, 2} for (i, j, _v) in hi)
    assert anticorrelated_pairs(res, threshold=10.0) == []
