"""Tests for fitness_score: dedup + per-sequence fitness from cached marginals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from protein_chisel.sampling.fitness_score import (
    AA_TO_IDX,
    deduplicate_by_sequence,
    fitness_from_seed_marginals,
    score_dataframe_fitness,
    seq_hash,
)


# ----------------------------------------------------------------------
# Hash determinism + case insensitivity
# ----------------------------------------------------------------------


def test_seq_hash_deterministic():
    assert seq_hash("MKVL") == seq_hash("MKVL")


def test_seq_hash_case_invariant():
    assert seq_hash("mkVL") == seq_hash("MKVL")


def test_seq_hash_different_sequences_collide_negligibly():
    h1 = seq_hash("MKVLAGD")
    h2 = seq_hash("MKVLAGE")
    assert h1 != h2


def test_seq_hash_typeerror_on_non_str():
    with pytest.raises(TypeError):
        seq_hash(b"MKVL")


# ----------------------------------------------------------------------
# Deduplication
# ----------------------------------------------------------------------


def test_dedup_empty_dataframe():
    df = pd.DataFrame(columns=["sequence", "id"])
    out = deduplicate_by_sequence(df)
    assert len(out) == 0
    assert "seq_hash" in out.columns
    assert "n_dupes" in out.columns


def test_dedup_no_duplicates():
    df = pd.DataFrame({"sequence": ["MKV", "AGD", "PLE"], "id": [1, 2, 3]})
    out = deduplicate_by_sequence(df)
    assert len(out) == 3
    assert (out["n_dupes"] == 1).all()


def test_dedup_all_identical():
    df = pd.DataFrame({"sequence": ["MKV"] * 10, "id": list(range(10))})
    out = deduplicate_by_sequence(df)
    assert len(out) == 1
    assert int(out["n_dupes"].iloc[0]) == 10
    # First-occurrence id should be 0 (stable).
    assert out["id"].iloc[0] == 0


def test_dedup_case_insensitive():
    df = pd.DataFrame({"sequence": ["MKv", "MKV", "mkv"], "id": [1, 2, 3]})
    out = deduplicate_by_sequence(df)
    assert len(out) == 1
    assert int(out["n_dupes"].iloc[0]) == 3


def test_dedup_stop_codons_preserved_distinctly():
    # `*` should be treated literally — collapses only with another `*`-bearing seq
    df = pd.DataFrame({"sequence": ["MKV*", "MKV", "MKV*"], "id": [1, 2, 3]})
    out = deduplicate_by_sequence(df).sort_values("id").reset_index(drop=True)
    assert len(out) == 2
    # MKV (no stop) and MKV* (with stop) are distinct
    seqs = sorted(out["sequence"].tolist())
    assert seqs == ["MKV", "MKV*"]


def test_dedup_missing_column_raises():
    df = pd.DataFrame({"foo": ["MKV"]})
    with pytest.raises(KeyError):
        deduplicate_by_sequence(df, sequence_col="sequence")


def test_dedup_none_input():
    out = deduplicate_by_sequence(None)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0


# ----------------------------------------------------------------------
# fitness_from_seed_marginals
# ----------------------------------------------------------------------


def _uniform_log_probs(L: int) -> np.ndarray:
    """(L, 20) uniform log-probs (each row sums to 1 in prob-space)."""
    return np.full((L, 20), -np.log(20.0))


def test_fitness_uniform_marginals_returns_log_1_over_20():
    L = 5
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    seq = "MKVLA"
    res = fitness_from_seed_marginals(seq, lp_e, lp_s, w)
    expected = -np.log(20.0)
    assert res.logp_esmc_mean == pytest.approx(expected, abs=1e-12)
    assert res.logp_saprot_mean == pytest.approx(expected, abs=1e-12)
    assert res.logp_fused_mean == pytest.approx(expected, abs=1e-12)
    assert res.method == "seed_marginal"
    assert res.seq_hash == seq_hash(seq)


def test_fitness_higher_logprob_at_seed_aa_gives_higher_score():
    # Make ESM-C strongly favor 'A' at every position, weakly favor others.
    L = 4
    lp_e = np.full((L, 20), -10.0)
    lp_e[:, AA_TO_IDX["A"]] = -0.1
    lp_e -= np.log(np.sum(np.exp(lp_e), axis=1, keepdims=True))
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    res_a = fitness_from_seed_marginals("AAAA", lp_e, lp_s, w)
    res_other = fitness_from_seed_marginals("KRDE", lp_e, lp_s, w)
    assert res_a.logp_esmc_mean > res_other.logp_esmc_mean


def test_fitness_shape_mismatch_raises():
    L = 5
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    with pytest.raises(ValueError):
        fitness_from_seed_marginals("MK", lp_e, lp_s, w)


def test_fitness_weights_shape_mismatch_raises():
    L = 5
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 3))  # bad
    with pytest.raises(ValueError):
        fitness_from_seed_marginals("MKVLA", lp_e, lp_s, w)


def test_fitness_non_canonical_aa_handled():
    # 'X' at one position should be ignored, fitness computed over canonical AAs only
    L = 3
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    res = fitness_from_seed_marginals("MXV", lp_e, lp_s, w)
    # Two valid AAs; mean = log(1/20)
    assert res.logp_esmc_mean == pytest.approx(-np.log(20.0), abs=1e-12)


def test_fitness_all_non_canonical_raises():
    L = 3
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    with pytest.raises(ValueError):
        fitness_from_seed_marginals("XXX", lp_e, lp_s, w)


def test_fitness_zero_weight_position_falls_back_to_unit_weight():
    """When β=γ=0 at a position, fused score uses 1/(1+1) average.

    This protects active-site rows where the fusion class weight is 0
    so the fitness is still defined (we don't want NaN downstream).
    """
    L = 3
    lp_e = np.array([[-1.0]*20, [-2.0]*20, [-3.0]*20])
    lp_s = np.array([[-2.0]*20, [-3.0]*20, [-4.0]*20])
    w = np.zeros((L, 2))  # all zero
    seq = "MKV"
    res = fitness_from_seed_marginals(seq, lp_e, lp_s, w)
    # With safe_w=1 and (β=γ=0), fused = 0/1 = 0 at each position
    # mean over 3 positions = 0
    assert res.logp_fused_mean == pytest.approx(0.0, abs=1e-12)


def test_fitness_reproducible():
    L = 7
    rng = np.random.default_rng(42)
    raw_e = rng.normal(0, 1, size=(L, 20))
    lp_e = raw_e - np.log(np.sum(np.exp(raw_e), axis=1, keepdims=True))
    raw_s = rng.normal(0, 1, size=(L, 20))
    lp_s = raw_s - np.log(np.sum(np.exp(raw_s), axis=1, keepdims=True))
    w = rng.uniform(0, 1, size=(L, 2))
    seq = "MKVLAGD"
    r1 = fitness_from_seed_marginals(seq, lp_e, lp_s, w)
    r2 = fitness_from_seed_marginals(seq, lp_e, lp_s, w)
    assert r1.logp_esmc_mean == r2.logp_esmc_mean
    assert r1.logp_saprot_mean == r2.logp_saprot_mean
    assert r1.logp_fused_mean == r2.logp_fused_mean


# ----------------------------------------------------------------------
# score_dataframe_fitness
# ----------------------------------------------------------------------


def test_score_dataframe_adds_columns_and_uses_cache():
    L = 4
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    df = pd.DataFrame({"sequence": ["MKVL", "AGDR", "MKVL"], "id": [1, 2, 3]})
    df = deduplicate_by_sequence(df)  # collapses duplicates first
    cache: dict = {}
    out = score_dataframe_fitness(df, lp_e, lp_s, w, fitness_cache=cache)
    assert "fitness__logp_esmc_mean" in out.columns
    assert "fitness__logp_fused_mean" in out.columns
    # Cache populated for each unique seq
    assert len(cache) == 2


def test_score_dataframe_cache_hit_avoids_recompute():
    L = 4
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    df1 = pd.DataFrame({"sequence": ["MKVL"]})
    df1 = deduplicate_by_sequence(df1)
    cache: dict = {}
    out1 = score_dataframe_fitness(df1, lp_e, lp_s, w, fitness_cache=cache)
    n_after_first = len(cache)

    df2 = pd.DataFrame({"sequence": ["MKVL"]})
    df2 = deduplicate_by_sequence(df2)
    out2 = score_dataframe_fitness(df2, lp_e, lp_s, w, fitness_cache=cache)
    # Cache size unchanged (hit)
    assert len(cache) == n_after_first
    assert (out1["fitness__logp_fused_mean"].iloc[0]
            == out2["fitness__logp_fused_mean"].iloc[0])
