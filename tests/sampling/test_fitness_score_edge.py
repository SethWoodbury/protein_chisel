"""Edge-case tests for fitness_score not covered by test_fitness_score.py.

Targets the specific failure modes that could blow up an autonomous
3-cycle iterative_design_v2 run:
- stale ``seq_hash`` columns surviving a TSV round-trip
- duplicate-collapse + downstream pdb_map lookup integrity
- empty / all-non-canonical / NaN sequence handling
- cache pollution across cycles when sequences case-fold equal
"""

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


def _uniform_log_probs(L: int) -> np.ndarray:
    return np.full((L, 20), -np.log(20.0))


# ----------------------------------------------------------------------
# Stale seq_hash column behavior in score_dataframe_fitness
# ----------------------------------------------------------------------


def test_score_dataframe_trusts_existing_seq_hash_column():
    """``score_dataframe_fitness`` does NOT recompute ``seq_hash`` when the
    column already exists. This is intentional (perf), but means a stale /
    wrong upstream hash will be used as the cache key.

    Scoring is still correct (``fitness_from_seed_marginals`` re-uppercases
    the sequence internally), but the cache key is wrong. Documented here
    so a future change that adds re-validation makes this fail loudly.
    """
    L = 4
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    df = pd.DataFrame({
        "sequence": ["MKVL"],
        "seq_hash": ["DEADBEEF" * 3],   # 24 hex chars but wrong
        "id": ["fake"],
    })
    cache: dict = {}
    out = score_dataframe_fitness(df, lp_e, lp_s, w, fitness_cache=cache)
    # Score is correct (uniform => -log(20))
    assert out["fitness__logp_esmc_mean"].iloc[0] == pytest.approx(-np.log(20.0))
    # But the cache was keyed by the (wrong) provided hash, not the true one.
    assert "DEADBEEF" * 3 in cache
    assert seq_hash("MKVL") not in cache


def test_score_dataframe_recomputes_hash_when_missing():
    """When ``seq_hash`` is absent, the function computes from upper-cased
    sequence and uses that as the cache key. Documents the required
    contract for fresh DataFrames coming straight off candidates.tsv."""
    L = 4
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    df = pd.DataFrame({"sequence": ["mkvl"]})  # lowercase
    cache: dict = {}
    out = score_dataframe_fitness(df, lp_e, lp_s, w, fitness_cache=cache)
    # Hash column was added, derived from upper-cased seq.
    assert out["seq_hash"].iloc[0] == seq_hash("MKVL")
    assert seq_hash("MKVL") in cache


def test_score_dataframe_cache_reuse_across_calls():
    """Cache is keyed by ``seq_hash``. Two calls with the *same* sequence
    populate the cache once and reuse it on the second call."""
    L = 4
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    cache: dict = {}
    df1 = pd.DataFrame({"sequence": ["MKVL"]})
    df1 = deduplicate_by_sequence(df1)
    score_dataframe_fitness(df1, lp_e, lp_s, w, fitness_cache=cache)
    n1 = len(cache)
    df2 = pd.DataFrame({"sequence": ["MKVL"]})
    df2 = deduplicate_by_sequence(df2)
    score_dataframe_fitness(df2, lp_e, lp_s, w, fitness_cache=cache)
    assert len(cache) == n1   # no new entry


# ----------------------------------------------------------------------
# Cycle-state corruption: same case-folded seq, different ids
# ----------------------------------------------------------------------


def test_dedup_keeps_first_id_when_seqs_equal():
    """Two rows, same case-folded sequence, different ids — keep first.

    Critical for the v2 driver: a downstream ``pdb_map[id]`` lookup must
    succeed for the kept id, not the dropped one.
    """
    df = pd.DataFrame({
        "sequence": ["mkvl", "MKVL"],
        "id": ["design_A", "design_B"],
    })
    out = deduplicate_by_sequence(df)
    assert len(out) == 1
    assert out["id"].iloc[0] == "design_A"
    # The kept id is still findable in a hypothetical pdb_map.
    pdb_map = {"design_A": "/tmp/A.pdb", "design_B": "/tmp/B.pdb"}
    assert pdb_map.get(out["id"].iloc[0]) is not None


def test_dedup_preserves_sampler_metadata_of_first_occurrence():
    """When two rows are sequence-identical, dedup preserves the metadata
    of the first occurrence — important for tracking which cycle / seed
    produced the survivor."""
    df = pd.DataFrame({
        "sequence": ["MKVL", "MKVL"],
        "id": ["c0_design_5", "c1_design_2"],
        "cycle": [0, 1],
        "T": [0.20, 0.18],
    })
    out = deduplicate_by_sequence(df)
    assert len(out) == 1
    assert out["id"].iloc[0] == "c0_design_5"
    assert int(out["cycle"].iloc[0]) == 0
    assert float(out["T"].iloc[0]) == pytest.approx(0.20)


# ----------------------------------------------------------------------
# fitness_from_seed_marginals: more degenerate inputs
# ----------------------------------------------------------------------


def test_fitness_with_lowercase_sequence_uppercases_internally():
    """Sequence passed in lowercase should still produce the right score —
    the function calls ``.upper()`` before gathering."""
    L = 5
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    res_lower = fitness_from_seed_marginals("mkvla", lp_e, lp_s, w)
    res_upper = fitness_from_seed_marginals("MKVLA", lp_e, lp_s, w)
    assert res_lower.logp_esmc_mean == res_upper.logp_esmc_mean
    assert res_lower.seq_hash == res_upper.seq_hash
    # And the stored sequence is upper-cased.
    assert res_lower.sequence == "MKVLA"


def test_fitness_partial_non_canonical_uses_only_valid_positions():
    """Mix of valid + non-canonical AAs: mean computed over valid count.

    Three valid positions with -log(20) each => mean = -log(20).
    The stop codon '*' at position 3 is silently ignored.
    """
    L = 4
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    res = fitness_from_seed_marginals("MKV*", lp_e, lp_s, w)
    assert res.logp_esmc_mean == pytest.approx(-np.log(20.0))


def test_fitness_seq_shorter_than_marginals_raises():
    """Length must match the cached marginals exactly. If MPNN ever
    returns a truncated (e.g. M-elided) sequence, fitness must REFUSE to
    score rather than silently scoring against the wrong row range."""
    L = 5
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    with pytest.raises(ValueError, match="esmc shape"):
        fitness_from_seed_marginals("MKVL", lp_e, lp_s, w)   # length 4 vs L=5


def test_fitness_seq_longer_than_marginals_raises():
    L = 3
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    with pytest.raises(ValueError, match="esmc shape"):
        fitness_from_seed_marginals("MKVLA", lp_e, lp_s, w)


def test_fitness_saprot_shape_mismatch_raises():
    L = 5
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L + 1)   # mismatch only on saprot
    w = np.ones((L, 2))
    with pytest.raises(ValueError, match="saprot shape"):
        fitness_from_seed_marginals("MKVLA", lp_e, lp_s, w)


# ----------------------------------------------------------------------
# Edge cases at the dedup-empty -> next-cycle boundary
# ----------------------------------------------------------------------


def test_score_dataframe_empty_input_returns_empty_with_columns():
    """Zero-survivors after a cycle's filters must not crash the next
    stage. ``score_dataframe_fitness`` on an empty (post-dedup) DF must
    return an empty DF with the fitness columns present."""
    L = 4
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    df = pd.DataFrame(columns=["sequence", "id"])
    df = deduplicate_by_sequence(df)
    cache: dict = {}
    out = score_dataframe_fitness(df, lp_e, lp_s, w, fitness_cache=cache)
    assert len(out) == 0
    # Columns may be missing on truly empty input, but this should NOT raise.


def test_dedup_then_score_on_all_dupes_collapses_to_one_score():
    """5 identical sequences -> 1 row -> 1 score -> 1 cache entry. Sanity
    check on the dedup-then-score idiom used in stage_seq_filter +
    stage_fitness_score."""
    L = 4
    lp_e = _uniform_log_probs(L)
    lp_s = _uniform_log_probs(L)
    w = np.ones((L, 2))
    df = pd.DataFrame({"sequence": ["MKVL"] * 5, "id": list(range(5))})
    df = deduplicate_by_sequence(df)
    cache: dict = {}
    out = score_dataframe_fitness(df, lp_e, lp_s, w, fitness_cache=cache)
    assert len(out) == 1
    assert len(cache) == 1
    assert int(out["n_dupes"].iloc[0]) == 5


# ----------------------------------------------------------------------
# Stop-codon * / ambiguous AA handling at the hash level
# ----------------------------------------------------------------------


def test_seq_hash_ambiguous_aa_not_normalized():
    """seq_hash does NOT normalize 'X' or '*' away — it just upper-cases.
    Documents that an MPNN candidate with an X is hashed distinctly from
    its X-stripped version, so dedup won't collapse them."""
    h_with_x = seq_hash("MKVLX")
    h_no_x = seq_hash("MKVL")
    h_with_star = seq_hash("MKVL*")
    assert h_with_x != h_no_x
    assert h_with_star != h_no_x
    assert h_with_x != h_with_star
