"""Edge-case tests for iterative_fusion not covered by test_iterative_fusion.py.

Targets specific ways the cross-cycle bias refinement could go wrong in
an autonomous v2 run:
- single-unique-survivor cycle (full consensus collapse)
- positions with class label not in cfg.class_weights / only_at_classes
- base_bias is never mutated (cycle reuse safety)
- cap behavior breaks ties deterministically
- augmented_resnos label semantics (currently array-index + 1, NOT
  protein resno — documented edge case)
"""

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


def _zero_bias(L: int) -> np.ndarray:
    return np.zeros((L, 20), dtype=np.float64)


# ----------------------------------------------------------------------
# Single-unique-survivor degenerate cycle
# ----------------------------------------------------------------------


def test_one_unique_survivor_augments_all_eligible_positions():
    """If only ONE survivor remains after filters (e.g. cycle 1 had one
    sequence pass), every eligible position has 100% consensus on that
    sequence's AA, so every eligible position gets augmented.

    This is intentional behavior — single-survivor cycles produce a
    sharply pointed bias that cycle 2 will explore around. Documents the
    'collapse to a point' regime so a future change to add a minimum
    n_survivors guard breaks this test loudly.
    """
    L = 6
    base = _zero_bias(L)
    survivors = ["AKLDGE"]   # one sequence, all positions
    cfg = IterationBiasConfig(
        consensus_threshold=0.85, consensus_strength=2.0,
        only_at_classes=("surface",), max_augmented_fraction=1.0,
    )
    out, telem = build_iteration_bias(
        base_bias=base,
        survivor_sequences=survivors,
        position_classes=["surface"] * L,
        config=cfg,
    )
    assert telem.n_survivors == 1
    assert telem.n_positions_augmented == L
    for i, c in enumerate("AKLDGE"):
        j = _AA_TO_IDX[c]
        assert out[i, j] == pytest.approx(2.0)


# ----------------------------------------------------------------------
# Positions with class label outside cfg.only_at_classes
# ----------------------------------------------------------------------


def test_unknown_class_label_treated_as_ineligible():
    """A class label like 'membrane' or '' that isn't in
    ``only_at_classes`` must NEVER be augmented, mirroring how
    ``fuse_plm_logits`` falls back to weight 0 for unknown classes."""
    L = 4
    base = _zero_bias(L)
    survivors = ["AAAA"] * 10
    classes = ["surface", "MYSTERY", "", "buried"]
    cfg = IterationBiasConfig(
        consensus_threshold=0.5, consensus_strength=2.0,
        only_at_classes=("surface", "buried"), max_augmented_fraction=1.0,
    )
    out, telem = build_iteration_bias(base, survivors, classes, config=cfg)
    a_idx = _AA_TO_IDX["A"]
    assert out[0, a_idx] == pytest.approx(2.0)   # surface
    assert out[1, a_idx] == 0.0                   # MYSTERY
    assert out[2, a_idx] == 0.0                   # empty
    assert out[3, a_idx] == pytest.approx(2.0)   # buried
    assert telem.n_positions_eligible == 2
    assert telem.n_positions_augmented == 2


# ----------------------------------------------------------------------
# base_bias mutation safety — critical for cycle reuse
# ----------------------------------------------------------------------


def test_build_iteration_bias_does_not_mutate_base_bias():
    """``run_cycle`` reuses the same ``base_bias`` array across all
    cycles. If ``build_iteration_bias`` accidentally mutated its input,
    cycle 2 would see cycle-1's augmentations baked into 'base'.

    Verify the function makes a copy and leaves the caller's array
    untouched, regardless of whether augmentation actually fires.
    """
    L = 5
    base = np.random.RandomState(7).normal(0, 0.5, size=(L, 20))
    base_snapshot = base.copy()
    survivors = ["AKLDG"] * 10
    cfg = IterationBiasConfig(
        consensus_threshold=0.5, consensus_strength=2.0,
        only_at_classes=("surface",), max_augmented_fraction=1.0,
    )
    out, _ = build_iteration_bias(
        base_bias=base,
        survivor_sequences=survivors,
        position_classes=["surface"] * L,
        config=cfg,
    )
    # The caller's array is unmodified...
    assert np.array_equal(base, base_snapshot)
    # ...even though the returned array IS modified.
    assert not np.array_equal(out, base)


def test_build_iteration_bias_is_idempotent_under_repeat_calls():
    """Calling ``build_iteration_bias`` twice with the same inputs gives
    the same result and does not double-augment. This is what ``run_cycle``
    relies on when it (logically) reapplies cycle-k consensus on the
    SAME ``base_bias``."""
    L = 5
    base = np.random.RandomState(0).normal(0, 0.5, size=(L, 20))
    survivors = ["AKLDG"] * 10
    cfg = IterationBiasConfig(
        consensus_threshold=0.5, consensus_strength=2.0,
        only_at_classes=("surface",), max_augmented_fraction=1.0,
    )
    out1, _ = build_iteration_bias(base, survivors, ["surface"] * L, config=cfg)
    out2, _ = build_iteration_bias(base, survivors, ["surface"] * L, config=cfg)
    assert np.array_equal(out1, out2)


# ----------------------------------------------------------------------
# Cap behavior + ordering
# ----------------------------------------------------------------------


def test_cap_picks_highest_agreement_first():
    """When more positions exceed the consensus threshold than the cap
    allows, ``build_iteration_bias`` keeps the positions with highest
    top-AA frequency. Verify that a position with 100% consensus beats
    one with 90% consensus when only one slot remains."""
    L = 10
    base = _zero_bias(L)
    # Position 0: 100% A; Position 1: 90% A + 10% K; rest: 50/50 (below).
    seqs = []
    # 10 sequences total
    for k in range(10):
        # First two positions: sample chooses A or K according to plan
        c0 = "A"
        c1 = "A" if k < 9 else "K"   # 90% A
        rest = "AG" * 4   # length 8, deterministic but non-consensus
        seqs.append(c0 + c1 + rest)
    cfg = IterationBiasConfig(
        consensus_threshold=0.85, consensus_strength=2.0,
        only_at_classes=("surface",),
        max_augmented_fraction=0.10,   # cap = 1
    )
    out, telem = build_iteration_bias(base, seqs, ["surface"] * L, config=cfg)
    a_idx = _AA_TO_IDX["A"]
    # Only ONE position augmented, and it must be position 0 (100% > 90%).
    n_aug = int((out[:, a_idx] >= 1.99).sum())
    assert n_aug == 1
    assert out[0, a_idx] == pytest.approx(2.0)
    assert out[1, a_idx] == 0.0
    assert telem.capped is True
    assert telem.n_positions_augmented == 1


def test_cap_zero_when_max_fraction_zero():
    """``max_augmented_fraction=0.0`` => cap = 0 => never augment."""
    L = 5
    base = _zero_bias(L)
    seqs = ["AAAAA"] * 10
    cfg = IterationBiasConfig(
        consensus_threshold=0.5, consensus_strength=2.0,
        only_at_classes=("surface",),
        max_augmented_fraction=0.0,
    )
    out, telem = build_iteration_bias(base, seqs, ["surface"] * L, config=cfg)
    assert np.array_equal(out, base)
    assert telem.n_positions_augmented == 0
    assert telem.capped is True   # candidates>cap=0 triggered the cap path


# ----------------------------------------------------------------------
# Resno labelling: array-index + 1 vs PDB resseq
# ----------------------------------------------------------------------


def test_augmented_resnos_are_array_index_plus_one_not_pdb_resseq():
    """``IterationBiasTelemetry.augmented_resnos`` is currently
    ``array_index + 1``, NOT the protein PDB resseq. For PTE_i1 these
    coincide because protein_resnos start at 1, but if the construct
    started at e.g. resseq 5 the labels would be off.

    The v2 driver only logs these for human inspection (telemetry.json),
    not for any downstream lookup, so this is acceptable. Test
    documents the actual semantics so a future caller doesn't get
    surprised.
    """
    L = 5
    base = _zero_bias(L)
    seqs = ["AAAAA"] * 10
    cfg = IterationBiasConfig(
        consensus_threshold=0.5, consensus_strength=2.0,
        only_at_classes=("surface",), max_augmented_fraction=1.0,
    )
    _, telem = build_iteration_bias(base, seqs, ["surface"] * L, config=cfg)
    # Augmented at array indices 0..4 -> labels 1..5.
    assert telem.augmented_resnos == [1, 2, 3, 4, 5]


# ----------------------------------------------------------------------
# Catalytic-residue protection — fixed_resnos_zero_indexed must dominate
# ----------------------------------------------------------------------


def test_fixed_resnos_protect_position_even_at_full_consensus():
    """100% consensus on a fixed (catalytic) position must NOT augment.
    This is the structural-integrity guarantee for HIS 60/64/128/132 +
    LYS 157 + GLU 131 in the PTE_i1 pipeline."""
    L = 6
    base = _zero_bias(L)
    seqs = ["HHHHHH"] * 20   # full consensus on H at every position
    cfg = IterationBiasConfig(
        consensus_threshold=0.5, consensus_strength=2.0,
        only_at_classes=("surface",), max_augmented_fraction=1.0,
    )
    # Pretend positions 1, 3, 4 are catalytic (zero-indexed).
    out, telem = build_iteration_bias(
        base, seqs, ["surface"] * L,
        fixed_resnos_zero_indexed=[1, 3, 4],
        config=cfg,
    )
    h_idx = _AA_TO_IDX["H"]
    # Only positions 0, 2, 5 should have been augmented.
    assert out[0, h_idx] == pytest.approx(2.0)
    assert out[1, h_idx] == 0.0   # fixed
    assert out[2, h_idx] == pytest.approx(2.0)
    assert out[3, h_idx] == 0.0   # fixed
    assert out[4, h_idx] == 0.0   # fixed
    assert out[5, h_idx] == pytest.approx(2.0)
    assert telem.n_positions_augmented == 3


def test_fixed_resnos_dedup_internally():
    """Duplicates in the fixed_resnos list shouldn't break anything."""
    L = 4
    base = _zero_bias(L)
    seqs = ["AAAA"] * 10
    cfg = IterationBiasConfig(
        consensus_threshold=0.5, consensus_strength=2.0,
        only_at_classes=("surface",), max_augmented_fraction=1.0,
    )
    out, telem = build_iteration_bias(
        base, seqs, ["surface"] * L,
        fixed_resnos_zero_indexed=[2, 2, 2, 2],
        config=cfg,
    )
    a_idx = _AA_TO_IDX["A"]
    assert out[0, a_idx] == pytest.approx(2.0)
    assert out[1, a_idx] == pytest.approx(2.0)
    assert out[2, a_idx] == 0.0
    assert out[3, a_idx] == pytest.approx(2.0)
    assert telem.n_positions_augmented == 3


# ----------------------------------------------------------------------
# consensus_aa_frequencies - more degenerate inputs
# ----------------------------------------------------------------------


def test_consensus_position_with_only_non_canonical_yields_zero_freq():
    """A position that is 'X' in every survivor has zero canonical
    counts; ``consensus_aa_frequencies`` must return all zeros for that
    row (not NaN), so ``top_freq`` for that position is 0 and it never
    crosses the consensus threshold."""
    L = 3
    seqs = ["MXV", "MXV", "MXV"]
    out = consensus_aa_frequencies(seqs, L)
    # Position 1 is 'X' for all survivors -> entire row zero.
    assert (out[1, :] == 0).all()
    # Other positions look normal.
    assert out[0, _AA_TO_IDX["M"]] == pytest.approx(1.0)
    assert out[2, _AA_TO_IDX["V"]] == pytest.approx(1.0)


def test_consensus_lowercase_input_normalized():
    """Sequences arriving in lowercase (which shouldn't happen post-dedup
    but might if someone calls consensus_aa_frequencies directly) are
    upper-cased before counting."""
    out = consensus_aa_frequencies(["mkvla", "MKVLA"], L=5)
    for i, c in enumerate("MKVLA"):
        assert out[i, _AA_TO_IDX[c]] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Telemetry shape sanity
# ----------------------------------------------------------------------


def test_telemetry_n_survivors_matches_input_length():
    L = 4
    base = _zero_bias(L)
    seqs = ["AAAA"] * 7
    out, telem = build_iteration_bias(base, seqs, ["active_site"] * L)
    assert telem.n_survivors == 7
    # active_site is not in default only_at_classes ... wait, default *is*
    # ('buried', 'surface', 'first_shell', 'pocket'). So eligible=0.
    assert telem.n_positions_eligible == 0
    assert telem.n_positions_augmented == 0


def test_telemetry_n_eligible_excludes_fixed_positions():
    """Eligible count subtracts fixed positions even if their class is
    in only_at_classes — exactly matches the eligible_mask construction."""
    L = 5
    base = _zero_bias(L)
    seqs = ["AAAAA"] * 10
    cfg = IterationBiasConfig(only_at_classes=("surface",))
    out, telem = build_iteration_bias(
        base, seqs, ["surface"] * L,
        fixed_resnos_zero_indexed=[0, 2],
        config=cfg,
    )
    assert telem.n_positions_eligible == 3   # 5 surface - 2 fixed
