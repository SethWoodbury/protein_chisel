"""Tests for AA-composition helpers used by WS-C composition control.

``over_cap_aas`` underpins the opt-in ``--aa_fraction_cap``: any amino acid
whose fraction in the survivor pool is at/over the cap is hard-omitted in the
next cycle, breaking runaway single-AA over-representation (e.g. the 26%-Ala
design that motivated the solubility-steering overhaul).
"""

from __future__ import annotations

import pytest

from protein_chisel.expression.aa_composition import over_cap_aas


# 26% A, 20% L, 20% G, 12% S, 8% T, 7% E, 7% D (sums to 200 residues).
_POOL = "A" * 52 + "L" * 40 + "G" * 40 + "S" * 24 + "T" * 16 + "E" * 14 + "D" * 14


def test_over_cap_aas_returns_members_at_or_over_cap():
    capped = over_cap_aas(_POOL, cap=0.15)
    # A (26%), L (20%), G (20%) are over a 15% cap.
    assert set(capped) == {"A", "G", "L"}
    # Minor members stay allowed.
    for aa in ("S", "T", "E", "D"):
        assert aa not in capped


def test_over_cap_aas_is_sorted_and_deduplicated():
    capped = over_cap_aas(_POOL, cap=0.15)
    assert capped == sorted(set(capped))
    assert isinstance(capped, list)
    assert all(isinstance(a, str) and len(a) == 1 for a in capped)


def test_over_cap_aas_boundary_is_inclusive():
    # Exactly 20% G with cap 0.20 must be capped (>= semantics).
    assert "G" in over_cap_aas(_POOL, cap=0.20)
    # Just above 20% excludes it.
    assert "G" not in over_cap_aas(_POOL, cap=0.2001)


def test_over_cap_aas_respects_exclude():
    seq = "C" * 60 + "A" * 140          # 30% C, 70% A
    capped = over_cap_aas(seq, cap=0.15, exclude_aas="C")
    assert "C" not in capped            # excluded even though 30% >= cap
    assert "A" in capped


def test_over_cap_aas_empty_when_none_over_cap():
    seq = "ACDEFGHIKLMNPQRSTVWY" * 10   # 5% each
    assert over_cap_aas(seq, cap=0.15) == []


def test_over_cap_aas_empty_sequence():
    assert over_cap_aas("", cap=0.15) == []
