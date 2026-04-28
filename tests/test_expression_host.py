"""Host tests for filters/expression_host."""

from __future__ import annotations

import pytest

from protein_chisel.filters.expression_host import (
    E_COLI_PROTEASE_SITES, GENERAL_FORBIDDEN, YEAST_PROTEASE_SITES,
    YEAST_PTM_SITES, get_host_patterns,
)
from protein_chisel.filters.protease_sites import find_protease_sites


def test_get_host_patterns_ecoli_includes_ompT():
    pats = get_host_patterns("ecoli")
    names = [n for n, _ in pats]
    assert "ompT_RR" in names
    assert "ompT_KK" in names
    assert "ompT_KR" in names


def test_get_host_patterns_yeast_includes_kex2():
    pats = get_host_patterns("yeast")
    names = [n for n, _ in pats]
    assert "kex2_KR" in names
    assert "kex2_RR" in names


def test_get_host_patterns_yeast_n_glycosylation():
    """Yeast pattern catches NX[ST] N-glycosylation."""
    pats = get_host_patterns("yeast")
    names = dict(pats)
    assert "n_glycosylation_NXS_NXT" in names


def test_get_host_patterns_unknown_raises():
    with pytest.raises(ValueError):
        get_host_patterns("mammalian")


def test_find_protease_sites_yeast_n_glycosylation():
    """A sequence with NXS should hit the yeast NXS pattern."""
    res = find_protease_sites("MGGGNASGGG", host="yeast")
    names = res.by_name()
    assert "n_glycosylation_NXS_NXT" in names


def test_find_protease_sites_ecoli_ompT_RR():
    res = find_protease_sites("MGGGRRGGG", host="ecoli")
    names = res.by_name()
    assert "ompT_RR" in names


def test_general_forbidden_poly_runs():
    """GENERAL_FORBIDDEN is included only when host= is set."""
    res = find_protease_sites("MGGGGGGGGG", host="ecoli")
    names = res.by_name()
    assert "poly_G_long" in names


def test_kex2_intended_kcx_not_filtered_default():
    """E. coli host pattern set must NOT include 'intended_kcx' by default."""
    pats = get_host_patterns("ecoli")
    names = [n for n, _ in pats]
    assert "intended_kcx" not in names
