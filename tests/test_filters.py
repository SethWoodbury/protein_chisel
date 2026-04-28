"""Tests for filters/protparam, filters/protease_sites, filters/length.

protparam needs Biopython, which is in esmc.sif and likely present
on host pythons that have biopython installed.
"""

from __future__ import annotations

import pytest


# ---- protparam ------------------------------------------------------------


# Skip protparam tests if biopython isn't available on the runner.
biopython = pytest.importorskip("Bio.SeqUtils.ProtParam")


def test_protparam_metrics_basic():
    from protein_chisel.filters.protparam import protparam_metrics

    # Ubiquitin (76 aa) — well-characterized reference protein
    ub = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    res = protparam_metrics(ub)
    assert res.length == 76
    # Ubiquitin pI is ~6.5 (per Expasy ProtParam)
    assert 5.5 < res.pi < 7.5
    # Molecular weight ~8.5 kDa
    assert 8000 < res.molecular_weight < 9500


def test_protparam_charge_at_pH7_no_HIS():
    from protein_chisel.filters.protparam import protparam_metrics

    # Add 5 lysines and 5 aspartates; expect charge_no_HIS ≈ 0
    seq = "K" * 5 + "D" * 5
    res = protparam_metrics(seq)
    # K is fully protonated at pH 7 (pKa 10.5), D is fully deprotonated (pKa 3.65)
    # Net = +5 - 5 + N-term(+1) - C-term(-1) ≈ 0
    assert -1.0 < res.charge_at_pH7_no_HIS < 1.0


def test_protparam_charge_with_histidines():
    from protein_chisel.filters.protparam import protparam_metrics

    # 10 histidines: charge_at_pH7 includes them, no_HIS excludes them.
    seq = "H" * 10
    res = protparam_metrics(seq)
    # standard charge_at_pH7 sees ~half-protonated histidines (small +)
    assert res.charge_at_pH7 > -1
    # no_HIS variant excludes them, only N-term (+1) and C-term (-1) → ~0
    assert -0.5 < res.charge_at_pH7_no_HIS < 1.5


def test_protparam_to_dict_keys():
    from protein_chisel.filters.protparam import protparam_metrics

    res = protparam_metrics("MGGGGAAAAA")
    d = res.to_dict()
    assert "protparam__pi" in d
    assert "protparam__charge_at_pH7_no_HIS" in d
    assert "protparam__instability_index" in d
    assert "protparam__gravy" in d


def test_protparam_strips_non_canonical():
    """Non-canonical AAs are stripped before analysis (no exception)."""
    from protein_chisel.filters.protparam import protparam_metrics

    res = protparam_metrics("MGXJZBAAA")
    # X J Z B all stripped → only canonical: MGAAA (5 chars)
    assert res.length == 5


# ---- protease_sites -------------------------------------------------------


def test_find_protease_sites_default():
    from protein_chisel.filters.protease_sites import find_protease_sites

    # RR matches kex2_RR; LVPR matches thrombin
    seq = "MGGRRGGGGGGLVPRGSAAA"
    res = find_protease_sites(seq)
    names = res.by_name()
    assert "kex2_RR" in names
    assert "thrombin" in names


def test_find_protease_sites_clean():
    from protein_chisel.filters.protease_sites import find_protease_sites

    # No motifs in pure GGGS linker
    res = find_protease_sites("GGGGSGGGGSGGGGS")
    assert not res.has_any()


def test_protease_extra_patterns():
    from protein_chisel.filters.protease_sites import find_protease_sites

    res = find_protease_sites(
        "MGGFOOBARGGM",
        extra_patterns=[("custom_foobar", r"FOOBAR")],
    )
    assert any(h.name == "custom_foobar" for h in res.hits)


def test_protease_to_dict():
    from protein_chisel.filters.protease_sites import find_protease_sites

    res = find_protease_sites("RR" * 3)
    d = res.to_dict()
    assert "protease__n_total" in d
    assert d["protease__n_total"] >= 3


# ---- length ---------------------------------------------------------------


def test_length_filter_passes():
    from protein_chisel.filters.length import LengthFilterConfig, passes_length_filter

    ok, reason = passes_length_filter("MGAAAA", LengthFilterConfig(min_length=3, max_length=10))
    assert ok and reason == ""


def test_length_filter_too_short():
    from protein_chisel.filters.length import LengthFilterConfig, passes_length_filter

    ok, reason = passes_length_filter("MG", LengthFilterConfig(min_length=5))
    assert not ok
    assert "min_length" in reason


def test_length_filter_terminal_constraints():
    from protein_chisel.filters.length import LengthFilterConfig, passes_length_filter

    cfg = LengthFilterConfig(must_start_with="M")
    ok, _ = passes_length_filter("AGGG", cfg)
    assert not ok
    ok, _ = passes_length_filter("MGGG", cfg)
    assert ok


def test_length_filter_forbidden_terminal():
    from protein_chisel.filters.length import LengthFilterConfig, passes_length_filter

    cfg = LengthFilterConfig(forbidden_n_terminal=("P",))
    ok, _ = passes_length_filter("PGGG", cfg)
    assert not ok
    ok, _ = passes_length_filter("MGGG", cfg)
    assert ok
