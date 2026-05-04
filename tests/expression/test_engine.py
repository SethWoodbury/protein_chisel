"""Tests for the ExpressionRuleEngine + builtin rules.

Covers each rule's positive case, negative case, severity overrides,
structure-aware modulation, and the engine's aggregation logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from protein_chisel.expression import (
    ExpressionRuleEngine, ExpressionProfile, REGISTRY, Severity,
)
import protein_chisel.expression.builtin_rules  # noqa: F401 — registers


def _engine(profile=None):
    if profile is None:
        profile = ExpressionProfile.bl21_cytosolic_streptag()
    return ExpressionRuleEngine(profile=profile)


def _no_tag_engine():
    """Profile with no tags — terminus rules NOT short-circuited.

    Used for tests that want to exercise the terminus rules directly.
    With the default Strep-tag profile, ssra/hydrophobic_c_tail/
    n_end_rule/signal_peptide/tat/lipobox short-circuit because the
    designed body's termini are internal in the full construct.
    """
    p = ExpressionProfile.bl21_cytosolic_streptag()
    return ExpressionRuleEngine(
        profile=type(p)(**{**p.__dict__, "n_tag": "", "c_tag": ""}),
    )


# ----------------------------------------------------------------------
# Severity ordering + overrides
# ----------------------------------------------------------------------


def test_severity_promote_demote():
    assert Severity.WARN_ONLY.promote() == Severity.SOFT_BIAS
    assert Severity.SOFT_BIAS.promote() == Severity.HARD_OMIT
    assert Severity.HARD_OMIT.promote() == Severity.HARD_FILTER
    assert Severity.HARD_FILTER.promote() == Severity.HARD_FILTER  # ceiling
    assert Severity.HARD_FILTER.demote() == Severity.HARD_OMIT
    assert Severity.WARN_ONLY.demote() == Severity.WARN_ONLY  # floor


def test_profile_overrides_string_parse():
    base = ExpressionProfile.bl21_cytosolic_streptag()
    out = ExpressionProfile.from_overrides_string(
        base, "kr_neighbor_dibasic=SOFT_BIAS,polyproline_stall=HARD_FILTER",
    )
    assert out.severity_overrides["kr_neighbor_dibasic"] == Severity.SOFT_BIAS
    assert out.severity_overrides["polyproline_stall"] == Severity.HARD_FILTER


def test_profile_overrides_string_empty_returns_base():
    base = ExpressionProfile.bl21_cytosolic_streptag()
    assert ExpressionProfile.from_overrides_string(base, "") is base
    assert ExpressionProfile.from_overrides_string(base, "  ") is base


def test_profile_overrides_string_bad_severity_raises():
    base = ExpressionProfile.bl21_cytosolic_streptag()
    with pytest.raises(ValueError):
        ExpressionProfile.from_overrides_string(base, "rule_x=BOGUS")


def test_profile_overrides_string_missing_eq_raises():
    base = ExpressionProfile.bl21_cytosolic_streptag()
    with pytest.raises(ValueError):
        ExpressionProfile.from_overrides_string(base, "rule_x SOFT_BIAS")


# ----------------------------------------------------------------------
# C-terminal degron rules
# ----------------------------------------------------------------------


def test_ssra_exact_fires():
    res = _no_tag_engine().evaluate("MSGDESIGNAANDENYALAA")
    fail = [h for h in res.hits if h.rule_name == "ssra_tag_cterm"]
    assert fail, "ssrA_tag_cterm did not fire on exact AANDENYALAA"
    assert fail[0].severity == Severity.HARD_FILTER
    assert not res.passes_hard_filter()


def test_ssra_does_not_fire_on_strep_tag():
    """Strep-tag II ends in WSHPQFEK — must not collide with ssrA-like."""
    res = _engine().evaluate("MSGDESIGNGSAWSHPQFEK")
    fail = [h for h in res.hits if h.rule_name == "ssra_tag_cterm"]
    assert not fail


def test_hydrophobic_c_tail_fires_on_VLAAA():
    res = _no_tag_engine().evaluate("MSGEEEDDDLLLLEEDDLAAGSVAAVLAAA")
    fail = [h for h in res.hits if h.rule_name == "hydrophobic_c_tail"]
    assert fail, "hydrophobic_c_tail did not fire on VLAAA"


def test_hydrophobic_c_tail_does_not_fire_on_charged_tail():
    res = _no_tag_engine().evaluate("MSGEEEDDDLLLLEEDDLAAGSEEDDDK")
    fail = [h for h in res.hits if h.rule_name == "hydrophobic_c_tail"]
    assert not fail


# ----------------------------------------------------------------------
# N-terminal rules
# ----------------------------------------------------------------------


def test_metap_warning_on_MSG():
    res = _engine().evaluate("MSGDESIGNGSEEDDK")
    metap = [h for h in res.hits if h.rule_name == "metap_warning"]
    assert metap, "metap_warning should fire when M-S- starts the sequence"
    assert metap[0].severity == Severity.WARN_ONLY


def test_metap_warning_does_not_fire_on_ML():
    """L is not in MetAP small-AA set, so leading M is NOT cleaved."""
    res = _engine().evaluate("MLAGGGSEEEDDLAARRGSEEDDK")
    metap = [h for h in res.hits if h.rule_name == "metap_warning"]
    assert not metap


def test_n_end_rule_fires_on_no_M_start_with_L():
    """Sequence starting with L (no M, no MetAP) -> mature N-term L = degron."""
    res = _no_tag_engine().evaluate("LDESIGNGSAEEDDLLEEDDDDK")
    n_end = [h for h in res.hits if h.rule_name == "n_end_rule_destabilizing"]
    assert n_end, "n_end_rule should fire when seq[0] = L"
    assert n_end[0].severity == Severity.HARD_FILTER


def test_signal_peptide_fires_on_synthetic_sec():
    """Synthetic Sec signal: basic n-region + hydrophobic h-region + AXA."""
    res = _no_tag_engine().evaluate(
        "MKKLLALAVLAAFAQAAAGSAEEEDDDLLLLEKKK" + "A" * 50,
    )
    sp = [h for h in res.hits if h.rule_name == "signal_peptide_n_term"]
    assert sp, "signal_peptide_n_term should fire on synthetic Sec signal"


def test_signal_peptide_does_not_fire_on_acidic_n_term():
    res = _engine().evaluate("MSGEEEDDDLLLLEDDDLAAEEEDDLLEEDDLAARRGGGGGGGGGGSEEDDK")
    sp = [h for h in res.hits if h.rule_name == "signal_peptide_n_term"]
    assert not sp


def test_tat_motif_fires():
    res = _no_tag_engine().evaluate("MNSRRTFLKAAEEDDLLEEDDLAAGSEEDDK")
    tat = [h for h in res.hits if h.rule_name == "tat_signal_motif"]
    assert tat


def test_lipobox_fires():
    res = _no_tag_engine().evaluate("MSGAAAALAGCAAEEDDLLEEDDLAAGSEEDDK")
    lb = [h for h in res.hits if h.rule_name == "lipobox_n_term"]
    assert lb


# ----------------------------------------------------------------------
# Internal rules
# ----------------------------------------------------------------------


def test_kr_neighbor_dibasic_fires_in_loop_as_hard_omit():
    """Surface-loop neighbor of fixed K -> HARD_OMIT."""
    seq = "AAA" + "K" + "AAA" + "G" * 100   # K at index 3 = catalytic
    L = len(seq)
    ss = "L" * L                              # all loop
    sasa = np.full(L, 60.0)                   # all surface
    res = _engine().evaluate(
        seq, ss_reduced=ss, sasa=sasa,
        catalytic_resnos=[4],                 # 1-indexed = index 3
        fixed_resnos=[4],
    )
    kr = [h for h in res.hits if h.rule_name == "kr_neighbor_dibasic"]
    assert any(h.severity == Severity.HARD_OMIT for h in kr)
    omit = res.hard_omit_per_residue()
    # neighbors of index 3 are 2 and 4
    assert 2 in omit and 4 in omit
    assert omit[2] == "KR" and omit[4] == "KR"


def test_kr_neighbor_dibasic_in_helix_is_soft_bias():
    """Surface-helix neighbor of fixed K -> SOFT_BIAS, not HARD_OMIT.

    This is the PTE_i1 K157-K158 case: helix context, so we don't
    forbid sample-time, just downweight in the bias matrix.
    """
    seq = "AAA" + "K" + "AAA" + "GGG"
    L = len(seq)
    ss = "H" * L
    sasa = np.full(L, 60.0)
    res = _engine().evaluate(
        seq, ss_reduced=ss, sasa=sasa,
        catalytic_resnos=[4], fixed_resnos=[4],
    )
    kr = [h for h in res.hits if h.rule_name == "kr_neighbor_dibasic"]
    assert kr
    assert all(h.severity == Severity.SOFT_BIAS for h in kr)
    assert not res.hard_omit_per_residue()
    assert res.soft_bias_per_residue()


def test_kr_neighbor_dibasic_buried_is_warn_only():
    """Buried neighbor of fixed K -> WARN_ONLY only."""
    seq = "AAAKAAA"
    L = len(seq)
    ss = "H" * L
    sasa = np.full(L, 5.0)                    # buried
    res = _engine().evaluate(
        seq, ss_reduced=ss, sasa=sasa,
        catalytic_resnos=[4], fixed_resnos=[4],
    )
    kr = [h for h in res.hits if h.rule_name == "kr_neighbor_dibasic"]
    assert kr
    assert all(h.severity == Severity.WARN_ONLY for h in kr)


def test_kr_neighbor_skipped_without_structure():
    """Without SS/SASA, requires_structure=True rules are silently skipped."""
    res = _engine().evaluate("AAAKAAA", catalytic_resnos=[4], fixed_resnos=[4])
    kr = [h for h in res.hits if h.rule_name == "kr_neighbor_dibasic"]
    assert not kr


def test_kr_neighbor_severity_override_forces_hard_omit():
    """User can override structure-aware behavior with a forced severity."""
    seq = "AAAKAAA"
    L = len(seq)
    ss = "H" * L
    sasa = np.full(L, 60.0)
    profile = ExpressionProfile.bl21_cytosolic_streptag().with_overrides({
        "kr_neighbor_dibasic": Severity.HARD_OMIT,
    })
    res = ExpressionRuleEngine(profile).evaluate(
        seq, ss_reduced=ss, sasa=sasa,
        catalytic_resnos=[4], fixed_resnos=[4],
    )
    kr = [h for h in res.hits if h.rule_name == "kr_neighbor_dibasic"]
    assert all(h.severity == Severity.HARD_OMIT for h in kr)


def test_polyproline_fires_on_PPP():
    res = _engine().evaluate("MSGDESPPPDESIGNGSEEDDLLEEDDLAARRGGGGGSEEDDK")
    pp = [h for h in res.hits if h.rule_name == "polyproline_stall"]
    assert pp
    assert pp[0].severity == Severity.SOFT_BIAS


def test_secm_arrest_fires_on_exact_motif():
    res = _engine().evaluate(
        "MSGAEEFSTPVWISQAQGIRAGPDESIGNGSEEDDLLEEDDLAARRGSEEDDK",
    )
    sm = [h for h in res.hits if h.rule_name == "secm_arrest"]
    assert sm
    assert sm[0].severity == Severity.HARD_FILTER


def test_cytosolic_disulfide_overload_fires_with_5_cys():
    res = _engine().evaluate(
        "MSGCCCCCEEEDDLLEEDDLAARRGGGGGGGGGGGSEEDDK",
    )
    cy = [h for h in res.hits if h.rule_name == "cytosolic_disulfide_overload"]
    assert cy
    assert cy[0].severity == Severity.SOFT_BIAS


def test_cytosolic_disulfide_does_not_fire_with_2_cys():
    res = _engine().evaluate(
        "MSGCCEEEEDDLLEEDDLAARRGGGGGGGGGGGSEEDDK",
    )
    cy = [h for h in res.hits if h.rule_name == "cytosolic_disulfide_overload"]
    assert not cy


# ----------------------------------------------------------------------
# Tag-protease rules
# ----------------------------------------------------------------------


def test_tev_internal_warns_when_no_cleave_protease():
    res = _engine().evaluate("MSGENLYFQGAEEEDDLLEKKGGSEEDDK")
    tev = [h for h in res.hits if h.rule_name == "tev_site_internal"]
    assert tev
    assert tev[0].severity == Severity.WARN_ONLY


def test_tev_internal_filters_when_user_uses_TEV():
    profile = ExpressionProfile.bl21_cytosolic_streptag()
    profile = type(profile)(**{**profile.__dict__, "cleave_protease": "TEV"})
    res = ExpressionRuleEngine(profile=profile).evaluate(
        "MSGENLYFQGAEEEDDLLEKKGGSEEDDK",
    )
    tev = [h for h in res.hits if h.rule_name == "tev_site_internal"]
    assert tev
    assert tev[0].severity == Severity.HARD_FILTER


# ----------------------------------------------------------------------
# Engine aggregation
# ----------------------------------------------------------------------


def test_engine_aggregates_multiple_hard_omit_to_per_residue_dict():
    """Two rules each forbidding K at the same position -> KR set merged."""
    L = 10
    seq = "AAAAKAAAAA"
    ss = "L" * L
    sasa = np.full(L, 60.0)
    res = _engine().evaluate(
        seq, ss_reduced=ss, sasa=sasa,
        catalytic_resnos=[5], fixed_resnos=[5],
    )
    omit = res.hard_omit_per_residue()
    # neighbors of index 4 are 3 and 5
    for pos in (3, 5):
        assert pos in omit
        assert "K" in omit[pos] and "R" in omit[pos]


def test_engine_to_omit_AA_json_uses_protein_resnos():
    L = 10
    seq = "AAAAKAAAAA"
    ss = "L" * L
    sasa = np.full(L, 60.0)
    # Protein chain with a gap: resnos 100..109 except 105 missing -> we
    # supply [100,101,102,103,104,106,107,108,109,110] (length 10).
    # Catalytic at PDB resno 104 (= body index 4).
    protein_resnos = [100, 101, 102, 103, 104, 106, 107, 108, 109, 110]
    res = _engine().evaluate(
        seq, ss_reduced=ss, sasa=sasa,
        catalytic_resnos=[104], fixed_resnos=[104],
        protein_resnos=protein_resnos,
    )
    j = res.to_omit_AA_json("A", protein_resnos=protein_resnos)
    # neighbors of body index 4 are 3 and 5 -> PDB resnos 103 and 106
    assert "A103" in j and "A106" in j


def test_engine_passes_hard_filter_when_clean():
    # 200-residue length, AA composition close to enzyme distribution
    # so the AA-comp rules don't fire. Short test sequences (~25 aa)
    # naturally show extreme z-scores due to small-sample noise.
    seq = (
        "MSGAEEDDLLEEDDLAARRGSEEDDKAAGGGSSTTLLIIVVAAAGSEEDDLLAARRGS"
        "MEEEDDLLAARRGSGSGSEEDDLLAARRGSAAAEEDDLLAARRGSEEDDLLAARRGSEED"
        "GSGSGSAEEDDLLAARRGSEEDDLLAARRGSAEEDDLLAARRGSEEDDLLAARRGSEED"
        "AAGGGSEEDDLLAARRGSEED"
    )
    res = _engine().evaluate(seq)
    assert res.passes_hard_filter(), f"unexpected fails: {res.fail_reasons()}"


def test_engine_returns_zero_hits_for_nonsense_short_sequence():
    res = _engine().evaluate("M")
    # Should not crash; rules that need >= some length silently produce 0.
    assert isinstance(res.hits, list)
