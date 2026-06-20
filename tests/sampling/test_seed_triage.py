"""Seed triage: detect a pathologically hydrophobic / over-represented INPUT scaffold.

The PLM fusion bias is conditioned on the seed, so on a pathological seed it AMPLIFIES
the bad composition (an ~8700x lock at T=0.15 — see sampling.bias_scale). Empirically,
dropping the PLM (plm_strength=0) on such a seed took GRAVY 1.34->-0.50 and Ala 27%->0.5%.
``assess_seed`` is the pure detector that an opt-in policy uses to decide whether to skip
the PLMs for a run. Pure + reference-free so it generalizes to any scaffold.
"""
import pytest

from protein_chisel.sampling.seed_triage import (
    assess_seed, should_skip_plm, SeedAssessment,
)


def _patho(p):
    return SeedAssessment(pathological=p, reasons=["x"] if p else [],
                          gravy=0.0, max_aa="A", max_aa_frac=0.3, hydrophobic_frac=0.6)


def test_should_skip_plm_only_when_enabled_pathological_and_plm_on():
    # the actionable policy: skip PLM iff enabled AND pathological AND plm currently > 0
    assert should_skip_plm(_patho(True), enabled=True, current_plm_strength=1.25) is True
    assert should_skip_plm(_patho(True), enabled=False, current_plm_strength=1.25) is False  # off
    assert should_skip_plm(_patho(False), enabled=True, current_plm_strength=1.25) is False  # clean seed
    assert should_skip_plm(_patho(True), enabled=True, current_plm_strength=0.0) is False    # already off
    assert should_skip_plm(None, enabled=True, current_plm_strength=1.25) is False           # no assessment


def test_clean_soluble_seed_is_not_pathological():
    # Balanced, charge-rich, low hydrophobic-fraction sequence.
    seq = ("DEKRNQSTHGDEKRNQSTHGYDEKRNQSTHG" * 6)
    a = assess_seed(seq, gravy=-0.4)
    assert isinstance(a, SeedAssessment)
    assert a.pathological is False
    assert a.reasons == []


def test_high_gravy_trips_even_with_balanced_composition():
    seq = ("ACDEFGHIKLMNPQRSTVWY" * 10)            # perfectly even composition
    a = assess_seed(seq, gravy=1.34)
    assert a.pathological
    assert any("gravy" in r.lower() for r in a.reasons)


def test_over_represented_single_aa_trips():
    seq = ("A" * 60) + ("DEKRNQSTHG" * 14)          # Ala ~30%
    a = assess_seed(seq, gravy=-0.2)                # GRAVY fine; composition is the problem
    assert a.pathological
    assert a.max_aa == "A"
    assert a.max_aa_frac > 0.16
    assert any(r.startswith("A ") or "Ala" in r or "'A'" in r for r in a.reasons)


def test_high_hydrophobic_fraction_trips():
    seq = ("AVLIMFWC" * 20) + ("DEKR" * 5)          # ~89% hydrophobic, no single AA huge
    a = assess_seed(seq, gravy=0.39)                # just under the gravy cap on purpose
    assert a.pathological
    assert any("hydrophobic" in r.lower() for r in a.reasons)


def test_thresholds_are_configurable():
    seq = ("A" * 40) + ("DEKRNQ" * 10)              # Ala ~40%, GRAVY-ish moderate
    strict = assess_seed(seq, gravy=0.5)
    assert strict.pathological                       # trips default thresholds
    loose = assess_seed(seq, gravy=0.5, gravy_max=1.0,
                        max_aa_frac=0.5, hydrophobic_frac_max=0.95)
    assert loose.pathological is False               # all thresholds loosened past the seed


def test_assessment_carries_measured_values():
    seq = ("L" * 50) + ("DEKRNQSTHG" * 15)
    a = assess_seed(seq, gravy=0.8)
    assert a.gravy == 0.8
    assert 0.0 <= a.hydrophobic_frac <= 1.0
    assert a.max_aa == "L"
    assert a.max_aa_frac == pytest.approx(50 / (50 + 150))
