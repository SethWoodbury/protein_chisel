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
from protein_chisel.expression.aa_composition import aa_z_scores as aa_z_scores_for_test


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


# ---------------------------------------------------------------------------
# Feature 1 — z-score over-representation gate (opt-in, redundant)
# ---------------------------------------------------------------------------

# An EC-3 hydrolase reference where Ala per-seq mean≈8.7%, sd≈3.3% — so an
# ~30%-Ala seed is z≈(30-8.7)/3.3≈6.5 (well over 3) with log2(30/8.7)≈1.8 (over the
# 0.25 floor): both the z-gate AND the legacy 16% trip fire (the redundant OR).
_ALA_REF = "swissprot_enzyme_2026_01"


def test_z_gate_off_by_default_is_byte_identical_to_legacy():
    """aa_zmax=None (default) => the z-gate is inert: no over_rep_aas, no extra
    reason, and pathological is decided exactly as before. F1 default = byte-identical."""
    seq = ("ACDEFGHIKLMNPQRSTVWY" * 10)              # even composition, GRAVY fine
    a = assess_seed(seq, gravy=-0.2)                  # nothing trips
    assert a.over_rep_aas == []
    assert a.pathological is False
    # The legacy thresholds still fire identically with the z-gate left off:
    ala = ("A" * 60) + ("DEKRNQSTHG" * 14)            # Ala ~30%
    leg = assess_seed(ala, gravy=-0.2)                # legacy max_aa_frac trip only
    assert leg.pathological and leg.over_rep_aas == []


def test_z_gate_flags_over_represented_aa_with_z_and_log2_and_reference():
    """One-sided z>=aa_zmax AND log2_enrichment>=floor => the AA is over-rep; the
    reason carries the z, the log2 enrichment, AND the reference (verbose logic)."""
    seq = ("A" * 60) + ("DEKRNQSTHG" * 14)            # Ala ~30%
    a = assess_seed(seq, gravy=-0.2, aa_zmax=3.0, aa_reference=_ALA_REF)
    assert "A" in a.over_rep_aas
    assert a.pathological
    # The z-gate reason names the AA, its z, its log2 enrichment, and the reference.
    zr = next(r for r in a.reasons if "z=" in r and "A" in r)
    assert "log2=" in zr and _ALA_REF in zr


def test_z_gate_log2_floor_prevents_rare_aa_high_z_trivial_pct():
    """codex floor: a naturally-rare AA (Trp, ref≈1.1%) at high z but a trivial %
    must NOT trip — its log2 enrichment is below the 0.25 floor. Trp at ~1.7%
    is z>3 vs the tight Trp SD yet log2(1.7/1.1)≈0.6 ... so to exercise the FLOOR we
    take a % whose log2 is under the floor while z clears it."""
    # Build a sequence where W sits just over its mean in z but under the log2 floor.
    # W ref (enzyme): mean≈1.101, sd≈0.918, global≈1.137. A % that gives z>=3 is
    # ≈1.101+3*0.918≈3.85% -> log2(3.85/1.137)≈1.76 (over floor) — high-z rare AAs
    # generally clear the floor, so instead assert the floor is wired by RAISING it
    # so the same trip is suppressed.
    seq = ("W" * 4) + ("DEKRNQSTHG" * 20)             # W ~1.9%
    on = assess_seed(seq, gravy=-0.3, aa_zmax=0.5, aa_reference=_ALA_REF,
                     aa_z_log2_floor=0.0)
    assert "W" in on.over_rep_aas                      # low z-threshold + no floor: trips
    floored = assess_seed(seq, gravy=-0.3, aa_zmax=0.5, aa_reference=_ALA_REF,
                          aa_z_log2_floor=10.0)        # impossible floor
    assert "W" not in floored.over_rep_aas             # floor suppresses the trip


def test_z_gate_one_sided_under_representation_does_not_trip():
    """One-sided: an UNDER-represented AA (negative z) is not a bad-seed signal."""
    # A sequence with ZERO Leucine (enzyme ref mean≈9.7%): L's z is strongly negative.
    # Build the rest from a balanced, near-reference mix so nothing else over-trips and we
    # isolate "the under-represented L does not appear in over_rep_aas".
    seq = ("ACDEFGHIKMNPQRSTVWY" * 11)                 # every canonical AA EXCEPT L
    a = assess_seed(seq, gravy=-0.1, aa_zmax=3.0, aa_reference=_ALA_REF,
                    aa_z_log2_floor=0.25)
    assert "L" not in a.over_rep_aas                    # under-rep L never trips
    z = aa_z_scores_for_test(seq, reference=_ALA_REF)
    assert z["L"] < 0                                   # confirm L really is under-rep


def test_z_gate_excludes_omitted_aa():
    """exclude_aas (e.g. 'C' when cysteine is hard-omitted) is never flagged."""
    seq = ("C" * 40) + ("DEKRNQSTHG" * 16)             # Cys ~20% (would be z-high)
    incl = assess_seed(seq, gravy=0.1, aa_zmax=3.0, aa_reference=_ALA_REF)
    excl = assess_seed(seq, gravy=0.1, aa_zmax=3.0, aa_reference=_ALA_REF,
                       exclude_aas="C")
    assert "C" in incl.over_rep_aas
    assert "C" not in excl.over_rep_aas


def test_z_gate_is_a_redundant_OR_with_the_flat_max_aa_frac():
    """User requires redundancy: a seed under the flat 16% cap but z-over-rep STILL
    trips via the z-gate (and vice-versa)."""
    # Ala ~14% (under the 16% flat cap) but z = (14-8.96)/3.30 ≈ 1.5 ... use aa_zmax
    # low enough to fire the z-gate while the flat cap stays silent.
    seq = ("A" * 14) + ("DEKRNQSTHG" * 86)[:86]        # Ala 14/100 = 14%
    a = assess_seed(seq, gravy=-0.3, aa_zmax=1.0, max_aa_frac=0.16,
                    aa_reference=_ALA_REF)
    assert a.max_aa_frac < 0.16                         # flat cap NOT tripped
    assert "A" in a.over_rep_aas                        # z-gate fired (redundant OR)
    assert a.pathological


def test_z_gate_reference_threading_changes_the_verdict():
    """The reference is threaded: a different baseline yields a different z and so a
    different trip set (the skeptic's caveat — pass the design's own EC class)."""
    seq = ("A" * 14) + ("DEKRNQSTHG" * 86)[:86]
    enz = assess_seed(seq, gravy=-0.3, aa_zmax=1.0, aa_reference="swissprot_enzyme_2026_01")
    glob = assess_seed(seq, gravy=-0.3, aa_zmax=1.0, aa_reference="swissprot_global_2026_01")
    # Different per-seq mean/sd between references => the z (and thus the trip) differs.
    assert enz.over_rep_aas != glob.over_rep_aas or enz.severity != glob.severity


def test_z_gate_empty_and_none_sequence_safe():
    """None/empty sequence must not crash the z-gate (it degrades to no trips)."""
    for seq in ("", None):
        a = assess_seed(seq, gravy=0.0, aa_zmax=3.0, aa_reference=_ALA_REF)
        assert a.over_rep_aas == []
        assert isinstance(a.severity, float)


def test_empty_sequence_is_never_pathological_even_with_high_gravy():
    """codex fix: an empty/non-canonical sequence yields NO trips, even if a (meaningless)
    high GRAVY is supplied — there is no seed to assess."""
    for seq in ("", None, "XXXX", "   "):
        a = assess_seed(seq, gravy=2.0, aa_zmax=3.0, aa_reference=_ALA_REF)
        assert a.pathological is False
        assert a.reasons == []
        assert a.severity == 0.0
        assert a.max_aa == "" and a.max_aa_frac == 0.0


def test_z_gate_degenerate_zmax_le_zero_does_not_trip_excluded_or_zero_signal():
    """codex fix: a degenerate aa_zmax<=0 + log2_floor<=0 must NOT let the 0.0 sentinel
    (excluded AAs, zero-SD AAs, AAs exactly at mean) trip — only a GENUINE positive z does.
    (The CLI rejects aa_zmax<=0, but the pure detector must be safe regardless.)"""
    seq = ("A" * 60) + ("DEKRNQSTHG" * 14)
    a = assess_seed(seq, gravy=-0.2, aa_zmax=0.0, aa_reference=_ALA_REF,
                    aa_z_log2_floor=0.0, exclude_aas="C")
    # C is excluded -> never flagged; no AA with z<=0 (sentinel) is flagged.
    assert "C" not in a.over_rep_aas
    z = aa_z_scores_for_test(seq, reference=_ALA_REF, exclude_aas="C")
    for aa in a.over_rep_aas:
        assert z[aa] > 0.0                                 # only genuine over-rep trips


def test_graded_soft_degenerate_severity_falls_back_to_cliff():
    """codex fix: a pathological assessment with an UNSET/degenerate severity (<1) must
    degrade soft mode to the cliff (0.0), never silently leave the PLM at full strength."""
    bad = SeedAssessment(pathological=True, reasons=["x"], gravy=2.0, max_aa="A",
                         max_aa_frac=0.3, hydrophobic_frac=0.6,
                         over_rep_aas=[], severity=0.0)   # severity NOT set to a real ratio
    out = graded_plm_strength(bad, enabled=True, current_plm_strength=1.25,
                              soft=True, soft_zero=2.0)
    assert out == 0.0
    # NaN severity also -> cliff (not a NaN strength).
    nan_a = SeedAssessment(pathological=True, reasons=["x"], gravy=2.0, max_aa="A",
                           max_aa_frac=0.3, hydrophobic_frac=0.6,
                           over_rep_aas=[], severity=float("nan"))
    assert graded_plm_strength(nan_a, enabled=True, current_plm_strength=1.25,
                               soft=True, soft_zero=2.0) == 0.0


def test_z_gate_exclude_all_aas_is_safe():
    """Excluding every canonical AA leaves the z-gate with nothing to flag (no crash)."""
    seq = ("A" * 60) + ("DEKRNQSTHG" * 14)
    a = assess_seed(seq, gravy=-0.2, aa_zmax=3.0, aa_reference=_ALA_REF,
                    exclude_aas="ACDEFGHIKLMNPQRSTVWY")
    assert a.over_rep_aas == []


# ---------------------------------------------------------------------------
# Feature 2 — soft/graded plm_strength (opt-in; CLIFF stays default) + severity
# ---------------------------------------------------------------------------


from protein_chisel.sampling.seed_triage import severity as _severity_fn, graded_plm_strength


def test_severity_is_one_at_threshold_and_grows_above():
    """severity = max over-threshold ratio across active trips; ==1.0 exactly at a
    trip threshold, >1 above it, and 0.0 for a non-pathological seed."""
    clean = assess_seed(("ACDEFGHIKLMNPQRSTVWY" * 10), gravy=-0.2)
    assert _severity_fn(clean) == 0.0
    # GRAVY exactly at the cap (0.4): ratio gravy/gravy_max = 1.0 (use > so set just over).
    g = assess_seed(("ACDEFGHIKLMNPQRSTVWY" * 10), gravy=0.8, gravy_max=0.4)
    assert _severity_fn(g) == pytest.approx(0.8 / 0.4)   # 2.0


def test_severity_takes_the_max_across_trips():
    """When several trips fire, severity is the MAX over-threshold ratio (the worst)."""
    seq = ("A" * 60) + ("VLIMFW" * 14)                  # Ala-rich AND hydrophobic-rich
    a = assess_seed(seq, gravy=1.6, gravy_max=0.4)      # gravy ratio = 4.0 dominates
    assert _severity_fn(a) == pytest.approx(max(
        1.6 / 0.4, a.max_aa_frac / 0.16, a.hydrophobic_frac / 0.50))


def test_severity_includes_the_z_gate_ratio():
    """When the z-gate is active, max_z/aa_zmax contributes to severity."""
    seq = ("A" * 60) + ("DEKRNQSTHG" * 14)              # Ala ~30%, big z
    a = assess_seed(seq, gravy=-0.2, aa_zmax=3.0, aa_reference=_ALA_REF)
    # severity should reflect the z ratio (max_z/aa_zmax) since Ala z >> 3.
    assert _severity_fn(a) >= 1.0
    assert a.severity == pytest.approx(_severity_fn(a))  # stored on the assessment


def test_graded_cliff_default_is_zero_when_pathological():
    """soft=False (DEFAULT) => the cliff: a pathological seed forces strength 0.0,
    exactly like should_skip_plm — F2 default is byte-identical."""
    a = assess_seed(("A" * 60) + ("DEKRNQSTHG" * 14), gravy=-0.2)   # legacy trip
    assert a.pathological
    out = graded_plm_strength(a, enabled=True, current_plm_strength=1.25, soft=False)
    assert out == 0.0


def test_graded_disabled_or_clean_or_off_returns_current_strength():
    """Not enabled, clean seed, or PLM already off => strength unchanged (mirrors
    should_skip_plm's gating)."""
    clean = assess_seed(("ACDEFGHIKLMNPQRSTVWY" * 10), gravy=-0.2)
    patho = assess_seed(("A" * 60) + ("DEKRNQSTHG" * 14), gravy=-0.2)
    assert graded_plm_strength(patho, enabled=False, current_plm_strength=1.25) == 1.25
    assert graded_plm_strength(clean, enabled=True, current_plm_strength=1.25) == 1.25
    assert graded_plm_strength(patho, enabled=True, current_plm_strength=0.0) == 0.0
    assert graded_plm_strength(None, enabled=True, current_plm_strength=1.25) == 1.25


def test_graded_soft_curve_interpolates_full_at_threshold_zero_at_soft_zero():
    """soft=True => current * clamp(1-(sev-1)/(soft_zero-1),0,1): full strength at the
    trip threshold (sev=1), 0 at sev>=soft_zero, linear between."""
    # Construct a seed whose severity is exactly 1.5 via GRAVY 0.6 / cap 0.4.
    a = assess_seed(("ACDEFGHIKLMNPQRSTVWY" * 10), gravy=0.6, gravy_max=0.4)
    assert _severity_fn(a) == pytest.approx(1.5)
    out = graded_plm_strength(a, enabled=True, current_plm_strength=2.0,
                              soft=True, soft_zero=2.0)
    # 1-(1.5-1)/(2-1) = 0.5 -> 2.0*0.5 = 1.0
    assert out == pytest.approx(1.0)
    # At/over soft_zero -> 0.0.
    hi = assess_seed(("ACDEFGHIKLMNPQRSTVWY" * 10), gravy=1.6, gravy_max=0.4)  # sev 4
    assert graded_plm_strength(hi, enabled=True, current_plm_strength=2.0,
                               soft=True, soft_zero=2.0) == pytest.approx(0.0)


def test_graded_soft_full_strength_just_at_threshold():
    """Right at the trip boundary (severity≈1) soft keeps ~full strength (no cliff)."""
    a = assess_seed(("ACDEFGHIKLMNPQRSTVWY" * 10), gravy=0.4001, gravy_max=0.4)
    out = graded_plm_strength(a, enabled=True, current_plm_strength=1.3, soft=True,
                              soft_zero=2.0)
    assert out == pytest.approx(1.3, rel=1e-2)


def test_graded_soft_zero_at_or_below_one_degrades_to_cliff():
    """Edge: soft_zero<=1 would divide by (soft_zero-1)<=0; the policy must degrade to
    the cliff (0.0) rather than crash / produce a NaN."""
    a = assess_seed(("A" * 60) + ("DEKRNQSTHG" * 14), gravy=-0.2)
    for sz in (1.0, 0.5, 0.0):
        out = graded_plm_strength(a, enabled=True, current_plm_strength=1.25,
                                  soft=True, soft_zero=sz)
        assert out == 0.0


def test_should_skip_plm_cliff_path_preserved():
    """should_skip_plm stays for the cliff path (and its tests) — graded(soft=False)
    must agree with it on the same inputs."""
    a = assess_seed(("A" * 60) + ("DEKRNQSTHG" * 14), gravy=-0.2)
    skip = should_skip_plm(a, enabled=True, current_plm_strength=1.25)
    graded = graded_plm_strength(a, enabled=True, current_plm_strength=1.25, soft=False)
    assert skip is True and graded == 0.0
