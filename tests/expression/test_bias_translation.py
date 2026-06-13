"""Tests for translating expression-engine soft-bias hits into an MPNN bias.

``soft_bias_to_bias_array`` is the pure kernel behind the opt-in
``--composition_soft_bias`` tier: it turns the (previously dead)
``EngineResult.soft_bias_per_residue()`` map of {body_position -> AAs to
downweight} into an additive (L, 20) per-residue bias the LigandMPNN sampler
consumes, in the canonical PLM/MPNN amino-acid column order.
"""

from __future__ import annotations

import numpy as np

from protein_chisel.expression.aa_composition import AA_ORDER_REF
from protein_chisel.expression.engine import (
    EngineResult, aggregate_pool_soft_bias, soft_bias_to_bias_array,
)
from protein_chisel.expression.severity import RuleHit, Severity


def _soft_hit(start, end, aas, name="r"):
    return RuleHit(
        rule_name=name, severity=Severity.SOFT_BIAS, start=start, end=end,
        matched="", reason="", suggested_omit_AAs=aas,
    )


def test_downweights_named_position_aa_cells():
    L = 5
    sb = {0: "L", 2: "FW"}
    arr = soft_bias_to_bias_array(sb, L, magnitude=1.5, aa_order=AA_ORDER_REF)
    assert arr.shape == (L, 20)
    assert arr[0, AA_ORDER_REF.index("L")] == -1.5
    assert arr[2, AA_ORDER_REF.index("F")] == -1.5
    assert arr[2, AA_ORDER_REF.index("W")] == -1.5
    # Only the three named cells are touched; everything else is zero.
    assert int((arr != 0).sum()) == 3


def test_is_a_pure_downweight_never_positive():
    arr = soft_bias_to_bias_array({0: "AVL", 1: "K"}, 3, magnitude=2.0)
    assert arr.max() <= 0.0
    # magnitude is taken by absolute value (downweight regardless of sign).
    arr_neg = soft_bias_to_bias_array({0: "AVL"}, 3, magnitude=-2.0)
    assert np.allclose(arr_neg[0, AA_ORDER_REF.index("A")], -2.0)


def test_repeated_aa_letter_at_one_position_applied_once():
    """A repeated AA letter within one position's string must NOT stack (a
    source-agnostic map could carry 'LL'); the cell gets -magnitude once."""
    arr = soft_bias_to_bias_array({0: "LL", 1: "KKK"}, 3, magnitude=2.0)
    assert arr[0, AA_ORDER_REF.index("L")] == -2.0      # once, not -4.0
    assert arr[1, AA_ORDER_REF.index("K")] == -2.0      # once, not -6.0
    assert int((arr != 0).sum()) == 2


def test_skips_out_of_range_positions_and_unknown_aas():
    arr = soft_bias_to_bias_array({99: "L", 0: "XZ-"}, 3, magnitude=1.0)
    # pos 99 is out of range; X/Z/- are not canonical AAs -> nothing applied.
    assert np.array_equal(arr, np.zeros((3, 20)))


def test_empty_map_is_all_zero():
    arr = soft_bias_to_bias_array({}, 4, magnitude=2.0)
    assert arr.shape == (4, 20)
    assert not arr.any()


# ---------------------------------------------------------------------------
# WS-C hybrid soft-bias source: span filter + per-cycle pool aggregation
# ---------------------------------------------------------------------------


def test_soft_bias_per_residue_span_filter_excludes_whole_protein_hits():
    """max_span_frac drops whole-protein composition hits (handled by the
    suppress-all / fraction-cap levers) while keeping LOCAL structural ones."""
    L = 20
    er = EngineResult(
        sequence="A" * L, profile_name="t",
        hits=[
            _soft_hit(5, 8, "P", "polyproline_stall"),          # local span 3
            _soft_hit(0, L, "A", "aa_composition_out_of_distribution"),  # whole
        ],
    )
    full = er.soft_bias_per_residue()                # no filter: both included
    assert full[0] == "A" and full[19] == "A" and "P" in full[5]
    filt = er.soft_bias_per_residue(max_span_frac=0.5)
    assert set(filt.keys()) == {5, 6, 7}             # only the local P hit
    assert all(v == "P" for v in filt.values())


def test_soft_bias_per_residue_default_unchanged_byte_identical():
    """max_span_frac defaults None → identical to the legacy (all-hits) map."""
    er = EngineResult(
        sequence="A" * 12, profile_name="t",
        hits=[_soft_hit(2, 5, "LF"), _soft_hit(0, 12, "A")],
    )
    assert er.soft_bias_per_residue() == er.soft_bias_per_residue(max_span_frac=None)


def test_soft_bias_per_residue_excludes_aggregate_hits_under_filter():
    """An aggregate SOFT_BIAS hit (e.g. dibasic-count: too many KR motifs overall)
    spans first-to-last motif — a region, not a per-position liability. Under the
    local-only filter it is EXCLUDED even when its span passes the span gate, so
    intervening non-motif positions are not biased."""
    L = 30
    local = _soft_hit(5, 8, "P", "polyproline_stall")
    agg = RuleHit(
        rule_name="dibasic_motif_count_cap", severity=Severity.SOFT_BIAS,
        start=2, end=12, matched="6", reason="", suggested_omit_AAs="KR",
        metadata={"aggregate": True},                  # span 10 <= 0.5*30=15 (passes span gate)
    )
    er = EngineResult(sequence="A" * L, profile_name="t", hits=[local, agg])
    # Legacy default (None): aggregate INCLUDED (byte-identical to before).
    assert "K" in er.soft_bias_per_residue().get(2, "")
    # Local-only filter: aggregate EXCLUDED; only the local polyproline hit remains.
    filt = er.soft_bias_per_residue(max_span_frac=0.5)
    assert set(filt.keys()) == {5, 6, 7}
    assert all("K" not in v and "R" not in v for v in filt.values())


def test_dibasic_count_rule_marks_its_soft_bias_hit_aggregate():
    """The global dibasic-count rule tags its hit aggregate so the per-residue
    soft-bias tier (local-only) drops it."""
    from protein_chisel.expression.builtin_rules import DibasicMotifCountCapRule
    from protein_chisel.expression.profiles import ExpressionProfile
    from protein_chisel.expression.rules import StructureContext
    rule = DibasicMotifCountCapRule()
    ctx = StructureContext(sequence="KR" * 6 + "A" * 60)   # 6 KR motifs > cap 5
    hits = rule.evaluate(ctx, ExpressionProfile.bl21_cytosolic_streptag())
    assert hits, "rule should fire on 6 dibasic motifs"
    assert hits[0].metadata.get("aggregate") is True


def test_aggregate_pool_soft_bias_support_gate():
    """A liability flagged in >= min_support of survivors is kept; a one-off
    (below support) is dropped. Whole-protein hits are filtered by span."""
    class _StubEngine:
        """Returns a polyproline-style local hit at a per-sequence position
        encoded by the sequence: 'P' triples drive a local hit at their index."""
        def evaluate(self, seq, **_):
            hits = []
            i = seq.find("PPP")
            if i >= 0:
                hits.append(_soft_hit(i, i + 3, "P", "polyproline_stall"))
            return EngineResult(sequence=seq, profile_name="t", hits=hits)

    eng = _StubEngine()
    # 3 of 4 survivors share a PPP at position 5; one has it at position 0.
    survivors = [
        "AAAAA" + "PPP" + "A" * 12,   # PPP at 5
        "AAAAA" + "PPP" + "A" * 12,   # PPP at 5
        "AAAAA" + "PPP" + "A" * 12,   # PPP at 5
        "PPP" + "A" * 17,             # PPP at 0 (one-off)
    ]
    out = aggregate_pool_soft_bias(
        eng, survivors, ss_reduced=None, sasa=None, position_class=None,
        catalytic_resnos=(), fixed_resnos=(), protein_resnos=None,
        min_support=0.5, max_span_frac=0.5,
    )
    # position 5 (3/4 = 0.75 >= 0.5) kept; position 0 (1/4 = 0.25) dropped.
    assert set(out.keys()) == {5, 6, 7}
    assert all(v == "P" for v in out.values())


def test_soft_bias_to_bias_array_rejects_nonfinite_magnitude():
    """A non-finite magnitude would cast to -inf in the (float32) sampler bias and
    poison the softmax — reject it at the kernel."""
    import pytest
    for bad in (float("inf"), -float("inf"), float("nan")):
        with pytest.raises(ValueError):
            soft_bias_to_bias_array({0: "L"}, 3, magnitude=bad)


def test_aggregate_pool_soft_bias_skips_failing_survivor():
    """A survivor whose engine.evaluate RAISES must be skipped (not abort the
    whole aggregation); support is taken over the SUCCESSFUL survivors."""
    class _Flaky:
        def evaluate(self, seq, **_):
            if seq == "BOOM":
                raise RuntimeError("a rule blew up on this sequence")
            hits = [_soft_hit(2, 5, "P", "polyproline_stall")] if "PPP" in seq else []
            return EngineResult(sequence=seq, profile_name="t", hits=hits)

    good = "xxPPPxx" + "A" * 10
    out = aggregate_pool_soft_bias(
        _Flaky(), [good, "BOOM", good],
        min_support=0.75, max_span_frac=0.9,
    )
    # 1 of 3 raised → denominator 2 successful; both flag pos 2-4 → 2/2 >= 0.75.
    assert set(out.keys()) == {2, 3, 4}


def test_aggregate_pool_soft_bias_all_failing_is_empty():
    class _AllBoom:
        def evaluate(self, seq, **_):
            raise RuntimeError("boom")
    assert aggregate_pool_soft_bias(_AllBoom(), ["a", "b"]) == {}


def test_aggregate_pool_soft_bias_empty_pool_is_empty():
    class _Eng:
        def evaluate(self, seq, **_):
            return EngineResult(sequence=seq, profile_name="t", hits=[])
    assert aggregate_pool_soft_bias(
        _Eng(), [], ss_reduced=None, sasa=None, position_class=None,
        catalytic_resnos=(), fixed_resnos=(), protein_resnos=None,
    ) == {}


def test_default_aa_order_locks_to_plm_mpnn_column_order():
    """The helper's default AA column order MUST equal the sampler's bias
    column order, or the soft bias would land on the wrong amino acids."""
    from protein_chisel.sampling.plm_fusion import AA_ORDER
    assert AA_ORDER_REF == AA_ORDER
    # And the helper defaults to that order when none is passed.
    arr = soft_bias_to_bias_array({0: "C"}, 1, magnitude=1.0)
    assert arr[0, AA_ORDER.index("C")] == -1.0
