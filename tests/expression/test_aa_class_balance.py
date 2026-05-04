"""Tests for class-aware compensatory bias_AA construction.

Covers the central swap behavior the user asked for: when one member of
an AA class is over-represented (e.g. E z=+5) and another is under-rep
(e.g. D z=-2), bias_AA should down-weight the over-rep AA AND up-weight
the under-rep AA — not just suppress one side, since that would only
reduce the property the class encodes (here: total negative charge).
"""

from __future__ import annotations

import pytest

from protein_chisel.expression.aa_class_balance import (
    AA_CLASSES,
    AaBalanceTelemetry,
    compute_class_balanced_bias_AA,
)
from protein_chisel.expression.aa_composition import aa_z_scores


def _build_seq(counts: dict[str, int], total: int = 200) -> str:
    """Build a length-`total` sequence with the given AA counts; pad with A."""
    parts = []
    used = 0
    for aa, n in counts.items():
        parts.append(aa * n)
        used += n
    if used < total:
        parts.append("A" * (total - used))
    return "".join(parts)[:total]


def test_e_rich_d_poor_triggers_swap():
    """E over-rep + D under-rep should yield bias_AA with E:-x and D:+y."""
    seq = _build_seq({"E": 30, "L": 40, "G": 30, "V": 20})  # ~ no D, ~15% E
    z = aa_z_scores(seq, reference="swissprot_ec3_hydrolases_2026_01")
    assert z["E"] > 2.0, f"E z={z['E']} not high enough for test setup"
    assert z["D"] < -2.0, f"D z={z['D']} not low enough for test setup"

    t = compute_class_balanced_bias_AA(seq, exclude_aas="C")

    # Both E and D appear in the bias dict, with the right signs.
    assert "E" in t.per_aa_bias and t.per_aa_bias["E"] < 0
    assert "D" in t.per_aa_bias and t.per_aa_bias["D"] > 0

    # The string is parseable AA:val,AA:val with both swap members present.
    parts = dict(p.split(":") for p in t.bias_AA_string.split(","))
    assert "E" in parts and float(parts["E"]) < 0
    assert "D" in parts and float(parts["D"]) > 0

    # The "negatively_charged" class is recorded in the swap audit log.
    classes_swapped = [s["class"] for s in t.swaps]
    assert "negatively_charged" in classes_swapped


def test_balanced_class_no_swap():
    """If a class has no over+under combo, no swap from that class."""
    # Use a sequence whose negatively-charged content matches reference: roughly
    # 6% E + 6% D = 12% → typical hydrolase composition. No imbalance triggers.
    seq = _build_seq({"E": 12, "D": 12, "L": 30, "A": 30, "G": 30, "S": 30,
                      "T": 20, "K": 12, "R": 12, "I": 12})
    t = compute_class_balanced_bias_AA(seq, exclude_aas="C")
    classes_swapped = [s["class"] for s in t.swaps]
    assert "negatively_charged" not in classes_swapped


def test_max_bias_clamp():
    """Per-AA bias must not exceed +/- max_bias_nats after summing classes."""
    # Push A extremely high (multiple class memberships → huge cumulative bias).
    seq = "A" * 200
    t = compute_class_balanced_bias_AA(seq, exclude_aas="C", max_bias_nats=1.5)
    for aa, val in t.per_aa_bias.items():
        assert -1.5 - 1e-9 <= val <= 1.5 + 1e-9, f"{aa} bias {val} out of clamp"


def test_excluded_aa_not_biased():
    """exclude_aas members are skipped — never appear in the bias output."""
    seq = _build_seq({"C": 30, "L": 50, "A": 60, "G": 30, "V": 30})
    t = compute_class_balanced_bias_AA(seq, exclude_aas="C")
    assert "C" not in t.per_aa_bias
    assert ":C:" not in (":" + t.bias_AA_string + ":")


def test_singleton_class_only_suppresses_extreme():
    """P / G singletons have no swap partner; only suppress when z>over_z."""
    # G very over-rep (10% G is z=~5), should be suppressed.
    seq = _build_seq({"G": 30, "L": 40, "A": 30, "V": 20, "S": 20, "E": 12,
                      "D": 12, "K": 8, "R": 8, "I": 12, "T": 8})
    t = compute_class_balanced_bias_AA(seq, exclude_aas="C", over_z_threshold=3.0)
    if "G" in t.per_aa_bias:
        assert t.per_aa_bias["G"] < 0  # G never up-weighted (singleton class)


def test_telemetry_to_dict_round_trip():
    seq = _build_seq({"E": 25, "L": 40, "G": 30, "V": 20, "S": 20})
    t = compute_class_balanced_bias_AA(seq, exclude_aas="C")
    d = t.to_dict()
    assert "bias_AA_string" in d
    assert "per_aa_bias" in d
    assert "swaps" in d
    assert "z_scores" in d
    assert d["reference"] == "swissprot_ec3_hydrolases_2026_01"


def test_class_definitions_consistent():
    """Sanity: AA_CLASSES values are upper-case 1-letter codes."""
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    for cls_name, members in AA_CLASSES.items():
        for m in members:
            assert m in valid, f"{cls_name} has bad member {m!r}"
        # No duplicates within a class.
        assert len(members) == len(set(members)), f"{cls_name} has duplicates"


def test_unknown_reference_raises():
    with pytest.raises(ValueError):
        compute_class_balanced_bias_AA("AAAAAAAAAA", reference="not_a_reference")
