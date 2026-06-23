"""Tests for the shared hydrophobicity / SAP primitives (scoring/sap.py).

WS-B: a single source of truth for the KD scale + SASA-max + the corrected
(centered, zero-clamped) hydrophobicity weight, shared by the driver's SAP proxy
and the controller. Pure -> fully host-testable (no freesasa).
"""
from __future__ import annotations

import numpy as np
import pytest

from protein_chisel.scoring import sap


def test_kd_mean_is_about_minus_point_49():
    assert sap.KD_MEAN == pytest.approx(-0.49, abs=1e-9)


def test_kd_weight_raw_is_signed_kyte_doolittle():
    assert sap.kd_weight_raw("I") == 4.5
    assert sap.kd_weight_raw("R") == -4.5
    assert sap.kd_weight_raw("A") == 1.8
    assert sap.kd_weight_raw("Z") == 0.0  # unknown -> 0


def test_kd_weight_corrected_clamps_below_average_to_zero():
    # Polar / charged residues (below-mean KD) contribute NOTHING -> they can no
    # longer cancel hydrophobic neighbours (the legacy proxy's core defect).
    for polar in ("D", "E", "K", "R", "N", "Q", "S", "T"):
        assert sap.kd_weight_corrected(polar) == 0.0
    assert sap.kd_weight_corrected("Z") == 0.0  # unknown -> 0


def test_kd_weight_corrected_makes_alanine_count():
    # Raw Ala KD is only +1.8 (floor of the hydrophobic set); centered it is +2.29
    # and clearly positive, so an alanine surface now registers.
    assert sap.kd_weight_corrected("A") == pytest.approx(1.8 - sap.KD_MEAN)
    assert sap.kd_weight_corrected("A") > 0
    # Ordering preserved within the hydrophobic set.
    assert (sap.kd_weight_corrected("A")
            < sap.kd_weight_corrected("L")
            < sap.kd_weight_corrected("I"))


def _line(n):
    """n residues on the x-axis, all mutually within the default radius."""
    return np.array([[float(i), 0.0, 0.0] for i in range(n)])


def test_reducer_matches_explicit_reference_raw():
    # Locks byte-identity of the refactored legacy proxy: hand-compute the exact
    # neighbourhood sum the historical loop produced.
    aas = ["A", "L", "D"]
    sasa_total = [60.5, 95.5, 93.5]  # = 0.5 * SASA_MAX for A/L/D
    cas = _line(3)
    out = sap.sap_neighborhood_metrics(
        aas, sasa_total, cas, weight_fn=sap.kd_weight_raw)
    # every residue sees all 3 neighbours:
    expect = (60.5 / 121) * 1.8 + (95.5 / 191) * 3.8 + (93.5 / 187) * (-3.5)
    assert out["max"] == pytest.approx(expect)
    assert out["mean"] == pytest.approx(expect)
    assert out["p95"] == pytest.approx(expect)


def test_reducer_corrected_removes_polar_cancellation():
    # Raw: the aspartate (KD -3.5) cancels hydrophobic contributions.
    # Corrected: aspartate contributes 0, so the score is strictly higher.
    aas = ["A", "L", "D"]
    sasa_total = [60.5, 95.5, 93.5]
    cas = _line(3)
    raw = sap.sap_neighborhood_metrics(aas, sasa_total, cas, weight_fn=sap.kd_weight_raw)
    corr = sap.sap_neighborhood_metrics(aas, sasa_total, cas, weight_fn=sap.kd_weight_corrected)
    assert corr["max"] > raw["max"]
    # corrected = 0.5*2.29 + 0.5*4.29 + 0  (D drops out)
    expect = 0.5 * (1.8 - sap.KD_MEAN) + 0.5 * (3.8 - sap.KD_MEAN) + 0.0
    assert corr["max"] == pytest.approx(expect)


def test_reducer_skips_unmapped_residues():
    aas = ["A", None, "L"]
    sasa_total = [60.5, 999.0, 95.5]
    cas = _line(3)
    out = sap.sap_neighborhood_metrics(aas, sasa_total, cas, weight_fn=sap.kd_weight_raw)
    expect = (60.5 / 121) * 1.8 + (95.5 / 191) * 3.8  # None skipped
    assert out["max"] == pytest.approx(expect)


def test_reducer_radius_excludes_far_residues():
    aas = ["A", "L"]
    sasa_total = [60.5, 95.5]
    cas = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])  # 50 Å apart
    out = sap.sap_neighborhood_metrics(
        aas, sasa_total, cas, weight_fn=sap.kd_weight_raw, radius=10.0)
    # each residue sees only itself
    assert out["max"] == pytest.approx((95.5 / 191) * 3.8)   # the L self-term


def test_reducer_empty_is_nan():
    out = sap.sap_neighborhood_metrics([], [], np.empty((0, 3)), weight_fn=sap.kd_weight_raw)
    assert np.isnan(out["max"]) and np.isnan(out["mean"]) and np.isnan(out["p95"])
