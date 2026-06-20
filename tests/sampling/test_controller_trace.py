"""Tests for the pure controller-trace formatter (sampling.controller_trace).

CF-5: opt-in verbose controller logging. ``controller_trace_rows`` turns the
per-cycle ``AxisOutcome`` objects + their declarative ``ControlAxis`` specs into a
flat list of long-format rows (one per axis per cycle) for an append-only
``controller_trace.tsv`` and a per-cycle CONTROLLER REPORT. It is PURE (no I/O) so
the chronological-validation surface is unit-testable without models or the cluster.

The formatter is observability only — it never runs unless ``--controller_verbose``
is set, so a default run is byte-identical (guarded in scripts/iterative_design.py).
"""
import math

import numpy as np
import pytest

from protein_chisel.sampling.adaptive_bias import AxisOutcome, ControlAxis
from protein_chisel.sampling.bias_scale import odds_for_nats
from protein_chisel.sampling.controller_trace import controller_trace_rows


def _axis(name, *, target, band_lo, band_hi, scope="global"):
    """Minimal ControlAxis fixture (only the trace-relevant fields matter)."""
    return ControlAxis(
        name=name,
        metric_column=f"{name}__mean",
        target=target,
        band_lo=band_lo,
        band_hi=band_hi,
        deadband=0.25,
        error_scale=1.0,
        scope=scope,
        aa_weights=np.zeros(20, dtype=np.float32),
    )


def _outcome(name, *, gate_open, reason, n, mean, signed_error, u, scope="global"):
    return AxisOutcome(
        name=name,
        gate_open=gate_open,
        reason=reason,
        n=n,
        mean=mean,
        t_stat=3.1,
        fail_fraction=0.42,
        signed_error=signed_error,
        e_norm=signed_error / 2.0,
        gain_correction=1.0,
        frozen_wrong_sign=False,
        u=u,
        scope=scope,
    )


def test_one_row_per_axis_with_expected_contents():
    T = 0.15
    charge_axis = _axis("charge", target=-10.0, band_lo=-15.0, band_hi=-5.0)
    surf_axis = _axis("surface_hydrophobicity", target=-0.2, band_lo=-0.4,
                      band_hi=0.0, scope="surface")
    outcomes = [
        _outcome("charge", gate_open=True, reason="gate_open", n=120,
                 mean=-3.0, signed_error=-7.0, u=0.31),
        _outcome("surface_hydrophobicity", gate_open=False, reason="below_t_min",
                 n=118, mean=-0.25, signed_error=0.0, u=0.0, scope="surface"),
    ]
    axes_by_name = {"charge": charge_axis,
                    "surface_hydrophobicity": surf_axis}

    rows = controller_trace_rows(outcomes, axes_by_name, cycle_idx=4, temperature=T)

    assert isinstance(rows, list) and len(rows) == 2
    r0, r1 = rows

    # one row per axis, in outcome order
    assert r0["axis"] == "charge"
    assert r1["axis"] == "surface_hydrophobicity"

    # cycle + temperature-independent fields straight from the outcome
    assert r0["cycle"] == 4
    assert r0["scope"] == "global"
    assert r0["measured_mean"] == pytest.approx(-3.0)
    assert r0["signed_error"] == pytest.approx(-7.0)
    assert r0["gate_open"] is True
    assert r0["reason"] == "gate_open"
    assert r0["drive_u"] == pytest.approx(0.31)
    assert r0["n"] == 120

    # band/target lifted from the matching ControlAxis
    assert r0["target"] == pytest.approx(-10.0)
    assert r0["band_lo"] == pytest.approx(-15.0)
    assert r0["band_hi"] == pytest.approx(-5.0)

    # effective_odds is EXACTLY odds_for_nats(u, T) — the legibility number
    assert r0["effective_odds"] == pytest.approx(odds_for_nats(0.31, T))
    assert r0["effective_odds"] == pytest.approx(math.exp(0.31 / T))

    # a closed (held) axis reports its surface scope + 1.0x odds (u=0 -> exp(0)=1)
    assert r1["scope"] == "surface"
    assert r1["gate_open"] is False
    assert r1["drive_u"] == pytest.approx(0.0)
    assert r1["effective_odds"] == pytest.approx(1.0)


def test_row_key_set_is_exactly_the_documented_schema():
    axis = _axis("charge", target=-10.0, band_lo=-15.0, band_hi=-5.0)
    outcome = _outcome("charge", gate_open=True, reason="gate_open", n=50,
                       mean=-4.0, signed_error=-6.0, u=0.2)
    rows = controller_trace_rows([outcome], {"charge": axis},
                                 cycle_idx=0, temperature=0.2)
    assert set(rows[0].keys()) == {
        "cycle", "axis", "scope", "measured_mean", "target", "band_lo",
        "band_hi", "signed_error", "gate_open", "reason", "drive_u",
        "effective_odds", "n",
    }


def test_effective_odds_tracks_temperature():
    # Same drive, lower T => a far larger effective odds multiplier (exp(u/T)).
    axis = _axis("charge", target=-10.0, band_lo=-15.0, band_hi=-5.0)
    outcome = _outcome("charge", gate_open=True, reason="gate_open", n=50,
                       mean=-4.0, signed_error=-6.0, u=0.3)
    hot = controller_trace_rows([outcome], {"charge": axis},
                                cycle_idx=1, temperature=0.30)[0]
    cold = controller_trace_rows([outcome], {"charge": axis},
                                 cycle_idx=1, temperature=0.15)[0]
    assert cold["effective_odds"] > hot["effective_odds"]
    assert hot["effective_odds"] == pytest.approx(math.exp(0.3 / 0.30))
    assert cold["effective_odds"] == pytest.approx(math.exp(0.3 / 0.15))


def test_axis_without_matching_spec_falls_back_gracefully():
    # An outcome whose axis name is missing from axes_by_name must still produce a
    # row (target/band None) rather than crash — the trace is advisory, never fatal.
    outcome = _outcome("mystery", gate_open=False, reason="no_data", n=0,
                       mean=float("nan"), signed_error=0.0, u=0.0)
    rows = controller_trace_rows([outcome], {}, cycle_idx=2, temperature=0.2)
    assert len(rows) == 1
    r = rows[0]
    assert r["axis"] == "mystery"
    assert r["target"] is None
    assert r["band_lo"] is None
    assert r["band_hi"] is None
    # scope still comes from the outcome; odds still computed from u
    assert r["effective_odds"] == pytest.approx(1.0)


def test_empty_outcomes_is_empty_rows():
    assert controller_trace_rows([], {}, cycle_idx=0, temperature=0.2) == []


def test_temperature_must_be_positive():
    axis = _axis("charge", target=-10.0, band_lo=-15.0, band_hi=-5.0)
    outcome = _outcome("charge", gate_open=True, reason="gate_open", n=50,
                       mean=-4.0, signed_error=-6.0, u=0.2)
    with pytest.raises(ValueError):
        controller_trace_rows([outcome], {"charge": axis},
                              cycle_idx=0, temperature=0.0)
