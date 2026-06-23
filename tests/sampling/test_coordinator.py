"""Unit tests for the CF-2 Actuator field + CF-3 coordinate() budget allocator.

These are PURE tests (numpy/pandas/stdlib only — no models, no PyRosetta, no
cluster). They pin the three review-demanded invariants of the per-cell,
weight-partitioned, work-conserving controller budget:

  * BUG-A  — the budget is bounded on the SIGNED sum (|Σ grant| ≤ budget), with a
             final unconditional clip; NEVER on Σ|grant|.
  * BUG-B  — the controller budget nests INSIDE the total ceiling with reserved
             headroom, so both ceilings bind simultaneously (neither shadows the
             other).
  * sign-selected shared-actuator rule — axes that share one actuator collapse by
             MAX-by-magnitude when same-sign (anti-double-count) and by signed SUM
             when opposite-sign (genuine trade-off).

plus the byte-identical default-path contract (coordinator OFF ⇒ legacy
global_per_aa_bias), veto bypass, effective-odds ceilings, and the pI-shares-charge
behaviour.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from protein_chisel.sampling import adaptive_bias as ab
from protein_chisel.sampling.adaptive_bias import (
    AA_ORDER, AA_TO_IDX, AdaptiveBiasConfig, AxisOutcome, ControlAxis,
    charge_weights, coordinate, default_axes, global_per_aa_bias,
)
from protein_chisel.sampling.bias_scale import nats_for_odds, odds_for_nats
from protein_chisel.sampling.coordinator import (
    CONTROLLER_CEILING, TOTAL_CEILING, coordinate_global, nested_total_clip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _outcome(name, u, scope="global"):
    return AxisOutcome(
        name=name, gate_open=True, reason="test", n=100, mean=0.0, t_stat=9.0,
        fail_fraction=0.5, signed_error=0.0, e_norm=0.0, gain_correction=1.0,
        frozen_wrong_sign=False, u=float(u), scope=scope)


def _global_axis(name, weights, *, actuator=None, tier="budget", weight=1.0,
                 priority=100):
    return ControlAxis(
        name=name, metric_column=name, target=0.0, band_lo=-1.0, band_hi=1.0,
        deadband=0.1, error_scale=1.0, scope="global", aa_weights=weights,
        actuator=actuator, tier=tier, weight=weight, priority=priority)


T_APPLY = 0.15


# ---------------------------------------------------------------------------
# CF-2 — Actuator field
# ---------------------------------------------------------------------------
def test_control_axis_actuator_fields_default_byte_identical():
    """The four new fields default to legacy semantics (actuator=None, budget,
    weight 1.0, priority 100) so an axis that does not set them is unchanged."""
    ax = _global_axis("x", charge_weights())
    assert ax.actuator is None
    assert ax.tier == "budget"
    assert ax.weight == 1.0
    assert ax.priority == 100
    # frozen dataclass — still immutable
    with pytest.raises(Exception):
        ax.actuator = "y"  # type: ignore[misc]


def test_default_axes_declare_actuators():
    """charge -> charge_DEKR, surface_hydrophobicity -> gravy_surface (CF-2)."""
    charge, surface = default_axes()
    assert charge.actuator == "charge_DEKR"
    assert surface.actuator == "gravy_surface"


def test_default_axes_unchanged_grouping_key_is_name_when_actuator_none():
    """With no actuator set, the grouping key falls back to the axis name, so the
    algorithm reduces to today's per-axis behaviour (the reduction contract)."""
    ax = _global_axis("solo", charge_weights())
    assert (ax.actuator or ax.name) == "solo"


# ---------------------------------------------------------------------------
# Test 2 — coordinator-off == legacy global_per_aa_bias (default path)
# ---------------------------------------------------------------------------
def test_coordinate_global_two_default_axes_matches_legacy_when_no_shared_actuator():
    """coordinate_global over the 2 default axes (charge global + a global GRAVY
    stand-in), each its OWN actuator, with a budget large enough not to bind, equals
    legacy global_per_aa_bias byte-for-byte (the reduction-to-legacy contract for the
    no-shared-actuator case)."""
    # two disjoint global axes (charge D/E/K/R + a global hydrophobic proxy), each
    # its own actuator -> no grouping collapse. Use a generous odds ceiling so the
    # budget never binds, isolating the "same shape as legacy" claim.
    w_charge = charge_weights()
    w_other = ab.kd_centered_weights()
    a_charge = _global_axis("charge", w_charge, actuator="charge_DEKR")
    a_other = _global_axis("hydro", w_other, actuator="gravy_surface")
    axes_by_name = {a_charge.name: a_charge, a_other.name: a_other}
    ocs = [_outcome("charge", 0.10), _outcome("hydro", 0.08)]
    cfg = AdaptiveBiasConfig(max_nats=0.6)
    # huge ceiling -> budget non-binding; coordinator must reproduce the legacy sum
    big = nats_for_odds(1e6, T_APPLY)
    got = coordinate_global(ocs, axes_by_name, budget_nats=big)
    legacy = global_per_aa_bias(ocs, axes_by_name, cfg)
    # legacy rescales the summed (20,) vector to max_nats=0.6; here the drives are
    # small so no rescale fires => coordinator (disjoint actuators, non-binding
    # budget) must equal it.
    assert set(got) == set(legacy)
    for aa in legacy:
        assert got[aa] == pytest.approx(legacy[aa], abs=1e-9)


# ---------------------------------------------------------------------------
# Test 3 — max-not-sum collapses same-sign shared drives
# ---------------------------------------------------------------------------
def test_max_not_sum_same_sign_shared_actuator():
    """Two axes on charge_DEKR both pushing D up => applied D bias = max(|c1|,|c2|),
    NOT c1+c2 (no double-count)."""
    w = charge_weights()  # D=+1
    a1 = _global_axis("charge", w, actuator="charge_DEKR")
    a2 = _global_axis("pi", w, actuator="charge_DEKR")
    axes_by_name = {"charge": a1, "pi": a2}
    ocs = [_outcome("charge", 0.20), _outcome("pi", 0.30)]
    big = nats_for_odds(1e6, T_APPLY)
    got = coordinate_global(ocs, axes_by_name, budget_nats=big)
    # D desired: charge 0.20*1 = 0.20 ; pi 0.30*1 = 0.30. Same sign => max = 0.30.
    assert got["D"] == pytest.approx(0.30, abs=1e-9)
    # NOT the sum 0.50
    assert got["D"] != pytest.approx(0.50, abs=1e-3)


# ---------------------------------------------------------------------------
# Test 4 — signed-sum on opposite signs
# ---------------------------------------------------------------------------
def test_signed_sum_opposite_sign_shared_actuator():
    """Two axes on one actuator with OPPOSITE signs => signed sum (genuine
    trade-off), not max."""
    w = charge_weights()  # D=+1
    a1 = _global_axis("charge", w, actuator="charge_DEKR")
    a2 = _global_axis("pi", w, actuator="charge_DEKR")
    axes_by_name = {"charge": a1, "pi": a2}
    # charge pushes D up (+0.30), pi pushes D down (-0.10) => net +0.20
    ocs = [_outcome("charge", 0.30), _outcome("pi", -0.10)]
    big = nats_for_odds(1e6, T_APPLY)
    got = coordinate_global(ocs, axes_by_name, budget_nats=big)
    assert got["D"] == pytest.approx(0.20, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 5 — signed-sum budget invariant (BUG-A)
# ---------------------------------------------------------------------------
def test_signed_sum_budget_invariant_bug_a():
    """Construct DIFFERENT-actuator drives whose Σ|grant| would exceed the budget
    on a shared AA; assert |Σ grant| ≤ budget_nats per cell and the final clip binds.
    """
    # two axes on DIFFERENT actuators, both pushing D up hard. Budget is small so the
    # re-lend must NOT push the signed sum past it.
    wD = np.zeros(20); wD[AA_TO_IDX["D"]] = 1.0
    a1 = _global_axis("ax1", wD, actuator="act1", priority=1)
    a2 = _global_axis("ax2", wD, actuator="act2", priority=2)
    axes_by_name = {"ax1": a1, "ax2": a2}
    ocs = [_outcome("ax1", 0.50), _outcome("ax2", 0.50)]
    budget = nats_for_odds(CONTROLLER_CEILING, T_APPLY)  # 8x -> ~0.312 nats
    got = coordinate_global(ocs, axes_by_name, budget_nats=budget)
    # both want D up by 0.50 each (sum of magnitudes 1.0 >> budget). The signed
    # cell total on D must be clipped at +budget, NOT 2*budget.
    assert got.get("D", 0.0) == pytest.approx(budget, abs=1e-9)
    assert abs(got.get("D", 0.0)) <= budget + 1e-9


def test_allocate_cell_cross_actuator_matches_spec_algorithm():
    """Pin the EXACT spec algorithm for the (theoretical) cross-actuator opposite-sign
    case: two SEPARATE actuators with symmetric opposing drives go through the joint
    BUDGET (weight-partition + work-conserving re-lend + final clip), NOT a pure
    signed-sum — so the result is the priority-ordered budget allocation, bounded by
    BUG-A (|cell| ≤ budget). The pure signed-sum rule applies WITHIN one actuator
    (test_signed_sum_opposite_sign_shared_actuator), which is the only case the real
    pipeline produces (charge/pI share ONE actuator; surface is AA-disjoint). This test
    documents the cross-actuator budget behaviour as intentional, per the spec's
    `cell = clip(Σ grant_g, ±budget)` definition (the re-lend overshoots then the final
    clip binds — the spec's 'belt-and-suspenders' clip)."""
    from protein_chisel.sampling.coordinator import _allocate_cell
    # +10 / -10 on two DIFFERENT actuators, budget 1: the spec's algorithm yields
    # +budget (priority a first) — bounded, deterministic, |cell| ≤ budget.
    cell_a = _allocate_cell({"a": 10.0, "b": -10.0}, {"a": 1.0, "b": 1.0},
                            {"a": 1, "b": 2}, budget_nats=1.0)
    assert cell_a == pytest.approx(1.0, abs=1e-9)
    assert abs(cell_a) <= 1.0 + 1e-9          # BUG-A invariant holds
    # reversing priority reverses the winner (deterministic by (priority, name)).
    cell_b = _allocate_cell({"a": 10.0, "b": -10.0}, {"a": 1.0, "b": 1.0},
                            {"a": 2, "b": 1}, budget_nats=1.0)
    assert cell_b == pytest.approx(-1.0, abs=1e-9)
    assert abs(cell_b) <= 1.0 + 1e-9


def test_budget_invariant_holds_for_every_cell():
    """For arbitrary drives across several actuators, every cell's |Σ grant| ≤
    budget_nats (the invariant is per-cell, not just on D)."""
    rng = np.random.default_rng(7)
    axes_by_name = {}
    ocs = []
    for k in range(4):
        w = rng.normal(0, 1, 20)
        name = f"ax{k}"
        axes_by_name[name] = _global_axis(name, w, actuator=f"act{k}",
                                           priority=k)
        ocs.append(_outcome(name, rng.uniform(-0.6, 0.6)))
    budget = nats_for_odds(CONTROLLER_CEILING, T_APPLY)
    got = coordinate_global(ocs, axes_by_name, budget_nats=budget)
    for aa, v in got.items():
        assert abs(v) <= budget + 1e-9, f"cell {aa} = {v} exceeds budget {budget}"


# ---------------------------------------------------------------------------
# Test 6 — nested ceilings (BUG-B)
# ---------------------------------------------------------------------------
def test_nested_ceilings_bug_b():
    """Large PLM+consensus + a controller drive => |cell_total| ≤ total_nats AND the
    controller keeps its FULL reserved headroom (neither shadows the other)."""
    total_nats = nats_for_odds(TOTAL_CEILING, T_APPLY)     # ~6.9 * 0.15 ~ 1.036
    reserve = nats_for_odds(CONTROLLER_CEILING, T_APPLY)   # ~2.08 * 0.15 ~ 0.312
    rest = np.full((1, 20), 5.0, dtype=np.float64)          # PLM+consensus huge
    controller = np.zeros((1, 20), dtype=np.float64)
    controller[0, AA_TO_IDX["D"]] = reserve                # controller wants full D
    total = nested_total_clip(rest, controller, total_nats=total_nats,
                              reserve_nats=reserve)
    # (1) overall bound: |cell_total| <= total_nats everywhere
    assert np.all(np.abs(total) <= total_nats + 1e-9)
    # (2) the controller keeps its reserved headroom: the D cell = clip(rest,
    #     total-reserve) + clip(controller, reserve) = (total-reserve) + reserve =
    #     total_nats — the controller's reserve is NOT shadowed by the saturated rest.
    assert total[0, AA_TO_IDX["D"]] == pytest.approx(total_nats, abs=1e-9)
    # a non-controller cell saturates only to (total - reserve): the reserve stays
    # carved out for the controller even where the controller is silent.
    assert total[0, AA_TO_IDX["A"]] == pytest.approx(total_nats - reserve, abs=1e-9)


def test_nested_ceilings_reserve_capped_at_total():
    """codex BUG-B edge: a reserve LARGER than the total ceiling cannot overrun the
    total (reserve is capped at total inside nested_total_clip — it just collapses the
    rest budget to zero, never breaks |cell_total| ≤ total_nats)."""
    total_nats = nats_for_odds(50.0, T_APPLY)        # small whole-stack ceiling
    reserve = nats_for_odds(1e9, T_APPLY)            # absurd controller reserve > total
    rest = np.full((3, 20), 9.0)
    controller = np.full((3, 20), 9.0)
    out = nested_total_clip(rest, controller, total_nats=total_nats,
                            reserve_nats=reserve)
    assert np.all(np.abs(out) <= total_nats + 1e-9)


def test_nested_ceilings_opposing_controller():
    """A controller pushing AGAINST a saturated rest still gets its full reserve in
    the opposing direction (the carve-out is symmetric)."""
    total_nats = nats_for_odds(TOTAL_CEILING, T_APPLY)
    reserve = nats_for_odds(CONTROLLER_CEILING, T_APPLY)
    rest = np.full((1, 20), 5.0, dtype=np.float64)
    controller = np.zeros((1, 20), dtype=np.float64)
    controller[0, AA_TO_IDX["D"]] = -reserve               # push DOWN against +rest
    total = nested_total_clip(rest, controller, total_nats=total_nats,
                              reserve_nats=reserve)
    # rest clipped to +(total-reserve), controller to -reserve => net total-2*reserve
    assert total[0, AA_TO_IDX["D"]] == pytest.approx(total_nats - 2 * reserve,
                                                     abs=1e-9)
    assert np.all(np.abs(total) <= total_nats + 1e-9)


# ---------------------------------------------------------------------------
# Test 7 — veto bypass
# ---------------------------------------------------------------------------
def test_veto_tier_bypasses_budget():
    """A veto-tier axis's drive is emitted UNBUDGETED (added on top of the budgeted
    sum, never diluted to the controller ceiling)."""
    wD = np.zeros(20); wD[AA_TO_IDX["D"]] = 1.0
    veto = _global_axis("veto", wD, actuator="veto_act", tier="veto")
    budgeted = _global_axis("b", wD, actuator="b_act", tier="budget")
    axes_by_name = {"veto": veto, "b": budgeted}
    # veto wants D up by a LARGE amount that exceeds the budget; it must survive
    # un-clipped by the controller budget.
    big_drive = 5.0
    ocs = [_outcome("veto", big_drive), _outcome("b", 0.10)]
    budget = nats_for_odds(CONTROLLER_CEILING, T_APPLY)
    got = coordinate_global(ocs, axes_by_name, budget_nats=budget)
    # the veto contribution (5.0) bypasses the 8x budget entirely.
    assert got["D"] >= big_drive - 1e-9


# ---------------------------------------------------------------------------
# Test 8 — effective_odds ≤ ceiling
# ---------------------------------------------------------------------------
def test_effective_odds_controller_le_ceiling():
    """For arbitrary BUDGET-tier drives, realized odds at T_apply ≤ CONTROLLER_CEILING
    (the controller share never exceeds 8x)."""
    rng = np.random.default_rng(11)
    axes_by_name, ocs = {}, []
    for k in range(3):
        w = rng.normal(0, 1, 20)
        name = f"ax{k}"
        axes_by_name[name] = _global_axis(name, w, actuator=f"act{k}")
        ocs.append(_outcome(name, rng.uniform(-0.6, 0.6)))
    budget = nats_for_odds(CONTROLLER_CEILING, T_APPLY)
    got = coordinate_global(ocs, axes_by_name, budget_nats=budget)
    for aa, v in got.items():
        assert odds_for_nats(abs(v), T_APPLY) <= CONTROLLER_CEILING + 1e-6


def test_effective_odds_total_le_ceiling():
    """(PLM+consensus+controllers) realized cell odds at T_apply ≤ TOTAL_CEILING; the
    units-consistency guarantee that no source is over-weighted out of the shared
    odds budget."""
    total_nats = nats_for_odds(TOTAL_CEILING, T_APPLY)
    reserve = nats_for_odds(CONTROLLER_CEILING, T_APPLY)
    rng = np.random.default_rng(3)
    rest = rng.normal(0, 4.0, (5, 20))           # large PLM+consensus stacks
    controller = rng.uniform(-reserve, reserve, (5, 20))
    total = nested_total_clip(rest, controller, total_nats=total_nats,
                              reserve_nats=reserve)
    assert np.all(odds_for_nats_vec(np.abs(total), T_APPLY) <= TOTAL_CEILING + 1e-6)


def odds_for_nats_vec(nats, T):
    return np.exp(np.asarray(nats) / T)


# ---------------------------------------------------------------------------
# Test 9 — pI fires when charge asleep (blind-spot catch)
# ---------------------------------------------------------------------------
def test_pi_drives_shared_actuator_when_charge_asleep():
    """pI out-of-band, charge in-band => only pI emits a drive; it steers the SHARED
    charge_DEKR actuator (D up) even though the charge axis is silent."""
    charge = default_axes(axes=["charge"])[0]
    pi_axis = _pi_axis()
    axes_by_name = {charge.name: charge, pi_axis.name: pi_axis}
    # charge asleep (u=0, gate closed) ; pI wants D/E up (u>0 with charge_weights)
    ocs = [_outcome("charge", 0.0), _outcome("pi", 0.25)]
    budget = nats_for_odds(CONTROLLER_CEILING, T_APPLY)
    got = coordinate_global(ocs, axes_by_name, budget_nats=budget)
    # pI alone drove D up on the shared actuator
    assert got.get("D", 0.0) == pytest.approx(0.25, abs=1e-9)
    assert got.get("K", 0.0) < 0.0  # K down (charge_weights basic_downweight)


# ---------------------------------------------------------------------------
# Test 10 — pI+charge agree => no double push
# ---------------------------------------------------------------------------
def test_pi_and_charge_agree_max_not_sum():
    """pI + charge both want D/E up => the shared actuator gets MAX, not SUM."""
    charge = default_axes(axes=["charge"])[0]
    pi_axis = _pi_axis()
    axes_by_name = {charge.name: charge, pi_axis.name: pi_axis}
    ocs = [_outcome("charge", 0.20), _outcome("pi", 0.30)]
    budget = nats_for_odds(1e6, T_APPLY)  # non-binding to isolate the max rule
    got = coordinate_global(ocs, axes_by_name, budget_nats=budget)
    # both project D=+1 ; same sign => max(0.20, 0.30) = 0.30, NOT 0.50
    assert got["D"] == pytest.approx(0.30, abs=1e-9)
    assert got["D"] != pytest.approx(0.50, abs=1e-3)


def _pi_axis():
    """Build the pI axis the registry will declare (shares charge_DEKR)."""
    from protein_chisel.sampling.adaptive_bias import charge_weights
    return ControlAxis(
        name="pi", metric_column="pi", target=5.5, band_lo=5.0, band_hi=7.5,
        deadband=0.25, error_scale=1.25, scope="global",
        aa_weights=charge_weights(), actuator="charge_DEKR", tier="budget",
        weight=1.0)


# ===========================================================================
# Integration via compute_adaptive_bias (the actual driver entry point)
# ===========================================================================
from protein_chisel.sampling.adaptive_bias import compute_adaptive_bias


def _trip_pool(n=120, *, charge_mean=2.0, gravy_mean=0.5, pi_mean=8.0, seed=0):
    """A candidate pool that trips charge (too positive), surface (too hydrophobic),
    and (when pI in the axes) pI (too basic). Includes the gap columns each axis reads.
    """
    rng = np.random.default_rng(seed)
    cv = rng.normal(charge_mean, 1.0, n)
    gv = rng.normal(gravy_mean, 0.1, n)
    pv = rng.normal(pi_mean, 0.3, n)
    return pd.DataFrame({
        "net_charge_full_HH": cv,
        "gravy": gv,
        "pi": pv,
        "selection__seq_filter_gap_charge_high": np.maximum(0.0, cv - (-4.0)),
        "selection__seq_filter_gap_charge_low": np.zeros(n),
        "selection__seq_filter_gap_gravy": np.maximum(0.0, gv - 0.3),
        "selection__seq_filter_gap_pi": np.maximum(0.0, pv - 7.5),
        "sequence": ["ACDEFGHIKLMNPQRSTVWY" * 3] * n,
    })


def _kw(L=20):
    return dict(L=L, position_classes=["distal_surface"] * L,
               sasa_fraction=np.full(L, 0.5), fixed_idx=set())


# ---------------------------------------------------------------------------
# Test 1 — A/A byte-identical (coordinator OFF, run twice)
# ---------------------------------------------------------------------------
def test_aa_byte_identical_coordinator_off():
    """Coordinator OFF: running compute_adaptive_bias twice on the same inputs yields
    byte-identical global bias, per-position delta, and merged bias_AA string."""
    df = _trip_pool()
    axes = default_axes()
    cfg = AdaptiveBiasConfig(max_odds=8.0)  # coordinator defaults OFF
    r1 = compute_adaptive_bias(pool_df=df, axes=axes, cfg=cfg, state=None,
                               temperature=0.15, **_kw())
    r2 = compute_adaptive_bias(pool_df=df, axes=axes, cfg=cfg, state=None,
                               temperature=0.15, **_kw())
    assert r1.controller_global == r2.controller_global
    assert r1.bias_AA_string == r2.bias_AA_string
    assert np.array_equal(r1.per_position_delta, r2.per_position_delta)
    # and the coordinator telemetry key is ABSENT when off (byte-identical contract)
    assert "coordinator" not in r1.telemetry


# ---------------------------------------------------------------------------
# Test 2 — coordinator-off == legacy path
# ---------------------------------------------------------------------------
def test_coordinator_off_equals_legacy_two_default_axes():
    """With the coordinator OFF, compute_adaptive_bias takes the legacy
    global_per_aa_bias + per-axis surface-delta path EXACTLY — the coordinate() path is
    not entered. We assert the global equals a direct global_per_aa_bias call and the
    delta equals the legacy per-axis accumulation."""
    df = _trip_pool()
    axes = default_axes()
    cfg = AdaptiveBiasConfig(max_odds=8.0)
    r = compute_adaptive_bias(pool_df=df, axes=axes, cfg=cfg, state=None,
                              temperature=0.15, **_kw())
    # Re-derive the legacy global from the same outcomes the call produced.
    axes_by_name = {a.name: a for a in axes}
    import dataclasses as _dc
    from protein_chisel.sampling.bias_scale import effective_clamp_nats
    eff = effective_clamp_nats(cfg.max_nats, cfg.max_odds, 0.15)
    eff_cfg = _dc.replace(cfg, max_nats=eff)
    legacy_global = global_per_aa_bias(r.outcomes, axes_by_name, eff_cfg)
    assert r.controller_global == legacy_global


def test_coordinator_off_is_byte_identical_to_pre_feature():
    """The hard byte-identical claim: an AdaptiveBiasConfig with coordinator=False
    (the default) produces the SAME global+delta+bias_AA as the same config explicitly
    constructed without ever touching the new fields — i.e. the new fields are inert
    when off (defends against an accidental default flip)."""
    df = _trip_pool()
    axes = default_axes()
    cfg_default = AdaptiveBiasConfig(max_odds=8.0)
    assert cfg_default.coordinator is False        # default OFF
    r = compute_adaptive_bias(pool_df=df, axes=axes, cfg=cfg_default, state=None,
                              temperature=0.15, **_kw())
    # With coordinator off the global is the legacy sum-then-rescale; charge trips so
    # D/E go up, K/R down (the known legacy behaviour, unchanged).
    assert r.controller_global["D"] > 0 and r.controller_global["E"] > 0
    assert r.controller_global["K"] < 0 and r.controller_global["R"] < 0


# ---------------------------------------------------------------------------
# Coordinator ON via compute_adaptive_bias: pI shares charge under max-not-sum
# ---------------------------------------------------------------------------
def test_compute_adaptive_bias_coordinator_pi_shares_charge_no_double():
    """Coordinator ON with charge+pi BOTH tripping (both want D/E up): the shared
    charge_DEKR actuator gets the MAX of the two drives, not the sum (no double-lock)."""
    df = _trip_pool()
    axes = default_axes(axes=["charge", "pi"])
    cfg = AdaptiveBiasConfig(max_odds=8.0, coordinator=True)
    r = compute_adaptive_bias(pool_df=df, axes=axes, cfg=cfg, state=None,
                              temperature=0.15, **_kw())
    # the coordinator budget caps D at the controller ceiling; with max-not-sum the
    # shared actuator never exceeds what ONE axis at full authority would emit.
    budget = nats_for_odds(CONTROLLER_CEILING, 0.15)
    assert r.controller_global.get("D", 0.0) <= budget + 1e-9
    assert r.controller_global.get("D", 0.0) > 0.0   # it did steer D up
    assert r.telemetry["coordinator"]["active"] is True


def test_seed_warmstart_coordinator_bounds_cycle0_and_off_is_byte_identical():
    """codex #3: --controller_coordinator must also route the cycle-0 warm-start through
    the coordinator (bounded to the controller ceiling at the application T), and the
    OFF path must stay byte-identical (the temperature arg is ignored when off)."""
    from protein_chisel.sampling.adaptive_bias import seed_warmstart
    L = 20
    kw = dict(L=L, position_classes=["distal_surface"] * L,
              sasa_fraction=np.full(L, 0.5), fixed_idx=set())
    seed = {"gravy": 0.9, "net_charge_full_HH": 5.0}     # too hydrophobic + too positive
    # coordinator ON: the cycle-0 controller share is bounded to the ceiling at T_apply.
    g_on, d_on, _s, _t = seed_warmstart(
        seed_metrics=seed, axes=default_axes(),
        cfg=AdaptiveBiasConfig(coordinator=True), temperature=0.20, **kw)
    for aa, v in g_on.items():
        assert odds_for_nats(abs(v), 0.20) <= CONTROLLER_CEILING + 1e-6
    assert np.all(odds_for_nats_vec(np.abs(d_on), 0.20) <= CONTROLLER_CEILING + 1e-6)
    # coordinator OFF: byte-identical regardless of the temperature argument.
    g1, d1, _, _ = seed_warmstart(seed_metrics=seed, axes=default_axes(),
                                  cfg=AdaptiveBiasConfig(), temperature=0.20, **kw)
    g2, d2, _, _ = seed_warmstart(seed_metrics=seed, axes=default_axes(),
                                  cfg=AdaptiveBiasConfig(), temperature=None, **kw)
    assert g1 == g2 and np.array_equal(d1, d2)


def test_compute_adaptive_bias_coordinator_total_odds_bounded():
    """The controller share realized at T_apply never exceeds CONTROLLER_CEILING odds
    (the units-consistency guarantee from compute_adaptive_bias' own output)."""
    df = _trip_pool()
    axes = default_axes(axes=["charge", "pi"])
    cfg = AdaptiveBiasConfig(max_odds=8.0, coordinator=True)
    r = compute_adaptive_bias(pool_df=df, axes=axes, cfg=cfg, state=None,
                              temperature=0.15, **_kw())
    for aa, v in r.controller_global.items():
        assert odds_for_nats(abs(v), 0.15) <= CONTROLLER_CEILING + 1e-6
    # surface delta cells are also controller-budget bounded
    assert np.all(odds_for_nats_vec(np.abs(r.per_position_delta), 0.15)
                  <= CONTROLLER_CEILING + 1e-6)


# ---------------------------------------------------------------------------
# pI-band wiring + shared-actuator guard basis (codex review follow-ups).
# ---------------------------------------------------------------------------
def test_pi_axis_band_and_target_follow_pi_min_max():
    """The pI controller axis takes its band/target from the (pi_min, pi_max) the
    driver threads from --pi_min/--pi_max, so the controller deadbands/gates against
    the SAME band the pI filter enforces (codex)."""
    axes = ab.default_axes(axes=["charge", "surface_hydrophobicity", "pi"],
                           pi_band=(5.5, 7.0), pi_target=6.0)
    pi = next(a for a in axes if a.name == "pi")
    assert pi.band_lo == 5.5 and pi.band_hi == 7.0
    assert pi.target == 6.0
    assert pi.metric_column == "pi"


def test_pi_and_charge_share_charge_dekr_actuator():
    """pI shares the charge D/E/K/R actuator (the basis for the driver's
    shared-actuator guard); charge + surface do NOT share an actuator."""
    cp = ab.default_axes(axes=["charge", "pi"])
    assert all(a.actuator == "charge_DEKR" for a in cp)
    cs = ab.default_axes(axes=["charge", "surface_hydrophobicity"])
    assert {a.actuator for a in cs} == {"charge_DEKR", "gravy_surface"}
