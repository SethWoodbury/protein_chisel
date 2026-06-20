"""Unit tests for the closed-loop adaptive-bias controller (pure, no models).

Covers the four hard requirements:
  (a) bidirectional anti-overshoot (reverse when the pool over-corrects),
  (b) fires only when statistically in trouble (gate silence otherwise),
  (c) negligible/clean control logic (sign/clamp/compose correctness),
  (d) byte-identical when off (no-op when there's nothing to do),
plus a closed-loop simulation proving convergence + hold + anti-overshoot on a
(near-)static plant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from protein_chisel.sampling import adaptive_bias as ab
from protein_chisel.sampling.adaptive_bias import (
    AA_TO_IDX, AdaptiveBiasConfig, AxisState, ControlAxis,
    charge_weights, compute_adaptive_bias, default_axes, kd_centered_weights,
    merge_bias_AA_strings, step_axis, signed_deadband_error, wald_lower_bound,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_pool(axis: ControlAxis, *, n: int, mean: float, std: float, rng) -> pd.DataFrame:
    """Synthesize a candidate pool with the metric + gap columns for one axis."""
    vals = rng.normal(mean, std, n)
    df = pd.DataFrame({axis.metric_column: vals})
    scale = max(1e-9, abs(axis.error_scale))
    if axis.fail_high_column and axis.fail_low_column:
        # two separate one-sided gap columns (charge: high + low)
        df[axis.fail_high_column] = np.maximum(0.0, vals - axis.band_hi) / scale
        df[axis.fail_low_column] = np.maximum(0.0, axis.band_lo - vals) / scale
    elif axis.fail_high_column:
        # single two-sided interval gap column (gravy): >0 outside [lo, hi]
        df[axis.fail_high_column] = (np.maximum(0.0, vals - axis.band_hi)
                                     + np.maximum(0.0, axis.band_lo - vals)) / scale
    df["passed_seq_filter"] = (vals >= axis.band_lo) & (vals <= axis.band_hi)
    return df


CHARGE_AXIS = default_axes()[0]
GRAVY_AXIS = default_axes()[1]


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------
def test_kd_dict_is_single_source_from_scoring_sap():
    """WS-D dedup: the controller's KD scale is the SAME object as scoring.sap's
    (single source of truth), and the centered projection is byte-identical to a
    hand-computation off that shared dict (no silent value drift)."""
    from protein_chisel.scoring import sap
    assert ab.KD_HYDROPHOBICITY is sap.KD_HYDROPHOBICITY
    kd = np.array([sap.KD_HYDROPHOBICITY[a] for a in ab.AA_ORDER], dtype=float)
    centered = kd - kd.mean()
    assert np.array_equal(kd_centered_weights(), centered / np.abs(centered).max())


def test_kd_projection_signs_and_norm():
    w = kd_centered_weights()
    assert abs(np.abs(w).max() - 1.0) < 1e-9          # normalized to max 1
    for a in "IVLFMA":                                 # hydrophobic -> positive
        assert w[AA_TO_IDX[a]] > 0
    for a in "DEKRNQ":                                 # hydrophilic -> negative
        assert w[AA_TO_IDX[a]] < 0


def test_default_axes_unchanged_by_default():
    """No args => today's registry: [charge, surface_hydrophobicity], target -10,
    band (-18,-4), gap columns present (byte-identical default)."""
    axes = default_axes()
    assert [a.name for a in axes] == ["charge", "surface_hydrophobicity"]
    ch = axes[0]
    assert ch.target == -10.0 and (ch.band_lo, ch.band_hi) == (-18.0, -4.0)
    assert ch.fail_high_column == "selection__seq_filter_gap_charge_high"
    assert ch.fail_low_column == "selection__seq_filter_gap_charge_low"


def test_default_axes_charge_band_override_sets_band_target_and_nulls_gaps():
    """--adaptive_charge_band -15,-5: charge axis band=(-15,-5), target=midpoint
    -10, and the (cycle-band) gap columns are NULLED so the fail-fraction is
    evaluated on raw values vs the new band (codex fix)."""
    ch = default_axes(charge_band=(-15.0, -5.0))[0]
    assert (ch.band_lo, ch.band_hi) == (-15.0, -5.0)
    assert ch.target == pytest.approx(-10.0)
    assert ch.fail_high_column is None and ch.fail_low_column is None


def test_charge_band_override_fires_via_raw_value_eval_without_gap_columns():
    """With the override (gap columns nulled), an out-of-band charge pool that
    carries NO gap columns still fires through raw-value band evaluation."""
    ch = default_axes(charge_band=(-15.0, -5.0))[0]
    rng = np.random.default_rng(43)
    pool = make_pool(ch, n=200, mean=2.0, std=1.0, rng=rng)   # +2, well above -5
    assert "selection__seq_filter_gap_charge_high" not in pool.columns
    oc, _ = step_axis(pool, ch, AxisState(), AdaptiveBiasConfig())
    assert oc.gate_open and oc.fail_fraction > 0.9


def test_default_axes_charge_band_validates_lo_lt_hi():
    with pytest.raises(ValueError):
        default_axes(charge_band=(-5.0, -15.0))      # lo > hi
    with pytest.raises(ValueError):
        default_axes(charge_band=(float("nan"), -5.0))


def test_default_axes_selector_subset_and_unknown_raises():
    assert [a.name for a in default_axes(axes=["charge"])] == ["charge"]
    assert [a.name for a in default_axes(axes=["surface_hydrophobicity"])] == \
        ["surface_hydrophobicity"]
    with pytest.raises(ValueError):
        default_axes(axes=["charge", "bogus_axis"])


def test_default_axes_selector_rejects_empty_and_duplicates():
    """codex: an empty selection silently disables the controller, and a duplicate
    (e.g. the surface axis twice) would stack the actuator past its clamp — both are
    config errors, not silent behaviors."""
    with pytest.raises(ValueError):
        default_axes(axes=[])
    with pytest.raises(ValueError):
        default_axes(axes=["surface_hydrophobicity", "surface_hydrophobicity"])


def test_charge_projection():
    w = charge_weights()
    assert w[AA_TO_IDX["D"]] == 1.0 and w[AA_TO_IDX["E"]] == 1.0
    assert w[AA_TO_IDX["K"]] == pytest.approx(-0.35)
    assert w[AA_TO_IDX["R"]] == pytest.approx(-0.35)
    for a in "AVLIMFWCGHNQPSTY":
        assert w[AA_TO_IDX[a]] == 0.0


def test_wald_lower_bound():
    assert wald_lower_bound(0.0, 100) == 0.0
    assert wald_lower_bound(0.5, 100) > 0.3
    assert wald_lower_bound(0.05, 10) < 0.05            # small n, small f -> CI dips low


# ---------------------------------------------------------------------------
# (b) Gate silence
# ---------------------------------------------------------------------------
def test_silent_when_in_target():
    rng = np.random.default_rng(1)
    pool = make_pool(GRAVY_AXIS, n=200, mean=GRAVY_AXIS.target, std=0.1, rng=rng)
    oc, _ = step_axis(pool, GRAVY_AXIS, AxisState(), AdaptiveBiasConfig())
    assert not oc.gate_open and oc.u == 0.0


def test_silent_low_n():
    rng = np.random.default_rng(2)
    pool = make_pool(GRAVY_AXIS, n=10, mean=1.0, std=0.1, rng=rng)   # very out, but N<30
    oc, _ = step_axis(pool, GRAVY_AXIS, AxisState(), AdaptiveBiasConfig())
    assert not oc.gate_open and "N=" in oc.reason and oc.u == 0.0


def test_silent_high_variance_mean_in_band():
    # mean inside band, large spread -> |t| small -> gate closed even if a few fail
    rng = np.random.default_rng(3)
    pool = make_pool(GRAVY_AXIS, n=200, mean=0.0, std=2.0, rng=rng)
    oc, _ = step_axis(pool, GRAVY_AXIS, AxisState(), AdaptiveBiasConfig(t_min=2.5))
    assert not oc.gate_open


def test_silent_low_fail_fraction():
    # mean just past target but almost everything still in-band -> low fail-fraction
    rng = np.random.default_rng(4)
    pool = make_pool(GRAVY_AXIS, n=400, mean=0.28, std=0.01, rng=rng)  # band_hi 0.3
    oc, _ = step_axis(pool, GRAVY_AXIS, AxisState(), AdaptiveBiasConfig())
    assert oc.fail_fraction < 0.15
    assert not oc.gate_open


# ---------------------------------------------------------------------------
# (a) Reversal / bidirectional + drive direction
# ---------------------------------------------------------------------------
def test_too_hydrophobic_pushes_hydrophilic_at_surface():
    rng = np.random.default_rng(5)
    pool = make_pool(GRAVY_AXIS, n=200, mean=0.8, std=0.1, rng=rng)
    L = 10
    res = compute_adaptive_bias(
        pool_df=pool, axes=[GRAVY_AXIS], cfg=AdaptiveBiasConfig(), state=None,
        L=L, position_classes=["distal_surface"] * L, sasa_fraction=np.ones(L),
        fixed_idx=set())
    d = res.per_position_delta
    assert d[0, AA_TO_IDX["L"]] < 0          # surface Leu down-weighted
    assert d[0, AA_TO_IDX["I"]] < d[0, AA_TO_IDX["A"]]   # Ile down more than Ala (KD)


def test_compute_adaptive_bias_applies_odds_space_clamp():
    # CF-1: cfg.max_odds + temperature -> the |u| clamp becomes nats_for_odds(max_odds, T)
    # (temperature-invariant authority); no max_odds -> raw max_nats (byte-identical).
    from protein_chisel.sampling.bias_scale import nats_for_odds, ODDS_NUDGE
    rng = np.random.default_rng(7)
    pool = make_pool(GRAVY_AXIS, n=200, mean=0.8, std=0.1, rng=rng)
    L = 5
    common = dict(pool_df=pool, axes=[GRAVY_AXIS], state=None, L=L,
                  position_classes=["distal_surface"] * L,
                  sasa_fraction=np.ones(L), fixed_idx=set())
    res0 = compute_adaptive_bias(cfg=AdaptiveBiasConfig(), **common)
    assert res0.telemetry["config"]["max_nats"] == 0.6          # legacy path
    assert res0.telemetry["odds_clamp"]["max_odds"] is None
    T = 0.15
    res1 = compute_adaptive_bias(cfg=AdaptiveBiasConfig(max_odds=ODDS_NUDGE),
                                 temperature=T, **common)
    exp = nats_for_odds(ODDS_NUDGE, T)
    assert abs(res1.telemetry["config"]["max_nats"] - exp) < 1e-9
    assert abs(res1.telemetry["odds_clamp"]["eff_max_nats"] - round(exp, 4)) < 1e-9
    # a tighter clamp (NUDGE=2x=0.10 nats < 0.6) can only reduce the applied drive
    assert abs(res1.telemetry["axes"][0]["u"]) <= abs(res0.telemetry["axes"][0]["u"]) + 1e-9


def test_overshoot_hydrophilic_reverses():
    # An over-corrected pool (too hydrophilic, GRAVY well below target) must REVERSE:
    # surface term flips to UP-weight hydrophobic.
    rng = np.random.default_rng(6)
    pool = make_pool(GRAVY_AXIS, n=200, mean=-1.2, std=0.1, rng=rng)  # below band_lo -0.8
    L = 5
    res = compute_adaptive_bias(
        pool_df=pool, axes=[GRAVY_AXIS], cfg=AdaptiveBiasConfig(), state=None,
        L=L, position_classes=["distal_surface"] * L, sasa_fraction=np.ones(L),
        fixed_idx=set())
    assert res.outcomes[0].u < 0
    assert res.per_position_delta[0, AA_TO_IDX["L"]] > 0   # now UP-weight surface Leu


def test_charge_too_positive_raises_DE():
    rng = np.random.default_rng(7)
    pool = make_pool(CHARGE_AXIS, n=200, mean=2.0, std=2.0, rng=rng)  # too positive
    res = compute_adaptive_bias(
        pool_df=pool, axes=[CHARGE_AXIS], cfg=AdaptiveBiasConfig(), state=None,
        L=5, position_classes=["distal_surface"] * 5, sasa_fraction=np.ones(5),
        fixed_idx=set())
    d = ab.parse_bias_AA(res.bias_AA_string)
    assert d.get("D", 0) > 0 and d.get("E", 0) > 0
    assert d.get("K", 0) < 0 and d.get("R", 0) < 0


def test_charge_overshoot_too_negative_reverses():
    rng = np.random.default_rng(8)
    pool = make_pool(CHARGE_AXIS, n=200, mean=-22.0, std=2.0, rng=rng)  # below band_lo -18
    res = compute_adaptive_bias(
        pool_df=pool, axes=[CHARGE_AXIS], cfg=AdaptiveBiasConfig(), state=None,
        L=5, position_classes=["distal_surface"] * 5, sasa_fraction=np.ones(5),
        fixed_idx=set())
    d = ab.parse_bias_AA(res.bias_AA_string)
    assert d.get("D", 0) < 0 and d.get("E", 0) < 0     # down-weight acidic
    assert d.get("K", 0) > 0 and d.get("R", 0) > 0     # up-weight basic


def test_signed_deadband_error_zero_in_band():
    # within deadband -> 0; beyond -> signed, deadband-subtracted
    assert signed_deadband_error(GRAVY_AXIS.target, GRAVY_AXIS) == 0.0
    e = signed_deadband_error(GRAVY_AXIS.target + GRAVY_AXIS.deadband + 0.2, GRAVY_AXIS)
    assert e == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# (c) Clamp / budget / protect / merge
# ---------------------------------------------------------------------------
def test_clamp_respected():
    rng = np.random.default_rng(9)
    pool = make_pool(CHARGE_AXIS, n=300, mean=50.0, std=3.0, rng=rng)  # absurdly off
    cfg = AdaptiveBiasConfig(max_nats=0.6)
    res = compute_adaptive_bias(
        pool_df=pool, axes=[CHARGE_AXIS], cfg=cfg, state=None, L=5,
        position_classes=["distal_surface"] * 5, sasa_fraction=np.ones(5), fixed_idx=set())
    for v in ab.parse_bias_AA(res.bias_AA_string).values():
        assert abs(v) <= 0.6 + 1e-9
    assert abs(res.outcomes[0].u) <= 0.6 + 1e-9


def test_merge_skips_same_direction_class_balance_wins():
    # class-balance already moved E down; controller wants E up -> conflict, CB wins.
    s, conflicts = merge_bias_AA_strings("E:-1.40,D:1.45", {"E": 0.5, "D": 0.5, "N": 0.3})
    d = ab.parse_bias_AA(s)
    assert d["E"] == pytest.approx(-1.40)     # class-balance retained
    assert d["D"] == pytest.approx(1.45)      # same-direction -> CB retained (skip stack)
    assert d["N"] == pytest.approx(0.3)       # untouched by CB -> controller applies
    kinds = {c["aa"]: c["resolution"] for c in conflicts}
    assert kinds["E"] == "class_balance_wins" and kinds["D"] == "skip_same_dir"


def test_merge_clamps_to_class_balance_cap():
    s, _ = merge_bias_AA_strings("", {"D": 5.0})
    assert ab.parse_bias_AA(s)["D"] == pytest.approx(ab.CLASS_BALANCE_MAX_NATS)


def test_global_budget_no_double_push():
    # Two synthetic global axes both up-weighting D should not exceed max_nats.
    ax2 = ControlAxis(name="charge2", metric_column="net_charge_full_HH",
                      target=-10.0, band_lo=-18.0, band_hi=-4.0, deadband=3.0,
                      error_scale=6.0, scope="global", aa_weights=charge_weights(),
                      fail_high_column="selection__seq_filter_gap_charge_high",
                      fail_low_column="selection__seq_filter_gap_charge_low")
    rng = np.random.default_rng(10)
    pool = make_pool(CHARGE_AXIS, n=300, mean=5.0, std=2.0, rng=rng)
    cfg = AdaptiveBiasConfig(max_nats=0.6)
    res = compute_adaptive_bias(
        pool_df=pool, axes=[CHARGE_AXIS, ax2], cfg=cfg, state=None, L=5,
        position_classes=["distal_surface"] * 5, sasa_fraction=np.ones(5), fixed_idx=set())
    assert abs(ab.parse_bias_AA(res.bias_AA_string).get("D", 0)) <= 0.6 + 1e-9


# ---------------------------------------------------------------------------
# Surface delta: surface-only, fixed-skip
# ---------------------------------------------------------------------------
def test_surface_delta_only_at_surface_and_not_fixed():
    rng = np.random.default_rng(11)
    pool = make_pool(GRAVY_AXIS, n=200, mean=0.8, std=0.1, rng=rng)
    L = 4
    classes = ["distal_surface", "distal_buried", "primary_sphere", "distal_surface"]
    res = compute_adaptive_bias(
        pool_df=pool, axes=[GRAVY_AXIS], cfg=AdaptiveBiasConfig(), state=None, L=L,
        position_classes=classes, sasa_fraction=np.ones(L), fixed_idx={3})
    d = res.per_position_delta
    assert d[0, AA_TO_IDX["L"]] < 0          # surface, not fixed -> active
    assert np.all(d[1] == 0)                  # buried -> untouched
    assert np.all(d[2] == 0)                  # active-site -> untouched
    assert np.all(d[3] == 0)                  # surface BUT fixed -> untouched


# ---------------------------------------------------------------------------
# WS-D: non_tunnel_surface scope
# ---------------------------------------------------------------------------
# L=6 scaffold exercising every branch:
#   i0 distal_surface, exposed, BUT tunnel-lining   i3 distal_buried (buried)
#   i1 nearby_surface, exposed (binding region)     i4 distal_surface, low SASA
#   i2 primary_sphere (active site)                 i5 nearby_surface, but FIXED
_SCOPE_CLASSES = ["distal_surface", "nearby_surface", "primary_sphere",
                  "distal_buried", "distal_surface", "nearby_surface"]
_SCOPE_SASA = np.array([0.5, 0.5, 0.5, 0.05, 0.1, 0.5])
_SCOPE_FIXED = {5}
_SCOPE_TUNNEL = frozenset({0})


def test_surface_scope_legacy_matches_distal_surface_only():
    """sasa_gate=None reproduces today's distal_surface + sasa>0 + not-fixed gate
    EXACTLY (tunnel/throat sets ignored in legacy mode) — byte-identical."""
    mask = ab.surface_scope(
        L=6, position_classes=_SCOPE_CLASSES, sasa_fraction=_SCOPE_SASA,
        fixed_idx=_SCOPE_FIXED, sasa_gate=None, tunnel_lining_idx=_SCOPE_TUNNEL)
    # distal_surface with sasa>0 and not fixed: i0 (tunnel ignored in legacy), i4.
    assert list(mask) == [True, False, False, False, True, False]


def test_surface_scope_non_tunnel_adds_exposed_nearby_drops_tunnel_and_lowsasa():
    """sasa_gate set => the non_tunnel_surface set: exposed non-active-site, not
    tunnel/throat/fixed. Adds back exposed nearby_surface (i1); drops the
    tunnel-lining distal_surface (i0) and the low-SASA distal_surface (i4)."""
    mask = ab.surface_scope(
        L=6, position_classes=_SCOPE_CLASSES, sasa_fraction=_SCOPE_SASA,
        fixed_idx=_SCOPE_FIXED, sasa_gate=0.2, tunnel_lining_idx=_SCOPE_TUNNEL)
    assert list(mask) == [False, True, False, False, False, False]


def test_surface_scope_excludes_unvetted_classes_positively_gated():
    """SEV-1 (review): the non_tunnel branch POSITIVELY gates on the steerable
    exposed classes ({distal_surface, nearby_surface}) rather than only excluding
    the active site — so a legacy/unknown CORE class (e.g. 'buried', 'first_shell')
    with high total-SASA is NOT swept into the surface scope."""
    classes = ["distal_surface", "nearby_surface", "buried", "first_shell", "weird"]
    sasa = np.array([0.5, 0.5, 0.5, 0.5, 0.5])     # all exposed by the (total-SASA) gate
    mask = ab.surface_scope(
        L=5, position_classes=classes, sasa_fraction=sasa, fixed_idx=set(),
        sasa_gate=0.2)
    # Only the two vetted exposed classes; the core/unknown classes are excluded
    # even though their SASA clears the gate.
    assert list(mask) == [True, True, False, False, False]


def test_surface_scope_steerable_classes_configurable():
    """A caller can restrict the scope (e.g. distal_surface only) for a cautious run."""
    classes = ["distal_surface", "nearby_surface"]
    sasa = np.array([0.5, 0.5])
    mask = ab.surface_scope(
        L=2, position_classes=classes, sasa_fraction=sasa, fixed_idx=set(),
        sasa_gate=0.2, steerable_classes=frozenset({"distal_surface"}))
    assert list(mask) == [True, False]          # nearby_surface now excluded


def test_surface_scope_throat_band_excluded():
    mask = ab.surface_scope(
        L=6, position_classes=_SCOPE_CLASSES, sasa_fraction=_SCOPE_SASA,
        fixed_idx=set(), sasa_gate=0.2, throat_band_idx=frozenset({1}))
    assert mask[1] == False          # i1 is in the throat band -> excluded


def test_build_surface_delta_honors_provided_mask():
    """A provided surface_mask overrides the legacy class gate: a nearby_surface
    position masked-in gets the down-weight; a distal_surface position masked-out
    does not."""
    oc = ab.AxisOutcome(name="x", gate_open=True, reason="", n=200, mean=0.8,
                        t_stat=9.0, fail_fraction=0.9, signed_error=0.5,
                        e_norm=0.5, gain_correction=1.0, frozen_wrong_sign=False,
                        u=0.4, scope="surface")
    classes = ["distal_surface", "nearby_surface"]
    mask = np.array([False, True])   # invert the legacy expectation
    d = ab.build_surface_delta(
        oc, L=2, position_classes=classes, sasa_fraction=np.ones(2),
        fixed_idx=set(), cfg=AdaptiveBiasConfig(), surface_mask=mask)
    assert np.all(d[0] == 0)                      # distal_surface masked OUT
    assert d[1, AA_TO_IDX["L"]] < 0               # nearby_surface masked IN


def test_build_surface_delta_mask_none_is_legacy_byte_identical():
    """surface_mask=None preserves the exact legacy behavior."""
    oc = ab.AxisOutcome(name="x", gate_open=True, reason="", n=200, mean=0.8,
                        t_stat=9.0, fail_fraction=0.9, signed_error=0.5,
                        e_norm=0.5, gain_correction=1.0, frozen_wrong_sign=False,
                        u=0.4, scope="surface")
    classes = ["distal_surface", "nearby_surface", "primary_sphere"]
    kw = dict(L=3, position_classes=classes, sasa_fraction=np.ones(3),
              fixed_idx=set(), cfg=AdaptiveBiasConfig())
    legacy = ab.build_surface_delta(oc, **kw)                 # no surface_mask
    explicit = ab.build_surface_delta(oc, surface_mask=None, **kw)
    assert np.array_equal(legacy, explicit)
    assert d_nonzero_rows(legacy) == [0]          # only distal_surface (legacy)


def d_nonzero_rows(d):
    return [i for i in range(d.shape[0]) if np.any(d[i])]


def test_compute_adaptive_bias_threads_surface_mask():
    rng = np.random.default_rng(31)
    pool = make_pool(GRAVY_AXIS, n=200, mean=0.8, std=0.1, rng=rng)
    classes = ["distal_surface", "nearby_surface"]
    mask = np.array([False, True])
    res = compute_adaptive_bias(
        pool_df=pool, axes=[GRAVY_AXIS], cfg=AdaptiveBiasConfig(), state=None,
        L=2, position_classes=classes, sasa_fraction=np.ones(2), fixed_idx=set(),
        surface_mask=mask)
    d = res.per_position_delta
    assert np.all(d[0] == 0) and d[1, AA_TO_IDX["L"]] < 0


# ---------------------------------------------------------------------------
# WS-D: multi-pool plumbing (pool_key + pools) — enables future struct-stage axes
# ---------------------------------------------------------------------------
def test_control_axis_pool_key_defaults_to_seq():
    assert CHARGE_AXIS.pool_key == "seq" and GRAVY_AXIS.pool_key == "seq"


def test_pools_routes_each_axis_to_its_pool():
    """An axis with pool_key='struct' MEASURES the struct pool, not the seq pool:
    a healthy seq pool + a too-hydrophobic struct pool must fire the struct axis."""
    import dataclasses
    struct_axis = dataclasses.replace(GRAVY_AXIS, name="gravy_struct", pool_key="struct")
    rng = np.random.default_rng(40)
    seq_healthy = make_pool(GRAVY_AXIS, n=200, mean=GRAVY_AXIS.target, std=0.1, rng=rng)
    struct_bad = make_pool(struct_axis, n=200, mean=0.8, std=0.1, rng=rng)
    res = compute_adaptive_bias(
        pools={"seq": seq_healthy, "struct": struct_bad}, axes=[struct_axis],
        cfg=AdaptiveBiasConfig(), state=None, L=3,
        position_classes=["distal_surface"] * 3, sasa_fraction=np.ones(3),
        fixed_idx=set())
    assert res.outcomes[0].gate_open                       # fired off the STRUCT pool
    assert res.per_position_delta[0, AA_TO_IDX["L"]] < 0
    assert res.outcomes[0].n == 200


def test_pools_back_compat_single_pool_df():
    """Passing pool_df= (no pools=) is identical to pools={'seq': pool_df}."""
    rng = np.random.default_rng(41)
    pool = make_pool(GRAVY_AXIS, n=200, mean=0.8, std=0.1, rng=rng)
    kw = dict(axes=[GRAVY_AXIS], cfg=AdaptiveBiasConfig(), state=None, L=3,
              position_classes=["distal_surface"] * 3, sasa_fraction=np.ones(3),
              fixed_idx=set())
    a = compute_adaptive_bias(pool_df=pool, **kw)
    b = compute_adaptive_bias(pools={"seq": pool}, **kw)
    assert np.array_equal(a.per_position_delta, b.per_position_delta)
    assert a.outcomes[0].u == b.outcomes[0].u


def test_pools_missing_key_holds_not_crashes():
    """An axis whose pool_key isn't in pools holds (no measurement) — no crash."""
    import dataclasses
    struct_axis = dataclasses.replace(GRAVY_AXIS, name="g_struct", pool_key="struct")
    rng = np.random.default_rng(42)
    res = compute_adaptive_bias(
        pools={"seq": make_pool(GRAVY_AXIS, n=200, mean=0.8, std=0.1, rng=rng)},
        axes=[struct_axis], cfg=AdaptiveBiasConfig(), state=None, L=3,
        position_classes=["distal_surface"] * 3, sasa_fraction=np.ones(3),
        fixed_idx=set())
    assert not res.outcomes[0].gate_open               # struct pool absent -> hold
    assert "no pool" in res.outcomes[0].reason


# ---------------------------------------------------------------------------
# (d) No-op when there's nothing to do
# ---------------------------------------------------------------------------
def test_empty_pool_is_noop():
    res = compute_adaptive_bias(
        pool_df=None, axes=default_axes(), cfg=AdaptiveBiasConfig(), state=None,
        L=8, position_classes=["distal_surface"] * 8, sasa_fraction=np.ones(8),
        fixed_idx=set(), class_balance_bias_AA="E:-1.0")
    assert res.bias_AA_string == "E:-1.00"   # class-balance unchanged
    assert np.count_nonzero(res.per_position_delta) == 0


def test_healthy_pool_is_noop():
    rng = np.random.default_rng(12)
    pool = pd.concat([
        make_pool(GRAVY_AXIS, n=200, mean=GRAVY_AXIS.target, std=0.1, rng=rng),
        make_pool(CHARGE_AXIS, n=200, mean=CHARGE_AXIS.target, std=2.0, rng=rng),
    ], axis=1)
    res = compute_adaptive_bias(
        pool_df=pool, axes=default_axes(), cfg=AdaptiveBiasConfig(), state=None,
        L=8, position_classes=["distal_surface"] * 8, sasa_fraction=np.ones(8),
        fixed_idx=set())
    assert res.bias_AA_string == ""
    assert np.count_nonzero(res.per_position_delta) == 0


# ---------------------------------------------------------------------------
# Wrong-sign freeze
# ---------------------------------------------------------------------------
def test_wrong_sign_freezes_axis():
    # Build a state whose history says: we pushed u=+0.3 and the mean went UP
    # (wrong direction for our projection) -> next step must freeze (u=0).
    rng = np.random.default_rng(13)
    pool = make_pool(GRAVY_AXIS, n=200, mean=0.9, std=0.1, rng=rng)
    # last cycle we applied u=0.3; step_axis will append (0.3, mean_now~0.9). Paired
    # with the prior (0.0, 0.5) that gives K=(0.9-0.5)/0.3>0 -> wrong sign -> freeze.
    st = AxisState(last_u=0.3, history=[(0.0, 0.5)])
    oc, _ = step_axis(pool, GRAVY_AXIS, st, AdaptiveBiasConfig())
    assert oc.frozen_wrong_sign and oc.u == 0.0


# ---------------------------------------------------------------------------
# Closed-loop convergence + hold + anti-overshoot (the headline)
# ---------------------------------------------------------------------------
def _simulate(axis, cfg, *, m0, K, n_cycles, std, noise=0.0, seed=0):
    """Static-plant sim: pool mean responds to the bias applied THIS cycle.

    m_k = m0 + K * u_k (+ noise). The controller observes the pool produced by the
    previous u and computes the next u. Returns (means, us).
    """
    rng = np.random.default_rng(seed)
    state = AxisState()
    m = m0
    means, us = [m], [0.0]
    for _ in range(n_cycles):
        pool = make_pool(axis, n=150, mean=m, std=std, rng=rng)
        oc, state = step_axis(pool, axis, state, cfg)
        u = oc.u
        m = m0 + K * u + (rng.normal(0.0, noise) if noise > 0 else 0.0)
        means.append(m)
        us.append(u)
    return np.array(means), np.array(us)


def test_closed_loop_converges_and_holds():
    # Too hydrophobic (m0=+0.5); target -0.2. K<0: a hydrophilic push lowers GRAVY.
    cfg = AdaptiveBiasConfig()
    means, us = _simulate(GRAVY_AXIS, cfg, m0=0.5, K=-1.2, n_cycles=5, std=0.12, seed=0)
    # lands in the passing band and near target
    assert GRAVY_AXIS.band_lo <= means[-1] <= GRAVY_AXIS.band_hi
    assert abs(means[-1] - GRAVY_AXIS.target) <= GRAVY_AXIS.deadband + 0.12
    # HOLDS: the bias does not collapse back to 0 once healthy
    assert abs(us[-1]) > 0.1
    # no large overshoot past target
    assert means.min() >= GRAVY_AXIS.target - GRAVY_AXIS.deadband - 0.1
    # clamp respected throughout
    assert np.all(np.abs(us) <= cfg.max_nats + 1e-9)


def test_closed_loop_overshoot_then_recover():
    # Aggressive gain + strong plant so cycle 1 overshoots PAST the lower band edge
    # (too hydrophilic); the controller must detect it (two-sided gap) and REVERSE
    # the bias to bring the pool back into band — bidirectional anti-overshoot.
    cfg = AdaptiveBiasConfig(gain=1.6)
    means, us = _simulate(GRAVY_AXIS, cfg, m0=0.6, K=-2.6, n_cycles=8, std=0.12, seed=1)
    assert means.min() < GRAVY_AXIS.band_lo          # it did overshoot out of band (low)
    assert GRAVY_AXIS.band_lo <= means[-1] <= GRAVY_AXIS.band_hi   # recovered into band
    # the reversal actually reduced the bias from its peak (did not just keep pushing)
    assert abs(us[-1]) < np.abs(us).max()
    assert means.min() > GRAVY_AXIS.target - 1.0      # bounded, did not diverge


def test_closed_loop_strong_plant_does_not_diverge():
    # Plant gain 4x what the controller initially assumes. After an unavoidable
    # first-step overshoot (gain unknown until estimated), the online estimate must
    # tame it: settle bounded and IN BAND within a few cycles rather than diverging.
    cfg = AdaptiveBiasConfig()
    means, us = _simulate(GRAVY_AXIS, cfg, m0=0.5, K=-4.0, n_cycles=8, std=0.1, seed=2)
    assert np.all(np.isfinite(means))
    assert np.abs(means).max() < 5.0                  # bounded (did not diverge)
    assert GRAVY_AXIS.band_lo <= means[-1] <= GRAVY_AXIS.band_hi   # settled in band
    # and the late trajectory is stable (held, not oscillating)
    assert np.std(means[-3:]) < 0.05


def test_seed_warmstart_from_input():
    # A hydrophobic, slightly-positive seed should warm-start cycle 0: surface
    # hydrophobic down-weight + global D/E up-weight, with state seeding last_u.
    axes = default_axes()
    g, delta, state, tele = ab.seed_warmstart(
        seed_metrics={"gravy": 0.9, "net_charge_full_HH": 2.0}, axes=axes,
        cfg=AdaptiveBiasConfig(), L=6, position_classes=["distal_surface"] * 6,
        sasa_fraction=np.ones(6), fixed_idx=set())
    assert g.get("D", 0) > 0 and g.get("E", 0) > 0          # acidify
    assert delta[0, AA_TO_IDX["L"]] < 0                      # surface Leu down
    assert state["surface_hydrophobicity"]["last_u"] > 0     # seeds last_u for gain est
    assert state["charge"]["last_u"] > 0


def test_seed_warmstart_in_band_is_noop():
    axes = default_axes()
    g, delta, state, tele = ab.seed_warmstart(
        seed_metrics={"gravy": -0.2, "net_charge_full_HH": -10.0}, axes=axes,
        cfg=AdaptiveBiasConfig(), L=6, position_classes=["distal_surface"] * 6,
        sasa_fraction=np.ones(6), fixed_idx=set())
    assert g == {} and np.count_nonzero(delta) == 0


def _emitted_surface_u(res, surf_idx=0):
    """Recover the applied surface drive u from the EMITTED (L,20) delta."""
    wL = ab.hydrophobic_surface_weights()[AA_TO_IDX["L"]]
    return -res.per_position_delta[surf_idx, AA_TO_IDX["L"]] / wL


def test_hold_survives_actuator_when_gate_closes():
    # Regression for the "HOLD is inert" bug: an axis that was biased last cycle and
    # whose pool is now healthy (gate closed) must STILL emit the held bias, not 0.
    rng = np.random.default_rng(20)
    healthy = make_pool(GRAVY_AXIS, n=200, mean=GRAVY_AXIS.target, std=0.1, rng=rng)
    state = {"surface_hydrophobicity": AxisState(last_u=0.5, history=[(0.5, -0.2)]).to_dict()}
    res = compute_adaptive_bias(
        pool_df=healthy, axes=[GRAVY_AXIS], cfg=AdaptiveBiasConfig(), state=state,
        L=4, position_classes=["distal_surface"] * 4, sasa_fraction=np.ones(4),
        fixed_idx=set())
    assert not res.outcomes[0].gate_open               # pool is healthy -> gate closed
    assert np.any(res.per_position_delta)              # ...but the held bias is EMITTED
    assert _emitted_surface_u(res) == pytest.approx(0.5, abs=0.05)   # held (pure)


def _simulate_emission(cfg, *, m0, K, alpha, n_cycles, std, seed):
    """Closed-loop sim through the EMISSION path (compute_adaptive_bias) on a LEAKY
    plant m_k = alpha*m_{k-1} + (1-alpha)*(m0 + K*u_k) — partial reversion, so the
    controller MUST hold the bias to keep the pool on target."""
    rng = np.random.default_rng(seed)
    state = None
    m = m0
    means = [m]
    for _ in range(n_cycles):
        pool = make_pool(GRAVY_AXIS, n=150, mean=m, std=std, rng=rng)
        res = compute_adaptive_bias(
            pool_df=pool, axes=[GRAVY_AXIS], cfg=cfg, state=state, L=4,
            position_classes=["distal_surface"] * 4, sasa_fraction=np.ones(4),
            fixed_idx=set())
        state = res.new_state
        u = _emitted_surface_u(res)
        m = alpha * m + (1.0 - alpha) * (m0 + K * u)
        means.append(m)
    return np.array(means)


def test_emission_path_converges_and_holds_on_leaky_plant():
    means = _simulate_emission(AdaptiveBiasConfig(), m0=0.5, K=-1.2, alpha=0.3,
                               n_cycles=7, std=0.12, seed=0)
    assert GRAVY_AXIS.band_lo <= means[-1] <= GRAVY_AXIS.band_hi
    assert abs(means[-1] - GRAVY_AXIS.target) <= GRAVY_AXIS.deadband + 0.05
    assert np.std(means[-3:]) < 0.03                   # held steady, not oscillating


def test_zero_variance_out_of_band_pool_fires():
    # A collapsed pool (all identical, se=0) that is out of band must NOT fool the
    # gate into closing (regression: t=0 when se=0).
    rng = np.random.default_rng(21)
    pool = make_pool(GRAVY_AXIS, n=150, mean=1.0, std=0.0, rng=rng)
    res = compute_adaptive_bias(
        pool_df=pool, axes=[GRAVY_AXIS], cfg=AdaptiveBiasConfig(), state=None, L=4,
        position_classes=["distal_surface"] * 4, sasa_fraction=np.ones(4), fixed_idx=set())
    assert res.outcomes[0].gate_open and np.any(res.per_position_delta)


def test_over_rep_mask_focuses_surface_downweight():
    mask = ab.hydrophobic_over_rep_mask(["L" * 60 + "AVGS" * 10])   # Leu-overloaded
    assert mask is not None and mask[AA_TO_IDX["L"]] == 1.0
    # hydrophilic-dominated seq: no HYDROPHOBIC AA is over-represented -> None
    assert ab.hydrophobic_over_rep_mask(["DEKRSTNQHG" * 12]) is None
    # with a mask, only the over-represented AA is down-weighted at the surface
    rng = np.random.default_rng(22)
    pool = make_pool(GRAVY_AXIS, n=200, mean=0.8, std=0.1, rng=rng)
    res = compute_adaptive_bias(
        pool_df=pool, axes=[GRAVY_AXIS], cfg=AdaptiveBiasConfig(), state=None, L=3,
        position_classes=["distal_surface"] * 3, sasa_fraction=np.ones(3),
        fixed_idx=set(), over_rep_mask=mask)
    d = res.per_position_delta
    assert d[0, AA_TO_IDX["L"]] < 0 and d[0, AA_TO_IDX["I"]] == 0.0   # only L


def test_over_rep_mask_ignored_on_reversal():
    # Pool overshot too hydrophilic (u<0 -> up-weight hydrophobics). Even with a mask
    # focusing on L, the up-weight must spread across ALL hydrophobic AAs, not re-add
    # the already-over-represented one preferentially.
    mask = np.zeros(20)
    mask[AA_TO_IDX["L"]] = 1.0
    rng = np.random.default_rng(24)
    pool = make_pool(GRAVY_AXIS, n=200, mean=-1.2, std=0.1, rng=rng)   # below band_lo
    res = compute_adaptive_bias(
        pool_df=pool, axes=[GRAVY_AXIS], cfg=AdaptiveBiasConfig(), state=None, L=2,
        position_classes=["distal_surface"] * 2, sasa_fraction=np.ones(2),
        fixed_idx=set(), over_rep_mask=mask)
    d = res.per_position_delta
    assert res.outcomes[0].u < 0
    assert d[0, AA_TO_IDX["I"]] > 0 and d[0, AA_TO_IDX["V"]] > 0   # spread, not just L


def test_freeze_clears_history_for_clean_recovery():
    rng = np.random.default_rng(23)
    pool = make_pool(GRAVY_AXIS, n=200, mean=0.9, std=0.1, rng=rng)
    st = AxisState(last_u=0.3, history=[(0.0, 0.5)])   # in-step append -> wrong sign
    oc, st_new = step_axis(pool, GRAVY_AXIS, st, AdaptiveBiasConfig())
    assert oc.frozen_wrong_sign and oc.u == 0.0
    assert st_new.history == []                        # cleared -> deterministic recovery


def test_online_gain_estimate_sign_and_value():
    cfg = AdaptiveBiasConfig()
    scale = GRAVY_AXIS.error_scale
    # history: pushed +0.3, mean dropped 0.4 -> K=-1.333 (correct sign) -> correction>0
    corr, wrong = ab.estimate_gain_correction([(0.0, 0.5), (0.3, 0.1)], cfg, scale)
    assert not wrong and corr > 0
    # units: corr ~ error_scale/(gain*|K|)
    assert corr == pytest.approx(scale / (cfg.gain * (0.4 / 0.3)), rel=0.01)
    # wrong sign: pushed +0.3, mean rose -> freeze
    corr2, wrong2 = ab.estimate_gain_correction([(0.0, 0.5), (0.3, 0.9)], cfg, scale)
    assert wrong2 and corr2 == 0.0


def test_gain_estimate_ignores_tiny_du():
    # converged/held axis: du ~ 0 but mean wiggles from noise -> must NOT explode/freeze
    cfg = AdaptiveBiasConfig()
    hist = [(0.55, -0.16), (0.5501, -0.10)]   # |du|=1e-4 < du_min
    corr, wrong = ab.estimate_gain_correction(hist, cfg, GRAVY_AXIS.error_scale)
    assert corr == 1.0 and not wrong
