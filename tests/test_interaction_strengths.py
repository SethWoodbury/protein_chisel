"""Host tests for the strength-weighted layer over chemical_interactions."""

from __future__ import annotations

import math

import pytest

from protein_chisel.tools.chemical_interactions import (
    INTERACTION_GEOMETRY,
    InteractionsResult,
    interaction_strengths,
    _gauss_strength,
)


def _make_interactions(
    *, hbonds=(), salt_bridges=(), pi_pi=(), pi_cation=()
) -> InteractionsResult:
    return InteractionsResult(
        hbonds=list(hbonds),
        salt_bridges=list(salt_bridges),
        pi_pi=list(pi_pi),
        pi_cation=list(pi_cation),
    )


def test_gauss_strength_peaks_at_d0():
    g = INTERACTION_GEOMETRY["salt_bridge"]
    assert _gauss_strength(g["d0"], **g) == pytest.approx(1.0)
    assert _gauss_strength(g["d0"] + 5 * g["sigma"], **g) < 1e-3
    assert _gauss_strength(float("nan"), **g) == 0.0


def test_strength_empty_input():
    res = interaction_strengths(_make_interactions())
    assert res.n_total_contacts if hasattr(res, "n_total_contacts") else True
    assert res.by_type_strength_sum["hbond"] == 0.0
    assert res.by_type_count["salt_bridge"] == 0
    assert res.per_residue_strength == {}


def test_strength_single_salt_bridge_at_optimum():
    sb = {"pos_res": 10, "pos_name3": "ARG", "pos_atom": "NH1",
          "neg_res": 25, "neg_name3": "ASP", "neg_atom": "OD1",
          "distance": 3.0}
    res = interaction_strengths(_make_interactions(salt_bridges=[sb]))
    assert res.by_type_count["salt_bridge"] == 1
    # at d0 = 3.0 → strength = 1.0
    assert res.by_type_strength_sum["salt_bridge"] == pytest.approx(1.0)
    # both partners get strength
    assert res.per_residue_strength[10] == pytest.approx(1.0)
    assert res.per_residue_strength[25] == pytest.approx(1.0)


def test_strength_far_salt_bridge_decays():
    sb_close = {"pos_res": 1, "neg_res": 2, "distance": 3.0,
                "pos_name3": "ARG", "pos_atom": "NH1",
                "neg_name3": "ASP", "neg_atom": "OD1"}
    sb_far = {**sb_close, "pos_res": 3, "neg_res": 4, "distance": 5.0}
    res = interaction_strengths(_make_interactions(salt_bridges=[sb_close, sb_far]))
    s_close = res.per_residue_strength[1]
    s_far = res.per_residue_strength[3]
    assert s_close > s_far
    assert s_far > 0.0


def test_strength_pi_pi_angle_factor_stacked_vs_tilted():
    """Stacked (0°) and t-shape (90°) get full strength; 45° gets half."""
    common = {"name3_a": "PHE", "name3_b": "TYR", "centroid_distance": 4.0,
              "geometry": "stacked"}
    stacked = {"res_a": 1, "res_b": 2, "plane_angle_deg": 0.0, **common}
    tilted = {"res_a": 3, "res_b": 4, "plane_angle_deg": 45.0, **common,
              "geometry": "tilted"}
    t_shape = {"res_a": 5, "res_b": 6, "plane_angle_deg": 90.0, **common,
               "geometry": "t_shape"}
    res = interaction_strengths(_make_interactions(pi_pi=[stacked, tilted, t_shape]))
    s_stacked = res.per_residue_strength_by_type[1]["pi_pi"]
    s_tilted = res.per_residue_strength_by_type[3]["pi_pi"]
    s_tshape = res.per_residue_strength_by_type[5]["pi_pi"]
    assert s_stacked > s_tilted
    assert s_tshape > s_tilted
    # 0° and 90° should be roughly equal
    assert s_stacked == pytest.approx(s_tshape, rel=1e-3)
    # 45° should be ~ half
    assert s_tilted == pytest.approx(s_stacked * 0.5, rel=0.05)


def test_strength_hbond_uses_energy_proxy():
    """Hbond strength uses the Rosetta energy as a soft proxy."""
    # Strong hbond (energy = -3 kcal/mol) → strength ~ 0.99
    strong = {"donor_res": 10, "donor_h_atom": "HD1", "donor_heavy_atom": "ND1",
              "donor_name3": "HIS", "acceptor_res": 25, "acceptor_atom": "OD1",
              "acceptor_name3": "ASP", "energy": -3.0}
    weak = {**strong, "donor_res": 11, "acceptor_res": 26, "energy": -0.3}
    res = interaction_strengths(_make_interactions(hbonds=[strong, weak]))
    s_strong = res.per_residue_strength[10]
    s_weak = res.per_residue_strength[11]
    assert s_strong > s_weak
    assert s_strong > 0.9
    assert s_weak < 0.5
    # weighted_hbond_energy aggregates -E * strength; both contributions
    # are positive (since energies are negative).
    assert res.weighted_hbond_energy > 0.0


def test_strength_per_residue_rollup_sums_across_types():
    sb = {"pos_res": 100, "neg_res": 200, "distance": 3.0,
          "pos_name3": "K", "pos_atom": "NZ",
          "neg_name3": "E", "neg_atom": "OE1"}
    pp = {"res_a": 100, "res_b": 50, "centroid_distance": 4.0,
          "name3_a": "F", "name3_b": "Y", "plane_angle_deg": 0.0,
          "geometry": "stacked"}
    res = interaction_strengths(_make_interactions(salt_bridges=[sb], pi_pi=[pp]))
    # Residue 100 participates in BOTH a salt bridge and a π-π pair → total
    # strength = both contributions added.
    assert "salt_bridge" in res.per_residue_strength_by_type[100]
    assert "pi_pi" in res.per_residue_strength_by_type[100]
    s100 = res.per_residue_strength[100]
    # Check additive
    assert s100 == pytest.approx(
        res.per_residue_strength_by_type[100]["salt_bridge"]
        + res.per_residue_strength_by_type[100]["pi_pi"],
        rel=1e-6,
    )


def test_strength_to_dict_keys():
    sb = {"pos_res": 1, "neg_res": 2, "distance": 3.0,
          "pos_name3": "K", "pos_atom": "NZ",
          "neg_name3": "E", "neg_atom": "OE1"}
    res = interaction_strengths(_make_interactions(salt_bridges=[sb]))
    d = res.to_dict()
    assert "interact_strength__salt_bridge__sum" in d
    assert "interact_strength__salt_bridge__count" in d
    assert "interact_strength__weighted_hbond_energy" in d


# ---- Codex review: NaN / edge-case regression tests --------------------


def test_strength_nan_hbond_energy_is_zero():
    """A NaN hbond energy must NOT poison the aggregate sum (codex review)."""
    nan_hb = {"donor_res": 10, "donor_h_atom": "HD1", "donor_heavy_atom": "ND1",
              "donor_name3": "HIS", "acceptor_res": 25, "acceptor_atom": "OD1",
              "acceptor_name3": "ASP", "energy": float("nan")}
    res = interaction_strengths(_make_interactions(hbonds=[nan_hb]))
    assert math.isfinite(res.by_type_strength_sum["hbond"])
    assert res.by_type_strength_sum["hbond"] == 0.0
    assert math.isfinite(res.weighted_hbond_energy)
    # the energy is NaN so it shouldn't add to the weighted sum either
    assert res.weighted_hbond_energy == 0.0


def test_strength_positive_hbond_energy_is_zero():
    """Repulsive (positive) hbond energy → 0 strength."""
    pos_hb = {"donor_res": 1, "donor_h_atom": "HD1", "donor_heavy_atom": "ND1",
              "donor_name3": "HIS", "acceptor_res": 2, "acceptor_atom": "OD1",
              "acceptor_name3": "ASP", "energy": +0.5}
    res = interaction_strengths(_make_interactions(hbonds=[pos_hb]))
    assert res.by_type_strength_sum["hbond"] == 0.0


def test_strength_nan_pi_pi_angle_is_zero():
    """NaN π-π plane angle must NOT poison the aggregate."""
    nan_pp = {"res_a": 1, "res_b": 2, "centroid_distance": 4.0,
              "name3_a": "F", "name3_b": "Y",
              "plane_angle_deg": float("nan"), "geometry": "stacked"}
    res = interaction_strengths(_make_interactions(pi_pi=[nan_pp]))
    assert math.isfinite(res.by_type_strength_sum["pi_pi"])
    assert res.by_type_strength_sum["pi_pi"] == 0.0


def test_strength_negative_pi_pi_angle_works():
    """Negative angle (rotation in either direction) gives same strength as positive."""
    pp_pos = {"res_a": 1, "res_b": 2, "centroid_distance": 4.0,
              "name3_a": "F", "name3_b": "Y",
              "plane_angle_deg": 30.0, "geometry": "tilted"}
    pp_neg = {**pp_pos, "res_a": 3, "res_b": 4, "plane_angle_deg": -30.0}
    res = interaction_strengths(_make_interactions(pi_pi=[pp_pos, pp_neg]))
    s_pos = res.per_residue_strength_by_type[1]["pi_pi"]
    s_neg = res.per_residue_strength_by_type[3]["pi_pi"]
    assert s_pos == pytest.approx(s_neg, rel=1e-6)


def test_strength_partial_geometry_override_keeps_d0():
    """Codex review: shallow merge dropped d0 when caller overrode only sigma."""
    sb = {"pos_res": 1, "neg_res": 2, "distance": 3.0,
          "pos_name3": "K", "pos_atom": "NZ",
          "neg_name3": "E", "neg_atom": "OE1"}
    res_default = interaction_strengths(_make_interactions(salt_bridges=[sb]))
    # Override sigma only — d0 should stay at 3.0 (the default), so distance
    # 3.0 still scores 1.0 even with a tighter sigma.
    res_overridden = interaction_strengths(
        _make_interactions(salt_bridges=[sb]),
        geometry={"salt_bridge": {"sigma": 0.2}},
    )
    assert res_default.by_type_strength_sum["salt_bridge"] == pytest.approx(1.0)
    assert res_overridden.by_type_strength_sum["salt_bridge"] == pytest.approx(1.0)


def test_strength_distance_zero_salt_bridge():
    """Zero-distance is finite and scores below 1 (further from d0=3.0)."""
    sb = {"pos_res": 1, "neg_res": 2, "distance": 0.0,
          "pos_name3": "K", "pos_atom": "NZ",
          "neg_name3": "E", "neg_atom": "OE1"}
    res = interaction_strengths(_make_interactions(salt_bridges=[sb]))
    s = res.by_type_strength_sum["salt_bridge"]
    assert math.isfinite(s)
    assert 0.0 < s < 1.0  # distance 0 is far from d0=3 in (d-d0)/sigma terms
