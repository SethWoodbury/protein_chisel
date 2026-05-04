"""Tests for utils/sidechain_geometry — phantom CB, centroid, orient, soft scores."""

from __future__ import annotations

import math

import numpy as np
import pytest

from protein_chisel.utils.sidechain_geometry import (
    BACKBONE_ATOMS,
    FUNCTIONAL_ATOMS,
    MAX_SASA_BY_AA,
    RING_ATOMS,
    _CHIRALITY_SIGN,
    functional_atom_position,
    max_sasa_for,
    orientation_angle_deg,
    phantom_cb,
    ring_centroid,
    sidechain_atom_names,
    sidechain_centroid,
    sigmoid,
)


# ---- Phantom CB -----------------------------------------------------------


def test_chirality_sign_was_validated():
    """Module-load auto-validation must have picked +1 or -1."""
    assert _CHIRALITY_SIGN in (+1, -1)


def test_phantom_cb_l_alanine_geometry():
    """Phantom CB on canonical L-Ala backbone matches the real Cβ to <0.15 Å."""
    # Same coords used in the chirality probe.
    N = np.array([1.458, 0.000, 0.000])
    CA = np.array([2.009, 1.420, 0.000])
    C = np.array([3.535, 1.420, 0.000])
    CB_canonical = np.array([1.483, 2.158, -1.190])
    cb_phantom = phantom_cb(N, CA, C)
    err = float(np.linalg.norm(cb_phantom - CB_canonical))
    assert err < 0.15, f"phantom CB error {err:.3f} Å exceeds 0.15 Å"


def test_phantom_cb_bond_length_is_1_522():
    """Phantom CB should sit exactly 1.522 Å from CA (Engh & Huber)."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        N = rng.normal(size=3)
        CA = rng.normal(size=3)
        C = rng.normal(size=3)
        # Skip degenerate cases
        if np.linalg.norm(N - CA) < 0.5 or np.linalg.norm(C - CA) < 0.5:
            continue
        cb = phantom_cb(N, CA, C)
        assert abs(float(np.linalg.norm(cb - CA)) - 1.522) < 1e-6


# ---- Sidechain helpers ----------------------------------------------------


def test_sidechain_atom_names_known_aa():
    """Should match the canonical clash_check table."""
    arg_atoms = sidechain_atom_names("ARG")
    assert {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"}.issubset(arg_atoms)
    # Backbone never appears.
    for bb in BACKBONE_ATOMS:
        assert bb not in arg_atoms


def test_sidechain_centroid_arg_at_typical_geometry():
    """Centroid of a 'real' Arg should be along the chi1-extended direction."""
    coords = {
        "N":   np.array([0.0, 0.0, 0.0]),
        "CA":  np.array([1.5, 0.0, 0.0]),
        "C":   np.array([2.0, 1.4, 0.0]),
        "CB":  np.array([1.7, 0.0, 1.5]),    # chi1 ~ trans
        "CG":  np.array([2.7, 0.0, 2.7]),
        "CD":  np.array([3.2, 0.0, 4.2]),
        "NE":  np.array([4.5, 0.5, 4.7]),
        "CZ":  np.array([5.2, 1.5, 5.4]),
        "NH1": np.array([6.4, 1.7, 5.0]),
        "NH2": np.array([4.8, 2.5, 6.4]),
    }
    cen = sidechain_centroid(coords, "ARG")
    assert cen is not None
    # Centroid should be roughly along +z, well above CA.
    assert cen[2] > 2.0


def test_sidechain_centroid_glycine_uses_phantom():
    """Gly with no CB: centroid falls back to provided phantom CB."""
    coords = {
        "N":  np.array([0.0, 0.0, 0.0]),
        "CA": np.array([1.5, 0.0, 0.0]),
        "C":  np.array([2.0, 1.4, 0.0]),
    }
    phantom = np.array([1.0, 1.0, 1.0])
    cen = sidechain_centroid(coords, "GLY", fallback_phantom_cb=phantom)
    assert cen is not None
    np.testing.assert_array_equal(cen, phantom)


def test_functional_atom_arg_is_guanidinium_mean():
    coords = {
        "NH1": np.array([6.4, 1.7, 5.0]),
        "NH2": np.array([4.8, 2.5, 6.4]),
    }
    pt = functional_atom_position(coords, "ARG")
    assert pt is not None
    np.testing.assert_allclose(pt, np.mean([coords["NH1"], coords["NH2"]], axis=0))


def test_functional_atom_his_picks_closer_to_ligand():
    """His tautomer-agnostic: returns the atom (ND1 or NE2) closer to ligand."""
    coords = {
        "ND1": np.array([0.0, 0.0, 0.0]),
        "NE2": np.array([5.0, 0.0, 0.0]),
    }
    # Ligand at +x → NE2 is closer.
    pt = functional_atom_position(coords, "HIS",
                                    ligand_centroid=np.array([6.0, 0.0, 0.0]))
    np.testing.assert_array_equal(pt, coords["NE2"])
    # Ligand at -x → ND1 is closer.
    pt = functional_atom_position(coords, "HIS",
                                    ligand_centroid=np.array([-6.0, 0.0, 0.0]))
    np.testing.assert_array_equal(pt, coords["ND1"])


def test_ring_centroid_phe_is_geometric_center():
    coords = {
        "CG":  np.array([0.0, 0.0, 0.0]),
        "CD1": np.array([1.4, 0.0, 0.0]),
        "CD2": np.array([-0.7, 1.21, 0.0]),
        "CE1": np.array([2.1, 1.21, 0.0]),
        "CE2": np.array([0.0, 2.42, 0.0]),
        "CZ":  np.array([1.4, 2.42, 0.0]),
    }
    cen = ring_centroid(coords, "PHE")
    assert cen is not None
    expected = np.mean(np.stack(list(coords.values()), axis=0), axis=0)
    np.testing.assert_allclose(cen, expected)


# ---- Orientation angle ----------------------------------------------------


def test_orientation_angle_pointing_toward_is_zero():
    ca = np.array([0.0, 0.0, 0.0])
    cb = np.array([1.0, 0.0, 0.0])
    lig = np.array([5.0, 0.0, 0.0])
    assert orientation_angle_deg(ca, cb, lig) == pytest.approx(0.0, abs=1e-6)


def test_orientation_angle_pointing_away_is_180():
    ca = np.array([0.0, 0.0, 0.0])
    cb = np.array([1.0, 0.0, 0.0])
    lig = np.array([-5.0, 0.0, 0.0])
    assert orientation_angle_deg(ca, cb, lig) == pytest.approx(180.0, abs=1e-6)


def test_orientation_angle_perpendicular_is_90():
    ca = np.array([0.0, 0.0, 0.0])
    cb = np.array([0.0, 1.0, 0.0])
    lig = np.array([5.0, 0.0, 0.0])
    assert orientation_angle_deg(ca, cb, lig) == pytest.approx(90.0, abs=1e-6)


def test_orientation_angle_zero_vector_returns_nan():
    ca = np.array([0.0, 0.0, 0.0])
    cb = np.array([0.0, 0.0, 0.0])
    lig = np.array([1.0, 0.0, 0.0])
    assert math.isnan(orientation_angle_deg(ca, cb, lig))


# ---- Sigmoid + max SASA ---------------------------------------------------


def test_sigmoid_at_zero_is_half():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(np.array([0.0])).item() == pytest.approx(0.5)


def test_sigmoid_extremes_no_overflow():
    # Very large positive → 1; very negative → 0; no overflow warning.
    assert sigmoid(1000.0) == pytest.approx(1.0)
    assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-12)


def test_max_sasa_known_values():
    assert max_sasa_for("ALA") == pytest.approx(129.0)
    assert max_sasa_for("ARG") == pytest.approx(274.0)
    # Modified-residue alias falls through to parent.
    assert max_sasa_for("KCX") == pytest.approx(236.0)
    # Unknown falls through to ALA.
    assert max_sasa_for("XXX") == pytest.approx(MAX_SASA_BY_AA["ALA"])
