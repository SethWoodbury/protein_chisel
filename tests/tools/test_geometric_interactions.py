"""Tests for tools.geometric_interactions."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from protein_chisel.tools.geometric_interactions import (
    InteractionPanel, detect_interactions,
)


def _write_pdb(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "tmp.pdb"
    p.write_text(textwrap.dedent(body).strip() + "\nEND\n")
    return p


# ----------------------------------------------------------------------
# Empty / trivial cases
# ----------------------------------------------------------------------


def test_empty_pdb_returns_empty_panel(tmp_path):
    p = _write_pdb(tmp_path, "")
    panel = detect_interactions(p)
    assert isinstance(panel, InteractionPanel)
    assert panel.n_total == 0


def test_protein_only_no_ligand(tmp_path):
    """No HETATMs -> no protein↔ligand interactions."""
    p = _write_pdb(tmp_path, """
        ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
        ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C
    """)
    panel = detect_interactions(p, selection="protein_vs_ligand")
    assert panel.n_total == 0


# ----------------------------------------------------------------------
# H-bond detection (canonical Ser-OG donor → ligand O acceptor)
# ----------------------------------------------------------------------


def test_hbond_canonical_distance_and_angle(tmp_path):
    """Ser-OG (donor) at 2.9 A from a HETATM O (acceptor), with the
    SER CB - OG vector pointing toward the ligand O."""
    p = _write_pdb(tmp_path, """
        ATOM      1  N   SER A   1       0.000   0.000   0.000  1.00  0.00           N
        ATOM      2  CA  SER A   1       1.500   0.000   0.000  1.00  0.00           C
        ATOM      3  CB  SER A   1       2.250   1.300   0.000  1.00  0.00           C
        ATOM      4  OG  SER A   1       3.700   1.300   0.000  1.00  0.00           O
        HETATM    5  O1  LIG B 100       6.600   1.300   0.000  1.00  0.00           O
    """)
    panel = detect_interactions(p)
    hb = [i for i in panel.interactions if i.type == "hbond"]
    assert hb, "expected an H-bond between SER OG and LIG O1"
    assert 2.5 < hb[0].distance < 3.5
    assert 0 <= hb[0].strength <= 1


def test_hbond_blocked_by_bad_antecedent_angle(tmp_path):
    """CB and ligand-O on the SAME side of OG. There's no room for an H
    between OG and ligand-O (antecedent-D-A angle ~0 deg) -- the
    putative H is forced AWAY from the acceptor, so the algorithm
    must reject."""
    p = _write_pdb(tmp_path, """
        ATOM      1  N   SER A   1       0.000   0.000   0.000  1.00  0.00           N
        ATOM      2  CA  SER A   1       1.500   0.000   0.000  1.00  0.00           C
        ATOM      3  CB  SER A   1       2.250   1.300   0.000  1.00  0.00           C
        ATOM      4  OG  SER A   1       3.700   1.300   0.000  1.00  0.00           O
        HETATM    5  C1  LIG B 100       3.200   2.500   0.000  1.00  0.00           C
        HETATM    6  O1  LIG B 100       1.700   3.500   0.000  1.00  0.00           O
    """)
    panel = detect_interactions(p)
    # Both directions fail their angle checks:
    # - SER-OG as donor: CB and LIG-O on same side of OG -> ~48 deg
    # - LIG-O as donor: C1 (its covalent partner) and SER-OG on same
    #   side of LIG-O -> antecedent angle small, also rejected.
    hb = [i for i in panel.interactions if i.type == "hbond"]
    assert not hb, f"unexpected h-bond: {hb}"


def test_hbond_distance_too_far(tmp_path):
    p = _write_pdb(tmp_path, """
        ATOM      1  CA  SER A   1       0.000   0.000   0.000  1.00  0.00           C
        ATOM      2  CB  SER A   1       1.500   0.000   0.000  1.00  0.00           C
        ATOM      3  OG  SER A   1       2.000   1.000   0.000  1.00  0.00           O
        HETATM    4  O1  LIG B 100       8.000   1.000   0.000  1.00  0.00           O
    """)
    panel = detect_interactions(p)
    hb = [i for i in panel.interactions if i.type == "hbond"]
    assert not hb


# ----------------------------------------------------------------------
# Salt bridge: K-NZ vs ligand O (or D-OD vs ligand N)
# ----------------------------------------------------------------------


def test_salt_bridge_lysine_to_ligand_carboxylate(tmp_path):
    p = _write_pdb(tmp_path, """
        ATOM      1  N   LYS A   1       0.000   0.000   0.000  1.00  0.00           N
        ATOM      2  CA  LYS A   1       1.500   0.000   0.000  1.00  0.00           C
        ATOM      3  CB  LYS A   1       2.250   1.300   0.000  1.00  0.00           C
        ATOM      4  CG  LYS A   1       3.700   1.300   0.000  1.00  0.00           C
        ATOM      5  CD  LYS A   1       4.450   2.600   0.000  1.00  0.00           C
        ATOM      6  CE  LYS A   1       5.900   2.600   0.000  1.00  0.00           C
        ATOM      7  NZ  LYS A   1       6.650   3.900   0.000  1.00  0.00           N
        HETATM    8  O1  LIG B 100       7.700   6.500   0.000  1.00  0.00           O
    """)
    panel = detect_interactions(p)
    sb = [i for i in panel.interactions if i.type == "salt_bridge"]
    assert sb
    assert sb[0].atom_a == "NZ"
    assert sb[0].res_b_name == "LIG"


# ----------------------------------------------------------------------
# Hydrophobic
# ----------------------------------------------------------------------


def test_hydrophobic_leu_to_ligand_carbon(tmp_path):
    p = _write_pdb(tmp_path, """
        ATOM      1  N   LEU A   1       0.000   0.000   0.000  1.00  0.00           N
        ATOM      2  CA  LEU A   1       1.500   0.000   0.000  1.00  0.00           C
        ATOM      3  CB  LEU A   1       2.250   1.300   0.000  1.00  0.00           C
        ATOM      4  CG  LEU A   1       3.700   1.300   0.000  1.00  0.00           C
        ATOM      5  CD1 LEU A   1       4.450   2.600   0.000  1.00  0.00           C
        ATOM      6  CD2 LEU A   1       4.450   0.000   0.000  1.00  0.00           C
        HETATM    7  C1  LIG B 100       8.000   1.300   0.000  1.00  0.00           C
    """)
    panel = detect_interactions(p)
    hyd = [i for i in panel.interactions if i.type == "hydrophobic"]
    assert hyd


# ----------------------------------------------------------------------
# vdW clash
# ----------------------------------------------------------------------


def test_vdw_clash_fires_on_overlap(tmp_path):
    p = _write_pdb(tmp_path, """
        ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
        HETATM    2  C1  LIG B 100       1.500   0.000   0.000  1.00  0.00           C
    """)
    panel = detect_interactions(p)
    cl = [i for i in panel.interactions if i.type == "vdw_clash"]
    assert cl
    assert cl[0].distance < 2.0


# ----------------------------------------------------------------------
# Panel aggregation
# ----------------------------------------------------------------------


def test_panel_to_dict_has_all_typed_keys(tmp_path):
    p = _write_pdb(tmp_path, "")
    panel = detect_interactions(p)
    d = panel.to_dict("foo__")
    expected_keys = {
        "foo__n_total",
        "foo__n_hbond", "foo__strength_hbond",
        "foo__n_salt_bridge", "foo__strength_salt_bridge",
        "foo__n_pi_pi", "foo__strength_pi_pi",
        "foo__n_pi_cation", "foo__strength_pi_cation",
        "foo__n_hydrophobic", "foo__strength_hydrophobic",
        "foo__n_vdw_clash", "foo__strength_vdw_clash",
        "foo__strength_total",
    }
    assert expected_keys.issubset(d.keys())


def test_panel_strength_zero_when_no_interactions(tmp_path):
    p = _write_pdb(tmp_path, "")
    panel = detect_interactions(p)
    d = panel.to_dict()
    assert d["geom_int__strength_total"] == 0
