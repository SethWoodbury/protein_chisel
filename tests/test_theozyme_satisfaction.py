"""Tests for tools/theozyme_satisfaction (host-side; pure Python)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from protein_chisel.tools.theozyme_satisfaction import theozyme_satisfaction


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"
REFINED_PDB = TEST_DIR / "refined.pdb"
APO_PDB = TEST_DIR / "af3_pred.pdb"


def test_theozyme_satisfaction_no_reference_returns_distances_only():
    res = theozyme_satisfaction(DESIGN_PDB)
    # No theozyme_pdb means no alignment-based metrics
    assert res.n_catalytic == 6
    assert res.motif_rmsd != res.motif_rmsd  # NaN
    # But ligand-distance pairs should be populated (catres atoms vs YYE)
    assert len(res.catres_to_ligand_distances) > 0


def test_theozyme_satisfaction_design_vs_design_zero_rmsd():
    """design vs itself → motif_rmsd == 0."""
    res = theozyme_satisfaction(DESIGN_PDB, theozyme_pdb=DESIGN_PDB)
    assert res.n_catalytic == 6
    assert res.motif_rmsd < 0.001
    assert res.motif_heavy_rmsd < 0.001


def test_theozyme_satisfaction_design_vs_refined_small_drift():
    """design vs AF3-refined should have small but nonzero motif RMSD."""
    res = theozyme_satisfaction(DESIGN_PDB, theozyme_pdb=REFINED_PDB)
    # AF3 refinement perturbs catalytic residues a bit; not zero, not huge
    assert 0.0 < res.motif_rmsd < 5.0
    assert res.n_catalytic == 6


def test_theozyme_satisfaction_apo_no_ligand():
    """Apo PDB has no ligand → no catres-to-ligand distances."""
    # We pass DESIGN_PDB's REMARK 666 explicitly (apo doesn't have it).
    res = theozyme_satisfaction(
        APO_PDB, explicit_catres=[("A", 188), ("A", 184), ("A", 64),
                                   ("A", 41), ("A", 148), ("A", 187)],
    )
    assert res.n_catalytic == 6
    # No ligand → empty distances
    assert res.catres_to_ligand_distances == {}


def test_theozyme_satisfaction_fixed_atoms_json(tmp_path: Path):
    """Read RFdiffusion3-style fixed_atoms_json instead of REMARK 666."""
    fixed = tmp_path / "fixed.json"
    fixed.write_text(json.dumps({
        str(DESIGN_PDB): ["A188", "A184", "A64", "A41"],
    }))
    res = theozyme_satisfaction(DESIGN_PDB, fixed_atoms_json=fixed)
    assert res.n_catalytic == 4
    # Only 4 residues but still produces ligand distances
    assert len(res.catres_to_ligand_distances) > 0


def test_theozyme_satisfaction_to_dict_keys():
    res = theozyme_satisfaction(DESIGN_PDB, theozyme_pdb=DESIGN_PDB)
    d = res.to_dict()
    assert "theozyme__motif_rmsd" in d
    assert "theozyme__motif_heavy_rmsd" in d
    assert "theozyme__n_catalytic" in d
    # per-residue keys
    assert any("theozyme__per_res__" in k for k in d)
