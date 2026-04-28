"""Test the contact_ms tool — runs in esmc.sif (where py_contact_ms is).

py_contact_ms is numpy-only and does not need PyRosetta, so this test
just needs the esmc.sif environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.cluster


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"
APO_PDB = TEST_DIR / "af3_pred.pdb"


def test_contact_ms_design_has_signal():
    from protein_chisel.tools.contact_ms import contact_ms_protein_ligand

    result = contact_ms_protein_ligand(DESIGN_PDB)
    # Designed enzyme: protein wraps the ligand → CMS > 0
    assert result.total_cms > 0.0, f"expected CMS > 0, got {result.total_cms}"
    assert result.n_target_atoms > 0
    assert result.n_binder_atoms > 100
    # py_contact_ms only returns per-atom for the target side
    assert len(result.per_atom_cms_target) == result.n_target_atoms


def test_contact_ms_apo_zero():
    from protein_chisel.tools.contact_ms import contact_ms_protein_ligand

    result = contact_ms_protein_ligand(APO_PDB)
    assert result.total_cms == 0.0
    assert result.n_target_atoms == 0


def test_contact_ms_to_dict():
    from protein_chisel.tools.contact_ms import contact_ms_protein_ligand

    result = contact_ms_protein_ligand(DESIGN_PDB)
    d = result.to_dict()
    assert "cms__total" in d
    assert d["cms__total"] == result.total_cms
