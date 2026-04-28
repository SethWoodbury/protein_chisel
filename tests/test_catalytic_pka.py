"""Tests for tools/catalytic_pka. Runs in esmc.sif (where propka lives)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.cluster

TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"


def test_catalytic_pka_runs_on_design():
    from protein_chisel.tools.catalytic_pka import catalytic_pka

    res = catalytic_pka(DESIGN_PDB)
    # 6 catalytic residues from REMARK 666; PROPKA should hit all titratable ones.
    # Out of 6 (4 HIS, 1 LYS, 1 GLU), all 6 are titratable so n should be 6.
    assert res.n_catres_evaluated == 6
    assert len(res.catres_pka_shift) == 6
    # Per-residue dict has at least the catalytic residues
    assert len(res.per_residue_pka) >= 6


def test_catalytic_pka_to_dict_keys():
    from protein_chisel.tools.catalytic_pka import catalytic_pka

    res = catalytic_pka(DESIGN_PDB)
    d = res.to_dict()
    assert "pka__n_catres_evaluated" in d
    # At least one per-residue pKa key
    assert any("__pka" in k and "catres__" in k for k in d)


def test_catalytic_pka_explicit_catres_subset():
    from protein_chisel.tools.catalytic_pka import catalytic_pka

    # Only LYS64 from REMARK 666
    res = catalytic_pka(DESIGN_PDB, catres=[("A", 64)])
    assert res.n_catres_evaluated == 1
    keys = list(res.catres_pka.keys())
    assert keys[0][:2] == ("A", 64)
