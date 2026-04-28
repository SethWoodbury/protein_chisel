"""Cluster tests for chemistry tools batch:
chemical_interactions, buns, catres_quality.

py_contact_ms test runs in esmc.sif (different sif), see test_contact_ms.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.cluster


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"
PARAMS_DIR = Path(
    "/home/woodbuse/testing_space/scaffold_optimization/"
    "ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params"
)


# ---- chemical_interactions ------------------------------------------------


def test_chemical_interactions_design():
    from protein_chisel.tools.chemical_interactions import chemical_interactions

    res = chemical_interactions(DESIGN_PDB, params=[PARAMS_DIR])

    # Designed proteins should have many hbonds
    assert len(res.hbonds) > 50, f"unexpectedly few hbonds: {len(res.hbonds)}"
    # All hbond energies negative (favorable)
    assert all(h["energy"] < 0 for h in res.hbonds)

    # Each hbond row has the canonical fields
    for h in res.hbonds[:3]:
        assert {"donor_res", "acceptor_res", "donor_atom", "acceptor_atom", "energy"} <= set(h)

    # Salt bridges, π-π, π-cation may or may not occur but should return lists
    assert isinstance(res.salt_bridges, list)
    assert isinstance(res.pi_pi, list)
    assert isinstance(res.pi_cation, list)

    summary = res.summary()
    assert summary["interact__n_hbonds"] == len(res.hbonds)
    # sum_hbond_energy should be negative (favorable)
    assert summary["interact__sum_hbond_energy"] < 0


def test_chemical_interactions_pi_pi_geometry():
    """Check that any detected π-π pair has a sensible plane angle."""
    from protein_chisel.tools.chemical_interactions import chemical_interactions

    res = chemical_interactions(DESIGN_PDB, params=[PARAMS_DIR])
    for p in res.pi_pi:
        assert 0.0 <= p["plane_angle_deg"] <= 90.0
        assert p["geometry"] in ("stacked", "tilted", "t_shape")
        assert p["res_a"] != p["res_b"]


# ---- BUNS -----------------------------------------------------------------


def test_buns_no_whitelist():
    from protein_chisel.tools.buns import buns

    res = buns(DESIGN_PDB, params=[PARAMS_DIR])
    # Plausible range — well-designed structures have low BUNS counts
    # but our test PDB has metal coordination (Zn) which can leave
    # apparent unsats around the active site.
    assert res.n_buried_polar_total > 0
    assert res.n_buried_unsat >= 0
    # Each unsat row has the required fields
    for u in res.buried_unsat_atoms:
        assert {"resno", "atom_name", "sasa", "name3"} <= set(u)


def test_buns_whitelist_reduces_count():
    """Adding the catalytic residues to the whitelist removes them from BUNS."""
    from protein_chisel.tools.buns import buns, whitelist_from_remark_666

    base = buns(DESIGN_PDB, params=[PARAMS_DIR])
    wl = whitelist_from_remark_666(DESIGN_PDB)
    assert len(wl) > 0
    with_wl = buns(DESIGN_PDB, params=[PARAMS_DIR], whitelist=wl)

    # Whitelist should never increase the unsat count
    assert with_wl.n_buried_unsat <= base.n_buried_unsat
    assert with_wl.n_whitelisted >= 0


def test_buns_whitelist_skips_atoms():
    """Whitelist of every buried polar should drive count to zero."""
    from protein_chisel.tools.buns import buns

    base = buns(DESIGN_PDB, params=[PARAMS_DIR])
    wl = [(u["resno"], u["atom_name"]) for u in base.buried_unsat_atoms]
    if not wl:
        pytest.skip("no unsat atoms to whitelist (already clean design)")
    with_full_wl = buns(DESIGN_PDB, params=[PARAMS_DIR], whitelist=wl)
    assert with_full_wl.n_buried_unsat == 0


# ---- catres_quality -------------------------------------------------------


def test_catres_quality_design():
    from protein_chisel.tools.catres_quality import catres_quality

    res = catres_quality(DESIGN_PDB, params=[PARAMS_DIR])

    # 6 catalytic residues from REMARK 666
    assert res.n_residues == 6
    assert len(res.per_residue) == 6

    # Catalytic residues are HIS / LYS / GLU per REMARK 666
    name3s = {row.name3 for row in res.per_residue}
    assert name3s.issubset({"HIS", "LYS", "GLU"})

    # bondlen_max_dev should be small for non-broken residues (< 0.5 Å)
    # Use a generous threshold; the test only flags catastrophic breaks.
    assert res.bondlen_max_dev < 0.5

    # n_broken_sidechains should be 0 for a sensible design
    assert res.n_broken_sidechains == 0


def test_catres_quality_explicit_resnos():
    """Pass explicit resnos instead of REMARK 666."""
    from protein_chisel.tools.catres_quality import catres_quality

    res = catres_quality(DESIGN_PDB, params=[PARAMS_DIR], catres_resnos=[64])
    assert res.n_residues == 1
    assert res.per_residue[0].resno == 64
    assert res.per_residue[0].name3 == "LYS"
