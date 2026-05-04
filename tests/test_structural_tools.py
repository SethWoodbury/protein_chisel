"""Cluster tests for the structural tools batch:
classify_positions, backbone_sanity, shape_metrics, secondary_structure,
ss_summary, ligand_environment.

Single test module so the PyRosetta init + pose-loading happen once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.cluster


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"
AF3_APO_PDB = TEST_DIR / "af3_pred.pdb"
REFINED_PDB = TEST_DIR / "refined.pdb"

PARAMS_DIR = Path(
    "/home/woodbuse/testing_space/scaffold_optimization/"
    "ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params"
)


# ---- classify_positions ---------------------------------------------------


def test_classify_positions_design_pdb():
    from protein_chisel.tools.classify_positions import classify_positions

    pt = classify_positions(DESIGN_PDB, params=[PARAMS_DIR])
    df = pt.df

    # 208 protein + 1 YYE = 209 rows
    assert len(df) == 209

    # Catalytic residues from REMARK 666 are tagged primary_sphere (new)
    # or active_site (legacy). Accept either so the test is resilient
    # across the directional-classifier rewrite.
    catalytic_resnos = {41, 64, 148, 184, 187, 188}
    cat_rows = df[df["resno"].isin(catalytic_resnos)]
    assert all(cat_rows["class"].isin(["active_site", "primary_sphere"]))
    assert all(cat_rows["is_catalytic"])
    assert len(cat_rows) == 6

    # All non-catalytic protein rows are one of the known class strings
    # (legacy or new vocabulary).
    non_cat_protein = df[df["is_protein"] & ~df["is_catalytic"]]
    valid_classes = {
        "first_shell", "pocket", "buried", "surface",   # legacy
        "primary_sphere", "secondary_sphere", "nearby_surface",
        "distal_buried", "distal_surface",              # new
    }
    assert all(non_cat_protein["class"].isin(valid_classes))

    # Ligand row is class=ligand
    lig_rows = df[~df["is_protein"]]
    assert len(lig_rows) == 1
    assert lig_rows.iloc[0]["class"] == "ligand"
    assert lig_rows.iloc[0]["name3"] == "YYE"

    # Distance fields are sane
    protein = df[df["is_protein"]]
    assert (protein["dist_ligand"] >= 0).all()
    assert (protein["dist_catalytic"] >= 0).all()


def test_classify_positions_apo_pdb():
    """Apo PDB has no ligand and no REMARK 666 — every residue is buried/surface."""
    from protein_chisel.tools.classify_positions import classify_positions

    pt = classify_positions(AF3_APO_PDB)
    df = pt.df

    # No catalytic, no ligand → no primary/secondary sphere either.
    assert df["is_catalytic"].sum() == 0
    assert (df["class"] == "ligand").sum() == 0
    assert (df["class"].isin(["first_shell", "primary_sphere"])).sum() == 0
    assert (df["class"].isin(["active_site", "primary_sphere"])).sum() == 0
    # All rows fall into a buried/surface bucket (legacy or distal_*).
    valid_no_lig = {
        "buried", "surface",                    # legacy
        "distal_buried", "distal_surface",      # new
    }
    assert all(df["class"].isin(valid_no_lig))


def test_classify_positions_persistence(tmp_path: Path):
    """Round-trip via parquet/TSV preserves required columns."""
    from protein_chisel.io.schemas import PositionTable
    from protein_chisel.tools.classify_positions import classify_positions

    pt = classify_positions(DESIGN_PDB, params=[PARAMS_DIR])
    out = tmp_path / "pos.parquet"
    actual = pt.to_parquet(out)
    loaded = PositionTable.from_parquet(actual)
    assert len(loaded.df) == len(pt.df)


# ---- backbone_sanity ------------------------------------------------------


def test_backbone_sanity_design():
    from protein_chisel.tools.backbone_sanity import backbone_sanity

    res = backbone_sanity(DESIGN_PDB, params=[PARAMS_DIR])

    # A clean designed backbone has no chainbreak above 4.5 Å
    assert res.chainbreak_max < 4.5
    assert res.chainbreak_above_4_5 == 0
    # rCA_nonadj is a metric, not a filter; in a sensible structure it's
    # typically > 3.5 Å. Tight turns (e.g. β-hairpin) can push it as low
    # as ~3.8 Å between i and i+3.
    assert res.rCA_nonadj_min > 3.5
    # Termini not buried at the active site
    assert res.term_n_mindist_to_lig > 5.0
    assert res.term_c_mindist_to_lig > 5.0
    assert res.n_residues == 208


def test_backbone_sanity_apo():
    """Apo PDB has no ligand → terminus distances are NaN."""
    import math
    from protein_chisel.tools.backbone_sanity import backbone_sanity

    res = backbone_sanity(AF3_APO_PDB)
    assert math.isnan(res.term_n_mindist_to_lig)
    assert math.isnan(res.term_c_mindist_to_lig)
    assert res.n_residues == 208


# ---- shape_metrics --------------------------------------------------------


def test_shape_metrics_design():
    from protein_chisel.tools.shape_metrics import shape_metrics

    res = shape_metrics(DESIGN_PDB, params=[PARAMS_DIR])
    # A 208-residue globular protein has Rg ≈ 12-18 Å typically.
    assert 10.0 < res.rg < 25.0
    # Length-normalized Rg should be modest (~Rg / sqrt(N) ≈ 1.0-1.4)
    assert 0.5 < res.rg_norm < 2.5
    # Globular proteins have small relative shape anisotropy (well below 1)
    assert 0.0 <= res.rel_shape_anisotropy < 0.5
    assert res.n_residues == 208


# ---- secondary_structure / ss_summary -------------------------------------


def test_secondary_structure_design():
    from protein_chisel.tools.secondary_structure import secondary_structure

    res = secondary_structure(DESIGN_PDB, params=[PARAMS_DIR])
    assert len(res.ss_full) == 208
    assert len(res.ss_reduced) == 208
    # All reduced labels should be in {H, E, L}
    assert set(res.ss_reduced.values()) <= {"H", "E", "L"}


def test_ss_summary_with_catalytic_residues():
    from protein_chisel.tools.secondary_structure import ss_summary

    cat = {41, 64, 148, 184, 187, 188}
    res = ss_summary(DESIGN_PDB, params=[PARAMS_DIR], catalytic_resnos=cat)

    assert res.n_protein == 208
    # Sanity: fractions sum to ~1
    assert abs(res.helix_frac + res.sheet_frac + res.loop_frac - 1.0) < 0.001
    # At least one of helix/sheet exists in any reasonable design
    assert res.helix_count > 0 or res.sheet_count > 0
    # Sum of catalytic-in-* equals number of catalytic residues
    assert res.catalytic_in_helix + res.catalytic_in_sheet + res.catalytic_in_loop == 6


# ---- ligand_environment ---------------------------------------------------


def test_ligand_environment_design():
    from protein_chisel.tools.ligand_environment import ligand_environment

    results = ligand_environment(
        DESIGN_PDB,
        params=[PARAMS_DIR],
        target_atoms=["C1", "O1", "O2"],  # the carbamate cap atoms (KCX)
        compute_relative=True,
    )
    assert len(results) == 1
    r = results[0]
    assert r.ligand_name3 == "YYE"
    assert r.ligand_seqpos == 209
    # Designed enzyme: ligand sits at the active site, lig_dist ~3-5 Å
    assert 0.0 <= r.lig_dist <= 5.0
    # Some residues within 5 Å (active site)
    assert r.n_residues_within_5A > 0
    # Even more within 8 Å
    assert r.n_residues_within_8A >= r.n_residues_within_5A
    # The carbamate cap atoms are present in per_atom_sasa
    assert set(r.per_atom_sasa) == {"C1", "O1", "O2"}
    # Ligand SASA is finite and bounded
    assert r.ligand_sasa >= 0.0
    # Relative SASA should be 0..1 (slightly outside is OK due to method differences)
    assert 0.0 <= r.ligand_sasa_relative <= 1.5


def test_ligand_environment_apo():
    """Apo PDB returns an empty list."""
    from protein_chisel.tools.ligand_environment import ligand_environment

    results = ligand_environment(AF3_APO_PDB)
    assert results == []
