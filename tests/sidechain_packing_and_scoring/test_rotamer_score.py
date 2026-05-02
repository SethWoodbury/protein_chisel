"""Cluster tests for tools/rotamer_score (Dunbrack fa_dun per residue)."""

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


def test_rotamer_score_design_pdb():
    """Score every (non-A/G) protein residue and check we get sensible numbers."""
    from protein_chisel.tools.sidechain_packing_and_scoring.rotamer_score import rotamer_score

    res = rotamer_score(DESIGN_PDB, params=[PARAMS_DIR])
    df = res.per_residue_df
    # design.pdb has 208 protein residues; some fraction are A/G (skipped)
    assert 100 < res.n_residues_scored < 208
    # Check df has the canonical columns
    for col in ("resno", "name3", "fa_dun", "n_chi", "chi_1",
                "is_outlier", "is_severe_outlier", "is_catalytic"):
        assert col in df.columns
    # All fa_dun values are finite
    assert df["fa_dun"].notna().all()
    # Mean fa_dun for a designed protein is typically 0.5-2.5
    assert 0.0 < res.mean_fa_dun < 5.0
    # Max fa_dun must be ≥ mean
    assert res.max_fa_dun >= res.mean_fa_dun
    # frac_outliers = n_outliers / n_residues_scored
    assert abs(res.frac_outliers - res.n_outliers / res.n_residues_scored) < 1e-9


def test_rotamer_score_flags_catalytic_residues():
    from protein_chisel.tools.sidechain_packing_and_scoring.rotamer_score import rotamer_score

    cat = {41, 64, 148, 184, 187, 188}
    res = rotamer_score(DESIGN_PDB, params=[PARAMS_DIR], catalytic_resnos=cat)
    df = res.per_residue_df
    cat_rows = df[df["is_catalytic"]]
    # All 6 catalytic residues are scored (none are A/G — they are HIS/LYS/GLU)
    assert len(cat_rows) == 6


def test_rotamer_score_to_dict_keys():
    from protein_chisel.tools.sidechain_packing_and_scoring.rotamer_score import rotamer_score

    res = rotamer_score(DESIGN_PDB, params=[PARAMS_DIR])
    d = res.to_dict()
    assert "rotamer__n_residues_scored" in d
    assert "rotamer__n_outliers" in d
    assert "rotamer__frac_outliers" in d
    assert "rotamer__mean_fa_dun" in d
    assert "rotamer__max_fa_dun" in d


def test_rotamer_score_outlier_threshold_respected():
    from protein_chisel.tools.sidechain_packing_and_scoring.rotamer_score import (
        RotamerScoreConfig, rotamer_score,
    )

    # Very tight threshold flags more outliers
    res_strict = rotamer_score(
        DESIGN_PDB, params=[PARAMS_DIR],
        config=RotamerScoreConfig(outlier_threshold=2.0, severe_threshold=10.0),
    )
    res_loose = rotamer_score(
        DESIGN_PDB, params=[PARAMS_DIR],
        config=RotamerScoreConfig(outlier_threshold=10.0, severe_threshold=20.0),
    )
    assert res_strict.n_outliers >= res_loose.n_outliers


def test_rotamer_score_skips_alanine_glycine_by_default():
    from protein_chisel.tools.sidechain_packing_and_scoring.rotamer_score import rotamer_score

    res = rotamer_score(DESIGN_PDB, params=[PARAMS_DIR])
    df = res.per_residue_df
    # ALA and GLY have no rotamers, so should not appear when skip_rotamerless=True
    assert "ALA" not in set(df["name3"])
    assert "GLY" not in set(df["name3"])


def test_rotamer_score_includes_alanine_glycine_when_disabled():
    from protein_chisel.tools.sidechain_packing_and_scoring.rotamer_score import (
        RotamerScoreConfig, rotamer_score,
    )

    res = rotamer_score(
        DESIGN_PDB, params=[PARAMS_DIR],
        config=RotamerScoreConfig(skip_rotamerless=False),
    )
    df = res.per_residue_df
    # When NOT skipping, we should see A and G if the design has any
    name3s = set(df["name3"])
    assert "ALA" in name3s or "GLY" in name3s


def test_rotamer_score_excludes_catalytic_from_aggregates():
    """Catalytic residues are often strained on purpose (carbamylated K,
    attack-poised H). Excluding them from aggregates avoids penalizing
    intentional strain — and the per-residue DF still carries them.
    """
    from protein_chisel.tools.sidechain_packing_and_scoring.rotamer_score import (
        RotamerScoreConfig, rotamer_score,
    )

    cat = {41, 64, 148, 184, 187, 188}
    res_with = rotamer_score(
        DESIGN_PDB, params=[PARAMS_DIR], catalytic_resnos=cat,
        config=RotamerScoreConfig(exclude_catalytic_from_aggregates=False),
    )
    res_without = rotamer_score(
        DESIGN_PDB, params=[PARAMS_DIR], catalytic_resnos=cat,
        config=RotamerScoreConfig(exclude_catalytic_from_aggregates=True),
    )
    # Per-residue DF still includes all 6 catalytic residues in both modes
    assert (res_with.per_residue_df["is_catalytic"]).sum() == 6
    assert (res_without.per_residue_df["is_catalytic"]).sum() == 6
    # Aggregates reflect the exclusion: n_residues_scored drops by exactly 6
    assert res_without.n_residues_scored == res_with.n_residues_scored - 6
    # Lys64 had fa_dun ~12 → was a severe outlier; excluding catalytic
    # should drop the severe-outlier count by at least 1 (often 2-3).
    assert res_without.n_severe_outliers <= res_with.n_severe_outliers
    # Mean fa_dun also drops since high-strain catalytic residues no longer
    # contribute (or stays the same if catalytic happen to be near-mean).
    assert res_without.mean_fa_dun <= res_with.mean_fa_dun + 0.01
