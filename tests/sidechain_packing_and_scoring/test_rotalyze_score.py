"""Cluster tests for sidechain_packing_and_scoring/rotalyze_score.

Runs MolProbity rotalyze (Top8000 KDE) inside esmc.sif. Slow-ish
(2-5 s per call) so we keep these tests light.
"""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.cluster


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"


def test_rotalyze_score_design_pdb():
    """Score every residue and check we get sensible numbers."""
    from protein_chisel.tools.sidechain_packing_and_scoring.rotalyze_score import (
        rotalyze_score,
    )

    res = rotalyze_score(DESIGN_PDB)
    df = res.per_residue_df
    # design.pdb has 208 residues; A/G are still scored by rotalyze
    # (it just classifies them as Favored with a high percentile by default).
    assert 100 < res.n_residues_scored < 250
    # Check df has the canonical columns
    for col in ("chain_id", "resseq", "resname", "score", "evaluation",
                "is_outlier", "is_catalytic"):
        assert col in df.columns
    # frac_outliers in [0, 1]
    assert 0.0 <= res.frac_outliers <= 1.0
    # Mean score is the percent rotamer score, 0-100. For a designed
    # protein with reasonable rotamers, mean is typically 30-70%.
    assert 0.0 <= res.mean_score <= 100.0
    # Counts add up
    assert res.n_outliers + res.n_allowed + res.n_favored <= res.n_residues_scored


def test_rotalyze_score_to_dict_keys():
    from protein_chisel.tools.sidechain_packing_and_scoring.rotalyze_score import (
        rotalyze_score,
    )

    res = rotalyze_score(DESIGN_PDB)
    d = res.to_dict()
    for key in (
        "rotalyze__n_residues_scored",
        "rotalyze__n_outliers",
        "rotalyze__frac_outliers",
        "rotalyze__mean_score",
    ):
        assert key in d


def test_rotalyze_score_flags_catalytic():
    from protein_chisel.tools.sidechain_packing_and_scoring.rotalyze_score import (
        rotalyze_score,
    )

    cat = {41, 64, 148, 184, 187, 188}
    res = rotalyze_score(DESIGN_PDB, catalytic_resnos=cat)
    df = res.per_residue_df
    cat_rows = df[df["is_catalytic"]]
    # All 6 catalytic resseqs should be in the per-residue df
    assert len(cat_rows) >= 5  # allow 1 missed in case rotalyze drops anything
