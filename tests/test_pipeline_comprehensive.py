"""End-to-end test of the comprehensive_metrics pipeline.

Runs in pyrosetta.sif (which has biopython too — protparam works).
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


def test_comprehensive_metrics_single_pdb(tmp_path: Path):
    from protein_chisel.io.schemas import PoseSet
    from protein_chisel.pipelines.comprehensive_metrics import (
        ComprehensiveMetricsConfig, run_comprehensive_metrics,
    )

    ps = PoseSet.from_single_pdb(DESIGN_PDB, sequence_id="design", fold_source="designed")
    cfg = ComprehensiveMetricsConfig(ligand_target_atoms=("C1", "O1", "O2"))

    result = run_comprehensive_metrics(
        ps, out_dir=tmp_path, params=[PARAMS_DIR], config=cfg, skip_existing=False,
    )

    df = result.metric_table.df
    assert len(df) == 1

    # Required ID columns
    assert "sequence_id" in df.columns
    assert "conformer_index" in df.columns

    # Position table summary
    assert int(df["positions__n_residues"].iloc[0]) == 208
    assert int(df["positions__n_active_site"].iloc[0]) == 6  # REMARK 666

    # Backbone sanity
    assert df["backbone__chainbreak_above_4_5"].iloc[0] == 0
    assert df["backbone__n_residues"].iloc[0] == 208

    # Shape
    assert 10.0 < float(df["shape__rg"].iloc[0]) < 25.0

    # SS summary fractions sum to 1
    h = float(df["ss__helix_frac"].iloc[0])
    e = float(df["ss__sheet_frac"].iloc[0])
    l = float(df["ss__loop_frac"].iloc[0])
    assert abs(h + e + l - 1.0) < 0.01

    # Ligand environment
    assert "ligand__lig_dist" in df.columns
    assert df["ligand__name3"].iloc[0] == "YYE"

    # Chemical interactions
    assert int(df["interact__n_hbonds"].iloc[0]) > 0

    # BUNS (whitelist applied since REMARK 666 catalytic residues present)
    assert "buns__n_buried_unsat" in df.columns

    # Catres quality
    assert int(df["catres__n_residues"].iloc[0]) == 6
    assert int(df["catres__n_broken_sidechains"].iloc[0]) == 0

    # ProtParam
    assert "protparam__pi" in df.columns
    assert "protparam__charge_at_pH7_no_HIS" in df.columns

    # PositionTable saved per pose
    pos_table = result.per_pose_outputs["design"]["position_table"]
    assert pos_table.exists() or pos_table.with_suffix(".tsv").exists()

    # Manifest exists
    assert (tmp_path / "per_pose" / "design" / "conf0" / "_manifest.json").exists()

    # MetricTable parquet/tsv on disk
    metrics_path = tmp_path / "metrics.parquet"
    if not metrics_path.exists():
        metrics_path = tmp_path / "metrics.tsv"
    assert metrics_path.exists()


def test_comprehensive_metrics_multi_pose(tmp_path: Path):
    """Run on design + AF3 apo + refined; aggregations should be exposed.

    Each pose gets its own metric row; the overall MetricTable has one row
    per pose. Aggregation across the rows is the caller's job (scoring/aggregate).
    """
    from protein_chisel.io.schemas import PoseEntry, PoseSet
    from protein_chisel.pipelines.comprehensive_metrics import (
        ComprehensiveMetricsConfig, run_comprehensive_metrics,
    )

    ps = PoseSet([
        PoseEntry(path=str(DESIGN_PDB), sequence_id="ub", fold_source="designed", conformer_index=0),
        PoseEntry(path=str(AF3_APO_PDB), sequence_id="ub", fold_source="AF3_seed1", conformer_index=1, is_apo=True),
        PoseEntry(path=str(REFINED_PDB), sequence_id="ub", fold_source="AF3_refined", conformer_index=2),
    ], name="design_plus_af3")

    cfg = ComprehensiveMetricsConfig(
        ligand_target_atoms=("C1", "O1", "O2"),
        run_protparam=True,
    )
    result = run_comprehensive_metrics(
        ps, out_dir=tmp_path, params=[PARAMS_DIR], config=cfg, skip_existing=False,
    )
    df = result.metric_table.df
    assert len(df) == 3

    # All three rows share sequence_id ub
    assert (df["sequence_id"] == "ub").all()
    # conformer_index spans 0..2
    assert sorted(df["conformer_index"].tolist()) == [0, 1, 2]
    # The apo conformer's ligand fields are NaN-or-missing.
    apo = df[df["fold_source"] == "AF3_seed1"]
    # ligand__lig_dist should NOT exist for apo OR be NaN; check either.
    if "ligand__lig_dist" in df.columns:
        # Apo row must not have a finite ligand distance
        import math
        v = apo["ligand__lig_dist"].iloc[0]
        assert v is None or (isinstance(v, float) and math.isnan(v)) or pd_isnan(v)
    # Designed pose's interaction count > 0
    designed = df[df["fold_source"] == "designed"]
    assert int(designed["interact__n_hbonds"].iloc[0]) > 0


def pd_isnan(v):
    import pandas as pd
    return pd.isna(v)


def test_comprehensive_metrics_restart(tmp_path: Path):
    """Re-running with skip_existing=True reuses prior outputs."""
    from protein_chisel.io.schemas import PoseSet
    from protein_chisel.pipelines.comprehensive_metrics import run_comprehensive_metrics

    ps = PoseSet.from_single_pdb(DESIGN_PDB, sequence_id="design", fold_source="designed")
    # First run
    r1 = run_comprehensive_metrics(ps, out_dir=tmp_path, params=[PARAMS_DIR])
    n1 = len(r1.metric_table.df)
    # Second run with skip
    r2 = run_comprehensive_metrics(ps, out_dir=tmp_path, params=[PARAMS_DIR], skip_existing=True)
    assert len(r2.metric_table.df) == n1
