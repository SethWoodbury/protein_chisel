"""Comprehensive structural metrics pipeline.

Modernizes ~/special_scripts/design_filtering/metric_monster__MAIN.py.
Takes a PoseSet (or single PDB) and runs the full battery of descriptive
structural tools, emitting:

- one PositionTable parquet per pose,
- one MetricTable parquet aggregating all per-pose metric rows.

This pipeline lives in pyrosetta.sif (PyRosetta-bound). PLM-based scoring
and contact_ms (which need esmc.sif) are separate pipelines.

The pipeline is restartable: each pose's outputs land in
``out_dir/per_pose/<sequence_id>/`` and we skip poses whose manifest
matches an existing one.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from protein_chisel.io.pdb import parse_remark_666
from protein_chisel.io.schemas import (
    Manifest,
    MetricTable,
    PoseEntry,
    PoseSet,
    PositionTable,
    manifest_matches,
)


LOGGER = logging.getLogger("protein_chisel.comprehensive_metrics")


@dataclass
class ComprehensiveMetricsConfig:
    """Knobs for which tools to run and their parameters."""

    run_position_table: bool = True
    run_backbone_sanity: bool = True
    run_shape_metrics: bool = True
    run_ss_summary: bool = True
    run_ligand_environment: bool = True
    run_chemical_interactions: bool = True
    run_buns: bool = True
    run_catres_quality: bool = True
    run_protparam: bool = True
    run_protease_sites: bool = True

    # Per-tool params
    salt_bridge_cutoff: float = 4.0
    pi_pi_cutoff: float = 6.0
    pi_cation_cutoff: float = 6.0
    buns_sasa_cutoff: float = 1.0

    # Ligand atoms to compute per-atom SASA on; user-supplied per design
    ligand_target_atoms: tuple[str, ...] = ()


@dataclass
class ComprehensiveMetricsResult:
    metric_table: MetricTable
    out_dir: Path
    per_pose_outputs: dict[str, dict[str, Path]] = field(default_factory=dict)


def run_comprehensive_metrics(
    pose_set: PoseSet,
    out_dir: str | Path,
    params: list[str | Path] = (),
    config: Optional[ComprehensiveMetricsConfig] = None,
    skip_existing: bool = True,
) -> ComprehensiveMetricsResult:
    """Run the structural metric battery over a PoseSet."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or ComprehensiveMetricsConfig()
    rows: list[dict[str, Any]] = []
    per_pose_outputs: dict[str, dict[str, Path]] = {}

    for entry in pose_set:
        per_dir = out_dir / "per_pose" / entry.sequence_id / f"conf{entry.conformer_index}"
        per_dir.mkdir(parents=True, exist_ok=True)

        manifest = _manifest_for_pose(entry, params, cfg)
        manifest_path = per_dir / "_manifest.json"
        row_path = per_dir / "metrics.tsv"
        # Unique key per pose so multi-conformer outputs don't overwrite.
        out_key = f"{entry.sequence_id}__conf{entry.conformer_index}"

        if skip_existing and manifest_matches(manifest, manifest_path) and row_path.exists():
            LOGGER.info("skipping %s (manifest matches)", out_key)
            row = pd.read_csv(row_path, sep="\t").iloc[0].to_dict()
            rows.append(row)
            per_pose_outputs[out_key] = _list_outputs(per_dir)
            continue

        LOGGER.info("processing %s", entry.path)
        row = _run_one_pose(entry, params, cfg, per_dir)
        rows.append(row)
        manifest.to_json(manifest_path)
        pd.DataFrame([row]).to_csv(row_path, sep="\t", index=False)
        per_pose_outputs[out_key] = _list_outputs(per_dir)

    df = pd.DataFrame(rows)
    # MetricTable requires (sequence_id, conformer_index) ID columns; ensure they're present.
    if "sequence_id" not in df.columns:
        df["sequence_id"] = [e.sequence_id for e in pose_set]
    if "conformer_index" not in df.columns:
        df["conformer_index"] = [e.conformer_index for e in pose_set]
    metric_table = MetricTable(df=df)
    metric_table.to_parquet(out_dir / "metrics.parquet")

    return ComprehensiveMetricsResult(
        metric_table=metric_table,
        out_dir=out_dir,
        per_pose_outputs=per_pose_outputs,
    )


def _manifest_for_pose(
    entry: PoseEntry, params: list[str | Path], cfg: ComprehensiveMetricsConfig
) -> Manifest:
    # Capture ALL config fields (booleans + numeric thresholds + tuples) so
    # changes to e.g. salt_bridge_cutoff trigger a re-run.
    serializable_cfg: dict[str, Any] = {}
    for k, v in cfg.__dict__.items():
        if isinstance(v, (bool, int, float, str, type(None))):
            serializable_cfg[k] = v
        elif isinstance(v, (list, tuple)):
            serializable_cfg[k] = list(v)
        else:
            serializable_cfg[k] = repr(v)

    from protein_chisel import __version__ as chisel_version

    return Manifest.for_stage(
        stage="comprehensive_metrics",
        input_paths=[entry.path],
        config={
            "tool_config": serializable_cfg,
            "params": [str(p) for p in params],
            "metadata": {
                "sequence_id": entry.sequence_id,
                "conformer_index": entry.conformer_index,
                "fold_source": entry.fold_source,
            },
        },
        tool_versions={"protein_chisel": chisel_version},
    )


def _list_outputs(per_dir: Path) -> dict[str, Path]:
    return {
        "metrics_tsv": per_dir / "metrics.tsv",
        "manifest": per_dir / "_manifest.json",
        "position_table": per_dir / "positions.tsv",
    }


def _run_one_pose(
    entry: PoseEntry,
    params: list[str | Path],
    cfg: ComprehensiveMetricsConfig,
    per_dir: Path,
) -> dict[str, Any]:
    """Run all enabled tools on one pose; return a metrics row."""
    pdb = entry.path
    pose_id = f"{entry.sequence_id}__conf{entry.conformer_index}"
    row: dict[str, Any] = {
        "sequence_id": entry.sequence_id,
        "conformer_index": entry.conformer_index,
        "fold_source": entry.fold_source,
        "is_apo": entry.is_apo,
        "pdb_path": str(pdb),
        "pose_id": pose_id,
    }

    catres = parse_remark_666(pdb)
    catalytic_resnos = set(catres.keys())

    # Position classification + position table
    if cfg.run_position_table:
        from protein_chisel.tools.classify_positions import classify_positions

        pt = classify_positions(pdb, pose_id=pose_id, catres=catres, params=params)
        pt.to_parquet(per_dir / "positions.tsv")
        row["positions__n_residues"] = int((pt.df["is_protein"]).sum())
        # Count BOTH new (directional 6-class) and legacy (5-class) names
        # so the column set stays stable across the rewrite. The new
        # classifier emits new names in `class` and legacy names in
        # `class_legacy`; legacy PositionTables only have `class`.
        cls_col = pt.df["class"]
        legacy_col = pt.df.get("class_legacy", cls_col)
        # Legacy buckets (back-compat).
        row["positions__n_active_site"] = int((legacy_col == "active_site").sum())
        row["positions__n_first_shell"] = int((legacy_col == "first_shell").sum())
        row["positions__n_buried"] = int((legacy_col == "buried").sum())
        row["positions__n_surface"] = int((legacy_col == "surface").sum())
        # Directional buckets (new).
        row["positions__n_primary_sphere"]   = int((cls_col == "primary_sphere").sum())
        row["positions__n_secondary_sphere"] = int((cls_col == "secondary_sphere").sum())
        row["positions__n_nearby_surface"]   = int((cls_col == "nearby_surface").sum())
        row["positions__n_distal_buried"]    = int((cls_col == "distal_buried").sum())
        row["positions__n_distal_surface"]   = int((cls_col == "distal_surface").sum())

    # Backbone sanity
    if cfg.run_backbone_sanity:
        from protein_chisel.tools.backbone_sanity import backbone_sanity

        res = backbone_sanity(pdb, params=params)
        row.update(res.to_dict())

    # Shape metrics
    if cfg.run_shape_metrics:
        from protein_chisel.tools.shape_metrics import shape_metrics

        res = shape_metrics(pdb, params=params)
        row.update(res.to_dict())

    # SS summary
    if cfg.run_ss_summary:
        from protein_chisel.tools.secondary_structure import ss_summary

        res = ss_summary(pdb, params=params, catalytic_resnos=catalytic_resnos)
        row.update(res.to_dict())

    # Ligand environment (skips silently for apo)
    if cfg.run_ligand_environment:
        from protein_chisel.tools.ligand_environment import ligand_environment

        results = ligand_environment(
            pdb, params=params, target_atoms=cfg.ligand_target_atoms,
        )
        if results:
            # First ligand under the canonical "ligand__" prefix; subsequent
            # ligands (rare; multi-ligand designs) under ligand_<i>__ prefixes.
            row.update(results[0].to_dict())
            for i, extra in enumerate(results[1:], start=1):
                row.update(extra.to_dict(prefix=f"ligand_{i}__"))
            row["ligand__n_ligands"] = len(results)

    # Chemical interactions
    if cfg.run_chemical_interactions:
        from protein_chisel.tools.chemical_interactions import chemical_interactions

        res = chemical_interactions(
            pdb,
            params=params,
            salt_bridge_cutoff=cfg.salt_bridge_cutoff,
            pi_pi_centroid_cutoff=cfg.pi_pi_cutoff,
            pi_cation_cutoff=cfg.pi_cation_cutoff,
        )
        row.update(res.summary())

    # BUNS
    if cfg.run_buns:
        from protein_chisel.tools.buns import buns, whitelist_from_remark_666

        wl = whitelist_from_remark_666(pdb) if catres else None
        res = buns(pdb, params=params, sasa_buried_cutoff=cfg.buns_sasa_cutoff, whitelist=wl)
        row.update(res.to_dict())

    # Catres quality (only if catalytic residues present)
    if cfg.run_catres_quality and catalytic_resnos:
        from protein_chisel.tools.catres_quality import catres_quality

        res = catres_quality(pdb, params=params)
        row.update(res.to_dict())

    # ProtParam (sequence-only)
    if cfg.run_protparam:
        from protein_chisel.filters.protparam import protparam_metrics
        from protein_chisel.io.pdb import extract_sequence

        seq = extract_sequence(pdb)
        if seq:
            try:
                pp = protparam_metrics(seq)
                row.update(pp.to_dict())
            except Exception as e:  # biopython may not be present in all sifs
                LOGGER.warning("protparam failed for %s: %s", entry.path, e)

    # Protease sites
    if cfg.run_protease_sites:
        from protein_chisel.filters.protease_sites import find_protease_sites
        from protein_chisel.io.pdb import extract_sequence

        seq = extract_sequence(pdb)
        if seq:
            res = find_protease_sites(seq)
            row.update(res.to_dict())

    return row


__all__ = [
    "ComprehensiveMetricsConfig",
    "ComprehensiveMetricsResult",
    "run_comprehensive_metrics",
]
