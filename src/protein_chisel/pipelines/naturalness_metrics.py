"""Naturalness scoring pipeline (PLM-based).

Runs in esmc.sif. Computes ESM-C and SaProt pseudo-perplexities for each
pose, plus the calibrated PLM fusion bias matrix (saved as a numpy file
per pose).

Complements `comprehensive_metrics` (which is structural). They can be
run in parallel and merged via MetricTable.merge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from protein_chisel.io.pdb import extract_sequence
from protein_chisel.io.schemas import (
    Manifest,
    MetricTable,
    PoseSet,
    manifest_matches,
)


LOGGER = logging.getLogger("protein_chisel.naturalness_metrics")


@dataclass
class NaturalnessConfig:
    esmc_model: str = "esmc_300m"
    saprot_model: str = "saprot_35m"
    device: str = "auto"
    score_pseudo_perplexity: bool = True
    save_logits: bool = True
    save_fusion_bias: bool = True
    fusion_classes: Optional[list[str]] = None


@dataclass
class NaturalnessResult:
    metric_table: MetricTable
    out_dir: Path
    per_pose_outputs: dict[str, dict[str, Path]] = field(default_factory=dict)


def run_naturalness_metrics(
    pose_set: PoseSet,
    out_dir: str | Path,
    config: Optional[NaturalnessConfig] = None,
    position_table_dir: Optional[str | Path] = None,
    skip_existing: bool = True,
) -> NaturalnessResult:
    """Run ESM-C + SaProt scoring (and optional fusion) over a PoseSet.

    Args:
        pose_set: PoseSet of structures.
        out_dir: directory where per-pose outputs (logits, fusion bias)
            are written.
        config: NaturalnessConfig.
        position_table_dir: optional directory containing per-pose
            PositionTable parquets named ``<sequence_id>/conf<i>/positions.tsv``
            (or .parquet). If provided AND save_fusion_bias is True, the
            fusion uses those classes; otherwise fusion is skipped.
    """
    cfg = config or NaturalnessConfig()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    per_pose_outputs: dict[str, dict[str, Path]] = {}

    for entry in pose_set:
        per_dir = out_dir / "per_pose" / entry.sequence_id / f"conf{entry.conformer_index}"
        per_dir.mkdir(parents=True, exist_ok=True)
        manifest = _manifest_for_pose(entry, cfg)
        manifest_path = per_dir / "_manifest.json"
        row_path = per_dir / "metrics.tsv"

        if skip_existing and manifest_matches(manifest, manifest_path) and row_path.exists():
            LOGGER.info("skip %s", entry.sequence_id)
            row = pd.read_csv(row_path, sep="\t").iloc[0].to_dict()
            rows.append(row)
            per_pose_outputs[entry.sequence_id] = _list_outputs(per_dir)
            continue

        LOGGER.info("naturalness on %s", entry.path)
        row = _run_one_pose(entry, cfg, per_dir, position_table_dir)
        rows.append(row)
        manifest.to_json(manifest_path)
        pd.DataFrame([row]).to_csv(row_path, sep="\t", index=False)
        per_pose_outputs[entry.sequence_id] = _list_outputs(per_dir)

    df = pd.DataFrame(rows)
    if "sequence_id" not in df.columns:
        df["sequence_id"] = [e.sequence_id for e in pose_set]
    if "conformer_index" not in df.columns:
        df["conformer_index"] = [e.conformer_index for e in pose_set]
    metric_table = MetricTable(df=df)
    metric_table.to_parquet(out_dir / "metrics.parquet")
    return NaturalnessResult(metric_table=metric_table, out_dir=out_dir, per_pose_outputs=per_pose_outputs)


def _manifest_for_pose(entry, cfg: NaturalnessConfig) -> Manifest:
    return Manifest.for_stage(
        stage="naturalness_metrics",
        input_paths=[entry.path],
        config={
            "esmc_model": cfg.esmc_model,
            "saprot_model": cfg.saprot_model,
            "device": cfg.device,
            "score_pseudo_perplexity": cfg.score_pseudo_perplexity,
            "save_logits": cfg.save_logits,
            "save_fusion_bias": cfg.save_fusion_bias,
        },
        tool_versions={"protein_chisel": "0.0.1"},
    )


def _list_outputs(per_dir: Path) -> dict[str, Path]:
    return {
        "metrics_tsv": per_dir / "metrics.tsv",
        "manifest": per_dir / "_manifest.json",
        "esmc_log_probs": per_dir / "esmc_log_probs.npy",
        "saprot_log_probs": per_dir / "saprot_log_probs.npy",
        "fusion_bias": per_dir / "fusion_bias.npy",
    }


def _run_one_pose(
    entry,
    cfg: NaturalnessConfig,
    per_dir: Path,
    position_table_dir: Optional[Path],
) -> dict:
    pdb = entry.path
    row: dict = {
        "sequence_id": entry.sequence_id,
        "conformer_index": entry.conformer_index,
        "fold_source": entry.fold_source,
        "pdb_path": str(pdb),
    }

    seq = extract_sequence(pdb)
    if not seq:
        LOGGER.warning("no sequence extracted from %s", pdb)
        return row

    # ESM-C
    from protein_chisel.tools.esmc import esmc_logits, esmc_score

    esmc_lp = esmc_logits(seq, model_name=cfg.esmc_model, device=cfg.device)
    if cfg.save_logits:
        np.save(per_dir / "esmc_log_probs.npy", esmc_lp.log_probs)
    if cfg.score_pseudo_perplexity:
        esmc_s = esmc_score(seq, model_name=cfg.esmc_model, device=cfg.device)
        row.update(esmc_s.to_dict())

    # SaProt
    from protein_chisel.tools.saprot import saprot_logits, saprot_score

    saprot_lp = saprot_logits(pdb, model_name=cfg.saprot_model, device=cfg.device)
    if cfg.save_logits:
        np.save(per_dir / "saprot_log_probs.npy", saprot_lp.log_probs)
    if cfg.score_pseudo_perplexity:
        saprot_s = saprot_score(pdb, model_name=cfg.saprot_model, device=cfg.device)
        row.update(saprot_s.to_dict())

    # PLM fusion (requires position classes)
    if cfg.save_fusion_bias and position_table_dir:
        pt_path = _find_position_table(position_table_dir, entry.sequence_id, entry.conformer_index)
        if pt_path and pt_path.exists():
            from protein_chisel.io.schemas import PositionTable
            from protein_chisel.sampling.plm_fusion import fuse_plm_logits

            pt = PositionTable.from_parquet(pt_path)
            protein_rows = pt.df[pt.df["is_protein"]].sort_values("resno")
            if len(protein_rows) == esmc_lp.log_probs.shape[0]:
                pos_classes = protein_rows["class"].tolist()
                fusion = fuse_plm_logits(
                    esmc_lp.log_probs, saprot_lp.log_probs, pos_classes,
                )
                np.save(per_dir / "fusion_bias.npy", fusion.bias)
                row["fusion__mean_abs_bias"] = float(np.abs(fusion.bias).mean())
                row["fusion__max_abs_bias"] = float(np.abs(fusion.bias).max())

    return row


def _find_position_table(base: Path, sequence_id: str, conformer_index: int) -> Optional[Path]:
    base = Path(base)
    for ext in (".parquet", ".tsv"):
        p = base / "per_pose" / sequence_id / f"conf{conformer_index}" / f"positions{ext}"
        if p.exists():
            return p
    return None


__all__ = [
    "NaturalnessConfig",
    "NaturalnessResult",
    "run_naturalness_metrics",
]
