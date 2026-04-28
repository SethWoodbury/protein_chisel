"""Orchestrate plm_fusion → ligand_mpnn for biased sequence sampling.

This is a thin convenience layer that:
1. Loads PLM logits (ESM-C + SaProt) for the seed sequence/structure.
2. Loads / classifies positions to derive position-class labels.
3. Builds the calibrated fusion bias.
4. Calls LigandMPNN with that bias and the active-site residues fixed.
5. Returns a CandidateSet with sampler metadata.

It does NOT recompute PLM logits inside the sampling loop (that would
need a refresh strategy — see architecture.md). Use the static-bias path
for v1; refresh / reranker / allowed-set variants are TODO.

This module *coordinates* multiple sifs:
- ESM-C / SaProt logits → esmc.sif (caller must run there or precompute)
- LigandMPNN sampling → mlfold.sif (handled internally via apptainer)
- classify_positions → pyrosetta.sif (caller must run there or precompute)

The simplest entry point ``biased_sample(...)`` accepts pre-computed
log-prob arrays and a PositionTable so the orchestration here is
container-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from protein_chisel.io.pdb import parse_remark_666
from protein_chisel.io.schemas import CandidateSet, PositionTable
from protein_chisel.sampling.plm_fusion import (
    FusionConfig, FusionResult, fuse_plm_logits,
)
from protein_chisel.tools.ligand_mpnn import (
    LigandMPNNConfig, LigandMPNNResult, sample_with_ligand_mpnn,
)


LOGGER = logging.getLogger("protein_chisel.biased_mpnn")


@dataclass
class BiasedSampleConfig:
    fusion: FusionConfig = field(default_factory=FusionConfig)
    ligand_mpnn: LigandMPNNConfig = field(default_factory=LigandMPNNConfig)
    n_samples: int = 100
    fix_active_site: bool = True
    fix_first_shell: bool = False
    chain: str = "A"


@dataclass
class BiasedSampleResult:
    candidate_set: CandidateSet
    fusion: FusionResult
    ligand_mpnn: LigandMPNNResult


def biased_sample(
    pdb_path: str | Path,
    ligand_params: str | Path,
    position_table: PositionTable,
    log_probs_esmc: np.ndarray,
    log_probs_saprot: np.ndarray,
    out_dir: str | Path,
    config: Optional[BiasedSampleConfig] = None,
) -> BiasedSampleResult:
    """Run biased LigandMPNN sampling with a fused PLM bias.

    Args:
        pdb_path: input PDB.
        ligand_params: ligand .params file.
        position_table: per-residue PositionTable for the *protein* chain.
            Rows must be sorted by resno; only protein rows are used for
            class labels and length checking.
        log_probs_esmc: (L, 20) ESM-C masked-LM log-probs over the 20 AAs.
        log_probs_saprot: (L, 20) SaProt log-probs (3Di-marginalized).
        out_dir: where LigandMPNN writes its outputs.
        config: BiasedSampleConfig (fusion + LigandMPNN configs + sampler knobs).
    """
    cfg = config or BiasedSampleConfig()

    protein_rows = (
        position_table.df[position_table.df["is_protein"]].sort_values("resno")
    )
    L = len(protein_rows)

    if log_probs_esmc.shape[0] != L:
        raise ValueError(
            f"ESM-C log-probs have {log_probs_esmc.shape[0]} positions "
            f"but PositionTable has {L} protein residues"
        )
    if log_probs_saprot.shape[0] != L:
        raise ValueError(
            f"SaProt log-probs have {log_probs_saprot.shape[0]} positions "
            f"but PositionTable has {L} protein residues"
        )

    pos_classes = protein_rows["class"].tolist()

    fusion = fuse_plm_logits(
        log_probs_esmc=log_probs_esmc,
        log_probs_saprot=log_probs_saprot,
        position_classes=pos_classes,
        config=cfg.fusion,
    )

    fixed_resnos: set[int] = set()
    if cfg.fix_active_site:
        fixed_resnos.update(
            int(r) for r in protein_rows.loc[protein_rows["class"] == "active_site", "resno"]
        )
    if cfg.fix_first_shell:
        fixed_resnos.update(
            int(r) for r in protein_rows.loc[protein_rows["class"] == "first_shell", "resno"]
        )
    # Force REMARK 666 catalytic residues into fixed set even if their
    # class somehow drifted.
    catres = parse_remark_666(pdb_path)
    fixed_resnos.update(catres.keys())

    LOGGER.info(
        "biased sample: L=%d, fixed=%d, mean_abs_bias=%.4f",
        L, len(fixed_resnos), float(np.abs(fusion.bias).mean()),
    )

    lmpnn = sample_with_ligand_mpnn(
        pdb_path=pdb_path,
        ligand_params=ligand_params,
        chain=cfg.chain,
        fixed_resnos=sorted(fixed_resnos),
        bias_per_residue=fusion.bias,
        n_samples=cfg.n_samples,
        config=cfg.ligand_mpnn,
        out_dir=out_dir,
        parent_design_id=Path(pdb_path).stem,
    )

    # Annotate the candidate set with where the bias came from.
    cs_df = lmpnn.candidate_set.df.copy()
    cs_df["plm_fusion_mean_abs_bias"] = float(np.abs(fusion.bias).mean())
    cs_df["n_fixed_positions"] = len(fixed_resnos)
    cs_df["fusion_class_weights"] = "|".join(
        f"{c}={w:.2f}" for c, w in cfg.fusion.class_weights.items()
    )
    return BiasedSampleResult(
        candidate_set=CandidateSet(df=cs_df),
        fusion=fusion,
        ligand_mpnn=lmpnn,
    )


__all__ = ["BiasedSampleConfig", "BiasedSampleResult", "biased_sample"]
