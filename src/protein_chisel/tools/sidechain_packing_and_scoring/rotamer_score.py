"""Per-residue rotamer scoring via Rosetta's Dunbrack 2010 library.

Rosetta's ``fa_dun`` score type is ``-log P(rotamer | φ, ψ)`` — the
backbone-dependent rotamer probability from the Dunbrack 2010 library.
Lower is better:

    fa_dun ≲ 1     well-modeled rotamer (canonical, dominant for that φ/ψ)
    fa_dun ≈ 2-3   acceptable, common
    fa_dun > 5     outlier — rotamer is ~e^-5 ≈ 0.7% of its φ/ψ-bin mass
    fa_dun > 8     severe outlier; treat as broken sidechain

This is roughly equivalent to MolProbity's rotamer "favored / allowed /
outlier" classification but exposed as a continuous energy.

The tool scores a pose with the canonical bcov fix_scorefxn pattern
(``decompose_bb_hb_into_pair_energies`` so per-residue energies sum
correctly), reads ``fa_dun`` per residue, classifies outliers, and
emits per-residue rows + aggregate metrics.

Run inside pyrosetta.sif.

Reference:
    Shapovalov, M.V. & Dunbrack, R.L. (2011). A smoothed
    backbone-dependent rotamer library for proteins derived from
    adaptive kernel density estimates and regressions. Structure 19:844.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("protein_chisel.rotamer_score")


# Default thresholds. Adjustable via RotamerScoreConfig.
DEFAULT_OUTLIER_THRESHOLD = 5.0   # fa_dun above this → outlier
DEFAULT_SEVERE_THRESHOLD = 8.0    # fa_dun above this → severe outlier


# Residues that have no rotamers (no chi angles) — fa_dun is 0/undefined.
_ROTAMERLESS = {"ALA", "GLY"}


@dataclass
class RotamerScoreConfig:
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD
    severe_threshold: float = DEFAULT_SEVERE_THRESHOLD
    skip_rotamerless: bool = True       # True: don't count A/G against any aggregate
    exclude_catalytic_from_aggregates: bool = False
    # ^^ When True, residues passed via `catalytic_resnos` are still scored
    # and present in the per-residue df but NOT counted in n_outliers /
    # frac_outliers / mean_fa_dun. Catalytic residues are often in strained
    # rotamers by design (carbamylated lysines, attack-poised histidines, etc.).
    # Same pattern as buns whitelist — avoids penalizing intentional strain.


@dataclass
class RotamerScoreResult:
    """Per-residue rotamer scoring + aggregate stats.

    `per_residue_df` rows: resno, name3, name1, fa_dun, n_chi,
    chi_1..chi_4 (NaN where absent), is_outlier, is_severe_outlier.
    """

    per_residue_df: pd.DataFrame
    n_residues_scored: int = 0
    n_outliers: int = 0
    n_severe_outliers: int = 0
    frac_outliers: float = 0.0
    mean_fa_dun: float = 0.0
    median_fa_dun: float = 0.0
    max_fa_dun: float = 0.0
    sum_fa_dun: float = 0.0
    config: Optional[RotamerScoreConfig] = None
    outlier_resnos: list[int] = field(default_factory=list)

    def to_dict(self, prefix: str = "rotamer__") -> dict[str, float | int]:
        return {
            f"{prefix}n_residues_scored": self.n_residues_scored,
            f"{prefix}n_outliers": self.n_outliers,
            f"{prefix}n_severe_outliers": self.n_severe_outliers,
            f"{prefix}frac_outliers": self.frac_outliers,
            f"{prefix}mean_fa_dun": self.mean_fa_dun,
            f"{prefix}median_fa_dun": self.median_fa_dun,
            f"{prefix}max_fa_dun": self.max_fa_dun,
            f"{prefix}sum_fa_dun": self.sum_fa_dun,
        }


def rotamer_score(
    pdb_path: str | Path,
    params: list[str | Path] = (),
    config: Optional[RotamerScoreConfig] = None,
    catalytic_resnos: Optional[set[int]] = None,
) -> RotamerScoreResult:
    """Score every protein residue's rotamer probability via Dunbrack.

    Args:
        pdb_path: input PDB.
        params: ligand .params files for PyRosetta init.
        config: thresholds + skip-rotamerless option.
        catalytic_resnos: optional set of seqposes that are catalytic;
            adds an ``is_catalytic`` column in the per-residue df. Does
            NOT change the scoring (fa_dun applies regardless).

    Returns RotamerScoreResult with the per-residue dataframe and
    aggregates. `outlier_resnos` lists every residue above the outlier
    threshold so callers can flag them quickly.
    """
    cfg = config or RotamerScoreConfig()

    from protein_chisel.utils.pose import (
        get_default_scorefxn, init_pyrosetta, pose_from_file,
    )
    import pyrosetta.rosetta as ros

    init_pyrosetta(params=list(params))
    pose = pose_from_file(pdb_path)
    sfxn = get_default_scorefxn()
    sfxn(pose)

    fa_dun_term = ros.core.scoring.score_type_from_name("fa_dun")

    rows: list[dict] = []
    catalytic_set = set(catalytic_resnos or ())
    for r in pose.residues:
        if not r.is_protein():
            continue
        sp = int(r.seqpos())
        n3 = r.name3()
        if cfg.skip_rotamerless and n3 in _ROTAMERLESS:
            continue
        try:
            fa_dun = float(
                pose.energies().residue_total_energies(sp).get(fa_dun_term)
            )
        except Exception as e:
            LOGGER.warning("fa_dun read failed for residue %d (%s): %s", sp, n3, e)
            continue
        n_chi = int(r.nchi())
        chis = [float(r.chi(i)) for i in range(1, n_chi + 1)]
        chi_pad = chis + [float("nan")] * (4 - len(chis))
        rows.append({
            "resno": sp,
            "name3": n3,
            "name1": r.name1(),
            "fa_dun": fa_dun,
            "n_chi": n_chi,
            "chi_1": chi_pad[0],
            "chi_2": chi_pad[1],
            "chi_3": chi_pad[2],
            "chi_4": chi_pad[3],
            "is_outlier": fa_dun > cfg.outlier_threshold,
            "is_severe_outlier": fa_dun > cfg.severe_threshold,
            "is_catalytic": sp in catalytic_set,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return RotamerScoreResult(
            per_residue_df=df, config=cfg,
        )

    # Aggregate over the rows minus the catalytic exclusion (when enabled).
    if cfg.exclude_catalytic_from_aggregates and catalytic_set:
        agg_df = df[~df["is_catalytic"]]
    else:
        agg_df = df

    fa = agg_df["fa_dun"].to_numpy(dtype=float)
    outliers = agg_df[agg_df["is_outlier"]]
    severe = agg_df[agg_df["is_severe_outlier"]]

    return RotamerScoreResult(
        per_residue_df=df,  # full per-residue DF (including catalytic)
        n_residues_scored=int(len(agg_df)),
        n_outliers=int(len(outliers)),
        n_severe_outliers=int(len(severe)),
        frac_outliers=float(len(outliers) / len(agg_df)) if len(agg_df) else 0.0,
        mean_fa_dun=float(np.mean(fa)) if len(fa) else 0.0,
        median_fa_dun=float(np.median(fa)) if len(fa) else 0.0,
        max_fa_dun=float(np.max(fa)) if len(fa) else 0.0,
        sum_fa_dun=float(np.sum(fa)) if len(fa) else 0.0,
        config=cfg,
        outlier_resnos=outliers["resno"].astype(int).tolist(),
    )


__all__ = [
    "DEFAULT_OUTLIER_THRESHOLD",
    "DEFAULT_SEVERE_THRESHOLD",
    "RotamerScoreConfig",
    "RotamerScoreResult",
    "rotamer_score",
]
