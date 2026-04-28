"""Sequence-design pipeline v1.

Composition:
    classify_positions  → PositionTable
    esmc_logits + saprot_logits  → fused PLM bias matrix
    LigandMPNN sampling with bias + active-site fixed  → CandidateSet
    Hard sequence filters (regex, ProtParam range)  → filtered CandidateSet
    Diversity selection on mutable positions  → top-N library

Multi-sif by design — each stage runs in whichever sif owns its native
deps. The pipeline coordinates via on-disk artifacts; you can run
each stage by hand if you'd rather.

Stage manifest:

    out_dir/
      0_classify/positions.tsv             # pyrosetta.sif
      0_classify/_manifest.json
      1_logits/esmc_log_probs.npy          # esmc.sif
      1_logits/saprot_log_probs.npy
      1_logits/_manifest.json
      2_fusion/bias.npy                    # any sif with numpy
      2_fusion/_manifest.json
      3_sample/candidates.fasta            # mlfold.sif
      3_sample/candidates.tsv
      3_sample/_manifest.json
      4_filter/candidates.fasta            # any sif w/ Bio
      4_filter/candidates.tsv
      4_filter/_manifest.json
      5_diversity/library.fasta            # host
      5_diversity/library.tsv
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from protein_chisel.io.pdb import parse_remark_666
from protein_chisel.io.schemas import (
    CandidateSet, Manifest, PositionTable,
)
from protein_chisel.sampling.plm_fusion import FusionConfig, fuse_plm_logits


LOGGER = logging.getLogger("protein_chisel.sequence_design_v1")


@dataclass
class HardFilters:
    pi_min: float = 4.0
    pi_max: float = 9.0
    instability_max: float = 60.0
    charge_at_pH7_no_HIS_min: float = -10.0
    charge_at_pH7_no_HIS_max: float = 10.0
    forbid_protease_sites: bool = True
    require_start_M: bool = True


@dataclass
class SequenceDesignV1Config:
    n_samples: int = 100
    sampling_temp: float = 0.1
    fusion: FusionConfig = field(default_factory=FusionConfig)
    fix_active_site: bool = True
    fix_first_shell: bool = False
    chain: str = "A"
    esmc_model: str = "esmc_300m"
    saprot_model: str = "saprot_35m"
    device: str = "auto"
    hard_filters: HardFilters = field(default_factory=HardFilters)
    target_n_diverse: int = 50
    diversity_min_distance: int = 2  # over mutable positions only


@dataclass
class SequenceDesignV1Result:
    out_dir: Path
    raw_candidates: CandidateSet
    filtered_candidates: CandidateSet
    diverse_candidates: CandidateSet
    position_table: PositionTable


# ---------------------------------------------------------------------------
# Stage 0: classify_positions  (in pyrosetta.sif)
# ---------------------------------------------------------------------------


def stage_classify(
    pdb: Path, params: list[Path], out_dir: Path, sequence_id: str = "design",
) -> PositionTable:
    from protein_chisel.tools.classify_positions import classify_positions

    pt_path = out_dir / "positions.tsv"
    if pt_path.exists():
        return PositionTable.from_parquet(pt_path)
    pt = classify_positions(pdb, pose_id=sequence_id, params=params)
    pt.to_parquet(pt_path)
    LOGGER.info("classify: wrote %d rows to %s", len(pt.df), pt_path)
    return pt


# ---------------------------------------------------------------------------
# Stage 1: PLM logits  (in esmc.sif)
# ---------------------------------------------------------------------------


def stage_plm_logits(
    pdb: Path, sequence: str, out_dir: Path, cfg: SequenceDesignV1Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute masked-LM ESM-C + SaProt log-probs."""
    from protein_chisel.tools.esmc import esmc_logits
    from protein_chisel.tools.saprot import saprot_logits

    esmc_path = out_dir / "esmc_log_probs.npy"
    saprot_path = out_dir / "saprot_log_probs.npy"

    if esmc_path.exists():
        esmc_lp = np.load(esmc_path)
    else:
        LOGGER.info("ESM-C masked logits (L forward passes)")
        esmc_lp = esmc_logits(
            sequence, model_name=cfg.esmc_model, device=cfg.device, masked=True,
        ).log_probs
        np.save(esmc_path, esmc_lp)

    if saprot_path.exists():
        saprot_lp = np.load(saprot_path)
    else:
        LOGGER.info("SaProt masked logits")
        saprot_lp = saprot_logits(
            pdb, chain=cfg.chain, model_name=cfg.saprot_model,
            device=cfg.device, masked=True,
        ).log_probs
        np.save(saprot_path, saprot_lp)

    return esmc_lp, saprot_lp


# ---------------------------------------------------------------------------
# Stage 2: fusion  (any sif with numpy)
# ---------------------------------------------------------------------------


def stage_fusion(
    esmc_lp: np.ndarray,
    saprot_lp: np.ndarray,
    position_table: PositionTable,
    out_dir: Path,
    cfg: SequenceDesignV1Config,
) -> np.ndarray:
    bias_path = out_dir / "bias.npy"
    if bias_path.exists():
        return np.load(bias_path)

    protein_rows = (
        position_table.df[position_table.df["is_protein"]].sort_values("resno")
    )
    pos_classes = protein_rows["class"].tolist()
    fusion = fuse_plm_logits(esmc_lp, saprot_lp, pos_classes, config=cfg.fusion)
    np.save(bias_path, fusion.bias)
    LOGGER.info("fusion: bias (L=%d, mean_abs=%.4f)", len(pos_classes), np.abs(fusion.bias).mean())
    return fusion.bias


# ---------------------------------------------------------------------------
# Stage 3: LigandMPNN sampling  (in mlfold.sif)
# ---------------------------------------------------------------------------


def stage_sample(
    pdb: Path,
    ligand_params: Path,
    bias: np.ndarray,
    fixed_resnos: list[int],
    protein_resnos: list[int],
    out_dir: Path,
    cfg: SequenceDesignV1Config,
) -> CandidateSet:
    from protein_chisel.tools.ligand_mpnn import (
        LigandMPNNConfig, sample_with_ligand_mpnn,
    )

    fasta_path = out_dir / "candidates.fasta"
    meta_path = out_dir / "candidates.tsv"
    if fasta_path.exists() and meta_path.exists():
        return CandidateSet.from_disk(meta_path)

    lmpnn_cfg = LigandMPNNConfig(temperature=cfg.sampling_temp)
    res = sample_with_ligand_mpnn(
        pdb_path=pdb,
        ligand_params=ligand_params,
        chain=cfg.chain,
        fixed_resnos=fixed_resnos,
        bias_per_residue=bias,
        protein_resnos=protein_resnos,
        n_samples=cfg.n_samples,
        config=lmpnn_cfg,
        out_dir=out_dir / "_lmpnn",
    )
    res.candidate_set.to_disk(fasta_path, meta_path)
    LOGGER.info("sample: %d candidates from LigandMPNN", len(res.candidate_set.df))
    return res.candidate_set


# ---------------------------------------------------------------------------
# Stage 4: hard filters  (any sif with Bio)
# ---------------------------------------------------------------------------


def stage_filter(
    candidates: CandidateSet,
    out_dir: Path,
    cfg: SequenceDesignV1Config,
) -> CandidateSet:
    from protein_chisel.filters.protparam import protparam_metrics
    from protein_chisel.filters.protease_sites import find_protease_sites

    rows: list[dict] = []
    for _, row in candidates.df.iterrows():
        seq = row["sequence"]
        # Sequence-level metrics
        try:
            pp = protparam_metrics(seq)
        except Exception as e:
            LOGGER.warning("protparam fail: %s", e)
            continue
        if not (cfg.hard_filters.pi_min <= pp.pi <= cfg.hard_filters.pi_max):
            continue
        if pp.instability_index > cfg.hard_filters.instability_max:
            continue
        if not (
            cfg.hard_filters.charge_at_pH7_no_HIS_min
            <= pp.charge_at_pH7_no_HIS
            <= cfg.hard_filters.charge_at_pH7_no_HIS_max
        ):
            continue
        if cfg.hard_filters.require_start_M and (not seq or seq[0] != "M"):
            continue
        if cfg.hard_filters.forbid_protease_sites:
            sites = find_protease_sites(seq)
            if sites.has_any():
                continue
        out_row = dict(row)
        out_row.update({
            "protparam__pi": pp.pi,
            "protparam__charge_at_pH7_no_HIS": pp.charge_at_pH7_no_HIS,
            "protparam__instability_index": pp.instability_index,
            "protparam__gravy": pp.gravy,
        })
        rows.append(out_row)
    df = pd.DataFrame(rows) if rows else candidates.df.iloc[0:0].copy()
    cs = CandidateSet(df=df)
    cs.to_disk(out_dir / "candidates.fasta", out_dir / "candidates.tsv")
    LOGGER.info(
        "filter: kept %d / %d candidates", len(df), len(candidates.df),
    )
    return cs


# ---------------------------------------------------------------------------
# Stage 5: diversity selection  (host)
# ---------------------------------------------------------------------------


def stage_diversity(
    candidates: CandidateSet,
    position_table: PositionTable,
    out_dir: Path,
    cfg: SequenceDesignV1Config,
) -> CandidateSet:
    from protein_chisel.scoring.diversity import (
        mask_from_position_table, select_diverse,
    )

    if len(candidates.df) == 0:
        empty = candidates.df.copy()
        cs = CandidateSet(df=empty)
        cs.to_disk(out_dir / "library.fasta", out_dir / "library.tsv")
        return cs

    mask = mask_from_position_table(position_table.df)
    score_col = "protparam__pi" if "protparam__pi" in candidates.df.columns else None

    selected = select_diverse(
        candidates.df,
        sequence_col="sequence",
        mask=mask,
        k=cfg.target_n_diverse,
        min_distance=cfg.diversity_min_distance,
        score_col=score_col,
    )
    cs = CandidateSet(df=selected)
    cs.to_disk(out_dir / "library.fasta", out_dir / "library.tsv")
    LOGGER.info("diversity: %d / %d selected", len(selected), len(candidates.df))
    return cs


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_sequence_design_v1(
    pdb_path: str | Path,
    ligand_params: str | Path,
    out_dir: str | Path,
    sequence_id: str = "design",
    extra_pyrosetta_params: list[str | Path] = (),
    config: Optional[SequenceDesignV1Config] = None,
) -> SequenceDesignV1Result:
    """Run all five stages end-to-end. Each stage is restartable via
    output-existence; outputs accumulate per stage in ``out_dir``.
    """
    cfg = config or SequenceDesignV1Config()
    pdb = Path(pdb_path).resolve()
    ligand = Path(ligand_params).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    s0 = out / "0_classify"; s0.mkdir(exist_ok=True)
    s1 = out / "1_logits";   s1.mkdir(exist_ok=True)
    s2 = out / "2_fusion";   s2.mkdir(exist_ok=True)
    s3 = out / "3_sample";   s3.mkdir(exist_ok=True)
    s4 = out / "4_filter";   s4.mkdir(exist_ok=True)
    s5 = out / "5_diversity"; s5.mkdir(exist_ok=True)

    # 0. Classify positions
    pt = stage_classify(
        pdb, list(extra_pyrosetta_params) + [ligand.parent], s0, sequence_id=sequence_id,
    )

    # Sequence is needed for ESM-C
    from protein_chisel.io.pdb import extract_sequence
    seq = extract_sequence(pdb, chain=cfg.chain)
    if not seq:
        raise RuntimeError(f"no protein sequence on chain {cfg.chain} in {pdb}")

    # 1. PLM logits
    esmc_lp, saprot_lp = stage_plm_logits(pdb, seq, s1, cfg)

    # 2. Fusion
    bias = stage_fusion(esmc_lp, saprot_lp, pt, s2, cfg)

    # 3. Sample
    catres = parse_remark_666(pdb)
    fixed_resnos = list(catres.keys())
    protein_rows = pt.df[pt.df["is_protein"]].sort_values("resno")
    protein_resnos = protein_rows["resno"].astype(int).tolist()
    raw = stage_sample(
        pdb, ligand, bias, fixed_resnos, protein_resnos, s3, cfg,
    )

    # 4. Filter
    filtered = stage_filter(raw, s4, cfg)

    # 5. Diversity
    diverse = stage_diversity(filtered, pt, s5, cfg)

    return SequenceDesignV1Result(
        out_dir=out,
        raw_candidates=raw,
        filtered_candidates=filtered,
        diverse_candidates=diverse,
        position_table=pt,
    )


__all__ = [
    "HardFilters",
    "SequenceDesignV1Config",
    "SequenceDesignV1Result",
    "run_sequence_design_v1",
    "stage_classify",
    "stage_diversity",
    "stage_filter",
    "stage_fusion",
    "stage_plm_logits",
    "stage_sample",
]
