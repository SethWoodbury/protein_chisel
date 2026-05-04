"""Quick MPNN-ablation test: 4 conditions x 100 designs each.

Tests whether (a) ligand_mpnn_use_side_chain_context=1 over-constrains
the design toward WT identities at first-shell positions, and (b) the
``plddt_residpo_alpha`` enhanced checkpoint improves fitness/diversity.

Conditions:
    A: use_side_chain_context=1, no enhance (CURRENT default)
    B: use_side_chain_context=0, no enhance
    C: use_side_chain_context=1, enhance=plddt_residpo_alpha
    D: use_side_chain_context=0, enhance=plddt_residpo_alpha

Reports per-condition:
    - First-shell diversity (mean unique AAs / position)
    - Pairwise Hamming (min/mean)
    - Mean fitness logp_fused
    - K/R surface counts
    - WT-identity recovery rate at non-fixed positions

Run inside universal.sif. Reuses cached PLM artifacts from a previous
production run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("mpnn_ablation_test")


SEED = "/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb"
PT = "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260503-175033/classify/positions.tsv"
PLM_BIAS = "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260503-175033/plm_artifacts/fusion_bias.npy"
PLM_E = "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260503-175033/plm_artifacts/esmc_log_probs.npy"
PLM_S = "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260503-175033/plm_artifacts/saprot_log_probs.npy"
WEIGHTS = "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260503-175033/plm_artifacts/fusion_weights.npy"
LMPNN_CKPT = "/net/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt"
SC_CKPT = "/net/databases/mpnn/packer_weights/s_300756.pt"
ENHANCE_CKPT_NAME = "plddt_residpo_alpha_20250116-aec4d0c4"

CATALYTIC_RESNOS = [60, 64, 128, 131, 132, 157]
CHAIN = "A"

CONDITIONS = [
    ("A_sc1_noEnhance",   {"use_side_chain_context": 1, "enhance": None}),
    ("B_sc0_noEnhance",   {"use_side_chain_context": 0, "enhance": None}),
    ("C_sc1_plddtAlpha",  {"use_side_chain_context": 1, "enhance": ENHANCE_CKPT_NAME}),
    ("D_sc0_plddtAlpha",  {"use_side_chain_context": 0, "enhance": ENHANCE_CKPT_NAME}),
]


def run_one_condition(
    name: str,
    cfg: dict,
    *,
    seed_pdb: Path,
    bias: np.ndarray,
    protein_resnos: list[int],
    fixed_resnos: list[int],
    out_root: Path,
    n_samples: int = 100,
) -> Path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.ligand_mpnn import (
        LigandMPNNConfig, sample_with_ligand_mpnn,
    )
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("=== running %s ===", name)
    lcfg = LigandMPNNConfig(
        temperature=0.20,
        batch_size=10,
        repack_everything=0,
        pack_side_chains=1,
        ligand_mpnn_use_side_chain_context=cfg["use_side_chain_context"],
        omit_AA="CX",
        enhance=cfg["enhance"],
        extra_flags=(
            "--checkpoint_ligand_mpnn", LMPNN_CKPT,
            "--checkpoint_path_sc", SC_CKPT,
        ),
    )
    res = sample_with_ligand_mpnn(
        pdb_path=seed_pdb, chain=CHAIN,
        fixed_resnos=fixed_resnos,
        bias_per_residue=bias,
        protein_resnos=protein_resnos,
        n_samples=n_samples,
        config=lcfg,
        out_dir=out_dir,
        parent_design_id=f"PTE_i1_ablation_{name}",
        via_apptainer=False,
    )
    cand_tsv = out_dir / "candidates.tsv"
    res.candidate_set.to_disk(out_dir / "candidates.fasta", cand_tsv)
    return cand_tsv


def analyze(name: str, cand_tsv: Path, *, wt_seq: str, fs_resnos: list[int],
             fixed_set: set[int], lp_e: np.ndarray, lp_s: np.ndarray,
             weights: np.ndarray) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.sampling.fitness_score import (
        deduplicate_by_sequence, score_dataframe_fitness,
    )
    df = pd.read_csv(cand_tsv, sep="\t")
    if "is_input" in df.columns:
        df = df[~df["is_input"].astype(bool)].copy()
    df = deduplicate_by_sequence(df)
    df = score_dataframe_fitness(df, lp_e, lp_s, weights)

    seqs = df["sequence"].tolist()
    arr = np.array([list(s) for s in seqs])
    L = arr.shape[1]

    # First-shell diversity
    fs_idx = [r - 1 for r in fs_resnos]
    fs_unique = [len(set(arr[:, i])) for i in fs_idx]
    # WT-identity recovery rate at NON-FIXED, NON-FIRST-SHELL positions
    nonfix = [i for i in range(L) if (i + 1) not in fixed_set]
    wt_match = []
    for i in nonfix:
        wt_match.append((arr[:, i] == wt_seq[i]).sum() / len(seqs))
    # Pairwise Hamming over a 30-design subset (full pairwise is N^2)
    sub = arr[:30]
    hams = []
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            hams.append((sub[i] != sub[j]).sum())
    return {
        "name": name,
        "n": int(len(df)),
        "n_unique": int(df["seq_hash"].nunique()),
        "fs_mean_unique_aas_per_pos": float(np.mean(fs_unique)),
        "fs_max_unique_aas_per_pos": int(np.max(fs_unique)),
        "wt_recovery_rate_nonfixed_mean": float(np.mean(wt_match)),
        "fitness_logp_fused_mean": float(df["fitness__logp_fused_mean"].mean()),
        "fitness_logp_fused_max": float(df["fitness__logp_fused_mean"].max()),
        "pairwise_hamming_min": int(min(hams)) if hams else 0,
        "pairwise_hamming_mean": float(np.mean(hams)) if hams else 0,
        "k_count_mean": float(np.mean([s.count("K") for s in seqs])),
        "r_count_mean": float(np.mean([s.count("R") for s in seqs])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_root", type=Path,
                   default=Path("/net/scratch/woodbuse/mpnn_ablation_v1"))
    p.add_argument("--n_samples", type=int, default=100)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args.out_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("out root: %s", args.out_root)

    pt_df = pd.read_csv(PT, sep="\t")
    prot = pt_df[pt_df["is_protein"]].sort_values("resno").reset_index(drop=True)
    protein_resnos = prot["resno"].astype(int).tolist()
    fs_resnos = prot[prot["class"] == "first_shell"]["resno"].astype(int).tolist()
    fixed_set = set(CATALYTIC_RESNOS)

    bias = np.load(PLM_BIAS)
    lp_e = np.load(PLM_E)
    lp_s = np.load(PLM_S)
    weights = np.load(WEIGHTS)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.io.pdb import extract_sequence
    wt_seq = extract_sequence(SEED, chain=CHAIN)

    rows = []
    for name, cfg in CONDITIONS:
        cand_tsv = run_one_condition(
            name, cfg,
            seed_pdb=Path(SEED), bias=bias,
            protein_resnos=protein_resnos,
            fixed_resnos=CATALYTIC_RESNOS,
            out_root=args.out_root,
            n_samples=args.n_samples,
        )
        stats = analyze(
            name, cand_tsv, wt_seq=wt_seq,
            fs_resnos=fs_resnos, fixed_set=fixed_set,
            lp_e=lp_e, lp_s=lp_s, weights=weights,
        )
        rows.append(stats)
        LOGGER.info("%s -> %s", name, stats)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_root / "summary.tsv", sep="\t", index=False)
    print()
    print("=== SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
