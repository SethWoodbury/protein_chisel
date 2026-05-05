#!/bin/bash
#SBATCH --job-name=PTE_i1_iter_v2
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
# Resource notes (validated 2026-05-04 GPU PLM sweep + 6-way matrix +
# L=275 empirical memory probe):
#   GPU runs (full pipeline, default scaffold L=202):
#     - 300m+35m PLMs:  3:51 wall,  5.0 GB MaxRSS
#     - 600m+650m PLMs: 7:06 wall,  9.5 GB MaxRSS
#     - 600m+1.3b PLMs: 3:40 wall,  8.0 GB MaxRSS  <-- production default
#   CPU runs (cpus=4, full pipeline, L=202): 14.3 GB MaxRSS for 1.3b.
#   L=275 stage-2 probe (CPU): 10.9 GB peak; full pipeline estimate
#   ~16-17 GB. cpus=4 is the sweet spot (super-linear from fpocket
#   Pool; hyperthread oversubscription kicks in past 4).
#   --mem=20G is whole-job total (NOT per-cpu); ~1.2-1.4x headroom
#   for typical L=200-280 scaffolds. Bump to 24G for L>300.
#   See docs/sweep_parameters.md for the full memory model.
#SBATCH --output=/net/scratch/woodbuse/slurm-iter-v2-%j.out
#SBATCH --error=/net/scratch/woodbuse/slurm-iter-v2-%j.err

# PLM-fusion-driven iterative design for the PTE_i1 scaffold.
#
# Three stages, three sifs:
#   1. classify_positions   -> pyrosetta.sif (CPU only, but already on a GPU
#                              node so we just use what we have)
#   2. precompute PLM logits + fusion  -> esmc.sif (GPU)
#   3. iterative driver (sample/filter/score/dedup/diverse)  -> universal.sif
#
# Outputs go under /net/scratch/woodbuse/iterative_design_v2_PTE_i1_<ts>/.
# Within each run: classify/, plm_artifacts/, cycle_00/, cycle_01/,
#   cycle_02/, final_topk/, manifest.json.

set -euo pipefail

REPO=/home/woodbuse/codebase_projects/protein_chisel
SEED_PDB="${SEED_PDB:-/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb}"
LIG_PARAMS="${LIG_PARAMS:-/home/woodbuse/testing_space/scaffold_optimization/ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params/YYE.params}"
TARGET_K="${TARGET_K:-50}"
MIN_HAMMING="${MIN_HAMMING:-3}"
N_CYCLES="${N_CYCLES:-3}"
# AAs MPNN should never sample. PTE_i1 has no catalytic Cys, so we
# exclude C as well as X (UNK) by default. Override per-scaffold.
OMIT_AA="${OMIT_AA:-CX}"

# LigandMPNN side-chain-context flag. 0 (default) = MPNN sees only
# backbone + ligand atoms. The clash failure mode (Y/F/W at clash-prone
# first-shell positions) is now prevented at sample time by auto-
# detected per-residue omits.
USE_SIDE_CHAIN_CONTEXT="${USE_SIDE_CHAIN_CONTEXT:-0}"

# Post-translational modifications declared for catalytic residues. The
# final --protonate_final stage records these in the output PDB's
# REMARK 668 block so downstream consumers (docking, MD setup) know the
# residue should carry the PTM even if the seed PDB stores it
# unmodified. PTE_i1 has Lys 157 as the carbamate-bearing metal-bridging
# lysine (KCX). For other scaffolds, override via env or pass empty.
PTM="${PTM:-A:157=KCX}"

# Optional pLDDT-enhanced LigandMPNN checkpoint (--enhance flag).
# Empty string = use the standard ligand_mpnn checkpoint. Available
# names listed in iterative_design_v2.py AVAILABLE_ENHANCE_CHECKPOINTS;
# enhance can boost mean fitness ~+0.02 nats/residue at a small
# diversity cost. Pass via env: ENHANCE=plddt_residpo_alpha_20250116-aec4d0c4
ENHANCE="${ENHANCE:-}"

# PLM model variants (only matter for stage 2 — precompute_plm_artifacts).
#   ESMC_MODEL  : esmc_300m (~46s GPU) | esmc_600m (default, ~90s GPU)
#   SAPROT_MODEL: saprot_35m (~10s GPU) | saprot_650m | saprot_650m_af2
#                 | saprot_1.3b (default, ~30s GPU, 6 GB VRAM)
# Defaults set by 6-way validation 2026-05-04: esmc_600m + saprot_1.3b
# delivered the best balance of fitness (-1.79 mean), sap_max (~0.85,
# 2x lower than 35m default), pocket volume (+30%), and pairwise
# diversity (+20%) at no wall-time cost vs the small-model default
# (3:40 GPU). See docs/sweep_parameters.md for the full PLM-combo
# matrix. Override per scaffold if memory is tight (esmc_300m +
# saprot_35m fits in ~5 GB).
# Stage 2 runs ONCE per scaffold so its cost is amortized across all
# subsequent design rounds.
ESMC_MODEL="${ESMC_MODEL:-esmc_600m}"
SAPROT_MODEL="${SAPROT_MODEL:-saprot_1.3b}"

# Working dir for this job (ms + PID for collision safety with concurrent jobs)
TS=$(python3 -c 'import datetime,os; t=datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]; print(f"{t}-pid{os.getpid()}")' 2>/dev/null || date +%Y%m%d-%H%M%S)
WORK_ROOT=/net/scratch/woodbuse
WORK_DIR="$WORK_ROOT/iterative_design_v2_PTE_i1_${TS}"
mkdir -p "$WORK_DIR"
echo "=== work dir: $WORK_DIR ==="

CLASSIFY_DIR="$WORK_DIR/classify"
PLM_DIR="$WORK_DIR/plm_artifacts"
mkdir -p "$CLASSIFY_DIR" "$PLM_DIR"

# Some pyrosetta.sif images export PYTHONNOUSERSITE — so PYTHONPATH alone
# lets us put protein_chisel on the import path. The default container
# PYTHONPATH for esmc.sif and universal.sif holds /cifutils/src etc. that
# we must preserve.

echo "=== STAGE 1: classify_positions (pyrosetta.sif) ==="
apptainer exec \
    --bind "$REPO:/code" \
    --bind /net/software \
    --bind /net/scratch \
    --bind /home/woodbuse \
    --env "PYTHONPATH=/code/src:/pyrosetta" \
    /net/software/containers/pyrosetta.sif \
    python "$REPO/scripts/classify_positions_pte_i1.py" \
        --seed_pdb "$SEED_PDB" \
        --ligand_params "$LIG_PARAMS" \
        --out_dir "$CLASSIFY_DIR"

echo "=== STAGE 2: PLM logits + fusion (esmc.sif) ==="
apptainer exec --nv \
    --bind "$REPO:/code" \
    --bind /net/software \
    --bind /net/databases \
    --bind /net/scratch \
    --bind /home/woodbuse \
    --bind /net/databases/huggingface/esmc \
    --bind /net/databases/huggingface/saprot \
    --env "PYTHONPATH=/code/src" \
    --env "HF_HOME=/net/databases/huggingface/esmc" \
    --env "HF_HUB_CACHE=/net/databases/huggingface/esmc/hub" \
    --env "SAPROT_HF_CACHE=/net/databases/huggingface/saprot/hub" \
    /net/software/containers/users/woodbuse/esmc.sif \
    python "$REPO/scripts/precompute_plm_artifacts.py" \
        --seed_pdb "$SEED_PDB" \
        --position_table "$CLASSIFY_DIR/positions.tsv" \
        --out_dir "$PLM_DIR" \
        --esmc_model "$ESMC_MODEL" \
        --saprot_model "$SAPROT_MODEL"

echo "=== STAGE 3: iterative driver (universal.sif) ==="
apptainer exec --nv \
    --bind "$REPO:/code" \
    --bind /net/software \
    --bind /net/databases \
    --bind /net/scratch \
    --bind /home/woodbuse \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    /net/software/containers/universal.sif \
    python "$REPO/scripts/iterative_design_v2.py" \
        --seed_pdb "$SEED_PDB" \
        --ligand_params "$LIG_PARAMS" \
        --plm_artifacts_dir "$PLM_DIR" \
        --position_table "$CLASSIFY_DIR/positions.tsv" \
        --out_root "$WORK_ROOT" \
        --target_k "$TARGET_K" \
        --min_hamming "$MIN_HAMMING" \
        --cycles "$N_CYCLES" \
        --omit_AA "$OMIT_AA" \
        --use_side_chain_context "$USE_SIDE_CHAIN_CONTEXT" \
        ${PTM:+--ptm "$PTM"} \
        ${ENHANCE:+--enhance "$ENHANCE"}

echo "=== DONE -- $WORK_DIR ==="
