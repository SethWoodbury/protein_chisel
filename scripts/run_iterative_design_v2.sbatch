#!/bin/bash
#SBATCH --job-name=PTE_i1_iter_v2
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
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

# LigandMPNN side-chain-context flag. 1 (default) = MPNN sees catalytic
# sidechain rotamers (clash-safe). 0 trades safety for first-shell
# diversity but on PTE_i1 caused 90%+ design rejection due to severe
# clashes (designed Y@35 vs catalytic E@131).
USE_SIDE_CHAIN_CONTEXT="${USE_SIDE_CHAIN_CONTEXT:-1}"

# Optional pLDDT-enhanced fused_mpnn checkpoint name (without .pth);
# empty string = disabled. Adds ~+0.02 nats/residue mean fitness.
ENHANCE="${ENHANCE:-}"

# Working dir for this job
TS=$(date +%Y%m%d-%H%M%S)
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
    /net/software/containers/users/woodbuse/esmc.sif \
    python "$REPO/scripts/precompute_plm_artifacts.py" \
        --seed_pdb "$SEED_PDB" \
        --position_table "$CLASSIFY_DIR/positions.tsv" \
        --out_dir "$PLM_DIR"

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
        ${ENHANCE:+--enhance "$ENHANCE"}

echo "=== DONE -- $WORK_DIR ==="
