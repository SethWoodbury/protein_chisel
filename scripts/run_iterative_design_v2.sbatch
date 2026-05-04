#!/bin/bash
#SBATCH --job-name=PTE_i1_iter_v2
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=01:00:00
# Resource notes (validated 2026-05-04 sweep E):
#   - cpus=2 mem=8G: ~7.5 min wall, 3.5 GB peak (44% of 8G)
#   - cpus=4 mem=10G: better with fpocket parallelization (Pool(4) cuts
#     fpocket time ~4x; was ~41% of total wall before)
#   - cpus=8 mem=12G: optimal if you want max parallelization, more
#     priority cost. Most users want cpus=4.
# Original was cpus=4 mem=16G --time=08:00:00 — overprovisioned 2x on
# memory and 8x on time. Production runs land in 7-10 min.
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

# Optional pLDDT-enhanced LigandMPNN checkpoint (--enhance flag).
# Empty string = use the standard ligand_mpnn checkpoint. Available
# names listed in iterative_design_v2.py AVAILABLE_ENHANCE_CHECKPOINTS;
# enhance can boost mean fitness ~+0.02 nats/residue at a small
# diversity cost. Pass via env: ENHANCE=plddt_residpo_alpha_20250116-aec4d0c4
ENHANCE="${ENHANCE:-}"

# PLM model variants (only matter for stage 2 — precompute_plm_artifacts).
# Defaults are the smallest+fastest in each family.
#   ESMC_MODEL  : esmc_300m (default, ~46s) | esmc_600m
#   SAPROT_MODEL: saprot_35m (default, ~10s) | saprot_650m | saprot_1.3b
# Larger models give marginally better masked-LM probs but cost more
# VRAM and time. Stage 2 runs ONCE per scaffold so the time cost is
# amortized across all subsequent design runs.
ESMC_MODEL="${ESMC_MODEL:-esmc_300m}"
SAPROT_MODEL="${SAPROT_MODEL:-saprot_35m}"

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
        ${ENHANCE:+--enhance "$ENHANCE"}

echo "=== DONE -- $WORK_DIR ==="
