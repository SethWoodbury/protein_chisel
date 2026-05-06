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
#SBATCH --output=/net/scratch/%u/slurm-iter-v2-%j.out
#SBATCH --error=/net/scratch/%u/slurm-iter-v2-%j.err

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

# === Portability ====================================================
# REPO is auto-detected from this script's location, so any user can
# `git clone` into their own home and the sbatch just works. Override
# REPO=... only if you're calling this script from a different checkout.
# Other site-specific paths (apptainer images, model weights at
# /net/software, /net/databases) are CLUSTER-SHARED resources that any
# user with read access can hit; nothing here writes into another
# user's home.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="${REPO:-$(dirname "$SCRIPT_DIR")}"

# === Required inputs (per-design caller MUST set these) ============
# SEED_PDB: starting backbone (with the desired catalytic motif and
#   ligand placement; REMARK 666 lines drive auto-derivation of the
#   catalytic residues so you don't hardcode resnos here)
# LIG_PARAMS: Rosetta params for the ligand. We auto-detect a sensible
#   default for the PTE_i1 substrate (YYE), but for any other ligand
#   the caller MUST set LIG_PARAMS=...
SEED_PDB="${SEED_PDB:?Set SEED_PDB to your input PDB (with REMARK 666)}"
LIG_PARAMS="${LIG_PARAMS:?Set LIG_PARAMS to your ligand .params file}"
TARGET_K="${TARGET_K:-50}"
MIN_HAMMING="${MIN_HAMMING:-3}"
N_CYCLES="${N_CYCLES:-3}"
# AAs MPNN should never sample. Default 'X' = exclude UnknownX only.
# Scaffolds with no catalytic Cys (e.g. PTE_i1) typically set 'CX' to
# avoid spurious disulfides. Set per-scaffold.
OMIT_AA="${OMIT_AA:-X}"

# LigandMPNN side-chain-context flag. 0 (default) = MPNN sees only
# backbone + ligand atoms. The clash failure mode (Y/F/W at clash-prone
# first-shell positions) is now prevented at sample time by auto-
# detected per-residue omits.
USE_SIDE_CHAIN_CONTEXT="${USE_SIDE_CHAIN_CONTEXT:-0}"

# Post-translational modifications declared for catalytic residues.
# Records the modification in the output PDB's REMARK 668 block so
# downstream consumers (docking, MD setup) know the residue should
# carry the PTM even if the seed PDB stores it unmodified.
#
# ANNOTATION ONLY — residues are kept as their unmodified form for
# Rosetta / sequence / protonation. PTM is metadata.
#
# Default: empty (no PTM annotation). Set per-scaffold using motif-
# index form 'CHAIN/EXPECTED_RESNAME/MOTIF_INDEX:CODE'. Example for
# PTE_i1 (catalytic Lys at REMARK 666 motif index 3 is carbamylated):
#   PTM='A/LYS/3:KCX'
# Multiple: 'A/LYS/3:KCX,A/HIS/1:MSE'
PTM="${PTM:-}"

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

# === Output base ====================================================
# WORK_ROOT is the top-level directory under which work_dir/ and run_dir/
# get created. Defaults to /net/scratch/$USER. Override per-call (e.g.
# from a sweep notebook) to put each input PDB's outputs in its own
# isolated subdirectory:
#   WORK_ROOT=/net/scratch/aruder2/.../chisel_out/i1__ref_pdbs/<pdb_stem>
#   bash run_iterative_design_v2.sbatch
WORK_ROOT="${WORK_ROOT:-/net/scratch/$USER}"
mkdir -p "$WORK_ROOT"

# Working dir for this job (ms + PID for collision safety with concurrent jobs)
TS=$(python3 -c 'import datetime,os; t=datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]; print(f"{t}-pid{os.getpid()}")' 2>/dev/null || date +%Y%m%d-%H%M%S)
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
    --bind "$HOME" \
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
    --bind "$HOME" \
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

# Stage 3 sif selection: prefer the user-built derivative that adds
# pyKVFinder + RDKit on top of universal.sif (used by --tunnel_metrics).
# Canonical location is /net/software/containers/users/woodbuse/ —
# world-readable on this cluster. Override via TUNNEL_SIF=... env var
# if you have a custom build. Falls back to system universal.sif if
# the tools sif isn't available (means pyKVFinder skipped — homegrown
# ray-cast still runs).
SIF_CANON=/net/software/containers/users/woodbuse/universal_with_tunnel_tools.sif
SIF_SCRATCH=/net/scratch/woodbuse/sif/universal_with_tunnel_tools.sif
if [[ -n "${TUNNEL_SIF:-}" && -f "$TUNNEL_SIF" ]]; then
    STAGE3_SIF="$TUNNEL_SIF"
elif [[ -f "$SIF_CANON" ]]; then
    STAGE3_SIF="$SIF_CANON"
elif [[ -f "$SIF_SCRATCH" ]]; then
    STAGE3_SIF="$SIF_SCRATCH"   # legacy NFS-propagation workaround
else
    STAGE3_SIF=/net/software/containers/universal.sif
    echo "WARN: tunnel-tools sif not found at canonical or fallback paths;"
    echo "      using system universal.sif (--tunnel_metrics will skip pyKVFinder)"
fi
echo "=== STAGE 3: iterative driver ($STAGE3_SIF) ==="
apptainer exec --nv \
    --bind "$REPO:/code" \
    --bind /net/software \
    --bind /net/databases \
    --bind /net/scratch \
    --bind "$HOME" \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    "$STAGE3_SIF" \
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
        --run_dir_marker "$WORK_DIR/run_dir.txt" \
        ${PTM:+--ptm "$PTM"} \
        ${ENHANCE:+--enhance "$ENHANCE"} \
        ${EXTRA_DRIVER_FLAGS:-}

# Stage 3 wrote run_dir's path into $WORK_DIR/run_dir.txt as soon as
# it knew the timestamped run_dir name. Read it for stage 4. This is
# more robust than scanning timestamps (avoids races with concurrent
# jobs that share $WORK_ROOT).
RUN_DIR=""
if [[ -f "$WORK_DIR/run_dir.txt" ]]; then
    RUN_DIR=$(< "$WORK_DIR/run_dir.txt")
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR/final_topk/topk_pdbs" ]]; then
    echo "WARNING: could not locate stage 3 run dir from $WORK_DIR/run_dir.txt;"
    echo "         skipping stage 4 (protonate). Run protonate_final_topk.py"
    echo "         manually if needed."
else
    echo "=== STAGE 4: protonate final top-K + shipping layout (pyrosetta.sif) ==="
    echo "    run_dir = $RUN_DIR"
    # SAVE_INTERMEDIATES=1 to keep cycle_NN/ subtrees for diagnostics.
    # Default (unset): clean shipping layout — run_dir/designs/*.pdb +
    # designs.tsv + designs.fasta + cycle_metrics.tsv + manifest.json.
    SHIPPING_FLAGS="--shipping_layout"
    if [[ "${SAVE_INTERMEDIATES:-}" == "1" ]]; then
        SHIPPING_FLAGS="--shipping_layout --no_strip_intermediates"
    fi
    if [[ "${MINIMAL:-}" == "1" ]]; then
        # Minimal mode: just designs/ + designs.tsv (with embedded
        # RUN_META JSON header). Drops manifest.json and all aux files.
        # /net/scratch space-saver for production sweeps.
        SHIPPING_FLAGS="$SHIPPING_FLAGS --minimal_layout"
    fi
    apptainer exec \
        --bind "$REPO:/code" \
        --bind /net/software \
        --bind /net/scratch \
        --bind "$HOME" \
        --env "PYTHONPATH=/code/src:/pyrosetta" \
        /net/software/containers/pyrosetta.sif \
        python "$REPO/scripts/protonate_final_topk.py" \
            --topk_dir "$RUN_DIR/final_topk/topk_pdbs" \
            --seed_pdb "$SEED_PDB" \
            --ligand_params "$LIG_PARAMS" \
            --out_dir "$RUN_DIR/final_topk/topk_pdbs_protonated" \
            --summary_json "$RUN_DIR/protonation_summary.json" \
            $SHIPPING_FLAGS \
            ${PTM:+--ptm "$PTM"}
fi

# === Per-PDB-sweep consolidation ====================================
# When the user pre-set WORK_ROOT to a PER-PDB output dir AND requested
# MINIMAL mode, flatten the output one more level: move RUN_DIR's
# contents up into WORK_ROOT and remove the (now empty) RUN_DIR +
# WORK_DIR, so the final layout is exactly:
#     WORK_ROOT/<pdb>_chisel_NNN.pdb         (N PDBs, flat)
#     WORK_ROOT/chiseled_design_metrics.tsv  (1 TSV with embedded RUN_META)
#
# Triggered when CONSOLIDATE_TO_WORK_ROOT=1 (default ON when MINIMAL=1
# AND WORK_ROOT was explicitly set by the caller — i.e. you're running
# a sweep). Safe-guards:
#   - skipped if WORK_ROOT was the default /net/scratch/$USER (since
#     then collisions between concurrent runs are a real risk)
#   - skipped if RUN_DIR doesn't exist (pipeline failed earlier)
#   - skipped if the merge would clobber existing files
USER_DEFAULT_WORK_ROOT="/net/scratch/$USER"
if [[ "${MINIMAL:-}" == "1" \
      && "${CONSOLIDATE_TO_WORK_ROOT:-1}" == "1" \
      && "$WORK_ROOT" != "$USER_DEFAULT_WORK_ROOT" \
      && -d "${RUN_DIR:-/nonexistent}" ]]; then
    echo "=== CONSOLIDATE: $RUN_DIR/* -> $WORK_ROOT/ ==="
    # Use a temp-then-rename to avoid mid-move racing
    shopt -s dotglob nullglob
    moved_any=0
    for f in "$RUN_DIR"/*; do
        bn=$(basename "$f")
        if [[ -e "$WORK_ROOT/$bn" ]]; then
            echo "WARN: refusing to overwrite $WORK_ROOT/$bn — skipping"
            continue
        fi
        mv "$f" "$WORK_ROOT/$bn"
        moved_any=1
    done
    shopt -u dotglob nullglob
    # Clean up empty timestamped dirs
    rmdir "$RUN_DIR" 2>/dev/null || true
    rm -rf "$WORK_DIR" 2>/dev/null || true
    if (( moved_any )); then
        echo "=== DONE -- consolidated to $WORK_ROOT ==="
    else
        echo "WARN: consolidation moved no files; check $RUN_DIR / $WORK_DIR"
    fi
else
    echo "=== DONE -- $WORK_DIR ==="
fi
