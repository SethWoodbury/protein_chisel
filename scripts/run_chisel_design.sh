#!/bin/bash
#SBATCH --job-name=chisel_design
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
#SBATCH --output=/net/scratch/%u/slurm-chisel-%j.out
#SBATCH --error=/net/scratch/%u/slurm-chisel-%j.err

# protein_chisel — PLM-fusion-driven iterative design pipeline.
#
# Four stages, three sifs (composed via shared filesystem; apptainer
# can only exec one image at a time, so each stage runs in its image):
#   1. classify_positions    -> pyrosetta.sif                 (CPU)
#   2. precompute PLM logits -> protein_chisel_plm.sif        (GPU recommended)
#   3. iterative driver      -> protein_chisel_design.sif     (GPU or CPU)
#   4. protonate top-K       -> pyrosetta.sif                 (CPU)
#
# Outputs go under $OUTPUT_DIR (alias WORK_ROOT). Default
# /net/scratch/$USER/. Each run creates a timestamped subdir
# unless MINIMAL=1 + a non-default OUTPUT_DIR triggers consolidation
# to a flat layout (PDBs + chiseled_design_metrics.tsv only).
#
# Two callable patterns:
#   (a) sbatch run_chisel_design.sh   — honors the #SBATCH directives above.
#   (b) bash run_chisel_design.sh     — useful inside an outer slurm
#                                       array job (the #SBATCH lines
#                                       become inert comments in this
#                                       mode; outer array's resources win).
# Both are first-class supported.

set -euo pipefail

# === Portability ====================================================
# REPO is auto-detected from this script's location so any user can
# `git clone` into their own home and the sbatch just works. Detection
# is two-pronged because direct `sbatch <script>` copies the script to
# /var/slurm/spool/, breaking BASH_SOURCE-based detection — but the
# bash <script> pattern (used inside outer slurm array jobs, the
# lab-standard) preserves it.
#
# Order of resolution:
#   1. REPO=... if explicitly set by caller (always wins).
#   2. BASH_SOURCE[0] dirname (works for `bash <full-path-to-script>`).
#   3. SLURM_SUBMIT_DIR (works for `sbatch <script>` if user submitted
#      from inside the repo checkout).
#   4. Hard error with actionable hint.
# A canary file (scripts/run_chisel_design.sh itself) is checked at
# each candidate location to confirm it really is a protein_chisel
# checkout before accepting it.
_CANARY="scripts/run_chisel_design.sh"
_validate_repo() { [[ -f "$1/$_CANARY" ]]; }

if [[ -n "${REPO:-}" ]]; then
    _validate_repo "$REPO" || { echo "ERROR: REPO=$REPO is not a protein_chisel checkout (no $_CANARY)" >&2; exit 1; }
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
    REPO_GUESS="$(dirname "$SCRIPT_DIR")"
    if _validate_repo "$REPO_GUESS"; then
        REPO="$REPO_GUESS"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && _validate_repo "$SLURM_SUBMIT_DIR"; then
        REPO="$SLURM_SUBMIT_DIR"
    else
        echo "ERROR: cannot auto-detect protein_chisel checkout." >&2
        echo "  BASH_SOURCE -> $REPO_GUESS  (canary missing)" >&2
        echo "  SLURM_SUBMIT_DIR -> ${SLURM_SUBMIT_DIR:-(unset)}" >&2
        echo "  Fix: either run 'bash /full/path/to/run_chisel_design.sh', or" >&2
        echo "       'sbatch run_chisel_design.sh' from inside the repo, or" >&2
        echo "       set REPO=/path/to/protein_chisel explicitly." >&2
        exit 1
    fi
fi
echo "REPO=$REPO"

# === Required inputs (per-design caller MUST set these) ============
# Both names accepted at the sbatch surface (caller picks what reads
# best). Internally the script + python driver use the SEED_*/WORK_*
# names because that's their semantic role (the seed for a design
# campaign + the working-dir root). The INPUT_PDB / OUTPUT_DIR aliases
# are caller ergonomics only.
#
# SEED_PDB | INPUT_PDB: starting backbone with REMARK 666 catalytic
#   motif + ligand placement. REMARK 666 drives auto-derivation of
#   catalytic residues so you don't hardcode resnos here. REQUIRED.
# LIG_PARAMS: Rosetta .params file for the ligand. REQUIRED.
SEED_PDB="${SEED_PDB:-${INPUT_PDB:-}}"
if [[ -z "$SEED_PDB" ]]; then
    echo "ERROR: set SEED_PDB or INPUT_PDB to your input PDB (with REMARK 666)" >&2
    exit 1
fi
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
# OUTPUT_DIR | WORK_ROOT (caller picks; WORK_ROOT remains the internal
# variable). Top-level directory under which work_dir/ and run_dir/ get
# created. Defaults to /net/scratch/$USER. Override per-call (e.g. from
# a sweep notebook) to put each input PDB's outputs in its own isolated
# subdirectory:
#   OUTPUT_DIR=/net/scratch/aruder2/.../chisel_out/<pdb_stem> \
#       bash scripts/run_chisel_design.sh
WORK_ROOT="${WORK_ROOT:-${OUTPUT_DIR:-/net/scratch/$USER}}"
mkdir -p "$WORK_ROOT"

# Working dir for this job (ms + PID for collision safety with concurrent jobs)
TS=$(python3 -c 'import datetime,os; t=datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]; print(f"{t}-pid{os.getpid()}")' 2>/dev/null || date +%Y%m%d-%H%M%S)
WORK_DIR="$WORK_ROOT/chisel_design_${TS}"
mkdir -p "$WORK_DIR"
echo "=== work dir: $WORK_DIR ==="

CLASSIFY_DIR="$WORK_DIR/classify"
PLM_DIR="$WORK_DIR/plm_artifacts"
mkdir -p "$CLASSIFY_DIR" "$PLM_DIR"

# Some pyrosetta.sif images export PYTHONNOUSERSITE — so PYTHONPATH alone
# lets us put protein_chisel on the import path. The default container
# PYTHONPATH for esmc.sif and universal.sif holds /cifutils/src etc. that
# we must preserve.

# === GPU/CPU mode selection ========================================
# Auto-detect by default; override with USE_GPU=1 (force --nv) or
# USE_GPU=0 (force CPU, drop --nv from stages 2 + 3). Stages 1 + 4
# never use --nv (PyRosetta is CPU-only) and are unaffected.
#
# Why this matters: when this script is run via `bash <file>` inside
# an outer slurm array job (the lab-standard pattern), the #SBATCH
# directives at the top become inert comments and the job inherits
# whatever resources the outer array requested. If that outer array
# is on a CPU partition, `apptainer exec --nv` fails (or silently
# yields no GPU). Auto-detection avoids that footgun.
NV_FLAGS=()
if [[ "${USE_GPU:-auto}" == "0" ]]; then
    GPU_MODE="CPU (USE_GPU=0 override)"
elif [[ "${USE_GPU:-auto}" == "1" ]]; then
    NV_FLAGS=("--nv")
    GPU_MODE="GPU (USE_GPU=1 override)"
elif command -v nvidia-smi &>/dev/null && nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
    NV_FLAGS=("--nv")
    GPU_MODE="GPU (auto-detected: $(nvidia-smi -L 2>/dev/null | head -1 | sed 's/ (UUID.*//'))"
else
    GPU_MODE="CPU (no GPU detected)"
fi
echo "================================================================"
echo "  protein_chisel mode: $GPU_MODE"
if [[ "$GPU_MODE" == CPU* ]]; then
    echo "  CPU NOTE: stage 2 (ESM-C $ESMC_MODEL + SaProt $SAPROT_MODEL)"
    echo "  is ~10-14x slower than GPU. Benchmark 2026-05-04 (L=202,"
    echo "  600m+1.3b, cpus=4): 50:32 wall, 14.3 GB MaxRSS. Allow 60-90"
    echo "  min walltime; bump cpus to 4 if slurm priority allows."
fi
echo "================================================================"

echo
echo "################################################################"
echo "###  STAGE 1: classify_positions (pyrosetta.sif)             ###"
echo "################################################################"
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

# Stage 2 sif: ESM-C + SaProt + foldseek stack. Canonical name is
# protein_chisel_plm.sif (symlink to esmc.sif at the same path).
PLM_SIF_NEW=/net/software/containers/users/woodbuse/protein_chisel_plm.sif
PLM_SIF_OLD=/net/software/containers/users/woodbuse/esmc.sif
if [[ -n "${PLM_SIF:-}" && -e "$PLM_SIF" ]]; then
    STAGE2_SIF="$PLM_SIF"
elif [[ -e "$PLM_SIF_NEW" ]]; then
    STAGE2_SIF="$PLM_SIF_NEW"
else
    STAGE2_SIF="$PLM_SIF_OLD"
fi
echo
echo "################################################################"
echo "###  STAGE 2: PLM logits + fusion                            ###"
echo "###  sif: $STAGE2_SIF"
echo "################################################################"
apptainer exec "${NV_FLAGS[@]}" \
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
    "$STAGE2_SIF" \
    python "$REPO/scripts/precompute_plm_artifacts.py" \
        --seed_pdb "$SEED_PDB" \
        --position_table "$CLASSIFY_DIR/positions.tsv" \
        --out_dir "$PLM_DIR" \
        --esmc_model "$ESMC_MODEL" \
        --saprot_model "$SAPROT_MODEL"

# Stage 3 sif selection. The protein_chisel suite at the canonical
# user-shared dir uses friendly names:
#   protein_chisel_plm.sif    -> esmc.sif                       (stage 2)
#   protein_chisel_design.sif -> universal_with_tunnel_tools.sif (stage 3)
# Both are symlinks at /net/software/containers/users/woodbuse/, so
# either name works. Override via TUNNEL_SIF=... env var if you have a
# custom build. Final fallback is the system universal.sif (loses
# pyKVFinder; homegrown ray-cast in tunnel_metrics still runs).
SIF_CANON_NEW=/net/software/containers/users/woodbuse/protein_chisel_design.sif
SIF_CANON_OLD=/net/software/containers/users/woodbuse/universal_with_tunnel_tools.sif
if [[ -n "${TUNNEL_SIF:-}" && -e "$TUNNEL_SIF" ]]; then
    STAGE3_SIF="$TUNNEL_SIF"
elif [[ -e "$SIF_CANON_NEW" ]]; then
    STAGE3_SIF="$SIF_CANON_NEW"
elif [[ -e "$SIF_CANON_OLD" ]]; then
    STAGE3_SIF="$SIF_CANON_OLD"
else
    STAGE3_SIF=/net/software/containers/universal.sif
    echo "WARN: protein_chisel_design.sif not found; using system"
    echo "      universal.sif (--tunnel_metrics will skip pyKVFinder)"
fi
echo
echo "################################################################"
echo "###  STAGE 3: iterative driver (sample / filter / score)     ###"
echo "###  sif: $STAGE3_SIF"
echo "################################################################"
apptainer exec "${NV_FLAGS[@]}" \
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
    echo
    echo "################################################################"
    echo "###  STAGE 4: protonate final top-K + shipping layout         ###"
    echo "###  sif: pyrosetta.sif                                       ###"
    echo "################################################################"
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
