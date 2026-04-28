#!/bin/bash
#SBATCH --job-name=chisel_metrics
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-chisel-metrics-%j.out
#SBATCH --error=slurm-chisel-metrics-%j.err

# Run protein_chisel.pipelines.comprehensive_metrics on a PDB.
#
# Usage:
#   sbatch run_comprehensive_metrics.sbatch <pdb> <out_dir> [params_dir]
#
# All paths are absolute and live under /net or /home.

set -euo pipefail

PDB="$1"
OUT="$2"
PARAMS="${3:-}"

REPO=/home/woodbuse/codebase_projects/protein_chisel
SIF=/net/software/containers/pyrosetta.sif

PARAMS_FLAG=""
if [[ -n "$PARAMS" ]]; then
    PARAMS_FLAG="--params $PARAMS"
fi

apptainer exec \
    --bind "${REPO}:/code" \
    --env "PYTHONPATH=/code/src:/pyrosetta" \
    "$SIF" \
    python -m protein_chisel.cli comprehensive-metrics "$PDB" \
        --out "$OUT" \
        ${PARAMS_FLAG}
