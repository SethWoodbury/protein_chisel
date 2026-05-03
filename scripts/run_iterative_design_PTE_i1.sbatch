#!/bin/bash
#SBATCH --job-name=PTE_i1_iter_design
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/net/scratch/woodbuse/slurm-iter-design-%j.out
#SBATCH --error=/net/scratch/woodbuse/slurm-iter-design-%j.err

# Iterative-design driver for the PTE_i1 scaffold.
# Runs the driver inside universal.sif which bundles:
#   - python + Bio + pandas + numpy
#   - fused_mpnn (LigandMPNN sampling)
#   - fpocket (when /net/software is bound in)

set -euo pipefail

REPO=/home/woodbuse/codebase_projects/protein_chisel
DRIVER=$REPO/scripts/iterative_design_PTE_i1.py

mkdir -p /net/scratch/woodbuse

# Need /net/software for fused_mpnn + fpocket binary, /net/databases for
# any model weights, /net/scratch for outputs, /home/woodbuse for inputs.
# Preserve universal.sif's default PYTHONPATH (which contains /cifutils/src
# needed by fused_mpnn) and prepend our protein_chisel src/.
apptainer exec --nv \
    --bind "${REPO}:/code" \
    --bind /net/software \
    --bind /net/databases \
    --bind /net/scratch \
    --bind /home/woodbuse \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    /net/software/containers/universal.sif \
    python "$DRIVER" "$@"
