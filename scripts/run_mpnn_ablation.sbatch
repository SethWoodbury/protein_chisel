#!/bin/bash
#SBATCH --job-name=mpnn_ablation
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --exclude=g2132
#SBATCH --output=/net/scratch/woodbuse/slurm-mpnn-abl-%j.out
#SBATCH --error=/net/scratch/woodbuse/slurm-mpnn-abl-%j.err

set -euo pipefail

REPO=/home/woodbuse/codebase_projects/protein_chisel

apptainer exec --nv \
    --bind "$REPO:/code" \
    --bind /net/software \
    --bind /net/databases \
    --bind /net/scratch \
    --bind /home/woodbuse \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    /net/software/containers/universal.sif \
    python "$REPO/scripts/mpnn_ablation_test.py" "$@"
