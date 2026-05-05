#!/bin/bash
#SBATCH --job-name=probe_275aa
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4 --mem=24G --time=01:00:00
#SBATCH --output=/net/scratch/woodbuse/slurm-probe-275aa-%j.out
#SBATCH --error=/net/scratch/woodbuse/slurm-probe-275aa-%j.err
set -euo pipefail
apptainer exec \
    --bind /home/woodbuse/codebase_projects/protein_chisel:/code \
    --bind /net/software --bind /net/databases --bind /net/scratch --bind /home/woodbuse \
    --bind /net/databases/huggingface/esmc \
    --bind /net/databases/huggingface/saprot \
    --env "PYTHONPATH=/code/src" \
    --env "HF_HOME=/net/databases/huggingface/esmc" \
    --env "HF_HUB_CACHE=/net/databases/huggingface/esmc/hub" \
    --env "SAPROT_HF_CACHE=/net/databases/huggingface/saprot/hub" \
    /net/software/containers/users/woodbuse/esmc.sif \
    python /home/woodbuse/codebase_projects/protein_chisel/scripts/probe_plm_memory.py \
        --length 275 --esmc_model esmc_600m --saprot_model saprot_1.3b --device cpu
