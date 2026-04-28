#!/bin/bash
#SBATCH --job-name=chisel_naturalness
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-chisel-naturalness-%j.out
#SBATCH --error=slurm-chisel-naturalness-%j.err

# Run protein_chisel naturalness pipeline (ESM-C + SaProt + PLM fusion).
#
# Usage:
#   sbatch run_naturalness_metrics.sbatch <pdb> <out_dir> [position_table_dir]

set -euo pipefail

PDB="$1"
OUT="$2"
POS_DIR="${3:-}"

REPO=/home/woodbuse/codebase_projects/protein_chisel
SIF=/net/software/containers/users/woodbuse/esmc.sif

apptainer exec --nv \
    --bind "${REPO}:/code" \
    --bind /net/databases/huggingface/esmc \
    --bind /net/databases/huggingface/saprot \
    --env "PYTHONPATH=/code/src" \
    --env "HF_HOME=/net/databases/huggingface/esmc" \
    --env "HF_HUB_CACHE=/net/databases/huggingface/esmc/hub" \
    "$SIF" \
    python -c "
from pathlib import Path
import sys
from protein_chisel.io.schemas import PoseSet
from protein_chisel.pipelines.naturalness_metrics import (
    NaturalnessConfig, run_naturalness_metrics,
)

ps = PoseSet.from_single_pdb('$PDB')
cfg = NaturalnessConfig()
pos_dir = '$POS_DIR' or None
result = run_naturalness_metrics(
    ps, out_dir='$OUT', config=cfg,
    position_table_dir=pos_dir if pos_dir else None,
)
print(f'Wrote naturalness MetricTable with {len(result.metric_table.df)} rows.')
"
