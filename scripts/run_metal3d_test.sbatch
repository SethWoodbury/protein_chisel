#!/bin/bash
#SBATCH --job-name=chisel_metal3d_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-chisel-metal3d-%j.out
#SBATCH --error=slurm-chisel-metal3d-%j.err

# Run the slow Metal3D end-to-end test on a GPU node. The wrapper's
# scripts/run_metal3d.py will auto-launch metal3d.sif from this host
# (it self-detects "outside container" and relaunches with --nv).
#
# Usage:
#   sbatch scripts/run_metal3d_test.sbatch

set -euo pipefail

REPO=/home/woodbuse/codebase_projects/protein_chisel
cd "$REPO"

PYTHONPATH=src python -m pytest -m slow \
    tests/test_metal3d.py::test_metal3d_runs_end_to_end -v
