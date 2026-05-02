#!/bin/bash
#SBATCH --job-name=chisel_sidechain_pack_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-chisel-sidechain-%j.out
#SBATCH --error=slurm-chisel-sidechain-%j.err

# Run all four GPU-required sidechain packer tests against the freshly
# rebuilt esmc.sif. Each tool's test_*_runs_end_to_end exercise full
# inference on design.pdb. Continue past failures so we get reports on
# all four even if one bombs.
#
# Usage:
#   sbatch scripts/run_sidechain_packing_tests.sbatch

set -uo pipefail

REPO=/home/woodbuse/codebase_projects/protein_chisel
cd "$REPO"

echo "=== nvidia-smi ==="
nvidia-smi -L
echo

# Trigger autofs to mount /mnt/net/{databases,software}/... paths the
# wrappers will bind. Compute nodes use autofs and only mount paths
# that have been touched -- if a path hasn't been accessed, apptainer's
# --bind <path>:<path> fails with "mount source doesn't exist".
echo "=== priming autofs for /net paths used by the wrappers ==="
for p in \
    /net/databases/lab/pippack/model_weights \
    /net/databases/lab/flowpacker/checkpoints \
    /net/databases/lab/attnpacker/AttnPackerPTM_V2 \
    /net/databases/lab/opus_rota5/opus_rota5/Rota5/models \
    /net/databases/lab/opus_rota5/opus_rota5/Rota5/unet3d/models \
    /net/databases/lab/molprobity/monomers \
    /net/databases/lab/molprobity/chem_data \
    /net/software/utils \
    /net/software/containers/users/woodbuse \
    ; do
    if [ -e "$p" ]; then
        echo "  ok: $p"
    else
        echo "  MISSING: $p"
    fi
done
echo

# Run each test in its own pytest invocation so they don't share Python
# process state (different tools have conflicting torch/TF imports if
# loaded in the same process).
for tool in pippack flowpacker attnpacker opus_rota5; do
    echo "============================================================"
    echo "  $tool : starting at $(date)"
    echo "============================================================"
    PYTHONPATH=src python3 -m pytest -m slow \
        tests/sidechain_packing_and_scoring/test_${tool}_*.py -v --tb=long \
        || echo "*** $tool tests FAILED -- continuing to next tool ***"
    echo
done

echo "All sidechain packer tests done at $(date)"
