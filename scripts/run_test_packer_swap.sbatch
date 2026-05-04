#!/bin/bash
#SBATCH --job-name=chisel_packer_swap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-packer-swap-%j.out
#SBATCH --error=slurm-packer-swap-%j.err

# Run scripts/test_packer_swap.py against the GPU-resident packers
# (PIPPack, AttnPacker). FASPR is CPU-only and was already exercised
# directly on the host.
#
# Usage:
#   sbatch scripts/run_test_packer_swap.sbatch
#
# Output:
#   slurm-packer-swap-<jobid>.{out,err}
#   ./test_packer_swap_results/results.tsv

set -uo pipefail

REPO=/home/woodbuse/codebase_projects/protein_chisel
cd "$REPO"

# Prime autofs so the apptainer wrappers can bind these paths.
for p in \
    /net/databases/lab/pippack/model_weights \
    /net/databases/lab/attnpacker/AttnPackerPTM_V2 \
    /net/software/containers/users/woodbuse \
    /net/scratch/woodbuse \
    ; do
    [ -e "$p" ] && echo "ok: $p" || echo "MISSING: $p"
done

OUTDIR=$REPO/test_packer_swap_results
mkdir -p "$OUTDIR"

echo "=== nvidia-smi ==="
nvidia-smi -L
echo

echo "=== running FASPR (host CPU) for full coverage ==="
PYTHONPATH=src python3 scripts/test_packer_swap.py \
    --packers faspr \
    --out-dir "$OUTDIR/faspr"
echo

echo "=== running PIPPack ==="
PYTHONPATH=src python3 scripts/test_packer_swap.py \
    --packers pippack \
    --out-dir "$OUTDIR/pippack" \
    || echo "*** PIPPack failed -- continuing ***"
echo

echo "=== running AttnPacker ==="
PYTHONPATH=src python3 scripts/test_packer_swap.py \
    --packers attnpacker \
    --out-dir "$OUTDIR/attnpacker" \
    || echo "*** AttnPacker failed -- continuing ***"
echo

echo "=== merging results ==="
{
    head -n1 "$OUTDIR/faspr/results.tsv"
    for sub in faspr pippack attnpacker; do
        if [ -f "$OUTDIR/$sub/results.tsv" ]; then
            tail -n +2 "$OUTDIR/$sub/results.tsv"
        fi
    done
} > "$OUTDIR/results.tsv"
echo "merged → $OUTDIR/results.tsv"
column -t -s $'\t' "$OUTDIR/results.tsv" | head -40

echo
echo "Done at $(date)"
