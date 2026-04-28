# Setup

## Prerequisites

Required cluster resources (already present):

- Apptainer ≥ 1.4 on login + compute nodes.
- The `esmc.sif` container at `/net/software/containers/users/woodbuse/esmc.sif`. Rebuild from `/net/software/containers/users/woodbuse/spec/esmc.spec` if missing.
- Read access to `/net/databases/huggingface/esmc/` and `/net/databases/huggingface/saprot/`.

## Install the package

The package itself is a thin orchestrator and is light to install. The heavy runtime deps (torch, esm, transformers, foldseek, etc.) come from the apptainer images.

```bash
git clone <repo-url> ~/codebase_projects/protein_chisel
cd ~/codebase_projects/protein_chisel
pip install -e .
```

If you want to import the package from inside a sif, two options:

### Option A: bind-mount and PYTHONPATH (no rebuild)

```bash
apptainer exec \
    --bind ~/codebase_projects/protein_chisel:/code \
    --env PYTHONPATH=/code/src \
    /net/software/containers/users/woodbuse/esmc.sif \
    python -m protein_chisel.cli --help
```

This is the recommended way during development.

### Option B: pip install -e inside the sif

Possible but requires an interactive `apptainer shell --writable-tmpfs` session and re-doing it whenever the sif is rebuilt. Not worth it for normal use.

## Running pipelines

```bash
chisel <pipeline> --config configs/<your-config>.yaml
```

Pipelines write a single output directory (configurable in the config) with intermediate files per stage. Restart-safe — re-running skips stages whose outputs already exist.

## Slurm

Tools and pipelines emit slurm-ready helpers in `scripts/`. Default partition / GPU type / memory are set in those templates; override per-job with sbatch flags.

## Cluster-specific gotchas

- **NFS attribute caching.** `/net/software` has 10-minute attribute caches on compute nodes. After rebuilding a sif, expect 5–10 min of stale-content errors on first sbatch. Just retry.
- **Heterogeneous gres buckets.** `--gres=gpu:a4000` schedules either RTX A4000 (Ampere sm_86) or RTX 4000 Ada (sm_89). Both work with the cu128 wheels in `esmc.sif`. Use `--gres=gpu:b4000:1` to specifically target Blackwell.
- **HF cache writability.** `/net/databases/huggingface/cache` symlinks to per-node `/scratch` and is not writable from login. Use the parallel directories listed in `paths.py` instead.
