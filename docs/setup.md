# Setup

## Prerequisites

Required cluster resources (already present):

- Apptainer ≥ 1.4 on login + compute nodes.
- The `esmc.sif` container at `/net/software/containers/users/woodbuse/esmc.sif`. Rebuild from `/net/software/containers/users/woodbuse/spec/esmc.spec` if missing.
- Read access to `/net/databases/huggingface/esmc/` and `/net/databases/huggingface/saprot/`.

## Install the package

The package itself is a thin orchestrator and is light to install. The heavy runtime deps (torch, esm, transformers, foldseek, pyrosetta, ...) come from the apptainer images.

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

This is the recommended way during development. The `utils/apptainer.ApptainerCall` helper does this automatically.

### Option B: pip install -e inside the sif

Possible but requires an interactive `apptainer shell --writable-tmpfs` session and re-doing it whenever the sif is rebuilt. Not worth it for normal use.

## What runs where

| Stage / tool | sif | GPU? | Why |
|---|---|---|---|
| classify_positions, backbone_sanity, shape_metrics, secondary_structure / ss_summary, ligand_environment, chemical_interactions, buns, catres_quality, preorganization | `pyrosetta.sif` | no | PyRosetta + Coventry SASA recipe |
| comprehensive_metrics pipeline | `pyrosetta.sif` | no | wraps the structural tools |
| esmc_logits / esmc_score | `esmc.sif` | yes (recommended) | new evolutionaryscale `esm` package + HF cache |
| saprot_logits / saprot_score | `esmc.sif` | yes | foldseek (bundled in sif) + transformers + saprot HF cache |
| naturalness_metrics pipeline | `esmc.sif` | yes | PLM scoring + fusion-bias artifacts |
| contact_ms | `esmc.sif` (or any sif w/ py_contact_ms) | no | numpy-only; PyRosetta-free |
| catalytic_pka | `esmc.sif` | no | PROPKA was added there |
| sample_with_ligand_mpnn | `universal.sif` | yes | fused_mpnn build at `/net/software/lab/fused_mpnn/seth_temp/` |
| sequence_design_v1 pipeline | mixed: pyrosetta.sif (stage 0) → esmc.sif (stages 1-2) → mlfold.sif (stage 3) → esmc.sif (stages 4-5) | yes | orchestrates 4 sifs |
| theozyme_satisfaction | host (numpy + io/pdb only) | no | pure-Python Kabsch alignment |
| iterative_optimize pipeline | host (numpy) | no | proposal sampling + acceptance only |
| Filters (protparam, protease_sites, length, expression_host) | host | no | sequence-only |
| Scoring (aggregate, pareto, diversity) | host | no | numpy + pandas |
| fpocket_run | (any sif w/ fpocket binary) | no | **binary not yet installed**, see below |
| metal3d_score | `metal3d.sif` (planned) | yes | inference path **stubbed**, only HETATM scan active |

## Container deps recap (per [docs/dependencies.md](dependencies.md))

| sif | Python | Key extras for chisel |
|---|---|---|
| `pyrosetta.sif` | 3.12 | PyRosetta + Coventry SASA + DSSP |
| `esmc.sif` (lab user-built) | 3.12 | torch 2.11+cu128, evolutionaryscale `esm` 3.2.3, transformers 4.48.1, foldseek (avx2 static), `py_contact_ms`, PROPKA, biopython |
| `universal.sif` | 3.11 | fused_mpnn runner |
| `mlfold.sif` | — | LigandMPNN sampling alternative path |
| `metal3d.sif` | — | Metal3D weights (inference NOT wired yet) |
| `rosetta.sif`, `esmfold.sif`, `af3.sif` | varies | not directly used yet |

### Pending sif additions (user TODO)

- `fpocket` binary into a sif (see [tools/fpocket_run.py:13](../src/protein_chisel/tools/fpocket_run.py#L13)).
- ProLIF + pdbe-arpeggio (planned for a richer interaction-fingerprint sif; the wrappers are TBD — see [docs/future_plans.md](future_plans.md)).
- CAVER (tunnel detection — TBD).

## Running pipelines

```bash
# CLI
chisel comprehensive-metrics <pdb> --out <dir> [--params <p>]
chisel classify-positions <pdb> --out <path> [--params <p>] [--catres-spec ...]
chisel esmc-score <sequence> [--model esmc_300m]
chisel saprot-score <pdb> [--chain A] [--model saprot_35m]

# sbatch wrappers
sbatch scripts/run_comprehensive_metrics.sbatch <pdb> <out_dir> [params_dir]
sbatch scripts/run_naturalness_metrics.sbatch <pdb> <out_dir> [position_table_dir]
sbatch scripts/run_sequence_design_v1.sbatch <pdb> <ligand_params> <out_dir>
```

Pipelines write a single output directory (see per-pipeline layout in [docs/pipelines.md](pipelines.md)). Restart-safe via manifest-hash matching for `comprehensive_metrics` and `naturalness_metrics`; file-existence only for `sequence_design_v1` (a known weakness).

## Slurm

Default sbatch templates in `scripts/`:

- `run_comprehensive_metrics.sbatch` — `--partition=cpu`, `--mem=8G`, `--time=02:00:00`.
- `run_naturalness_metrics.sbatch` — `--partition=gpu --gres=gpu:a4000:1`, `--mem=24G`, `--time=02:00:00`.
- `run_sequence_design_v1.sbatch` — `--partition=gpu --gres=gpu:a4000:1`, `--mem=32G`, `--time=04:00:00`.

Override per-job with sbatch flags (e.g. `sbatch --time=08:00:00 ...`).

## Cluster-specific gotchas

- **NFS attribute caching.** `/net/software` has 10-minute attribute caches on compute nodes. After rebuilding a sif, expect 5–10 min of stale-content errors on first sbatch. Just retry.
- **Heterogeneous gres buckets.** `--gres=gpu:a4000` schedules either RTX A4000 (Ampere sm_86) or RTX 4000 Ada (sm_89). Both work with the cu128 wheels in `esmc.sif`. Use `--gres=gpu:b4000:1` to specifically target Blackwell.
- **HF cache writability.** `/net/databases/huggingface/cache` symlinks to per-node `/scratch` and is not writable from login. Use the parallel directories listed in `paths.py` (`HF_CACHE_ESMC`, `HF_CACHE_SAPROT`) instead.
- **`pyrosetta.sif` sets `PYTHONNOUSERSITE=1`.** Host-installed pytest etc. won't be importable. The `ApptainerCall.with_pytest()` helper in [utils/apptainer.py:117](../src/protein_chisel/utils/apptainer.py#L117) prepends the host's user-site so they work.
- **First-run NFS errors after sif rebuild** look like `lstat: no such file or directory` even though the file is there — wait 5–10 min for NFS attributes to refresh.
