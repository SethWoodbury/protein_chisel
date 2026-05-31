# scripts/

Standalone drivers, SLURM wrappers, and diagnostic tools that aren't part of
the importable `protein_chisel` package. The production design pipeline lives
here (it's a multi-`sif` orchestration, so it can't be a normal package entry
point).

## Production pipeline (the chisel design run)

| File | Role |
|---|---|
| `run_chisel_design.sh` | **Production entry.** 4-stage / 3-`sif` SLURM wrapper. Submit with `sbatch`. |
| `classify_positions_pte_i1.py` | Stage 1 — directional position classifier (`pyrosetta.sif`). |
| `precompute_plm_artifacts.py` | Stage 2 — ESM-C + SaProt logits + fusion bias (`plm.sif`, GPU). |
| `iterative_design.py` | Stage 3 — the main iterative design driver (`design.sif`). |
| `protonate_final_topk.py` | Stage 4 — protonation + shipping reorganization (`pyrosetta.sif`). |
| `load_chiseled_runs.py` | JupyterHub helper: glob-load run output TSVs into one DataFrame. |

See the top-level [README](../README.md) and [docs/architecture.md](../docs/architecture.md).

## Package-pipeline wrappers

Thin `sbatch` wrappers around the `protein_chisel.pipelines.*` orchestrators:

- `run_comprehensive_metrics.sbatch` — `pipelines.comprehensive_metrics`
- `run_naturalness_metrics.sbatch` — `pipelines.naturalness_metrics`

## Offline / validation tools

- `run_caver.sh` — offline CAVER 3.0.3 tunnel validation on a top-K (see [docs/tunnel_analysis.md](../docs/tunnel_analysis.md)).
- `run_metal3d.py` / `run_metal3d_test.sbatch` — Metal3D batch runner + its test.
- `run_sidechain_packing_tests.sbatch` — runs the `tests/sidechain_packing_and_scoring/` suite on the cluster.

## Diagnostics / one-off analysis

These are ad-hoc tools — many hard-code specific `/net/scratch` run dirs and are
meant to be edited per investigation, not run as-is:

- Audits: `audit_aa_comp.py`, `audit_clash_omits.py`, `audit_pocket_metrics.py`
- Reports / comparisons: `full_diagnostic_report.py`, `compare_directional_to_baseline.py`, `aggregate_plm_metrics.py`, `flexibility_score.py`
- Ablations / benchmarks: `mpnn_ablation_test.py` (+ `run_mpnn_ablation.sbatch`), `test_packer_swap.py` (+ `run_test_packer_swap.sbatch`)
- Profiling: `probe_plm_memory.py` (+ `probe_plm_memory.sbatch`)
- Variant scoring: `generate_variant_pdbs.py`, `score_variants_esmc.py` (+ `score_variants_esmc.sbatch`)
- JupyterHub cell templates: `notebook_driver_cell.py`, `notebook_driver_cell_array_job.py`

## Legacy

- `iterative_design_PTE_i1.py` / `run_iterative_design_PTE_i1.sbatch` — the original
  **v1 single-shot** driver, superseded by `iterative_design.py`. Kept for reference;
  not part of the production pipeline.
