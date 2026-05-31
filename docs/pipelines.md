# Pipelines

Pipelines orchestrate tools with file-based handoffs. Each pipeline writes a single output directory; intermediate stage outputs allow restart.

| Pipeline | File | sif(s) | Purpose |
|---|---|---|---|
| `comprehensive_metrics` | [pipelines/comprehensive_metrics.py](../src/protein_chisel/pipelines/comprehensive_metrics.py) | `pyrosetta.sif` | Descriptive structural battery |
| `naturalness_metrics` | [pipelines/naturalness_metrics.py](../src/protein_chisel/pipelines/naturalness_metrics.py) | `esmc.sif` (GPU) | ESM-C + SaProt scoring + fusion bias artifacts |
| `iterative_optimize` | [pipelines/iterative_optimize.py](../src/protein_chisel/pipelines/iterative_optimize.py) | host (numpy) | Single-mutation walk (constrained-LS or MH) |

> The production de-novo design pipeline is the standalone driver
> `scripts/iterative_design.py` (4-stage SLURM wrapper `scripts/run_chisel_design.sh`),
> documented in [`docs/architecture.md`](architecture.md) — not one of the package
> pipelines below.

---

## `comprehensive_metrics`

Modernized replacement for `~/special_scripts/design_filtering/metric_monster__MAIN.py`. Takes a `PoseSet` (or single PDB) and runs the full descriptive structural battery on each pose; emits one `MetricTable` row per pose plus a per-pose `PositionTable`.

### Inputs

- `pose_set: PoseSet` — `PoseSet.from_single_pdb(path)` for a single PDB; multi-conformer inputs become PoseEntry rows with distinct `conformer_index`.
- `out_dir: Path`.
- `params: list[Path]` — ligand `.params` files / dirs.
- `config: ComprehensiveMetricsConfig` — booleans for which tools to run; salt-bridge / π-π / π-cation / BUNS-SASA cutoffs; `ligand_target_atoms` tuple for per-atom SASA.
- `skip_existing: bool` — if True, poses with a manifest match are reloaded from disk instead of recomputed.

### Tools called (in order)

```
classify_positions  →  positions.parquet (per pose)
backbone_sanity     →  backbone__*
shape_metrics       →  shape__*
ss_summary          →  ss__*
ligand_environment  →  ligand__*  (skipped if apo)
chemical_interactions  →  interact__*
buns                →  buns__*  (whitelist auto-derived from REMARK 666)
catres_quality      →  catres__*  (only if catalytic residues present)
protparam           →  protparam__*  (sequence-only, may fail if Bio not in sif)
protease_sites      →  protease__*  (sequence-only)
```

### Output layout

```
out_dir/
├── per_pose/
│   └── <sequence_id>/
│       └── conf<i>/
│           ├── _manifest.json     # provenance hash
│           ├── metrics.tsv        # one row, restart cache
│           └── positions.tsv      # PositionTable
└── metrics.parquet                # MetricTable (one row per pose)
```

### sbatch invocation

```bash
sbatch scripts/run_comprehensive_metrics.sbatch \
    /path/to/design.pdb \
    out/comprehensive \
    /path/to/params_dir   # optional
```

The sbatch wrapper at [scripts/run_comprehensive_metrics.sbatch](../scripts/run_comprehensive_metrics.sbatch) runs in `pyrosetta.sif` on a CPU partition (`--partition=cpu`, `--mem=8G`, `--time=02:00:00`). Multi-PDB inputs go via the `chisel comprehensive-metrics` CLI ([cli.py:27](../src/protein_chisel/cli.py#L27)), which assigns each PDB a unique `conformer_index`.

### Restart behavior

`skip_existing=True` (default) reuses prior outputs iff `_manifest.json` matches. The manifest captures input file SHA-256, the full `ComprehensiveMetricsConfig` dataclass, params paths, sequence_id/conformer_index/fold_source — any change → re-run. See [comprehensive_metrics.py:129](../src/protein_chisel/pipelines/comprehensive_metrics.py#L129).

### Tested

[tests/test_pipeline_comprehensive.py](../tests/test_pipeline_comprehensive.py) (cluster) — single-PDB run, multi-pose run with apo, restart-skip, manifest match.

---

## `naturalness_metrics`

ESM-C + SaProt pseudo-perplexity per pose, plus saved logits and (if a `position_table_dir` is supplied) the calibrated PLM fusion bias as a `.npy`.

### Inputs

- `pose_set: PoseSet`.
- `out_dir: Path`.
- `config: NaturalnessConfig` — `esmc_model` (`esmc_300m`/`esmc_600m`), `saprot_model` (`saprot_35m`/`saprot_650m`/`saprot_1.3b`), `device`, `score_pseudo_perplexity`, `save_logits`, `save_fusion_bias`.
- `position_table_dir: Path | None` — if present, fusion bias is computed using class labels from `<dir>/per_pose/<sequence_id>/conf<i>/positions.tsv`.

### Output layout

```
out_dir/
├── per_pose/<sequence_id>/conf<i>/
│   ├── _manifest.json
│   ├── metrics.tsv
│   ├── esmc_log_probs.npy       # (L, 20) masked-LM marginals
│   ├── saprot_log_probs.npy     # (L, 20)
│   └── fusion_bias.npy          # (L, 20) only if position_table_dir given
└── metrics.parquet              # MetricTable
```

### MetricTable columns

`esmc__pseudo_perplexity`, `esmc__mean_loglik`, `esmc__min_loglik`, `saprot__pseudo_perplexity`, `saprot__mean_loglik`, `saprot__min_loglik`, and (if fusion ran) `fusion__mean_abs_bias`, `fusion__max_abs_bias`.

### sbatch invocation

```bash
sbatch scripts/run_naturalness_metrics.sbatch \
    /path/to/design.pdb \
    out/naturalness \
    out/comprehensive   # position_table_dir, optional but needed for fusion
```

Runs in `esmc.sif` on a GPU partition (`--gres=gpu:a4000:1`, `--mem=24G`). Sets `HF_HOME` and `HF_HUB_CACHE` to the cluster's HF cache dirs at `/net/databases/huggingface/{esmc,saprot}`.

### Restart behavior

Same manifest pattern as `comprehensive_metrics`. The `NaturalnessConfig` is folded into the manifest.

### Tested

**End-to-end test not in CI.** Underlying tools tested in [tests/test_plm_tools.py](../tests/test_plm_tools.py) (cluster).

---

## `iterative_optimize`

Two modes (single tool, both modes in one function):

1. **`constrained_local_search`** (default) — propose from `(L, 20)` PLM-fused marginals at one position, accept iff a user-supplied `accept_fn(s) → bool` returns True. **No proposal-correction term, no scalar energy.** It's NOT Metropolis-Hastings — biased by the proposal distribution. See [docs/architecture.md](architecture.md) "Mode 1: constrained local search".
2. **`mh`** — real Metropolis-Hastings: scalar `energy_fn(s) → float` (lower better) + linear-anneal MH temperature schedule. Includes the **q-correction term** (`log p(old_aa) − log p(new_aa)` at position i) so non-uniform proposals are correctly normalized.

### Inputs

- `starting_sequence: str` (length L).
- `per_position_log_probs: np.ndarray` shape `(L, 20)` in `PLM_AA_ORDER`. Calibrated PLM-fused marginals.
- `fixed_positions: set[int]` — 0-indexed positions to NEVER mutate (typically catalytic residues).
- `energy_fn: Callable[[str], float]` — required for `mh` mode.
- `accept_fn: Callable[[str], bool]` — required for `constrained_local_search` mode.
- `config: IterativeOptimizeConfig` — `mode`, `n_iterations`, `sample_temperature`, `initial_mh_temperature`, `final_mh_temperature`, `n_chains` (independent chains for diagnostics), `convergence_window` (no-improvement iters → stop).

### Output

`IterativeOptimizeResult` with:
- `final_sequences: list[str]` — best per chain.
- `final_scores: list[float]`.
- `candidate_set: CandidateSet` — every accepted move across all chains.
- `walk_log: pd.DataFrame` — per-step `(chain, iter, position, old_aa, new_aa, accepted, score)`.
- `converged: bool`, `n_iterations_run: int`.

If `out_dir` is given: writes `walk_log.tsv` and (if any candidates) `candidates.fasta` + `candidates.tsv`.

### Tested

[tests/test_iterative_optimize.py](../tests/test_iterative_optimize.py) (host) — convergence to peaked target, fixed-positions invariant, MH energy reduction with uniform proposal, multiple independent chains, on-disk output writing.

### Limitations

- **No block moves, no parallel tempering** (mentioned in architecture.md but not implemented). Cannot do compensatory mutations across multiple positions.
- **No convergence diagnostics** beyond the simple "no_improvement window" — no R-hat, no autocorrelation, no acceptance-rate trend (architecture.md flags these as "should have").
- **No sbatch wrapper** — driven from a Python entry point.
