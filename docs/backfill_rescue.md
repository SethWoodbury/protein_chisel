# Backfill + deferred rescue (final selection)

Added in commit `b4f5509` (merge of `aruder2/ANTONIA-TEMP-FIXES`). Makes the final-selection stage robust to two situations that previously caused empty / short shipments:

1. **Strict druggability cutoff empties the pool.** Before: pipeline aborted with `EMPTY_POOL_AFTER_FPOCKET_FILTER`. After: the pipeline reaches into the per-cycle `02_seq_filter/` artifacts (rejects + survivors), backfills with the most-promising "near-miss" candidates, and re-scores them through the rest of the pipeline so they carry the same metric set as primary survivors.

2. **Cycle survivors don't reach `target_k` after diversity selection.** Before: shipped fewer than `target_k` PDBs and called it done. After: same backfill path tops up to `target_k` from the seq-stage rescue pool.

Default is **on** (`--final_filter_backfill true`). Disable with `--final_filter_backfill false` if you specifically want the strict-cutoff behavior for a benchmark.

## Algorithm sketch

```
1. Run 3 cycles as usual -> all_ranked, all_pdb_maps.
2. Concat + dedup -> primary_pool.
3. Apply strict fpocket-druggability filter -> survivors_strict.
4. If len(survivors_strict) >= target_k:
       run progressive-diversity TOPSIS selection -> top
   Else:
       deficit = target_k - len(survivors_strict)
       Build seq-stage rescue pool from cycle_NN/02_seq_filter/{rejects,survivors}
         (in-memory re-scoring with fitness + TOPSIS; no fpocket yet).
       Pick top-K rescue candidates by deficit + cap (default 200).
       Re-score that shortlist through stage_struct_filter ->
         stage_tunnel_metrics -> stage_fitness_score -> stage_fpocket_rank
         (so backfilled rows carry the full metric set).
       Overlay rescued metrics onto the rescue rows ->
         augmented_pool = primary_pool ∪ overlay_rows_by_id(rescue rows).
       Run progressive-diversity TOPSIS on augmented_pool -> top.
```

## What you'll see in the output

`chiseled_design_metrics.tsv` has new columns to make rescue activity inspectable:

| column | meaning |
|---|---|
| `selection__bucket` | which selection bucket the row landed in (`primary`, `rescue`, ...) |
| `selection__bucket_priority` | numeric tiebreaker within bucket |
| `selection__hard_final_filter_passed` | boolean — did the row pass the strict druggability cutoff? |
| `selection__fpocket_gap` | distance from druggability cutoff (0 = at threshold; positive = above) |
| `selection__deferred_rescue_requested` | True if the row came from the seq-stage rescue pool |
| `selection__deferred_rescue_attempted` | True if rescue re-scoring was actually run for this row |
| `selection__deferred_rescue_struct_failed` | True if struct-filter rejected during rescue |
| `selection__deferred_rescue_tunnel_failed` | True if tunnel-metrics failed during rescue |

A run that ships `target_k` purely from primary survivors will show all rescue columns as False / NaN. A run that backfilled will show some rows with `selection__bucket=rescue` and the deferred-rescue flags populated.

## Solubility veto (opt-in — `--ship_solubility_veto` / `SHIP_SOLUBILITY_VETO=1`)

Backfill exists to recover **fpocket-druggability** near-misses (and to top up to
`target_k`). It was **never** meant to override the **solubility** band. But the rescue
re-scoring replays struct/tunnel/fpocket only and does **not** re-apply the GRAVY/charge
seq filter, so a `passed_seq_filter=False` design (e.g. GRAVY=1.05) could be bucketed
`rescued_final_filters` and shipped as rank-0. **Caveat on the columns above:**
`selection__hard_final_filter_passed` means *"passed the strict fpocket-druggability
cutoff"* — **not** solubility; it can read `True` for a GRAVY-failing design.

Set `--ship_solubility_veto` (default OFF → byte-identical) to make the final top-K writer
**drop any design outside the final-cycle GRAVY + net-charge band before shipping** — the
single hard chokepoint both selection branches pass through, covering rescued rows *and*
annealed primary rows that pass an early-cycle GRAVY band but fail the final one. It adds a
truthful `selection__solubility_passed` column. With the veto on, **fewer than `target_k`
designs may ship** (intended: better to ship 35 soluble than 40 with garbage). The band
comes from the final `CycleConfig` (`gravy_min/gravy_max`, `net_charge_min/net_charge_max`);
charge bounds are exclusive and GRAVY bounds inclusive, mirroring `stage_seq_filter`.

## Limits

The rescue path is **intra-process only**. It looks at `cycle_NN/02_seq_filter/` artifacts that the same `iterative_design.py` process wrote earlier in its own execution — not at any prior partial run dir. It is robustness against *late-stage filter wipeouts*, not a `--resume` mechanism. (See "Resume from a partial run dir" below for the separate question.)

What rescue *cannot* recover:
- A run that crashed before cycle 0's `02_seq_filter/` was written.
- Rescuing PDBs from a *different* run dir (no cross-process recovery).

## Resume from a partial run dir (NOT implemented)

If a run crashes mid-cycle, rerunning from scratch is currently the only option. True `--recovery` (re-using cycle output across separate `iterative_design.py` invocations) would require persisting these in-memory arrays to disk after each stage:

- `log_probs_esmc`, `log_probs_saprot` (already on disk via `plm_artifacts/`)
- `weights_per_position`, `position_class_array`, `pt`, `ss`, `seed_dfi_metrics` (currently RAM-only)
- `fitness_cache` (currently RAM-only)
- `all_ranked` cumulative across completed cycles

…plus a manifest reload step at startup. None of that exists today. With the P0+P1 fixes + the merged backfill path, the pipeline is now robust enough that the marginal value of `--recovery` is small — most failures it would have been useful for (fpocket subprocess flakes, druggability-zero collapse, IS_ parser confusion, empty-pool TOPSIS) are now fixed at the source.

## New env vars / CLI flags

| name | type | default | role |
|---|---|---|---|
| `--final_filter_backfill` | CLI | `true` | enable seq-stage backfill + deferred rescue when strict cutoff would empty the pool |
| `--debug-short-test` | CLI | off | preset: `target_k=20`, `n_cycles=2`. Overrides user `target_k` / `cycles`. For dev iteration only. |
| `USE_NODE_LOCAL_SCRATCH` | env / CLI | `true` | stage intermediate work in node-local fast scratch (when detected), then atomically republish to `OUTPUT_DIR` at the end. Falls back to `OUTPUT_DIR` directly if no node scratch is available. |
| `CLOBBER_EXISTING_OUTPUTS` | env / CLI | `false` | when republishing to an `OUTPUT_DIR` that already contains wrapper-owned artifacts, remove them first. Allowlist (only `*.pdb`, `chiseled_design_metrics.tsv`, `cycle_*`, `final_topk`, `wrk_*`, etc.). User files outside that allowlist are never deleted. |
| `COPY_INPUT_STRUCTURE_INTO_OUT_DIR` | env / CLI | `true` | also runs the seed PDB through the full scoring pipeline and appends an `input_reference` row to `chiseled_design_metrics.tsv`, so the seed is directly comparable to designs in the same TSV |

CLI flags (e.g. `--use-node-local-scratch true`, `--clobber-existing-outputs true`) override env vars when both are set.

## How to read the rescue columns in JupyterHub

```python
# load_runs lives in scripts/load_chiseled_runs.py (standalone script, not packaged):
import sys; sys.path.insert(0, "/path/to/protein_chisel/scripts")
from load_chiseled_runs import load_runs
df = load_runs("/net/scratch/$USER/chisel_sweep/*/chiseled_design_metrics.tsv")

# How many designs in each top-K were primary vs rescued?
df.groupby(["run_id", "selection__bucket"]).size().unstack(fill_value=0)

# When rescue fired, did struct-filter reject most rescue candidates?
rescued = df[df["selection__deferred_rescue_attempted"] == True]
rescued[["selection__deferred_rescue_struct_failed",
         "selection__deferred_rescue_tunnel_failed"]].mean()
```
