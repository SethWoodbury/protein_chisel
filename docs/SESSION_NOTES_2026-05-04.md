# Session notes 2026-05-04

What happened in the autonomous-work block while you were sleeping (5:30 AM → ~6:00 AM PT). Read this first when you're back.

## Production-recommended config (TL;DR)

```bash
python scripts/iterative_design_v2.py \
    --seed_pdb $SEED_PDB --ligand_params $LIG_PARAMS \
    --plm_artifacts_dir $WORK/plm_artifacts \
    --position_table  $WORK/classify/positions.tsv \
    --target_k 50 --min_hamming 3 --cycles 3 \
    --strategy annealing \
    --plm_strength 1.25 \
    --consensus_threshold 0.90 --consensus_strength 1.0 --consensus_max_fraction 0.15 \
    --n_term_pad MSG --c_term_pad GSA \
    --design_ph 7.8
```

This is **Sweep B's config** from the autonomous sweep. Best balance found:
- Best diversity: global hamming 56, primary-sphere unique-AAs 6.2/pos
- Best sap_max: 1.24
- Tied-best preorganization strength: 27.98
- Within target charge / pI / clash bands

## The 5-sweep autonomous experiment

Submitted 5 jobs (4 GPU + 1 CPU). Real bug found and fixed mid-sweep.

| Sweep | strategy | consensus | result | run_dir |
|---|---|---|---|---|
| A | constant | 0.90 / 1.0 / 0.15 | fitness −1.901, ham 46.1, prim 2.4 | `..._053529` |
| B | **annealing** | **0.90 / 1.0 / 0.15** | **fitness −1.914, ham 56.0, prim 6.2** | `..._054455-127-pid3231842` |
| C | constant | 0.95 / 0.5 / 0.10 | fitness −1.900, ham 44.4, prim 2.1 | `..._053530` |
| D | CPU test (constant default) | — | fitness −1.865 (1 cycle) | `..._053541` |
| E | annealing + lean resources (cpus=2 mem=8G) | 0.90 / 1.0 / 0.15 | running at 06:00 | TBD |

**Bug found:** Sweeps B and C started in the same second; both wrote to the same `run_dir` and overwrote each other. **Fixed** in commit `8c298cc` — timestamps now include ms+PID. Sweep B was re-submitted as job 14008985.

**CPU pipeline works end-to-end.** ~6.4× slower than GPU per-cycle on the bottleneck (MPNN sample), or ~3.6× slower in wall-clock for typical pipelines (other stages are CPU-bound on both). Fully viable for thousands of jobs without burning slurm GPU priority.

## Codex reviews (both autonomous)

Two codex agent reviews this morning:

### 1. Full codebase review (after all session changes)

**Verdict: nothing to block deployment.** Pipeline internally consistent. Three soft items flagged:

- Expression engine sees unpadded design body while ProtParam sees padded — **deliberate trade-off** documented in stage_seq_filter docstring (per your spec: "sequence-specific, structure-agnostic" calculations only).
- `_normalize_axis` docstring lied about NaN handling — **fixed**.
- `length` field reports padded length — pre-flagged, no consumer affected.

### 2. Efficiency-plan review

**Caught 3 quantitative errors in my draft** (all corrected):

- fpocket cost: I said 0.3 s/design; actual is **~0.9 s/design**. fpocket fraction of total wall is ~41%, not 37%. Upside: parallelization savings ~3× larger than I estimated.
- CPU slowdown: I said 3.6×; actual per-cycle is **~6.4×** (full-pipeline 3.6× was diluted by mixing 1-cycle CPU vs 3-cycle GPU runs).
- cpus=2 + Pool(8) was **internally contradictory**. Resolution: cpus=2 if no parallelism, cpus=8 if parallel; default cpus=4 is safe.

## Bugs found and fixed this session block

1. `8c298cc` — concurrent-job run_dir collision (timestamp ms+PID).
2. `ef6569c` (efficiency plan revision) — quantitative claims corrected per codex.
3. `c4f0555` earlier — termini padding bug (now fixed).
4. argparse `%%` escape — `--consensus_threshold` help string had a literal `%` that broke `--help` under Python 3.14. **Fixed**.

## Documentation deployed (Phase 2)

```
README.md                                118 lines — top-level + Mermaid quickstart
docs/architecture.md                     254 lines — full pipeline diagram, per-cycle data flow
docs/usage.md                            278 lines — slurm/interactive/CPU patterns
docs/cli_reference.md                    329 lines — all 37 flags grouped + presets
docs/metrics_reference.md                  TBD   — agent still running (the big one)
docs/dependencies.md                     323 lines — sif/binaries/data paths/apptainer templates
docs/troubleshooting.md                  185 lines — 10 known issues + strengths/weaknesses
docs/examples/pte_default_run.md         140 lines — production config explained
docs/examples/pte_diverse_run.md          97 lines — exploration config
docs/examples/new_scaffold_setup.md      180 lines — adapting to non-PTE targets
docs/plans/efficiency_plan.md            TBD   — efficiency recommendations
docs/plans/documentation_plan.md          81 lines — what got built (this work)
```

Total new doc text: ~2200 lines so far (excluding the in-flight metrics_reference).

## Phase 3 (efficiency, post-codex revision)

Revised efficiency recommendations after codex pushback:

- **`--cpus-per-task=2-4 --mem=10G`** (was 4/16; observed peak 6.8 GB). Saves ~50% slurm priority cost per job.
- **CPU partition for bulk sweeps** — fully validated, ~6.4× slower per cycle but zero GPU priority impact.
- **fpocket parallelization** (deferred; ~20 LOC + Pool(8)) would save ~100 s/run = ~20% wall time.
- **DFI seed-only refactor** (deferred; ~50 LOC) would save ~16 s/run, fixed-backbone designs only.
- **A4000 fully compatible** (only needs 2-4 GB VRAM); B100/B200 backward-compatible if user has access.

Plan committed at `docs/plans/efficiency_plan.md`. Execution deferred to your direction.

## What's still running at 6:00 AM

- Doc subagent: metrics_reference (the big ~1500-line doc — most important reference)
- Lean-resource sweep E (validating `cpus=2 mem=8G`; ~3 min in)

Will land notifications as those complete. No further code changes pending — codebase is in a stable, reviewed state.

## Recommended things to do when you're back

1. **Skim the doc tree.** README → architecture → metrics_reference (when it lands) → cli_reference. Total ~3000 lines if you read all of them; ~10 min skim.
2. **Decide whether to enable fpocket parallelization** (~20 LOC, ~20% wall-time savings). Defer if you'd rather burn-in current state.
3. **Decide whether to drop GPU resources** (`--cpus-per-task=2 --mem=10G`). Free win on slurm priority.
4. **Decide whether you want a `--propka_final` opt-in.** PROPKA works in universal.sif (160 ms/design) but has parameterization issues with PTE's binuclear Zn + YYE ligand without manual setup. Currently declined.

## Questions I left for you

(From the in-line discussions; collected here for convenience.)

- Are you on Hyak/digs cluster? sbatch partitions assumed `gpu-bf` and `cpu`.
- What's the priority cost / CPU-hour vs GPU-hour ratio?
- For "thousands of jobs" — `submitit`-style scaffolding or just `sbatch --array`?
- Should we wire fpocket parallelization next, or roll with current and revisit later?
