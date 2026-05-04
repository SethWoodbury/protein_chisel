# Efficiency / resource minimization plan

**Status:** Planning only. Execute under user direction.
**Author:** Claude (Opus 4.7), 2026-05-04 ~05:55am.

## Revisions after codex review (2026-05-04 06:00am)

Codex flagged 3 quantitative errors in the original plan; corrected here:

- **fpocket cost:** plan said "0.3 s/design"; actual is **~0.9 s/design**
  (76s for 79 designs in cycle 0). Real fpocket fraction of total wall
  is **~41%, not ~37%**. Upside: fpocket parallelization savings are
  ~3× larger than the original estimate (~100s/run, not ~30s).
- **CPU slowdown:** plan said "3.6×"; apples-to-apples per-cycle
  measurement is **~6.4×** (GPU cycle-0 MPNN = 83s for n=500;
  CPU cycle-0 MPNN = 535s for n=500). The 3.6× number divided
  full GPU run (3 cycles, 1200 samples) by full CPU run (1 cycle,
  500 samples) — different cycle counts polluted the ratio.
  CPU is still viable, just slower than originally claimed.
- **Resource contradiction:** plan said `--cpus-per-task=2` AND
  recommended fpocket `Pool(8)` parallelization. Inconsistent.
  Resolution: cpus=2 if no fpocket parallelism; cpus=8 if
  parallelism is enabled. Use cpus=4 as a safe default.
- Memory headroom: 8 GB has only ~17% slack on a 6.8 GB peak.
  Bumping to **--mem=10G** gives 47% slack — safer.

## Empirical baseline (this session)

GPU sweep (g2157, A100-style): 3 cycles, 1200 samples, full pipeline → **8.2 min wall**.
CPU sweep (cpu node, 8 cores): 1 cycle, 500 samples, full pipeline → **9.8 min wall**.

Per-cycle: GPU 2.7 min/cycle, CPU 9.8 min/cycle. **CPU is ~3.6× slower** — not 30× as feared. For "thousands of jobs at low slurm-priority" workflows, CPU is fully viable.

## Where time goes per cycle (best estimate from logs)

| Stage | GPU (s) | CPU (s) | Notes |
|---|---|---|---|
| stage_sample (fused_mpnn) | ~80 | ~470 | LigandMPNN forward pass; CPU-bound on CPU |
| stage_restore_pdbs | ~5 | ~5 | stdlib parsing |
| stage_seq_filter | ~1 | ~1 | numpy + protparam (sub-ms each) |
| stage_struct_filter | ~10 | ~10 | freesasa fallback path; clash check |
| stage_fitness_score | <1 | <1 | numpy gather |
| stage_fpocket_rank | ~60 | ~60 | fpocket binary, CPU-bound on both |

Summary: **fpocket dominates the GPU pipeline** (~37% of cycle time). On CPU, fused_mpnn dominates (~80%). Optimizing fpocket benefits both.

## Slurm resource recommendations

### GPU jobs (current default)

`#SBATCH --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00`

**Adjustments worth trying:**

1. `--cpus-per-task=2` (was 4). Only fpocket + numpy use CPU heavily; MPNN is GPU. Save 2 CPU slots per job. **Test in next batch.**

2. `--mem=8G` (was 16G). Observed peak was ~6.8 GB. 8 GB has slack. Save 8 GB per job. **Test in next batch.**

3. `--time=01:00:00` (was 02:00:00). Real runs land in 8–10 min. 1 hour is generous. **Test in next batch.**

4. Add `--gres=gpu:1` constraints by GPU type: `--gres=gpu:rtx_a4000:1` or `--gres=gpu:gtx_3090:1`. Smaller cards are fine — fused_mpnn uses ~2-4 GB VRAM. **Verify on RTX A4000.**

### CPU jobs (new)

For runs you don't need fast turnaround on:

`#SBATCH --partition=cpu --cpus-per-task=8 --mem=24G --time=04:00:00`

Empirical: 1 cycle ≈ 10 min, 3 cycles ≈ 30 min. 4 hours is generous. Slurm priority NOT impacted by GPU usage.

### A4000 / B-series compatibility

**A4000** (16 GB VRAM, ~13 TFLOPS FP32): yes, fully compatible. fused_mpnn uses ~2-4 GB and is FP32; A4000 is overkill for memory and adequate for compute. **Use this when GPU partition allows.**

**B-series**: NVIDIA hasn't released a "B4000" SKU. If user means the **B100/B200** Blackwell architecture (data-center GPUs), yes — backward compatible with PyTorch 2.x; fused_mpnn would just be wasted on it. If they mean the **AMD Radeon Pro B-series**, no — fused_mpnn requires CUDA, not ROCm.

What we ACTUALLY ran on this session: g1105, g2157, g2314, g2505 nodes, partition `gpu-bf` (A100s). Standard.

## Code-level optimizations (in priority order)

### 1. fpocket parallelization (HIGH impact)

fpocket runs **per design** sequentially in `stage_fpocket_rank`. ~60s for 200 designs = 0.3 s/design. Could parallelize trivially with `multiprocessing.Pool(8)`. **Estimated savings: 50% of fpocket time = ~30 s/cycle = ~90 s/run.**

Risk: fpocket binary is single-threaded; multiple instances should be fine. Each call writes to its own tmp dir.

### 2. Skip per-design DFI (already noted)

DFI was found to be design-invariant for fixed-backbone designs. ~80 ms × 200 designs = 16 s/cycle wasted. **Refactor: compute once at startup, reuse the value.** Trivial.

### 3. fused_mpnn batch_size on CPU (MEDIUM impact)

CPU runs use `batch_size=10`, `number_of_batches=50`. PyTorch CPU is more efficient with smaller batches (less memory pressure). Could try `batch_size=4 number_of_batches=125`. **Estimated savings: maybe 10-20% of CPU sample time.**

### 4. PLM artifact precompute caching (LOW impact)

Currently re-fuses at runtime. The expensive part (ESM-C / SaProt forward passes) is already cached. Re-fusion is ~5 ms. Not worth optimizing.

### 5. PositionTable migration on legacy parquet (MEDIUM impact, but rare)

Currently re-classifies at every load if `class` is legacy. ~50 ms one-time per run. Not worth optimizing unless we run thousands of pipelines on the same dataset, in which case migrate the parquet once.

### 6. Strip dead code (clean up)

`stage_diverse_topk` legacy function is now unused (replaced by inline `select_diverse_topk_two_axis`). Remove.

## Resource sweet spot for production (recommended)

For 1× design run:
```
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2     # was 4
#SBATCH --mem=8G              # was 16G
#SBATCH --time=00:30:00       # was 02:00:00
```

**Cuts slurm priority cost ~50%** (mem and CPUs each 50% lower).

For 1000× design runs (low-priority):
```
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
```

**Zero GPU usage** → GPU priority preserved for time-sensitive work. Pipeline takes ~3.6× longer per job but you can run 8-16× more concurrent jobs in the cpu partition.

## Open questions for the user

1. Are you on Hyak/digs cluster or another? cpu/gpu-bf partitions assumed; verify partition names.
2. What's the priority cost / CPU-hour vs GPU-hour ratio? That dictates the GPU/CPU split sweet spot.
3. For the "thousands of jobs" workflow — is there a reasonable scaffolding tool like `submitit` or just plain `sbatch --array`?
4. Do you want me to actually implement fpocket parallelization + DFI seed-only refactor next?

## Estimated impact summary

| Change | Wall-time saving | Slurm-priority saving | Implementation cost |
|---|---|---|---|
| `--cpus-per-task=2 --mem=8G` | none | **~50% per job** | 0 (sbatch edit) |
| fpocket parallel (Pool) | ~30s/run = 6% | none | ~20 LOC |
| DFI seed-only | ~16s/run = 3% | none | ~50 LOC |
| Use A4000 GPU class | none | partition shift; possibly +priority | 1-line sbatch |
| CPU partition for non-urgent | none | **100% GPU-priority preserved** | sbatch template |

**Headline recommendation**: drop CPU/mem requests on GPU sbatch immediately (free win). Move bulk runs to CPU partition. Defer fpocket parallel + DFI refactor until user prioritizes.
