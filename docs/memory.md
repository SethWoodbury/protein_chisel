# Memory efficiency (PLM precompute)

The pipeline's memory peak is **Stage 2** (`precompute_plm_artifacts.py`, in `esmc.sif`),
which runs the per-position masked-LM marginals for **ESM-C** and **SaProt**. SaProt-1.3B
(`saprot_1.3b`, the default) is the single heaviest model.

## What's already efficient

- **Experts run one at a time** (`load → compute → free`): ESM-C loads, computes its
  `(L,20)` log-probs, and goes out of scope **before** SaProt loads. The two models are
  **never co-resident** — peak ≈ `max(ESM-C, SaProt)`, not the sum. The precompute loop
  also does an explicit `gc.collect()` + `torch.cuda.empty_cache()` after each expert so
  the previous model's allocations are reclaimed immediately.
- **Stage 2 is its own process** (`apptainer exec esmc.sif …`) that **exits** when done,
  releasing everything. **Stage 3 (the design driver) never loads the PLMs** — it reads
  the small precomputed `.npy` artifacts. So a SaProt OOM can only happen in Stage 2.
- **`low_cpu_mem_usage=True`** on the SaProt load streams the checkpoint into pre-allocated
  tensors (no transient 2× CPU copy), saving load-time host RAM. **It requires the
  `accelerate` package**; when that's absent (e.g. the current `esmc.sif`) the loader falls
  back to the standard path automatically — same weights/logits, just without the saving.
- **Per-artifact cache**: re-running the same scaffold skips model loading entirely.

## Measured peak (default `esmc_600m` + `saprot_1.3b`, GPU `gpu-bf`)

| Metric (L=120 probe) | fp32 | fp16 |
|---|---|---|
| peak VRAM (`max_memory_allocated`) | ~5.3 GB | ~4.7 GB |
| peak host RSS | ~6.9 GB | ~9.3 GB* |

Production-length (L≈200–280) host MaxRSS: ~8 GB on GPU, **~14–15 GB CPU-only** (SaProt-1.3B
float32 in host RAM). So on a **GPU partition** (recommended) `--mem=22g` is comfortable;
CPU-only is where host RAM is tight.

> \* **Important:** `fp16`'s benefit is **VRAM**, not host RAM. Without `accelerate`, the
> `torch_dtype=fp16` load materializes the fp32 checkpoint then casts (double-buffer), so the
> host-load transient is *higher* than fp32. Use `fp16`/`bf16` when **GPU VRAM** is the binding
> constraint (smaller GPU / larger model / longer sequence); installing `accelerate` in the
> container would recover the host-RAM saving via `low_cpu_mem_usage`.

## Knobs to reduce memory

| Knob (env / CLI) | Effect | Byte-identical? |
|---|---|---|
| `SAPROT_MODEL=saprot_650m` (or `saprot_35m`) | smaller model, less memory + faster | No (changes results — a model choice) |
| `ESMC_MODEL=esmc_300m` | smaller ESM-C | No (model choice) |
| `PLM_DTYPE=fp16` (`--plm_dtype fp16`) / `bf16` | **lowers GPU VRAM** (half-precision weights); host RAM only drops if `accelerate` is present | **No — opt-in.** Default `fp32` is byte-identical |
| `SAPROT_MODEL=saprot_650m` is the most reliable **host-RAM** reducer | smaller model in RAM | No (model choice) |
| run on a GPU partition | model in VRAM, lower host RAM | Yes |

### `PLM_DTYPE` (opt-in half precision)

`PLM_DTYPE=fp16` (or `bf16`) runs the PLM forward passes in half precision, roughly halving
the model footprint. The logit reduction still upcasts to float, but the model output
differs slightly from fp32, so it is **opt-in** and the default stays **fp32 (byte-identical)**.

To keep fp16 and fp32 from contaminating each other, all Stage-2 artifacts are written to
**dtype-suffixed** files when non-fp32 — `esmc_log_probs.fp16.npy`, `saprot_log_probs.fp16.npy`,
`fusion_bias.fp16.npy`, … (fp32 keeps the legacy unsuffixed names). An fp16 run therefore
never reuses an fp32 cache (or vice-versa), even in the same `out_dir`, and the driver loads
the matching dtype (and asserts the precompute manifest's `plm_dtype` agrees). `PLM_DTYPE` is
passed to **both** Stage 2 (which writes the artifacts) and Stage 3 (which loads them).

> `bf16` is preferred over `fp16` on CPU (native CPU fp16 compute is limited); both are fine
> on GPU. `low_cpu_mem_usage` applies to SaProt only — ESM-C's loader has no equivalent kwarg.

## Memory warning

At Stage-2 startup the driver estimates the configured PLMs' footprint vs the detected
budget (`SLURM_MEM_PER_NODE` / `SLURM_MEM_PER_CPU` / cgroup `memory.max` / `/proc/meminfo`,
whichever is the binding cap) and **logs a WARNING** suggesting a smaller `SAPROT_MODEL`
and/or `PLM_DTYPE=fp16` if it looks tight (`utils/resources.py::warn_if_plm_mem_tight`). It
**only warns** — it never changes the run. If you see an OOM/SIGKILL in Stage 2, this warning
(and the table above) tells you what to dial down.
