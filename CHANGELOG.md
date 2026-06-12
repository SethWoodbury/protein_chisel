# Changelog

All notable changes to **protein_chisel** are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions use semver.

## [Unreleased]

### Added — Adaptive solubility-bias controller (opt-in, default OFF, byte-identical)
- New closed-loop controller (`src/protein_chisel/sampling/adaptive_bias.py`) that, across
  design cycles, measures the candidate pool's net charge + surface hydrophobicity and steers
  the next cycle's LigandMPNN biases toward target solubility. **Default OFF** → byte-identical
  to before; enable with `ADAPTIVE_BIAS=1` (`--adaptive_bias`).
- **Integral-hold control** that holds the achieved bias once in-band (a near-static plant would
  revert if the bias were released), **reverses on overshoot** (bidirectional), and fires only
  when the pool is **statistically out of target** (t-stat + Wald-bounded fail-fraction + min-N
  gate); per-AA surface down-weight is focused on **way-over-represented** hydrophobics (z>3 vs
  the hydrolase reference). Online two-point gain estimate + wrong-sign freeze for stability.
- **Two actuators, partitioned by safety**: charge → global D/E up / K/R down (composed with the
  existing class-balance `bias_AA`, which wins conflicts); hydrophobicity → per-position
  down-weight **only at `distal_surface`, non-fixed positions** (never the buried core or
  catalytic/binding region). Disjoint AA sets → no double-count; all magnitudes clamped below the
  PLM fusion scale.
- Opt-in `--adaptive_bias_seed_from_input` warm-starts cycle 0 from the input scaffold's own
  GRAVY/charge. Tunable knobs: gain / max_nats / carry / deadband / tmin / fmin / min_n / mode
  (`proportional`|`bangbang`). Env passthrough in `run_chisel_design.sh` (emitted only when set).
- **Input-hydrophobicity warning** (logging only): a seed `GRAVY > +0.4` logs that designs will
  likely fail solubility filters and the controller can only steer the surface, not fix the fold.
- Per-cycle telemetry `cycle_NN/00_bias/adaptive_bias_telemetry.json`. Docs: `docs/adaptive_bias.md`.

### Added / Changed — Stage-2 (PLM precompute) memory efficiency (default byte-identical)
- `low_cpu_mem_usage=True` on the SaProt load (no transient 2× CPU weight copy;
  byte-identical weights). Explicit `gc.collect()` + `torch.cuda.empty_cache()` between
  experts so the previous model is reclaimed before the next loads.
- **Opt-in half precision**: `PLM_DTYPE` / `--plm_dtype {fp32,fp16,bf16}` (default `fp32`)
  roughly halves PLM memory. Changes the logits, so all Stage-2 artifacts are written to
  **dtype-suffixed** files when non-fp32 (`*_log_probs.fp16.npy`, `fusion_bias.fp16.npy`, …)
  — fp16 never reuses an fp32 cache; the driver loads the matching dtype and asserts the
  precompute manifest's `plm_dtype` agrees. ESM-C casts to half *before* the device move
  (so fp16 only allocates the half-size VRAM). `fp32` is the exact prior behavior.
- **Memory warning** (`utils/resources.py`): `detect_available_mem_mb` (SLURM / cgroup /
  /proc, binding cap) + `estimate_plm_footprint_mb` (GPU- vs CPU-aware) + a Stage-2-startup
  `warn_if_plm_mem_tight` that suggests a smaller `SAPROT_MODEL` / `PLM_DTYPE=fp16` when
  tight. Logging-only — never changes behavior.
- Docs: `docs/memory.md`. See for the variant × dtype × memory table.

## [1.0.0] — 2026-06-09

First tagged release. A suite of **opt-in, modular add-ons** toward plug-and-play
experts / metrics / sampling, plus provenance — with a hard contract: **with no new
flags or env, the pipeline behaves exactly as before (byte-identical designs, filters,
ranking, and PDBs).** Every add-on defaults to the prior behavior and is independently
selectable and tunable.

### Added — pluggable experts (PLM fusion)
- `src/protein_chisel/experts/` registry (`Expert` base + `esmc`/`saprot`) and
  `sampling/plm_fusion.fuse_experts` (N-way calibrated product-of-experts; the N=2
  fast path delegates to the untouched `fuse_plm_logits`). `--experts` (default
  `esmc,saprot`) in precompute + driver + shell. *Borrowed from Sebastian (sebols) /
  Joe Mi `fused_mpnn_poe`.*

### Added — decode-time Product-of-Experts (PoE) sampling backend
- `--mpnn_backend {bias,poe}` (default `bias`). `poe` runs `fused_mpnn_poe`
  (`poe_mpnn.sif`) as a **separate host stage** (nested apptainer is blocked), mixing
  context-aware experts (`hermes/esm/e1/vesm/msa/dms`) into per-position log-probs on
  top of our calibrated bias, then the driver scores/ranks that pool (one-shot).
  `sampling/mpnn_backends.py` + `scripts/poe_emit_command.py`; env
  `MPNN_BACKEND/ADDITIONAL_EXPERTS/EXPERT_LAMBDAS/POE_NUM_DESIGNS/POE_TEMPERATURE/POE_HERMES_PROBS`.
  Cluster-validated end-to-end (`esm`, `hermes`). See `docs/poe_backend.md`.
  *Borrowed from Sebastian (sebols) / Joe Mi; HERMES by Visani et al. (Nourmohammad lab, UW).*
- **HERMES** reachable via `ADDITIONAL_EXPERTS=hermes` (on-the-fly from in-container
  weights; optional precomputed CSV via `POE_HERMES_PROBS`).

### Added — pluggable metric registry + selection/gating
- `src/protein_chisel/metrics/` declarative catalog of every `stage_*` metric
  (role/columns/deps/cost; reuses `multi_objective` ranking labels — no duplicate
  registry). `--metrics`/`--filters` (+ `METRICS`/`FILTERS` env) gate the ranking
  basket, optional stages, and which filters may drop designs — byte-identical at the
  default `all`. See `docs/metrics_registry.md`.

### Added — active-site interaction-network growth
- `conserved_hbonds.build_interaction_network` (recursive H-bond/interaction shells);
  `--conserve_hbond_depth/_interaction_types/_shell_decay/--conserve_grow_network`
  (default depth=1 = legacy single-shell conserved-hbond fixing).

### Added — provenance & diagnostics
- `provenance.py` → `provenance.json` per run (experts + versions, fusion version,
  `mpnn_backend`, metric/filter selection, conserve-network, PoE experts/lambdas/commit,
  `chisel_version`).
- `scoring/covariation.py` — APC-corrected MIp covariation/epistasis diagnostic
  (experimental, off by default).
- `sampling/mpnn_with_refresh.py` — PLM-bias refresh orchestrator (rounds=0 = current).
  *Recompute wiring deferred post-1.0 (host-stage multi-round; PoE now covers the
  context-aware-expert use case).*

### Fixed
- Stage-4 His tautomer: map Amber/tautomer codes to Rosetta restypes
  (`HIE/HIS_E→HIS`, `HID→HIS_D`, `HIP→HIS_P`) before `rts.has_name()` — catalytic His
  keeps its protonation state; 0 "unknown tautomer" warnings.
- `scoring/dfi.py`: sort per-class keys so the `dfi__*__<class>` TSV column order is
  deterministic (was PYTHONHASHSEED-dependent).
- HF offline: Stage-2 forces `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` (read-only cache).

### Changed — final output naming/provenance (on by default)
- Final designs renamed `<stem>_chisel_<NN>.pdb`, rank-ordered (best = `_chisel_00`),
  padded to the largest 0-based index width, in a collision-/crash-safe rename.
- `DESIGN_PATH` collapsed to a single `chisel_iterative_design output <abs path>` line
  (upstream provenance + 665/666/667/668/QCB preserved).

### Engineering
- 605 host tests (+2 skip); dual review (independent subagent + codex) on every
  substantive change; cluster integration validation (default byte-identity + each
  opt-in path).
