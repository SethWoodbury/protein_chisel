# Changelog

All notable changes to **protein_chisel** are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions use semver.

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
