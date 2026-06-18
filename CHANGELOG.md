# Changelog

All notable changes to **protein_chisel** are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions use semver.

## [Unreleased]

### Added/Changed — WS-D adaptive-controller expansion (opt-in, default byte-identical)
- **`non_tunnel_surface` scope (`--adaptive_surface_sasa_gate` / `ADAPTIVE_SURFACE_SASA_GATE`)** —
  the controller's surface hydrophobic down-weight no longer acts only on `distal_surface`
  (a ligand-distance shell). New pure `sampling/adaptive_bias.surface_scope(...)` returns an
  `(L,)` membership mask = SASA fraction ≥ gate AND class in a vetted exposed set
  (`{distal_surface, nearby_surface}`, configurable) AND not tunnel-lining / throat-band / fixed —
  a *superset* of `distal_surface` (adds back exposed `nearby_surface`, drops the 10 Å gate)
  *minus* the tunnel mouth. This is the user's "steer what I can see by eye" surface. The mask
  is structure-invariant (built once from the seed; tunnel-lining read from the seed annotation,
  throat-band source not yet wired). `build_surface_delta(surface_mask=…)` consumes it;
  **`sasa_gate=None` reproduces the legacy `distal_surface` + SASA>0 gate byte-for-byte** (proven
  over 200 random trials). **Review fix (SEV-1):** the scope POSITIVELY gates on the vetted exposed
  classes rather than negatively excluding the active site — a negative gate would sweep in any
  *other* class (legacy `buried`/`first_shell`, or an unknown class on a new scaffold) whose total-
  SASA cleared the gate, down-weighting a load-bearing core hydrophobic; positive gating closes that
  (and the total-residue-SASA proxy caveat for `nearby_surface` is documented — restrict to
  `{distal_surface}` for a conservative run).
- **Configurable charge band (`--adaptive_charge_band LO,HI` / `ADAPTIVE_CHARGE_BAND`, e.g.
  `-15,-5`)** — overrides the controller's net-charge band, sets the target to the band MIDPOINT,
  and **nulls the charge axis's precomputed gap columns** so the fail-fraction is evaluated on raw
  net charge: those gaps were computed against the *cycle filter* band, so a custom adaptive band
  must fall back to raw-value evaluation (codex). Validated finite `lo < hi`; bad band / unknown
  axis fail fast at startup (the per-cycle controller is defensively wrapped, so otherwise they'd
  silently degrade the run).
- **Axis selector (`--adaptive_bias_axes` / `ADAPTIVE_BIAS_AXES`)** — `default_axes(axes=…)` returns
  an ordered subset of the registry (default `charge,surface_hydrophobicity`); the extension point
  for future registry entries. Rejects an empty selection and duplicates (codex: a repeated axis
  would double-stack one actuator past its clamp). `--adaptive_charge_band` requires exactly two
  finite fields (`_parse_charge_band_arg`).
- **Multi-pool plumbing** — `ControlAxis.pool_key` (default `"seq"`) + `compute_adaptive_bias(pools=…)`
  (back-compat: a lone `pool_df` is `{"seq": pool_df}`) route each axis to the stage pool it measures.
  This is the clean, tested enabler for a future struct-stage axis without touching the control law.
- **Dedup** — `KD_HYDROPHOBICITY` is now imported from the single source of truth in `scoring/sap.py`
  (re-exported for back-compat); the duplicated dict the 2026-06 audit flagged is gone. Byte-identical
  (same values; `kd.mean()` arithmetic unchanged).
- **Design debate (codex + two independent subagents) deliberately DEFERRED two of the planned axes:**
  - *composition axis* — a scalar integral controller is the wrong model for composition (the offending
    AA changes each cycle; holding a scalar `u` while recomputing 20-dim weights is incoherent; reversal
    / secant gain are undefined). And per-cycle over-representation is **already corrected** by WS-C's
    class-balance + `suppress_all_overrep`; the `merge_bias_AA_strings` policy (class-balance wins) makes
    a naive composition axis vanish. So nothing sound to add — composition stays in WS-C.
  - *SAP axis* — `sap_corr` ≡ the GRAVY surface actuator (identical AA set, same `build_surface_delta`)
    → an independent SAP axis would double-push; `sap_corr_*` lives only on the struct-stage pool (the
    controller measures seq-stage); struct-survivor counts fall below `min_n` exactly when solubility is
    failing; and the corrected-SAP threshold is uncalibrated. Deferred pending calibration + an
    actuator-group controller; the `pool_key`/`pools` plumbing makes it a one-axis follow-up.
- The control law (`step_axis`, gain estimate, merge, wald gate) is **untouched** → the convergence /
  hold / anti-overshoot suite cannot regress. With `--adaptive_bias` absent (or on with no WS-D flags),
  the run is byte-identical (independently verified: codex + two subagents, 6000+ fuzz trials, the
  default registry compared field-by-field vs HEAD). Their findings — the SEV-1 scope leak, the
  duplicate/empty axis selector, and the charge-band field count — are folded in above. 22 new host
  tests; full host suite 784 passed.

### Added — WS-C composition control that actually corrects (opt-in, default OFF, byte-identical)
- Three independent opt-in levers attack the over-/single-AA-representation failure
  mode (the shipped rank-0 design was 26% Ala / 20% Leu); each defaults to current
  behavior so a no-new-flag run is byte-for-byte identical.
- **`--composition_suppress_all_overrep` / `COMPOSITION_SUPPRESS_ALL_OVERREP=1`** —
  the per-cycle class-balanced `bias_AA` now down-weights **every** over-represented
  member of an AA class (`z > --balance_z_threshold`), not just the single class
  maximum. The legacy path touches only the class max, so when Alanine is the
  hydrophobic-aliphatic max an also-over-represented Leucine **escaped** correction
  entirely; it no longer does. The property-conserving within-class swap up-weight of
  the most under-represented partner is preserved (attached to the class-max
  down-weight). Singleton classes (P, G) are unaffected — their only member already
  *is* the class max. Lives in `expression/aa_class_balance.py` behind a new
  `suppress_all_overrep=False` kwarg (off-path is the legacy code verbatim).
- **`--aa_fraction_cap FRAC` / `AA_FRACTION_CAP`** — hard-omit any amino acid whose
  fraction in a cycle's survivor pool is `>= FRAC` (e.g. 0.15) at every **non-fixed
  designable** position the next cycle, bounding runaway single-AA over-representation.
  Recomputed each cycle from that cycle's survivors, so an AA is re-allowed once it
  falls back under the cap; catalytic/fixed positions keep their identity. Reference-
  free (caps on the raw observed fraction, not a per-class z) so it generalises to any
  scaffold/objective. New pure `expression/aa_composition.over_cap_aas` kernel +
  testable `_build_fraction_cap_omit` driver helper, merged into the existing
  per-residue omit (`None` cap / nothing-over-cap → empty dict → byte-identical merge).
  **Safety (independent-review):** validated as a finite fraction in `(0, 1]`; a cap so
  low it would leave a position with fewer than 3 sampleable AAs is **skipped** that
  cycle with an ERROR — never producing an all/most-AA omit, which fused MPNN encodes
  as equal `-1e8` logits and would then sample *uniformly from the "forbidden" set*.
- **`--composition_soft_bias` (+ `--composition_soft_bias_nats`, default 0.5) /
  `COMPOSITION_SOFT_BIAS` / `COMPOSITION_SOFT_BIAS_NATS`** — activate the previously
  **dead** per-residue SOFT_BIAS tier of the expression engine (long-hydrophobic-
  stretch, KR-near-catalytic-on-helix, polyproline, repetitive-segment, …) as a
  negative per-`(position, AA)` bias added to each cycle's sampler bias in the same
  additive slot as the throat/adaptive deltas. The map is **pool-derived per cycle**
  (`expression/engine.aggregate_pool_soft_bias`): the engine is re-evaluated on the
  cycle's survivors and a `(position, AA)` liability is applied only if it recurs in
  `>= 50%` of survivors (support gate) — so the tier tracks the liabilities the
  *designs* introduce as the pool drifts (polyproline/repeat/hydrophobic-stretch are
  sequence-determined; a seed-only map is blind to them), not just the WT's. The seed
  map bootstraps cycle 0. **Whole-protein composition hits are excluded** via a new
  `soft_bias_per_residue(max_span_frac=…)` span filter — those are the suppress-all /
  fraction-cap levers' job, and including them would freeze a whole-protein single-AA
  ban. Aggregate rules (dibasic-motif-count, composition-out-of-distribution,
  methionine-overrepresented) additionally self-declare `metadata["aggregate"]`, so
  their region-envelope hits are dropped from the per-residue tier regardless of span
  (a dibasic-count hit spans first-to-last motif and would otherwise bias every
  intervening non-motif position). Pure `soft_bias_to_bias_array` helper translates
  `{position → AAs}` into an
  `(L, 20)` downweight in the canonical PLM/LigandMPNN column order (a test locks
  `AA_ORDER_REF == plm_fusion.AA_ORDER`; repeated AA letters de-dup). **Magnitude
  (independent-review):** the bias is added in **logit space** (before the softmax
  temperature divide), so the effective odds penalty is `exp(nats / T)`; at the
  pipeline's `T≈0.15–0.20` the default **0.5** nats is ~12–28× (a firm nudge near the
  adaptive controller's ~0.6 clamp), where the initially-chosen 1.5 would have been a
  ~1800–22000× near-hard ban. Magnitude validated finite `>= 0` (a NaN would poison the
  sampling softmax).
- Under the one-shot **PoE** backend (`--mpnn_backend poe`) the suppress-all and
  fraction-cap levers act on a survivor pool that never exists, so they are explicit
  no-ops and now log a WARNING (the soft-bias still applies via its cycle-0 seed
  bootstrap).
- Reviewed across three rounds by codex + two independent subagents; every P1/P2/P3 is
  folded in above: the fraction-cap all-omit (incl. the post-merge cap × structural-omit
  interaction, guarded by `_enforce_min_sampleable_after_cap`, which reverts the cap and
  re-verifies the structural base), the soft-bias magnitude, seed-vs-pool source, the
  aggregate-rule over-application, and per-survivor evaluation robustness.
  51 new host tests (24 unit + 27 integration); full host suite 762 passed.

### Added — Corrected SAP + shared `scoring/sap.py` module (opt-in, default OFF, byte-identical)
- New `src/protein_chisel/scoring/sap.py` — single source of truth for the Kyte-Doolittle scale,
  Tien max-SASA, the 3→1 map, and a pure `sap_neighborhood_metrics` reduction. Removes the dicts
  duplicated between `iterative_design.py` and `sampling/adaptive_bias.py` (per the 2026-06 audit)
  and gives the controller + driver one place to evolve the hydrophobicity model.
- `_compute_sap_proxy` now uses the shared module — the legacy `sap_*` columns are **byte-identical**
  (a reference test locks the reducer's arithmetic). `--sap_corrected` / `SAP_CORRECTED=1`
  (default OFF) additionally emits `sap_corr_{max,mean,p95}`: a **centered, zero-clamped** KD weight
  (`max(0, KD − mean)`) so exposed *polar* residues no longer cancel hydrophobic ones and **alanine
  surfaces register** (raw KD scores Ala only +1.8; centered +2.29) — the two blind spots that let a
  GRAVY=1.05 / 26%-Ala design read sap_max≈17. Threaded worker → `stage_struct_filter` → `run_cycle`
  → driver → shell. Rescued backfill rows carry NaN `sap_corr_*` (not re-scored). 9 host tests.
  Independently reviewed (codex): confirmed legacy `sap_*` bit-identical (500-case float-hex match)
  and caught a worker-tuple arity crash — `_build_input_reference_row` (input-reference scoring, ON
  by default) still passed the old 7-tuple to the 8-field worker → `ValueError` on every default run;
  fixed by threading `sap_corrected` to that second producer too. (Foundation for the planned
  per-residue SAP ControlAxis; see `docs/plans/solubility_steering_plan.md`.)

### Fixed / Added — Hard solubility veto in final selection (opt-in, default OFF, byte-identical)
- **Bug:** deferred-rescue/backfill re-scored fallback candidates through struct/tunnel/fpocket
  but **never re-applied the GRAVY/charge seq band**, so a `passed_seq_filter=False` design
  (observed: GRAVY=1.05, the user's shipped rank-0) was bucketed `rescued_final_filters` and
  shipped as the best design. `selection__hard_final_filter_passed` is fpocket-druggability only
  (misleading name) and read `True` for it.
- **Fix (opt-in):** `--ship_solubility_veto` / `SHIP_SOLUBILITY_VETO=1` (default OFF → byte-identical)
  applies a modular `_apply_solubility_veto` step at both final-selection branches — *before* the
  row-count bookkeeping, so a legitimate veto drop is never mis-logged as a PDB-export failure —
  dropping any design outside the final cycle's **actual** `stage_seq_filter` band *before* any PDB
  is copied. Bounds are strategy-correct (GRAVY = `args.gravy_*` under `constant`, `cyc.gravy_*`
  under `annealing`; charge = final `CycleConfig` band), so the veto matches what the seq filter
  enforced. Pure helper `_within_solubility_band` mirrors `stage_seq_filter` exactly (charge bounds
  exclusive on `net_charge_full_HH`, GRAVY inclusive; missing/NaN fail closed). Adds a truthful
  `selection__solubility_passed` column. May ship fewer than `target_k` (intended; the
  backfill-underfill ERROR is suppressed when the veto is active). `_write_final_topk_artifacts`
  stays a pure writer (single responsibility). Robust env parsing (`false`/`off`/`0` all disable).
  15 host tests. Independently reviewed (subagent + codex). Docs: `docs/backfill_rescue.md`.

### Added — Configurable design-name suffix (`CHISEL_SUFFIX` / `--design_token`, default byte-identical)
- The shipped-PDB filename token is now configurable: `CHISEL_SUFFIX=chiseli2` →
  `<stem>_chiseli2_NNN.pdb` (env passthrough in `run_chisel_design.sh` →
  `finalize_design_names.py --design_token` → `finalize_names.finalize_design_names`).
  Default `chisel` is **byte-identical**. Token must be non-empty alphanumeric (no `_`/`.`)
  so the trailing-index strip stays unambiguous; the strip also recognises the legacy
  `chisel` token, so a run minting `_chisel_<idx>` ids renames cleanly to a custom token and
  a same-token re-run is idempotent. An input stem that itself contains `_chisel_62` keeps it
  (only the trailing design index is replaced). Removed the now-unused module-level
  `_CHISEL_RE` (per-call regex built from the token).

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
