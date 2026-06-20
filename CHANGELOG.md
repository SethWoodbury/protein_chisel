# Changelog

All notable changes to **protein_chisel** are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions use semver.

## [Unreleased] — diverse-backbone effectiveness (post-1.1.0, opt-in, byte-identical)

Root-cause work after cluster validation revealed two reasons steering under-performed on
hydrophobic seeds: (1) **bias mis-scaling** — LigandMPNN samples `softmax((logits+bias)/T)`,
so a bias of `b` nats shifts odds by `exp(b/T)`; at T≈0.15 the PLM fusion (~1.36 nats) is an
~8,700× *lock* while the charge controller (~0.6 nats, ~55×) is dwarfed (independent codex +
subagent audit). (2) the composition correctors are gated behind survivors that never exist
when the GRAVY band rejects 100% of samples. Empirically, dropping the PLM (`plm_strength=0`)
on a GRAVY=1.34 seed took **GRAVY → −0.50, Ala 27% → 0.5%, hydrophobic 66% → 32%**, and
survivors finally appeared so the composition cap fired.

### Added — seed triage: opt-in PLM auto-skip on a pathological input (`--plm_autoskip_bad_input`)
- New pure, reference-free `sampling/seed_triage.py` (`assess_seed` + `should_skip_plm`): flags an
  input scaffold as pathological when GRAVY > `--plm_autoskip_gravy` (0.4), any single AA ≥
  `--plm_autoskip_max_aa_frac` (0.16), or hydrophobic fraction > `--plm_autoskip_hydrophobic_frac`
  (0.50). When `--plm_autoskip_bad_input` is set (+ env `PLM_AUTOSKIP_BAD_INPUT=1`) and the seed
  trips, the driver forces `plm_strength → 0` **before** building the fusion, so LigandMPNN
  regenerates from structure + fixed residues instead of the PLM amplifying the bad seed. Default
  OFF → byte-identical (imports + work happen only inside the opt-in branch; `--help` needs no PLM).
- New `sampling/bias_scale.py` (`nats_for_odds = T·ln(M)`, `odds_for_nats = exp(b/T)`, odds
  vocabulary) — the temperature-invariant foundation for the bias recalibration.

### Added — CF-1: odds-space controller clamp (`--adaptive_bias_max_odds`)
- First step of the unified multi-objective controller framework (design: `docs/plans/controller_framework.md`).
  The adaptive controller's `|bias|` clamp was a fixed `max_nats=0.6`; because MPNN samples
  `softmax((logits+bias)/T)`, that is 55× odds at T=0.15 but only 20× at T=0.20 — the controller's
  authority silently varied with the temperature schedule. `--adaptive_bias_max_odds X` (env
  `ADAPTIVE_BIAS_MAX_ODDS`) instead clamps in **odds space**: `max_nats := T·ln(X)` per cycle (via new
  `bias_scale.effective_clamp_nats`), so the authority is temperature-invariant and legible (X=2 nudge …
  8 strong). `AdaptiveBiasConfig.max_odds` defaults `None`, `compute_adaptive_bias` gains an optional
  `temperature`; when unset the effective cfg **is** the original cfg → byte-identical (61 controller
  tests + full 839 green). Validated `> 1.0` at parse time. The `step_axis` control law is untouched.

## [1.1.0] — 2026-06-18 — Solubility-steering overhaul (8 opt-in workstreams, byte-identical default)

The whole **solubility-steering campaign** (WS-A..H): make `protein_chisel` actually steer designs
toward solubility — fix the rescue bug that shipped a GRAVY=1.05 / 26%-Ala design as rank-0, add a
corrected SAP, real composition control, an expanded adaptive controller, sampling-core safety rails,
a PLM-refresh toolkit, and a tunnel-lining omit — **every behavior behind a flag/env that defaults to
the prior behavior, so with no new flags the pipeline is byte-for-byte identical**. Each workstream
was built TDD-first and put through independent review (codex + ≥1 subagent), which deferred several
planned-but-redundant/unsound levers (the WS-D composition & SAP axes, the WS-E anti-repeat &
plm_off_mode, the WS-F driver activation) with documented reasons. Host suite: 809 passed.

### Fixed — graded-clash bias Lys/Arg symmetry (⚠️ behavior change, NOT byte-identical vs 1.0.0)
- The always-on `compute_graded_clash_bias` call site passed `bulky_aas="YFWHMR"`, dropping **Lys**
  while keeping **Arg** — even though the function's own default was `"YFWHMRK"` and K/R are the same
  length tier (Cb→NZ ~5.5 Å, Cb→CZ ~6 Å). K can collide with a fixed catalytic atom just as R can, so
  it now gets the same graded clash down-weight. Both the function default and the call site reference a
  single `_CLASH_BULKY_AAS = "YFWHMRK"` constant (guard-tested) so they cannot drift apart again. **This
  is the one intentional exception to the "byte-identical default" rule above:** the clash bias has no
  flag, and `base_bias = base_bias + clash_bias` is the fusion baseline carried into *every* cycle (not
  just cycle 0), so this shifts design output at clash-prone positions where Lys was previously
  un-penalised. Folded into the unreleased 1.1.0. Companion to the WS-G bulky-set fix.

### Added — WS-G omit tunnel-lining (opt-in/experimental, default OFF, byte-identical)
- **`--omit_tunnel_lining` (+ `--omit_tunnel_lining_aas`, default `FHKRWY`) / `OMIT_TUNNEL_LINING` /
  `OMIT_TUNNEL_LINING_AAS`** — hard-omit bulky/aromatic AAs at the seed's tunnel-lining positions to
  keep the substrate channel open from cycle 0. New pure `_build_tunnel_lining_omit` +
  `_read_seed_tunnel_lining` helpers; the lining set is the seed `is_tunnel_lining` annotation — now
  the **single source of truth**, shared with WS-D's surface scope (WS-D's inline read was refactored
  to call `_read_seed_tunnel_lining`). Merged into `omit_AA_per_residue` only inside the flag's `if`
  (merge re-sorts AA strings, so even a `{}` merge isn't a guaranteed no-op — codex), so a no-flag run
  is byte-identical. **Default = the throat's own bulky-blocker set:** new shared
  `tunnel_metrics.bulky_blocker_aas(0.70)` derives the set from `_BLOCKER_WEIGHT >= 0.70`, so it's one
  source of truth with the throat-feedback bias — **`FHKRWY`** (aromatics W/F/Y/H plus the long charged
  R/K). Lysine and Arginine ARE bulky here (Lys Cb→NZ ~5.5 Å, Arg ~6 Å — genuine channel constrictors,
  same classification the throat applies); hard-omitting them at lining positions also nudges net charge
  negative, which *agrees* with WS-D's charge axis. NOT the plan's draft `FILMVWYA` — Alanine is excluded
  (small, can't constrict; composition is WS-C's job) and the medium hydrophobics I/L/M/V are left to the
  throat-feedback controller's capped/decaying pressure rather than a permanent hard ban; it warns when
  both `--omit_tunnel_lining` and
  `--throat_feedback` are on (the hard omit shadows the soft bias at lining∩throat positions).
  Off by default — it's the bluntest of the channel levers (complementary to, and overlapping with,
  the soft throat-feedback). Catalytic/fixed positions are never omitted. 10 host tests; the actual
  fpocket lining run is cluster-only.

### Added — WS-F PLM-bias refresh toolkit (orchestrator + helpers wired + tested; driver activation deferred)
- The previously-dead `sampling/mpnn_with_refresh.py` is now a complete, host-tested refresh toolkit:
  the `run_with_refresh` orchestrator (already present) plus two new **pure** helpers —
  `choose_inband_representative` (median-fitness sequence among in-band, seq-filter-passing survivors;
  reuses the shared solubility band) and `refuse_esmc_only` (re-fuse fresh ESM-C marginals against the
  *unchanged seed SaProt* marginals — SaProt is structure-aware, so refreshing it on the drifted
  sequence would erase the 3Di signal; the same `fusion_cfg` flows `--plm_strength` /
  `--plm_class_strength` through). New shared `scoring/solubility.within_solubility_band` (single
  source of truth; the WS-A veto's `_within_solubility_band` is now a thin wrapper — byte-identical).
- **Driver activation deliberately DEFERRED** (design debate, codex + 2 subagents): the ESM-C recompute
  on a drifted sequence is a *between-stage* operation — the design sif where the driver runs has no
  `torch`/`esm` and nested apptainer is blocked, so a `--plm_refresh_rounds` flag would be an
  always-no-op in the current topology ("misleading"). Activating the refresh needs a PoE-style host
  stage that re-precomputes ESM-C mid-run, and a cluster ablation to justify the cost (one full
  masked-LM recompute per round). The toolkit + helpers are ready for that follow-up; provenance
  already reserves `plm_refresh_rounds`. (WS-F is the *only* lever that re-grounds the static seed
  prior, so it is NOT redundant with WS-C/D/E — it is the missing axis, just unverifiable on this host.)

### Added/Changed — WS-E sampling-core safety (opt-in, default byte-identical)
- **`--bias_total_clamp NATS` / `BIAS_TOTAL_CLAMP`** — bound the **effective** per-`(position, AA)`
  sampling bias (`bias_per_residue` + the separately-applied global `bias_AA`) to `±NATS` via
  `_clamp_bias_total` (`clip(bias_k + g, ±N) − g`, parsed from the final serialized `bias_AA`).
  Today nothing caps the *sum*: the consensus (+2.0, uncapped) and PLM peak (~2.5) stack to ~4.5 nats
  → ~10¹³× odds at T≈0.15, a de-facto hard lock that defeats the diversity injection. Applied to the
  bias the sampler sees (the `bias.npy` diagnostic stays the un-clamped per-position fusion bias).
  Default `None` → byte-identical; suggested production ~3.0 (an overflow guard, not a regularizer).
- **`--sampling_temperature_floor T` / `SAMPLING_TEMPERATURE_FLOOR`** — raise any cycle's sampling
  temperature to `max(cycle_T, T)`, applied once after the schedule is built so the sampler and the
  logged temperature agree. At T≈0.15 a 0.5-nat bias is ~28× (near-deterministic); ~0.3 restores
  genuine multinomial diversity. Overrides annealing; no effect under the PoE backend (warned).
  Default `None` → byte-identical.
- **PLM strength ↔ fitness decouple** (byte-identical for `--plm_strength > 0`) — at `--plm_strength 0`
  the seed-marginal fitness used to collapse every design to a 0-tie (the fused mean `(β·lp_e+γ·lp_s)/
  (β+γ)` with `β=γ=0`), silently zeroing the weight-2.0 fitness objective. Now, **only at strength 0**,
  the fitness ranking substitutes the structural **strength-1.0** weights
  (`plm_fusion.decoupled_fitness_weights`) so the rank is meaningful (run PLM-off-for-sampling and
  still rank by PLM naturalness). For `--plm_strength > 0` the ranking **reuses the exact
  strength-scaled fusion weights** — so the fitness TSV column is bit-for-bit identical to before
  (recomputing the scale-invariant mean at 1.0 would have jittered ~25% of rows by 1 ULP; the
  independent review caught this). The sampling bias always honors `--plm_strength`; `>2`-expert runs
  are unchanged. Rank-invariance + the strength-0 rescue are pinned by a mixed-class test.
- **`--plm_class_strength K=V,…` / `PLM_CLASS_STRENGTH`** — absolute per-class overrides of the
  PLM-fusion `class_weights` (e.g. `distal_surface=0.3,primary_sphere=0.0`); `--plm_strength` still
  multiplies globally on top. Unknown class names and non-finite/negative values are rejected
  (`_parse_plm_class_strength`). Default `""` → byte-identical. Also: `--plm_strength` now rejects a
  non-finite value (the prior `<0` / `>5` checks let NaN through).
- **Design debate (codex + two subagents) DEFERRED two planned features as redundant:**
  - *`--anti_repeat_bias`* — the per-cycle bias is regenerated from `survivors_prev` (no cross-cycle
    composition memory), so "over-represented in the running pool" ≈ "over-represented in this cycle's
    survivors", already handled by WS-C's `--aa_fraction_cap` (a hard-omit, reference-free, *stronger*
    anti-mode-collapse lever) + `suppress_all_overrep`; and a naive `bias_AA` anti-repeat is swallowed
    by the class-balance-wins merge — the same structural reason WS-D deferred its composition axis.
  - *`--plm_off_mode`* — after the decouple it is identical to `--plm_strength 0` (both zero the
    sampling bias and keep the decoupled rank); a second flag aliasing an existing one invites the
    no-op/version-skew bug class. The `0.0` behavior is documented on `--plm_strength`.
- 8 new host tests; full host suite 790 passed.

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
