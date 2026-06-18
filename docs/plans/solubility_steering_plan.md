# Solubility-Steering Overhaul — Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` (inline) for the driver-touching
> workstreams (they edit the same 6800-line `scripts/iterative_design.py` and must not be
> parallelized), and `superpowers:subagent-driven-development` for the isolated modules
> (`finalize_names`, `adaptive_bias`, `mpnn_with_refresh`). Steps use `- [ ]` checkboxes.

**Goal:** Make `protein_chisel` actually steer designs toward solubility (net charge in a
configurable band, low/corrected SAP, controlled GRAVY, no single-AA over/under-representation),
fix the bug that ships solubility-failing designs, and break the static-hydrophobic-seed lock-in —
all **opt-in and byte-identical by default**.

**Architecture:** Every change is gated behind a new flag/env that defaults to *current behavior*.
Metrics whose values would change (SAP) are emitted as **new columns** alongside the untouched
originals. Nothing alters a default run's designs, filters, ranking, PDBs, or TSV columns until the
user opts in. We bump `chisel_version` and document each opt-in path.

**Tech stack:** Python 3.11 (design sif), numpy/pandas, LigandMPNN, ESM-C + SaProt (PLM, plm sif),
pyKVFinder/fpocket, pytest (host suite). Containers via `scripts/run_chisel_design.sh`.

**Version-skew finding (2026-06-12):** the real i2 commands
(`…/cmds/protein_chisel_i2`, 2467 of them) already set `ADAPTIVE_BIAS=1` + `CONSERVE_HBONDS=1`, but
the bad run's `provenance.json` reports `chisel_version=1.0.0`, and the adaptive controller is an
[Unreleased] feature *after* 1.0.0. The design sif bind-mounts the repo checkout (`--bind REPO:/code`),
so the running code was a 1.0.0 checkout that **ignored `ADAPTIVE_BIAS=1`** → the controller was a
silent no-op. **Action:** confirm the deployed checkout is this branch before any validation run; a
chunk of the user's pain is "the controller they asked for never ran." Still need WS-D (scope/axes)
to make it effective.

**Execution cadence (user-requested):** after each workstream lands green host tests, get an
**independent review** (a fresh review subagent **and** `codex exec -s read-only`) on the diff,
fold in edge cases, then proceed. Validate behavior on **Slurm CPU** with real seeds from the i2
campaign (below), smallest-and-PLM-off first, scaling up, PLMs on last.

**Non-negotiable contract (from the codebase culture + user decision "keep opt-in"):**
with no new flags/env, every byte of output is identical to `main` today. Verified per-workstream
by an A/A test (run default twice → identical) where feasible, and by additive-column / default-off
gating everywhere else.

---

## Root-cause summary (what these workstreams fix)

Triangulated across 5 subsystem audits + codex + the actual bad-run data (rank-0 design:
GRAVY=1.05, A=25.7%/L=19.5%/V=10.5%, shipped with `passed_seq_filter=False`, seed GRAVY=1.19):

| # | Defect | Workstream |
|---|---|---|
| 1 | **Backfill rescue ships seq-filter failures** as rank-0 (`iterative_design.py:3653-3680` gates on fpocket/tunnel/struct only; `selection__hard_final_filter_passed` = fpocket only) | **WS-A** |
| 2 | **SAP proxy** uses signed raw Kyte-Doolittle (polar cancels hydrophobic) + threshold 100 = off | **WS-B** |
| 3 | **Composition control is detect-but-don't-correct**: SOFT_BIAS tier dead-code, class-balance fixes only the single max-z AA per class (L escapes), no per-AA fraction cap, hard rule z>6 misses A's z=5.4 | **WS-C** |
| 4 | **Adaptive controller** was off; surface axis scoped to `distal_surface` only; no SAP/composition axes; charge band not user-set | **WS-D** |
| 5 | **No aggregate clamp on summed bias** + **temperature 0.15–0.20** makes every nudge near-deterministic (1 nat ≈ 787×, 2.5 nats ≈ 10⁷× at T=0.15); no anti-repetition term at sampling; `plm_strength=0` also zeros the weight-2.0 fitness rank | **WS-E** |
| 6 | **PLM refresh dead-code** (`mpnn_with_refresh.py` never called) → static-seed hydrophobic lock-in never broken | **WS-F** |
| 7 | **Tunnel-lining omit** has plumbing (`--omit_AA_per_residue`) but no consumed source set | **WS-G** |
| 8 | **`_chisel_` output token hard-coded** (`finalize_names.py:141`) — user renames by hand | **WS-H** |

User's "8 Å designable shell" theory is **moot**: design scope = all non-fixed positions
(~202/210); the `class` column never gates designability. No task changes that; WS-D/WS-G address
the real surface-steering and tunnel-omit levers instead.

---

## New flags / env (all default to legacy behavior)

| Flag | Env (run_chisel_design.sh) | Default | Workstream | Effect when set |
|---|---|---|---|---|
| `--ship_solubility_veto` | `SHIP_SOLUBILITY_VETO` | off | WS-A | rescue/selection may never ship a row outside the GRAVY/charge band |
| `--sap_corrected` | `SAP_CORRECTED` | off | WS-B | also emit `sap_corr_*` columns (centered, SASA-normalized) |
| `--sap_corrected_max` | `SAP_CORRECTED_MAX` | `None` | WS-B | gate on `sap_corr_max` (only if `--sap_corrected`) |
| `--composition_soft_bias` | `COMPOSITION_SOFT_BIAS` | off | WS-C | wire the SOFT_BIAS tier into the sampler bias |
| `--aa_fraction_cap` | `AA_FRACTION_CAP` | `None` | WS-C | hard-omit any AA already ≥ cap fraction in survivors (per-cycle) |
| `--composition_suppress_all_overrep` | `COMPOSITION_SUPPRESS_ALL_OVERREP` | off | WS-C | class-balance down-weights *every* z>threshold member, not just the max |
| `--adaptive_bias` (exists) | `ADAPTIVE_BIAS` | off | WS-D | enable controller |
| `--adaptive_surface_sasa_gate` | `ADAPTIVE_SURFACE_SASA_GATE` | `None` | WS-D | controller surface axis acts on all non-catalytic SASA≥gate positions, not just `distal_surface` |
| `--adaptive_bias_axes` | `ADAPTIVE_BIAS_AXES` | `charge,gravy` | WS-D | comma list incl. new `sap`, `composition` |
| `--adaptive_charge_band` | `ADAPTIVE_CHARGE_BAND` | cycle band | WS-D | e.g. `-15,-5` |
| `--bias_total_clamp` | `BIAS_TOTAL_CLAMP` | `None` | WS-E | clamp \|summed per-(pos,AA) bias\| to N nats |
| `--sampling_temperature_floor` | `SAMPLING_TEMPERATURE_FLOOR` | `None` | WS-E | raise any cycle temp below floor |
| `--anti_repeat_bias` | `ANTI_REPEAT_BIAS` | off | WS-E | per-cycle down-weight of AAs over-represented in the *running* pool at sampling |
| `--plm_class_strength` | `PLM_CLASS_STRENGTH` | `None` | WS-E | per-class PLM multiplier override, e.g. `distal_surface=0.3` |
| `--plm_off_mode` | `PLM_OFF_MODE` | off | WS-E | zero PLM sampling bias but keep raw-logp ranking (decoupled) |
| `--plm_refresh_rounds` | `PLM_REFRESH_ROUNDS` | `0` | WS-F | N ESM-C re-ground rounds on solubility-passing representatives |
| `--omit_tunnel_lining` | `OMIT_TUNNEL_LINING` | off | WS-G | omit hydrophobic/bulky AAs at pyKVFinder cavity-lining positions |
| `--design_token` | `CHISEL_SUFFIX` | `chisel` | WS-H | output naming `<stem>_<token>_<NN>.pdb` |

---

## WS-H: Configurable design-name token  *(start here — tiny, zero-risk, explicit ask)*

**Files:**
- Modify: `src/protein_chisel/tools/finalize_names.py` (`_CHISEL_RE` line 34, `new_id` line 141, add `design_token` param to `finalize_design_names`)
- Modify: `scripts/finalize_design_names.py` (add `--design_token`, default `chisel`; read `CHISEL_SUFFIX` env)
- Modify: `scripts/run_chisel_design.sh` (`_finalize_design_names()` ~line 143-150: pass `--design_token "${CHISEL_SUFFIX:-chisel}"`)
- Test: `tests/tools/test_finalize_names.py`

**Byte-identity:** default `design_token="chisel"` → identical regex strip + identical `new_id`.

- [ ] **Step 1 — failing test.** Add to `tests/tools/test_finalize_names.py`:
```python
def test_custom_design_token(tmp_path):
    # build a minimal finalized dir with 2 design PDBs + TSV, then re-finalize with token
    d = _make_min_run(tmp_path, ids=["seed_x_chisel_0", "seed_x_chisel_1"])  # existing helper or inline
    from protein_chisel.tools.finalize_names import finalize_design_names
    out = finalize_design_names(d, design_token="chiseli2")
    names = sorted(p.name for p in (d).glob("*.pdb"))
    assert names == ["seed_x_chiseli2_0.pdb", "seed_x_chiseli2_1.pdb"]
    # idempotent re-run with the same token does not double-append
    finalize_design_names(d, design_token="chiseli2")
    assert sorted(p.name for p in d.glob("*.pdb")) == names
```
- [ ] **Step 2 — run, expect FAIL** (`finalize_design_names() got unexpected kwarg 'design_token'`).
- [ ] **Step 3 — implement.** In `finalize_names.py`:
  - Signature: `def finalize_design_names(final_root, *, keep_intermediate=False, tsv_name=_TSV_NAME, design_token="chisel")`.
  - Build the strip regex from the token so re-runs strip the run's own token AND a literal prior `_chisel_<idx>` (back-compat): `strip_re = re.compile(rf"^(?P<stem>.+)_(?:{re.escape(design_token)}|chisel)_\d+.*$")`. Use it where `_CHISEL_RE` is used (line 139).
  - `new_id = f"{stem}_{design_token}_{rank:0{width}d}"` (line 141).
  - Validate token: `if not re.fullmatch(r"[A-Za-z0-9]+", design_token): raise ValueError(...)` (no underscores/dots → keeps the strip regex unambiguous; matches the no-`v1/v2` naming preference).
- [ ] **Step 4 — run test, expect PASS** + the existing `test_finalize_names.py` cases still green:
  `python -m pytest tests/tools/test_finalize_names.py -q`
- [ ] **Step 5 — CLI + shell.** `finalize_design_names.py`: `p.add_argument("--design_token", default=os.environ.get("CHISEL_SUFFIX","chisel"))` and pass through. `run_chisel_design.sh`: add `--design_token "${CHISEL_SUFFIX:-chisel}"` to the python invocation at line 149.
- [ ] **Step 6 — commit** `feat(naming): configurable --design_token / CHISEL_SUFFIX (default 'chisel', byte-identical)`.

---

## WS-A: Hard solubility veto in selection  *(the critical bug)*

**Files:**
- Modify: `scripts/iterative_design.py` — rescue bucketing `~3650-3680`; pre-export gate `~6253`; add `--ship_solubility_veto` arg `~4597`.
- Test: `tests/pipelines/` (new `test_solubility_veto.py`) — unit-test the bucketing helper on a synthetic DataFrame.

**Byte-identity:** flag default off → the `np.select` and pre-export path are unchanged. When ON,
a band-failing row can never enter a shippable bucket / the final top-K.

**Approach (two layers):**
1. Factor the band check into a pure helper next to the cycle config:
```python
def _within_solubility_band(df, cfg):
    g = pd.to_numeric(df.get("gravy"), errors="coerce")
    c = pd.to_numeric(df.get("net_charge_full_HH"), errors="coerce")
    return ((g >= cfg.gravy_min) & (g <= cfg.gravy_max)
            & (c > cfg.net_charge_min) & (c < cfg.net_charge_max)).fillna(False)
```
2. In `_deferred_rescue_score_candidates` (before the `np.select` at 3653), when `ship_solubility_veto`:
   compute `sol_ok = _within_solubility_band(rescued, cycle_cfg)`; AND it into the top-3 bucket
   predicates; add a terminal `~sol_ok → "rescued_solubility_veto"` bucket at priority 90. Also set
   an honest `rescued["selection__solubility_passed"] = sol_ok` (do NOT overload
   `selection__hard_final_filter_passed`; add the new column always so the TSV gains a truthful
   signal — additive, byte-safe).
3. Belt-and-suspenders: immediately before `_write_final_topk_artifacts` (~6253), when the flag is
   set, drop `top` rows failing `_within_solubility_band(top, final_cycle_cfg)`, `LOGGER.error` the count.

- [ ] Failing unit test: synthetic `rescued` with one GRAVY=1.05 row + one in-band row → with veto, the 1.05 row gets bucket `rescued_solubility_veto`/priority 90 and the in-band row gets `rescued_final_filters`; without veto, both as today.
- [ ] Implement helper + gating + new column.
- [ ] Test PASS; run `tests/pipelines/test_tier_filter.py` + any selection tests to confirm no regression.
- [ ] Env passthrough in `run_chisel_design.sh` (`SHIP_SOLUBILITY_VETO`).
- [ ] Commit `fix(selection): opt-in hard solubility veto so rescue can't ship GRAVY/charge failures`.
- [ ] **Also (always-on, additive, honest):** rename-by-addition — keep `selection__hard_final_filter_passed` but add `selection__solubility_passed` populated on every path (primary + rescue). Document in `docs/backfill_rescue.md`.

---

## WS-B: Corrected SAP (structure-aware) as new columns + optional gate

**Files:**
- Modify: `scripts/iterative_design.py` `_compute_sap_proxy` (~1504-1576) — add a corrected variant; emit `sap_corr_max/mean/p95` when `--sap_corrected`.
- Modify: shared KD/centering into a small helper (avoid the 3-way KD dict duplication noted in the audit — DRY with `adaptive_bias.py`). Put the centered scale in `src/protein_chisel/scoring/metrics.py` or a new `scoring/aggregation.py`.
- Test: `tests/scoring/test_sap.py` (new).

**Byte-identity:** existing `sap_*` columns + threshold 100 untouched. New columns only when flag set;
new gate only when `--sap_corrected_max` given.

**Corrected formula (fixes signed cancellation + Ala under-count):** per residue i,
`sap_corr_i = Σ_{j: CA_j within 10Å of CA_i} (SASA_j/SASA_ref_j) · max(0, KD_centered(aa_j))`
where `KD_centered = KD - mean(KD over 20 aa)` (so only above-average-hydrophobic residues
contribute and polar residues can't cancel). Calibrate the gate threshold against soluble EC3
hydrolases (validation task below) — until calibrated, ship the columns but keep the gate `None`.

- [ ] Failing test: a synthetic all-Ala-surface pose scores higher `sap_corr_max` than the legacy
  `sap_max` (relative assertion), and a polar-surface pose scores ~0.
- [ ] Implement centered helper + corrected proxy + columns; wire flag.
- [ ] Tests PASS; default run still emits only legacy `sap_*` (assert columns unchanged).
- [ ] Commit `feat(sap): opt-in centered/SASA-normalized sap_corr_* columns (legacy sap_* untouched)`.

---

## WS-C: Composition control that actually corrects

**Files:**
- Modify: `src/protein_chisel/expression/aa_class_balance.py` — `--composition_suppress_all_overrep` path (down-weight every z>threshold member, not just the class max).
- Modify: `scripts/iterative_design.py` — (a) wire `EngineResult.soft_bias_per_residue()` into the per-cycle bias when `--composition_soft_bias` (~4196 region, alongside class-balance); (b) `--aa_fraction_cap`: build per-cycle `omit_AA` for any AA whose survivor fraction ≥ cap, merged into the global omit (~5434/5470 omit-merge).
- Test: `tests/expression/test_aa_class_balance.py`, new `tests/test_composition_control.py`.

**Byte-identity:** all three behind flags default off/None.

- [ ] Failing test: pool with A=25.7%, L=19.5% → `compute_class_balanced_bias_AA(..., suppress_all_overrep=True)` returns down-weights for **both** A and L (today only A).
- [ ] Implement suppress-all-overrep (iterate all members with z>`balance_z_threshold`, down-weight by `min(max_bias_nats, bias_per_z*z)`; keep existing single-swap when flag off).
- [ ] Failing test: `--aa_fraction_cap 0.15` with survivors at A=0.257 → next-cycle omit dict contains `A` at all non-fixed designable positions (or global `--omit_AA` add) ; verify merge keeps existing omits.
- [ ] Implement cap → omit; wire `soft_bias_per_residue` (clamp to `max_bias_nats`).
- [ ] Tests PASS; default path byte-identical (flags off).
- [ ] Commit `feat(composition): opt-in suppress-all-overrep + per-AA fraction cap + wire SOFT_BIAS`.

---

## WS-D: Controller expansion (SASA scope + SAP/composition axes + charge band)

> **Status (shipped):** the `non_tunnel_surface` SASA scope, the configurable charge band, the
> `--adaptive_bias_axes` selector, the multi-pool (`pool_key`/`pools`) plumbing, and the KD dedup
> landed. A design debate (codex + 2 subagents) **deferred the composition and SAP axes**: a scalar
> integral controller is the wrong model for composition (per-cycle vector policy; over-rep is
> already handled by WS-C class-balance, and the merge makes a naive axis vanish), and the SAP axis
> ≡ the GRAVY surface actuator (double-push), needs the struct-stage pool, can't meet `min_n` when
> failing, and is uncalibrated. The `pool_key`/`pools` plumbing makes a future SAP axis a one-line
> follow-up. See `CHANGELOG.md` [Unreleased] WS-D.

**`non_tunnel_surface` scope (user ask — replace "10 Å from ligand" with a real designable-surface set):**
The user wants the surface controller to act on *any exposed residue that is not part of the active-site
mouth/tunnel* — "what I can see by eye," not a ligand-distance shell. Define at runtime (no new
classify stage needed; the driver already has `position_classes`, `sasa_fraction`, the WS-G
pyKVFinder tunnel-lining set, and the throat band from `tunnel_metrics`):

```
non_tunnel_surface(i) = (sasa_fraction[i] >= sasa_gate)            # genuinely exposed (default 0.20, configurable)
                        AND class[i] not in {primary_sphere, secondary_sphere, ligand}   # not active-site
                        AND i not in tunnel_lining_resnos           # WS-G pyKVFinder cavity-lining
                        AND i not in throat_band_resnos             # tunnel_metrics throat band (sc→ligand-centroid ∈[3.5,9.0]Å)
                        AND i not in fixed_idx                      # never catalytic/anchor
```
This is a *superset of `distal_surface`* (drops the 10 Å gate, adds back exposed `nearby_surface`)
*minus* the tunnel/mouth. Add it as `adaptive_bias.surface_scope(...)` returning the index mask, used
by `build_surface_delta` in place of the `SURFACE_CLASSES` test. **Re-evaluate cutoffs** as a
validation task: dump the per-class + per-scope counts for 3 real i2 seeds and confirm the steerable
set matches what the user sees by eye (tune `sasa_gate`, the throat band, and the lining cutoff).

**Files:**
- Modify: `src/protein_chisel/sampling/adaptive_bias.py` — (a) add `surface_scope()` computing the `non_tunnel_surface` mask; `build_surface_delta` gates on it (configurable; legacy `{distal_surface}` when `sasa_gate=None` and no tunnel sets → byte-identical); (b) new `ControlAxis` for `sap` (scope=surface, controlled var = `sap_corr_mean` from WS-B — **requires the seq-stage pool to carry it**, see note) and for `composition` (vector axis: down-weight any AA with pool-fraction z>thr); (c) `default_axes(..., charge_band=...)`.
- Modify: `scripts/iterative_design.py` controller call (~5697-5736) — pass sasa gate, axes list, charge band; ensure the measured pool carries the metric each axis needs.
- Test: `tests/test_adaptive_bias.py` (extend).

**Note / dependency:** the controller measures the **seq-stage** pool, which today lacks `sap_*` and
per-AA fractions. The `composition` axis can read the sequence directly (no extra compute). The `sap`
axis needs `sap_corr_*` in the seq-stage pool → either compute corrected SAP at seq-stage (needs a
pose; costly) or run the SAP axis off the **struct-stage** survivors. Plan: implement the
`composition` axis + SASA-scope + charge band now; gate the `sap` axis behind WS-B landing and add it
reading struct-stage survivors (documented limitation, like throat carry).

**Byte-identity:** controller already opt-in (`--adaptive_bias`); new sub-knobs default to current
scope/axes. With `--adaptive_bias` absent, zero change.

- [ ] Failing tests: surface delta acts on a `nearby_surface` SASA=0.5 position when `sasa_gate=0.2`
  (today it does not); composition axis down-weights A given a 25.7%-A pool; charge band `-15,-5`
  sets the axis target/band.
- [ ] Implement; keep all existing `test_adaptive_bias.py` green (control-law untouched).
- [ ] Commit `feat(controller): SASA-scoped surface axis + composition & sap axes + charge band`.

---

## WS-E: Sampling-core safety (clamp, temperature, anti-repeat, PLM decouple/off, per-class PLM)

**Files:**
- Modify: `scripts/iterative_design.py` bias assembly (~4099-4173): after summing all terms, when
  `--bias_total_clamp N`: `bias_k = np.clip(bias_k, -N, N)` (the audit/codex note: there is **no**
  aggregate clamp today). Temperature: when `--sampling_temperature_floor F`, `temp = max(temp, F)`
  per cycle (~860). Anti-repeat: when `--anti_repeat_bias`, compute running-pool AA z-scores and add
  a small negative bias to over-represented AAs at non-fixed positions before sampling.
- Modify: PLM decouple — `fitness_score` / the fitness call so ranking uses **unscaled** class
  weights (independent of `plm_strength`), making `--plm_strength 0` keep a meaningful rank; add
  `--plm_off_mode` (zero sampling bias, keep raw-logp rank).
- Modify: `src/protein_chisel/sampling/plm_fusion.py` + driver — `--plm_class_strength k=v,...`
  per-class override (FusionConfig already supports per-class; CLI only exposes the global today).
- Test: `tests/sampling/test_fitness_score.py`, `tests/test_*` for clamp/temp.

**Byte-identity:** clamp/floor/anti-repeat/off-mode all default None/off. Per-class override default
None. PLM decouple: since fitness is already invariant to *nonzero* `plm_strength`, the decoupled
rank equals today's for every current run (all use 1.25); only `plm_strength=0` changes (from ties
to a real rank) — and 0 is not a default. Add an A/A assertion at `plm_strength=1.25`.

- [ ] Failing tests: summed bias with a +5 and +4 term clamps to N=2; `temp_floor=0.3` raises a 0.15
  cycle; `plm_strength=0` + decouple → non-constant `fitness__logp_*` rank; per-class override sets
  `distal_surface` weight.
- [ ] Implement each behind its flag; A/A test default run unchanged.
- [ ] Commit `feat(sampling): opt-in bias clamp, temp floor, anti-repeat, PLM decouple/off, per-class PLM`.

---

## WS-F: Wire PLM refresh (break the static-seed lock-in)

**Files:**
- Modify: `scripts/iterative_design.py` — when `--plm_refresh_rounds K>0`, after a cycle pick a
  **solubility/composition-passing** representative (median-fitness among in-band survivors; if none
  in-band, skip refresh that round and log), recompute **ESM-C** marginals on it (host stage in the
  plm sif, like precompute), re-fuse, and use the refreshed `base_bias` for the next cycle. Use the
  existing `sampling/mpnn_with_refresh.py` orchestrator (`run_with_refresh`) by injecting the three
  callables.
- SaProt caveat (codex): masked-LM refresh erases the 3Di token too → refresh **ESM-C only** by
  default; keep SaProt at the seed marginals (document). Optionally `--plm_refresh_experts esmc`.
- Provenance: set `plm_refresh_rounds` truthfully (`provenance.py:31`).
- Test: `tests/sampling/test_mpnn_with_refresh.py` (extend — it already exists), + a driver-level
  smoke that `rounds=0` is a no-op (byte-identical).

**Byte-identity:** default `--plm_refresh_rounds 0` → `run_with_refresh` is the identity path
(module docstring guarantees rounds==0 == non-refresh).

- [ ] Failing test: `run_with_refresh(rounds=1, choose=least_hydrophobic_inband, recompute=stub)`
  re-grounds the bias off the chosen rep; `rounds=0` returns the un-refreshed bias.
- [ ] Implement the three injected callables + driver wiring + the in-band-representative guard.
- [ ] Tests PASS; `rounds=0` smoke is byte-identical.
- [ ] Commit `feat(plm): opt-in ESM-C bias refresh on solubility-passing representatives (rounds=0 default)`.

---

## WS-G: Omit tunnel-lining residues from design

**Files:**
- Modify: `scripts/iterative_design.py` — when `--omit_tunnel_lining`, derive a tunnel-lining resno
  set from **pyKVFinder** cavity-lining (reliable; `tunnel_metrics.pyKVFinder_score` runs in the
  design sif — the seed-fpocket `is_tunnel_lining` path needs an uninstalled binary) and merge
  `{f"A{resno}": "FILMVWYA..."}` into `omit_AA_per_residue` (existing path ~5470).
- Test: `tests/test_*` mocking the lining set → omit dict contains those positions.

**Byte-identity:** flag default off.

- [ ] Failing test: given a lining set `{50, 51}`, omit dict gains `A50`/`A51` with the bulky/hydrophobic AA string, merged with existing omits.
- [ ] Implement source + merge; guard double-stacking with the throat-bias delta (log if both active).
- [ ] Commit `feat(design): opt-in omit_tunnel_lining (pyKVFinder cavity-lining → omit_AA_per_residue)`.

---

## Validation (cluster)

**Slurm CPU smoke ladder (this host has ~4 GB free + no GPU → must use the `cpu` partition;
verify the `ipd`-account block is lifted with `sbatch --test-only` first).** Real i2 seeds from
`/net/scratch/woodbuse/organophosphatase/i4_design_260515/af3_out/filtered_i2/for_chisel/`,
ligand `…/theozymes/kcx_set1__pte_hbond/params/PBJ.params`, `PTM=A/LYS/4:KCX`. Ladder:
1. **Fastest sanity** — `USE_GPU=0 --debug-short-test` (target_k=20, n_cycles=2) + `PLM_OFF_MODE=1`
   (WS-E) on ONE small seed (e.g. `…FS024_T0_1_5_1_chisel_00.pdb`), with each new opt-in flag, to
   confirm the new paths run and produce designs (behavior, not quality).
2. **Scale up** — full `N_CYCLES=3 TARGET_K=40`, still PLMs-off, 2–3 diverse seeds; check the opt-in
   metrics actually move (GRAVY/charge/`sap_corr`/AA-composition distributions vs a legacy run).
3. **PLMs on last** — default `plm_strength`, `ADAPTIVE_BIAS=1` + new axes, on the same seeds;
   confirm the controller + refresh improve solubility without wrecking fitness/druggability.
- [ ] A/A byte-identity: a no-new-flags CPU run twice → identical TSV+PDBs (gates the contract).

- [ ] **Measure the actual PLM surface sign** (codex's open question): load a run's
  `fusion_runtime/base_bias.npy` (or recompute) and report the mean bias on hydrophobic vs polar AAs
  at `distal_surface` positions — proves/【dis】proves the "PLM favors surface hydrophobics" claim.
- [ ] **A/B ablation** (codex-recommended first cut): `--final_filter_backfill false` (or
  `--ship_solubility_veto`) × temperature floor 0.3 × `plm_strength {0, 1.25}` × adaptive seed
  warm-start, on 2–3 representative seeds; compare GRAVY/charge/sap_corr/AA-composition distributions.
- [ ] **Calibrate `sap_corr_max` threshold** against known-soluble EC3 hydrolases (feeds WS-B gate).
- [ ] **Seed quality:** report input GRAVY for the i1→i2 feed; add a doc note that feeding chisel's
  own hydrophobic outputs compounds the problem (the controller steers surface, not fold).

---

## Self-review (spec coverage)

- Hydrophobic surface → WS-A (stop shipping), WS-B (real SAP), WS-D (surface steering), WS-E/F (don't lock in). ✓
- Repeating/over-rep AAs → WS-C (cap + suppress-all + SOFT_BIAS), WS-E (anti-repeat + clamp + temp). ✓
- Controllers for all metrics → WS-D (charge/SAP/GRAVY/composition axes). ✓
- Net charge -5..-15 → WS-D `--adaptive_charge_band`. ✓
- Designable shell / omit tunnel-lining → WS-G (+ doc that the shell is moot). ✓
- "Are PLMs bad / skip PLMs for bad designs" → WS-E (decouple + off-mode + per-class) + WS-F (refresh). ✓
- Configurable suffix → WS-H. ✓
- future_plans alignment → WS-B & WS-D are its 2026-06-11 entries; WS-F is its PLM-refresh item. ✓
- Byte-identical/opt-in → every flag defaults legacy; SAP additive columns; PLM decouple A/A-safe. ✓

**Ordering for execution:** WS-H → WS-A → WS-B → WS-C → WS-D → WS-E → WS-F → WS-G
(fast win → critical bug → metric the controller needs → composition → controller → sampling core →
refresh → tunnel omit). WS-D's `sap` axis depends on WS-B; WS-F benefits from WS-A's in-band check.
