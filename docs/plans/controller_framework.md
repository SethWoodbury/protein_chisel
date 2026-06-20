# Unified multi-objective controller framework

**Status:** design complete (4-member committee: architecture + objective-catalog + coordination + codex, all convergent). Implementation in incremental opt-in steps CF-1..CF-7, each TDD + codex-reviewed, **byte-identical by default**. Raw committee reports: `/tmp/{controller_framework_arch,objective_catalog,controller_coordination,codex_ctrl_out}` (snapshots; this doc is the canonical synthesis).

## Goal

Many enzyme-design objectives optimised **in parallel** by modular plug-in controllers that **coordinate** ("talk to each other"): instability↓, pI-in-range, thermostability/expression/behaviour proxies (cheap metrics, **liberal ranges — catch extremes only**), charge, GRAVY, SAP, composition, tunnel/throat access, H-bond networks, PLM fitness. Modular add/remove, **odds-space** (temperature-invariant), generalisable to **any enzyme**, **diversity-preserving**, rational/tunable/efficient, with **verbose chronological logging** for step-by-step validation.

## Foundation already in place

- `sampling/adaptive_bias.py`: `ControlAxis` (declarative axis: metric → target/band → `aa_weights` actuator → `pool_key`), `step_axis` (gated leaky-integral PID + online secant gain — **sound, do not edit**), `compute_adaptive_bias` (multi-pool), `default_axes` (only **charge** + **surface_hydrophobicity** today), `global_per_aa_bias` (sums + rescales to `max_nats`), `merge_bias_AA_strings` (class-balance-wins).
- `sampling/bias_scale.py`: `nats_for_odds(M,T)=T·ln(M)`, `odds_for_nats=exp(b/T)`, `ODDS_NUDGE=2 / ODDS_STRONG=8 / ODDS_LOCK=100`.

## The verified problem (why steering was weak)

MPNN samples `softmax((logits+bias)/T)` → odds `= exp(bias/T)`. At T=0.15 the controller clamp `max_nats=0.6` is only **55×**, while PLM fusion ~1.36 nats is **~8700×** and the additive terms (PLM+consensus+clash+...) **multiply in odds space to ~10¹³×**, locking cells. The controller is dwarfed and the stack kills diversity. Fix = odds-space + temperature-aware budgets, recalibrated so each source is a single legible knob.

## Key design decisions (convergent)

1. **Don't touch `step_axis`** (control law is correct). Recalibrate the *clamp* and add a *coordinator*.
2. **`ControlAxis → Actuator → Coordinator`.** `ControlAxis` stays the measure+control-law leg (one registry entry per *scalar* objective). Add a defaulted `actuator` field (byte-identical when `None`); many axes can **share** one actuator (coordinator takes the **dominant** drive, not the sum) → pI shares the **charge** actuator (D/E/K/R), SAP shares the **GRAVY/surface** actuator → **no second lock**. Admit a `vector` actuator for composition. Structural/H-bond objectives are **out of scope for bias** — they stay filters/ranking + `conserve`.
3. **Odds-space recalibration.** Thread `cycle_cfg.sampling_temperature` into `compute_adaptive_bias`; express clamps as `nats_for_odds(max_odds, T)`. Recommended ceilings: charge/surface **8× (STRONG)**; instability/composition **2× (NUDGE)**; pI/SAP share their actuator; **coordinator total budget 8×** — all deliberately subordinate to the PLM, but now T-invariant single knobs. Legacy raw-nats path kept for the 2 current axes (byte-identical when odds-mode off).
4. **Coordinator — 3 tiers in odds space.** `veto` (clash/omits/fraction-cap — bans, always win, unbudgeted) → `budget` (preference controllers + PLM/consensus share a joint odds budget; opposite pulls **sum** with per-axis weights) → `nudge` (diversity ≤2×, shrinks the budget). New `ControlAxis` fields `tier/priority/weight/conflict_policy`. A joint **odds-space SUM clamp** on (PLM+consensus+controllers) is the actual fix for the 10¹³× lock; veto bypasses.
5. **Most "new objectives" are FILTERS + RANKING, not new actuating controllers** (objective-catalog): pI≡charge, SAP≡GRAVY, aliphatic⊂GRAVY, instability=dipeptide (no clean per-AA projection). They're already computed (`instability_index`, `pi`, `aliphatic_index`, `boman_index`, `net_charge_*`, `gravy`, `sap_*`, ...). Expose **liberal configurable bands** (catch extremes) + low-weight TOPSIS `target` specs. The **actuating** set stays small + orthogonal: charge, GRAVY/surface, composition, throat.
6. **Diversity is a first-class (nudge-tier) controller** + a hard per-position entropy floor at sampling + optional MIN_HAMMING coupling. Measures/logs even when actuation is off (observe collapse first).
7. **Shared once-per-cycle `compute_pool_stats()`** (efficiency = per-metric gate stats; consistency = diversity stats) read by all axes + the diversity controller + `cycle_metrics.tsv`.
8. **Verbose logging:** opt-in `--controller_verbose` per-cycle CONTROLLER REPORT (per axis: measured/target/band/error/gate+why/drive `u`/**odds@T**/applied per-AA bias + pool diversity) and an append-only `run_dir/controller_trace.tsv` (one row per axis per cycle) for chronological offline validation.

## Incremental plan (each opt-in, TDD, byte-identical default, codex-reviewed)

- **CF-1 — odds-space clamp.** Thread `T` into `compute_adaptive_bias`; opt-in `--controller_max_odds` clamps the combined controller bias via `nats_for_odds(max_odds, T)`; default `None` → current `max_nats` path (byte-identical). *Unlocks effective controllers — the crux.*
- **CF-2 — `Actuator` abstraction.** Defaulted `actuator` field; charge/surface plugins reproduce current output byte-identically alone; admit shared + vector actuators.
- **CF-3 — `Coordinator`.** 3-tier veto/budget/nudge + joint odds SUM clamp + report of *actually-applied* `u` back to axis state. `legacy_sum` default == today.
- **CF-4 — registry + declarative config.** Name→factory registry (clone `experts.registry`); TOML/dict profile lifting PTE setpoints out of code (default == current → byte-identical); `--controller_profile`.
- **CF-5 — verbose logging + `controller_trace.tsv`.** (Can land early — pure observability, byte-identical.)
- **CF-6 — first new controller** proving "one entry" (pI sharing the charge actuator, delivering the user's pI-range ask) + expose liberal band filters (instability/aliphatic/boman/pI) + fix the `--pi_min` help bug (references nonexistent `--net_charge_max`).
- **CF-7 — follow-ups:** SAP (shares GRAVY actuator), composition-as-controller view, thermostability/expression rank specs.

## Validation protocol (the user's emphasis)

Every CF step: host TDD green → run on **easy (dir-B low-GRAVY) + hard (dir-A/C high-GRAVY)** seeds on SLURM with `--controller_verbose` → read the `controller_trace.tsv` to confirm each controller measures/drives/clamps as intended chronologically → compare GRAVY/charge/pI/instability/composition/diversity vs the prior step. Seeds + sweep harness: `/net/scratch/woodbuse/chisel_solub_test/sweep/`.
