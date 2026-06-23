# Solubility-steering validation record (v1.3.0)

Evidence base for the controller-coordinator + configurable-ranges release. All runs are
full-pipeline (PLM fusion → LigandMPNN → struct/tunnel/fpocket filters → ship) on the
organophosphatase (PTE) i4 campaign seeds, GPU `gpu-bf`, 3 cycles, T=0.20/0.18/0.15,
target_k=16. Seeds + harness live in `/net/scratch/woodbuse/chisel_solub_test/`.

## 1. The problem, measured (A0 baseline = the pre-release default)

A 9-seed sweep of the *default* config (adaptive controller on, no coordinator, #29 off)
showed the default path ships unhealthy designs on hydrophobic seeds:

| seed class | GRAVY (shipped) | dominant AA | pass-QC | note |
|---|---|---|---|---|
| soluble seeds (low) | −0.03 … −0.22 | A (19–26%) | 16/16 | healthy but high single-AA |
| hydrophobic seeds (high) | **+0.78 … +1.46** | L (21–23%) | **0/16** | 100% backfill — no soluble design |

Two live failure modes in the default deploy path: **(a) runaway single-AA composition**
(21–26% one residue) and **(b) hydrophobic backfill** on hard seeds (the pipeline ships the
least-bad GRAVY>+0.8 designs when nothing passes the band).

Root cause traced to ground: on the hard seeds the controller *engaged correctly* (gate open,
drive ramping) and seed-triage *fired correctly* (PLM skipped), but the composition cap
(`AA_FRACTION_CAP=0.15`) and suppress-overrep **never fired because they gate on survivors, and
there were 0 survivors** — a vicious cycle (23% Leu → GRAVY +1.2 → 0 survivors → cap can't fire).

## 2. Independent weighting/units audit (codex)

Confirms the bias stack is **dimensionally consistent** — MPNN logits, PLM fusion, consensus,
class-balance, throat, and controller drives are all additive **nats**, sampled as `exp(bias/T)`.
The defect is **not the space**; it is that the default has **no bound on the sum**
(`--bias_total_clamp_odds` defaults `None`), so an aligned cell reaches **6–7 nats =
2.3e17–1.8e20×** at T=0.15 — a deterministic lock. The fix is a default-on SUM clamp (the
coordinator's nested `TOTAL_CEILING≈1e3×`), not re-scaling any single source.

## 3. Hydrophilic rescue — #29 composition-pool fallback

`--composition_pool_fallback` makes the cap/suppress fire on the previous cycle's *sampled*
pool when survivors are empty. 6-job experiment on the 3 hard seeds (`hydval/`):

| hard seed | A0 GRAVY | +#29 GRAVY | survivors (cycle 1) |
|---|---|---|---|
| Chigh (i2) | +0.78 | **+0.02** | 0 → **137** |
| Ahigh (predesign) | +0.97 | +0.35 | 0 → 17 |
| Bhigh (af3, GRAVY+1.46 seed) | +1.46 | +0.92 | resists (genuine input limit) |

**#29 rescues the moderate hard seeds to hydrophilic.** The most pathological seed (GRAVY +1.46,
27% Leu, 68.6% hydrophobic) resists — a genuine input-quality limit the pipeline already warns
about ("cannot make a hydrophobic *fold* soluble"); such seeds should be pre-filtered upstream.

**Key finding that motivated the pI controller:** the rescued designs are hydrophilic (GRAVY
mean +0.08, 13/16 ≤ 0.3) but fail QC by **hairline charge/pI band-edge misses in both
directions** — `net_charge −3.99 vs −4.0` (not negative enough) and `pI 4.98 vs 5.0` (pushed too
acidic). The charge controller alone is too weak to land designs *firmly* in-band; a pI
controller sharing the charge actuator is exactly the joint authority needed. Brute-force gain
cranking made it *worse* (oscillation), confirming the bounded coordinator over raw gain.

## 4. The coordinator (CF-2/CF-3 + pI) — what shipped

`--controller_coordinator` (opt-in) replaces the naive bias sum with a per-cell, weight-
partitioned, signed-sum-bounded odds budget: shared-actuator **max-not-sum by sign** (pI+charge
double-count is structurally impossible), signed-sum invariant + final clip (BUG-A), controller
budget **nested inside** the ~1e3× whole-stack ceiling with reserved headroom (BUG-B), all at the
application-cycle temperature. The pI-in-range axis shares the charge D/E/K/R actuator. A
**shared-actuator guard** rejects `pi` without the coordinator (the double-lock is unrunnable, not
just discouraged) — so the "naive redundancy" negative control is prevented by construction and
pinned by a unit test.

Reviews: the build agent's own codex pass + an **independent codex review (mine)** that caught two
driver-level footguns (the shared-actuator guard and the pI-band-from-`--pi_min/--pi_max` wiring),
both fixed. Host suite: **1067 passed**.

## 5. Coordinator A/B (A0 / A1 / A2 × hard+easy seeds)

3 arms (the A3 naive-double-lock control is now prevented by the guard + its unit test):
A0 = baseline (+#29), A1 = coordinator-only, A2 = +pI sharing the charge actuator. Pre-registered
criteria: A1/A2 diversity ≥ A0; A2 in-band yield > A0; every `effective_odds ≤ ceiling` (no lock).

12-job run (`abval/`, all #29 on; A3 prevented by the shared-actuator guard + its unit test):

| arm | Chigh GRAVY / in-band | Ahigh GRAVY / in-band | Blow,Clow (easy) | Shannon bits | max ctrl odds |
|---|---|---|---|---|---|
| A0 baseline | −0.01 / 1 | +0.33 / 0 | 16/16, 16/16 | 3.67–3.82 | ≤6.0 |
| A1 coordinator | **−0.10 / 3** | +0.38 / 0 | 16/16 | 3.70–3.83 | ≤6.0 |
| A2 +pI | +0.07 / 1 | +0.37 / 0 | 16/16 | 3.70–3.80 | ≤6.0 |

**Findings (honest):**
1. **Diversity preserved** — every arm ships 16/16 unique designs; Shannon entropy and
   %-positions-with-2+-AAs are ≥ A0 within noise (often slightly higher on easy seeds). The
   coordinator does **not** collapse diversity. ✓
2. **No regression on easy seeds** — A0/A1/A2 all 16/16 in-band on Blow/Clow. ✓
3. **No lock** — every controller `effective_odds ≤ 6.0× < 8× ceiling` at every cycle. ✓
4. **No double-lock possible** — the shared-actuator guard makes the naive-redundancy config a
   startup error (the "A3" negative control is structurally prevented, pinned by a unit test). ✓
5. **The dominant effectiveness lever is #29 + seed-triage**, not the coordinator: A0 already sits
   at GRAVY ~0 on Chigh (vs +0.78 without #29). The coordinator adds a *bounded, safe* refinement
   (A1 best on Chigh) + the structural stack-bound (units audit §2) + enables the pI redundancy
   safely. On these *triaged* seeds it is ~neutral-to-slightly-positive — matching the math
   review's prediction that the recalibration matters most for the non-triaged / aligned-prior
   path and as a structural safety bound, not as a yield multiplier on already-triaged seeds.
6. **The pI controller is correctly *dormant*** here — all four seeds have in-band pI (5.2–6.0 ∈
   [5,7.5]), so its gate stays closed (it shares the charge actuator and fires only when pI drifts
   out of band, e.g. the pI 4.98 over-acidification seen in the max-steer hydval arm). It is a
   feedback controller: redundancy that engages when needed, harmless when not.

**Ship conclusion:** the coordinator + pI are a sound, safe, diversity-preserving framework that
delivers the requested redundancy with a structural guard against misuse; the proven effectiveness
gains come from #29 + seed-triage (already validated). All opt-in / byte-identical by default; a
recommended driver-cell env template turns on the validated stack.

## 6. Recommended deploy config (driver cell)

All features are opt-in; defaults are byte-identical. This env template turns on the validated
stack for a hydrophobic-prone backbone (set via `scripts/run_chisel_design.sh`):

```bash
# --- effectiveness stack (validated) ---
ADAPTIVE_BIAS=1                       # closed-loop charge/surface controller
PLM_AUTOSKIP_BAD_INPUT=1             # seed-triage: skip the PLM on a pathological (hydrophobic) seed
COMPOSITION_POOL_FALLBACK=1         # #29: fire the composition cap on the sampled pool when 0 survivors (THE big lever)
COMPOSITION_SUPPRESS_ALL_OVERREP=1  # down-weight every over-represented class member
AA_FRACTION_CAP=0.15                # hard-cap any single AA at 15% of the pool
COMPOSITION_SOFT_BIAS=1             # per-residue expression soft-bias
SAP_CORRECTED=1                     # centered-KD SAP proxy
OMIT_TUNNEL_LINING=1               # keep bulky/aromatic residues out of the tunnel throat
# --- coordinator + redundant pI controller (optional; bounds the stack + safe redundancy) ---
CONTROLLER_COORDINATOR=1            # joint odds budget (8x controller / ~1e3x whole-stack), max-not-sum
ADAPTIVE_BIAS_AXES=charge,surface_hydrophobicity,pi   # add the pI controller (REQUIRES the coordinator)
CONTROLLER_CEILING=8               # joint controller odds ceiling (default 8)
CONTROLLER_VERBOSE=1               # write per-cycle controller_trace.tsv for chronological validation
# --- acceptance ranges (override any; these are the defaults) ---
# NET_CHARGE_MIN=-18  NET_CHARGE_MAX=-4   PI_MIN=5.0  PI_MAX=7.5
# GRAVY_MIN=-0.8  GRAVY_MAX=0.3  INSTABILITY_MAX=60  ALIPHATIC_MIN=40  BOMAN_MAX=4.5  SAP_MAX=100
```

Notes: the acceptance ranges set the **final/strictest** cycle; annealing relaxes earlier cycles
from them. Selecting `pi` without `CONTROLLER_COORDINATOR=1` is a deliberate startup error (it would
double-count). For a *known-soluble* backbone, `PLM_AUTOSKIP_BAD_INPUT` is a no-op (it only fires on
a pathological seed) and the PLM is kept. Seeds with GRAVY ≳ +1.4 (very hydrophobic *folds*) cannot
be made soluble by surface design and should be pre-filtered upstream.
