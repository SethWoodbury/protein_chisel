# Adaptive solubility-bias controller (opt-in)

A closed-loop controller that, across design cycles, measures the candidate pool's
**net charge** and **surface hydrophobicity** and steers the next cycle's LigandMPNN
biases toward target solubility. **Opt-in and default OFF** — with `--adaptive_bias`
absent the pipeline is byte-identical to before.

> **As of v1.2.0**, when the controller *does* run, its control-law **damping is ON by
> default** (EWMA measurement + derivative-on-measurement + slew-limit + soft 'ramp'
> deadband): it *regulates* setpoint instead of limit-cycling against a drifting pool
> (the legacy law swung charge −6.8→−8.8→−1.3). `--no_controller_damping` reverts to the
> legacy under-damped law. The base law described below is the *un-damped* integral
> controller; see `docs/cli_reference.md` (controller section) and `docs/architecture.md`
> (adaptive-controller framework + the independent math review) for the damping and the
> odds-space clamps (`--adaptive_bias_max_odds`, `--bias_total_clamp_odds`).

Source: `src/protein_chisel/sampling/adaptive_bias.py` (pure, unit-tested) +
integration in `scripts/iterative_design.py`. Env passthrough: `ADAPTIVE_BIAS*` in
`scripts/run_chisel_design.sh`.

## Why

Structure-conditioned LigandMPNN tracks the input backbone: on a hydrophobic scaffold
designs come out hydrophobic and often fail the solubility filters (net charge, pI,
GRAVY, SAP). The existing class-balance `bias_AA` only rebalances composition *within*
a biophysical class and can't respond to property gaps. This controller closes the
loop on the actual property statistics. It **complements, not replaces, input-scaffold
quality control** — it steers the *surface*, it cannot make a hydrophobic *fold*
soluble (a `GRAVY > +0.4` input logs a warning to that effect).

## Control design

We treat the survivor-pool mean of a property as the controlled variable and the
additive MPNN bias (nats) as the actuator. The pool mean responds to the bias applied
*this* cycle and largely forgets earlier cycles, so the controller is an **integral
controller that HOLDS its accumulated bias** (not a proportional one that resets, which
would limit-cycle on a near-static plant).

Per axis, per cycle `k → k+1`:

```
GATE open & out of deadband:  u_{k+1} = clip( carry*u_k + g_eff*e_norm , ±max_nats )
GATE closed or in deadband:   u_{k+1} = u_k            (purely HOLD)
wrong-sign plant response:     u_{k+1} = 0             (freeze + clear history)
```

- **Gate (binary, statistical)** — fire only when the pool is genuinely in trouble:
  `N ≥ min_n` AND `|t| = |mean−target|/se > tmin` AND fail-fraction `> fmin` with a
  Wald lower bound clearing 0. A zero-variance out-of-band pool is treated as `|t| = ∞`
  (not 0) so a collapsed pool isn't mistaken for healthy.
- **Drive (continuous, physical)** — signed, deadbanded error vs target (deadband =
  `deadband_frac` × band half-width). Its sign flips when the pool overshoots past
  target, so the controller **reverses** to pull an over-corrected pool back.
- **Hold** — when the pool is in-band the bias is held *exactly* and still emitted
  (a healthy pool does not drop the bias and revert).
- **Online gain estimate** — a two-point secant `K̂ = Δmean/Δu` (using the last
  distinct-`u` points) sets `g_eff = g·error_scale/(g·|K̂|)` for near-deadbeat steps,
  self-calibrating over the ~2 correction cycles. Skipped for the charge axis (its bias
  is merged with class-balance, so the requested `u` ≠ applied `u`).
- **Wrong-sign freeze** — if the pool moved the *wrong* way after a bias, the axis is
  frozen (`u=0`) and its history cleared, rather than doubling down.

## Actuators (partitioned by safety)

- **charge → GLOBAL per-AA** (`net_charge_full_HH`, target −10): up-weight D/E, mild
  down-weight K/R; reverses if the pool overshoots too negative. Composed with the
  class-balance `bias_AA` (class-balance wins conflicts).
- **surface hydrophobicity → PER-POSITION** (`gravy`, target −0.2): down-weight
  hydrophobic AAs (KD-scaled) **only at `distal_surface`, non-fixed positions** — never
  the buried core, never the catalytic/binding region (so conserved-H-bond anchors are
  excluded by construction). When a hydrophobic AA is *way over-represented* (z > 3 vs
  the hydrolase reference) the down-weight focuses on it.

The two AA sets are disjoint (D/E/K/R vs A/V/L/I/M/F/W/C) → no double-counting. All
magnitudes are clamped to `max_nats` (default 0.6, below the PLM fusion mean_abs ~0.85),
so the controller nudges and never overrides the structure/PLM signal.

`pI` is monitored only (it is a function of charge) to avoid two controllers fighting
the same D/E/K/R knobs. New axes (e.g. a future CamSol/DeepSP predictor) are one
`ControlAxis` registry entry.

## Usage

```bash
# default: controller OFF, byte-identical
ADAPTIVE_BIAS=1 bash scripts/run_chisel_design.sh         # turn it on
ADAPTIVE_BIAS=1 ADAPTIVE_BIAS_SEED_FROM_INPUT=1 ...        # also warm-start cycle 0
```

Env (each emitted only when set) → driver flags:

| Env | Flag | Default | Meaning |
|---|---|---|---|
| `ADAPTIVE_BIAS=1` | `--adaptive_bias` | off | enable the controller |
| `ADAPTIVE_BIAS_GAIN` | `--adaptive_bias_gain` | 0.6 | initial integral gain |
| `ADAPTIVE_BIAS_MAX_NATS` | `--adaptive_bias_max_nats` | 0.6 | per-AA / per-cell clamp |
| `ADAPTIVE_BIAS_CARRY` | `--adaptive_bias_carry` | 0.9 | integral leak during active correction |
| `ADAPTIVE_BIAS_DEADBAND` | `--adaptive_bias_deadband` | 0.25 | deadband (fraction of band half-width) |
| `ADAPTIVE_BIAS_TMIN` | `--adaptive_bias_tmin` | 2.5 | gate `|t|` threshold |
| `ADAPTIVE_BIAS_FMIN` | `--adaptive_bias_fmin` | 0.15 | gate fail-fraction threshold |
| `ADAPTIVE_BIAS_MIN_N` | `--adaptive_bias_min_n` | 30 | min pool size to act |
| `ADAPTIVE_BIAS_MODE` | `--adaptive_bias_mode` | proportional | `proportional` (integral) or `bangbang` |
| `ADAPTIVE_BIAS_SEED_FROM_INPUT=1` | `--adaptive_bias_seed_from_input` | off | warm-start cycle 0 from the input scaffold |

**Tuning** — defaults are deliberately conservative ("not overly strict / not overly
liberal"). Raise `gain` for faster correction (risk: first-cycle overshoot before the
gain estimate engages; the controller then reverses). Lower `tmin`/`fmin` to act on
milder problems. `bangbang` mode is a provably non-divergent fallback (only the sign of
the plant response matters).

## Telemetry

Per cycle: `cycle_NN/00_bias/adaptive_bias_telemetry.json` — per-axis gate open/closed
+ reason, `t`/fail-fraction stats, the drive `u`, gain correction, freeze flag, the
global per-AA bias, surface positions touched, and class-balance conflicts. The driver
log prints `cycle N adaptive controller: axes_active=… global=… surface_positions=…`.

## Cost & byte-identity

The controller is a few pandas reductions + one z-score + one vectorized `(L,20)` op per
cycle (~10⁴ ops) — negligible vs MPNN sampling hundreds of sequences. When off,
`run_cycle` receives `None` for both carried biases and the new code paths are skipped →
`bias_k` and `bias_AA` are bit-identical to the legacy pipeline (the input-GRAVY warning
is logging-only).
