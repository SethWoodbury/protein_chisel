# Throat-Blocker Feedback — Iterative MPNN Bias for Pocket Accessibility

## What it does

After cycle k of `iterative_design_v2`, the top survivors are scored for
*pocket entrance accessibility* by `score_tunnels()` (homegrown ray-cast
+ pyKVFinder cavity detection). The per-design output includes a
**breakdown** — a list of `(resno, resname, mass_weight)` for designable
side chains in the throat band that block the best escape cone.

These observations are aggregated across the cycle's top survivors,
turned into a `(L, 20)` per-(position, AA) bias delta, and added (with
exponential decay) to the LigandMPNN bias matrix for cycle k+1. This
actively pressures MPNN to swap bulky residue types at recurring throat
positions in the next sample.

## Algorithm

```
for each design in cycle_k_top_100:
    breakdown = score_tunnels(design, return_breakdown=True)
    for (resno, resname, mass_weight) in breakdown:
        pos_weight[resno] += mass_weight
        pos_aa_count[resno][resname] += 1

for each resno where avg_weight = pos_weight[resno] / N >= threshold:
    for each AA with bulky mass-weight (Trp, Phe, Tyr, His, Arg, Lys):
        delta[res_idx, aa_idx] -= base_strength * mass_weight
    delta[res_idx, top_observed_aa_idx] -= observed_extra
    delta[res_idx, :] = max(delta[res_idx, :], -max_total_per_aa)

bias_for_cycle_k+1 = base_bias + (decayed throat_bias_prev) + new_delta
```

The bias is **additive on top of** the existing class-balance and
consensus-reinforcement biases. Previous-cycle bias decays exponentially
(`decay = 0.5` by default — halve each cycle), then the new cycle's
observations are added — so positions that fix themselves release
pressure, while positions that recur build cumulative penalty.

## Production defaults (after 4-round A/B sweep)

In `src/protein_chisel/tools/tunnel_metrics.py::build_throat_bias_delta`:

| param | value | rationale |
|---|---|---|
| `avg_weight_threshold` | 0.10 | Captures positions where ~12% of designs have a bulky blocker — typical for throat residues. Lower threshold gave 0 positions targeted. |
| `base_strength` | 1.0 nat | Per-AA penalty per bulky residue at each targeted position. Stronger (1.5) hurt diversity. |
| `bulky_threshold` | 0.70 | Only Trp/Phe/Tyr/His/Arg/Lys get pre-emptive downweight (mass-weight ≥ 0.70). Met/Leu/Ile/Val are NOT pre-emptively downweighted — preserves medium-aliphatic rotamer choices. |
| `observed_extra` | 0.5 nat | Additional penalty on the OBSERVED top blocker AA (on top of the bulky pre-emption). |
| `max_total_per_aa` | 2.0 nats | Cap per (position, AA) cell to prevent catastrophic over-constraint. |

In `src/protein_chisel/scoring/multi_objective.py::DEFAULT_METRIC_SPECS`,
new TOPSIS criteria with halved weights:

| criterion | direction | weight | rationale |
|---|---|---|---|
| `tunnel__sidechain_blocked_fraction` | min | 0.30 | The fixable-blockage signal |
| `tunnel__throat_bulky_designable_count` | min | 0.25 | Mass-weighted throat clutter |
| `tunnel__best_cone_mean_path` | max | 0.20 | Free-path through the entrance |
| `pkvf__cavity_depth_max` | max | 0.20 | Real-pocket-with-depth signal |
| `pkvf__cavity_volume` | max | 0.15 | Larger cavities are usually more accessible |

Sum of new weights: 1.10 (vs the original 2.20 which was over-aggressive
and crashed both catalytic h-bonds and diversity).

## A/B sweep results (FS148 scaffold, 3 cycles, gpu-train L40)

| Round | Settings | hamming | pkvf vol | pkvf depth | h-bonds | fitness | sap_max | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | base 1.0 / threshold 0.30 / TOPSIS+0 | 38 | 179 | 0.18 | 1.11 | -1.80 | 1.61 | only 1 position targeted; effectively no bias |
| 2 | base 1.5 / threshold 0.10 / TOPSIS+2.2 | **38** | 107 | 1.23 | 2.28 | -1.81 | 2.06 | 4 positions, 41 cells; cavity gains; diversity drop |
| 3 | base 1.0 / threshold 0.10 / TOPSIS+2.2 | 32 | 121 | 0.54 | **1.00** | -1.79 | 1.41 | gentler bias *worse* on h-bonds; TOPSIS too aggressive |
| 4 ★ | base 1.0 / threshold 0.10 / TOPSIS+1.1 | **41** | **140** | 1.01 | 2.08 | -1.81 | 1.93 | **production default** |

### Round-4 vs baseline (same TOPSIS, only difference is `--throat_feedback`):

```
metric                                         BASELINE   TREATMENT      Δ
pkvf__cavity_volume                              120.1       140.3    +20  ✓✓
pkvf__cavity_depth_max                            0.63        1.01   +0.38
tunnel__best_cone_mean_path                       11.2        11.9   +0.70
tunnel__sidechain_blocked_fraction               0.587       0.535  -0.053  ✓
pairwise_hamming                                 36.28       41.05   +4.77  ✓ diversity GREW
fitness__logp_fused_mean                        -1.800      -1.815  -0.015
sap_max                                           1.57        1.93   +0.36  worse
n_hbonds_to_cat_his                               2.46        2.08   -0.38  worse
ligand_int__strength_total                       62.85       61.91   -0.94
fpocket__druggability                            0.969       0.961  -0.008
pi                                                5.22        5.19  -0.030
charge / aliphatic / boman / instability / aromaticity        ≈ comparable
```

**Trade analysis**: pocket-accessibility gains are substantial (cavity
volume +17%; tunnel mean path +6%; sidechain-blocked −9%). Diversity
actually grew (+13%). Costs: −15% catalytic h-bonds and ~+0.36 sap_max.
Net positive for tunnel-aware design; the h-bond cost is the main thing
to monitor going forward.

## CLI

```
--throat_feedback              # default ON
--no_throat_feedback           # disable for diagnostic runs
--throat_feedback_decay 0.5    # exponential decay across cycles
```

## Telemetry

Per-cycle:
- `cycle_NN/throat_blocker_telemetry.json` (stripped if `--shipping_layout`)
- `cycle_NN/00_bias/telemetry.json` includes `throat_bias_n_positions`
  and `throat_bias_max_penalty`

Aggregate (always survives shipping cleanup):
- `<run_dir>/throat_blocker_telemetry.json` — per-cycle list of
  `{blocker_stats, throat_telemetry}` covering every cycle that
  produced a bias.

Each `throat_telemetry["positions"]` row has:
- `resno`, `top_aa`, `avg_weight`, `n_observed`, `n_aas_biased`,
  `max_penalty_nats`

## When to disable

Use `--no_throat_feedback` when:
- The seed PDB has open access already (n_escape_cones ≥ 3 from
  `score_tunnels` on the seed) — no entrance-blockage to fix.
- You're iterating on fitness/sap and want pure baseline behavior.
- Catalytic h-bond geometry is the primary objective and you can't
  afford the −15% cost.

## Files

- `src/protein_chisel/tools/tunnel_metrics.py` — `aggregate_blocker_stats`,
  `build_throat_bias_delta`, plus the existing `score_tunnels` /
  `pyKVFinder_score`.
- `scripts/iterative_design_v2.py` — `run_cycle` threads
  `throat_bias_prev` between cycles, applies decay, accumulates new
  observations, returns `cycle_telem` with `throat_bias_delta` for the
  next cycle.
- `src/protein_chisel/scoring/multi_objective.py` — `DEFAULT_METRIC_SPECS`
  carries 5 tunnel/pkvf criteria with halved weights.


## Cross-scaffold validation (SEED1 FS269 vs SEED2 FS148)

Same production defaults, two different scaffolds. SEED1 has unusually
high baseline diversity (hamming ~53) while SEED2 is more constricted.
Treatment behavior is scaffold-dependent:

| metric | FS148 baseline | FS148 treatment | FS269 baseline | FS269 treatment |
|---|---|---|---|---|
| pairwise_hamming | 36.3 | **41.1** ↑ | **53.3** | 38.4 ↓ |
| sap_max | 1.57 | 1.93 ↑ | 1.68 | **1.27** ↓ ✓✓ |
| n_hbonds_to_cat_his | 2.46 | 2.08 ↓ | **1.06** | **1.76** ↑ ✓ |
| pkvf__cavity_volume | 120 | **140** ↑ | 131.5 | 125.1 ↓ (small) |
| druggability | 0.969 | 0.961 | 0.949 | **0.974** ↑ |
| fitness | -1.800 | -1.815 | -1.779 | -1.783 |

Reading: when a scaffold has a **clear** entrance-blockage problem
(FS148 with 4+ recurring blockers), treatment substantially improves
pocket geometry while preserving (even improving) diversity.
For scaffolds that are already accessible (FS269 with only 1 position
flagged at 1.0 nat), treatment still helps sap and h-bonds but at
the cost of diversity (which was already abundant).

This is the expected behavior: **bias matters when there's a problem
to fix, and is roughly neutral when there isn't**. For production
runs, pick `--throat_feedback` (default ON) when designing on a
scaffold known or suspected to have constricted entrance, and
consider `--no_throat_feedback` for scaffolds with naturally open
pockets where extra constraint is unnecessary.

A `--throat_feedback_decay` of 0.3 (vs default 0.5) would release
cumulative pressure faster and preserve diversity better at the cost
of weaker convergence — try if FS269-style scaffolds keep showing
diversity loss.
