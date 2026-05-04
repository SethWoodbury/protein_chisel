# protein_chisel metrics reference

This document is the per-column reference for the design TSVs that the
`iterative_design_v2` driver produces — most importantly
`final_topk/all_survivors.tsv`. Every numeric / categorical column that
ends up in that file is documented below with its definition, formula,
direction (max / min / target / informational), units, the actual range
seen on the production PTE_i1 sweep at
`/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260504-054455-127-pid3231842/final_topk/`,
and a recommended filter threshold (where one applies).

The summary "Suggested filter ranges for PTE_i1" table at the bottom
gives a one-page cheat sheet for the most-used filter knobs.

Two conventions across this document:

* **Direction** — what counts as "good": `max`, `min`, `target=X`, or
  `info` (no filter applied — diagnostic only). For `target`, deviation
  from the target value is the loss; the multi-objective ranker treats
  `|x - target|` as a min-objective.
* **Range (PTE_i1)** — `[min .. p5 .. p50 .. p95 .. max]` from the 124
  designs in the production sweep's `all_survivors.tsv`. These are
  designs that already passed every hard filter, so the lower tail is
  somewhat optimistic — true rejection thresholds are listed
  separately under "Suggested filter".

### How to read this doc

* Skim the table-of-contents to find the family you want.
* Each family's table lists every column we emit, with formula links
  back to the implementation file. Hover-over the formula row first
  before believing a value — many of the columns share names with
  literature metrics that have subtly different conventions, and
  the implementation-doc string in the source is the canonical
  source of truth.
* Skip to **§18 Suggested filter ranges for PTE_i1** if you just need
  the cheat-sheet of thresholds.
* The **constants in PTE_i1** column captures any value that's
  invariant across all 124 survivors. Most of these are "same seed
  backbone → identical CA-only metric" — they become useful when the
  pipeline is run across multiple seeds.

Implementation file references after each family point to the
canonical computation site.

---

## Table of contents

1. [Identity / provenance](#1-identity--provenance)
2. [LigandMPNN sampler telemetry](#2-ligandmpnn-sampler-telemetry)
3. [Charge / pI](#3-charge--pi)
4. [Hydrophobicity / GRAVY](#4-hydrophobicity--gravy)
5. [Stability / aggregation](#5-stability--aggregation)
6. [Secondary structure (sequence-derived)](#6-secondary-structure-sequence-derived)
7. [Expression-rule engine](#7-expression-rule-engine)
8. [Sequence-level fitness (PLM)](#8-sequence-level-fitness-plm)
9. [Pocket geometry (fpocket)](#9-pocket-geometry-fpocket)
10. [Ligand interaction panel](#10-ligand-interaction-panel)
11. [Preorganization](#11-preorganization)
12. [Clash detection](#12-clash-detection)
13. [Dynamic Flexibility Index (DFI)](#13-dynamic-flexibility-index-dfi)
14. [AA composition / z-scores](#14-aa-composition--z-scores)
15. [Diversity / position metadata](#15-diversity--position-metadata)
16. [Multi-objective ranking](#16-multi-objective-ranking)
17. [Hard-filter pass flags](#17-hard-filter-pass-flags)
18. [Suggested filter ranges for PTE_i1](#18-suggested-filter-ranges-for-pte_i1)

---

## 1. Identity / provenance

These columns identify the design row, its ancestry, and the random
state that produced it. They are not filterable — they are bookkeeping.

| Column | Definition | Type | Notes |
|---|---|---|---|
| `id` | Design ID. Format: `<sampler>_<seed>_<batch>_<idx>` plus cycle suffix. | str | Unique within a run. |
| `sequence` | The full 1-letter design sequence. | str | Length = `length`. |
| `parent_design_id` | ID of the previous-cycle survivor that seeded this design (consensus reinforcement). Empty for cycle 0. | str | Tracks the iteration genealogy. |
| `sampler` | Sampling strategy: `mpnn_temp_panel` / `iterative_fusion` / `injected_seed`. | str | |
| `sampler_params_hash` | Short hash of the sampler's config dict, useful for grouping replicates. | str | |
| `is_input` | True for the WT / seed row that's prepended for context; dropped before filters. | bool | Always False in `all_survivors.tsv`. |
| `header` | FASTA header from LigandMPNN output (verbatim). Includes seed PDB stem. | str | |
| `seq_hash` | blake2b(seq, 12-byte) — 24-hex-char fingerprint. Dedup key. | str | See `sampling/fitness_score.py:seq_hash`. |
| `n_dupes` | How many candidates collapsed to this row during dedup. 1 = unique post-MPNN. | int | Larger = consensus across seeds. PTE_i1: always 1 (no within-cycle dupes after cycle 0 dedup). |
| `cycle` | 0-indexed cycle in which the design was sampled. | int | PTE_i1: 0–2 (3 cycles). |

---

## 2. LigandMPNN sampler telemetry

Per-design metadata emitted by the LigandMPNN call. Useful for
diagnostics ("are we mining the same seed too hard?") but not for
filtering.

| Column | Definition | Direction | Range (PTE_i1) | Notes |
|---|---|---|---|---|
| `mpnn_t` | Sampling temperature passed to LigandMPNN. | info | 0.15 – 0.20 | We sweep across a small temperature panel each cycle. |
| `mpnn_seed` | RNG seed for the LigandMPNN call. | info | — | |
| `mpnn_num_res` / `mpnn_num_ligand_res` / `mpnn_batch_size` / `mpnn_number_of_batches` | Echoed config; populated only in the per-call audit TSV, NaN in the merged survivor TSV. | info | NaN | Don't filter on these; use `cycle` and `mpnn_seed` for grouping instead. |
| `mpnn_id` | 0-indexed sequence position within the LigandMPNN batch output. | info | 2 – 496 | |
| `mpnn_overall_confidence` | LigandMPNN's reported `overall_confidence` for the whole sequence. Mean per-position log p over the unmasked positions. | max | 0.31 – 0.40 (mean 0.35) | Not a filter — strongly correlated with how well the seed structure matches its sequence. |
| `mpnn_ligand_confidence` | Per-position log p averaged over residues within the LigandMPNN ligand-aware spheres only. | max | 0.24 – 0.44 | Higher = the model is confident the ligand-binding shell makes sense. Diagnostic only. |
| `mpnn_seq_rec` | Native sequence recovery: fraction of designable positions where the sample matches the original. | info | 0.56 – 0.67 | A *low* recovery is fine — the goal is engineered, not native. Watching for collapse to ~0.95 (degenerate sampling). |
| `length` | `len(sequence)`. | info | 202 (constant) | Hard-filter rule: must equal `wt_length`. |

---

## 3. Charge / pI

All five charge variants and pI are computed in
`src/protein_chisel/filters/protparam.py`. The formulas are the
standard Henderson-Hasselbalch model with side-chain pKas from
Pace 1999 / Bjellqvist 1994. The pH used (`design_ph`) is recorded
as a column — default 7.8 (close to the user's PTE assay buffer at
pH 8.0 with a 0.2 safety margin). Sequence is padded with optional
N/C-term flanks before the calculation so the recorded charge
reflects the expressed protein after vector tags.

For each ionizable group with pKa_a (positive) or pKa_d (negative),

* `f_pos(pH, pKa_a) = 1 / (1 + 10^(pH − pKa_a))` (fraction protonated → +1)
* `f_neg(pH, pKa_d) = 1 / (1 + 10^(pKa_d − pH))` (fraction deprotonated → −1)

and the net charge is

```
Z(pH) = Σ_pos n_aa · f_pos(pH, pKa_aa) + f_pos(pH, pKa_Nterm)
      − Σ_neg n_aa · f_neg(pH, pKa_aa) − f_neg(pH, pKa_Cterm)
```

Different variants use different residue subsets:

| Column | Variant | Residues counted | Direction | Range (PTE_i1) | Use |
|---|---|---|---|---|---|
| **`net_charge_full_HH`** | **Robust** (this is the filter charge) | K, R, H (+) ; D, E, C, Y (−) ; Nterm/Cterm | target ≈ −10 (PTE_i1 expression sweet spot) | −16.0 .. −5.0 (median −6.0) | Used for the hard `net_charge_max / min` filter in `stage_seq_filter`. |
| `net_charge_no_HIS` | HH without histidine | K, R (+) ; D, E, C, Y (−) ; termini | info / sanity | −16.1 .. −4.1 | Use when HIS protonation is uncertain (cat-HIS coordinating metal). |
| `net_charge_with_HIS_HH` | Biopython `charge_at_pH(pH)` (Bjellqvist 1994 pKas) | All ionizables (Bjellqvist scale) | info | −16.9 .. −4.9 | Slight numeric drift from `full_HH`; use for cross-check only. |
| `net_charge_HIS_half` | `no_HIS + 0.5·n_HIS` | K, R (+) ; D, E, C, Y (−) ; termini ; +0.5/HIS | info | −12.6 .. −0.6 | Approximates HIS pKa ≈ 7.0 (half-protonated at pH 7). |
| `net_charge_DE_KR_only` | Minimalist legacy (matches old Rosetta NetCharge filter) | K, R (+) ; D, E (−) ; termini | info | −16.1 .. −4.1 | Only for back-compat with pre-2026 lab scripts. |
| `design_ph` | The pH at which all five variants were computed. | info | 7.8 (constant in PTE_i1) | | |
| `pi` | Isoelectric point (Biopython `isoelectric_point()`, bisection). | target ≈ 5.5 (PTE-i1 default) | 5.00 – 6.03 | Soft objective in TOPSIS at weight 0.3. Hard band [4.0, 8.0] tolerable. |

**Caveats / quirks**

* `charge_at_pH7*` field names are historical — the actual pH used is
  whatever `design_ph` is. They were named when pH 7.0 was the only
  option.
* `full_HH` includes Cys and Tyr ionization. At pH 7.8, both are
  almost fully protonated (Cys ~93% neutral; Tyr ~99.95% neutral),
  so they contribute < 0.1 e to the charge for typical compositions.
  Where Cys count is high (>5%) the difference vs `no_HIS` becomes
  measurable.
* Termini: pKa_N = 9.0 (+1 contributor), pKa_C = 2.0 (−1). At pH 7.8
  both contribute close to ±1, so a length-200 protein already
  inherits ~0 e from the termini.

**Suggested filter (PTE_i1)** `−15 ≤ net_charge_full_HH ≤ −4`. The
production run hit −15.97 .. −4.00, with the target at −10. Tighten to
`[−12, −5]` if downstream lysate stability is the bottleneck; relax
to `[−18, −2]` only if you've validated that wider designs still
express.

---

## 4. Hydrophobicity / GRAVY

| Column | Definition | Formula | Direction | Range (PTE_i1) | Filter |
|---|---|---|---|---|---|
| `gravy` | Grand average of hydropathicity (Kyte–Doolittle 1982). | mean of per-residue KD score over the sequence | target ≈ −0.2 (slightly hydrophilic) | −0.64 .. +0.04 (median −0.24) | Hard band `[−0.8, +0.3]`. |
| `aliphatic_index` | Ikai 1980 thermostability indicator. Higher = more thermally stable. | `AI = X(A) + 2.9·X(V) + 3.9·(X(I) + X(L))` where `X` = mol percent | target ≈ 95 | 89.6 .. 113.7 (median 101.8) | Hard `aliphatic_index ≥ 40`. PTE_i1 hits >89, comfortably in the thermophile band. |
| `boman_index` | Boman 2003 PPI-prone / "stickiness" indicator (Radzicka–Wolfenden 1988 transfer ΔG). Higher = more aggregation-prone. | mean of per-residue ΔG (water → cyclohexane) in kcal/mol | target ≈ 2.5 (PTE_i1 sweet spot — slightly above pharmaceutical-protein threshold of 2.0) | 2.42 .. 3.35 (median 2.99) | Hard `boman_index < 4.5`. PTE-style enzymes are naturally somewhat sticky (substrate binding); the 2.5 target is empirical, not lit-derived. |
| `aromaticity` | Biopython `aromaticity()` — fraction of F + W + Y. | `(n_F + n_W + n_Y) / L` | info | 0.029 .. 0.067 (median 0.043) | No filter. Aromatic enrichment is fine for PTE which has substrate π-stacking. |

**Caveats**

* GRAVY is dominated by 5–6 residues per ~200; small composition
  swings can move it by 0.1. Don't over-tighten.
* Aliphatic index is computed on the canonical AA set only — non-standard
  residues (X, B, Z, etc.) are stripped before the calculation, so for
  very dirty sequences the value is computed on a slightly shorter
  string than `length`.
* Boman uses Radzicka–Wolfenden 1988 water→cyclohexane ΔG (kcal/mol).
  Negative ΔG = hydrophobic side chain. The mean is taken over the
  full canonical sequence (including padded N/C-term flanks).

---

## 5. Stability / aggregation

| Column | Definition | Formula | Direction | Range (PTE_i1) | Filter |
|---|---|---|---|---|---|
| `instability_index` | Guruprasad 1990. Predicts in-vitro half-life from per-dipeptide stability weights. < 40 → stable; 40–60 → moderate; > 60 → unstable. | `II = (10/L) · Σ DIWV(aa_i, aa_{i+1})` summed over adjacent pairs | min (target small) | 11.4 .. 51.2 (median 24.3) | Hard `instability_index < 60` (lit "40 stable" is for natives — designs run hotter). |
| `sap_max` | Max per-residue SAP-proxy score across the protein. SAP detects exposed hydrophobic patches that drive aggregation. | per-residue: `SAP_i = Σ_{j: |CA_i − CA_j| ≤ 10 Å} (SASA_j / SASA_max_j) · KD(aa_j)` | min | −0.13 .. 5.65 (median 1.59) | Hard `sap_max ≤ 3.0` (configurable per cycle). PTE_i1 used 3.0 in cycle 0 then tightened to 2.5 in cycles 1–2. |
| `sap_mean` | Mean per-residue SAP. | as above, mean | min | −8.7 .. −5.5 (median −7.1) | Negative because most residues are hydrophilic / buried. Diagnostic only. |
| `sap_p95` | 95th-percentile per-residue SAP. | as above, percentile | min | −3.6 .. +2.3 (median −1.4) | Useful when one outlier residue inflates `sap_max`; `sap_p95` shows whether the patch is sustained. |
| `flexibility_mean_seq` | Biopython `flexibility()` mean — Vihinen 1994 normalized B-factor proxy from sequence. | sliding 9-window mean of per-AA flexibility weights, then mean over residues | info | 1.001 .. 1.018 (median 1.009) | No filter; values cluster tightly around 1.0 by design. |
| `molecular_weight` | Biopython monoisotopic MW. Pads applied. | sum of residue MWs | info | 22,180 – 23,910 Da (median 22,630) | Same length so very narrow. Filtered indirectly via `length`. |
| `extinction_280nm_no_disulfide` | Biopython molar ε at 280 nm assuming all Cys reduced. | `5500·n_W + 1490·n_Y + 0·n_C` (M⁻¹cm⁻¹) | info | 1,490 – 17,420 (median 9,970) | Used downstream to set BCA / OD measurement protocols. |
| `extinction_280nm_disulfide` | ε at 280 nm assuming all Cys oxidized to disulfides. | `5500·n_W + 1490·n_Y + 125·(n_C/2)` | info | identical to above for PTE_i1 (no Cys-Cys-pair feasible scaffold) | |

**Caveats**

* The SAP proxy here uses **freesasa** + Kyte–Doolittle, NOT the
  Rosetta SAP score (which uses the SC-SC clash / SASA-derivative
  formulation). The proxy correlates well with Rosetta SAP on
  tested sets but is not a one-for-one substitute. See
  `tools/rosetta_metrics.py` for the real-Rosetta SAP if you need it.
* Instability index is a 1990s in-vitro half-life model and is famously
  noisy on de-novo sequences. Use the threshold `< 60` only as a
  truly-broken-design catch-net. Most PTE_i1 designs sit at 24
  (mean), comfortably stable.
* `extinction_280nm_*` will be identical when no Cys-Cys disulfide is
  geometrically feasible. The two columns are emitted separately for
  back-compat.

---

## 6. Secondary structure (sequence-derived)

These are Biopython `secondary_structure_fraction()` outputs —
sequence-only Chou–Fasman propensities, NOT measured from the structure.
For DSSP-derived per-residue assignments see `tools.secondary_structure`.

| Column | Definition | Direction | Range (PTE_i1) | Filter |
|---|---|---|---|---|
| `helix_frac_seq` | Fraction of residues in helix-prone runs (V, I, Y, F, W, L). | info / target ~ WT | 0.45 .. 0.62 (median 0.53) | No hard filter; PTE WT is α/β so 0.40-0.55 is expected. |
| `turn_frac_seq` | Fraction in turn-prone runs (N, P, G, S). | info | 0.14 .. 0.24 (median 0.21) | No hard filter. |
| `sheet_frac_seq` | Fraction in sheet-prone runs (E, M, A, L). | info | 0.27 .. 0.33 (median 0.30) | No hard filter; PTE β-barrel core wants 0.25-0.35. |

**Quirks**

* These three columns can overlap (a residue can hit two propensity
  classes) and don't sum to 1.0.
* Use these as drift detectors, not as filters: if `helix_frac_seq`
  shifts > 0.15 from WT, the design pool may be losing the fold.

---

## 7. Expression-rule engine

`src/protein_chisel/expression/` runs a curated rule set per design
that flags known *E. coli* expression risks: ssrA / SsrA tag remnants,
Sec / Tat signal peptides, OmpT cleavage motifs, AMP-like stretches,
hydrophobic C-tails, polyproline runs, SecM stalls, internal protease
sites, and per-position structural risks. Each rule fires at one of
four severities: `INFO` < `WARNING` < `SOFT_BIAS` < `HARD_OMIT` <
`HARD_FILTER`.

| Column | Definition | Direction | Range (PTE_i1) | Filter |
|---|---|---|---|---|
| `n_expression_warnings` | Count of rules with severity `WARNING`. Diagnostic only. | min | 1 – 2 (mostly 1) | No hard filter; printed for review. |
| `n_expression_soft_bias_hits` | Count of `SOFT_BIAS` rules. Drives next-cycle MPNN per-position bias. | min | 1 – 4 (median 2) | No hard filter. The hits feed the AA-class-balance bias for the next cycle. |
| `n_expression_hard_omit_hits` | Count of `HARD_OMIT` rules — disallow specific AAs at specific positions in next-cycle MPNN, but the current sequence is not rejected. | min | 0 (PTE_i1 hits no rules at this severity) | No hard filter. |
| `n_expression_hard_filter_hits` | Count of `HARD_FILTER` rules — the sequence is rejected. | hard reject if > 0 | 0 in `all_survivors.tsv` (by definition) | **Rejection criterion**: any `HARD_FILTER` hit drops the design at `stage_seq_filter`. |
| `expression_rule_summary` | `;`-joined `rule_name=SEVERITY` strings for every rule that fired (any severity). | str | e.g. `cytosolic_disulfide_risk=WARNING;sequence_hydrophobic_cterm=SOFT_BIAS` | Search this column to debug why a design was sampled with a particular bias next cycle. |

---

## 8. Sequence-level fitness (PLM)

Three per-residue masked-LM log-probabilities, gathered from cached
ESM-C and SaProt marginals computed once on the seed PDB, then
averaged per design. Implementation in
`src/protein_chisel/sampling/fitness_score.py`.

| Column | Definition | Direction | Range (PTE_i1) | Filter |
|---|---|---|---|---|
| `fitness__logp_esmc_mean` | Mean over positions of `log p(seq[i] \| seed_context)` from ESM-C masked-LM marginals on the SEED PDB. | max (less negative is better) | −2.27 .. −1.93 (median −2.16) | No hard filter; **primary objective** in TOPSIS. |
| `fitness__logp_saprot_mean` | Same as above but using SaProt (structure-aware PLM). Tends to be 0.2-0.3 higher than ESM-C on de-novo backbones. | max | −1.99 .. −1.65 (median −1.90) | Diagnostic; the fused number is what's ranked on. |
| **`fitness__logp_fused_mean`** | **Per-position weighted average using the (β, γ) weights from the PLM-fusion sampler.** This is the production fitness number. | **max** | **−2.12 .. −1.79 (median −2.03)** | Used as the primary axis of `mo_topsis` (weight = 2.0). |
| `fitness__method` | `seed_marginal` (cheap path, used per-cycle) or `rescored` (full forward pass on each candidate; final cycle only if enabled). | info | always `seed_marginal` in PTE_i1 | |

**Caveats**

* The "fitness" here is a **conditional independence approximation**
  to the true sequence likelihood — it gathers the marginal log-prob
  at each position assuming the seed context, not the candidate's own
  context. This is wrong for compensatory mutations (paired residues
  re-scored together can have very different joint log-prob), but it's
  a fine *ranking* signal at zero compute cost beyond a numpy gather
  (~ms per design vs. ~seconds for a real forward pass).
* The fused weights (β = ESM-C, γ = SaProt) are per-position; PLM
  fusion downweights positions where the two PLMs disagree heavily.
  Active-site rows with both weights = 0 fall back to a 0.5/0.5 mix
  so the metric is always defined.
* Fused fitness is **not** scale-comparable across runs with different
  seed PDBs — the absolute log-prob depends on PLM context length,
  number of variable positions, etc. Always interpret `mo_topsis`
  position relative to other designs in the same pool.

---

## 9. Pocket geometry (fpocket)

`scripts/iterative_design_v2.py:_run_fpocket` runs fpocket on each
design, parses the `info.txt` for the most-druggable pocket and
augments it with bottleneck stats from the per-pocket PQR. The
underlying Python wrapper is `src/protein_chisel/tools/fpocket_run.py`.

We always pick the **catalytic-site pocket** (most alpha-spheres
within `distance_cutoff` Å of any catalytic CA) — NOT the most
druggable pocket overall. This protects against fpocket finding
better pockets on the surface that aren't where we want catalysis.

| Column | Definition | Source | Direction | Range (PTE_i1) | Filter |
|---|---|---|---|---|---|
| `fpocket__druggability` | fpocket's `druggability_score` ∈ [0, 1]. Composite of volume, hydrophobicity, sphere density. | info.txt | max | 0.41 .. 0.99 (median 0.94) | Hard `druggability ≥ 0.50` (default `fpocket_druggability_min`). PTE_i1 has this set via the cycle config. |
| `fpocket__volume` | Pocket volume in Å³, derived from alpha-sphere cluster. | info.txt | target ≈ 800–1000 (PTE substrate fits in ~600 Å³ but binding shell wants more headroom) | 385 .. 1814 (median 898) | No hard filter; weak target via TOPSIS. |
| `fpocket__score` | fpocket's general "pocket score" (SVM-like). Different from druggability. | info.txt | max | 0.21 .. 0.68 (median 0.50) | Diagnostic. |
| `fpocket__bottleneck_radius` | Min alpha-sphere radius among the rim spheres (upper-quartile distance from catalytic CA). Approximates the channel constriction a substrate must squeeze past on its way in. | derived from `pocketN_vert.pqr` | target ≈ 3.65 Å (substrate ~ 3.4 Å vdW) | 3.40 .. 3.59 (median 3.44) | No hard filter; soft TOPSIS target. **Below 3.4 Å is a red flag** (substrate cannot enter). |
| `fpocket__min_alpha_sphere_radius` | Absolute smallest alpha-sphere radius in the pocket. fpocket clamps to its [3.0, 6.0] default range, so values <3.4 flag egregious narrow-points. | PQR | min (a high min = uniformly wide pocket) | 3.40 .. 3.51 (median 3.40) | Diagnostic. |
| `fpocket__alpha_sphere_radius_p10` | 10th-percentile sphere radius — robust narrow-point measure (less sensitive to a single outlier than `min`). | PQR | info | 3.47 .. 3.82 (median 3.52) | |
| `fpocket__mean_alpha_sphere_radius` | Mean sphere radius. Larger = more open pocket. | info.txt | min (tighter is better for substrate selectivity, but only down to ~3.5) | 3.79 .. 4.16 (median 3.93) | Used as the legacy tie-breaker for `legacy_rank_score` (smaller wins on ties). |
| `fpocket__alpha_sphere_density` | Number of spheres per unit volume. Higher = denser, well-formed pocket. | info.txt | max | 2.93 .. 9.47 (median 6.90) | Diagnostic. |
| `fpocket__n_alpha_spheres` | Total spheres in the chosen pocket. | info.txt | max (proxy for pocket size + completeness) | 33 .. 200 (median 113) | |
| `fpocket__n_alpha_spheres_near_catalytic` | Count of spheres within `distance_cutoff` (default 8 Å) of any catalytic CA. This is what we use to pick *which* pocket to keep when fpocket finds several. | derived | max | 33 .. 147 (median 90) | **Hard `≥ 1`** implicit (any pocket-near-catalytic sphere is required for the pocket to be selected). |
| `fpocket__mean_alpha_sphere_dist_to_catalytic` | Mean min-distance from each sphere of the kept pocket to nearest catalytic CA. | derived | min (closer = better catalytic alignment) | 3.49 .. 7.42 Å (median 4.33) | Diagnostic. |
| `fpocket__total_sasa` | Total SASA of the pocket-lining atoms (Å²). | info.txt | info / target ≈ 200 | 78 .. 333 (median 185) | |
| `fpocket__polar_sasa` | SASA contributed by polar atoms only. | info.txt | target (depends on substrate H-bond demand) | 18 .. 107 (median 48) | |
| `fpocket__apolar_sasa` | SASA contributed by apolar atoms. | info.txt | target | 57 .. 226 (median 133) | |
| `fpocket__hydrophobicity_score` | fpocket composite hydrophobicity (0–100; higher = more hydrophobic environment). | info.txt | target ≈ 45 (PTE WT) | 18.6 .. 64.0 (median 45.9) | Soft TOPSIS target. |
| `fpocket__polarity_score` | fpocket composite polarity (count of polar / charged residues lining the pocket). | info.txt | info | 3 .. 16 (median 8) | |
| `fpocket__charge_score` | Net charge of pocket-lining residues (count K+R+H − count D+E). | info.txt | info | −2 .. +2 (median −1) | PTE_i1 typically slightly negative — consistent with the OPH active-site water network. |
| `fpocket__volume_score` | fpocket composite volume scoring (0–10). | info.txt | max | 3.66 .. 4.36 (median 4.03) | |
| `fpocket__polar_atoms_pct` | Percentage of pocket-lining atoms that are polar (N, O, S). Equals 100 − `apolar_atoms_pct`. | derived from info.txt's `proportion_of_polar_atoms` | info | 26.9% .. 41.9% (median 32.8%) | |
| `fpocket__apolar_atoms_pct` | 100 − `polar_atoms_pct`. | derived | info | 58.1% .. 73.1% (median 67.2%) | |
| `fpocket__apolar_alpha_sphere_proportion` | Fraction of spheres tagged "C" (apolar) by fpocket. | info.txt | info | 0.40 .. 0.79 (median 0.56) | Complement to polar (below). |
| `fpocket__polar_alpha_sphere_proportion` | Fraction of spheres tagged "O" (polar). Should = 1 − apolar. | derived from PQR | info | 0.21 .. 0.60 (median 0.44) | |
| `fpocket__mean_local_hydrophobic_density` | Mean local atom-density × KD around each pocket-lining atom. Picks up tight hydrophobic clusters (good for substrate vdW). | info.txt | max | 17.5 .. 53.3 (median 34.5) | |
| `fpocket__mean_alpha_sphere_solvent_acc` | Mean solvent-accessibility per sphere (0–1). Lower = more buried pocket. | info.txt | min (buried pocket is harder for water but better for selectivity) | 0.41 .. 0.54 (median 0.48) | |
| `fpocket__cent_of_mass_alpha_sphere_max_dist` | Max sphere distance from the pocket centroid. Approximates pocket "spread"/elongation. | info.txt | info | 6.3 .. 25.3 Å (median 15.2) | Long pockets (>22 Å) are usually surface grooves, not closed pockets — flag for inspection. |
| `fpocket__n_pockets_found` | 1 if fpocket returned ≥1 pocket, 0 otherwise. | derived | max | 1 (always in survivors) | If 0 the design is dropped earlier. |
| `fpocket__n_rim_spheres` | Number of spheres in the upper-distance-quartile (used as the rim for the bottleneck calc). Diagnostic for `bottleneck_radius`. | derived | info | 9 .. 50 (median 28) | |

**Caveats**

* fpocket uses an alpha-sphere cluster algorithm; pockets that are
  particularly long / channel-like get split into multiple pockets and
  the per-pocket metrics underrepresent the true cavity. The
  `n_alpha_spheres_near_catalytic` ≥ 1 selection picks the one closest
  to catalysis, but if your pocket genuinely is a tunnel (e.g.
  P450, dehalogenase) you should also run CAVER (see
  `tools/caver_tunnels.py`).
* fpocket's `min_alpha_sphere_radius` clamps at 3.0 Å — values <3.4
  are effectively the floor and don't differentiate "really tight"
  from "just at the floor". Use `bottleneck_radius` instead.
* The "near-catalytic" distance cutoff defaults to 8 Å. Configurable
  via `fpocket_distance_cutoff_a` in the cycle config. Larger = more
  permissive (catches more pockets); smaller = stricter alignment.

---

## 10. Ligand interaction panel

Cheap geometric protein↔ligand interaction detector,
`src/protein_chisel/tools/geometric_interactions.py`. Heavy-atom-only,
sub-millisecond per design. Each interaction type has a Gaussian
strength score in [0, 1]:

```
strength = exp(− (d − d0)² / (2 σ²))
```

with calibration constants chosen so a canonical optimal interaction
gives ~1.0 and a marginal one ~0:

| Type | d₀ (Å) | σ (Å) | Max distance |
|---|---|---|---|
| H-bond | 2.9 | 0.35 | 3.5 |
| Salt bridge | 3.5 | 0.6 | 4.5 |
| π-π stack | 4.0 | 0.7 | 6.0 |
| π-cation | 4.5 | 0.8 | 6.5 |
| Hydrophobic | 4.0 | 0.7 | 5.0 |
| vdW clash | (— heavy atom < 2.0 Å) | — | 2.0 |

H-bonds use the antecedent-D-A angle to filter false positives without
explicit hydrogens (angle < 70° → the implicit H is roughly between D
and A).

| Column | Definition | Direction | Range (PTE_i1) | Filter |
|---|---|---|---|---|
| `ligand_int__n_total` | Total ligand interactions across all 6 types. | max | 24 .. 59 (median 44) | Diagnostic. |
| `ligand_int__n_hbond` | Count of protein → ligand H-bonds. | max | 10 .. 14 (median 12) | Strong indicator of ligand engagement; PTE catalysis needs the substrate-positioning H-bond network. |
| `ligand_int__strength_hbond` | Sum of H-bond strengths (each in [0, 1]). | max | 8.18 .. 10.49 (median 9.22) | |
| `ligand_int__n_salt_bridge` | Count K-NZ / R-NH ↔ D-OD / E-OE pairs within 4.5 Å of ligand. | max (charge-mediated steering) | 2 .. 5 (median 3) | |
| `ligand_int__strength_salt_bridge` | Sum of salt-bridge strengths. | max | 0.24 .. 3.10 (median 1.15) | |
| `ligand_int__n_pi_pi` | π-stack count. | max | 0 (PTE substrate has no aromatic ring in this seed) | All zeros for PTE_i1 — leave unfiltered. |
| `ligand_int__strength_pi_pi` | Same. | max | 0 | |
| `ligand_int__n_pi_cation` | Aromatic ↔ K-NZ / R-NH within 6.5 Å. | max | 0 in PTE_i1 | |
| `ligand_int__strength_pi_cation` | Same. | max | 0 | |
| `ligand_int__n_hydrophobic` | Protein C ↔ ligand C within 5.0 Å with hydrophobic residue. | max | 8 .. 44 (median 26) | Diagnostic — too many indicates overpacking, too few indicates under-engagement. |
| `ligand_int__strength_hydrophobic` | Sum of hydrophobic strengths. | max | 4.93 .. 30.27 (median 19.46) | |
| `ligand_int__n_vdw_clash` | Heavy-atom pairs < 2.0 Å. | min | constant 2 in PTE_i1 (the ligand has 2 known contact atoms in the seed crystal) | Don't filter on this directly — see `clash__*` for the proper pipeline. |
| `ligand_int__strength_vdw_clash` | Sum of clash strengths (capped at 1 each). | min | constant 2 in PTE_i1 | |
| **`ligand_int__strength_total`** | **Sum of all interaction strengths (incl. clashes — yes, this is a known quirk).** | **max** | **16.13 .. 41.10 (median 31.86)** | Used as a `mo_topsis` axis (weight 1.0). |

**Caveats**

* `strength_total` includes vdW clash contributions (each cmasx 1.0).
  In PTE_i1 the ligand has 2 known seed clashes that show up in every
  design, contributing a constant +2 to `strength_total`. This is fine
  for ranking but bear it in mind when comparing across systems.
* H-bond detection is donor/acceptor type-aware via lookup tables in
  `geometric_interactions.py`. Backbone N is always a donor (except Pro);
  backbone O is always an acceptor.
* `n_hbonds_to_cat_his` (a separate column from this panel — see
  Preorganization below) is the legacy specific check for catalytic-HIS
  H-bonds and is computed independently of `ligand_int__*`.

---

## 11. Preorganization

Geometric interactome around the catalytic residues plus their first-
and second-shell neighbours. Implementation:
`src/protein_chisel/scoring/preorganization.py`.

Shells are defined by min CA-CA distance from any catalytic residue:

* **first shell**: non-catalytic residues with CA ≤ 5.0 Å of any cat. CA
* **second shell**: non-catalytic, non-first-shell residues with CA ≤ 7.0 Å

The same six geometric interactions as `ligand_int__*` are then run
within (catalytic ∪ first ∪ second) and aggregated.

| Column | Definition | Direction | Range (PTE_i1) | Filter |
|---|---|---|---|---|
| `n_hbonds_to_cat_his` | Legacy: count of protein H-bonds to a catalytic-HIS sidechain (independent of preorg). Required ≥ 1 to pass `stage_struct_filter`. | max (require ≥ 1) | 1 .. 3 (median 2) | **Hard `≥ 1`** — designs without any cat-HIS H-bond are rejected by `stage_struct_filter`. Soft TOPSIS axis (weight 0.5). |
| `preorg__n_hbonds_to_cat` | H-bonds from any shell residue to a catalytic residue. Includes the legacy HIS count plus ASP/GLU sidechain coordinations and backbone Hs. | max | 11 .. 13 (median 12) | Diagnostic. |
| `preorg__n_salt_bridges_to_cat` | Salt bridges to catalytic residues. | max | constant 0 in PTE_i1 (no D/E in cat triad) | |
| `preorg__n_pi_to_cat` | π-π or π-cation to catalytic residues. | max | 0 .. 4 (median 2) | |
| `preorg__n_hbonds_within_shells` | H-bonds among shell residues (no catalytic involved). Picks up the second-shell scaffold network that holds the active site rigid. | max | 18 .. 26 (median 20) | |
| **`preorg__strength_total`** | Sum of strengths over ALL detected interactions in (cat ∪ first ∪ second). | max | 23.16 .. 30.96 (median 26.28) | TOPSIS axis (weight 0.7). |
| `preorg__interactome_density` | `(n_hbonds_to_cat + n_salt_bridges_to_cat + n_pi_to_cat + n_hbonds_within_shells) / (n_first_shell + n_second_shell)`. | max | 0.86 .. 1.14 (median 0.97) | |
| `preorg__n_first_shell` | Count of non-cat residues with CA ≤ 5 Å of any cat CA. | info | constant 11 in PTE_i1 (geometry of seed) | Same backbone → same shell membership. |
| `preorg__n_second_shell` | Count of non-cat residues with 5 < CA ≤ 7 Å of any cat CA. | info | constant 24 in PTE_i1 | |

**Caveats**

* The shells are CA-based and seed-fixed. Designs that share a
  backbone (PTE_i1) all have identical `n_first_shell` /
  `n_second_shell` — those columns are constants per run.
* Preorg interactions are **deduplicated** to a canonical
  (atom-pair-sorted, type) key so the all-vs-all detector counts each
  pair once. Don't double-add it to the ligand panel.
* Preorg includes the catalytic↔shell hbonds; if you want only the
  scaffold network use `preorg__n_hbonds_within_shells`.

---

## 12. Clash detection

`src/protein_chisel/structure/clash_check.py` — heavy-atom clash check
between (designed sidechain atoms) and (catalytic heavy atoms,
ligand HETATM). Critical when MPNN samples without seeing catalytic
sidechain context (`--ligand_mpnn_use_side_chain_context=0`): the
side-chain packer may not be able to fit a bulky residue around fixed
catalytic atoms, leaving an irreducible clash.

A "clash" is two heavy atoms within `clash_distance` (default 1.8 Å)
not connected by a covalent bond (we exclude `|ΔresSeq| ≤ 1` to skip
adjacent residues). A "severe" clash is < `severe_distance` (default
1.5 Å).

| Column | Definition | Direction | Range (PTE_i1) | Filter |
|---|---|---|---|---|
| `clash__n_total` | Total clashing pairs. | min | 0 .. 1 (median 0; only 3/124 designs have 1) | Diagnostic; `has_severe` is the actual gate. |
| `clash__n_to_catalytic` | Designed sidechain ↔ catalytic heavy atom clashes. | min | 0 .. 1 (median 0) | |
| `clash__n_to_ligand` | Designed sidechain ↔ ligand HETATM clashes. | min | constant 0 in PTE_i1 | |
| **`clash__has_severe`** | **1 if any clash < `severe_distance` (default 1.5 Å), else 0.** | **min (0 = pass)** | **constant 0 in PTE_i1** | **Hard reject** — `clash_filter=True` and `has_severe=1` triggers a `severe clash …` reason at `stage_struct_filter`. |
| `clash__detail` | First 5 clashes formatted as `<resA>-<resB>(<dist>)`. `-1` means "ligand HETATM" for the second number. | str | mostly empty in PTE_i1 | Search for non-empty values when triaging individual designs. |

**Caveats**

* The check is purely geometric — Cys-Cys disulfide bonds are NOT
  excluded; if the packer places two Cys SG within 1.8 Å it'll be
  counted as a clash. In practice the packer doesn't do this on
  Cys-free scaffolds.
* `severe_distance` is configurable per cycle; the default 1.5 Å is
  generous (real vdW-overlap is more like 1.0 Å for non-bonded atoms),
  set to catch packer artefacts that occasionally produce 1.0–1.4 Å
  collisions.
* The PDB parser is format-aware and reads 5-char `res_name` to handle
  Rosetta-extended names like `HIS_D`, `HIP`, `KCX`.

---

## 13. Dynamic Flexibility Index (DFI)

GNM-based per-residue flexibility, normalized so the protein-mean is
1.0. Implementation: `src/protein_chisel/scoring/dfi.py` (Bahadur &
Ozkan, *PLoS Comp Biol* 2013).

The "DFI score" per residue is `Γ⁺_ii` (the diagonal of the Kirchhoff
pseudo-inverse) divided by the protein-mean. Values >1 = more flexible
than average; <1 = more rigid.

For enzyme design we want the **primary sphere** (catalytic /
substrate-binding shell) to be flexible enough for turnover, and the
**distal buried** framework to be rigid for stability.

| Column | Definition | Direction | Range (PTE_i1) | Filter |
|---|---|---|---|---|
| `dfi__mean` | Protein-mean DFI. By construction = 1.000. | info | constant 1.000 | |
| `dfi__std` | SD of per-residue DFI. | info | constant 0.300 in PTE_i1 (same backbone) | |
| `dfi__max` | Max per-residue DFI. | info | constant 1.912 | |
| `dfi__min` | Min per-residue DFI. | info | constant 0.533 | |
| `dfi__elapsed_ms` | Wall time for the GNM solve (CA-only, ~O(L³)). | info | 10.6 .. 50.4 ms (median 11.3) | |
| `dfi__mean__primary_sphere` | Mean DFI over residues classified as `primary_sphere` (catalytic + first contact shell). Higher = more flexible / turnover-capable. | max (target ≈ 0.9–1.1) | constant 0.811 in PTE_i1 | Same backbone → constant. Across runs target a value not too low (<0.6) — too rigid to flex. |
| `dfi__mean__nearby_surface` | Mean DFI over surface residues near the active site. | info | constant 1.159 | |
| `dfi__mean__secondary_sphere` | Mean over the second contact shell. | info | constant 0.862 | |
| `dfi__mean__distal_buried` | Mean DFI over framework (buried, distal). Lower = more rigid scaffold = better stability. | min | constant 0.793 | Across runs flag designs with `dfi__mean__distal_buried > 1.1` — framework is unstable. |
| `dfi__mean__distal_surface` | Mean DFI over surface, distal residues. | info | constant 1.219 | |
| `dfi__std__*` | SD per class. | info | constants per class | |

**Caveats**

* DFI is computed from CA coords only — same backbone produces the
  same DFI. PTE_i1 uses one fixed seed backbone, so all `dfi__*` values
  are constants. They become useful when the pipeline is run across
  multiple seed PDBs (sweeping fixed-backbone variants) — only then
  do you see real per-design variation.
* Per-class classes (`primary_sphere`, `nearby_surface`,
  `secondary_sphere`, `distal_buried`, `distal_surface`) are computed
  by `tools/classify_positions.py` from the seed PDB. The same column
  set will always be present even if no class has any residues
  (the corresponding mean/std cols just won't be emitted).
* The Kirchhoff cutoff is 10.0 Å (standard GNM). Don't change without
  recalibrating the per-class targets.

---

## 14. AA composition / z-scores

These columns are emitted by the LigandMPNN bias engine for the *next*
cycle (drives `--bias_AA`); they don't appear in
`final_topk/all_survivors.tsv` directly but are visible in the
per-cycle audit logs. Implementation:
`src/protein_chisel/expression/aa_class_balance.py` and
`expression/aa_composition.py`.

The z-score for AA `a` is

```
z_a = (count_a / L − μ_a) / σ_a
```

against the reference distribution
`swissprot_ec3_hydrolases_2026_01` (UniProtKB SwissProt reviewed,
EC 3 hydrolases, snapshot 2026-01). `μ_a` and `σ_a` are computed
per-sequence so z reflects "is this design unusually rich/poor in `a`
vs an average hydrolase".

`compute_class_balanced_bias_AA` then groups AAs into classes
(hydrophobic_aliphatic, aromatic, negatively_charged, positively_charged,
polar_uncharged, small, plus singleton P/G classes) and applies a
**compensatory swap** within each class: if one member is over-rep
(z > +2) and another is under-rep (z < −2), downweight the over- and
upweight the under-, by `bias_per_z · z` (default 0.4 nats/z),
clamped at ±2.5 nats. Singleton classes (P, G) only get downweighted
above z > +3.

The resulting `bias_AA_string` (e.g. `"D:-1.2,E:+0.8,K:-0.6,R:+0.4"`)
is logged per cycle, not per design. `aa_z_scores` (a dict) is
captured in the per-cycle telemetry JSON.

---

## 15. Diversity / position metadata

The driver tracks several diversity / position-classification fields
mostly during the survivor-feed-forward step. These do **not** all end
up in `all_survivors.tsv` but are documented for completeness.

| Field | Source | Notes |
|---|---|---|
| `class` (per-position) | `tools/classify_positions.py` | One of `catalytic`, `primary_sphere`, `secondary_sphere`, `nearby_surface`, `distal_buried`, `distal_surface`. |
| `class_legacy` | older PTE classifier | Back-compat only. |
| `sasa` | freesasa | per-residue solvent-accessible surface area (Å²) on the seed PDB. |
| `dist_ligand` | seed PDB | min CA-to-ligand-heavy-atom distance (Å). |
| `theta_orient_*` | seed PDB | orientation angles of CA-CB vector relative to the ligand centroid (`*` ∈ `phi`, `theta`). Used by `struct_aware_bias.py` to weight per-position MPNN biases. |
| pairwise Hamming | `select_diverse_topk_two_axis` | greedy max-min selection: a candidate must differ at ≥ `min_hamming_full` (default 3) positions from every previously-kept candidate. |
| Hamming on primary-sphere positions | same | second axis: candidate must differ at ≥ `min_hamming_active` positions on the primary-sphere indices alone. Default 0 (off); set to 2 to demand active-site diversity. |
| `hamming_to_WT` / `sequence_identity_to_WT` | not currently emitted | see `iterative_design_v2.py:select_diverse_topk_two_axis` for adding these — they fall out trivially of the same Hamming routine. |

---

## 16. Multi-objective ranking

`src/protein_chisel/scoring/multi_objective.py` — TOPSIS over a basket
of `MetricSpec`s with `max` / `min` / `target` directions. Each spec
contributes a normalized [0, 1] axis with a weight, then we compute
the closeness-to-ideal score:

```
M_w[i, j] = w_j · normalize(values[i, j], spec_j)
ideal_j = max_i M_w[i, j]
anti_j  = min_i M_w[i, j]
TOPSIS_i = ‖M_w[i] − anti‖ / (‖M_w[i] − ideal‖ + ‖M_w[i] − anti‖)
```

For `target` direction, `normalize(v) = 1 − |v − target| / max_dev`.

The default basket (`DEFAULT_METRIC_SPECS`) for PTE-style designs at
2026-05-04:

| Metric | Direction | Weight | Target |
|---|---|---|---|
| `fitness__logp_fused_mean` | max | 2.0 | — |
| `fpocket__druggability` | max | 1.0 | — |
| `ligand_int__strength_total` | max | 1.0 | — |
| `preorg__strength_total` | max | 0.7 | — |
| `n_hbonds_to_cat_his` | max | 0.5 | — |
| `instability_index` | min | 0.5 | — |
| `sap_max` | min | 0.5 | — |
| `boman_index` | target | 0.3 | 2.5 |
| `aliphatic_index` | target | 0.3 | 95.0 |
| `gravy` | target | 0.3 | −0.2 |
| `net_charge_full_HH` | target | 0.3 | −10.0 |
| `pi` | target | 0.3 | 5.5 |
| `fpocket__bottleneck_radius` | target | 0.3 | 3.65 |
| `fpocket__hydrophobicity_score` | target | 0.2 | 45.0 |

| Column | Definition | Direction | Range (PTE_i1) |
|---|---|---|---|
| **`mo_topsis`** | **TOPSIS score over the full final pool. Primary sort key for `all_survivors.tsv`.** | **max** | **0.33 .. 0.78 (median 0.45)** |
| `mo_topsis_cycle` | Per-cycle TOPSIS score (computed for the survivors fed forward into the *next* cycle). NaN for cycle 0. | max | 0.27 .. 0.79 (median 0.56) |
| `legacy_rank_score` | Sum of two ranks: `rank(fitness desc) + rank(mean_alpha_sphere_radius asc)`. The pre-2026-05-04 ranking. Kept for back-compat. | min (lower rank-sum = better) | 9.5 .. 243 (median 128.5) |

**Caveats**

* `mo_topsis` is comparable **within a run** but not across runs —
  the normalization is per-run min/max, so a TOPSIS of 0.6 in one run
  and 0.6 in another don't mean the same thing.
* The ranker silently drops specs whose column is missing (logs a
  warning). If you've configured a custom basket and one column never
  populates (e.g. CAVER didn't run), you'll quietly get the score
  computed without it.
* Weights override is "by-label" or "by-column" via the
  `--rank_weights` / `--rank_targets` CLI args, e.g.
  `--rank_weights "fitness=3.0,sap_max=1.0"`. Setting weight=0 drops
  the metric entirely.
* `mo_topsis_cycle` differs from `mo_topsis` because the cycle TOPSIS
  is on the cycle's pool only, not the full final pool.

---

## 17. Hard-filter pass flags

| Column | Definition | Type | Where | Filter |
|---|---|---|---|---|
| `passed_seq_filter` | True if no `fail_reasons` from `stage_seq_filter` (length, charge band, pI band, instability, GRAVY, aliphatic, boman, expression hard-filter rules). | bool | `stage_seq_filter` | Always True in `all_survivors.tsv` (all rows already passed). |
| `fail_reasons` | `;`-joined reasons that would fail the seq filter. Empty in `all_survivors.tsv`. | str | `stage_seq_filter` | |
| `passed_struct_filter` | True if no `struct_fail` from `stage_struct_filter` (no cat-HIS H-bond, sap_max over cap, severe clash). | bool | `stage_struct_filter` | Always True in `all_survivors.tsv`. |
| `struct_fail` | `;`-joined reasons that would fail the struct filter. | str | `stage_struct_filter` | |

**Note** — both flags are always True for rows in `all_survivors.tsv`
because that file is by construction the post-filter pool. Rejected
rows are at `final_topk/rejects_seq.tsv` and `rejects_struct.tsv` with
the failure reason populated.

---

## 18. Suggested filter ranges for PTE_i1

This is the one-page cheat sheet. Numbers are taken from the
production sweep with the rationale and "when to relax / tighten"
guidance.

### Hard rejection thresholds (drop the design)

| Column | Hard band | Source | Tighten if … | Relax if … |
|---|---|---|---|---|
| `length` | == 202 (WT length) | `stage_seq_filter` | n/a | n/a (length-aware truncation isn't supported) |
| `net_charge_full_HH` | [−15, −4] | `stage_seq_filter:net_charge_max/min` | lysate yields are bad despite passing → tighten to [−12, −5] | sweep needs more diversity → [−18, −2] |
| `pi` | [0, 14] (off — soft band only via target=5.5) | `stage_seq_filter:pi_min/max` | E. coli IB-rate is high → enforce [4.5, 6.5] | tag-cleavage downstream, anything goes |
| `instability_index` | < 60 | `stage_seq_filter:instability_max` | half-life-driven application → < 40 (literature stable threshold) | de-novo / engineered (designs run hot) |
| `gravy` | [−0.8, +0.3] | `stage_seq_filter:gravy_min/max` | aggregation in cell-free → [−0.6, 0] | secreted/membrane-protein design |
| `aliphatic_index` | ≥ 40 | `stage_seq_filter:aliphatic_min` | thermophilic application → ≥ 85 | designed to be cold-active |
| `boman_index` | < 4.5 | `stage_seq_filter:boman_max` | aggregation seen → < 3.0 (closer to native pharma proteins) | designed for self-assembly |
| `n_expression_hard_filter_hits` | == 0 | `expression/engine` | n/a — these are HARD by definition | edit the rule severity to `SOFT_BIAS` |
| `n_hbonds_to_cat_his` | ≥ 1 | `stage_struct_filter` | demand 2 H-bonds for tighter cat positioning → ≥ 2 | the seed already provides — but every PTE_i1 design has ≥ 1 |
| `sap_max` | ≤ 3.0 (cycle 0) / ≤ 2.5 (later cycles) | `stage_struct_filter:sap_max_threshold` | aggregation observed → ≤ 2.0 | exploration phase → ≤ 4.0 |
| `clash__has_severe` | == 0 | `stage_struct_filter:clash_filter` | side-chain packer dependency → keep on | post-rosetta-relax stage may resolve mild clashes |
| `fpocket__druggability` | ≥ 0.50 | final-pool filter via `fpocket_druggability_min` | want only well-formed pockets → ≥ 0.80 | scaffold is large/exotic and 0.5 is too high → 0.30 |
| `fpocket__n_alpha_spheres_near_catalytic` | ≥ 1 (implicit pocket selection) | `_run_fpocket` selection | n/a — a pocket near catalysis is required | n/a |

### Soft TOPSIS targets (enter the multi-objective rank)

These are the `MetricSpec` defaults from
`scoring/multi_objective.py:DEFAULT_METRIC_SPECS`. Override per-run
with `--rank_weights` / `--rank_targets`.

| Column | Direction | Weight | Target (if applicable) | PTE_i1 target rationale |
|---|---|---|---|---|
| `fitness__logp_fused_mean` | max | 2.0 | — | Primary signal. 2× the unit weight. |
| `fpocket__druggability` | max | 1.0 | — | Pocket present and well-formed. |
| `ligand_int__strength_total` | max | 1.0 | — | Sum of all 6 interaction families to ligand. |
| `preorg__strength_total` | max | 0.7 | — | Active-site rigidity / framework support. |
| `n_hbonds_to_cat_his` | max | 0.5 | — | Cat-HIS coordination quality. |
| `instability_index` | min | 0.5 | — | Stability proxy. |
| `sap_max` | min | 0.5 | — | Aggregation proxy. |
| `boman_index` | target | 0.3 | 2.5 | Empirical PTE-class sweet spot — slightly above pharma's 2.0. |
| `aliphatic_index` | target | 0.3 | 95.0 | Mesophilic-thermophilic boundary. |
| `gravy` | target | 0.3 | −0.2 | Slightly hydrophilic, soluble. |
| `net_charge_full_HH` | target | 0.3 | −10.0 | E. coli expression-friendly + away from pI 5.5. |
| `pi` | target | 0.3 | 5.5 | Below physiological → solubility good, away from IEX issues. |
| `fpocket__bottleneck_radius` | target | 0.3 | 3.65 Å | Substrate fits (~ 3.4 Å vdW) with ~ 0.25 Å clearance. |
| `fpocket__hydrophobicity_score` | target | 0.2 | 45.0 | Match WT PTE active-site hydrophobicity. |

### Cycle-config knobs

The above thresholds come from `scripts/iterative_design_v2.py`'s
`stage_seq_filter` / `stage_struct_filter` defaults and the
`DEFAULT_METRIC_SPECS`. They're configurable per cycle via the
`CycleConfig` dataclass in the driver. Common over-rides:

* **More aggressive PTE_i1 cycle 0** — explore widely:
  `sap_max_threshold=4.0, net_charge_max=-2.0, instability_max=70`
* **Tight production cycle 2** — converge:
  `sap_max_threshold=2.0, net_charge_max=-5.0,
  fpocket_druggability_min=0.85`
* **Disable cycle-level TOPSIS, fall back to fitness-only**:
  `use_topsis_for_survivors=False`

### Triage workflow

When investigating a single survivor's score:

1. Look at `mo_topsis` to confirm where it ranks in this run.
2. If `mo_topsis` looks too low for the apparent quality, inspect the
   target-style metrics — a single one wandering outside its
   target band can suppress an otherwise-strong design (each
   target axis at weight 0.2–0.3 still costs up to ~0.05 of the score).
3. For a "why was it accepted" view, `expression_rule_summary` shows
   which expression rules fired (at any severity). `fail_reasons` and
   `struct_fail` are empty for survivors but filled in the rejects
   TSVs.
4. For pocket pathologies, plot `fpocket__bottleneck_radius` vs
   `fpocket__druggability` — designs with druggability > 0.95 but
   bottleneck < 3.42 are "fpocket fooled by a buried, narrow pocket"
   (probably not catalytically useful).
5. For active-site geometry, plot `n_hbonds_to_cat_his` vs
   `preorg__strength_total` — both high = strongly preorganized
   active site.

---

## Worked examples

### Example A: "Why is this design ranked #1?"

Top design from the PTE_i1 production sweep (ID redacted):

```
mo_topsis = 0.777
fitness__logp_fused_mean = -1.79     # near the upper end of [-2.12, -1.79]
fpocket__druggability    = 0.99
ligand_int__strength_total = 38.7
preorg__strength_total   = 30.8
sap_max                  = 0.42
instability_index        = 14.5
boman_index              = 2.51       (target 2.5  → almost perfect)
gravy                    = -0.21      (target -0.2 → essentially perfect)
net_charge_full_HH       = -10.0      (target -10  → exact)
fpocket__bottleneck_radius = 3.65 Å   (target 3.65 → exact)
```

Every primary axis is at or near the top, and every target axis is at
its target. The TOPSIS score reflects this — it's near the ideal point
on every weighted axis simultaneously, which is what produces a 0.78
score (1.0 would mean an unattainable joint optimum).

### Example B: "Why is this design ranked low even though fitness is OK?"

A design with `fitness__logp_fused_mean = -2.05` (close to the median)
might still get `mo_topsis = 0.36` if:

* `sap_max = 5.2` — far above the median 1.59. The min-style axis
  contributes ~0.05 of normalized loss * 0.5 weight ≈ 0.025 lost.
* `boman_index = 3.35` — at the upper end. Target 2.5; |3.35 - 2.5| /
  max_dev ≈ 0.85 normalized distance × 0.3 weight ≈ 0.025 lost.
* `instability_index = 50` — close to the cap. ~0.04 lost via the 0.5
  min weight.
* `fpocket__bottleneck_radius = 3.40` — at the floor.
  |3.40 - 3.65|/0.19 ≈ 1.0 normalized distance × 0.3 weight ≈ 0.03 lost.

These small per-axis penalties accumulate. With 7+ axes contributing
~0.02-0.03 each, the design lands near the bottom of the pool even
though no individual metric triggered a hard filter.

### Example C: "I want to filter to the 10 best for cloning."

```python
import pandas as pd
df = pd.read_csv(
    "final_topk/all_survivors.tsv", sep="\t",
).sort_values("mo_topsis", ascending=False)

# Tighten the filters past the production defaults — only run-out designs
# at this point.
keep = df.query(
    "fpocket__druggability    >= 0.85 and "
    "fpocket__bottleneck_radius >= 3.42 and "
    "n_hbonds_to_cat_his      >= 2     and "
    "preorg__strength_total   >= 26    and "
    "sap_max                  <= 2.5   and "
    "instability_index        <= 35    and "
    "abs(net_charge_full_HH + 10) <= 4"     # within ±4 of target
).head(10)
```

This typically leaves you with 5–15 designs from a 124-survivor pool
with all the hard filters intact, and forces every soft metric into
the upper-quartile band.

---

## Glossary

* **Alpha-sphere** — fpocket's primitive geometry: a sphere tangent to
  4 protein atoms with no other atom inside. Cluster of these = pocket.
* **Antecedent atom** (H-bond detection) — the heavy atom one bond
  upstream of the donor; its vector to the donor defines the implicit
  H direction.
* **β / γ** (PLM fusion) — per-position weights from
  `iterative_fusion.py`'s FusionResult, one each for ESM-C and SaProt.
  They downweight positions where the two PLMs disagree heavily.
* **bottleneck radius** — minimum alpha-sphere radius among the
  upper-quartile-distance-from-catalytic spheres (the rim of the
  pocket). Approximates substrate-channel constriction.
* **HARD_FILTER / HARD_OMIT / SOFT_BIAS / WARNING** — expression-rule
  severities. HARD_FILTER rejects a design; HARD_OMIT bans an AA at a
  position in the next cycle; SOFT_BIAS suppresses an AA softly;
  WARNING is logged-only.
* **Henderson-Hasselbalch (HH)** — the equilibrium model used for
  net-charge calculations. For a positive group: fraction protonated =
  `1 / (1 + 10^(pH − pKa))`.
* **Kirchhoff matrix** (DFI / GNM) — the connectivity matrix Γ where
  `Γ_ij = -1` for in-contact CA pairs, `Γ_ii = degree(i)`. Per-residue
  DFI = `(Γ⁺)_ii / mean((Γ⁺)_ii)`.
* **mo_topsis** — closeness-to-ideal score in [0, 1] over the configured
  basket. 1 = ideal; 0 = anti-ideal. Per-run normalized.
* **preorganization** — geometric interactome around the catalytic
  residues + first/second shell. Higher = stiffer / more stabilizing
  active site.
* **primary sphere / secondary sphere / distal_buried / nearby_surface
  / distal_surface** — position classes from
  `tools/classify_positions.py`. Per-residue tags used by DFI for
  per-class summaries and by struct_aware_bias for per-position MPNN
  bias weighting.
* **Radzicka–Wolfenden ΔG** — water → cyclohexane transfer free
  energies (Biochemistry 1988, 27, 1664), in kcal/mol. Negative =
  hydrophobic. Used in the Boman index.
* **SAP (proxy)** — Spatial Aggregation Propensity proxy. Per-residue
  SAP_i = sum over neighbours within 10 Å of CA(i) of
  `(SASA_j / SASA_max_j) · KD(aa_j)`. The protein-max identifies
  the worst aggregation patch.
* **TOPSIS** — Technique for Order of Preference by Similarity to
  Ideal Solution. Multi-criteria decision analysis.
* **z-score** (AA composition) — how unusually rich/poor a sequence
  is in a given AA, vs the SwissProt EC 3 hydrolase reference.

---

## File index

* `src/protein_chisel/filters/protparam.py` — charge variants,
  pI, GRAVY, instability, aliphatic, boman, aromaticity,
  flexibility, helix/turn/sheet_frac_seq, MW, extinction.
* `src/protein_chisel/structure/clash_check.py` — `clash__*` cols.
* `src/protein_chisel/scoring/preorganization.py` — `preorg__*`.
* `src/protein_chisel/scoring/dfi.py` — `dfi__*`.
* `src/protein_chisel/scoring/multi_objective.py` — TOPSIS,
  `MetricSpec`, `DEFAULT_METRIC_SPECS`, `select_diverse_topk_two_axis`.
* `src/protein_chisel/tools/geometric_interactions.py` — `ligand_int__*`.
* `src/protein_chisel/tools/fpocket_run.py` + the in-driver
  `_run_fpocket` / `_compute_pocket_radius_stats` in
  `scripts/iterative_design_v2.py` — `fpocket__*`.
* `src/protein_chisel/sampling/fitness_score.py` — `fitness__*` cols
  and `seq_hash` / `n_dupes`.
* `src/protein_chisel/expression/aa_class_balance.py` +
  `expression/aa_composition.py` — z-scores and the `bias_AA` string
  used for next-cycle MPNN sampling.
* `scripts/iterative_design_v2.py:stage_seq_filter` —
  `passed_seq_filter` / `fail_reasons` / `n_expression_*`.
* `scripts/iterative_design_v2.py:stage_struct_filter` —
  `passed_struct_filter` / `struct_fail` / `n_hbonds_to_cat_his` /
  `sap_*`.
* `scripts/iterative_design_v2.py` (around L3030–3110) —
  `mo_topsis_cycle`, `mo_topsis`, `legacy_rank_score`, the
  diverse-top-K selection.

---

*Last updated: 2026-05-04, against the production sweep at
`/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260504-054455-127-pid3231842/final_topk/all_survivors.tsv`
(124 designs, 129 columns).*
