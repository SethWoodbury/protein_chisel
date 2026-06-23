# `iterative_design.py` CLI reference

Reference for every flag accepted by
`/home/woodbuse/codebase_projects/protein_chisel/scripts/iterative_design.py`
(package version **1.2.0**). The opt-in solubility-steering / controller /
generalizability flags in §13 landed across 1.1.0–1.2.0; in **1.2.0 the
adaptive controller's damping is ON by default** (see `--no_controller_damping`
in §13). Source of truth: the `argparse` block in `scripts/iterative_design.py`
and the env passthrough in `scripts/run_chisel_design.sh`. Always defer to the
live `--help` if it disagrees.

Run via:

```bash
cd /home/woodbuse/codebase_projects/protein_chisel
PYTHONPATH=src:scripts python scripts/iterative_design.py \
    --plm_artifacts_dir <dir> --position_table <csv> [flags...]
```

---

## 1. Required inputs

### `--seed_pdb` (Path)
- **Default:** `/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb`
- **Description:** Reference PDB whose backbone + ligand context drives MPNN sampling. Catalytic residues are read via REMARK 666.
- **Change when:** designing on a non-PTE_i1 scaffold.
- **Example:** `--seed_pdb /path/to/scaffold.pdb`

### `--plm_artifacts_dir` (Path, **required**)
- **Default:** none — must be supplied.
- **Description:** Directory containing `esmc_log_probs.npy`, `saprot_log_probs.npy`, `fusion_bias.npy`, `fusion_weights.npy` (precomputed by the PLM artifact builder).
- **Change when:** every run — point at the artifacts for the current scaffold.
- **Example:** `--plm_artifacts_dir /net/scratch/woodbuse/plm_artifacts/PTE_i1`

### `--position_table` (Path, **required**)
- **Default:** none.
- **Description:** CSV defining design / fixed / catalytic positions and per-position omit lists. Schema in `src/protein_chisel/io/schemas.py::PositionTable`.
- **Change when:** every run.
- **Example:** `--position_table data/PTE_i1/position_table.csv`

### `--ligand_params` (Path)
- **Default:** `/home/woodbuse/testing_space/scaffold_optimization/ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params/YYE.params`
- **Description:** Rosetta `.params` file for the bound ligand. Used by Rosetta-based metrics and the optional Rosetta final stage.
- **Change when:** running on a different ligand.
- **Example:** `--ligand_params data/ligands/MYL.params`

---

## 2. Output

### `--out_root` (Path)
- **Default:** `/net/scratch/woodbuse`
- **Description:** Parent directory for `iterative_design_v2_PTE_i1_<ts>-pid<pid>/` run dirs. Timestamps are millisecond + PID to avoid collisions in parallel sweeps.
- **Change when:** writing somewhere other than the default scratch.
- **Example:** `--out_root /net/scratch/woodbuse/sweep_2026-05-04`

### `--target_k` (int)
- **Default:** `50`
- **Description:** Number of designs returned by the final diverse-top-K stage.
- **Change when:** want fewer/more candidates for downstream screening.
- **Example:** `--target_k 100`

### `--min_hamming` (int)
- **Default:** `3`
- **Description:** Minimum full-sequence Hamming distance enforced between any two designs in the final top-K.
- **Change when:** want stricter (≥5) or looser (1) global diversity.
- **Example:** `--min_hamming 5`

### `--min_hamming_active` (int)
- **Default:** `0` (disabled)
- **Description:** Minimum Hamming on the primary-sphere active-site positions, applied alongside `--min_hamming`.
- **Change when:** require active-site-level diversity even between globally-different designs.
- **Example:** `--min_hamming_active 2`

---

## 3. Sampling

### `--cycles` (int)
- **Default:** `3`
- **Description:** Number of consensus-bias iterations. `1` runs a single short-test cycle; `3` is the production schedule.
- **Change when:** smoke-testing (`1`) or running deeper iteration (rare; not validated > 3).
- **Example:** `--cycles 1`

### `--omit_AA` (str)
- **Default:** `"X"` (UNK only)
- **Description:** Concatenated single-letter codes of AAs MPNN may never sample. `"CX"` is recommended for scaffolds with no catalytic Cys.
- **Change when:** scaffold has a catalytic Cys (use `"X"`); want to forbid Met or aromatics globally (e.g. `"MX"`, `"WYFX"`).
- **Example:** `--omit_AA CX`

### `--use_side_chain_context` (int, choices `{0, 1}`)
- **Default:** `0`
- **Description:** LigandMPNN context flag. `0` = backbone + ligand only (better first-shell diversity; clash-prone bulky AAs auto-omitted). `1` = catalytic sidechain rotamers visible (more WT-conservative).
- **Change when:** designs collapse near WT (use `0`); aggressive samples clash with catalytic packing (use `1`).
- **Example:** `--use_side_chain_context 1`
- **Recommendation (1.4.0 review, F4):** **keep `sc=0`.** Clash is already prevented by the auto-omit of bulky AAs at clash-prone first-shell positions (`compute_clash_prone_first_shell_omits`), and a uniform `sc=1` **collapses first-shell diversity** (it would revert the deliberate v2 `1→0` decision). If — and only if — a specific scaffold shows a **measured** residual clash, prefer the opt-in **`--use_side_chain_context_schedule '1,1,0'`** (sc on early, off late) so the final cycle still samples with full first-shell diversity; never run `sc=1` in the final cycle.

### `--enhance` (str | None)
- **Default:** `None` (base ligand_mpnn)
- **Description:** Optional pLDDT-enhanced fused_mpnn checkpoint. Choices: `plddt_residpo_alpha_20250116-aec4d0c4`, `plddt_residpo_combine_from_timo_100k_20250905-36329ea5`, `plddt_preetham_20241018-5cb969e8`, `plddt_3_20240930-f9c9ea0f`, `plddt_4_20241003-a358098e`, `plddt_16_20240910-b65a33eb`.
- **Change when:** comparing pLDDT-finetuned checkpoints; baseline runs leave at default.
- **Example:** `--enhance plddt_residpo_alpha_20250116-aec4d0c4`

### `--plm_strength` (float, ≥ 0)
- **Default:** `1.25`
- **Description:** Global multiplier on PLM (ESM-C + SaProt) fusion class weights at every position. Empirical sweep on PTE_i1 (rounds 1–5, 2026-05-04) found 1.2–1.3 the sweet spot. `0.0` disables PLM bias entirely.
- **Change when:** want more MPNN structural fidelity (`0.7`); stress-test PLM influence (`1.5+`, diminishing returns and inflated charge SD).
- **Example:** `--plm_strength 1.25`

---

## 4. Strategy

### `--strategy` (str, choices `{"constant", "annealing"}`)
- **Default:** `"annealing"`
- **Description:** Cycle schedule. `constant` = same light-filter thresholds + TOPSIS weights every cycle (the legacy default); `annealing` = light filters loose in cycle 0, tightening to defaults by cycle 2; TOPSIS weights fitness-heavy in cycle 0, balanced cycle 1+; cycles 1+ pick survivors by TOPSIS instead of fitness alone. Hard filters (charge band, pI band, severe clash) are constant under both.
- **Change when:** want broader cycle-0 exploration before tightening.
- **Example:** `--strategy annealing`

### `--consensus_threshold` (float, [0, 1])
- **Default:** `0.90`
- **Description:** Cycle k+1 consensus reinforcement: AA frequency among cycle-k survivors needed before that AA's bias is reinforced. Raise to 0.95 to require stronger agreement and preserve diversity.
- **Change when:** later cycles collapse (raise); want faster convergence (lower, but watch diversity loss — rounds 6/7 at 0.85 lost ~50% pairwise hamming once a class-name bug was fixed).
- **Example:** `--consensus_threshold 0.90`

### `--consensus_strength` (float, nats)
- **Default:** `1.0`
- **Description:** Bias magnitude added at consensus-agreed `(position, AA)` pairs.
- **Change when:** lower (e.g. `0.5`) reduces over-collapse to consensus.
- **Example:** `--consensus_strength 0.5`

### `--consensus_max_fraction` (float, [0, 1])
- **Default:** `0.15`
- **Description:** Maximum fraction of eligible positions consensus may augment per cycle.
- **Change when:** lower (e.g. `0.10`) preserves more positional diversity by reinforcing only the strongest-agreement positions.
- **Example:** `--consensus_max_fraction 0.10`

---

## 5. Charge / pI

### `--design_ph` (float)
- **Default:** `7.8`
- **Description:** pH for charge / pI computations. `7.8` = PTE assay buffer pH 8.0 with safety margin. Robust filter charge uses Henderson-Hasselbalch on K/R/H + D/E/C/Y + termini (Pace 1999 / Bjellqvist 1994 pKas). Four diagnostic variants (no_HIS, HIS_half, DE_KR_only, Biopython) are also recorded.
- **Change when:** designing for a different assay/storage pH.
- **Example:** `--design_ph 7.4`

### `--pi_min` (float)
- **Default:** `5.0`
- **Description:** Minimum theoretical pI. With `net_charge_no_HIS < -10`, default 5.0 selects the least-acidic ~1% of cycle-0 designs. Low cycle-0 pass rate is acceptable: cycle 1+ consensus pulls toward less-acidic sequences.
- **Change when:** relax to `4.7` for higher cycle-0 pass rate at the cost of weaker selection pressure.
- **Example:** `--pi_min 4.7`

### `--pi_max` (float)
- **Default:** `7.5`
- **Description:** Maximum theoretical pI.
- **Change when:** designing toward neutral / mildly basic targets.
- **Example:** `--pi_max 8.0`

---

## 6. Light filters

### `--instability_max` (float)
- **Default:** `60.0`
- **Description:** Guruprasad 1990 instability index upper bound. Lit threshold for native E. coli expression is 40; de novo designs run higher, so default catches only truly broken sequences. Set `9999` to disable.
- **Change when:** strict expression filter (`50`); fully open (`9999`).
- **Example:** `--instability_max 50`

### `--gravy_min` / `--gravy_max` (float, float)
- **Defaults:** `-0.8`, `0.3`
- **Description:** Kyte-Doolittle GRAVY band. Typical soluble proteins fall in `[-0.4, 0]`; default is generous.
- **Change when:** target hydrophilic-only (`gravy_max=0.0`); allow membrane-adjacent (`gravy_max=0.5`).
- **Example:** `--gravy_min -0.5 --gravy_max 0.1`

### `--aliphatic_min` (float)
- **Default:** `40.0`
- **Description:** Ikai 1980 aliphatic-index lower bound. Thermostable native ~85–100; default catches only extreme low-aliphatic outliers.
- **Change when:** want a thermostability proxy (`70+`).
- **Example:** `--aliphatic_min 70`

### `--boman_max` (float)
- **Default:** `4.5`
- **Description:** Boman index upper bound (PPI / sticky propensity). Boman 2003 threshold ~2.5; default catches only extreme cases.
- **Change when:** want stricter sticky-protein filter (`3.5` or `2.5`).
- **Example:** `--boman_max 3.5`

---

## 7. Termini

### `--n_term_pad` (str)
- **Default:** `"MSG"`
- **Description:** N-terminal pad prepended to the design body BEFORE sequence-only metric computation (charge, pI, GRAVY, instability, aliphatic, Boman). `"MSG"` matches a typical E. coli vector tag — actual expressed protein is `M-S-G-[design]-G-S-A`. `""` disables.
- **Change when:** different expression vector / no tag.
- **Example:** `--n_term_pad MSGSHHHHHHSSGLVPRGSHM`

### `--c_term_pad` (str)
- **Default:** `"GSA"`
- **Description:** C-terminal pad — see `--n_term_pad`. `""` disables.
- **Change when:** vector lacks a C-terminal linker.
- **Example:** `--c_term_pad ""`

### `--no_omit_M_at_pos1` (flag)
- **Default:** off (M at body position 1 is hard-omitted)
- **Description:** By default position 1 of the design body is hard-omitted from M (start codon Met lives in the vector tag, not the design). Pass to allow MPNN to sample M there.
- **Change when:** designing without an N-terminal Met-providing tag.
- **Example:** `--no_omit_M_at_pos1`

---

## 8. Class balance

### `--balance_z_threshold` (float)
- **Default:** `2.0`
- **Description:** Class-balanced bias_AA z-cutoff. A swap fires only when one class member is over-represented (`> +z`) AND another under-represented (`< -z`). User notes: 2–3 reasonable; ≤1.5 too aggressive.
- **Change when:** want stronger class rebalancing (`1.5`); lighter touch (`3.0`).
- **Example:** `--balance_z_threshold 2.5`

---

## 9. Multi-objective ranking

### `--rank_weights` (str, `k=v,k=v`)
- **Default:** `""` (use built-in defaults)
- **Description:** TOPSIS weight overrides. Keys: `fitness, druggability, lig_int_strength, preorg_strength, hbonds_to_cat, instability, sap_max, boman, aliphatic, gravy, charge, pi, bottleneck, pocket_hydrophobicity`. Built-in defaults: `fitness=2.0, druggability=1.0, lig_int_strength=1.0, preorg_strength=0.7, hbonds_to_cat=0.5, instability=0.5, sap_max=0.5`; target metrics 0.3 (`boman, aliphatic, gravy, charge, pi, bottleneck`) and 0.2 (`pocket_hydrophobicity`). Set a weight to `0` to drop a metric from ranking.
- **Change when:** rebalancing TOPSIS for an objective you care about.
- **Example:** `--rank_weights druggability=2.0,instability=1.0,gravy=0`

### `--rank_targets` (str, `k=v,k=v`)
- **Default:** `""`
- **Description:** Target-value overrides for target-direction metrics; same keys as `--rank_weights`.
- **Change when:** different ideal charge / aliphatic / boman target than the built-ins.
- **Example:** `--rank_targets aliphatic=100,boman=2.0,charge=-12`

---

## 10. Hard filters

### `--no_clash_filter` (flag)
- **Default:** off (clash filter ENABLED)
- **Description:** Disables the heavy-atom clash check between catalytic + ligand and designed sidechains. Severe clashes (any heavy-atom pair < 1.5 Å) drop the design by default.
- **Change when:** debugging clash false positives only — not for production.
- **Example:** `--no_clash_filter`

### `--fpocket_druggability_min` (float)
- **Default:** `0.30`
- **Description:** Drops designs whose fpocket druggability score on the active-site pocket is below this threshold (no detectable cavity = bad design). Set `0` to disable.
- **Change when:** stricter pocket selection (`0.50`); disable when fpocket misbehaves on a non-standard scaffold (`0`).
- **Example:** `--fpocket_druggability_min 0.50`

---

## 11. Final-stage opt-ins

### `--cms_final` (flag)
- **Default:** off
- **Description:** After `stage_diverse_topk`, runs Coventry Contact Molecular Surface on the top-K only (~3–4 s/design). Adds a `cms__total` column. Requires `esmc.sif`.
- **Change when:** want CMS as an extra metric on the final shortlist.
- **Example:** `--cms_final`

### `--rosetta_final` (flag)
- **Default:** off
- **Description:** After `stage_diverse_topk`, runs the comprehensive Rosetta no-repack metrics panel (DDG + interface energy + Rosetta SAP + …) on the top-K only. ~30–60 s/design. Requires `pyrosetta.sif`.
- **Change when:** ranking the final shortlist with Rosetta scores; skip during fast iteration sweeps.
- **Example:** `--rosetta_final`

---

## 12. Expression

### `--expression_profile` (str)
- **Default:** `"bl21_cytosolic_streptag"`
- **Choices:** `bl21_cytosolic_streptag`, `k12_cytosolic`, `bl21_periplasmic`
- **Description:** Host-expression profile feeding the rule engine (codon / chaperone / signal-peptide / dibasic-cluster / polyproline rules etc.).
- **Change when:** expressing in K12, in periplasm, or any non-default host.
- **Example:** `--expression_profile bl21_periplasmic`

### `--expression_overrides` (str, `rule_name=SEVERITY,...`)
- **Default:** `""`
- **Description:** Comma-separated severity overrides for expression rules. SEVERITY ∈ `{WARN_ONLY, SOFT_BIAS, HARD_OMIT, HARD_FILTER}`.
- **Change when:** want to demote (`WARN_ONLY`) or promote (`HARD_FILTER`) a specific rule for a specific run.
- **Example:** `--expression_overrides kr_neighbor_dibasic=HARD_OMIT,polyproline_stall=WARN_ONLY`

---

## 13. Opt-in solubility steering, adaptive controller & generalizability

Almost every flag in this section is **opt-in and OFF/None by default → byte-identical
to the legacy pipeline when unset**, with a few **default-on / non-`None`-default**
exceptions: the adaptive controller's **damping is ON by default** (1.2.0; opt-OUT via
`--no_controller_damping`); the generalizability flags `--chain` (default `"A"`) and
`--aa_reference` (default EC-3 hydrolases) carry a non-trivial default value that
reproduces the prior hard-coded behavior; and — **new in 1.4.0, a deliberate
default-path change** — the bias-sum safety cap **`--bias_total_clamp` is ON by default at
3.0 nats** (opt-OUT via `--no_bias_total_clamp`). Each flag has an environment-variable
equivalent honored by `scripts/run_chisel_design.sh` (booleans truthy on
`1/true/yes/on`, case-insensitive; value knobs forwarded only when set). The
**generalizability** flags (`--catalytic_resnos`, `--chain`, `--no_require_cat_his_hbond`,
`--aa_reference`) let the pipeline run on **any enzyme**, not just PTE. See
`docs/architecture.md` (adaptive-controller framework + odds-space scaling, the
independent math review, seed triage), `docs/adaptive_bias.md` (the controller in
depth), and the design synthesis in `docs/plans/controller_framework.md`.

### `--catalytic_resnos` (str `R1,R2,...`) — env `CATALYTIC_RESNOS`
- **Default:** `None`.
- **Description:** Comma-separated 1-indexed catalytic resnos (in the design chain, see `--chain`) to fix/protect, e.g. `'41,64,187'`. Resolution priority is **this flag > the seed PDB's `REMARK 666` motif block > the hard-coded PTE_i1 builtin (`60,64,128,131,132,157`)**. When it falls through to the builtin (no override AND no REMARK 666) it emits a **loud warning** naming the PTE residues as almost certainly wrong for a non-PTE scaffold. Set this for ANY non-PTE scaffold whose seed lacks REMARK 666 — otherwise the run pins the wrong residues.
- **Change when:** designing on any non-PTE scaffold whose seed PDB has no REMARK 666 block.
- **Example:** `--catalytic_resnos '41,64,187'` / `CATALYTIC_RESNOS='41,64,187'`

### `--chain` (str `CHAIN_ID`) — env `CHAIN`
- **Default:** `"A"` (byte-identical; was previously hard-coded to `"A"`).
- **Description:** Single-character chain id of the catalytic/design chain in the seed PDB. **Every** structural read uses this chain: H-bond / clash / preorganization / geometric-interaction detection, sequence extraction, secondary structure, and tunnel lining. The env passthrough only forwards `--chain` when `CHAIN != "A"`, so the default is byte-identical.
- **Change when:** the scaffold's design chain is not `A` (e.g. a multi-chain seed where catalysis lives on chain B).
- **Example:** `--chain B` / `CHAIN=B`

### `--no_require_cat_his_hbond` (flag) — env `REQUIRE_CAT_HIS=0`
- **Default:** off → the requirement stays **ON** (byte-identical).
- **Description:** Disables **only** the structural-filter criterion that requires ≥1 side-chain H-bond to a catalytic HIS. Set this for an enzyme whose mechanism has **no** catalytic His — otherwise that criterion rejects every design. The other struct-filter criteria (SAP-proxy, severe-clash gate) are unaffected. The driver flag is `--no_require_cat_his_hbond`; the env form is the inverse boolean `REQUIRE_CAT_HIS` (default `1` = ON; set `REQUIRE_CAT_HIS=0` to forward the flag).
- **Change when:** designing an enzyme that catalyzes without a His (no His-acid/base in the mechanism).
- **Example:** `--no_require_cat_his_hbond` / `REQUIRE_CAT_HIS=0`

### `--aa_reference` (str `NAME`) — env `AA_REFERENCE`
- **Default:** `"swissprot_ec3_hydrolases_2026_01"` (EC-3 hydrolases) → unchanged, so an un-passed flag is **byte-identical**.
- **Description:** AA-composition baseline distribution that the **over-representation checks** score against — both the per-cycle class-balanced `bias_AA` and the adaptive-bias hydrophobic over-rep mask. PTE is a hydrolase, so the default is correct for it; for a non-hydrolase enzyme select the design's **own** EC class (e.g. `swissprot_ec2_transferases_2026_01`, `swissprot_ec1_oxidoreductases_2026_01`, `swissprot_ec4_lyases_2026_01`) or the all-enzyme baseline (`swissprot_enzyme_2026_01`) so compositions are compared to the right distribution rather than to hydrolases. The value is **validated at parse time** against the bundled `REFERENCE_DISTRIBUTIONS` keys — an unknown name errors out listing the valid keys.
- **Change when:** designing any non-hydrolase enzyme (so over-rep z-scores reflect that enzyme class's natural composition).
- **Example:** `--aa_reference swissprot_ec2_transferases_2026_01` / `AA_REFERENCE=swissprot_enzyme_2026_01`

### `--ship_solubility_veto` (flag) — env `SHIP_SOLUBILITY_VETO` (WS-A)
- **Default:** off.
- **Description:** When set, the final top-K writer drops any design **outside the final-cycle GRAVY + net-charge band** before shipping, so the deferred-rescue / backfill path can never ship a seq-filter-failing design (e.g. `GRAVY=1.05`) as rank-0. May ship **fewer than `--target_k`** (intended). Adds a truthful `selection__solubility_passed` column (distinct from `selection__hard_final_filter_passed`, which is the fpocket-druggability gate).
- **Change when:** you need a hard guarantee that no out-of-band design ships, even at the cost of a smaller shortlist.
- **Example:** `--ship_solubility_veto` / `SHIP_SOLUBILITY_VETO=1`

### `--sap_corrected` (flag) — env `SAP_CORRECTED` (WS-B)
- **Default:** off.
- **Description:** Additionally emits `sap_corr_{max,mean,p95}` columns on per-cycle designs — a centered, polar-cancellation-free SAP (shared `scoring.sap` module) that, unlike the legacy signed-Kyte-Doolittle `sap_*`, **registers alanine-rich hydrophobic surfaces and is not cancelled by exposed polar residues**. The legacy `sap_*` columns are unchanged. Rescued/backfill rows carry `NaN` for `sap_corr_*` (they are not re-scored).
- **Change when:** you want a hydrophobic-patch metric that catches Ala-rich surfaces the legacy SAP misses.
- **Example:** `--sap_corrected` / `SAP_CORRECTED=1`

### `--composition_suppress_all_overrep` (flag) — env `COMPOSITION_SUPPRESS_ALL_OVERREP` (WS-C)
- **Default:** off.
- **Description:** In the per-cycle class-balanced `bias_AA`, down-weight **every** over-represented member of an AA class (`z > --balance_z_threshold`), not just the single class maximum. Without it, when Alanine is the hydrophobic-class max an also-over-represented Leucine escapes correction. The property-conserving within-class swap up-weight is preserved.
- **Change when:** a single class has multiple over-represented members and you want all of them suppressed.
- **Example:** `--composition_suppress_all_overrep` / `COMPOSITION_SUPPRESS_ALL_OVERREP=1`

### `--aa_fraction_cap` (float `FRAC` in `(0, 1]`) — env `AA_FRACTION_CAP` (WS-C)
- **Default:** `None`.
- **Description:** Hard-omit any amino acid whose fraction in a cycle's survivor pool is `>= FRAC` (e.g. `0.15`) at every non-fixed designable position in the next cycle, bounding runaway single-AA over-representation (the 26%-Ala failure mode). **Recomputed per cycle**, so an AA is re-allowed once it falls back under the cap. A cap so low it would omit nearly every AA for a low-diversity pool is **skipped that cycle with an ERROR** (it never over-constrains the sampler).
- **Change when:** you want a hard ceiling on any one residue's pool fraction.
- **Example:** `--aa_fraction_cap 0.15` / `AA_FRACTION_CAP=0.15`

### `--composition_soft_bias` (flag) — env `COMPOSITION_SOFT_BIAS` (WS-C)
- **Default:** off.
- **Description:** Activates the expression engine's per-residue `SOFT_BIAS` tier (long-hydrophobic-stretch, KR-near-catalytic-on-helix, polyproline, repetitive-segment, …): adds a negative per-`(position, AA)` bias at those sites to every cycle's sampler bias. **Pool-derived per cycle** — the engine is re-evaluated on the survivors and a liability is applied only if it recurs in `>=` half of them, so it tracks the liabilities the designs actually introduce; the seed map bootstraps cycle 0. Whole-protein composition hits are excluded (those are the suppress-all / fraction-cap levers' job).
- **Change when:** you want sequence-liability motifs steered against at sampling, not just filtered after.
- **Example:** `--composition_soft_bias` / `COMPOSITION_SOFT_BIAS=1`

### `--composition_soft_bias_nats` (float, nats) — env `COMPOSITION_SOFT_BIAS_NATS` (WS-C)
- **Default:** `0.5`.
- **Description:** Down-weight magnitude applied at each `SOFT_BIAS (position, AA)` cell when `--composition_soft_bias` is set. The bias is added in **logit space** (before the softmax temperature divide), so the effective odds penalty is `exp(nats / T)`: at `T≈0.15–0.20` a 0.5-nat bias is ~12–28× (a firm nudge), whereas 1.5 would be ~1800–22000× (a near-hard ban). Keep it near the adaptive controller's ~0.6 clamp. No effect unless `--composition_soft_bias`.
- **Change when:** tuning how firmly the SOFT_BIAS tier nudges (lower = gentler).
- **Example:** `--composition_soft_bias_nats 0.7` / `COMPOSITION_SOFT_BIAS_NATS=0.7`

### `--omit_tunnel_lining` (flag) — env `OMIT_TUNNEL_LINING` (WS-G)
- **Default:** off (experimental).
- **Description:** Hard-omit bulky AAs (`--omit_tunnel_lining_aas`) at the seed's tunnel-lining positions (`is_tunnel_lining`) to keep the substrate channel open from cycle 0. Complementary to the soft, reactive throat-feedback bias; a permanent hard ban is blunter, so this is off by default. Catalytic/fixed positions are never omitted.
- **Change when:** the substrate channel keeps collapsing and the soft throat-feedback bias isn't enough.
- **Example:** `--omit_tunnel_lining` / `OMIT_TUNNEL_LINING=1`

### `--omit_tunnel_lining_aas` (str `AAS`) — env `OMIT_TUNNEL_LINING_AAS` (WS-G)
- **Default:** `FHKRWY` (the throat's bulky-blocker set, `tunnel_metrics._BLOCKER_WEIGHT >= 0.70`).
- **Description:** AAs to hard-omit at tunnel-lining positions when `--omit_tunnel_lining` is set. The default = aromatics W/F/Y/H plus the long charged R/K (Lys Cβ→NZ ~5.5 Å, Arg ~6 Å — genuine channel constrictors, same set as the throat-feedback bias). The medium hydrophobics I/L/M/V are deliberately left to the throat controller's capped/decaying pressure; Ala can't constrict. No effect unless `--omit_tunnel_lining`.
- **Change when:** you want a different channel-blocker set than the bulky default.
- **Example:** `--omit_tunnel_lining_aas WFYH` / `OMIT_TUNNEL_LINING_AAS=WFYH`

### `--plm_autoskip_bad_input` (flag) — env `PLM_AUTOSKIP_BAD_INPUT` (seed triage)
- **Default:** off.
- **Description:** If the **input scaffold** is pathologically hydrophobic / over-represented (per the `--plm_autoskip_*` thresholds below), **force `--plm_strength` to 0 for the whole run** so LigandMPNN regenerates from structure + fixed residues instead of the PLM bias amplifying the bad seed. The PLM fusion is conditioned on the seed, so on a hydrophobic seed it would otherwise amplify the bad composition (a ~1.36-nat fusion peak is a ~8700× lock at `T=0.15`). Empirically on a `GRAVY=1.34` seed this took GRAVY → −0.5 and Ala 27% → 0.5%.
- **Change when:** running on scaffolds that may be pathologically hydrophobic and you want automatic PLM bypass on the bad ones.
- **Example:** `--plm_autoskip_bad_input` / `PLM_AUTOSKIP_BAD_INPUT=1`

### `--plm_autoskip_gravy` (float `G`) — env `PLM_AUTOSKIP_GRAVY` (seed triage)
- **Default:** `0.4`.
- **Description:** Seed-triage GRAVY ceiling — the input trips the auto-skip above this value. No effect unless `--plm_autoskip_bad_input`.
- **Change when:** tuning how hydrophobic a seed must be to trigger the PLM bypass.
- **Example:** `--plm_autoskip_gravy 0.5` / `PLM_AUTOSKIP_GRAVY=0.5`

### `--plm_autoskip_max_aa_frac` (float `F` in `(0, 1]`) — env `PLM_AUTOSKIP_MAX_AA_FRAC` (seed triage)
- **Default:** `0.16`.
- **Description:** Seed-triage single-AA fraction ceiling — an input whose most-frequent residue exceeds this fraction trips the auto-skip. No effect unless `--plm_autoskip_bad_input`.
- **Change when:** tuning the single-AA over-representation trigger.
- **Example:** `--plm_autoskip_max_aa_frac 0.20` / `PLM_AUTOSKIP_MAX_AA_FRAC=0.20`

### `--plm_autoskip_hydrophobic_frac` (float `F` in `(0, 1]`) — env `PLM_AUTOSKIP_HYDROPHOBIC_FRAC` (seed triage)
- **Default:** `0.50`.
- **Description:** Seed-triage hydrophobic-fraction ceiling — an input whose hydrophobic-residue fraction exceeds this trips the auto-skip. No effect unless `--plm_autoskip_bad_input`.
- **Change when:** tuning the overall-hydrophobicity trigger.
- **Example:** `--plm_autoskip_hydrophobic_frac 0.55` / `PLM_AUTOSKIP_HYDROPHOBIC_FRAC=0.55`

### `--plm_autoskip_aa_zmax` (float `Z`) — env `PLM_AUTOSKIP_AA_ZMAX` (seed triage; F1, 1.4.0)
- **Default:** `None` → the z-gate is **off** → byte-identical (the flat `--plm_autoskip_max_aa_frac` is the only single-AA signal).
- **Description:** Adds a **distribution-aware**, per-AA single-AA over-representation signal to the seed triage, **redundant with (ORed to)** the flat `--plm_autoskip_max_aa_frac` so a naturally-abundant AA (Leu ~9.7%, Ala ~8.7%) and a naturally-rare one (Trp ~1.1%, Cys ~1.3%) are judged **fairly** against their own per-AA mean±SD instead of one flat 16% line. An AA trips iff **one-sided `z ≥ Z` (default 3.0) AND `log2_enrichment ≥ --plm_autoskip_aa_log2_floor`** — reusing `aa_composition.aa_z_scores` / `aa_log2_enrichment` (the existing `|z|>3 AND |log2|>0.25` precedent). Flagged AAs both contribute to the pathological verdict (driving the PLM cliff/soft reduction) **and** are armed into the **cycle-0 composition bootstrap** (the `--composition_soft_bias` seed-map), so the composition cap targets them from cycle 0. No effect unless `--plm_autoskip_bad_input`.
- **⚠️ Caveat (read before using):** the z is a **population distance, not a significance test** — it divides by the reference's **between-sequence SD**, so a high `z` means "far from the typical member of this family", **not** "statistically significant". And the **default `--aa_reference` (EC-3 hydrolases) is WRONG for a non-hydrolase seed** — always pass the design's **own** EC class (e.g. `--aa_reference swissprot_ec2_transferases_2026_01`) so the comparison distribution is right. The **log2 floor + one-sidedness + `exclude_aas`** (an already-omitted Cys is dropped) are what keep a legit Trp/Cys/Pro-rich family from being falsely triaged.
- **Change when:** you want a fair, per-AA over-representation trigger (and have set the correct `--aa_reference`).
- **Example:** `--plm_autoskip_bad_input --plm_autoskip_aa_zmax 3.0 --aa_reference swissprot_ec2_transferases_2026_01` / `PLM_AUTOSKIP_BAD_INPUT=1 PLM_AUTOSKIP_AA_ZMAX=3.0 AA_REFERENCE=swissprot_ec2_transferases_2026_01`

### `--plm_autoskip_aa_log2_floor` (float `L`) — env `PLM_AUTOSKIP_AA_LOG2_FLOOR` (seed triage; F1, 1.4.0)
- **Default:** `0.25` (matches the existing `aa_quality_check` `|log2| > 0.25` precedent).
- **Description:** Fold-change **floor** for the z-gate: a flagged AA must ALSO have `log2(design% / ref-global%) ≥ L`, so a naturally-rare AA at high `z` but a trivial absolute % does **not** falsely trip. Only used with `--plm_autoskip_aa_zmax`.
- **Change when:** tuning how large a fold-change is required alongside the z-threshold (raise to require a bigger enrichment).
- **Example:** `--plm_autoskip_aa_log2_floor 0.5` / `PLM_AUTOSKIP_AA_LOG2_FLOOR=0.5`

### `--plm_autoskip_soft` (flag) — env `PLM_AUTOSKIP_SOFT` (seed triage; F2, 1.4.0)
- **Default:** off → the **cliff** (a pathological seed forces `--plm_strength` to 0), the validated default.
- **Description:** Reduce `--plm_strength` **gradually** on a pathological seed instead of the 0/1 cliff: full strength at the trip threshold (severity 1) decaying linearly to 0 at `--plm_autoskip_soft_zero`. **The cliff stays the default** because a soft reduction does **not** rescue a pathological seed — at `T ≈ 0.15` even `plm_strength = 0.4` is ~602× odds, far past the ~8× controller authority (parity with "off" only near `plm_strength ≈ 0.13`). Soft is provided for **cluster A/B comparison**; the realized strength is recorded in `fusion_config.json`. No effect unless `--plm_autoskip_bad_input`.
- **Change when:** running the soft-vs-cliff validation, or you have measured that a gentle PLM reduction outperforms the cliff on borderline seeds.
- **Example:** `--plm_autoskip_bad_input --plm_autoskip_soft` / `PLM_AUTOSKIP_BAD_INPUT=1 PLM_AUTOSKIP_SOFT=1`

### `--plm_autoskip_soft_zero` (float `S`) — env `PLM_AUTOSKIP_SOFT_ZERO` (seed triage; F2, 1.4.0)
- **Default:** `2.0` (twice over the trip threshold).
- **Description:** Severity at which the soft curve reaches `plm_strength = 0`. Only used with `--plm_autoskip_soft`; `S ≤ 1` degrades to the cliff (full strength at the threshold and zero just above it).
- **Change when:** tuning how fast the soft reduction reaches zero (lower = steeper, closer to the cliff).
- **Example:** `--plm_autoskip_soft_zero 1.5` / `PLM_AUTOSKIP_SOFT_ZERO=1.5`

### `--adaptive_bias` (flag) — env `ADAPTIVE_BIAS=1`
- **Default:** off → byte-identical to the legacy pipeline.
- **Description:** Master switch for the closed-loop solubility controller. After each cycle it measures the candidate pool's **net charge** and **surface hydrophobicity** and steers the next cycle's MPNN biases toward target solubility (global D/E up-weight; per-position hydrophobic down-weight at solvent-exposed surface positions). It fires **only when the pool is statistically out of target**, holds the bias once in-band, and reverses on overshoot. All the `--adaptive_bias_*`, `--adaptive_*`, `--controller_*`, and `--no_controller_damping` knobs below have effect **only** when this is set. Full design: `docs/adaptive_bias.md`. The two default axes are `charge` + `surface_hydrophobicity`.
- **Change when:** you want the pipeline to actively regulate charge / surface hydrophobicity across cycles.
- **Example:** `--adaptive_bias` / `ADAPTIVE_BIAS=1`

The base-law tuning knobs (each forwarded by `run_chisel_design.sh` only when its env is set, so an unset knob is byte-identical):

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--adaptive_bias_gain` | `ADAPTIVE_BIAS_GAIN` | `0.6` | initial integral gain (band-normalized error) |
| `--adaptive_bias_max_nats` | `ADAPTIVE_BIAS_MAX_NATS` | `0.6` | raw per-AA / per-cell `|bias|` clamp in nats (superseded by `--adaptive_bias_max_odds` when set) |
| `--adaptive_bias_carry` | `ADAPTIVE_BIAS_CARRY` | `0.9` | integral leak during active correction (`1` = pure integral; held exactly when in-band) |
| `--adaptive_bias_deadband` | `ADAPTIVE_BIAS_DEADBAND` | `0.25` | deadband as a fraction of band half-width (avoids chasing noise) |
| `--adaptive_bias_tmin` | `ADAPTIVE_BIAS_TMIN` | `2.5` | gate `|t|`-stat threshold (pool mean vs target) |
| `--adaptive_bias_fmin` | `ADAPTIVE_BIAS_FMIN` | `0.15` | gate fail-fraction threshold |
| `--adaptive_bias_min_n` | `ADAPTIVE_BIAS_MIN_N` | `30` | minimum pool size to act on an axis |
| `--adaptive_bias_mode` | `ADAPTIVE_BIAS_MODE` | `proportional` | control law: `proportional` (integral) or `bangbang` (provably non-divergent) |
| `--adaptive_bias_seed_from_input` | `ADAPTIVE_BIAS_SEED_FROM_INPUT=1` | off | warm-start cycle 0 from the **input** scaffold's charge/hydrophobicity instead of waiting for cycle-0 output |

WS-D scope / band overrides (default `None` → the legacy `distal_surface` scope + the cycle's net-charge filter band; byte-identical):

- **`--adaptive_surface_sasa_gate FRAC`** — env `ADAPTIVE_SURFACE_SASA_GATE`. When set (e.g. `0.20`), the surface hydrophobic down-weight acts on the `non_tunnel_surface` set — every exposed (sidechain-SASA fraction ≥ FRAC) **non-active-site, non-tunnel-lining, non-fixed** position (a superset of `distal_surface`: adds exposed `nearby_surface`, drops the ligand-distance gate, minus the tunnel mouth). Steers "what you can see by eye."
- **`--adaptive_charge_band LO,HI`** — env `ADAPTIVE_CHARGE_BAND`. Override the controller's net-charge target band, e.g. `-15,-5`. The target becomes the band midpoint and the fail-fraction is evaluated on raw net charge (the cycle-band gap columns no longer apply). Steers net charge to your band without changing the seq filter.
- **`--adaptive_bias_axes LIST`** — env `ADAPTIVE_BIAS_AXES`. Comma list of controller axes to run (default `charge,surface_hydrophobicity`). Restrict (e.g. `charge`) or extend as registry entries land; an unknown axis name is rejected.

### `--no_controller_damping` (flag, opt-OUT) — env `CONTROLLER_DAMPING=0` (v1.2.0)
- **Default:** damping is **ON by default as of 1.2.0** (`--controller_damping` is the default; pass `--no_controller_damping` to revert). No effect without `--adaptive_bias`.
- **Description:** Control-law **damping**, the most important 1.2.0 change. The base (legacy) integral law **under-damped against a *drifting* plant**: it under-corrected, relaxed its integral the moment the pool was momentarily in-band, then ramped hard when the pool drifted back — a live trace limit-cycled charge **−6.8 → −8.8 → −1.3** (a 49× ramp). The damped law instead **held charge stable at the −10 target across mid and hard seeds** (regression: drive swing 0.434 → 0.090, ~4.8×). Damping bundles four stabilizers in one switch:
  - **EWMA of the pool mean** (`measurement_ewma_alpha = 0.5`) — filters per-cycle noise so the controller reacts to the trend, not a single sample.
  - **derivative-on-measurement** (`derivative_gain = 0.5·gain`) — anticipates drift and starts correcting before the hard ramp (measurement-derivative, not setpoint-derivative, so a setpoint change doesn't kick).
  - **per-cycle slew limit** (`slew_limit_frac = 0.15` of `max_nats`) — no full-range lurch in one cycle.
  - **soft 'ramp' deadband** — a continuous drive *through* target, killing the stick-slip of the hard on/off band.
  - Passing `--no_controller_damping` reverts to the legacy under-damped integral law.
- **Why default-ON / deliberate default-path change:** this only affects runs that **already** use `--adaptive_bias`; the `AdaptiveBiasConfig` library defaults stay no-op (the controller unit tests remain byte-identical). It is a deliberate default-path change (like the graded-clash Lys/Arg fix) made after cluster validation + an independent math review.
- **Change when:** you specifically want to reproduce the pre-1.2.0 (under-damped) controller behavior.
- **Example:** `--adaptive_bias --no_controller_damping` / `ADAPTIVE_BIAS=1 CONTROLLER_DAMPING=0`

### `--adaptive_bias_max_odds` (float `X > 1.0`) — env `ADAPTIVE_BIAS_MAX_ODDS` (CF-1)
- **Default:** `None` (keeps the raw `--adaptive_bias_max_nats` clamp → byte-identical).
- **Description:** Clamps the adaptive controller's `|bias|` in **odds space** at `X`-fold instead of in raw nats. It is **temperature-invariant**: each cycle the clamp becomes `max_nats := T·ln(X)`. Because MPNN's odds shift is `exp(bias/T)`, a fixed-nats clamp silently amplifies as `T` anneals; this knob makes the controller's authority invariant to the temperature schedule. Typical `2` (nudge) to `8` (strong). Only takes effect alongside `--adaptive_bias`; the value must be a finite multiplier `> 1.0`.
- **Change when:** running `--adaptive_bias` and you want a stable controller authority across the annealing schedule rather than a raw-nats clamp that tracks `T`.
- **Example:** `--adaptive_bias --adaptive_bias_max_odds 8` / `ADAPTIVE_BIAS=1 ADAPTIVE_BIAS_MAX_ODDS=8`

### `--controller_verbose` (flag) — env `CONTROLLER_VERBOSE` (CF-5)
- **Default:** off.
- **Description:** With `--adaptive_bias`, after each cycle append one row per axis to `<run_dir>/controller_trace.tsv` (long format: cycle, axis, scope, measured vs target/band, signed_error, gate + why, drive `u`, `effective_odds = exp(u/T)`, n) and emit a per-cycle **CONTROLLER REPORT** to the log, for step-by-step chronological validation. **Advisory only** — a trace write error never stops the run. No effect without `--adaptive_bias`.
- **Change when:** validating or debugging the adaptive controller cycle-by-cycle.
- **Example:** `--adaptive_bias --controller_verbose` / `ADAPTIVE_BIAS=1 CONTROLLER_VERBOSE=1`

### `--bias_total_clamp` (float `NATS ≥ 0`) — env `BIAS_TOTAL_CLAMP` (WS-E; ⚠️ default-on in 1.4.0)
- **Default:** **`3.0` nats — ON by default as of 1.4.0** (was `None`). This is a **deliberate default-path change** (like the 1.2.0 damping flip): a bare run now caps the bias stack at 3 nats. Pass an explicit value to override the default, or `--no_bias_total_clamp` to disable it (the exact pre-1.4.0 unclamped path).
- **Description:** Bound the **effective** per-`(position, AA)` sampling bias (`bias_per_residue` + the separately-applied global `bias_AA`) to `±NATS`. The consensus (`+2.0`, uncapped) + PLM-peak (~2.5) stack is otherwise uncapped; at `T ≈ 0.15` that locks a cell near-deterministically (~10¹³× odds), and the pathological double-count stacks to **1e17–1e20×**. Applied to the bias the sampler sees (the `bias.npy` diagnostic stays the un-clamped per-position fusion bias). **Why 3 nats:** it preserves a legit ~3-nat **single-source** PLM peak (so a clean PLM-on run is essentially unaffected) while capping the 6–20-nat double-count lock — a fixed-nats cap is the right semantic here (it allows ≤3 nats at **every** temperature), unlike the odds form or the coordinator's 1e3× whole-stack ceiling, both of which would clip a legit peak. It is a **safety cap, not the un-locker** — the opt-in coordinator (`--controller_coordinator`) remains the budgeted un-locker. **Independent of `--adaptive_bias`** (it guards the whole bias stack on every run, incl. pure-PLM). When the firing telemetry shows it adjusted N `(pos,AA)` cells, that count is the number bound.
- **Change when:** you want a different cap than 3 nats (raise for a stronger legit peak; lower to clamp harder).
- **Example:** `--bias_total_clamp 4.0` / `BIAS_TOTAL_CLAMP=4.0`

### `--no_bias_total_clamp` (flag, opt-OUT) — env `NO_BIAS_TOTAL_CLAMP=1` (1.4.0)
- **Default:** off → the 3-nat cap stays **ON** (the 1.4.0 default).
- **Description:** Opt **out** of the default 3-nat bias-sum safety cap, restoring the **exact pre-1.4.0 unclamped** sampling bias. Mirrors the `--no_controller_damping` idiom. **Mutually exclusive** with an explicit `--bias_total_clamp` / `--bias_total_clamp_odds` (passing both errors at parse time — opt-out vs set-a-value is contradictory).
- **Change when:** you specifically want to reproduce the pre-1.4.0 (unclamped) bias stack.
- **Example:** `--no_bias_total_clamp` / `NO_BIAS_TOTAL_CLAMP=1`

### `--bias_total_clamp_odds` (float `X > 1.0`) — env `BIAS_TOTAL_CLAMP_ODDS` (CF-3a)
- **Default:** `None` → byte-identical. **Mutually exclusive** with `--bias_total_clamp`.
- **Description:** Like `--bias_total_clamp` but the ceiling is expressed in **odds space** at `X`-fold: the per-`(pos, AA)` clamp magnitude becomes `nats_for_odds(X, T) = T·ln(X)` **each cycle**, so "no cell's total bias exceeds `X`-fold odds" holds **invariant to the temperature schedule** (MPNN samples `softmax((logits + bias)/T)`, so a fixed-nats clamp silently tracks `T` as it anneals). Must be `> 1.0` (an odds ceiling ≤ 1 is a non-positive clamp). Suggested `~8–100` (a stacking guard). Same temperature-invariance principle as `--adaptive_bias_max_odds`, applied to the **combined** bias instead of the controller's contribution.
- **Change when:** you want the bias-stack cap to track the annealing temperature rather than a fixed nats budget.
- **Example:** `--bias_total_clamp_odds 100` / `BIAS_TOTAL_CLAMP_ODDS=100`

### `--sampling_temperature_floor` (float `T`, ≤ 2.0) — env `SAMPLING_TEMPERATURE_FLOOR` (WS-E)
- **Default:** `None` → byte-identical.
- **Description:** Raise any cycle's sampling temperature to at least `T` (overriding the annealing schedule), applied once after the schedule is built so the sampler and the logged temperature agree. At `T ≈ 0.15` a 0.5-nat bias is ~28× (near-deterministic); `~0.3` restores genuine multinomial diversity. **No effect under the PoE backend** (warned).
- **Change when:** the annealed low temperature is collapsing diversity and you want a multinomial floor.
- **Example:** `--sampling_temperature_floor 0.3` / `SAMPLING_TEMPERATURE_FLOOR=0.3`

### `--plm_class_strength` (str `K=V,...`) — env `PLM_CLASS_STRENGTH` (WS-E)
- **Default:** `""` → byte-identical.
- **Description:** **Absolute** per-class overrides of the PLM-fusion class weights, e.g. `distal_surface=0.3,primary_sphere=0.0`. The global `--plm_strength` still multiplies on top. Unknown class names and non-finite/negative values are rejected.
- **Change when:** you want to retune (or zero) the PLM influence on a specific position class without changing the global strength.
- **Example:** `--plm_class_strength distal_surface=0.3,primary_sphere=0.0` / `PLM_CLASS_STRENGTH='distal_surface=0.3'`

---

## Recommended config presets

### PTE_i1 production (best from tonight, 2026-05-04)
```bash
--strategy annealing \
--plm_strength 1.25 \
--consensus_threshold 0.90 \
--consensus_strength 1.0 \
--consensus_max_fraction 0.15
```

### Diversity exploration
```bash
--strategy annealing \
--plm_strength 1.25 \
--consensus_threshold 0.95 \
--consensus_strength 0.5 \
--consensus_max_fraction 0.10
```

### Strict filtering (add to either preset above)
```bash
--instability_max 50 \
--boman_max 3.5
```

### CPU-only run
Identical command line, run via `apptainer exec` WITHOUT the `--nv` GPU flag.
MPNN forward passes will fall back to CPU (slower, but functional). Example:

```bash
apptainer exec /net/software/containers/universal.sif \
    python scripts/iterative_design.py \
    --plm_artifacts_dir <dir> --position_table <csv> \
    --strategy annealing --plm_strength 1.25 \
    --consensus_threshold 0.90 --consensus_strength 1.0 \
    --consensus_max_fraction 0.15
```

---

## Notes

- All paths are absolute. Relative paths are resolved against the cwd at launch.
- Run dirs are timestamped `iterative_design_v2_PTE_i1_<YYYYMMDD-HHMMSS-mmm>-pid<PID>` to prevent collisions in parallel sweeps.
- `--plm_strength` is validated: negative values error out; values > 5.0 emit a warning (PLM bias may dominate MPNN's structure-conditioned logits).
- Hard filters (charge band, pI band, severe clash) are constant across cycles under both strategies — only light filters and TOPSIS weights anneal.
