# `iterative_design_v2.py` CLI reference

Reference for every flag accepted by
`/home/woodbuse/codebase_projects/protein_chisel/scripts/iterative_design_v2.py`.
Defaults reflect the source on `main` as of 2026-05-04. Source of truth:
`scripts/iterative_design_v2.py`, lines 2536–2725.

Run via:

```bash
cd /home/woodbuse/codebase_projects/protein_chisel
PYTHONPATH=src:scripts python scripts/iterative_design_v2.py \
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
- **Default:** `"constant"`
- **Description:** Cycle schedule. `constant` = same light-filter thresholds + TOPSIS weights every cycle; `annealing` = light filters loose in cycle 0, tightening to defaults by cycle 2; TOPSIS weights fitness-heavy in cycle 0, balanced cycle 1+; cycles 1+ pick survivors by TOPSIS instead of fitness alone. Hard filters (charge band, pI band, severe clash) are constant under both.
- **Change when:** want broader cycle-0 exploration before tightening.
- **Example:** `--strategy annealing`

### `--consensus_threshold` (float, [0, 1])
- **Default:** `0.85`
- **Description:** Cycle k+1 consensus reinforcement: AA frequency among cycle-k survivors needed before that AA's bias is reinforced. Raise to 0.90+ to require stronger agreement and preserve diversity.
- **Change when:** later cycles collapse (raise); want faster convergence (lower, but watch diversity loss — rounds 6/7 at 0.85 lost ~50% pairwise hamming once a class-name bug was fixed).
- **Example:** `--consensus_threshold 0.90`

### `--consensus_strength` (float, nats)
- **Default:** `2.0`
- **Description:** Bias magnitude added at consensus-agreed `(position, AA)` pairs.
- **Change when:** lower (e.g. `1.0`) reduces over-collapse to consensus.
- **Example:** `--consensus_strength 1.0`

### `--consensus_max_fraction` (float, [0, 1])
- **Default:** `0.30`
- **Description:** Maximum fraction of eligible positions consensus may augment per cycle.
- **Change when:** lower (e.g. `0.15`) preserves more positional diversity by reinforcing only the strongest-agreement positions.
- **Example:** `--consensus_max_fraction 0.15`

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
    python scripts/iterative_design_v2.py \
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
