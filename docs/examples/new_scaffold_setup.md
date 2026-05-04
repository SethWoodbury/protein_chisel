# Adapting the pipeline to a new target

The reference implementation is hard-coded for PTE_i1 (YYE substrate,
binuclear Zn, KCX-capped K157, catalytic resnos 60/64/128/131/132/157).
Adapting to another scaffold requires touching a small set of inputs +
constants, plus re-calibrating two metric bands.

## Inputs to swap

These three replace cleanly, no code edits required:

1. **Seed PDB** (`--seed_pdb`). Must contain the design backbone +
   ligand HETATM(s) + REMARK 666 / HETNAM / LINK records that
   `iterative_design_v2.py:_pull_remark_666_hetnam_link` will copy
   forward into the packed designs. If your seed lacks those records,
   add them before submitting — restoration is what keeps the
   catalytic geometry stable across cycles.

2. **Ligand .params** (`--ligand_params`). Standard Rosetta params
   file. Used by stage-1 PyRosetta `classify_positions` to identify
   ligand neighbors for the primary/secondary/distal sphere classification
   and by stage-3 PyRosetta repacker for restoring catalytic rotamers.

3. **REMARK 666 / catres spec.** The catalytic resnos baked into
   `iterative_design_v2.py` are PTE-specific:

   ```python
   # scripts/iterative_design_v2.py:73-75
   DEFAULT_CATRES = (60, 64, 128, 131, 132, 157)
   CATALYTIC_HIS_RESNOS = (60, 64, 128, 132)
   CHAIN = "A"
   ```

   For a new scaffold, edit those three constants before calling the
   driver, OR (cleaner) fork `classify_positions_pte_i1.py` →
   `classify_positions_<scaffold>.py` and add a CLI arg to
   `iterative_design_v2.py` that overrides `DEFAULT_CATRES`. At
   present the driver does NOT expose catres on the CLI; this is
   the one edit you'll make.

## Class-balance reference distribution

`stage_bias` in `iterative_design_v2.py` calls
`compute_class_balanced_bias_AA(reference="swissprot_ec3_hydrolases_2026_01")`.
The default reference is the **EC-3 hydrolase Swiss-Prot subset**
(`src/protein_chisel/expression/data/aa_composition_baselines_2026_01.json`,
n=64,973), which is correct for PTE (EC 3.1.8.1) and any other hydrolase.

For a new target:

- **EC-3 hydrolase** (any of esterases, glycosidases, peptidases): keep
  the default. No change needed.
- **Other EC class** (oxidoreductase EC-1, transferase EC-2,
  isomerase EC-5, etc.): change the `reference=` argument in
  `iterative_design_v2.py:2363` to match. Available baselines in
  `aa_composition_baselines_2026_01.json` cover all six top-level EC
  classes plus an `all_swissprot` fallback. Run
  `python -c "import json; print(list(json.load(open('src/protein_chisel/expression/data/aa_composition_baselines_2026_01.json'))['baselines'].keys()))"`
  inside `universal.sif` to enumerate.
- **No EC annotation / unusual chemistry**: use
  `reference="all_swissprot_2026_01"` or accept the EC-3 default with
  a warning in your run log. The class-balance bias is gentle (max ±
  2.5 nats per AA, only fires at z > 2σ) — wrong reference adds noise
  but won't dominate.

## Calibration steps

Stage 1 + stage 2 are scaffold-independent in code: re-run them on
the new seed to regenerate `positions.tsv` + `plm_artifacts/`. They
cost ~30 s + ~60 s GPU; cache hits if you re-run on the same seed.

After the precompute, **inspect the class distribution in
`positions.tsv`**:

```bash
apptainer exec --bind ... /net/software/containers/users/woodbuse/esmc.sif \
    python -c "from protein_chisel.io.schemas import PositionTable; \
               pt = PositionTable.from_parquet('classify/positions.tsv'); \
               print(pt.df[pt.df.is_protein]['class'].value_counts())"
```

Sane PTE-class distribution for L=200:
~5–8 `primary_sphere`, ~15–25 `secondary_sphere`, ~40–80 `distal_buried`,
~80–120 `surface`. If primary_sphere is empty, your ligand isn't
actually contacting protein atoms — check the seed PDB. If everything
is `surface`, your params file isn't being parsed — check the
`HETATM` record names.

### pi band tuning

Default `--pi_min 5.0 --pi_max 7.5` was set on PTE_i1 to select the
least-acidic ~1% of cycle-0 designs at `--net_charge_max -10`. For a
new scaffold:

1. Run a single cycle 0 with `--cycles 1 --pi_min 0 --pi_max 14
   --net_charge_max 50 --net_charge_min -50` (effectively disable the
   bands).
2. Inspect `cycle_00/02_seq_filter/survivors_seq.tsv` `pi` column.
3. Re-run with `--pi_min` ≈ 5th percentile and `--pi_max` ≈ 95th
   percentile of that distribution, OR pick narrower bands matching
   your downstream assay buffer.

### Charge band tuning

Default `[-18, -4]` is calibrated to PTE_i1 surface chemistry +
assay pH 7.8. Re-run cycle 0 with the bands wide and look at the
`net_charge_full_HH` distribution; pick a band that matches your
scaffold's natural charge mode without forcing pathological surfaces.

## Things that ARE PTE-specific

These are NOT covered by simple flag overrides — they're hard-coded
restoration logic in `stage_restore_pdbs` and `compute_catalytic_neighbor_omit_dict`.

### KCX (carboxylated lysine) — catalytic K157

PTE_i1's K157 is post-translationally carboxylated to KCX (a CO2
adduct that bridges the binuclear Zn). The pipeline:

- Treats K157 as `fixed` (never designs it).
- In `stage_restore_pdbs`, restores the KCX cap atoms +
  catalytic hydrogens onto the packed MPNN PDBs (so downstream
  fpocket / clash detection sees the correct hetero atom set).
- Adds the `compute_catalytic_neighbor_omit_dict` rule to forbid K
  and R at PDB resnos 156 and 158, breaking the otherwise unsolvable
  KK-at-157-158 OmpT motif at sample time.

**To disable for a non-KCX scaffold:**

- Remove K157 from `DEFAULT_CATRES`.
- In `stage_restore_pdbs`, the KCX-handling code path is gated on
  `aa == "K" and is_kcx_resno(resno)`; if no resno qualifies, the
  branch is a no-op. No edit strictly required, but the KCX restoration
  code is in `protein_chisel.structure.restore_catalytic` if you ever
  need to disable it explicitly.
- The catalytic-neighbor-omit dict will return `{}` automatically if
  no catalytic resno is `K` or `R`, so K157-related logic stops
  firing once you remove it from `DEFAULT_CATRES`.

### Binuclear Zn handling

PTE_i1's two Zn2+ ions are restored from REMARK 666 / HETNAM / LINK
records via the per-PDB restoration path. As long as your new
scaffold's seed PDB carries the metal records the same way (HETATM
`ZN`, LINK records to coordinating HIS/KCX), restoration will copy
them forward unchanged. No code edit.

**For a non-metalloenzyme target** (apo enzyme, organic-cofactor
target, etc.): no action — there are no metal records to restore, so
the metal-restoration branch is a no-op. The general HETNAM / REMARK
666 / LINK propagation still handles your ligand correctly.

### Catalytic HIS tautomers

`CATALYTIC_HIS_RESNOS = (60, 64, 128, 132)` enumerates the four PTE
HIS residues whose tautomer state (HID / HIE / HIP) is restored after
MPNN packing. If your new scaffold has different catalytic HIS
positions, edit this constant. If you have NO catalytic HIS, set it
to `()` and the tautomer-restoration loop is a no-op. The h-bond
detection in `stage_struct_filter` (`_detect_hbond_to_his_sidechain`)
also keys off this set; with empty tuple, it produces 0 h-bonds for
every design (the column is recorded but doesn't filter).

### Recommended adaptation checklist

1. Edit `DEFAULT_INPUT_PDB`, `DEFAULT_LIG_PARAMS`, `DEFAULT_CATRES`,
   `CATALYTIC_HIS_RESNOS`, `CHAIN` constants in
   `scripts/iterative_design_v2.py:63-75`. (Or pass the first two via
   env / CLI; the rest must be source edits today.)
2. Optionally fork `scripts/classify_positions_pte_i1.py` → your
   scaffold name, change the `--pose_id` default.
3. Update `reference=` in the `compute_class_balanced_bias_AA` call
   if your target is not EC-3.
4. Run `--cycles 1 --pi_min 0 --pi_max 14 --net_charge_max 50
   --net_charge_min -50` once to read off natural pi / charge ranges.
5. Pick `--pi_min`, `--pi_max`, `--net_charge_max`, `--net_charge_min`
   from those distributions.
6. Run a 3-cycle production with the recommended Sweep B knobs from
   `pte_default_run.md` (those defaults — `--plm_strength 1.25
   --strategy annealing --consensus_*`) generalize across scaffolds).
