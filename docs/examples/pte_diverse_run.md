# Diversity-heavy PTE_i1 run

A variant of the standard production run that prioritizes sequence
diversity over consensus-driven fitness recovery. Use when seeding
early-stage exploration of new active-site arrangements, or when
downstream wet-lab work needs a less-correlated panel.

## Command

Same scaffolding as `pte_default_run.md`, with three knobs softened:

```bash
SEED_PDB=/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb
LIG_PARAMS=/home/woodbuse/testing_space/scaffold_optimization/ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params/YYE.params

# stage 3 (only this stage differs from the standard run)
apptainer exec --nv \
    --bind "$REPO:/code" --bind /net/software --bind /net/databases \
    --bind /net/scratch --bind /home/woodbuse \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    /net/software/containers/universal.sif \
    python "$REPO/scripts/iterative_design_v2.py" \
        --seed_pdb "$SEED_PDB" \
        --ligand_params "$LIG_PARAMS" \
        --plm_artifacts_dir "$PLM_DIR" \
        --position_table "$CLASSIFY_DIR/positions.tsv" \
        --out_root /net/scratch/woodbuse \
        --target_k 50 \
        --min_hamming 3 \
        --min_hamming_active 2 \
        --cycles 3 \
        --omit_AA CX \
        --plm_strength 1.25 \
        --strategy annealing \
        --consensus_threshold 0.95 \
        --consensus_strength 0.5 \
        --consensus_max_fraction 0.10
```

## What changes vs the default

| flag | default-run | diverse-run | effect |
|---|---|---|---|
| `--consensus_threshold` | 0.90 | **0.95** | even stronger cross-survivor agreement before reinforcement; only ~true consensus positions get a bump |
| `--consensus_strength` | 1.0 | **0.5** | half the per-(pos, AA) bias magnitude — barely a nudge |
| `--consensus_max_fraction` | 0.15 | **0.10** | tighter cap: at most 10% of L can be reinforced per cycle |

The combined effect is that consensus reinforcement becomes a *very
gentle* exploitation channel. Cycle 1+ still benefits from "AAs the
survivors agree on", but doesn't collapse the search space toward
those AAs the way the default settings do.

## When to use

- **First run on a new scaffold** — you have no priors yet on which
  AAs the survivors will converge to, and aggressive consensus would
  amplify whatever cycle 0 happens to find (good or bad).
- **Hypothesis-generation panels** — you're seeding downstream wet-lab
  work and want a panel of mechanistically distinct candidates rather
  than 50 minor variants on one optimum.
- **Diagnosing collapse** — if a previous default run produced a
  top-K with low pairwise Hamming or low primary-sphere diversity
  (the "rounds 6/7 with consensus_threshold 0.85 lost ~50% of pairwise
  Hamming" failure mode), this is the safe re-run to pull diversity
  back.
- **Comparing PLM influence** — paired with the default run and a
  `--plm_strength 0.7` run, this is the natural "low PLM, low
  consensus" point in a 2x2 sweep.

## When NOT to use

- **Final production rounds** — once you've validated which AAs the
  good survivors carry, the default consensus settings are deliberately
  tuned to amplify them. Diverse settings will produce a more
  scattered top-K with lower mean fitness.
- **When you've already filtered the panel down to a tight design
  space** — no need for diversity once the structural / catalytic
  requirements are doing the constraining.

## Expected impact on metrics

Compared to the default run on the same seed:

- Top-K mean fitness drops by ~0.05–0.10 nats/residue.
- Pairwise full-sequence Hamming on the top-K rises by ~30–50%.
- Primary-sphere unique-AA count typically rises from ~6 to 8–10.
- Druggability distribution widens; the median drops slightly but the
  top decile is comparable.
- Charge / pi / instability bands are unchanged (those are HARD
  filters and don't anneal in either run).

## Pairing with default

A common pattern: run the diverse panel first, pick the best
mechanistic clusters by hand, then run the default panel from each
cluster representative as a new seed. The diverse run is the cheap
exploration; the default is the cheap exploitation.
