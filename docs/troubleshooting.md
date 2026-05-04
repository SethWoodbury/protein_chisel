# Troubleshooting

This document collects known issues, fixes, and operational caveats for `protein_chisel`. Most entries trace back to specific commits or in-code comments; check git blame on the referenced files when in doubt.

---

## Known Issues

### 1. DAlphaBall / `libgfortran.so.5` failures in some apptainer containers

**Symptom.** Rosetta-based SASA computation aborts during position classification with an error similar to:

```
error while loading shared libraries: libgfortran.so.5: cannot open shared object file
```

**Cause.** Some apptainer images we run inside (notably older `rosetta.sif` builds) ship without the gfortran 5 runtime that DAlphaBall is linked against. Rosetta then refuses to run the SASA stage even though every other step works fine.

**Workaround.** `tools/classify_positions.py` auto-detects the failure and falls back to the `freesasa` Python bindings, which give equivalent SASA values for our purposes (correlation > 0.99 against Rosetta SASA on the test set). The fallback engages silently; you will see a single `[classify_positions] using freesasa fallback` line in the log. If you want the Rosetta path, use a container that bundles `libgfortran5` (the `universal.sif` image is fine).

---

### 2. Concurrent `run_dir` collisions

**Symptom (pre-fix).** When two SLURM jobs started in the same wall-clock second, both wrote into the same `run_dir`, clobbering each other's `seqs.fasta`, `position_table.parquet`, and metric outputs.

**Fix.** Commit `8c298cc` changed the run-directory timestamp from `YYYYMMDD-HHMMSS` to `YYYYMMDD-HHMMSS-<ms>-<PID>`. Collisions now require two jobs in the same millisecond *and* the same PID, which is essentially impossible.

**If you see old runs.** Anything before `8c298cc` may have silently merged outputs across jobs. Treat per-design metrics from those runs as suspect if you launched array jobs.

---

### 3. Position-table schema migration (5-class -> directional 6-class)

**Symptom.** Loading a `position_table.parquet` from before the directional taxonomy lands and you get a warning:

```
[position_table] legacy 5-class schema detected; reclassifying to directional 6-class
```

**Cause.** The original taxonomy was `{active_site, first_shell, pocket, buried, surface}`. The directional taxonomy adds *vector* information (does the side-chain point into the pocket, away from it, or laterally?), so the new classes are `{active_site, first_shell_in, first_shell_out, pocket_lining, buried, surface}`.

**Behavior.** Legacy tables are auto-reclassified on load. The reclassification config (cutoffs, vector projection threshold, neighborhood radius) is *stored back into the parquet file* under the `_classify_config` metadata key so that downstream consumers always see the same labels. Reproducible by design.

**If you do not want migration.** Pass `--no_migrate_position_table` to the loader; you will then get the legacy 5-class labels back, but most downstream filters will not understand them.

---

### 4. AA composition skew toward Glu (E)

**Symptom.** Across every run we have looked at so far, designed sequences over-use glutamate. Typical counts: E = 38-40 vs WT = 18. The other charged residues (D, K, R) are within +/- 2 of WT.

**Causes (multiple, additive).**

- **Charge filter pressure.** We require net charge in a band that excludes the WT (the WT has too many lysines for the binding-pocket electrostatics we want). The cheapest way for ProteinMPNN to satisfy that band is to add E.
- **MPNN's alpha-helix Glu prior.** ProteinMPNN was trained on a corpus where Glu is over-represented in alpha-helical positions, and our scaffold is helix-heavy. The model's marginal at exposed helical positions is genuinely Glu-biased.
- **Scaffold geometry.** The first-shell-out positions on this scaffold happen to be mostly helical and mostly solvent-exposed, which is exactly the regime where MPNN samples Glu most aggressively.

**Mitigation.** The class-balance fitness term penalizes residues whose per-class frequency is more than `balance_z_threshold` standard deviations above the EC-3 hydrolase reference. Once Glu hits z > 3 the *extreme-over fallback* engages and caps it hard.

**Tuning.** `--balance_z_threshold` (default `3.0`). Lower it (e.g. `2.0`) if you want to clamp Glu earlier; raise it if you are getting too aggressive penalties on residues that are legitimately enriched.

---

### 5. Position 1 always Met (M) before fix

**Symptom (pre-fix).** Every single design had Met at position 1. 100%, no variation.

**Cause.** The MSG vector tag (`Met-Ser-Gly...`) is *prepended* to the design body before MPNN scoring. ProteinMPNN sees the M at position 0 of the full chain and copies it into position 0 of the design body too because the marginal at the N-terminal cap is dominated by the start codon Met in the training set. The design body is *not* supposed to start with M - that M lives in the vector tag, not the protein.

**Fix.** Position 1 of the design body is now hard-omitted from M by default. The omit-list is set inside the sampling loop before MPNN is called.

**Disable.** `--no_omit_M_at_pos1` if you have a scaffold where the design body genuinely starts with the start codon (i.e. you are *not* using a vector tag).

---

### 6. Diversity collapse with default consensus (rounds 6-7)

**Symptom.** After we fixed the consensus-reinforcement bug (consensus was previously not being applied at all on most rounds), diversity in the surviving pool started collapsing around round 6-7 with the default settings. By round 8 most designs were within Hamming-3 of each other.

**Cause.** The default consensus parameters (`threshold=0.85`, `strength=2.0`, `max_frac=0.30`) are aggressive. They were tuned for *small, focused* design pools where you want fast convergence on a single mode. For diverse pools they collapse the population onto whichever mode happens to be ahead at round 5.

**Recommended for diverse pools.**

```
--consensus_threshold 0.90 \
--consensus_strength  1.0 \
--consensus_max_fraction 0.15
```

In English: only reinforce a position if 90%+ of survivors agree on it (was 85%), reinforce by half as much (was 2.0), and never reinforce more than 15% of positions in any one round (was 30%).

**Symptom of the right tuning.** Round-over-round mean Hamming distance to round-0 should stay above 8 through round 10. If it drops below 5, your consensus is too aggressive.

---

### 7. DFI is design-invariant for fixed-backbone designs

**Known limitation.** DFI (Dynamic Flexibility Index) is computed from a coarse-grained Gaussian Network Model on the CA trace. For fixed-backbone designs the CA trace is *identical* across every design - we are only changing side chains. So DFI gives the same answer for every design in the run. It cannot rank designs.

**What it is still good for.** Seed-level QC. If your seed backbone has DFI hot spots in the wrong places (e.g. a totally rigid active-site loop, or a wildly flexible catalytic helix), that is a real signal and you should pick a different seed. Compute DFI *once per seed*, never per-design.

**Future work.** A backbone-aware DFI variant (using predicted vs reference structures) would give a per-design signal, but is not implemented.

---

### 8. Expression engine sees unpadded design body; ProtParam sees padded

**Deliberate trade-off.** Documented in the docstring of `stage_seq_filter` in the pipeline module.

- **Expression-prediction engine** is given the bare design body, no vector tag, no padding. It was trained on bare ORFs and scoring it on padded sequences shifts its predictions by a constant.
- **ProtParam (instability index, GRAVY, charge)** is given the *padded* sequence (vector tag + design body + any C-terminal padding) because the user is going to express that exact construct, and the physicochemistry of the construct is what matters in the wet lab.

**Gap.** Junction-induced motifs - things like a polybasic stretch that only exists because the vector-tag end butts up against the design-body start - are *not caught*. The expression engine never sees the junction, and ProtParam aggregates over the whole sequence so a 4-residue motif at the junction is invisible in the bulk metrics.

**Mitigation.** If you are paranoid, scan the padded sequence yourself for junction motifs. There is no automated check for this in the current pipeline.

---

### 9. Cached PLM artifacts (`fusion_bias.npy`) become stale when class weights change

**Symptom (old).** You change `--class_weights` and re-run, but the fitness function gives the same answer as the previous run. Confused; pull hair.

**Cause.** `fusion_bias.npy` is a per-position bias vector derived from the PLM (ESM2) and the class weights. It used to be cached on disk and reloaded blindly. If you changed the class weights without changing anything else, the cache key did not invalidate and the stale bias was reused.

**Fix.** The driver now *re-fuses* the bias at runtime instead of trusting the on-disk cache. The cached value is still loaded, but it is shown alongside the runtime value at startup as `[fusion_bias] cached=...  runtime=...  drift=...` so you can see when the two have diverged. If `drift > 0` and you did not expect drift, your class weights changed.

**No flag needed.** Just check the startup log.

---

### 10. HIS protonation and metal coordination

**Why this is hard.** HIS pKa is one of the most context-dependent quantities in protein chemistry. In a binuclear-Zn active site coordinating a YYE ligand (our case), the apparent pKa of the coordinating histidines can be anywhere from 4.5 to 8.5 depending on Zn occupancy, second-shell hydrogen bonding, and solvent. There is no single "right" charge state.

**What we provide.** Five charge variants per design:

- `full_HH` - all HIS doubly protonated (HIP)
- `no_HIS`  - all HIS deprotonated/neutral (HIE/HID, default tautomer)
- `HIS_half` - exactly the metal-coordinating HIS deprotonated, all others HIP
- `HIE_all` - all HIS as HIE (epsilon-protonated neutral)
- `HID_all` - all HIS as HID (delta-protonated neutral)

You pick the one that matches your wet-lab pH and your beliefs about coordination geometry.

**PROPKA is available, but not auto-used.** PROPKA3 is bundled in `universal.sif` and runs in roughly 160 ms/design. We do not call it automatically because it has trouble parameterizing the binuclear Zn + YYE ligand without a manual ligand-parameter file - PROPKA's default ligand handling assigns wrong coordination numbers and the resulting HIS pKa values are off by a full pH unit in our hands.

**If you want PROPKA.** Build the ligand parameter file once for your active site, drop it in `external/propka_params/`, and set `--use_propka`. The pipeline will then use PROPKA-predicted protonation per design instead of the five-variant enumeration.

---

## Strengths

- **Directional 6-class position taxonomy.** Splits first-shell into "in" (pointing into pocket) vs "out" (pointing away). Captures information that the original 5-class scheme threw away.
- **Multi-objective ranking.** Pareto-style ranking across structure, expression, charge, AA composition, and (where applicable) catalytic geometry. No single hard-coded scalarization.
- **Consensus reinforcement.** When tuned correctly (see issue 6) it gives smooth convergence on the design landscape without collapsing diversity.
- **Diversity controls.** Hamming-distance floors, per-round Boltzmann temperature scheduling, and class-balance penalties prevent the pool from collapsing onto a single mode.
- **CPU-only pipeline works end-to-end.** No GPU required for the core design loop. ProteinMPNN runs on CPU at acceptable throughput; ESM2 fusion bias is computed once per round on CPU.
- **Comprehensive metrics.** Per-design we record SASA, DFI (seed-level), MPNN log-likelihood, ESM2 pseudo-perplexity, PLM fusion bias, instability index, GRAVY, net charge at 5 pH points, AA composition z-scores per class, and the catalytic geometry score where applicable.
- **Highly tunable.** Roughly 40 CLI flags expose every tunable knob. Defaults are conservative.
- **Scaffold-extensible.** Adding a new scaffold means writing a `scaffold.yaml` (anchors, ligand, vector tag, charge band). No code changes for new targets in the same family.
- **Reproducible.** Position-table reclassification config is stored in the parquet metadata; PLM fusion bias drift is reported at startup; every run gets a unique `run_dir`; all CLI args are dumped to `run_args.json`.
- **Provenance-aware.** Every per-design metric carries the engine name and version that produced it, so you can re-derive any value from raw outputs.

---

## Weaknesses

- **Defaults are tuned for PTE.** Consensus parameters, charge band, and class weights all have history with the PTE (organophosphate hydrolase) target. New targets will likely need at least the consensus parameters retuned (see issue 6).
- **AA composition reference is EC-3 hydrolases only.** The class-balance z-scores are computed against an EC-3 (hydrolase) reference frequency table. For non-hydrolase targets the reference is the wrong distribution and the balance penalty will fight you.
- **Fitness is ranking-only, not absolute.** The composite fitness score is meaningful for ordering designs *within a run*. Comparing absolute fitness values across runs (especially across different seeds or different scaffolds) is not meaningful - the components are normalized per-run.
- **No cross-design DFI signal.** See issue 7. DFI tells you something about the seed, nothing about which design within a seed is better.
- **Junction-motif gap.** See issue 8. Anything induced by the vector-tag/design-body boundary is invisible to both the expression engine and to ProtParam.

---

## Reporting new issues

Please include in any bug report:

1. The exact `run_dir` (so we can pull `run_args.json`).
2. The `git rev-parse HEAD` of the codebase at run time.
3. Which apptainer image was used.
4. The first 50 lines and last 100 lines of the SLURM `.out` file.
5. The `position_table.parquet` schema version (printed in the startup log).
