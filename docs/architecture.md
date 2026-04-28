# Architecture

## Three layers

```
pipelines/   ← orchestrators: chain tools with file-based handoffs.
   │
   ▼
tools/       ← single-purpose primitives, runnable on their own.
   │  uses
   ▼
filters/, scoring/, sampling/, io/, utils/  ← shared helpers (pure, importable).
```

**Tools** never call other tools (don't reach across the layer). Anything two tools want to share lives one level down — `filters/`, `sampling/`, etc. **Pipelines** call tools.

## Why file-based handoffs in pipelines

Each pipeline stage reads inputs from disk and writes outputs to disk. This means:
- **Restartable**: rerunning skips stages whose outputs already exist.
- **Inspectable**: at any point you can stop, look at the intermediate files, and reason about what happened.
- **Parallelizable**: independent stages can run on separate slurm jobs.
- **Crash-tolerant**: if stage 4 of 6 dies, you don't redo 1–3.

Cost: a little I/O overhead. Worth it for any pipeline >2 stages.

### Artifact contracts and provenance (avoid stale-TSV glue)

File handoffs are only safe if the receiving stage can verify what it's reading. `io/schemas.py` defines typed objects for the canonical artifacts:

| Artifact | Carried by | Contents |
|---|---|---|
| `PoseSet` | one or more pickled poses + sidecar `.json` | list of `Pose + metadata(sequence_id, fold_source, conformer_index, parent_design_id, is_apo)` |
| `PositionTable` | `position_table.parquet` | per-residue: class, sasa, dist_ligand, dist_catalytic, ss, ss_reduced, phi, psi, in_pocket, original_aa, chain |
| `CandidateSet` | `candidates.fasta` + sidecar `.parquet` | per-sequence: id, sequence, parent_design_id, sampler (mpnn|esmif|iterative), sampler_params_hash |
| `MetricTable` | `metrics.parquet` | one row per (sequence, conformer); columns prefixed by source (`rosetta__`, `fpocket__`, `esmc__`, ...). Pareto / ranking operates on this. |

**Provenance hashing**: every stage writes a `_manifest.json` next to its output containing:
- input file SHA-256s
- tool / model checkpoint version (e.g. esm 3.2.3, LigandMPNN commit, sif file mtime+inode)
- hashed CLI args / config
- python package versions

Restart logic checks the manifest hash, not just file existence. Mismatched manifest → re-run. This prevents the silent-reuse bug where a stage's output was written under different parameters than the current run.

## Inner-loop philosophy

This codebase is explicitly built to **avoid AF2/AF3/Boltz in the inner loop**. Use:
- LigandMPNN sampling (structure+ligand-conditioned) as the primary sequence engine.
- ESM-C and SaProt as bias / scoring (NOT primary samplers for de novo scaffolds — they pull toward natural priors that conflict with designed features).
- Cheap structural mini-eval: PyRosetta sidechain repack + Rosetta ligand interface ddG + fpocket geometry, all on the **fixed designed backbone** (no folding).
- AF3 only as the final filter on the top 50–100 candidates.

## Logit fusion — and an important asymmetry between samplers

LigandMPNN's logits are **not** the same kind of distribution as ESM-C / SaProt logits, and naive product-of-experts fusion across all three is incorrect.

| Model | What its per-position logits represent |
|---|---|
| LigandMPNN | Autoregressive: `p(aa_i | structure, ligand, aa_{decoded so far})`. Conditioned on a partial sequence determined by MPNN's own decoding order. |
| ESM-C | Masked-LM marginal: `p(aa_i | aa_rest)` with position `i` masked. |
| SaProt | Masked-LM marginal conditioned on AA + 3Di tokens. |

These distributions live in different conditional regimes. Don't average / multiply them as if they were peer distributions over the same conditioning set.

**The right pattern: PLM logits as a bias to MPNN's native sampler.**

```
log p_sample(aa_i) = log p_LigandMPNN(aa_i | structure, ligand, ctx)
                   + β · log p_ESM-C(aa_i | seq_minus_i)
                   + γ · log p_SaProt(aa_i | seq_minus_i, 3Di)
```

LigandMPNN exposes `--bias_AA_per_residue` which is added to its own per-position logits at decode time. Pre-compute the PLM marginals on the *original designed sequence* once, fold them into a bias matrix, and let MPNN sample with that bias. MPNN remains the primary generative process; the PLMs nudge the distribution.

Per position class:
- **active site** → freeze identity entirely (β = γ = 0; MPNN identity-fix on this position).
- **first shell** → small bias (β, γ small) so structure-aware MPNN dominates.
- **surface** → larger bias allowed; downstream hard filters (charge, pI) handle design objectives.

**Pure-PLM pseudo-logits as a fallback.** When LigandMPNN isn't an option (e.g. the iterative walk pipeline below, where you need a per-position conditional and MPNN's autoregressive ordering would have to be re-run for each position), use just `log p_ESM-C + log p_SaProt` — both are masked-LM marginals, so they fuse cleanly.

### Calibration of fused logits

Don't add raw log-probs naively. Apply, in order:

1. **Subtract the AA background** to convert per-position log-probs to log-odds:
   `log p(aa | ctx) − log p_background(aa)` where `p_background` is the marginal AA frequency in the training corpus (or UniRef, or a uniform 1/20 prior). This decouples "this AA is rare everywhere" from "this AA is wrong here."
2. **Entropy-match models**: each model has a different per-position entropy distribution. Rescale each model's logits by `τ_model` such that the median per-position entropy matches across models. Otherwise the high-entropy model dominates after summation.
3. **Position-class–dependent weights** (β, γ): low at first-shell / pocket-lining, higher at surface; zero at active-site (which is frozen anyway).
4. **Shrinkage at disagreement**: where models disagree strongly (low cosine similarity of their per-position distributions), shrink both contributions toward zero — let MPNN dominate. The PLMs are signaling low confidence; don't bias on disputed positions.

### Why a static PLM bias can drift, and three remedies

Computing the PLM marginals once on the seed sequence and freezing them as MPNN samples is fast but gets stale: as MPNN drifts the sequence away from the seed, the bias matrix encodes a context the new sequence no longer has. Three options, in order of complexity:

1. **Refresh on top samples** — sample N candidates, recompute PLM logits on the median-by-naturalness candidate, re-bias, re-sample. Two or three rounds typically suffice.
2. **PLM as reranker, not bias** — let MPNN sample freely, then rerank candidates by ESM-C / SaProt scores. Cleanest when you don't trust the bias calibration.
3. **PLM as allowed-set restrictor** — at each position, restrict MPNN's allowed AAs to the top-k under PLM marginals (e.g. k=5). Strong restriction; use only at non-active, non-pocket positions, and only when speed matters.

**Overconfidence warning**: a static PLM bias actively pushes toward natural-protein priors. For *de novo* enzymes whose function depends on intentionally unusual residues at pocket-adjacent positions, this can erase the design. Use refresh, reranker, or position-class–zero PLM weight at the pocket.

## Iterative walk pipeline — two flavors

Two related but distinct algorithms, useful when you start from an already-good sequence and want to refine it residue by residue, with each step seeing all previous steps. Both run as `pipelines/iterative_optimize.py` with a `--mode {constrained_local_search, mh}` flag.

### Mode 1: constrained local search (cheap; default)

```
s ← starting sequence
for t in 1..T:
    pick position i (uniform / round-robin / weighted by uncertainty)
    skip if i ∈ frozen (active site / first-shell-locked)
    p(aa | s_{-i}) ← fused PLM marginals (ESM-C + SaProt, calibrated)
    aa* ~ p with temperature τ
    s' ← s with i ← aa*
    accept iff hard filters pass on s' (regex / ProtParam range / repack Δ < threshold)
    update s ← s' on accept
```

This is **constrained local search**, not Metropolis-Hastings. There is no proposal-correction term and no scalar target energy — accept depends only on whether filters pass. It works for polishing a small mutable set, but it's biased: any path through "bad single-mutant intermediate to good double-mutant" is forbidden, so you cannot find compensatory mutation pairs. It can also stall in dense regions where every neighbor fails.

### Mode 2: Metropolis-Hastings (when compensatory moves matter)

Define a scalar score `E(s) = − log P_combined(s)` from a small fixed set of objectives (e.g. 0.5·E_rosetta + 1.0·log_perplexity + 0.5·max(0, charge_target − charge(s))²). For mutation `s → s'` proposed under `q(s'|s)`:

```
α = min(1,  exp(−ΔE) · q(s|s') / q(s'|s))
accept with probability α
```

Use a **temperature schedule** for simulated annealing (high τ early to traverse barriers, low τ late). Use **block moves** occasionally — propose two or more positions at once at known interaction networks (hbond partners, salt bridges, hydrophobic core clusters) — to allow coordinated changes. **Parallel tempering** runs N chains at different τ; periodically swap configurations between adjacent chains to escape local minima.

### Convergence diagnostics (not "no accepts in N sweeps")

That naive criterion catches stalled chains, not converged ones. Use:
- **Multiple chains from different seeds**; check **R-hat** (Gelman-Rubin) on objectives or top-cluster compositions.
- **Acceptance-rate trend**: collapsing to <5% near end suggests cooled enough; rising late means temperature too high.
- **Objective autocorrelation**: integrated autocorrelation time τ_int; report effective sample size = T / τ_int per chain.
- **Stability of top sequence clusters**: cluster sequences from late-iteration samples; pipeline considers "converged" when the top-cluster composition stops shifting between windowed slices.

### When this pipeline pathologically fails

- Compensatory multi-site moves (constrained-LS only): can't traverse a worse single-mutant intermediate.
- Charge-pair swaps: ditto.
- Backbone-coupled core mutations: PLMs see neither backbone nor your repacked sidechains.
- Catalytic-water-mediated networks: not modeled at all.
- Designs that need a residue PLMs heavily disfavor (the "intentionally weird" problem).

### When to prefer this vs. batch MPNN

- **Batch MPNN** (enzyme_optimize_v1): redesign large fraction of the protein at once.
- **Constrained local search**: polish a small set of mutable positions toward charge / pI / SAP / ddG targets.
- **MH with parallel tempering**: same use case but when compensatory moves are needed.

## Position classification (extended)

Done once per design. Per residue, record:

| Feature | How |
|---|---|
| `class` ∈ {active_site, first_shell, pocket, buried, surface} | distance to ligand/catalytic atoms + SASA + fpocket pocket map |
| `sasa` Å² | PyRosetta `getSASA` (Coventry recipe) |
| `dist_ligand` Å | min distance to any ligand heavy atom |
| `dist_catalytic` Å | min distance to any theozyme catalytic atom |
| `secondary_structure` ∈ {H, E, L, ...} | DSSP (PyRosetta `SecondaryStructureMetric`) — can be passed downstream as a per-position feature for sampling priors |
| `secondary_structure_reduced` ∈ {H, E, L} | DSSP reduced alphabet |
| `phi`, `psi` | backbone dihedrals |
| `chain` | for multi-chain inputs |
| `in_pocket` (bool) | fpocket membership |
| `original_aa` | wild-type / designed-sequence AA at this position |

Output is a single JSON consumed by every downstream tool. Secondary-structure labels can drive sampling priors directly (e.g., disallow proline in helix interiors except at known helix breaks).

## Theozyme satisfaction & active-site quality (separate from naturalness)

When the input PDB carries REMARK 666 catalytic geometry, treat the catalytic motif as a **specific geometric target**, not just "frozen residues." Per design / per conformer, compute:

- **Motif RMSD** vs. theozyme reference (catalytic residue heavy atoms only).
- **Catalytic distances**: each REMARK 666 motif atom pair (e.g. His Nε — ligand attack atom). Compare to the constraint definition.
- **Catalytic angles & dihedrals**: again from REMARK 666 / cstfile semantics.
- **Attack geometry**: for the substrate's reactive atom, the angle of approach to the nucleophile / proton donor.
- **Catalytic-residue rotamer strain**: cart_bonded + fa_dun on each motif residue.

These are sharper signals than total-pose Rosetta score for whether the active site is preserved.

## Active-site preorganization & flexibility

Static structures undersell active-site quality. A well-preorganized pocket has small geometric variance under perturbation. Use one of:

- **Restrained repack ensembles**: with the REMARK 666 constraints active, run N short repack-and-min trajectories. Compute variance of catalytic distances/angles across the ensemble. Low variance = preorganized.
- **Backrub ensembles**: PyRosetta `BackrubMover` for limited backbone perturbation; same variance metric.
- **AF3 conformer agreement**: when you do reach the final AF3 step, the per-conformer agreement on catalytic geometry is a free preorganization signal.

This is one of the few enzyme-quality metrics you can compute cheaply that captures something AF3 wouldn't catch in a single pass.

## Multi-pose inputs (single PDB or many)

Tools and pipelines accept either a single PDB or a `pose_set` — a list of poses with metadata. Common scenarios:

| Scenario | Composition | Aggregation strategy |
|---|---|---|
| Single design | one PDB | per-pose metrics, no aggregation |
| Design + AF3 conformers | designed model + N AF3 predictions of the same sequence | per-conformer metrics + agreement metrics: mean ± std of Rosetta total, ligand ddG, pocket volume; **conformational consistency** = how stable the metric is across conformers (low std → robust). |
| Family of designs | M sequences on the same backbone, each maybe with K conformers | nested aggregation: per-sequence (mean across its K conformers) → ranked across M; or "robust to conformer" = sequences whose worst conformer still passes filters. |
| Apo + holo | ligand-bound and ligand-free poses of the same sequence | apo-vs-holo metrics: ligand binding ΔΔG (the user's target), pocket geometry change, induced fit. |

`io/pose_set.py` carries metadata per pose: `sequence_id`, `fold_source` (designed | AF3_seedN | Boltz | RFdiffusion), `conformer_index`, `parent_design_id`, `is_apo`. **Internally, every tool operates on a `PoseSet`; a single PDB just becomes a `PoseSet` of size 1.** No separate codepath.

### Aggregation must be metric-specific

Don't just average everything. Different metrics need different aggregation strategies:

| Metric type | Aggregation across a PoseSet |
|---|---|
| Failure / clash / unsatisfiable | **worst-case** (max / 95th percentile). One bad conformer = the design fails. |
| Mean descriptive (Rg, SASA totals) | mean ± std. Std is itself informative — high std = conformationally unstable. |
| Pocket geometry | mean ± std, with a "min volume / max bottleneck" gate. |
| Catalytic geometry | mean ± std (preorganization signal). High std = brittle. |
| Apo vs holo paired | explicit paired delta (e.g. `ddG = E_holo − E_apo − E_ligand_alone`). Don't average across the two. |
| Fold-source agreement | compare *across sources* (designed-model vs. AF3 mean): low agreement → designed model probably overfit. |

**Don't average designed-model and AF3-conformer scores as exchangeable samples** — they come from different generative processes and reflect different things. Keep `source_model` in the metadata and report agreement separately from per-conformer aggregates.

This abstraction lets you pick **conformation-robust** designs in addition to **single-pose-good** ones — important because a design that scores well only on its idealized model but poorly on every AF3 conformer is probably a brittle design.

## Catalytic residues from REMARK 666

Theozyme matcher output writes lines like:

```
REMARK 666 MATCH TEMPLATE B YYE  209 MATCH MOTIF A HIS  188  1  1
```

Fields (left to right): ligand chain, ligand name3, ligand resno; catalytic chain, catalytic name3, catalytic resno; constraint number, constraint variant. `io/pdb.py` parses these into a `{resno: catres_info}` dict and the position classifier consumes it directly — every REMARK 666 residue is force-classed `active_site` with identity locked.

If REMARK 666 isn't present, fall back to a user-supplied `--catres "A94-96 B101"` spec (legacy `parse_ref_catres` pattern). One of the two is required.

## Chemical interactions and biophysics

Beyond the basic Rosetta metrics (`fa_elec`, `hbond_sc`, `hbond_bb_*`), planned tools cover:

- **Hydrogen bonds & networks** — PyRosetta `HBondSet`, classify donor/acceptor by residue+atom; track *which* protein positions hbond to ligand / catalytic residues, with energies. Modernize from `~/special_scripts/metrics_and_hbond_rosetta_seth_no_RELAX_V2.py` and `~/special_scripts/hbonding_network.py`.
- **Salt bridges & cation-π** — RosettaScripts `arg_cation_pi` reweighting (already used in legacy script); explicit detection via geometry.
- **Aromatic π-π stacking** — geometric: pairs of aromatic rings (FYW + ligand aromatics), centroid distance < 6 Å, plane angle 0° or 90° (parallel/T-shape). Custom helper.
- **Electrostatic complementarity** — RosettaScripts `ElectrostaticComplementarityMetric` (in legacy script).
- **Electric field at active site** — APBS-derived 3D potential, sampled at theozyme TS midpoint or specific atoms. Heavier; reserve for late-stage characterization, not inner loop.
- **Metal-binding suitability** — Metal3D (`/net/software/containers/pipelines/metal3d.sif`) when ligand is a metal cofactor or design has metal-coordinating residues.
- **Per-atom SASA on ligand** — Coventry SASA recipe, from legacy script.
- **Pocket geometry** — fpocket post-repack: volume, bottleneck, hydrophobicity, charge, druggability score.

## Position classification

Done once per design, before any sampling. Categories:
- `active_site` — within 4 Å of catalytic atoms (theozyme); identity frozen.
- `first_shell` — within 5 Å of ligand; conservative substitutions only.
- `pocket` — fpocket-defined pocket residues not covered above.
- `buried` — SASA < 20 Å²; allow stability mutations.
- `surface` — SASA ≥ 20 Å²; free, biased toward design objectives.

Output is a single JSON consumed by every downstream tool.

## Multi-objective ranking

**Hard filters first, then Pareto on what survives.** No weighted-sum scoring across incommensurable metrics — that path leads to elaborate weight tuning that doesn't generalize.

But Pareto has its own failure mode: with too many correlated objectives, almost everything is non-dominated and the front becomes the whole set. To prevent this:

1. **Cap at 3-5 *real* objectives.** Pick the ones with weakly correlated signal. Good candidate set:
   - Rosetta total-score Δ (or ligand interface ddG)
   - PLM naturalness (single fused score, not ESM-C and SaProt separately)
   - Pocket geometry preservation (single fpocket-derived score, not separate volume/bottleneck/hydrophobicity)
   - Catalytic preorganization (variance under repack ensembles, see below)
   - One design-objective metric (target charge match, target pI, target SAP)
2. **Hard constraints first**: charge in range, no protease sites, repack Δ within tolerance, BUNS below threshold. These reduce the set; Pareto then ranks within the survivors.
3. **ε-dominance** instead of strict dominance: bin objectives at meaningful resolution before comparing. Prevents trivial floating-point differences from creating spurious non-dominated points.
4. **Diversity selection on mutable / pocket positions only.** Hamming distance over full-length sequence is dominated by surface variation that doesn't matter for function. Compute identity over the union of (mutable positions ∪ pocket positions) — this gives you genuinely distinct designs.
5. **Calibration set required.** Without a benchmark of known-good and known-bad designs to set thresholds and weights, your filters are folklore. Maintain a small held-out set of past designs with known experimental outcomes; periodically check the filter stack reproduces expected pass/fail labels.

## Cheap filters: veto vs. ranker

Fixed-backbone PyRosetta repacking + ligand ddG + fpocket geometry + SAP / charge / protease filters are **a veto layer**, not a ranker. They reliably catch:
- Steric clashes
- Gross desolvation
- Pocket destruction
- Ligand-incompatible substitutions
- Obviously bad chemistry (excess net charge, exposed hydrophobics, runs of cysteines)

They are **not** reliably informative for:
- Foldability — backbone is fixed; you're not seeing whether the new sequence still folds to that backbone.
- Loop gating / conformational dynamics — you have one rotamer state per residue.
- Water-mediated catalysis — explicit waters absent.
- Protonation-sensitive networks — pKa estimation is a separate task.
- Induced-fit pockets — backbone movement isn't allowed.
- Multiple alternate ligand poses — only the docked pose is scored.

So: use cheap filters to *kill* obviously bad sequences, but treat near-tied survivors as equivalent. AF3 (the final filter outside the inner loop) is what tells you which of the survivors actually *work*.

## Synthetic-MSA caveats

A pool of N=1000 sampled sequences is a useful object — it gives per-position frequency profiles, mutual information between positions, and converged-vs-uncertain residues. **Do not** treat it like a natural MSA: classical MSA-derived signals (DCA, conservation, EVcouplings) extract evolutionary covariation, but a synthetic pool encodes only sampler priors, not biology. See `scoring/synthetic_msa.py` (when implemented) for the ways we use the pool.

## Where to add a new tool

1. Add `src/protein_chisel/tools/<your_tool>.py` — exposes a `cli` (`click` command) and a Python-callable function.
2. Register it in `protein_chisel/cli.py` so it appears under `chisel --help`.
3. Add an entry in `docs/dependencies.md` with the cluster path of whatever it wraps.
4. Add a test in `tests/test_<your_tool>.py` (smoke test is fine).
5. If the tool needs new paths, add them to `paths.py`.
