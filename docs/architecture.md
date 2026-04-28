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

## Iterative single-mutation walk (Gibbs / MH pipeline)

A separate pipeline alongside the batch `enzyme_optimize_v1`. Useful when you start from an already-good sequence and want to refine it one residue at a time, with each step seeing all previous steps.

```
s ← starting sequence
for t in 1..T:
    pick a position i (uniform / round-robin / weighted by uncertainty)
    skip if i is frozen (active site / first-shell-locked)
    compute p(aa | s_{-i}) from fused PLM marginals (ESM-C + SaProt)
    propose aa* ~ p (with temperature)
    accept aa* iff cheap filters still pass on s' = s with i ← aa*
        (regex / ProtParam / quick PyRosetta repack-and-score Δ)
    update s ← s' on accept
loop until convergence (no accepted moves in N consecutive sweeps)
```

Properties:
- **Honors structural epistasis implicitly**: each mutation is evaluated in the context of all previous mutations.
- **PLM-only** (no MPNN per step) keeps the inner loop cheap.
- **MH acceptance** with cheap filters as the energy function lets you steer toward target charge / SAP / ddG without a single weighted scoring formula.
- Use **temperature schedule** (anneal high → low) for simulated annealing variant.
- Output is one walked sequence; run many parallel chains from the same start to get a diverse library.

When to prefer this pipeline vs. batch MPNN:
- Batch MPNN: better for "redesign large fraction of the protein at once."
- Iterative walk: better for "polish a sequence with a small number of carefully chosen mutations" — closer to what directed evolution does.

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

## Multi-pose inputs (single PDB or many)

Tools and pipelines accept either a single PDB or a `pose_set` — a list of poses with metadata. Common scenarios:

| Scenario | Composition | Aggregation strategy |
|---|---|---|
| Single design | one PDB | per-pose metrics, no aggregation |
| Design + AF3 conformers | designed model + N AF3 predictions of the same sequence | per-conformer metrics + agreement metrics: mean ± std of Rosetta total, ligand ddG, pocket volume; **conformational consistency** = how stable the metric is across conformers (low std → robust). |
| Family of designs | M sequences on the same backbone, each maybe with K conformers | nested aggregation: per-sequence (mean across its K conformers) → ranked across M; or "robust to conformer" = sequences whose worst conformer still passes filters. |
| Apo + holo | ligand-bound and ligand-free poses of the same sequence | apo-vs-holo metrics: ligand binding ΔΔG (the user's target), pocket geometry change, induced fit. |

`io/pose_set.py` carries metadata per pose: `sequence_id`, `fold_source` (designed | AF3_seedN | Boltz | RFdiffusion), `conformer_index`, `parent_design_id`, `is_apo`. `scoring/aggregate.py` provides per-design rollups (mean / std / min / max / vote) over conformers.

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

**Hard filters first, then Pareto on what survives.** No weighted-sum scoring across incommensurable metrics — that path leads to elaborate weight tuning that doesn't generalize. Use Pareto fronts on (Rosetta ΔΔG, ligand binding ΔΔG, ESM-C/SaProt naturalness, pocket geometry preservation, charge-target match), then apply a sequence-identity diversity cap when picking the top N.

## Synthetic-MSA caveats

A pool of N=1000 sampled sequences is a useful object — it gives per-position frequency profiles, mutual information between positions, and converged-vs-uncertain residues. **Do not** treat it like a natural MSA: classical MSA-derived signals (DCA, conservation, EVcouplings) extract evolutionary covariation, but a synthetic pool encodes only sampler priors, not biology. See `scoring/synthetic_msa.py` (when implemented) for the ways we use the pool.

## Where to add a new tool

1. Add `src/protein_chisel/tools/<your_tool>.py` — exposes a `cli` (`click` command) and a Python-callable function.
2. Register it in `protein_chisel/cli.py` so it appears under `chisel --help`.
3. Add an entry in `docs/dependencies.md` with the cluster path of whatever it wraps.
4. Add a test in `tests/test_<your_tool>.py` (smoke test is fine).
5. If the tool needs new paths, add them to `paths.py`.
