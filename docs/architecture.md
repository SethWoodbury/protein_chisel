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

## Logit fusion (product of experts)

The intended way to combine multiple per-position log-prob distributions:

```
log p_combined(aa | pos) = α · log p_LigandMPNN(aa | structure, ligand)
                         + β · log p_ESM-C(aa | sequence)
                         + γ · log p_SaProt(aa | seq + 3Di)
```

Then sample (temperature τ) from `p_combined`. LigandMPNN already exposes per-position bias (`--bias_AA_per_residue`) so this is implemented by:

1. Compute ESM-C and SaProt logits on the original designed sequence — once.
2. Convert to a per-position bias matrix (β·log p_ESM-C + γ·log p_SaProt).
3. Pass to LigandMPNN at sampling time.

Per position class:
- **active site** → freeze (β=γ=0; identity also frozen by MPNN constraints).
- **first shell** → low bias (β=γ small).
- **surface** → bias allowed; charge/solubility filters do the heavy lifting downstream.

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
