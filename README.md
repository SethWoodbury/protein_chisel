# protein_chisel

Tools and pipelines for refining and ranking sequences for de novo enzyme designs.

The codebase is organized in three layers:

1. **Tools** (`src/protein_chisel/tools/`) — single-purpose primitives, each runnable on its own (e.g. *classify residue positions*, *score sequences with ESM-C*, *repack and score a PDB with PyRosetta*).
2. **Filters / scoring / sampling / I/O / utils** — shared helpers used by tools and pipelines.
3. **Pipelines** (`src/protein_chisel/pipelines/`) — orchestrators that chain tools together, with file-based handoffs so each stage is restartable.

Anything you'd want to run as a one-off (e.g. *only* the position classifier on a PDB) lives in `tools/` and is exposed by the `chisel` CLI. Anything you'd want to run as a recurring multi-step workflow lives in `pipelines/`.

## Big picture

For a designed enzyme structure (with ligand / theozyme catalytic residues defined), produce 50–100 diverse, biophysically vetted sequence variants for downstream AlphaFold validation — without paying for AF2/AF3 in the inner loop.

The intended core pipeline:

```
designed PDB + theozyme
   │
   ▼  classify_positions    (PyRosetta SASA + ligand distance + fpocket)
position_classes.json
   │
   ▼  esmc_logits / saprot_logits   (PLM per-position log-probs)
fused_bias.npy
   │
   ▼  ligand_mpnn (with --bias_AA_per_residue)
candidate_sequences.fasta   (~500)
   │
   ▼  filter cascade: protease regex → ProtParam → SAP → ESM-C ppx
filtered.fasta              (~200)
   │
   ▼  pyrosetta_repack + rosetta_ligand_ddg + fpocket_run
scored.tsv                  (per-sequence metrics)
   │
   ▼  pareto + diversity selection
final_library.fasta         (50–100 sequences for AF3)
```

## Documentation

Full docs in [`docs/`](docs/):

| File | What's inside |
|---|---|
| [`docs/capabilities.md`](docs/capabilities.md) | Quick "what can I do today?" bullets |
| [`docs/setup.md`](docs/setup.md) | Install, sif paths, what-runs-where, slurm gotchas |
| [`docs/architecture.md`](docs/architecture.md) | Three-layer design, logit-fusion math, multi-pose abstraction, ranking philosophy |
| [`docs/dependencies.md`](docs/dependencies.md) | Every external dep + per-tool installation status |
| [`docs/tools.md`](docs/tools.md) | Per-tool reference (inputs / outputs / sif / tests) |
| [`docs/filters.md`](docs/filters.md) | Sequence-only filters |
| [`docs/scoring.md`](docs/scoring.md) | Aggregate / Pareto / diversity, with code examples |
| [`docs/sampling.md`](docs/sampling.md) | PLM fusion calibration, biased MPNN, ligand_mpnn wrapper |
| [`docs/pipelines.md`](docs/pipelines.md) | Per-pipeline I/O, sbatch invocation, restart behavior |
| [`docs/io_and_schemas.md`](docs/io_and_schemas.md) | Manifest hashing, PoseSet / PositionTable / CandidateSet / MetricTable, sample JSONs |
| [`docs/testing.md`](docs/testing.md) | Host vs cluster vs gpu test markers, current coverage |
| [`docs/future_plans.md`](docs/future_plans.md) | Stubs, untested wrappers, sampling/scoring TODO, pending sif additions |
| [`docs/codex_review_2026-04-28.md`](docs/codex_review_2026-04-28.md) | Codex's prior critique (already addressed) |

## Cluster dependencies

All external models, containers, and weights are documented in [`docs/dependencies.md`](docs/dependencies.md) with their cluster paths. The Python module `protein_chisel.paths` centralizes those paths so everything else can import them.

## Install

```bash
git clone git@github.com:SethWoodbury/protein_chisel.git ~/codebase_projects/protein_chisel
```

This repo is bind-mounted into our apptainer images at `/code` with
`PYTHONPATH=/code/src`; you don't need a host-side `pip install` to use
it. See [`docs/setup.md`](docs/setup.md) for how each container handles
the package.

## Status (v0.0.1)

Implemented and tested:

- **Foundations**: `io/schemas` (PoseSet, PositionTable, CandidateSet,
  MetricTable, Manifest with stable hashing), `io/pdb` (REMARK 666
  parser/write-back, ATOM record, sequence extraction), `utils/pose`
  (PyRosetta wrappers + bcov fix_scorefxn pattern + Coventry SASA),
  `utils/geometry`, `utils/apptainer`.
- **Structural tools**: classify_positions, backbone_sanity,
  shape_metrics (proper Rg + asphericity), secondary_structure,
  ss_summary, ligand_environment, chemical_interactions (hbonds w/ heavy-
  and H-atom names + salt bridges + π-π geometry + π-cation), BUNS with
  whitelist, contact_ms (py_contact_ms), catres_quality.
- **Sequence filters**: protparam (Biopython + custom no-HIS net charge),
  protease_sites, length.
- **PLM tools**: esmc_logits/_score, saprot_logits/_score with proper
  masked-LM marginals (L forward passes per call).
- **Sampling**: plm_fusion (log-odds calibration + entropy-match +
  shrinkage), biased_mpnn (orchestrator), ligand_mpnn (wraps the
  fused_mpnn lab build at `/net/software/lab/fused_mpnn/seth_temp/`
  via `universal.sif`).
- **Scoring**: aggregate (metric-specific rollups across PoseSet),
  pareto (ε-dominance), diversity (Hamming over mutable positions).
- **Catalysis**: theozyme_satisfaction (Kabsch motif RMSD + per-residue +
  per-atom-to-ligand distances), catalytic_pka (PROPKA wrapper).
- **Pipelines**: comprehensive_metrics (descriptive structural battery),
  naturalness_metrics (PLM scoring + fusion-bias artifacts),
  sequence_design_v1 (5-stage pipeline), iterative_optimize (constrained
  local search + real Metropolis-Hastings).

Test coverage:
- 141 host tests
- 30 cluster tests in pyrosetta.sif (real design.pdb + apo + holo)
- 12 cluster tests in esmc.sif (PLM + contact_ms + PROPKA)

What's stubbed / pending (see [`docs/future_plans.md`](docs/future_plans.md)):
- preorganization (implemented; **untested**)
- fpocket_run (wrapper functional; **binary install pending**)
- metal3d_score (HETATM scan only; **inference path stubbed**)
- ProLIF / pdbe-arpeggio / CAVER wrappers (no source files yet; awaiting new sif)
- comprehensive_metrics × naturalness_metrics merge pipeline
- iterative_optimize block moves / parallel tempering
- scoring/calibration benchmark loop
- repo CI

## Quick start

```bash
# Run all structural metrics on the design PDB
sbatch scripts/run_comprehensive_metrics.sbatch \
    /home/woodbuse/testing_space/align_seth_test/design.pdb \
    out/comprehensive

# Run naturalness scoring (ESM-C + SaProt) — needs GPU
sbatch scripts/run_naturalness_metrics.sbatch \
    /home/woodbuse/testing_space/align_seth_test/design.pdb \
    out/naturalness \
    out/comprehensive  # for the PositionTable

# Full design pipeline (5 stages, multi-sif) — needs GPU
sbatch scripts/run_sequence_design_v1.sbatch \
    /home/woodbuse/testing_space/align_seth_test/design.pdb \
    /home/woodbuse/testing_space/scaffold_optimization/ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params/YYE.params \
    out/design
```

## Layout

```
protein_chisel/
├── docs/         # architecture, deps, setup
├── configs/      # YAML run configs
├── src/protein_chisel/
│   ├── paths.py           # all cluster-specific paths
│   ├── cli.py             # `chisel` CLI
│   ├── tools/             # standalone primitives
│   ├── filters/           # cheap sequence filters
│   ├── scoring/           # ranking + multi-objective selection
│   ├── sampling/          # logit fusion, biased MPNN
│   ├── pipelines/         # orchestrators
│   ├── io/                # FASTA / PDB / TSV
│   └── utils/             # apptainer, slurm, pose helpers
├── scripts/      # sbatch templates, CLI shims
├── tests/
└── examples/
```
