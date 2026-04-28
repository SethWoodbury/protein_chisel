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

## Cluster dependencies

All external models, containers, and weights are documented in [`docs/dependencies.md`](docs/dependencies.md) with their cluster paths. The Python module `protein_chisel.paths` centralizes those paths so everything else can import them.

## Install

```bash
git clone <this-repo> ~/codebase_projects/protein_chisel
cd ~/codebase_projects/protein_chisel
pip install -e .
```

For container-bound runs, see [`docs/setup.md`](docs/setup.md).

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
