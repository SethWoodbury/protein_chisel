# Documentation deployment plan

When triggered, deploy 6 parallel subagents to build comprehensive docs.

## Outputs

```
README.md                                # Top-level project overview, quickstart
docs/
├── architecture.md                      # Pipeline stages + data flow
├── usage.md                             # Step-by-step running guide
├── cli_reference.md                     # All ~40 flags with defaults + when to use
├── metrics_reference.md                 # Every metric in the TSV with definition + range
├── dependencies.md                      # SIFs, binaries, python packages, data paths
├── troubleshooting.md                   # Known issues + workarounds
├── plans/                               # Existing design docs (kept)
│   ├── directional_classification_plan.md
│   ├── documentation_plan.md           (this file)
│   └── efficiency_plan.md              (Phase 3)
└── examples/
    ├── pte_default_run.md               # Standard PTE_i1 production run
    ├── pte_diverse_run.md               # Diversity-recovery sweep
    └── new_scaffold_setup.md            # Adapting to a non-PTE target
```

## Subagent assignments (parallel)

### Agent 1 — README + architecture
**Reads:** repo structure, key modules, manifest.json from a recent run, plans/.
**Writes:** README.md (top-level orientation, what this pipeline does, quickstart, link tree), docs/architecture.md (Mermaid pipeline diagram, stage descriptions, data-flow + artifacts).

### Agent 2 — Usage + examples
**Reads:** scripts/run_chisel_design.sh, recent run dirs, plans/.
**Writes:** docs/usage.md (running locally vs slurm vs interactive, all 3 stages, common patterns), docs/examples/{pte_default_run.md, pte_diverse_run.md, new_scaffold_setup.md}.

### Agent 3 — CLI reference
**Reads:** scripts/iterative_design_v2.py argparse block. Generate `--help` output and convert to organized markdown grouped by topic (sampling, filters, multi-objective ranking, termini, strategy).
**Writes:** docs/cli_reference.md.

### Agent 4 — Metrics reference (the BIG one)
**Reads:** all module docstrings for protparam.py, dfi.py, multi_objective.py, classify_positions.py, scoring/preorganization.py, structure/clash_check.py, expression/aa_class_balance.py, tools/geometric_interactions.py. Plus a recent all_survivors.tsv to enumerate every column.
**Writes:** docs/metrics_reference.md — every metric in the TSV with:
- definition + formula
- units
- direction (max/min/target)
- typical range / threshold
- known caveats

This will be ~3000 lines. Most important doc.

### Agent 5 — Dependencies + apptainer
**Reads:** all sif paths in scripts/*.sbatch, all `from X import Y` lines, requirements.txt or pyproject.toml. Document SIF file purposes:
- `universal.sif`: PyRosetta, fpocket, freesasa, biopython, numpy, pandas
- `esmc.sif`: ESM-C, SaProt, py_contact_ms, GPU-bound
- `pyrosetta.sif`: Rosetta with full param libraries

Plus binaries:
- fused_mpnn at /net/software/lab/fused_mpnn/seth_temp/run.py
- LMPNN ckpt: /net/databases/mpnn/...
- PROPKA in universal.sif (160ms/design — declined for now)

Plus data paths:
- HF cache at /net/databases/huggingface/{esmc,saprot}
- params files

**Writes:** docs/dependencies.md.

### Agent 6 — Troubleshooting + strengths/weaknesses
**Reads:** session history (commits + log messages), known-issue comments in code, plans/.
**Writes:** docs/troubleshooting.md (known gotchas + fixes — DAlphaBall libgfortran, freesasa fallback, schema migration, AA composition skew toward E, position-1 M omit reasoning, consensus diversity trade-off). Plus strengths-vs-weaknesses summary in README.

## Timing estimate

Each subagent ~3-5 min. All run in parallel: ~5 min wall-clock for the full doc tree.

## After deployment

- Spot-check each output for accuracy
- Run `python scripts/iterative_design_v2.py --help` and verify CLI reference matches
- Run a smoke design and verify metrics reference matches columns
- Commit + push
