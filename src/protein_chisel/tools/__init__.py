"""Single-purpose primitives, each runnable on its own.

Conventions:
- Each tool is one module.
- Each module exposes a `click` command (`cli`) registered in
  `protein_chisel.cli` and a Python-callable function for in-process use.
- Tools should NOT call other tools internally — that's what pipelines are
  for. Cross-tool dependencies live in `filters/`, `scoring/`, `sampling/`,
  `io/`, or `utils/`.

Planned tools (add as implemented):
- classify_positions     # PyRosetta SASA + ligand distance + fpocket -> JSON
- esmc_logits            # ESM-C per-position log-probs (for fusion / scoring)
- esmc_score             # pseudo-perplexity per sequence
- saprot_logits          # SaProt per-position log-probs (structure-aware)
- saprot_score           # naturalness scoring per sequence
- ligand_mpnn            # LigandMPNN sampling, supports per-position bias
- esm_if                 # ESM-IF1 sampling (older, ensemble diversity)
- pyrosetta_repack       # sidechain repack + total score
- rosetta_ligand_ddg     # holo - apo binding ddG
- fpocket_run            # pocket geometry: volume, bottleneck, hydrophobicity
- thermompnn             # ML-based stability ddG (faster than Rosetta ddG)
"""
