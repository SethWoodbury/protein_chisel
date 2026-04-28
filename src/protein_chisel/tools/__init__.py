"""Single-purpose primitives, each runnable on its own.

Planned tools (add as implemented):

PLM / sampling:
- esmc_logits            # ESM-C masked-LM per-position log-probs (for fusion)
- esmc_score             # pseudo-perplexity per sequence
- saprot_logits          # SaProt structure-aware per-position log-probs
- saprot_score           # naturalness scoring per sequence
- ligand_mpnn            # LigandMPNN sampling, supports --bias_AA_per_residue
- esm_if                 # ESM-IF1 sampling (older fair-esm, ensemble diversity)

Structure / geometry:
- classify_positions     # SASA + ligand distance + DSSP + fpocket → JSON
- secondary_structure    # DSSP per-residue (H/E/L), reduced + full alphabet
- fpocket_run            # pocket volume / bottleneck / hydrophobicity / charge

Energy / stability:
- pyrosetta_repack       # sidechain repack on fixed backbone + Rosetta score Δ
- rosetta_ligand_ddg     # holo vs apo binding ΔΔG (no per-mutation; whole-design)
- thermompnn             # ML-based stability ddG (faster than Rosetta ddG)

Interactions:
- chemical_interactions  # hbonds (with energies), salt bridges, π-π, π-cation,
                         # ligand-protein contacts, per-atom ligand SASA
- metal3d_score          # metal-binding suitability (metal3d.sif)

Comprehensive metric assemblers:
- rosetta_metrics_xml    # the legacy ~25-metric RosettaScripts protocol from
                         # metrics_and_hbond_rosetta_seth_no_RELAX_V2.py, with
                         # parsed outputs in a single TSV row per pose

Conventions:
- Each tool is one module exposing a `click` command (registered in
  `protein_chisel.cli`) and a Python-callable function.
- Tools should NOT call other tools internally — that's what pipelines are
  for. Cross-tool dependencies live in `filters/`, `scoring/`, `sampling/`,
  `io/`, or `utils/`.
"""
