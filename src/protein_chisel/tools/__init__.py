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

Backbone / scaffold quality (modernized from process_diffusion3):
- backbone_sanity        # chainbreak (max sequential CA-CA), rCA_nonadj
                         #   (min non-adjacent CA-CA), term_mindist
                         #   (terminus to ligand)
- shape_metrics          # radius of gyration (proper sqrt-mean-square def),
                         #   Rg / sqrt(N) (length-normalized), gyration tensor
                         #   eigenvalues -> asphericity, acylindricity,
                         #   relative shape anisotropy, globularity
- ss_summary             # loop_frac, longest_helix, longest_sheet,
                         #   helix_count, sheet_count, loop_at_motif
                         #   (boolean: catalytic residue in loop region)

Catalytic geometry & active-site quality:
- catalytic_residues     # parse REMARK 666 -> {resno: {chain, name3,
                         #   target_chain, target_name3, target_resno,
                         #   cst_no, cst_no_var}}
- catres_quality         # per-catalytic-residue cart_bonded + fa_dun (rotamer
                         #   strain), and sidechain bond-length deviation vs.
                         #   ideal (broken-motif detector). From process_diff
                         #   `sidechain_connectivity` and `get_rosetta_scores`.
- theozyme_satisfaction  # catalytic distance/angle/dihedral deviations vs.
                         #   the REMARK 666 / cstfile-defined targets,
                         #   motif RMSD vs. theozyme reference, attack-geometry
                         #   angle for substrate reactive atoms.
- preorganization        # variance of catalytic geometry across a restrained
                         #   repack-min ensemble (or backrub ensemble); low
                         #   variance => preorganized active site.
- ligand_environment     # min backbone-ligand distance, ligand SASA, ligand
                         #   SASA_rel (vs. free ligand), per-atom ligand SASA
                         #   for user-specified atoms (Coventry recipe).
- buns                   # buried unsatisfied polar atoms / hbonds. Critical
                         #   metric; buried unsats are a major failure signal.
                         #   Two flavors:
                         #     a) Rosetta BuriedUnsatisfiedHbondsFilter (XML)
                         #     b) **Whitelist-aware BUNS** following bcov
                         #        parse_target_buns_recalculate_white.py:
                         #        accept that some catalytic atoms are
                         #        intentionally unsat (e.g. nucleophile lone
                         #        pairs poised for attack). Whitelist =
                         #        list of (resno, atom_name) tolerated unsat.
                         #   The whitelist-aware variant is essential for
                         #   theozyme-bearing designs.
- packing_quality        # Rosetta packstat score, buried cavity volume.
                         #   Distinct from holes (which is a Rosetta filter
                         #   targeting voids); packstat is a continuous
                         #   per-residue packing-quality metric.

Protonation / pKa / ionization (when relevant):
- catalytic_pka          # PROPKA or Rosetta-based pKa estimates for catalytic
                         #   residues; flag designs where catalytic histidines
                         #   end up at unworkable pKa.

Solubility (orthogonal predictors — don't rely on SAP alone):
- deepsp_score           # DeepSP solubility predictor (TBD, install pending)
- camsol_score           # CamSol solubility predictor (TBD, install pending)
                         # Use one of these as a second-opinion vs. SAP; they
                         # disagree often and the consensus is more reliable.

Energy / stability:
- pyrosetta_repack       # sidechain repack on fixed backbone + Rosetta score Δ
- rosetta_ligand_ddg     # holo vs apo binding ΔΔG (no per-mutation; whole-design)
- thermompnn             # ML-based stability ddG (faster than Rosetta ddG)
- per_residue_ddg        # per-residue ddG suite, modes from bcov ddg_per_res.py:
                         #   - basic per-residue contribution
                         #   - ala_scan (alanine substitution)
                         #   - with_repack (more accurate, more expensive)
                         #   - buried_elec (focused on buried electrostatics)
                         # Uses the canonical fix_scorefxn pattern (decomposed
                         # bb hbonds, proper double-bb check) so the per-residue
                         # contributions sum correctly.

Interactions:
- chemical_interactions  # hbonds (with energies), salt bridges, π-π, π-cation,
                         # ligand-protein contacts, per-atom ligand SASA.
                         # Use the canonical bcov fix_scorefxn pattern (set
                         # decompose_bb_hb_into_pair_energies + double-bb
                         # check) for proper hbond enumeration. Pattern from
                         # /home/bcov/util/dump_hbset.py.
- contact_ms             # **Distance-weighted contact molecular surface**
                         # via py_contact_ms (https://github.com/bcov77/
                         # py_contact_ms). Formula: area · exp(-0.5 d²).
                         # NumPy-only, PyRosetta-free, per-atom CMS output.
                         # Strictly better than Rosetta's
                         # ContactMolecularSurface filter for ranking
                         # interface quality. Use this in addition to or
                         # instead of the legacy filter.
- metal3d_score          # metal-binding suitability (metal3d.sif)
- surface_composition    # polars per SASA (polar atoms / unit SASA),
                         # hydrophobics per SASA, etc. From bcov
                         # polars_per_sasa.py — surface-chemistry signal
                         # complementary to SAP. Optional raytrace variant
                         # for more accurate SASA (uses npose).

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
