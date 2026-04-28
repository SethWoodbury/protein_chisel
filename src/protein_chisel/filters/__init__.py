"""Cheap sequence-level filters (pure functions; no side effects).

Each filter takes a sequence (or list of sequences) and returns either a
boolean mask, scored values, or a filtered subset. Filters never call out
to GPUs or network resources. They are sequence-only (no structure required)
unless explicitly noted.

Planned:
- protease_sites    # regex blacklist (Kex2 RR, trypsin K|R [^P], Lys-C,
                    #   OmpT, signal peptidase A-x-A, dibasic sites, ...)
- protparam         # Biopython.SeqUtils.ProtParam wrapper:
                    #   isoelectric_point() -> pI
                    #   instability_index()
                    #   gravy() -> grand average of hydropathy
                    #   molecular_weight()
                    #   aromaticity()
                    #   charge_at_pH(pH) and a "charge_at_pH7_no_HIS" variant
                    #   (matches the legacy NetCharge filter that excludes HIS
                    #   to avoid pH-uncertainty on histidine ionization)
                    #   secondary_structure_fraction() (helix/turn/sheet
                    #   propensity from sequence)
                    #   extinction_coefficient() (for spectroscopy / QC)
- sap_score         # spatial aggregation propensity (Lauer 2012). Needs a
                    #   structure; uses fixed designed backbone + repacked
                    #   sidechains. Strictly a structural filter but kept
                    #   here because most sequences will have a parent
                    #   structure on hand.
- codon_usage       # E. coli codon-table optimality (post-AA design)
- length            # length range, terminal AA constraints (no Pro at N+1,
                    #   no Cys at termini if not desired, etc.)
- forbidden_motifs  # internal Met start codons, NxS/NxT N-glycosylation
                    #   sites if expressing in eukaryote, runs of cysteines,
                    #   N-end rule destabilizers, single-codon repeats.
"""
