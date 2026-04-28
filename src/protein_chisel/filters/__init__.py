"""Cheap sequence-level filters (pure functions; no side effects).

Each filter takes a sequence (or list of sequences) and returns either a
boolean mask, scored values, or a filtered subset. Filters never call out
to GPUs or network resources.

Planned:
- protease_sites    # regex blacklist (Kex2, trypsin, signal peptidase, ...)
- protparam         # pI, net charge, instability index, GRAVY (Biopython)
- sap_score         # spatial aggregation propensity (Lauer 2012)
- codon_usage       # E. coli codon-table optimality (post-AA design)
- length            # length / terminal AA constraints
"""
