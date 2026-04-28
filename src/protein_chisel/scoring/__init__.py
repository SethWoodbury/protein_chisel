"""Multi-objective scoring and selection.

Once tools have produced per-sequence metrics, scoring chooses which
sequences to keep / promote.

Planned:
- pareto            # Pareto front extraction over N objectives
- diversity         # sequence-identity diversity penalty / clustering
- synthetic_msa     # mutual information & co-occurrence on a sample pool
                    # (NOT for natural-MSA-style covariation analysis;
                    #  see docs/architecture.md for caveats)
"""
