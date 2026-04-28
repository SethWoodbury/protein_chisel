"""Multi-objective scoring and selection.

Once tools have produced per-sequence metrics, scoring chooses which
sequences to keep / promote.

Planned:
- pareto            # Pareto front with **3-5 real objectives only**, not
                    # every metric you computed. Supports ε-dominance to
                    # avoid floating-point ties producing spurious
                    # non-dominated points. See architecture.md
                    # "Multi-objective ranking" for which objectives.
- diversity         # sequence-identity diversity over **mutable / pocket
                    # positions only** (full-length identity is dominated
                    # by surface noise). Hierarchical clustering on those
                    # positions; pick representatives.
- aggregate         # PoseSet -> per-design rollups, metric-specific:
                    #   worst-case for failure metrics
                    #   mean ± std for descriptive metrics
                    #   paired delta for apo/holo
                    #   cross-source agreement (designed vs. AF3) reported
                    #   separately, never averaged
- synthetic_msa     # mutual information & co-occurrence on a sample pool.
                    # **Not** for natural-MSA-style DCA / EVcouplings —
                    # synthetic pools encode sampler priors, not biology.
                    # For genuine evolutionary signal on de novo proteins,
                    # use Foldseek/PDB/AFDB structural-analog harvesting
                    # (planned future feature).
- calibration       # threshold management against a held-out benchmark
                    # of past designs with known outcomes. Without this,
                    # filter thresholds drift into folklore; with it,
                    # you can periodically re-fit thresholds and detect
                    # filter-stack regressions.
"""
