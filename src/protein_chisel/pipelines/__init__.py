"""Orchestrators that chain tools together.

Pipelines are thin: they read a config, invoke tools in sequence with
file-based handoffs, and write a single results directory. Each pipeline
is restartable (re-running skips stages whose outputs already exist).

Planned:
- enzyme_optimize_v1    # the MVP described in README:
                        #   classify -> fused logits -> biased LigandMPNN ->
                        #   filter cascade -> structural mini-eval -> Pareto select
"""
