"""Orchestrators that chain tools together.

Pipelines are thin: they read a config, invoke tools in sequence with
file-based handoffs, and write a single results directory. Each pipeline
is restartable (re-running skips stages whose outputs already exist).

Planned:
- enzyme_optimize_v1    # batch redesign:
                        #   classify -> PLM fusion -> biased LigandMPNN ->
                        #   filter cascade -> structural mini-eval -> Pareto select
                        # Best for "redesign large fraction of the protein at once."

- iterative_optimize    # single-mutation Gibbs / MH walk:
                        #   classify -> PLM-fused marginals -> position-by-position
                        #   propose+accept against cheap filters -> until convergence
                        # Best for "polish a sequence with a few targeted mutations,"
                        # closer to directed evolution. Run many parallel chains
                        # from the same start for library diversity.

- comprehensive_metrics # purely descriptive (no design): take a PDB and run the
                        # full Rosetta XML metric pack + chemical interactions +
                        # fpocket + metal3d + PLM scores. Modernized replacement
                        # for ~/special_scripts/design_filtering/metric_monster.

Conventions:
- Pipelines are restartable: rerunning skips stages whose outputs already exist.
- Configs in `configs/` drive runs (`chisel <pipeline> --config configs/<x>.yaml`).
- Outputs land in a single results directory with one subdir per stage.
"""
