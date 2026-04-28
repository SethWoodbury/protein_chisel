"""Cross-cutting utilities.

Planned:
- apptainer         # subprocess wrapper around `apptainer exec` with
                    # standard --bind/--env arguments and our SIF paths.
- slurm             # submitit wrapper with sensible defaults
- pose              # PyRosetta common operations (SASA, mutate-and-repack,
                    # hbond detection, residue selection)
- logging           # consistent logger setup across tools
"""
