"""Cross-cutting utilities.

Planned:
- apptainer         # subprocess wrapper around `apptainer exec` with
                    # standard --bind/--env arguments and our SIF paths.
- slurm             # submitit wrapper with sensible defaults
- pose              # PyRosetta common operations:
                    #   - get_per_atom_sasa(pose, probe=2.8) — bcov pattern
                    #   - get_residue_sasa, get_ligand_sasa, getSASA
                    #     (Coventry recipe; 1.4 Å for solvent, 2.8 Å for
                    #     contact-region computations)
                    #   - mutate_and_repack(pose, resno, new_aa)
                    #   - thread_seq_to_pose(pose, sequence) (process_diff)
                    #   - find_ligand_seqpos(pose)
                    #   - fix_scorefxn(sfxn, allow_double_bb=False) — the
                    #     canonical bcov hbond-decomposition setup used by
                    #     dump_hbset, ddg_per_res, polars_per_sasa, etc.
                    #     Required for per-residue energies to sum cleanly.
- hyperparam_search # Boltzmann grid sampler for tuning fusion weights,
                    # MH temperature schedules, filter thresholds. Adapted
                    # from /home/bcov/util/boltzmann_grid_sampler.py
                    # (originally Adam Moyer): pick next sample weighted
                    # by exp(-score/kT), supports continuous bounds and
                    # update-with-results loops.
- logging           # consistent logger setup across tools
"""
