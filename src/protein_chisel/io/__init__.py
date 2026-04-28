"""File I/O helpers (FASTA, PDB, results TSV).

Planned:
- fasta             # read/write FASTA, with metadata fields
- pdb               # extract chains, residues, ligand atoms; renumbering;
                    # **REMARK 666 parsing for catalytic residues from theozyme
                    # matcher output** (e.g. `REMARK 666 MATCH TEMPLATE B YYE
                    # 209 MATCH MOTIF A HIS 188  1  1`). Returns dict keyed by
                    # design resno. Re-emit on output PDBs so downstream tools
                    # always know the catalytic positions. Adapt from
                    # SETH_TEMP_UTILS/process_diffusion3_outputs__REORG.py
                    # (`get_matcher_residues`, `add_matcher_line_to_pose`).
- pose_set          # multi-pose abstraction: list of (pose, metadata) where
                    # metadata records: sequence_id, fold_source (designed |
                    # AF3_seed_N | Boltz | RFdiffusion), conformer_index,
                    # parent_design_id. Tools accept either a single pose or
                    # a pose_set; metrics that are pose-level are computed
                    # per-pose; aggregate metrics (mean, std, agreement,
                    # min, etc.) are computed by `aggregate_metrics()`.
- results           # canonical scored TSV/parquet for ranked outputs;
                    # adopt the prefix-by-source pattern from legacy
                    # design_filtering/metric_monster__MAIN.py.
- catres_spec       # parse user-provided catalytic-residue strings like
                    # ["A94-96", "B101"] (from process_diffusion3). Used
                    # when REMARK 666 isn't present.
- schemas           # **typed artifact contracts** for stage handoffs:
                    #   PoseSet, PositionTable, CandidateSet, MetricTable.
                    # Each carries a manifest sidecar with input file
                    # SHA-256s, tool/checkpoint versions, hashed CLI args
                    # / config, and python package versions. Restart logic
                    # checks the manifest hash, not just file existence —
                    # prevents silent reuse when params changed but files
                    # already exist.
"""
