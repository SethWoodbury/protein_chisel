# Capabilities — what protein_chisel can do today

Quick-reference for new users. Bullets only. For depth, follow the links into per-area docs.

## Read PDBs and parse theozyme inputs
- Parse REMARK 666 catalytic-residue lines into typed `CatalyticResidue` records ([io/pdb.parse_remark_666](../src/protein_chisel/io/pdb.py))
- Round-trip REMARK 666 across derived PDBs (write_remark_666 inserts before first ATOM)
- Multi-chain catalytic residues without collision (`key_by="chain_resno"`)
- Fallback `parse_catres_spec(["A94-96", "B101"])` when REMARK 666 is absent
- Quick PDB summary: chain composition, ligands, water, elements
- Detect apo vs holo, find ligand identity
- Extract 1-letter sequence from ATOM records (handles MSE/SEC/PYL)

## Classify residues
- Per-residue class: `active_site` / `first_shell` / `pocket` / `buried` / `surface` / `ligand`
- Per-residue features: SASA, distance to ligand / catalytic, DSSP (full + reduced), phi/psi
- Driven by REMARK 666 (or explicit catres spec); pocket residues injected by caller

## Score one structure with descriptive metrics
- **Backbone sanity**: chainbreak max, count above 4.5 Å, rCA-nonadj min, terminus-to-ligand
- **Shape**: proper sqrt-mean-square Rg, length-normalized Rg, asphericity, acylindricity, rel shape anisotropy, principal lengths
- **Secondary structure**: DSSP per residue + summary (helix/sheet/loop fractions, longest runs, loop_at_motif)
- **Ligand environment**: lig_dist, residues within 5/8 Å, ligand SASA (full + relative-to-free), per-atom SASA on user-named atoms
- **Chemical interactions**: hbonds with energies + heavy-atom + H-atom names, salt bridges, π-π (with plane angle classification), π-cation
- **BUNS**: buried unsatisfied polar atoms with whitelist support (bcov pattern)
- **Catres quality**: cart_bonded + fa_dun + sidechain bond-length deviation per catalytic residue

## Score sequences with PLMs
- ESM-C masked-LM marginals (proper L forward passes; not single-pass)
- ESM-C pseudo-perplexity (300m or 600m models)
- SaProt structure-aware logits (foldseek 3Di + AA), 3Di-marginalized over 20 AAs
- SaProt pseudo-perplexity (35m / 650m / 1.3b models)
- Calibrated PLM fusion: log-odds + entropy match + position-class weights + disagreement shrinkage

## Compare design to theozyme reference
- Kabsch alignment of catalytic motif (Cα RMSD, heavy-atom RMSD, max atom deviation)
- Per-residue RMSD (Cα and heavy)
- Per-(catalytic atom, ligand atom) distance pairs (always available, even without reference)
- Driven by REMARK 666, RFdiffusion3 fixed_atoms_json, or explicit catres list

## Active-site preorganization (untested but implemented)
- N-trajectory repack ensemble with catalytic residues + ligand frozen
- Per-residue centroid variance over the ensemble = preorganization signal

## Catalytic pKa
- PROPKA wrapper for catalytic residues (REMARK 666 by default)
- pKa shift vs solution reference per residue

## Interface scoring
- Distance-weighted contact molecular surface (`area · exp(-0.5 d²)`) via py_contact_ms
- Better signal than Rosetta's CMS filter for ranking; per-target-atom CMS available

## Sequence filtering
- Biopython ProtParam: pI, instability index, GRAVY, MW, aromaticity, charge_at_pH (with HIS), charge_at_pH7_no_HIS (matches legacy Rosetta NetCharge)
- Sequence-derived secondary-structure fractions
- Extinction coefficients at 280 nm
- Protease site detection: kex2, trypsin, ompT, thrombin, furin, caspase, custom regex
- Host-specific patterns: E. coli (ompT, trypsin) and yeast (kex2, N-glycosylation)
- General forbidden motifs: poly-G/N/Q runs, CCC, PPPP, internal Met
- Length + terminal AA constraints

## Sample sequences
- LigandMPNN sampling via the modern fused_mpnn build (`/net/software/lab/fused_mpnn/seth_temp/run.py`)
- `repack_everything=0` — catalytic residues stay locked
- `--bias_AA_per_residue_multi` — fed the calibrated PLM fusion bias matrix
- `--fixed_residues_multi` — catalytic residue identities locked
- `omit_AA="CX"` — no Cys / unknown by default

## Iterative single-mutation walks
- Constrained local search (PLM marginals propose, hard filters accept)
- Real Metropolis-Hastings (with q-correction term + linear-anneal temperature)
- Multiple independent chains for diagnostics
- Per-step walk log + every accepted candidate as a `CandidateSet`

## Multi-pose handling
- `PoseSet` abstraction: list of `PoseEntry` with `sequence_id`, `fold_source`, `conformer_index`, `parent_design_id`, `is_apo`, free-form `meta`
- Internally normalize: single PDB → PoseSet of size 1
- Filter / group by metadata
- Single PDB or multi-conformer inputs go through the same code path

## Aggregation across conformers
- Failure metrics → max (worst-case), `any_nonzero` flag
- Descriptive metrics → mean, std, min, max
- Sequence-only metrics → first
- Vote: per-conformer pass-fraction
- Apo/holo → explicit paired delta
- Cross-source preserved as `src__designed__*` / `src__AF3_seed1__*` columns (never averaged across sources as exchangeable samples)

## Multi-objective ranking
- Hard constraints first (`HardConstraint` with min/max; NaN fails)
- ε-Pareto on 3-5 real objectives (binned to avoid float-precision spurious non-dominated)
- NSGA-II crowding distance for spacing within the front
- Hamming-distance diversity selection over **mutable / pocket positions only** (not full-length)

## Provenance / restart
- Manifest hashing: SHA-256 of input files + JSON of config + tool versions + package versions
- Restart-skip in `comprehensive_metrics` and `naturalness_metrics` checks the manifest hash, not just file existence
- `MetricTable.merge` raises on column collisions (defaults to safe; explicit `on_collision="left"`/`"right"` override)
- All artifacts have stable schema definitions in [io/schemas.py](../src/protein_chisel/io/schemas.py)

## Pipelines
- **`comprehensive_metrics`** — full descriptive structural battery on a PoseSet, single sif (`pyrosetta.sif`)
- **`naturalness_metrics`** — ESM-C + SaProt scoring + PLM fusion bias artifacts (`esmc.sif`, GPU)
- **`sequence_design_v1`** — 5-stage end-to-end: classify → PLM logits → fusion → MPNN sample → hard filters → diversity selection (multi-sif)
- **`iterative_optimize`** — single-mutation walk pipeline (host, numpy)

## Cluster integration
- `ApptainerCall` helper: build commands with default repo bind, `PYTHONPATH=/code/src`, `--nv`, HF cache binds
- Pre-configured calls: `esmc_call`, `pyrosetta_call`, `rosetta_call`, `metal3d_call`, `universal_call`, `mlfold_call`
- sbatch templates for the three pipelines in [scripts/](../scripts/)
- Single source of truth for cluster paths: [paths.py](../src/protein_chisel/paths.py)

## CLI
- `chisel comprehensive-metrics <pdb> ...` — runs the full structural battery
- `chisel classify-positions <pdb> --out <path> ...`
- `chisel esmc-score <sequence>`
- `chisel saprot-score <pdb>`
