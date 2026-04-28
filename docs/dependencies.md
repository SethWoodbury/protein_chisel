# External dependencies

Living index of every external thing this codebase depends on, with cluster paths and provenance.

When you add a new tool wrapper, also add the underlying dependency here. Whenever a path changes, update both `src/protein_chisel/paths.py` and this file.

---

## Apptainer images

| Container | Cluster path | Built by | Spec | Notes |
|---|---|---|---|---|
| `esmc.sif` | `/net/software/containers/users/woodbuse/esmc.sif` | this lab | `/net/software/containers/users/woodbuse/spec/esmc.spec` | Python 3.12 + torch 2.11+cu128 + ESM-C 3.2.3 + transformers 4.48.1 + foldseek (avx2 static). Supports Ampere / Ada / Hopper / Blackwell. |
| `pyrosetta.sif` | `/net/software/containers/pyrosetta.sif` | softadmin | `/net/software/containers/spec/pyrosetta.spec` | Python 3.12 + PyRosetta. Used for SASA, repack, scoring, hbond detection. |
| `rosetta.sif` | `/net/software/containers/rosetta.sif` | softadmin | `/net/software/containers/spec/rosetta.spec` | Full Rosetta suite. Used for `rosetta_scripts` ligand interface ddG. |
| `universal.sif` | `/net/software/containers/universal.sif` | softadmin | (in `/net/software/containers/spec/`) | General-purpose Python 3.11 environment. **Has fair-esm 2.0.1 — not compatible with new evolutionaryscale `esm` package.** |
| `esmfold.sif` | `/net/software/containers/esmfold.sif` | softadmin | `/net/software/containers/spec/esmfold.spec` | Old ESM (`fair-esm` 2.0.1) on Python 3.7. Used by legacy `~/special_scripts/ESM/*.py`. |
| `af3.sif` | `/net/software/containers/af3.sif` | softadmin | (in spec/) | AlphaFold3. Used as the *final* filter outside the inner loop. |
| `metal3d.sif` | `/net/software/containers/pipelines/metal3d.sif` | jbutch (lab pipelines) | `/net/software/containers/pipelines/metal3d.def` | Metal3D — predicts metal-binding sites and scores existing metal coordination. Reference: https://github.com/baker-laboratory/pipelines/pull/258. |

### Cluster `.sif` build conventions

- User builds go in `/net/software/containers/users/$USER/`.
- Build with `apptainer build --fakeroot ...` (no sudo on this cluster).
- `/net/software` has 10-minute NFS attribute caching on compute nodes — first sbatch after a rebuild can fail with `lstat: no such file or directory` or stale-content errors. Retry after 5–10 min.

---

## HuggingFace weight caches

Cluster's canonical `/net/databases/huggingface/cache` is a symlink to per-node `/scratch` and not writable from login. We use parallel directories per model family, all writable to the `compute` group.

| Cache | Path | Models |
|---|---|---|
| ESM-C | `/net/databases/huggingface/esmc/` | EvolutionaryScale/esmc-300m-2024-12, esmc-600m-2024-12. esmc-6b is **API-only** (Forge), not downloadable. |
| SaProt | `/net/databases/huggingface/saprot/` | westlake-repl/SaProt_35M_AF2, SaProt_650M_PDB, SaProt_1.3B_AFDB_OMG_NCBI. |

### Pointing a job at a cache
```bash
apptainer exec --env HF_HOME=/net/databases/huggingface/saprot \
               --env HF_HUB_CACHE=/net/databases/huggingface/saprot/hub \
               $ESMC_SIF python …
```

---

## Static / source-installed model checkouts

| Tool | Cluster path | Source repo | Used by |
|---|---|---|---|
| LigandMPNN | `/net/software/lab/mpnn/proteinmpnn/ligandMPNN/` | https://github.com/dauparas/LigandMPNN | `tools/ligand_mpnn` (planned) |
| ProteinMPNN | `/net/software/lab/mpnn/proteinmpnn/` | https://github.com/dauparas/ProteinMPNN | (subsumed by LigandMPNN; kept for reference) |
| FAMPNN | `/net/software/lab/fampnn/` | (lab fork) | possible alternate sampler |

---

## Static weight files (legacy `.pt` format)

| Model | Path | Size | Notes |
|---|---|---|---|
| ESM-2 8M | `/net/databases/esmfold/esm2_t6_8M_UR50D.pt` | 29 MB | `fair-esm` format |
| ESM-2 35M | `/net/databases/esmfold/esm2_t12_35M_UR50D.pt` | 128 MB | |
| ESM-2 150M | `/net/databases/esmfold/esm2_t30_150M_UR50D.pt` | 566 MB | |
| ESM-2 650M | `/net/databases/esmfold/esm2_t33_650M_UR50D.pt` | 2.5 GB | |
| ESM-2 3B | `/net/databases/esmfold/esm2_t36_3B_UR50D.pt` | 5.3 GB | |
| ESMFold 3B | `/net/databases/esmfold/esmfold_3B_v1.pt` | 2.6 GB | |
| ESM-2 15B | (not downloaded) | 26 GB | Use only if 3B is insufficient. |

These weights are the **`fair-esm`** format and are loaded inside `esmfold.sif` (Python 3.7). They are NOT compatible with the new evolutionaryscale `esm` package in `esmc.sif`.

---

## Runtime tools

| Tool | Where | Notes |
|---|---|---|
| foldseek | bundled in `esmc.sif` (`/usr/local/bin/foldseek`) | Static avx2 build from https://mmseqs.com/foldseek/. The cluster's `/net/software/foldseek/foldseek` is dynamically linked and won't run inside our minimal sif. |
| fpocket | (not installed yet — TBD) | https://github.com/Discngine/fpocket. Add to a future sif or apt install in `pyrosetta.sif`. |
| ThermoMPNN | (not installed yet — TBD) | https://github.com/Kuhlman-Lab/ThermoMPNN. |
| ESM-IF1 | inside `esmfold.sif` | Old fair-esm; usable for ensemble diversity. |
| ProLIF | (not installed yet — TBD) | https://github.com/chemosim-lab/ProLIF — protein-ligand interaction fingerprints (hbonds, salt bridges, π-stacks, hydrophobic, etc). Useful for *characterizing* a design's interaction pattern with the ligand. |
| PLIP | (not installed yet — TBD) | https://github.com/pharmai/plip — alternative interaction profiler. Pick one of ProLIF or PLIP, not both. |
| APBS | (not installed yet — TBD) | https://www.poissonboltzmann.org/ — electrostatic potential maps. Heavy; reserve for late-stage characterization. |
| py_contact_ms | external pip / vendored | https://github.com/bcov77/py_contact_ms — distance-weighted contact molecular surface, NumPy-only, per-atom CMS. Strictly better than Rosetta's `ContactMolecularSurface` filter for interface ranking. Used by `tools/contact_ms`. MIT license. |
| npose | optional, in `/home/bcov/sc/random/npose` | Brian's lightweight pose abstraction (pure NumPy, faster than PyRosetta for pure geometry). Used by some `polars_per_sasa_raytrace` and other bcov scripts. We can either vendor a slice or rely on the cluster path; treat as optional. |
| PROPKA | (not installed yet — TBD) | https://github.com/jensengroup/propka — pKa estimator for catalytic ionizables. Used by `tools/catalytic_pka`. |

---

## Per-tool installation status (for the maintainer)

| Tool wrapper in this repo | Container | External weights | Status |
|---|---|---|---|
| `tools/esmc_score` | `esmc.sif` | `HF_CACHE_ESMC` | weights cached, not yet wrapped |
| `tools/esmc_logits` | `esmc.sif` | `HF_CACHE_ESMC` | weights cached, not yet wrapped |
| `tools/saprot_score` | `esmc.sif` | `HF_CACHE_SAPROT` | weights cached, not yet wrapped |
| `tools/saprot_logits` | `esmc.sif` | `HF_CACHE_SAPROT` | weights cached, not yet wrapped |
| `tools/ligand_mpnn` | mpnn-related sif (TBD) | `LIGAND_MPNN_DIR` | code present on cluster, not yet wrapped |
| `tools/classify_positions` | `pyrosetta.sif` | — | not yet wrapped |
| `tools/pyrosetta_repack` | `pyrosetta.sif` | — | not yet wrapped |
| `tools/rosetta_ligand_ddg` | `rosetta.sif` | — | not yet wrapped |
| `tools/fpocket_run` | TBD | fpocket binary | install pending |
| `tools/thermompnn` | TBD | ThermoMPNN weights | install pending |
| `tools/esm_if` | `esmfold.sif` | `ESM2_WEIGHTS_DIR` | not yet wrapped |
| `tools/secondary_structure` | `pyrosetta.sif` | — | DSSP per-residue labels |
| `tools/chemical_interactions` | `pyrosetta.sif` | — | hbonds, salt bridges, π-π, π-cation, fa_elec |
| `tools/metal3d_score` | `metal3d.sif` | (its own weights, in sif) | not yet wrapped |
| `tools/prolif_or_plip` | TBD | — | install pending |

---

## Reference scripts to modernize

Living in `~/special_scripts/`. These contain patterns and metric definitions we're porting into `protein_chisel`. Don't import from them — use them as reference and write modern equivalents in `tools/` or `filters/`.

| Legacy script | What it does | Where it lands here |
|---|---|---|
| `metrics_and_hbond_rosetta_seth_no_RELAX_V2.py` | RosettaScripts XML wrapping ~25 metrics: SAP, NetCharge (no HIS), Ddg, LigInterfaceEnergy, ContactMolecularSurface, ShapeComplementarity, Holes/InterfaceHoles, ExposedHydrophobics, SecondaryStructureCount, PreProline, LongestContPolar/Apolar, ElectrostaticComplementarity, plus per-atom hbonds and per-atom ligand SASA. | Split across `tools/pyrosetta_repack`, `tools/rosetta_ligand_ddg`, `tools/secondary_structure`, `tools/chemical_interactions`, `filters/sap_score`, `filters/protparam`. The XML pattern stays — we just decompose it into individually-callable metric functions. |
| `hbonding_network.py` | Iterates pose's `HBondSet`, classifies prot↔water / water↔water / lig↔water hbonds, dumps CSV. | `tools/chemical_interactions` (hbond detection module). |
| `design_filtering/metric_monster__MAIN.py` | Half-finished orchestrator: `contact_counter` + `protein_size_shape_metrics` + `execute_fpocket_on_holo_structure`, combines CSVs with prefixes, supports parallel runs. | The orchestration pattern → `pipelines/`. The individual metric calculators → `tools/`. The CSV-prefix-merge pattern is good and we'll keep it for results assembly. |
| `ESM/ESM_mutation_suggestions.py` | ESM-2 masked-token mutation suggestions per position. | `tools/esmc_score` (masked-token ESM-C scoring). API has changed — old fair-esm `Alphabet.get_batch_converter` becomes new evolutionaryscale `model.encode(ESMProtein(...))`. |
| `ESM/ESM_score_sequences.py` | ESM-2 pseudo-perplexity scoring of sequences in a fasta. | `tools/esmc_score` (perplexity mode). |
| `upgraded_fastMPNNdesign/run_pipeline.py` + `pipeline_constants.py` | A more polished pipeline runner with `PipelineRunner` class, structured StepOutputs dataclass, hash-based internal basenames, container/script registry. | Reference for `pipelines/` orchestrator structure. The `PipelineRunner` + step-defaults + arg-mapping pattern is solid; adopt the spirit, not the exact code. |
| `/net/software/lab/scripts/enzyme_design/SETH_TEMP_UTILS/process_diffusion3_outputs__REORG.py` | Post-processes diffusion outputs: REMARK 666 parsing, motif/template re-mapping, scaffold quality (chainbreak / rCA_nonadj / loop_frac / longest_helix / Rg / loop_at_motif), ligand environment (lig_dist, SASA, SASA_rel, term_mindist), motif geometry (bondlen_dev, cart_bonded_avg, fa_dun_avg), Rosetta scorefile output, multiprocessing over many designs. | Several modules here. `io/pdb.get_matcher_residues` + `add_matcher_line_to_pose` ports directly. `tools/backbone_sanity` (chainbreak, rCA_nonadj, term_mindist), `tools/shape_metrics` (replace its non-standard `get_ROG` with proper Rg + asphericity), `tools/ss_summary` (loop_frac, longest_helix, loop_at_motif), `tools/catres_quality` (sidechain bondlen + cart_bonded + fa_dun), `tools/ligand_environment` (lig_dist, ligand SASA, SASA_rel). The Rosetta-scorefile writer is a useful output format we may keep alongside TSV. |

### Brian Coventry (`/home/bcov/util/`) — patterns and metric definitions

Brian's util directory is 271 scripts of battle-tested PyRosetta + NumPy patterns. The ones we adapt:

| Script | Pattern / metric | Where it lands |
|---|---|---|
| `dump_hbset.py` | Canonical `fix_scorefxn` (decompose_bb_hb_into_pair_energies + bb_donor_acceptor_check) before `fill_hbond_set`. Required for proper hbond enumeration; without it, bb-bb hbonds collapse strangely. | `utils/pose.fix_scorefxn`; used by `tools/chemical_interactions`, `tools/per_residue_ddg`, others. |
| `per_atom_sasa.py` | Per-atom SASA via `core.scoring.packing.get_surf_vol(pose, atoms, probe)`. **Probe radius 2.8 Å** for contact-region calcs (vs. 1.4 Å for solvent). | `utils/pose.get_per_atom_sasa(pose, probe=2.8)`. |
| `parse_target_buns_recalculate_white.py` | BUNS computed with a **whitelist** of allowed unsats (resno+atom name pairs). | `tools/buns` whitelist-aware mode. Critical for theozyme designs. |
| `polars_per_sasa.py`, `polars_per_sasa_raytrace.py` | Polar atom count per SASA — surface-chemistry signal complementary to SAP. Raytrace variant uses npose for accurate SASA. | `tools/surface_composition` (basic and raytrace modes). |
| `ddg_per_res.py`, `ddg_per_res_ala_scan.py`, `ddg_per_res_repack.py`, `ddg_per_res_buried_elec.py` | Per-residue ddG suite: basic, alanine scan, with repack, buried-electrostatics-focused. | `tools/per_residue_ddg` with `--mode {basic,ala_scan,repack,buried_elec}`. |
| `boltzmann_grid_sampler.py` (Adam Moyer / Brian) | Boltzmann-weighted grid sampling for hyperparameter tuning. Continuous bounds, update-with-results loop. | `utils/hyperparam_search` for fusion-weight / MH-τ / threshold sweeps. |
| `find_full_pose_buried_atoms_near_interface_chainB.py` | Buried-atom detection at protein/ligand interface (the "chain B = ligand" assumption maps cleanly). | `tools/buns` companion. |
| `cage_max_sphere.py` | Max inscribed sphere — alternative pocket size measure. | `tools/fpocket_run` companion / cross-check. |
| `bfactor_lddt.py` | Map B-factor ↔ pLDDT (useful for AF prediction interpretation). | `utils/pose` helper. |
| `cluster_interface_sidechains.py` | Sidechain clustering at interfaces. | Reference for `scoring/diversity` if mutable-position-Hamming proves insufficient. |
| `dump_hbset.py` (the `fix_scorefxn` pattern itself) | Setting `decompose_bb_hb_into_pair_energies(True)` is mandatory for any per-residue energy decomposition — without it `pose.energies().residue_total_energies()` mixes bb energies between paired residues. | Consumed everywhere we score per-residue. Gets a comment in `utils/pose` explaining why. |
| `net_charge_sequences.py` | Net charge from sequence (no pH; just counts D, E, K, R, H). | Already covered by `filters/protparam.charge_at_pH` and `charge_at_pH7_no_HIS`. Keep bcov version as a fast sequence-only fallback. |
