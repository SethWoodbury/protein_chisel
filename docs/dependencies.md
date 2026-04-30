# External dependencies

Living index of every external thing this codebase depends on, with cluster paths and provenance.

When you add a new tool wrapper, also add the underlying dependency here. Whenever a path changes, update both `src/protein_chisel/paths.py` and this file.

---

## Apptainer images

| Container | Cluster path | Built by | Spec | Notes |
|---|---|---|---|---|
| `esmc.sif` | `/net/software/containers/users/woodbuse/esmc.sif` | this lab | `/net/software/containers/users/woodbuse/spec/esmc.spec` | Python 3.12 + torch 2.11+cu128 + ESM-C 3.2.3 + transformers 4.48.1 + foldseek (avx2 static) + py_contact_ms + PROPKA. Supports Ampere / Ada / Hopper / Blackwell. |
| `pyrosetta.sif` | `/net/software/containers/pyrosetta.sif` | softadmin | `/net/software/containers/spec/pyrosetta.spec` | Python 3.12 + PyRosetta. Used for SASA, repack, scoring, hbond detection. |
| `rosetta.sif` | `/net/software/containers/rosetta.sif` | softadmin | `/net/software/containers/spec/rosetta.spec` | Full Rosetta suite. Reserved for `rosetta_scripts` ligand interface ddG (planned, not wired). |
| `universal.sif` | `/net/software/containers/universal.sif` | softadmin | (in `/net/software/containers/spec/`) | General-purpose Python 3.11 environment. **Has fair-esm 2.0.1 — not compatible with new evolutionaryscale `esm` package.** Used by `tools/ligand_mpnn` (fused_mpnn build). |
| `mlfold.sif` | `/net/software/containers/mlfold.sif` | softadmin | — | Used in the legacy LigandMPNN runner path. The modern fused_mpnn runner uses `universal.sif`. |
| `mpnn_binder_design.sif` | `/net/software/containers/mpnn_binder_design.sif` | softadmin | — | Optional alternate MPNN routing. |
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
| LigandMPNN (legacy) | `/net/software/lab/mpnn/proteinmpnn/ligandMPNN/` | https://github.com/dauparas/LigandMPNN | (subsumed by fused_mpnn build) |
| ProteinMPNN | `/net/software/lab/mpnn/proteinmpnn/` | https://github.com/dauparas/ProteinMPNN | (subsumed by LigandMPNN; kept for reference) |
| **fused_mpnn (lab default)** | `/net/software/lab/fused_mpnn/seth_temp/run.py` | (lab) | `tools/ligand_mpnn` — **the modern runner**, honors `--repack_everything 0`, supports `--bias_AA_per_residue_multi`, supports `--enhance plddt_residpo_*` |
| LigandMPNN weights | `/net/databases/mpnn/ligand_model_weights` | — | LigandMPNN checkpoints |
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
| **fpocket** | (not installed yet — TBD) | https://github.com/Discngine/fpocket. The wrapper at [tools/fpocket_run.py](../src/protein_chisel/tools/fpocket_run.py) is functional once you point it at a binary; without one it raises an informative `RuntimeError`. **User TODO: add `apt install fpocket` to a future sif.** |
| ThermoMPNN | (not installed yet — TBD) | https://github.com/Kuhlman-Lab/ThermoMPNN. |
| ESM-IF1 | inside `esmfold.sif` | Old fair-esm; usable for ensemble diversity. |
| **ProLIF** | (not installed yet — TBD) | https://github.com/chemosim-lab/ProLIF — protein-ligand interaction fingerprints (hbonds, salt bridges, π-stacks, hydrophobic, etc). Useful for *characterizing* a design's interaction pattern with the ligand. **No wrapper yet.** |
| **pdbe-arpeggio** | (not installed yet — TBD) | https://github.com/PDBeurope/arpeggio — interaction profiler (alternative to PLIF). **No wrapper yet.** |
| PLIP | (not installed yet — TBD) | https://github.com/pharmai/plip — alternative interaction profiler. Pick one of ProLIF / Arpeggio / PLIP, not multiple. |
| APBS | (not installed yet — TBD) | https://www.poissonboltzmann.org/ — electrostatic potential maps. Heavy; reserve for late-stage characterization. |
| py_contact_ms | inside `esmc.sif` (vendored) | https://github.com/bcov77/py_contact_ms — distance-weighted contact molecular surface, NumPy-only, per-atom CMS. Strictly better than Rosetta's `ContactMolecularSurface` filter for interface ranking. Used by `tools/contact_ms`. MIT license. |
| PROPKA | inside `esmc.sif` | https://github.com/jensengroup/propka — pKa estimator for catalytic ionizables. Used by `tools/catalytic_pka`. |
| **CAVER** | (not bundled — TBD) | https://www.caver.cz/ — tunnel/channel detection. Could be useful for enzymes whose substrate must traverse a buried tunnel. **No wrapper yet.** |
| npose | optional, in `/home/bcov/sc/random/npose` | Brian's lightweight pose abstraction (pure NumPy, faster than PyRosetta for pure geometry). |

---

## Per-tool installation status

| Tool wrapper in this repo | Source file | Container | Status |
|---|---|---|---|
| `tools/esmc_score`, `tools/esmc_logits` | `tools/esmc.py` | `esmc.sif` | wrapped + tested |
| `tools/saprot_score`, `tools/saprot_logits` | `tools/saprot.py` | `esmc.sif` | wrapped + tested |
| `tools/contact_ms` | `tools/contact_ms.py` | `esmc.sif` | wrapped + tested |
| `tools/catalytic_pka` | `tools/catalytic_pka.py` | `esmc.sif` (PROPKA) | wrapped + tested |
| `tools/sample_with_ligand_mpnn` | `tools/ligand_mpnn.py` | `universal.sif` (fused_mpnn) | wrapped + helper-tested (no end-to-end) |
| `tools/classify_positions` | `tools/classify_positions.py` | `pyrosetta.sif` | wrapped + tested |
| `tools/backbone_sanity` | `tools/backbone_sanity.py` | `pyrosetta.sif` | wrapped + tested |
| `tools/shape_metrics` | `tools/shape_metrics.py` | `pyrosetta.sif` | wrapped + tested |
| `tools/secondary_structure`, `tools/ss_summary` | `tools/secondary_structure.py` | `pyrosetta.sif` | wrapped + tested |
| `tools/ligand_environment` | `tools/ligand_environment.py` | `pyrosetta.sif` | wrapped + tested |
| `tools/chemical_interactions` | `tools/chemical_interactions.py` | `pyrosetta.sif` | wrapped + tested |
| `tools/buns` | `tools/buns.py` | `pyrosetta.sif` | wrapped + tested |
| `tools/catres_quality` | `tools/catres_quality.py` | `pyrosetta.sif` | wrapped + tested |
| `tools/theozyme_satisfaction` | `tools/theozyme_satisfaction.py` | host (numpy) | wrapped + tested |
| `tools/preorganization` | `tools/preorganization.py` | `pyrosetta.sif` | **wrapped, untested** |
| `tools/fpocket_run` | `tools/fpocket_run.py` | (binary required) | **wrapped, binary install pending** |
| `tools/metal3d_score` | `tools/metal3d_score.py` | `metal3d.sif` | **HETATM scan only; inference path stubbed** |
| `tools/prolif_fingerprint` | (not started) | TBD (custom sif) | **stub planned, no source file** |
| `tools/arpeggio_interactions` | (not started) | TBD | **stub planned, no source file** |
| `tools/caver_tunnels` | (not started) | TBD | **stub planned, no source file** |
| `tools/pyrosetta_repack` | (not started) | `pyrosetta.sif` | not yet wrapped |
| `tools/rosetta_ligand_ddg` | (not started) | `rosetta.sif` | not yet wrapped |
| `tools/thermompnn` | (not started) | TBD | not yet wrapped, weights pending |
| `tools/per_residue_ddg` | (not started) | `pyrosetta.sif` | not yet wrapped |
| `tools/surface_composition` | (not started) | `pyrosetta.sif` | polars per SASA — not yet wrapped |
| `tools/rosetta_metrics_xml` | (not started) | `rosetta.sif` | not yet wrapped |
| `tools/esm_if` | (not started) | `esmfold.sif` | not yet wrapped |

---

## Reference scripts to modernize

Living in `~/special_scripts/`. These contain patterns and metric definitions we're porting into `protein_chisel`. Don't import from them — use them as reference and write modern equivalents in `tools/` or `filters/`.

| Legacy script | What it does | Where it lands here |
|---|---|---|
| `metrics_and_hbond_rosetta_seth_no_RELAX_V2.py` | RosettaScripts XML wrapping ~25 metrics: SAP, NetCharge (no HIS), Ddg, LigInterfaceEnergy, ContactMolecularSurface, ShapeComplementarity, Holes/InterfaceHoles, ExposedHydrophobics, SecondaryStructureCount, PreProline, LongestContPolar/Apolar, ElectrostaticComplementarity, plus per-atom hbonds and per-atom ligand SASA. | Split across `tools/pyrosetta_repack` (planned), `tools/rosetta_ligand_ddg` (planned), `tools/secondary_structure`, `tools/chemical_interactions`, `filters/sap_score` (planned), `filters/protparam`. The XML pattern stays — we just decompose it into individually-callable metric functions. |
| `hbonding_network.py` | Iterates pose's `HBondSet`, classifies prot↔water / water↔water / lig↔water hbonds, dumps CSV. | `tools/chemical_interactions` (hbond detection module). |
| `design_filtering/metric_monster__MAIN.py` | Half-finished orchestrator: `contact_counter` + `protein_size_shape_metrics` + `execute_fpocket_on_holo_structure`, combines CSVs with prefixes, supports parallel runs. | `pipelines/comprehensive_metrics`. The CSV-prefix-merge pattern is good and we kept it for results assembly. |
| `ESM/ESM_mutation_suggestions.py` | ESM-2 masked-token mutation suggestions per position. | `tools/esmc_logits` (masked-token marginal). API has changed — old fair-esm `Alphabet.get_batch_converter` becomes new evolutionaryscale `model.encode(ESMProtein(...))`. |
| `ESM/ESM_score_sequences.py` | ESM-2 pseudo-perplexity scoring of sequences in a fasta. | `tools/esmc_score` (perplexity mode). |
| `upgraded_fastMPNNdesign/run_pipeline.py` + `pipeline_constants.py` | A more polished pipeline runner with `PipelineRunner` class, structured StepOutputs dataclass, hash-based internal basenames, container/script registry. | Reference for `pipelines/` orchestrator structure. The `PipelineRunner` + step-defaults + arg-mapping pattern is solid; adopt the spirit, not the exact code. |
| `/net/software/lab/scripts/enzyme_design/SETH_TEMP_UTILS/process_diffusion3_outputs__REORG.py` | Post-processes diffusion outputs: REMARK 666 parsing, motif/template re-mapping, scaffold quality (chainbreak / rCA_nonadj / loop_frac / longest_helix / Rg / loop_at_motif), ligand environment (lig_dist, SASA, SASA_rel, term_mindist), motif geometry (bondlen_dev, cart_bonded_avg, fa_dun_avg), Rosetta scorefile output, multiprocessing over many designs. | Several modules here. `io/pdb.parse_remark_666` + `write_remark_666` ports directly. `tools/backbone_sanity` (chainbreak, rCA_nonadj, term_mindist), `tools/shape_metrics` (replaces non-standard `get_ROG` with proper Rg + asphericity), `tools/secondary_structure.ss_summary` (loop_frac, longest_helix, loop_at_motif), `tools/catres_quality` (sidechain bondlen + cart_bonded + fa_dun), `tools/ligand_environment` (lig_dist, ligand SASA, SASA_rel). |

### Brian Coventry (`/home/bcov/util/`) — patterns and metric definitions

Brian's util directory is 271 scripts of battle-tested PyRosetta + NumPy patterns. The ones we adapt:

| Script | Pattern / metric | Where it lands |
|---|---|---|
| `dump_hbset.py` | Canonical `fix_scorefxn` (decompose_bb_hb_into_pair_energies + bb_donor_acceptor_check) before `fill_hbond_set`. Required for proper hbond enumeration; without it, bb-bb hbonds collapse strangely. | [utils/pose.fix_scorefxn](../src/protein_chisel/utils/pose.py#L117); used by `tools/chemical_interactions`, `tools/buns`, `tools/catres_quality`. |
| `per_atom_sasa.py` | Per-atom SASA via `core.scoring.packing.get_surf_vol(pose, atoms, probe)`. **Probe radius 2.8 Å** for contact-region calcs (vs. 1.4 Å for solvent). | [utils/pose.get_per_atom_sasa](../src/protein_chisel/utils/pose.py#L225) (default 2.8). |
| `parse_target_buns_recalculate_white.py` | BUNS computed with a **whitelist** of allowed unsats (resno+atom name pairs). | `tools/buns` whitelist-aware mode. Critical for theozyme designs. |
| `polars_per_sasa.py`, `polars_per_sasa_raytrace.py` | Polar atom count per SASA — surface-chemistry signal complementary to SAP. | `tools/surface_composition` (planned). |
| `ddg_per_res.py`, `ddg_per_res_ala_scan.py`, `ddg_per_res_repack.py`, `ddg_per_res_buried_elec.py` | Per-residue ddG suite: basic, alanine scan, with repack, buried-electrostatics-focused. | `tools/per_residue_ddg` (planned). |
| `boltzmann_grid_sampler.py` (Adam Moyer / Brian) | Boltzmann-weighted grid sampling for hyperparameter tuning. Continuous bounds, update-with-results loop. | `utils/hyperparam_search` (planned). |
| `find_full_pose_buried_atoms_near_interface_chainB.py` | Buried-atom detection at protein/ligand interface. | `tools/buns` companion. |
| `cage_max_sphere.py` | Max inscribed sphere — alternative pocket size measure. | `tools/fpocket_run` companion / cross-check (planned). |
| `bfactor_lddt.py` | Map B-factor ↔ pLDDT (useful for AF prediction interpretation). | `utils/pose` helper (planned). |
| `cluster_interface_sidechains.py` | Sidechain clustering at interfaces. | Reference for `scoring/diversity` if mutable-position-Hamming proves insufficient. |
| `net_charge_sequences.py` | Net charge from sequence (no pH; just counts D, E, K, R, H). | Already covered by [`filters/protparam.charge_at_pH7_no_HIS`](../src/protein_chisel/filters/protparam.py#L111). |
