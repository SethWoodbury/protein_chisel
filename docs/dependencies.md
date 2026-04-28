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
