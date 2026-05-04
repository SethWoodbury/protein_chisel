# Dependencies & Apptainer

This document describes every external dependency `protein_chisel` requires:
container images (`.sif`), external binaries, model checkpoints, data paths,
Python packages, and the exact `apptainer` command patterns used to launch
each stage. It also covers the Slurm partitions we run on and the
scaffold-specific assets needed by the example PTE_i1 design campaign.

---

## 1. Container Images (`.sif`)

The pipeline is split across three Apptainer containers. Each stage of
`iterative_design_v2.py` (and the standalone scripts under `scripts/`) selects
the appropriate one. **Do not** swap containers casually — `esmc.sif` is the
only one with `py_contact_ms`, and `pyrosetta.sif` is the only one with the
full PyRosetta param library.

### 1.1 `universal.sif`

- **Path:** `/net/software/containers/universal.sif`
- **Role:** Main workhorse container. Most CPU stages run here.
- **Contains:**
  - PyRosetta (sufficient for scoring / sequence-recovery / SAP / classify)
  - `fpocket` binary (also available on host at
    `/net/software/lab/fpocket/bin/fpocket`)
  - `freesasa` (Shrake–Rupley fallback when `DAlphaBall` fails)
  - `biopython` (used for `ProteinAnalysis` / ProtParam-style descriptors)
  - `numpy`, `pandas`, `scipy`
  - `propka` **3.5.1** — present, but **not** auto-invoked by the current
    pipeline. Available for future `pKa` prediction stages.

Stages that use `universal.sif`:
- pocket detection (`run_fpocket.py`)
- SASA / SAP scoring
- biopython descriptor scoring
- `iterative_design_v2.py` orchestrator (top-level driver)

### 1.2 `esmc.sif`

- **Path:** `/net/software/containers/users/woodbuse/esmc.sif`
- **Role:** GPU-bound PLM and contact-surface container.
- **Contains:**
  - **ESM-C** (`esm` package) — PLM precompute (per-residue embeddings,
    log-likelihoods)
  - **SaProt** — structure-aware PLM
  - **`py_contact_ms`** (bcov77's contact molecular surface implementation) —
    used by the optional CMS final-stage filter
  - CUDA / cuDNN matched to the in-container PyTorch build

Stages that use `esmc.sif`:
- `plm_precompute.py` (ESM-C and SaProt embeddings, runs on GPU)
- optional CMS final-stage scoring

This is the only container that should be launched with `--nv` (see §5).

### 1.3 `pyrosetta.sif`

- **Path:** `/net/software/containers/pyrosetta.sif`
- **Role:** PyRosetta with the **full** params/database tree.
- **Contains:** PyRosetta (more complete library than what ships in
  `universal.sif`).

Stages that use `pyrosetta.sif`:
- `classify_positions` step. We saw cases where `universal.sif`'s PyRosetta
  was missing some param/HBNet auxiliary files. `pyrosetta.sif` is the
  reliable choice for any classification step that touches the param tree
  or full Rosetta database.

> Rule of thumb: if a stage works on `universal.sif`, prefer it (smaller,
> faster). Drop down to `pyrosetta.sif` only when classify or a Rosetta
> protocol complains about missing params.

---

## 2. External Binaries / Tools

### 2.1 `fused_mpnn` (LigandMPNN sampler)

- **Entry point:** `/net/software/lab/fused_mpnn/seth_temp/run.py`
- **Role:** LigandMPNN sequence sampler with a side-chain packer head. Used
  by the `mpnn_sample` stage of `iterative_design_v2.py`.
- **LigandMPNN checkpoint:**
  `/net/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt`
  (sequence model, T=300 sampling, 10% chain-noise, 25% schedule)
- **Side-chain packer checkpoint:**
  `/net/databases/mpnn/packer_weights/s_300756.pt`

Both checkpoints are read directly from `/net/databases` — bind that path
into the container (see §5).

### 2.2 `fpocket`

- Bundled in `universal.sif` (no host bind required).
- Host fallback: `/net/software/lab/fpocket/bin/fpocket`.

### 2.3 `DAlphaBall`

- Used by Rosetta's SASA implementation under some scorefunctions.
- **Known issue:** `DAlphaBall` startup fails inside several of our
  containers (missing helper binary in `PATH`). The pipeline transparently
  falls back to **`freesasa`** (Shrake–Rupley) when this happens. If you see
  a `DAlphaBall` error in the logs followed by SASA values appearing
  anyway, that fallback is what kicked in.

---

## 3. Data Paths

All of these are bind-mounted into the containers (see §5).

### 3.1 Hugging Face caches

ESM-C and SaProt weights are cached on shared storage so the containers
don't re-download per job:

- `/net/databases/huggingface/esmc`
- `/net/databases/huggingface/saprot`

When launching `esmc.sif` you must point `HF_HOME` at the right cache:

```bash
--env "HF_HOME=/net/databases/huggingface/esmc"
# or
--env "HF_HOME=/net/databases/huggingface/saprot"
```

If `HF_HOME` is unset or wrong, the model will try to fetch from the
internet — which usually fails on compute nodes.

### 3.2 Ligand `.params` files

Per-scaffold Rosetta `.params` files live next to the scaffold's seed
PDBs. For PTE_i1 see §7.

### 3.3 Seed PDBs

Per-scaffold; passed in via the run config (see §7 for the PTE_i1 path).

---

## 4. Python Packages

These are the imports the `src/` tree relies on. They are all already
installed inside `universal.sif` and/or `esmc.sif` — this list exists so
that anyone reproducing the environment outside of Apptainer (e.g. a
local conda env for unit tests) knows what to install.

### 4.1 In `universal.sif`
- `numpy`
- `pandas`
- `scipy`
- `biopython` (`Bio.SeqUtils.ProtParam.ProteinAnalysis`)
- `freesasa` (Shrake–Rupley SASA fallback)
- `propka` 3.5.1 (available; not currently auto-used)
- `pyrosetta`

### 4.2 In `esmc.sif`
- `esm` (Meta's ESM package, providing ESM-C)
- `saprot` (custom packaging — structure-aware PLM)
- `py_contact_ms` (bcov77 contact-molecular-surface, not in any other
  container)
- `torch` (CUDA build matched to the container)

See `pyproject.toml` at the repo root for the canonical list and version
pins.

---

## 5. Apptainer Command Patterns

All stages follow the same launch shape. The container choice and the
`HF_HOME` / `--nv` flags are what vary.

### 5.1 Common bind set

Every invocation binds the same four "infrastructure" paths plus the
repo:

```bash
--bind /net/software \
--bind /net/databases \
--bind /net/scratch \
--bind /home/woodbuse \
--bind <REPO>:/code
```

Where `<REPO>` is `/home/woodbuse/codebase_projects/protein_chisel`.
Inside the container the repo appears at `/code`, which keeps paths
stable across host changes.

### 5.2 `PYTHONPATH`

```bash
--env "PYTHONPATH=/code/src:/cifutils/src"
```

`/cifutils/src` is the in-container CIF helper tree (already present in
all three containers).

### 5.3 CPU stage on `universal.sif`

```bash
apptainer exec \
    --bind /net/software --bind /net/databases \
    --bind /net/scratch  --bind /home/woodbuse \
    --bind <REPO>:/code \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    /net/software/containers/universal.sif \
    python /code/scripts/iterative_design_v2.py [args...]
```

### 5.4 GPU stage on `esmc.sif` (PLM precompute, CMS)

`--nv` is required for CUDA. `HF_HOME` must point at the model's cache.

```bash
apptainer exec --nv \
    --bind /net/software --bind /net/databases \
    --bind /net/scratch  --bind /home/woodbuse \
    --bind <REPO>:/code \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    --env "HF_HOME=/net/databases/huggingface/esmc" \
    /net/software/containers/users/woodbuse/esmc.sif \
    python /code/scripts/plm_precompute.py [args...]
```

For SaProt, swap `HF_HOME` to `/net/databases/huggingface/saprot`.

### 5.5 Classify on `pyrosetta.sif`

```bash
apptainer exec \
    --bind /net/software --bind /net/databases \
    --bind /net/scratch  --bind /home/woodbuse \
    --bind <REPO>:/code \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    /net/software/containers/pyrosetta.sif \
    python /code/scripts/classify_positions.py [args...]
```

---

## 6. Slurm Partition Info

### 6.1 GPU: `gpu-bf`

- A100 nodes seen in this session: **g1105, g2157, g2314, g2505**.
- Use this partition for `plm_precompute.py` and any CMS final-stage run.
- Memory budget: ESM-C + SaProt fit comfortably in 40GB; LigandMPNN
  inference fits in 4GB. We have not seen OOMs on A100s.

### 6.2 CPU: `cpu`

- Used for non-GPU smoke tests and for running the entire pipeline in a
  pure-CPU mode.
- Wall-clock penalty: roughly **3.6× slower** than the GPU pipeline (this
  is because PLM precompute moves from seconds to minutes per design).
- Useful for cheap correctness tests and for stages with no GPU
  dependency at all.

### 6.3 Other GPU hardware

- **A4000** — fully compatible with the current pipeline. ESM-C + SaProt
  precompute needs only **2–4 GB VRAM**, which fits on every A4000 we've
  tested.
- **B100 / B200** (Blackwell) — backward-compatible if/when they become
  available; the current PyTorch builds in `esmc.sif` will run on them
  without rebuilding. No code changes required.

---

## 7. Scaffold-Specific Data — PTE_i1 Example

The PTE_i1 campaign is the canonical "live" test for the pipeline. Use
these paths verbatim in run configs.

### 7.1 Seed PDB

```
/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb
```

This is the post-AF3 filtered reference PDB (i1 round, ZAPP-p1D1
backbone).

### 7.2 Ligand `.params`

```
/home/woodbuse/testing_space/scaffold_optimization/ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params/YYE.params
```

Ligand three-letter code: `YYE`. This is the param file you pass to
PyRosetta and to LigandMPNN's ligand-aware sampling.

### 7.3 Catalytic residue numbers

```
60, 64, 128, 131, 132, 157
```

These positions are pinned (no MPNN resampling) and are used by the
`classify_positions` step as the catalytic anchor set. They appear in
the run config as `catalytic_resnos: [60, 64, 128, 131, 132, 157]`.

---

## Quick reference table

| Dependency | Provided by | Bind / env required |
| --- | --- | --- |
| PyRosetta (scoring, SAP) | `universal.sif` | repo + `/net/databases` |
| PyRosetta (classify, full params) | `pyrosetta.sif` | repo + `/net/databases` |
| ESM-C / SaProt | `esmc.sif` | `--nv` + `HF_HOME` to matching cache |
| `py_contact_ms` (CMS) | `esmc.sif` only | `--nv` |
| `fpocket` | bundled in `universal.sif` | — |
| `freesasa` (SASA fallback) | `universal.sif` | — |
| `propka` 3.5.1 | `universal.sif` (unused) | — |
| LigandMPNN sampler | `/net/software/lab/fused_mpnn/seth_temp/run.py` | `/net/software`, `/net/databases` |
| LigandMPNN weights | `/net/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt` | `/net/databases` |
| Side-chain packer weights | `/net/databases/mpnn/packer_weights/s_300756.pt` | `/net/databases` |
| HF model cache (ESM-C) | `/net/databases/huggingface/esmc` | `HF_HOME` |
| HF model cache (SaProt) | `/net/databases/huggingface/saprot` | `HF_HOME` |
