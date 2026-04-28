"""Single source of truth for cluster paths.

Every tool/pipeline imports from here so that a path change (new sif
location, weight cache moved, etc.) only edits one file.

Add new entries when you add a new external dependency, and document the
dependency in `docs/dependencies.md` at the same time.
"""

from pathlib import Path


# ---- Apptainer images ------------------------------------------------------

# ESM-C + SaProt + foldseek + transformers (built by us)
ESMC_SIF = Path("/net/software/containers/users/woodbuse/esmc.sif")

# Lab-maintained
PYROSETTA_SIF = Path("/net/software/containers/pyrosetta.sif")
ROSETTA_SIF = Path("/net/software/containers/rosetta.sif")
UNIVERSAL_SIF = Path("/net/software/containers/universal.sif")
ESMFOLD_SIF = Path("/net/software/containers/esmfold.sif")
AF3_SIF = Path("/net/software/containers/af3.sif")  # final filter only


# ---- HuggingFace caches ----------------------------------------------------

HF_CACHE_ESMC = Path("/net/databases/huggingface/esmc")
HF_CACHE_SAPROT = Path("/net/databases/huggingface/saprot")


# ---- Source-installed model checkouts -------------------------------------

LIGAND_MPNN_DIR = Path("/net/software/lab/mpnn/proteinmpnn/ligandMPNN")
PROTEIN_MPNN_DIR = Path("/net/software/lab/mpnn/proteinmpnn")
FAMPNN_DIR = Path("/net/software/lab/fampnn")


# ---- Weight files (older, .pt format) -------------------------------------

ESM2_WEIGHTS_DIR = Path("/net/databases/esmfold")  # ESM-2 + ESMFold checkpoints


# ---- Tools (binaries) -----------------------------------------------------

# Foldseek is bundled inside ESMC_SIF; the cluster's own binary is at
# /net/software/foldseek/foldseek but is dynamically linked and may not
# work outside its host environment.
FOLDSEEK_BIN_IN_SIF = "foldseek"


# ---- Cluster scratch / outputs --------------------------------------------

SLURM_LOG_DIR = Path.home() / "slurm_logs"
