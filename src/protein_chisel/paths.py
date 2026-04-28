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
METAL3D_SIF = Path("/net/software/containers/pipelines/metal3d.sif")
MLFOLD_SIF = Path("/net/software/containers/mlfold.sif")  # used for LigandMPNN
MPNN_BINDER_DESIGN_SIF = Path("/net/software/containers/mpnn_binder_design.sif")


# ---- HuggingFace caches ----------------------------------------------------

HF_CACHE_ESMC = Path("/net/databases/huggingface/esmc")
HF_CACHE_SAPROT = Path("/net/databases/huggingface/saprot")


# ---- Source-installed model checkouts -------------------------------------

LIGAND_MPNN_DIR = Path("/net/software/lab/mpnn/proteinmpnn/ligandMPNN")
LIGAND_MPNN_RUN = LIGAND_MPNN_DIR / "protein_mpnn_run.py"
LIGAND_MPNN_WEIGHTS = Path("/net/databases/mpnn/ligand_model_weights")
# Modern fused_mpnn runner (lab default for production design):
# correctly honors --repack_everything 0, supports --fixed_residues_multi
# JSON, supports --bias_AA_per_residue JSON, supports the --enhance flag
# for plddt-enhanced checkpoints. Runs in universal.sif.
FUSED_MPNN_DIR = Path("/net/software/lab/fused_mpnn/seth_temp")
FUSED_MPNN_RUN = FUSED_MPNN_DIR / "run.py"
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
