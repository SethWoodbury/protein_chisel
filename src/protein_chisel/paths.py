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

# fpocket lives in two places, both built from the vendored
# external/fpocket submodule:
# - inside esmc.sif at /usr/local/bin/fpocket (auto-discovered via
#   shutil.which when running inside the sif)
# - cluster-wide at /net/software/lab/fpocket/bin/fpocket so any user
#   running outside the sif can call it. tools/fpocket_run resolves
#   in order: --fpocket_exe, $FPOCKET, shutil.which, this fallback.
FPOCKET_CLUSTER_BIN = Path("/net/software/lab/fpocket/bin/fpocket")

# Metal3D — vendored source under external/metal-site-prediction (git
# submodule). The actual runtime (CNN weights + torch + moleculekit)
# lives inside metal3d.sif at /opt/metal-site-prediction; the local
# copy is for introspection, patching, and reproducibility (pinned commit).
# Aaron's scripts/run_metal3d.py auto-launches metal3d.sif when invoked
# from outside any container.
# __file__ is src/protein_chisel/paths.py. parents[2] is the repo root
# (above src/). NB: parents is 0-indexed and includes the file's own dir.
_REPO_ROOT_FROM_PATHS = Path(__file__).resolve().parents[2]
METAL3D_SOURCE_DIR = _REPO_ROOT_FROM_PATHS / "external" / "metal-site-prediction"
# Resolution for the runner script:
#   1. local checkout at scripts/run_metal3d.py (preferred when developing
#      from a clone of protein_chisel)
#   2. cluster-wide install at /net/software/lab/metal3d/run_metal3d.py
#      (so users who don't clone the repo can still call it)
METAL3D_RUNNER_SCRIPT_LOCAL = _REPO_ROOT_FROM_PATHS / "scripts" / "run_metal3d.py"
METAL3D_RUNNER_SCRIPT_CLUSTER = Path("/net/software/lab/metal3d/run_metal3d.py")
METAL3D_RUNNER_SCRIPT = (
    METAL3D_RUNNER_SCRIPT_LOCAL
    if METAL3D_RUNNER_SCRIPT_LOCAL.is_file()
    else METAL3D_RUNNER_SCRIPT_CLUSTER
)


# ---- Cluster scratch / outputs --------------------------------------------

SLURM_LOG_DIR = Path.home() / "slurm_logs"
