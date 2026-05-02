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


# ---- Sidechain packers + scorers (vendored as submodules) -----------------
#
# Each sidechain tool follows the same pattern as ESM-C / fpocket / metal3d:
#   - source vendored as a git submodule under external/<tool>/
#   - python runtime (torch, deps) lives inside an apptainer .sif
#   - heavyweight model weights live at /net/databases/lab/<tool>/, bind-
#     mounted at runtime so they aren't duplicated per-clone
#   - a thin protein_chisel wrapper at
#     src/protein_chisel/tools/sidechain_packing_and_scoring/<tool>_*.py
#     calls into the sif via apptainer exec
#
# All five neural-or-statistical sidechain tools currently ride inside
# esmc.sif (or a sibling sif when their dep stack conflicts). The
# decision per tool is documented in the corresponding wrapper module.

# --- FASPR (Huang 2020, MIT). Tiny C++ binary; no sif needed.
FASPR_SOURCE_DIR = _REPO_ROOT_FROM_PATHS / "external" / "faspr"
FASPR_CLUSTER_BIN = Path("/net/software/lab/faspr/bin/FASPR")
FASPR_DUNBRACK_BIN = Path("/net/software/lab/faspr/bin/dun2010bbdep.bin")

# --- PIPPack (Kuhlman lab 2024, MIT). torch + torch_geometric (pure-python,
# no scatter/sparse/cluster) + lightning + hydra + biopython. Lives in esmc.sif
# at /opt/pippack. Weights ship with the repo (~100MB, model_weights/).
PIPPACK_SOURCE_DIR = _REPO_ROOT_FROM_PATHS / "external" / "pippack"
PIPPACK_SIF = ESMC_SIF                # rides in esmc.sif
PIPPACK_GUEST_SOURCE_DIR = Path("/opt/pippack")  # path inside the sif
PIPPACK_WEIGHTS_DIR = Path("/net/databases/lab/pippack/model_weights")

# --- AttnPacker (McPartlon 2023, NO LICENSE — academic use only). The
# repo recommends python 3.8 + torch 1.11.0; whether it runs on esmc.sif's
# python 3.12 + torch 2.x is determined by attnpacker_install_research.
# Weights from Zenodo (record/7713779).
ATTNPACKER_SOURCE_DIR = _REPO_ROOT_FROM_PATHS / "external" / "attnpacker"
ATTNPACKER_SIF = ESMC_SIF             # provisional — confirm post-build
ATTNPACKER_GUEST_SOURCE_DIR = Path("/opt/attnpacker")
ATTNPACKER_WEIGHTS_DIR = Path("/net/databases/lab/attnpacker")

# --- FlowPacker (Lee 2025, MIT). torch + torch_geometric + e3nn + biotite.
# Includes a likelihood.py for per-residue scoring.
FLOWPACKER_SOURCE_DIR = _REPO_ROOT_FROM_PATHS / "external" / "flowpacker"
FLOWPACKER_SIF = ESMC_SIF
FLOWPACKER_GUEST_SOURCE_DIR = Path("/opt/flowpacker")
FLOWPACKER_WEIGHTS_DIR = Path("/net/databases/lab/flowpacker")

# --- OPUS-Rota5 (Xu 2024, GPL-3 — propagates downstream). README pins
# python 3.7 + TF 2.4, but the actual code uses only stable TF 2.x APIs;
# we run it on python 3.12 + TF 2.18+ inside esmc.sif with TF_USE_LEGACY_KERAS=1
# for .h5 weight loading. The vendored Rota5/mkdssp/mkdssp ELF is built
# against boost 1.53 and is unusable on Ubuntu 24.04 -- the wrapper
# bind-mounts the cluster's modern mkdssp at /net/software/utils/mkdssp
# and passes it via $OPUS_ROTA5_MKDSSP (run_opus_rota5.py is patched in
# the sif to honor that env var; see esmc.spec).
OPUS_ROTA5_SOURCE_DIR = _REPO_ROOT_FROM_PATHS / "external" / "opus_rota5"
OPUS_ROTA5_SIF = ESMC_SIF
OPUS_ROTA5_GUEST_SOURCE_DIR = Path("/opt/opus_rota5")
# Standalone-zip layout: /net/databases/lab/opus_rota5/opus_rota5/Rota5/...
OPUS_ROTA5_WEIGHTS_DIR = Path("/net/databases/lab/opus_rota5/opus_rota5")
OPUS_ROTA5_ROTAFORMER_WEIGHTS = OPUS_ROTA5_WEIGHTS_DIR / "Rota5" / "models"
OPUS_ROTA5_UNET3D_WEIGHTS = OPUS_ROTA5_WEIGHTS_DIR / "Rota5" / "unet3d" / "models"
# Cluster mkdssp 4.5.5 (modern build, dynamically linked to system libs).
MKDSSP_BIN = Path("/net/software/utils/mkdssp")

# --- MolProbity rotalyze via cctbx-base — statistical Top8000 KDE scorer
# complementary to Rosetta fa_dun. cctbx-base pip-installs into esmc.sif.
# At runtime, rotalyze needs:
#   - the CCP4 monomer library (env var CLIBD_MON)
#   - the Top8000 rotarama .pickle cache (libtbx finds via repository path
#     'chem_data/rotarama_data'). We bind-mount our chem_data over
#     /opt/esmc/chem_data inside the sif so libtbx.find_in_repositories
#     locates it.
MOLPROBITY_DIR = Path("/net/databases/lab/molprobity")
MOLPROBITY_MONOMERS_DIR = MOLPROBITY_DIR / "monomers"
MOLPROBITY_CHEM_DATA_DIR = MOLPROBITY_DIR / "chem_data"
# The guest path inside esmc.sif where chem_data must appear for cctbx
# to find it via libtbx.env.find_in_repositories('chem_data/rotarama_data').
MOLPROBITY_CHEM_DATA_GUEST = Path("/opt/esmc/chem_data")


# ---- Cluster scratch / outputs --------------------------------------------

SLURM_LOG_DIR = Path.home() / "slurm_logs"
