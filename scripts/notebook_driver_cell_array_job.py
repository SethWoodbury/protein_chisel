"""protein_chisel — JupyterHub cell for the lab-standard slurm array-job
pattern (one outer sbatch array, each task runs `bash <script>` for a
single input PDB).

Drop the body into a Jupyter cell. Assumes the surrounding notebook has
already defined: AF3_OUT_DIR, PARAMS_DIR, CHISEL_OUT_DIR, GIT_DIR,
CMDS_DIR, SUBMIT_DIR, LOGS_DIR (lab-convention path roots), plus the
`nb` helper module providing `nb.submit_array_job(...)`.

Output (per input PDB, with MINIMAL=1):
    <OUTPUT_DIR>/<pdb_stem>/
        ├── <pdb_stem>_chisel_NNN.pdb       (~50 PDBs)
        └── chiseled_design_metrics.tsv     (50 rows + RUN_META JSON header)
"""

#################################################################
### STAGE 3: PROTEIN_CHISEL SWEEP (AF3 seed PDBs -> top-K set) ###
#################################################################

### INPUTS ###
input_pdb_structures_dir = f"{AF3_OUT_DIR}filtered_i1/ref_pdbs/"
pdb_glob                 = "*.pdb"
lig_params               = f"{PARAMS_DIR}YYE.params"   # REQUIRED; ligand .params

### OUTPUTS ###
specific_chisel_output_dir = f"{CHISEL_OUT_DIR}i1__ref_pdbs/"

### CONSTANTS ###
repo_dir    = f"{GIT_DIR}protein_chisel"
script_path = f"{repo_dir}/scripts/run_chisel_design.sh"   # renamed from run_iterative_design_v2.sbatch (chisel-v1.0+)

### PROTEIN_CHISEL PARAMETERS ###

# --- Core knobs ----------------------------------------------------
target_k           = 50              # final designs shipped per PDB
n_cycles           = 3               # iterative-design cycles (3 = production)
min_hamming        = 3               # diversity floor in final top-K
omit_aa            = "CX"            # AAs MPNN never samples. "X" = unknown only;
                                     # "CX" excludes Cys+Unknown (PTE_i1 default).

# --- Sampling biases / PTM annotation ------------------------------
ptm_spec           = ""              # PTM annotation, motif-index form. Default empty.
                                     # Format: "CHAIN/EXPECTED_RESN/MOTIF_IDX:CODE"
                                     # PTE_i1 carbamylated-Lys example: "A/LYS/3:KCX"

# --- PLM model variants (stage 2) ----------------------------------
esmc_model         = "esmc_600m"     # esmc_300m | esmc_600m (default)
saprot_model       = "saprot_1.3b"   # saprot_35m | saprot_650m | saprot_1.3b (default)

# --- Throat-blocker feedback (default ON; helps constricted scaffolds) ---
throat_feedback    = True            # iterative bias against entrance-blocker AAs
throat_decay       = 0.5             # 0.5 default; lower releases pressure faster

# --- Tunnel / pocket scoring ---------------------------------------
tunnel_metrics     = True            # ray-cast + pyKVFinder pocket scoring
tunnel_hard_gate   = True            # drop "buried" designs before TOPSIS rank

# --- LigandMPNN extras ---------------------------------------------
enhance            = ""              # optional pLDDT-enhanced LigandMPNN ckpt name
                                     # (without .pth); empty = stock checkpoint

# --- Output layout -------------------------------------------------
minimal_mode       = True            # MINIMAL=1: flat run_dir, only PDBs + 1 TSV
save_intermediates = False           # keep cycle_NN/ subtrees (diagnostic only)

# --- Re-run safety / driver verbosity ------------------------------
force_overwrite    = False           # True = clobber existing *_chisel_*.pdb
verbose            = False           # --verbose passthrough to driver (DEBUG log level)

### SLURM SETTINGS ###
queue              = "cpu"      # "gpu-train" (L40) | "gpu-bf" (mixed fast GPU) | "gpu" | "cpu"
qtime              = "00:30:00" # bump to 01:30:00 for CPU mode (PLM is ~32 min on cpu=4)
cores              = 1          # CPU mode: bump to 4 for fpocket Pool() super-linear speedup
memory             = "20G"      # whole-job total; 24G for L>300 scaffolds
cmds_per_job       = 1          # 1 = one PDB per slurm task (recommended)

### COMMAND / SUBMIT FILE NAMES ###
commands_name      = "protein_chisel_i1_ref_pdbs"
job_name           = commands_name
commands_file_path = os.path.join(CMDS_DIR, commands_name)
submit_file        = f"{SUBMIT_DIR}{job_name}.sh"

### SANITY CHECKS ###
if queue not in {"cpu", "gpu", "gpu-train", "gpu-bf"}:
    raise ValueError("queue must be 'cpu', 'gpu', 'gpu-train', or 'gpu-bf'")

if not Path(repo_dir).is_dir():
    raise FileNotFoundError(f"repo_dir not found: {repo_dir}")
if not Path(script_path).is_file():
    raise FileNotFoundError(f"run_chisel_design.sh not found: {script_path}")
if not Path(input_pdb_structures_dir).is_dir():
    raise FileNotFoundError(f"input_pdb_structures_dir not found: {input_pdb_structures_dir}")
if not Path(lig_params).is_file():
    raise FileNotFoundError(f"lig_params not found: {lig_params}")

Path(specific_chisel_output_dir).mkdir(parents=True, exist_ok=True)
Path(CMDS_DIR).mkdir(parents=True, exist_ok=True)
Path(SUBMIT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

### QUICK LOGIC ###
input_pdbs = sorted(glob.glob(os.path.join(input_pdb_structures_dir, pdb_glob)))
if not input_pdbs:
    raise FileNotFoundError(f"No PDBs matched {os.path.join(input_pdb_structures_dir, pdb_glob)}")

print("repo_dir                    =", repo_dir)
print("input_pdb_structures_dir    =", input_pdb_structures_dir)
print("specific_chisel_output_dir  =", specific_chisel_output_dir)
print("\nNumber of Input PDBs =", len(input_pdbs))

### GENERATE COMMANDS ###
commands = []
skipped_existing = []

for pdb_file in input_pdbs:
    stem = Path(pdb_file).stem
    output_dir = os.path.join(specific_chisel_output_dir, stem)

    existing = list(Path(output_dir).glob("*_chisel_*.pdb")) if Path(output_dir).is_dir() else []
    if existing and not force_overwrite:
        skipped_existing.append(stem)
        continue

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # New surface env-var names (chisel-v1.0+): INPUT_PDB / OUTPUT_DIR.
    # Legacy SEED_PDB / WORK_ROOT still accepted by the sbatch.
    env_vars = {
        "INPUT_PDB":    pdb_file,
        "LIG_PARAMS":   lig_params,
        "TARGET_K":     str(target_k),
        "N_CYCLES":     str(n_cycles),
        "MIN_HAMMING":  str(min_hamming),
        "OMIT_AA":      omit_aa,
        "ESMC_MODEL":   esmc_model,
        "SAPROT_MODEL": saprot_model,
        "OUTPUT_DIR":   output_dir,
        "MINIMAL":      f"{int(minimal_mode)}",
    }

    if ptm_spec:
        env_vars["PTM"] = ptm_spec
    if save_intermediates:
        env_vars["SAVE_INTERMEDIATES"] = "1"
    if enhance:
        env_vars["ENHANCE"] = enhance

    extra_driver_flags = []
    if not throat_feedback:
        extra_driver_flags.append("--no_throat_feedback")
    if throat_feedback and throat_decay != 0.5:
        extra_driver_flags.extend(["--throat_feedback_decay", str(throat_decay)])
    if not tunnel_metrics:
        extra_driver_flags.append("--no_tunnel_metrics")
    if not tunnel_hard_gate:
        extra_driver_flags.append("--no_tunnel_hard_gate")
    if verbose:
        extra_driver_flags.append("--verbose")
    if extra_driver_flags:
        env_vars["EXTRA_DRIVER_FLAGS"] = " ".join(extra_driver_flags)

    env_prefix = " ".join(f"{key}={shlex.quote(value)}"
                          for key, value in env_vars.items())

    cmd = f"env {env_prefix} bash {shlex.quote(script_path)}"
    commands.append(cmd)

if skipped_existing:
    print(f"\nSkipped {len(skipped_existing)} PDBs with existing outputs (set force_overwrite = True to re-run).")
    for stem in skipped_existing[:5]:
        print("  -", stem)
    if len(skipped_existing) > 5:
        print("  ... and", len(skipped_existing) - 5, "more")

if not commands:
    raise RuntimeError("No commands generated. All inputs were skipped by rerun protection.")

with open(commands_file_path, "w") as f:
    f.write("\n".join(commands) + "\n")

### SETUP BATCH JOBS ###
num_jobs = (len(commands) + cmds_per_job - 1) // cmds_per_job
nb.submit_array_job(commands_file_path, qtime, str(cores), job_name, memory, submit_file, LOGS_DIR, num_jobs, cmds_per_job, queue)
