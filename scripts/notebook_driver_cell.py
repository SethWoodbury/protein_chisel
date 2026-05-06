"""protein_chisel — per-PDB sweep driver cell for jupyterhub.

Drop the body of this file into a Jupyter cell. It generates one
sbatch command per input PDB, each writing to a unique deterministic
subdirectory derived from the input PDB stem. Edit the paths in
section 1 + the ligand .params file in section 2 for your scaffold.

After all jobs land, every output dir contains EXACTLY:

    <output_base>/<seed_stem>/
        ├── <seed_stem>_chisel_001.pdb       (50 PDBs, fully hydrogenated,
        ├── <seed_stem>_chisel_005.pdb        REMARK 666 + REMARK 668)
        ├── ... (~48-50 .pdb files)
        └── chiseled_design_metrics.tsv      (50 rows × ~150 cols, with
                                              embedded RUN_META JSON
                                              as first comment line)

Wall time per PDB: ~6-8 min on gpu-train L40, ~10-15 min on gpu-bf A4000.

After the sweep, load EVERYTHING into one DataFrame:

    import sys
    sys.path.insert(0, '/home/<you>/codebase_projects/protein_chisel/scripts')
    from load_chiseled_runs import load_runs
    df = load_runs(f"{OUTPUT_BASE}/*/chiseled_design_metrics.tsv")
    # df has 50 * N_seeds rows × ~165 cols. Filter / groupby on
    # 'seed_basename', '_meta_ptm_spec', 'mo_topsis', etc.
"""

import glob
import os
import subprocess
from pathlib import Path

# ─── 1. Edit these paths for your sweep ───────────────────────────────
REPO        = "/home/aruder2/codebase_projects/protein_chisel"   # YOUR git clone
INPUT_DIR   = "/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs"
OUTPUT_BASE = "/net/scratch/aruder2/projects/PTE_i1/chisel_out/i1__ref_pdbs"

# Ligand .params file — REQUIRED. For PTE_i1 (YYE substrate):
LIG_PARAMS = "/net/scratch/aruder2/projects/PTE_i1/refs/YYE.params"

# ─── 2. Pipeline knobs (defaults are production-tested) ───────────────
PTM_SPEC           = "A/LYS/3:KCX"   # motif-index PTM; '' to disable
                                     # PTE_i1: motif idx 3 = catalytic Lys (KCX in active form)
                                     # Format: 'CHAIN/EXPECTED_RESN/MOTIF_IDX:CODE'
TARGET_K           = 50              # final designs per PDB
N_CYCLES           = 3               # iterative-design cycles
MIN_HAMMING        = 3               # diversity floor in final top-K
ESMC_MODEL         = "esmc_600m"     # esmc_300m | esmc_600m (default)
SAPROT_MODEL       = "saprot_1.3b"   # saprot_35m | saprot_650m | saprot_1.3b
THROAT_FEEDBACK    = True            # iterative bias against entrance blockers
TUNNEL_METRICS     = True            # ray-cast + pyKVFinder pocket scoring
SAVE_INTERMEDIATES = False           # keep cycle_NN/ for diagnostics

# ─── 3. Slurm settings ────────────────────────────────────────────────
PARTITION  = "gpu-train"             # 'gpu-train' (L40, faster) or 'gpu-bf' (A4000)
TIME_LIMIT = "00:30:00"              # generous; typical 6-8 min on gpu-train
MEM        = "20G"                   # whole-job; bump to 24G for L>300
CPUS       = 4

# ─── 4. Generate one sbatch command per input PDB ─────────────────────
os.makedirs(OUTPUT_BASE, exist_ok=True)
input_pdbs = sorted(glob.glob(f"{INPUT_DIR}/*.pdb"))
print(f"Found {len(input_pdbs)} input PDBs in {INPUT_DIR}")
print(f"Output base: {OUTPUT_BASE}")
print()

commands: list[tuple[str, str]] = []
for pdb_path in input_pdbs:
    stem = Path(pdb_path).stem                # e.g. ZAPP_..._FS148
    work_root = f"{OUTPUT_BASE}/{stem}"        # per-PDB output dir
    os.makedirs(work_root, exist_ok=True)

    env_vars = {
        "SEED_PDB":     pdb_path,
        "LIG_PARAMS":   LIG_PARAMS,
        "PTM":          PTM_SPEC,
        "TARGET_K":     str(TARGET_K),
        "N_CYCLES":     str(N_CYCLES),
        "MIN_HAMMING":  str(MIN_HAMMING),
        "ESMC_MODEL":   ESMC_MODEL,
        "SAPROT_MODEL": SAPROT_MODEL,
        "WORK_ROOT":    work_root,
        "MINIMAL":      "1",                    # flat layout: PDBs + 1 TSV
    }
    if SAVE_INTERMEDIATES:
        env_vars["SAVE_INTERMEDIATES"] = "1"

    extras: list[str] = []
    if not THROAT_FEEDBACK:
        extras.append("--no_throat_feedback")
    if not TUNNEL_METRICS:
        extras.append("--no_tunnel_metrics")
    if extras:
        env_vars["EXTRA_DRIVER_FLAGS"] = " ".join(extras)

    export_str = ",".join(f"{k}={v}" for k, v in env_vars.items() if v)

    cmd = (
        f"sbatch "
        f"--partition={PARTITION} "
        f"--gres=gpu:1 "
        f"--cpus-per-task={CPUS} "
        f"--mem={MEM} "
        f"--time={TIME_LIMIT} "
        f"--job-name=chisel_{stem[-12:]} "
        f"--export=ALL,{export_str} "
        f"{REPO}/scripts/run_iterative_design_v2.sbatch"
    )
    commands.append((stem, cmd))

# ─── 5a. Preview (dry run — print, don't submit) ──────────────────────
print(f"Generated {len(commands)} sbatch commands.")
print()
print("=== Preview (first 3) ===")
for stem, cmd in commands[:3]:
    print(f"# {stem}")
    print(cmd)
    print()

# ─── 5b. UNCOMMENT to submit them all ─────────────────────────────────
# print("Submitting...")
# for stem, cmd in commands:
#     out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
#     jid = out.stdout.strip().split()[-1] if out.stdout.strip() else "FAILED"
#     print(f"  {stem}: {jid}")
