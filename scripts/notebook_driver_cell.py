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

# Ligand .params file — REQUIRED. Update this path to wherever YOUR
# ligand .params file lives. For PTE_i1 the substrate is YYE; for
# other scaffolds use whatever ligand you docked into the seed.
LIG_PARAMS = "/net/scratch/aruder2/projects/PTE_i1/refs/YYE.params"

# ─── 2. Pipeline knobs (defaults are production-tested) ───────────────
# Default: empty (no PTM annotation). Set per-scaffold:
#   PTE_i1 example: "A/LYS/3:KCX"  (catalytic Lys at motif idx 3 is carbamylated)
#   Format:         "CHAIN/EXPECTED_RESN/MOTIF_IDX:CODE"
#   Multiple:       "A/LYS/3:KCX,A/HIS/1:MSE"
# ANNOTATION ONLY: residue is kept as its unmodified form for Rosetta /
# sequence / protonation; PTM is metadata in the output PDB's REMARK 668.
PTM_SPEC           = ""              # set to e.g. "A/LYS/3:KCX" for PTE_i1
TARGET_K           = 50              # final designs per PDB
N_CYCLES           = 3               # iterative-design cycles
MIN_HAMMING        = 3               # diversity floor in final top-K
OMIT_AA            = "X"             # AAs MPNN can never sample. PTE_i1 callers
                                     # typically set "CX" (exclude Cys+Unknown).
ESMC_MODEL         = "esmc_600m"     # esmc_300m | esmc_600m (default)
SAPROT_MODEL       = "saprot_1.3b"   # saprot_35m | saprot_650m | saprot_1.3b
THROAT_FEEDBACK    = True            # iterative bias against entrance blockers
THROAT_DECAY       = 0.5             # 0.5 default; lower releases pressure faster
TUNNEL_METRICS     = True            # ray-cast + pyKVFinder pocket scoring
TUNNEL_HARD_GATE   = True            # drop 'buried' designs before TOPSIS rank
SAVE_INTERMEDIATES = False           # keep cycle_NN/ for diagnostics
VERBOSE            = False           # add --verbose to driver (DEBUG log level)
ENHANCE            = ""              # optional pLDDT-enhanced LigandMPNN ckpt
                                     # name (without .pth). Empty = standard
                                     # checkpoint. See iterative_design_v2.py
                                     # AVAILABLE_ENHANCE_CHECKPOINTS for names.

# Re-run protection: by default we refuse to overwrite an existing
# per-PDB output that already has *_chisel_*.pdb in it. Set
# FORCE_OVERWRITE=True to clobber existing sweep outputs.
FORCE_OVERWRITE    = False

# ─── 3. Slurm settings ────────────────────────────────────────────────
PARTITION  = "gpu-train"             # 'gpu-train' (L40, faster) or 'gpu-bf' (A4000)
TIME_LIMIT = "00:30:00"              # generous; typical 6-8 min on gpu-train
MEM        = "20G"                   # whole-job; bump to 24G for L>300
CPUS       = 4

# ─── 4. Sanity checks + per-PDB command generation ───────────────────
# Verify the inputs exist BEFORE generating commands so we fail loudly
# rather than silently submit zero or bogus jobs.
if not Path(INPUT_DIR).is_dir():
    raise FileNotFoundError(f"INPUT_DIR not a directory: {INPUT_DIR}")
if not Path(LIG_PARAMS).is_file():
    raise FileNotFoundError(f"LIG_PARAMS file not found: {LIG_PARAMS}")
if not Path(REPO).is_dir() or not Path(f"{REPO}/scripts/run_iterative_design_v2.sbatch").is_file():
    raise FileNotFoundError(
        f"REPO doesn't look like a protein_chisel checkout: {REPO}\n"
        f"  expected scripts/run_iterative_design_v2.sbatch"
    )

os.makedirs(OUTPUT_BASE, exist_ok=True)
input_pdbs = sorted(glob.glob(f"{INPUT_DIR}/*.pdb"))
if not input_pdbs:
    raise FileNotFoundError(f"No *.pdb files in INPUT_DIR={INPUT_DIR}")
print(f"Found {len(input_pdbs)} input PDBs in {INPUT_DIR}")
print(f"Output base: {OUTPUT_BASE}")
print()

commands: list[tuple[str, str]] = []
skipped_existing: list[str] = []
for pdb_path in input_pdbs:
    stem = Path(pdb_path).stem                # e.g. ZAPP_..._FS148
    work_root = f"{OUTPUT_BASE}/{stem}"        # per-PDB output dir

    # Re-run protection
    existing = list(Path(work_root).glob("*_chisel_*.pdb")) if Path(work_root).is_dir() else []
    if existing and not FORCE_OVERWRITE:
        skipped_existing.append(stem)
        continue
    os.makedirs(work_root, exist_ok=True)

    env_vars = {
        "SEED_PDB":     pdb_path,
        "LIG_PARAMS":   LIG_PARAMS,
        "PTM":          PTM_SPEC,
        "TARGET_K":     str(TARGET_K),
        "N_CYCLES":     str(N_CYCLES),
        "MIN_HAMMING":  str(MIN_HAMMING),
        "OMIT_AA":      OMIT_AA,
        "ESMC_MODEL":   ESMC_MODEL,
        "SAPROT_MODEL": SAPROT_MODEL,
        "WORK_ROOT":    work_root,
        "MINIMAL":      "1",                    # flat layout: PDBs + 1 TSV
    }
    if SAVE_INTERMEDIATES:
        env_vars["SAVE_INTERMEDIATES"] = "1"
    if ENHANCE:
        env_vars["ENHANCE"] = ENHANCE

    extras: list[str] = []
    if not THROAT_FEEDBACK:
        extras.append("--no_throat_feedback")
    if THROAT_FEEDBACK and THROAT_DECAY != 0.5:
        extras += ["--throat_feedback_decay", str(THROAT_DECAY)]
    if not TUNNEL_METRICS:
        extras.append("--no_tunnel_metrics")
    if not TUNNEL_HARD_GATE:
        extras.append("--no_tunnel_hard_gate")
    if VERBOSE:
        extras.append("--verbose")
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

if skipped_existing:
    print(f"Skipped {len(skipped_existing)} PDBs with existing outputs "
          f"(set FORCE_OVERWRITE=True to re-run):")
    for s in skipped_existing[:5]:
        print(f"  - {s}")
    if len(skipped_existing) > 5:
        print(f"  ... and {len(skipped_existing) - 5} more")
    print()

# ─── 5a. Preview (dry run — print, don't submit) ──────────────────────
print(f"Generated {len(commands)} sbatch commands.")
print()
print("=== Preview (first 3) ===")
for stem, cmd in commands[:3]:
    print(f"# {stem}")
    print(cmd)
    print()

# ─── 5b. UNCOMMENT to submit them all (collects failures) ────────────
# print(f"Submitting {len(commands)} jobs...")
# submitted: list[tuple[str, str]] = []   # (stem, jid)
# failures:  list[tuple[str, str]] = []   # (stem, stderr)
# for stem, cmd in commands:
#     out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
#     stdout = (out.stdout or "").strip()
#     if out.returncode == 0 and stdout:
#         jid = stdout.split()[-1]
#         submitted.append((stem, jid))
#     else:
#         err = (out.stderr or "").strip() or stdout or "no stderr"
#         failures.append((stem, err.splitlines()[-1][:200]))
# print(f"\n=== SUMMARY: {len(submitted)} submitted, {len(failures)} failed ===")
# for stem, jid in submitted[:10]:
#     print(f"  OK   {stem} -> {jid}")
# if len(submitted) > 10:
#     print(f"  ... ({len(submitted)} total submitted)")
# for stem, err in failures:
#     print(f"  FAIL {stem}: {err}")
# print(f"\nMonitor: squeue -u $USER")
