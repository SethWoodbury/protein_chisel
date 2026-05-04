# Usage: iterative PLM-fusion design (`iterative_design_v2`)

End-to-end guide for running the three-stage iterative-design pipeline that
produces a diverse top-K of catalytically-constrained designs around a
ligand-bound seed scaffold. The reference implementation is the PTE_i1
scaffold but the same pattern works for any catalytic backbone (see
`docs/examples/new_scaffold_setup.md`).

The pipeline is split across three apptainer images so each stage
imports only what it needs:

| Stage | Container | Purpose |
|---|---|---|
| 1. classify positions | `pyrosetta.sif` | PositionTable: catres, primary/secondary/distal sphere classes |
| 2. PLM precompute | `esmc.sif` (GPU) | ESM-C + SaProt masked-LM marginals -> calibrated cycle-0 fusion bias |
| 3. iterative driver | `universal.sif` (GPU or CPU) | LigandMPNN sampling, filter cascade, fpocket scoring, TOPSIS rank, diverse top-K |

All three are wired together by `scripts/run_iterative_design_v2.sbatch`.

## Running on Slurm (recommended)

The reference sbatch produces a fresh run dir under `/net/scratch/woodbuse/`
and runs all three stages back-to-back on a GPU node:

```bash
# default PTE_i1 run: target_k=50, min_hamming=3, 3 cycles, no Cys
sbatch /home/woodbuse/codebase_projects/protein_chisel/scripts/run_iterative_design_v2.sbatch
```

The script holds the full apptainer pattern. Reproduced verbatim, the
key sbatch directives + the three apptainer invocations are:

```bash
#SBATCH --job-name=PTE_i1_iter_v2
#SBATCH --partition=gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/woodbuse/slurm-iter-v2-%j.out

REPO=/home/woodbuse/codebase_projects/protein_chisel
SEED_PDB=/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/<...>.pdb
LIG_PARAMS=/home/woodbuse/testing_space/scaffold_optimization/<...>/params/YYE.params
WORK_DIR=/net/scratch/woodbuse/iterative_design_v2_PTE_i1_$(date +%Y%m%d-%H%M%S)
CLASSIFY_DIR=$WORK_DIR/classify
PLM_DIR=$WORK_DIR/plm_artifacts

# --- STAGE 1: classify positions (CPU-only PyRosetta, but ride the GPU node)
apptainer exec \
    --bind "$REPO:/code" --bind /net/software --bind /net/scratch --bind /home/woodbuse \
    --env "PYTHONPATH=/code/src:/pyrosetta" \
    /net/software/containers/pyrosetta.sif \
    python "$REPO/scripts/classify_positions_pte_i1.py" \
        --seed_pdb "$SEED_PDB" --ligand_params "$LIG_PARAMS" --out_dir "$CLASSIFY_DIR"

# --- STAGE 2: ESM-C + SaProt marginals + calibrated fusion (GPU)
apptainer exec --nv \
    --bind "$REPO:/code" --bind /net/software --bind /net/databases \
    --bind /net/scratch --bind /home/woodbuse \
    --bind /net/databases/huggingface/esmc --bind /net/databases/huggingface/saprot \
    --env "PYTHONPATH=/code/src" \
    --env "HF_HOME=/net/databases/huggingface/esmc" \
    --env "HF_HUB_CACHE=/net/databases/huggingface/esmc/hub" \
    /net/software/containers/users/woodbuse/esmc.sif \
    python "$REPO/scripts/precompute_plm_artifacts.py" \
        --seed_pdb "$SEED_PDB" \
        --position_table "$CLASSIFY_DIR/positions.tsv" \
        --out_dir "$PLM_DIR"

# --- STAGE 3: iterative driver (LigandMPNN + filters + scoring + TOPSIS)
apptainer exec --nv \
    --bind "$REPO:/code" --bind /net/software --bind /net/databases \
    --bind /net/scratch --bind /home/woodbuse \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    /net/software/containers/universal.sif \
    python "$REPO/scripts/iterative_design_v2.py" \
        --seed_pdb "$SEED_PDB" --ligand_params "$LIG_PARAMS" \
        --plm_artifacts_dir "$PLM_DIR" \
        --position_table "$CLASSIFY_DIR/positions.tsv" \
        --out_root /net/scratch/woodbuse \
        --target_k 50 --min_hamming 3 --cycles 3 --omit_AA CX
```

Override the env knobs at submission time:

```bash
sbatch \
  --export=ALL,SEED_PDB=/path/to/other.pdb,LIG_PARAMS=/path/to/other.params,N_CYCLES=3,OMIT_AA=CX \
  scripts/run_iterative_design_v2.sbatch
```

A typical 3-cycle run on an A100-class GPU lands in **~8 min wall** and
writes ~2 GB of artifacts to `/net/scratch/woodbuse/iterative_design_v2_<scaffold>_<ts>/`.

## Running interactively in Apptainer (debug)

When a stage misbehaves, run the same containers interactively. Note
that each stage has its own bind set + PYTHONPATH; copy them from
`run_iterative_design_v2.sbatch` to avoid drift:

```bash
# stage 3 shell, GPU node:
salloc --partition=gpu-bf --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00

apptainer shell --nv \
    --bind /home/woodbuse/codebase_projects/protein_chisel:/code \
    --bind /net/software --bind /net/databases \
    --bind /net/scratch --bind /home/woodbuse \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    /net/software/containers/universal.sif

# inside the container:
python /code/scripts/iterative_design_v2.py --help
python /code/scripts/iterative_design_v2.py \
    --seed_pdb $SEED_PDB --ligand_params $LIG_PARAMS \
    --plm_artifacts_dir <prev_run>/plm_artifacts \
    --position_table   <prev_run>/classify/positions.tsv \
    --out_root /net/scratch/woodbuse \
    --cycles 1 --target_k 10
```

`--cycles 1` is a valuable short-test mode: ~3 min on GPU, exercises
every stage exactly once.

## Running on CPU (no GPU)

CPU runs work end-to-end and are validated empirically (this session,
2026-05-04) at **~3.6× slower than GPU per cycle**, not 30× as
sometimes feared. For low-priority "thousands of jobs" workflows, CPU
is fully viable and skips the GPU queue.

The only difference vs the GPU sbatch is dropping `--nv` from the
`apptainer exec` calls and switching the slurm partition to a CPU one:

```bash
#SBATCH --job-name=PTE_i1_iter_v2_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/woodbuse/slurm-iter-v2-cpu-%j.out

# stage 1 — same as GPU pattern, no --nv (PyRosetta is CPU anyway)
apptainer exec \
    --bind "$REPO:/code" --bind /net/software --bind /net/scratch --bind /home/woodbuse \
    --env "PYTHONPATH=/code/src:/pyrosetta" \
    /net/software/containers/pyrosetta.sif \
    python "$REPO/scripts/classify_positions_pte_i1.py" ...

# stage 2 — drop --nv. PLMs run on CPU via torch fallback.
apptainer exec \
    --bind "$REPO:/code" --bind /net/software --bind /net/databases \
    --bind /net/scratch --bind /home/woodbuse \
    --bind /net/databases/huggingface/esmc --bind /net/databases/huggingface/saprot \
    --env "PYTHONPATH=/code/src" \
    --env "HF_HOME=/net/databases/huggingface/esmc" \
    --env "HF_HUB_CACHE=/net/databases/huggingface/esmc/hub" \
    /net/software/containers/users/woodbuse/esmc.sif \
    python "$REPO/scripts/precompute_plm_artifacts.py" ...

# stage 3 — drop --nv. fused_mpnn falls back to CPU.
apptainer exec \
    --bind "$REPO:/code" --bind /net/software --bind /net/databases \
    --bind /net/scratch --bind /home/woodbuse \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    /net/software/containers/universal.sif \
    python "$REPO/scripts/iterative_design_v2.py" ...
```

Empirical timings (1 cycle / 500 samples / full pipeline, 2026-05-04):

| Hardware | Wall | per-cycle | vs GPU |
|---|---|---|---|
| 1× A100, 4 CPU | 8.2 min (3 cycles) | ~2.7 min | 1× |
| 8 CPU only | 9.8 min (1 cycle) | ~9.8 min | ~3.6× |

`fpocket` is the dominant cost on GPU (~37% of cycle); `fused_mpnn` is
the dominant cost on CPU (~80%). Both binaries run identically with
or without `--nv`.

## Stage-by-stage walkthrough

For run dir `$RD = /net/scratch/woodbuse/iterative_design_v2_PTE_i1_<ts>/`:

### Stage 1 — `classify_positions_pte_i1.py`

Reads `seed.pdb` + `ligand.params`, runs PyRosetta `classify_positions`,
writes `$RD/classify/positions.tsv` (a `PositionTable` parquet — column
`class` ∈ {`primary_sphere`, `secondary_sphere`, `distal_buried`,
`exposed`, `surface`, …} for protein rows). Idempotent: cache hit if
`positions.tsv` already exists. ~5–15 s.

### Stage 2 — `precompute_plm_artifacts.py`

Reads the seed sequence + PositionTable, runs ESM-C (`esmc_300m`) and
SaProt (`saprot_35m`) masked-LM in parallel (one forward pass per
position), then fuses into a calibrated bias matrix via
`protein_chisel.sampling.plm_fusion.fuse_plm_logits`. Writes to
`$RD/plm_artifacts/`:

- `esmc_log_probs.npy`, `saprot_log_probs.npy` — `(L, 20)` raw marginals
- `fusion_bias.npy` — `(L, 20)` cycle-0 bias in `AA_ORDER` (`ACDEFGHIKLMNPQRSTVWY`)
- `fusion_log_odds_{esmc,saprot}.npy` — calibrated per-PLM contributions
- `fusion_weights.npy` — `(L, 2)` per-position `(β_esmc, γ_saprot)`
- `manifest.json` — sha-16 of seed PDB, model names, fusion config

Idempotent (cache hit if every artifact exists). GPU: ~30–60 s. CPU: ~2 min.

### Stage 3 — `iterative_design_v2.py`

Three cycles by default. Each cycle (`$RD/cycle_NN/`) runs the same
five sub-stages and writes them to numbered subdirs:

| Subdir | Produced by | Notable artifacts |
|---|---|---|
| `00_bias/` | base PLM-fusion + class-balance + consensus reinforcement | `bias.npy`, `class_balance_telemetry.json`, `telemetry.json` |
| `01_sample/` | LigandMPNN forward sampling | `candidates.fasta`, `candidates.tsv`, `bias_per_residue.json`, `omit_AA_per_residue.json`, `pdbs_restored/` |
| `02_seq_filter/` | sequence-only filter cascade (charge band, pi band, instability, aliphatic, GRAVY, boman, expression rules) | `survivors_seq.tsv`, `rejects_seq.tsv` |
| `03_struct_filter/` | catalytic h-bond detection + clash check + SAP-proxy | `survivors_struct.tsv`, `rejects_struct.tsv`, `hbond_details.tsv` |
| `04_fitness/` | per-sequence fitness from cached PLM marginals | `scored.tsv` |
| `05_fpocket/` | constrained fpocket scoring at the active site | `ranked.tsv`, `per_design_fpocket/` |

After all cycles, `$RD/final_topk/` holds:

- `all_survivors.tsv` — concat + dedup across cycles, with `mo_topsis` column
- `topk.tsv`, `topk.fasta`, `topk_pdbs/` — the diverse top-K (greedy
  Hamming on full + active-site sequence)

`$RD/manifest.json` records every cycle config + all input paths. Run
dirs are tagged with millisecond + PID timestamps so concurrent
parallel sweeps never collide.

## Cycle config — exploration → exploitation schedule

The default 3-cycle schedule is a temperature ramp + a sample-budget
ramp + a filter-tightening ramp + (optionally) a TOPSIS-vs-fitness
ramp. Defined in `default_cycles()` in `scripts/iterative_design_v2.py`.

### Per-cycle defaults (`--strategy constant`, the legacy default)

| | n_samples | mpnn T | filter set | survivors selected by |
|---|---|---|---|---|
| cycle 0 | 500 | 0.20 | global defaults | fitness |
| cycle 1 | 400 | 0.18 | global defaults | fitness |
| cycle 2 | 300 | 0.15 | global defaults | fitness |

Same hard filters across all cycles (charge `[-18, -4]`, pi
`[5.0, 7.5]`, fpocket druggability ≥ 0.30, severe-clash 1.5 Å). Survivors
of cycle k are pooled and re-fed into cycle k+1's bias as a consensus
augmentation: AAs that appear in ≥ `consensus_threshold` (default 0.85)
of survivors at a non-fixed position get +`consensus_strength` (default
2.0) nats added to the bias for that (pos, AA), capped at
`consensus_max_fraction` (default 0.30) of L.

### Per-cycle defaults (`--strategy annealing`, recommended)

| | n_samples | mpnn T | instability_max | gravy band | aliphatic_min | boman_max | TOPSIS weights | survivors by |
|---|---|---|---|---|---|---|---|---|
| cycle 0 (explore) | 500 | 0.20 | 80 | [-1.0, 0.4] | 30 | 5.5 | fitness=3.0, others=0.1 | fitness |
| cycle 1 | 400 | 0.18 | 70 | [-0.9, 0.35] | 35 | 5.0 | defaults | TOPSIS |
| cycle 2 (exploit) | 300 | 0.15 | 60 | [-0.8, 0.3]  | 40 | 4.5 | defaults | TOPSIS |

Hard filters (charge, pi, severe-clash, druggability) **stay constant
across cycles in both strategies** — only the LIGHT filters and TOPSIS
weights anneal. This is intentional per Aaron's "the current range
should be the final one" rule.

Override the schedule from CLI:

- `--cycles 1` — short-test mode, single cycle 0
- `--cycles 2` — drop the exploit cycle
- `--strategy annealing` — switch to the schedule above
- `--consensus_threshold 0.90 --consensus_strength 1.0 --consensus_max_fraction 0.15` — tighter consensus, preserves diversity (recommended; see `docs/examples/pte_default_run.md`)
- `--rank_weights "fitness=2,druggability=1.5"` — TOPSIS reweighting
- `--rank_targets "charge=-12,aliphatic=80"` — TOPSIS retargeting

See `python scripts/iterative_design_v2.py --help` for the full CLI.
