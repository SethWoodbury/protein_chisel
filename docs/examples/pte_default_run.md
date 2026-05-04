# Standard PTE_i1 production run

The recommended invocation for a routine PTE_i1 (YYE/binuclear-Zn
phosphotriesterase) run, calibrated against the Sweep B tonight
(2026-05-04) which produced the best balance of fitness, druggability,
and active-site diversity in the empirical sweep.

## Command

Submit the reference sbatch and override only what differs from the
defaults baked into `run_iterative_design_v2.sbatch`:

```bash
SEED_PDB=/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb
LIG_PARAMS=/home/woodbuse/testing_space/scaffold_optimization/ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params/YYE.params

sbatch \
    --export=ALL,SEED_PDB=$SEED_PDB,LIG_PARAMS=$LIG_PARAMS,OMIT_AA=CX,N_CYCLES=3 \
    /home/woodbuse/codebase_projects/protein_chisel/scripts/run_iterative_design_v2.sbatch
```

If you need to pass non-default `iterative_design_v2.py` flags (the
sbatch only forwards `SEED_PDB`, `LIG_PARAMS`, `TARGET_K`, `MIN_HAMMING`,
`N_CYCLES`, `OMIT_AA`, `USE_SIDE_CHAIN_CONTEXT`, `ENHANCE`), copy the
sbatch and append to the stage-3 `apptainer exec` block. Stage-3 with
the recommended Sweep-B knobs:

```bash
apptainer exec --nv \
    --bind "$REPO:/code" --bind /net/software --bind /net/databases \
    --bind /net/scratch --bind /home/woodbuse \
    --env "PYTHONPATH=/code/src:/cifutils/src" \
    /net/software/containers/universal.sif \
    python "$REPO/scripts/iterative_design_v2.py" \
        --seed_pdb "$SEED_PDB" \
        --ligand_params "$LIG_PARAMS" \
        --plm_artifacts_dir "$PLM_DIR" \
        --position_table "$CLASSIFY_DIR/positions.tsv" \
        --out_root /net/scratch/woodbuse \
        --target_k 50 \
        --min_hamming 3 \
        --min_hamming_active 2 \
        --cycles 3 \
        --omit_AA CX \
        --plm_strength 1.25 \
        --strategy annealing \
        --consensus_threshold 0.90 \
        --consensus_strength 1.0 \
        --consensus_max_fraction 0.15
```

## Why these flags (Sweep B winners)

| flag | value | rationale |
|---|---|---|
| `--plm_strength 1.25` | sweet spot from the 0.7–1.5 sweep across rounds 1–5: best fitness recovery, tightest druggability distribution, best primary-shell diversity, no charge SD inflation |
| `--strategy annealing` | exploration in cycle 0 (fitness-heavy TOPSIS, loose light filters) → exploitation in cycle 2 (default TOPSIS weights, tight light filters, TOPSIS survivor selection) |
| `--consensus_threshold 0.90` | raised from default 0.85: requires stronger cross-survivor agreement before a (pos, AA) gets reinforced, prevents the rounds-6/7 collapse where 0.85 + a class-name bug killed pairwise Hamming |
| `--consensus_strength 1.0` | halved from default 2.0: lower per-(pos, AA) bias magnitude, consensus nudges instead of forces |
| `--consensus_max_fraction 0.15` | halved from default 0.30: only the strongest-agreement positions get reinforced, preserves positional diversity |
| `--min_hamming_active 2` | top-K must differ by ≥ 2 AAs in the primary-sphere, even when full-sequence Hamming is satisfied; useful diversity gate for fpocket variant analysis downstream |
| `--omit_AA CX` | PTE_i1 has no catalytic Cys; forbidding C avoids spurious disulfides + sample-time wasted budget |

The 6 catalytic resnos `(60, 64, 128, 131, 132, 157)` and binuclear-Zn
KCX handling (catalytic K157 capped as KCX) are hard-coded in
`iterative_design_v2.py` constants and are restored automatically by
`stage_restore_pdbs` (REMARK 666 + HETNAM + LINK + HIS tautomer
fixup). No flag needed for those.

## Expected output

```
/net/scratch/woodbuse/iterative_design_v2_PTE_i1_<TS>-<MS>-pid<PID>/
├── classify/
│   └── positions.tsv                # PositionTable
├── plm_artifacts/
│   ├── esmc_log_probs.npy           # (L, 20)
│   ├── saprot_log_probs.npy         # (L, 20)
│   ├── fusion_bias.npy              # (L, 20)  cycle-0 bias
│   ├── fusion_log_odds_{esmc,saprot}.npy
│   ├── fusion_weights.npy           # (L, 2)   per-pos β, γ
│   └── manifest.json
├── fusion_runtime/                  # the bias actually used (post --plm_strength)
│   ├── base_bias.npy
│   └── fusion_config.json
├── position_table_v2.parquet        # auto-saved sidecar if input was legacy 5-class
├── seed_tunnel_residues.tsv         # channel-lining residues w/in 6 Å of active-site
├── cycle_00/
│   ├── 00_bias/                     # bias.npy + class_balance_telemetry.json
│   ├── 01_sample/                   # candidates.tsv, pdbs_restored/, packed/
│   ├── 02_seq_filter/               # survivors_seq.tsv, rejects_seq.tsv
│   ├── 03_struct_filter/            # survivors_struct.tsv, hbond_details.tsv
│   ├── 04_fitness/scored.tsv
│   └── 05_fpocket/ranked.tsv
├── cycle_01/                        # same shape as cycle_00
├── cycle_02/                        # same shape
├── final_topk/
│   ├── all_survivors.tsv            # ~120-200 unique survivors across cycles
│   ├── topk.tsv                     # 50 diverse winners with mo_topsis
│   ├── topk.fasta
│   └── topk_pdbs/                   # 50 PDBs
└── manifest.json                    # full run config, cycle schedule, paths
```

`topk.tsv` has ~80 columns covering identity, MPNN telemetry, sequence
metrics (charge variants at design pH, instability, GRAVY, aliphatic,
boman, pi, MW, ε280), structural metrics (catalytic h-bond count,
ligand interaction strength + h-bond count, clash flags, SAP-proxy),
fpocket metrics (druggability, alpha-radius, hydrophobicity), and
ranking columns (`mo_topsis`, `legacy_rank_score`).

## Expected metric ranges (Sweep B, this session)

The reference Sweep B run sits at
`/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260504-054455-127-pid3231842/`.
After all three cycles + diverse top-K selection (50 designs), the
top-K mean values were:

| metric | Sweep B mean | column in topk.tsv |
|---|---|---|
| `fitness__logp_fused_mean` | ~ -1.91 | `fitness__logp_fused_mean` |
| pi (theoretical) | 5.12 | `pi` |
| net charge at pH 7.8 (HH) | -12.7 | `net_charge_full_HH` |
| SAP max | 1.24 | `sap__max` |
| fpocket druggability | 0.93 | `fpocket__druggability` |
| primary-sphere unique AAs | 6.2 | (computed from `min_hamming_active` gate; check `topk.tsv` `mo_topsis` ordering vs primary-sphere counts) |

Hard filters that should pass essentially everyone in `topk.tsv`:
charge in `[-18, -4]`, pi in `[5.0, 7.5]`, no severe (< 1.5 Å) clash,
druggability ≥ 0.30. If your run's top-K has < 30 designs after the
diversity gate, loosen `--min_hamming_active` (default 0) or
`--min_hamming` first; if there's an outright supply problem, drop to
`--cycles 2` and re-tune `--consensus_threshold` upward.

## Wall time

A100-class GPU node, 4 CPU, 16 GB: **~8 min wall** for all three
stages. ESM-C + SaProt precompute: ~30 s. Three cycles: ~7 min. Diverse
top-K + manifest: ~30 s. Slurm queue time on `gpu-bf` typically
dominates the experience.
