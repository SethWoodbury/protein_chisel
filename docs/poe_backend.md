# Decode-time Product-of-Experts (PoE) sampling backend

An **opt-in** alternative sampler. The default backend (`MPNN_BACKEND=bias`) is the
in-process LigandMPNN sampler with our calibrated static fusion bias — **byte-identical
to before**. `MPNN_BACKEND=poe` instead samples once via Sebastian (sebols)/Joe Mi's
`fused_mpnn_poe`, which mixes **context-aware experts** (HERMES / ESM / E1 / VESM / MSA
/ DMS) into the per-position log-probabilities at decode time, on top of *our same
calibrated bias*:

```text
log P_final = lambda_mpnn * log P_mpnn(+our bias) + sum_i lambda_i * log P_expert_i ,   lambda_mpnn = 1 - sum_i lambda_i
```

> Borrowed from Sebastian (sebols) and Joe Mi (`fused_mpnn_poe`); HERMES experts by
> Visani et al. (Nourmohammad lab, UW). The pinned container commit is recorded in
> `provenance.json`.

## Why it's a separate host stage (one-shot)

The stage-3 driver runs *inside* a container, and **nested `apptainer exec` is blocked**
on the cluster (verified empirically). So PoE cannot run per-cycle in-process like the
default sampler. Instead it runs as a **separate host stage**, one-shot, and the driver
**scores/ranks** the resulting pool (a single cycle — no per-cycle bias refinement).
The PoE output is structurally identical to our sampler's (`seqs/<stem>.fa` +
`packed/<stem>_packed_<idx>_1.pdb`), so the restore/filter/score/rank stages are unchanged.

Orchestration in `run_chisel_design.sh` when `MPNN_BACKEND=poe`:
1. **emit** — the driver (`--poe_emit_inputs`) writes this run's cycle-0
   `bias.json`/`fixed.json`/`omit.json` (the exact calibrated bias + fixed set incl.
   conserved-hbond rolls + omit) and exits. Reuses the driver's prep (no duplication).
2. **host PoE** — `apptainer exec poe_mpnn.sif python run.py --model_type ligand_mpnn
   --bias_AA_per_residue_multi bias.json …` (command built by the single source
   `mpnn_backends.build_poe_command` via `scripts/poe_emit_command.py`, exec'd at host).
3. **score-only** — the driver (`--mpnn_backend poe --poe_output_dir …`) loads that pool
   and runs filter/score/rank/finalize.

> The emit JSONs are keyed by the **literal** `--pdb_path` the host PoE receives (not
> `Path.resolve()`), so they survive symlinked `/net/scratch` → `/mnt` compute nodes.

## Using it (opt-in env)

| Env (`run_chisel_design.sh`) | Meaning | Default |
|---|---|---|
| `MPNN_BACKEND` | `bias` (default, byte-identical) or `poe` | `bias` |
| `ADDITIONAL_EXPERTS` | comma list, e.g. `esm` or `hermes,e1` (required for `poe`) | — |
| `EXPERT_LAMBDAS` | one weight per expert, each in (0,1), `sum < 1` | — |
| `POE_NUM_DESIGNS` | pool size to sample (rounds up to a multiple of 10) | `200` |
| `POE_TEMPERATURE` | PoE sampling temperature | `0.1` |

```bash
env INPUT_PDB=… LIG_PARAMS=… OUTPUT_DIR=… \
    MPNN_BACKEND=poe ADDITIONAL_EXPERTS=esm EXPERT_LAMBDAS=0.2 POE_NUM_DESIGNS=200 \
    bash run_chisel_design.sh
```

Supported experts: `hermes, dms, msa, esm, wt_esm, vesm, wt_vesm, e1, wt_e1`.
HERMES needs in-container weights (PoE-delegated). Our calibrated bias and the pipeline's
`OMIT_AA` / fixed residues / conserved-hbonds are all passed through, so PoE respects the
same constraints as the default sampler.

## Guarantees & provenance

- **Default unchanged**: with `MPNN_BACKEND=bias` (the default) nothing here runs; the
  stage-3 command and outputs are byte-identical.
- **Validation**: PoE requires experts+lambdas; lambdas must sum `< 1`; `--mpnn_backend poe`
  and `--poe_output_dir` must agree.
- **Provenance**: `provenance.json` records `mpnn_backend`, the experts + lambdas, and the
  pinned `poe_commit`; candidates are tagged `sampler="fused_mpnn_poe"`.

## Caveats

- PoE is **one-shot** (no per-cycle iterative bias refinement / throat feedback). For the
  iterative protocol, use the default `bias` backend.
- Requires GPU (the host PoE stage runs `--nv`).
- `fused_mpnn_poe` is under active development; the pinned commit is in provenance.
