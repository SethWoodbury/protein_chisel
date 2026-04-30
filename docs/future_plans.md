# Future plans / what's stubbed / what to test

What still needs work, drawn from [docs/architecture.md](architecture.md), [README.md](../README.md), and the source files themselves. **Items here are explicit TODOs — nothing speculative.**

---

## Stubbed wrappers (function exists, full functionality missing)

| Item | File | Status | What's needed |
|---|---|---|---|
| Metal3D inference | [tools/metal3d_score.py](../src/protein_chisel/tools/metal3d_score.py) | HETATM scan only; the inference path is **stubbed** because the upstream API is notebook-driven (see source line 84-90) | Wrap a small driver script invoked via `metal3d_call(nv=True)`. Output: clustered probability peaks → `predicted_sites`, then per-actual-metal `max_pred_prob_within_4A`. |
| `fpocket_run` | [tools/fpocket_run.py](../src/protein_chisel/tools/fpocket_run.py) | Wrapper functional; **fpocket binary not installed** on the cluster | User to add `apt install fpocket` (or build static) into a future sif. The wrapper resolution order (`fpocket_exe=` arg → `FPOCKET` env → `which fpocket`) is already in place. |
| CAVER tunnel detection | (not started) | No source file | **TBD**: bundle CAVER into a sif, write a wrapper `tools/caver_tunnels` that returns tunnel geometry / per-tunnel bottleneck. |
| ProLIF interaction fingerprints | (not started) | No source file | Build a sif containing ProLIF + dependencies; write `tools/prolif_fingerprint`. Returns a per-design interaction fingerprint suitable for `Tanimoto`-style comparison across designs. |
| pdbe-arpeggio | (not started) | No source file | Build a sif containing pdbe-arpeggio (alternative interaction profiler); write `tools/arpeggio_interactions`. |
| `theozyme_satisfaction` iterative-relax variant | [tools/theozyme_satisfaction.py](../src/protein_chisel/tools/theozyme_satisfaction.py) | One-shot Kabsch only; no iterative-relax mode | Add a `mode="iterative_relax"` flag that runs N short PyRosetta repack/min trajectories with catalytic residues constrained, then averages the resulting motif RMSD. Requires running inside `pyrosetta.sif`. |

---

## Untested wrappers

| Item | File | Reason |
|---|---|---|
| `preorganization` | [tools/preorganization.py](../src/protein_chisel/tools/preorganization.py) | Function written; no test in [tests/](../tests/). |
| End-to-end `sequence_design_v1` | [pipelines/sequence_design_v1.py](../src/protein_chisel/pipelines/sequence_design_v1.py) | Stages tested individually; no full pipeline run in CI. |
| End-to-end `naturalness_metrics` | [pipelines/naturalness_metrics.py](../src/protein_chisel/pipelines/naturalness_metrics.py) | Underlying tools tested; pipeline orchestration + fusion-bias artifacts not covered. |
| Live `sample_with_ligand_mpnn` | [tools/ligand_mpnn.py](../src/protein_chisel/tools/ligand_mpnn.py) | Helper-level tests only ([test_ligand_mpnn_unit.py](../tests/test_ligand_mpnn_unit.py)); no real fused_mpnn execution in CI. |
| ProLIF + Arpeggio | (planned wrappers above) | **Once new sif is built, add cluster tests** for both. |

---

## Pipelines not yet implemented

| Pipeline | File | What it does | Source for the design |
|---|---|---|---|
| `comprehensive_metrics × naturalness_metrics merge` | (planned) | Single pipeline that runs both batteries (in their respective sifs) and emits a single merged `MetricTable`. The two pipelines write compatible `MetricTable` parquets that can be merged via [`MetricTable.merge`](../src/protein_chisel/io/schemas.py#L394) — but there's no pipeline that orchestrates both yet. | README's "what's stubbed but not yet wired" |
| `iterative_optimize` block moves / parallel tempering | [pipelines/iterative_optimize.py](../src/protein_chisel/pipelines/iterative_optimize.py) | Currently only single-position MH proposals. Architecture.md mentions block moves (multiple positions in one MH step) and parallel tempering (N chains at different τ with periodic configuration swaps). | architecture.md "Mode 2: Metropolis-Hastings" |
| `iterative_optimize` convergence diagnostics | same | Currently uses a naive "no-improvement window". architecture.md calls for: R-hat (multi-chain), acceptance-rate trend, integrated autocorrelation time (effective sample size), top-cluster stability across late-iteration windows. | architecture.md "Convergence diagnostics" |

---

## Sampling / scoring future work

| Item | Status | What's needed |
|---|---|---|
| PLM-bias refresh-on-top-samples | not started | architecture.md "Why a static PLM bias can drift, and three remedies": sample N candidates, recompute PLM logits on the median-by-naturalness candidate, re-bias, re-sample. Two or three rounds typically suffice. Belongs in `sampling/mpnn_with_refresh.py`. |
| PLM as reranker | not started | Let MPNN sample freely, then rerank candidates by ESM-C / SaProt scores. Belongs in `sampling/plm_reranker.py`. |
| PLM allowed-set restrictor | not started | At each non-active, non-pocket position, restrict MPNN's allowed AAs to top-k under PLM marginals. Belongs in `sampling/plm_allowed_set.py`. |
| Calibration / benchmark loop | not started | A held-out set of known-good and known-bad designs to set thresholds and detect filter-stack regressions. Belongs in `scoring/calibration.py`. The architecture and codex review both flag this as essential to keep filter thresholds out of folklore. |
| Synthetic-MSA module | not started | Per-position frequency profiles + mutual information from a sampled pool. Critically, **NOT** for natural-MSA-style DCA / EVcouplings. Belongs in `scoring/synthetic_msa.py`. |
| Hyperparam search | not started | Boltzmann grid sampler for fusion-weight / MH-τ / threshold sweeps. Belongs in `utils/hyperparam_search.py`. |

---

## Other tools sketched but not implemented

From [tools/__init__.py](../src/protein_chisel/tools/__init__.py)'s planned-tool list:

- `tools/pyrosetta_repack` — sidechain repack on fixed backbone + Rosetta score Δ.
- `tools/rosetta_ligand_ddg` — holo vs apo binding ΔΔG (whole-design, not per-mutation).
- `tools/per_residue_ddg` — per-residue ddG suite (basic / ala_scan / repack / buried_elec).
- `tools/thermompnn` — ML stability predictor (faster than Rosetta ddG).
- `tools/esm_if` — ESM-IF1 sampling (older fair-esm; ensemble diversity).
- `tools/surface_composition` — polars per SASA, hydrophobics per SASA. Belongs adjacent to BUNS.
- `tools/rosetta_metrics_xml` — direct legacy XML wrapping for the ~25-metric protocol.
- `tools/deepsp_score`, `tools/camsol_score` — solubility predictors orthogonal to SAP.
- `tools/packing_quality` — Rosetta `packstat` + buried cavity volume.
- `filters/sap_score` — spatial aggregation propensity (Lauer 2012). Structural; needs sidechain-repacked pose.

---

## Repo / tooling

| Item | Status | What's needed |
|---|---|---|
| Repo CI | none | No GitHub Actions / cluster CI runner yet. Plan: a small workflow that runs the host-only test suite on PRs, plus a periodic cluster job that runs the `cluster`-marked tests. |
| Examples | empty | [examples/](../examples/) only has a stub README. End-to-end runnable examples would help onboarding. |
| Configs | empty | [configs/](../configs/) only has a stub README. YAML configs for each pipeline are planned (`chisel <pipeline> --config configs/<x>.yaml`); the CLI doesn't yet accept `--config` paths. |
| Manifest hashing for `sequence_design_v1` | weak | The pipeline uses file-existence-only restart logic; switching to manifest hashing (like `comprehensive_metrics`) would prevent the silent stale-config-reuse bug. |

---

## Newly-built sif (planned)

A new sif containing **ProLIF + pdbe-arpeggio + fpocket** would unlock three currently-blocked tools at once. Once built:
- Wire `tools/prolif_fingerprint` and add cluster tests.
- Wire `tools/arpeggio_interactions` and add cluster tests.
- Verify `tools/fpocket_run` end-to-end with a real PDB.

The cluster's existing `metal3d.sif` could be extended with a thin Metal3D-inference driver script, after which `tools/metal3d_score` also flips from "stub" to "wrapped + tested".
