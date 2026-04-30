# Testing

Three flavors of tests:

| Marker | Where it runs | How to invoke |
|---|---|---|
| (default — host) | host (login or laptop), no PyRosetta / no GPU / no PLM weights | `pytest` |
| `@pytest.mark.cluster` | Inside an apptainer image on a compute node | `pytest -m cluster` (inside sif) |
| `@pytest.mark.gpu` | Compute node with GPU and the relevant sif | `sbatch` only |

Pytest config in [pyproject.toml](../pyproject.toml#L41) has `addopts = "-m 'not cluster and not gpu'"` so the default invocation skips cluster + gpu tests.

## Coverage by file

Counts from `grep -c "^def test_" tests/test_*.py` as of 2026-04-28.

| File | Marker / sif | Tests | What it covers |
|---|---|---|---|
| [test_apptainer.py](../tests/test_apptainer.py) | mixed (6 host + 1 cluster pyrosetta.sif) | 7 | command construction, default binds, `--nv`, hf cache binding, 1 live python-inline call |
| [test_catalytic_pka.py](../tests/test_catalytic_pka.py) | cluster (esmc.sif) | 3 | PROPKA on real design.pdb, `to_dict` keys, explicit catres subset |
| [test_chemistry_tools.py](../tests/test_chemistry_tools.py) | cluster (pyrosetta.sif) | 7 | chemical_interactions hbond fields, π-π geometry, BUNS, BUNS+whitelist, catres_quality |
| [test_contact_ms.py](../tests/test_contact_ms.py) | cluster (esmc.sif) | 3 | CMS on design (>0), apo (=0), `to_dict` shape |
| [test_expression_host.py](../tests/test_expression_host.py) | host | 8 | E. coli / yeast / general patterns, `intended_kcx` exclusion |
| [test_filters.py](../tests/test_filters.py) | host (skips if no biopython) | 15 | protparam metrics, no-HIS charge, protease detection, length terminal constraints |
| [test_interaction_strengths.py](../tests/test_interaction_strengths.py) | host | 8 | Gaussian strength peak/decay, hbond energy proxy, salt bridge / π-π / π-cation distance scoring, per-residue rollup |
| [test_iterative_optimize.py](../tests/test_iterative_optimize.py) | host | 7 | constrained-LS convergence, MH energy reduction (q-correction), fixed positions, multi-chain, on-disk outputs |
| [test_ligand_mpnn_unit.py](../tests/test_ligand_mpnn_unit.py) | host | 11 | bias-matrix conversion, residue label format, fasta parsing, config hashing, default `repack_everything=0` |
| [test_pdb.py](../tests/test_pdb.py) | host | 23 | ATOM record parsing, REMARK 666 parse + round-trip + chain_resno keying, catres spec, summarize, sequence extraction |
| [test_pipeline_comprehensive.py](../tests/test_pipeline_comprehensive.py) | cluster (pyrosetta.sif) | 3 | single-PDB run, multi-pose with apo, restart-skip |
| [test_plm_fusion.py](../tests/test_plm_fusion.py) | host | 17 | log-odds, entropy match, cosine similarity, class-weight ordering, shrinkage at disagreement |
| [test_plm_tools.py](../tests/test_plm_tools.py) | cluster (esmc.sif) | 6 | ESM-C masked vs unmasked logits, pseudo-perplexity finite, SaProt logits shape, SaProt score, `to_dict` keys |
| [test_pose.py](../tests/test_pose.py) | cluster (pyrosetta.sif) | 9 | pose loading, ligand finding, scorefxn, hbonds, SASA, mutate / thread, per-atom SASA on KCX cap |
| [test_schemas.py](../tests/test_schemas.py) | host | 20 | sha256, manifest hash stability + match + missing-file, distinguishes same-basename in different dirs, PoseSet round-trip, PositionTable required-cols, CandidateSet round-trip, MetricTable merge + collision policy |
| [test_scoring.py](../tests/test_scoring.py) | host | 20 | aggregate failure/descriptive/first/per-source, paired apo/holo delta, hard constraints, ε-Pareto + epsilon binning, crowding distance, Hamming + diversity selection |
| [test_structural_tools.py](../tests/test_structural_tools.py) | cluster (pyrosetta.sif) | 10 | classify_positions design + apo + persistence, backbone_sanity, shape_metrics, secondary_structure, ss_summary, ligand_environment |
| [test_theozyme_satisfaction.py](../tests/test_theozyme_satisfaction.py) | host | 6 | no reference returns distances, design-vs-design zero RMSD, design-vs-refined small drift, apo no ligand, fixed_atoms_json, `to_dict` keys |

### Totals

- **141 host tests** (sum of all "host" rows above + the 6 host tests in test_apptainer.py).
- **30 cluster tests in `pyrosetta.sif`** (test_pose + test_structural_tools + test_chemistry_tools + test_pipeline_comprehensive + 1 cluster test in test_apptainer).
- **12 cluster tests in `esmc.sif`** (test_plm_tools + test_contact_ms + test_catalytic_pka).

## Running cluster tests

```bash
# Inside pyrosetta.sif
apptainer exec \
    --bind /home/woodbuse/codebase_projects/protein_chisel:/code \
    --env "PYTHONPATH=/code/src:/pyrosetta" \
    /net/software/containers/pyrosetta.sif \
    python -m pytest -m cluster /code/tests -v

# Inside esmc.sif
apptainer exec --nv \
    --bind /home/woodbuse/codebase_projects/protein_chisel:/code \
    --bind /net/databases/huggingface/esmc \
    --bind /net/databases/huggingface/saprot \
    --env "PYTHONPATH=/code/src" \
    --env "HF_HOME=/net/databases/huggingface/esmc" \
    --env "HF_HUB_CACHE=/net/databases/huggingface/esmc/hub" \
    /net/software/containers/users/woodbuse/esmc.sif \
    python -m pytest -m cluster /code/tests/test_plm_tools.py /code/tests/test_contact_ms.py /code/tests/test_catalytic_pka.py -v
```

Note: pyrosetta.sif sets `PYTHONNOUSERSITE=1` and ships only a minimal interpreter. `pytest` and other host-installed tools are invisible by default. The `ApptainerCall.with_pytest()` helper in [utils/apptainer.py:117](../src/protein_chisel/utils/apptainer.py#L117) prepends the host's user-site to `PYTHONPATH` so they're importable.

## Test fixtures

Real PDBs at `/home/woodbuse/testing_space/align_seth_test/`:

| File | Description |
|---|---|
| `design.pdb` | 208-residue PTE design with 6 catalytic residues (REMARK 666: HIS×4, LYS×1 (KCX), GLU×1) and YYE substrate at chain B 209 |
| `af3_pred.pdb` | AF3 prediction (apo) of the same designed sequence |
| `refined.pdb` | AF3 prediction with re-aligned ligand (carries REMARK 666) |

Ligand params at `/home/woodbuse/testing_space/scaffold_optimization/ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params/`. The PARAMS_DIR is reused across cluster tests.

## What's NOT tested

- **End-to-end `sequence_design_v1` pipeline** — only individual stages and underlying tools.
- **`naturalness_metrics` pipeline end-to-end** — the underlying ESM-C / SaProt tools are tested but the pipeline orchestration and fusion-bias artifacts are not.
- **`fpocket_run`** — binary not on cluster.
- **`metal3d_score`** — inference path stubbed.
- **`preorganization`** — function exists; no test.
- **Live LigandMPNN execution** — only helper-level tests for input/output format.
- **`utils/apptainer` live execution** — one test (`test_live_python_inline_in_pyrosetta_sif`) is `cluster`-marked.
- **No CI repository** — tests are run manually. See [docs/future_plans.md](future_plans.md).
