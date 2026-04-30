# Tools — exhaustive reference

Single-purpose primitives in `src/protein_chisel/tools/`. Each tool exposes a Python-callable function and a `Result` dataclass that produces a prefixed dict for `MetricTable` columns.

**Convention.** Tools never call other tools. They depend only on `io/`, `filters/`, `sampling/`, `scoring/`, and `utils/`. Pipelines compose tools.

## Table of tools

| Tool | File | sif | Returns | Tested |
|---|---|---|---|---|
| `classify_positions` | [tools/classify_positions.py](../src/protein_chisel/tools/classify_positions.py) | `pyrosetta.sif` | `PositionTable` | [tests/test_structural_tools.py](../tests/test_structural_tools.py) (cluster) |
| `backbone_sanity` | [tools/backbone_sanity.py](../src/protein_chisel/tools/backbone_sanity.py) | `pyrosetta.sif` | `BackboneSanityResult` (`backbone__*`) | [tests/test_structural_tools.py](../tests/test_structural_tools.py) (cluster) |
| `shape_metrics` | [tools/shape_metrics.py](../src/protein_chisel/tools/shape_metrics.py) | `pyrosetta.sif` | `ShapeMetricsResult` (`shape__*`) | [tests/test_structural_tools.py](../tests/test_structural_tools.py) (cluster) |
| `secondary_structure` / `ss_summary` | [tools/secondary_structure.py](../src/protein_chisel/tools/secondary_structure.py) | `pyrosetta.sif` | `SecondaryStructureResult`, `SSSummaryResult` (`ss__*`) | [tests/test_structural_tools.py](../tests/test_structural_tools.py) (cluster) |
| `ligand_environment` | [tools/ligand_environment.py](../src/protein_chisel/tools/ligand_environment.py) | `pyrosetta.sif` | `list[LigandEnvResult]` (`ligand__*`) | [tests/test_structural_tools.py](../tests/test_structural_tools.py) (cluster) |
| `chemical_interactions` | [tools/chemical_interactions.py](../src/protein_chisel/tools/chemical_interactions.py) | `pyrosetta.sif` | `InteractionsResult` (`interact__*`) | [tests/test_chemistry_tools.py](../tests/test_chemistry_tools.py) (cluster) |
| `interaction_strengths` | [tools/chemical_interactions.py:298](../src/protein_chisel/tools/chemical_interactions.py#L298) | host (numpy only — uses output of above) | `InteractionStrengthResult` (`interact_strength__*`) | [tests/test_interaction_strengths.py](../tests/test_interaction_strengths.py) (host) |
| `buns` | [tools/buns.py](../src/protein_chisel/tools/buns.py) | `pyrosetta.sif` | `BUNSResult` (`buns__*`) | [tests/test_chemistry_tools.py](../tests/test_chemistry_tools.py) (cluster) |
| `catres_quality` | [tools/catres_quality.py](../src/protein_chisel/tools/catres_quality.py) | `pyrosetta.sif` | `CatresQualityResult` (`catres__*`) | [tests/test_chemistry_tools.py](../tests/test_chemistry_tools.py) (cluster) |
| `theozyme_satisfaction` | [tools/theozyme_satisfaction.py](../src/protein_chisel/tools/theozyme_satisfaction.py) | host (numpy only) | `TheozymeSatisfactionResult` (`theozyme__*`) | [tests/test_theozyme_satisfaction.py](../tests/test_theozyme_satisfaction.py) (host) |
| `preorganization` | [tools/preorganization.py](../src/protein_chisel/tools/preorganization.py) | `pyrosetta.sif` | `PreorganizationResult` (`preorg__*`) | **untested** |
| `catalytic_pka` | [tools/catalytic_pka.py](../src/protein_chisel/tools/catalytic_pka.py) | `esmc.sif` (PROPKA) | `CatalyticPkaResult` (`pka__*`) | [tests/test_catalytic_pka.py](../tests/test_catalytic_pka.py) (cluster) |
| `fpocket_run` | [tools/fpocket_run.py](../src/protein_chisel/tools/fpocket_run.py) | host (binary required) | `FpocketResult` (`fpocket__*`) | **untested**, binary not installed |
| `metal3d_score` | [tools/metal3d_score.py](../src/protein_chisel/tools/metal3d_score.py) | `metal3d.sif` (stubbed) | `Metal3DResult` (`metal3d__*`) | **untested**, inference path stubbed |
| `contact_ms` | [tools/contact_ms.py](../src/protein_chisel/tools/contact_ms.py) | `esmc.sif` (py_contact_ms) | `ContactMSResult` (`cms__*`) | [tests/test_contact_ms.py](../tests/test_contact_ms.py) (cluster) |
| `esmc_logits` / `esmc_score` | [tools/esmc.py](../src/protein_chisel/tools/esmc.py) | `esmc.sif` (GPU recommended) | `ESMCLogitsResult`, `ESMCScoreResult` (`esmc__*`) | [tests/test_plm_tools.py](../tests/test_plm_tools.py) (cluster) |
| `saprot_logits` / `saprot_score` | [tools/saprot.py](../src/protein_chisel/tools/saprot.py) | `esmc.sif` (foldseek + transformers) | `SaProtLogitsResult`, `SaProtScoreResult` (`saprot__*`) | [tests/test_plm_tools.py](../tests/test_plm_tools.py) (cluster) |
| `sample_with_ligand_mpnn` | [tools/ligand_mpnn.py](../src/protein_chisel/tools/ligand_mpnn.py) | `universal.sif` (fused_mpnn build) | `LigandMPNNResult` w/ `CandidateSet` | [tests/test_ligand_mpnn_unit.py](../tests/test_ligand_mpnn_unit.py) (host, helpers only) |

---

## Per-tool detail

### `classify_positions`
[src/protein_chisel/tools/classify_positions.py:72](../src/protein_chisel/tools/classify_positions.py#L72)

Build a `PositionTable` for one PDB. Per-residue, derives: SASA (PyRosetta Coventry recipe), ligand distance, catalytic-residue distance, DSSP (full + reduced), phi/psi, in_pocket flag, residue class. The class is one of `active_site` (REMARK 666 / spec), `first_shell` (≤ 5 Å of any ligand atom), `pocket` (caller injects), `buried` (SASA < 20 Å²), `surface`, or `ligand`.

- **Inputs**: `pdb_path`, optional `catres` dict, optional `catres_spec` list (e.g. `["A94-96","B101"]`), optional `pocket_resnos` set, `params` (ligand `.params`), `ClassifyConfig`.
- **Outputs**: `PositionTable` with the columns in `POSITION_TABLE_REQUIRED` ([io/schemas.py:297](../src/protein_chisel/io/schemas.py#L297)) plus `atom_count`, `has_ca`.
- **Dependencies**: PyRosetta (init via `utils/pose.init_pyrosetta`), the bcov-style `getSASA` helper, `dssp` via `SecondaryStructureMetric`.
- **Limitations**: `pocket` class is only set when the caller passes `pocket_resnos`; the classifier itself never runs fpocket. No standalone CLI for fpocket integration yet.
- **CLI**: `chisel classify-positions <pdb> --out <path> --params <p>`.

### `backbone_sanity`
[src/protein_chisel/tools/backbone_sanity.py:42](../src/protein_chisel/tools/backbone_sanity.py#L42)

Three scalar metrics: `chainbreak_max` (largest sequential CA-CA gap), `chainbreak_above_4_5` (count > 4.5 Å), `rCA_nonadj_min` (smallest CA-CA distance for residues separated by ≥ 3 in sequence — clash detector), and `term_n_mindist_to_lig` / `term_c_mindist_to_lig` (terminus-to-ligand min distance, NaN if apo).

- **sif**: `pyrosetta.sif`. **Limitations**: Multi-chain inputs are handled — boundaries between chains are excluded from `chainbreak_max` ([utils/geometry.py:114](../src/protein_chisel/utils/geometry.py#L114)).

### `shape_metrics`
[src/protein_chisel/tools/shape_metrics.py:42](../src/protein_chisel/tools/shape_metrics.py#L42)

Proper sqrt-mean-square Rg + gyration-tensor descriptors: asphericity, acylindricity, relative shape anisotropy, and three principal lengths. Replaces process_diffusion3's non-standard `get_ROG`.

- **Inputs**: `pdb_path`, optional `chain_id` to restrict.
- **Outputs**: `ShapeMetricsResult` → `shape__rg`, `shape__rg_norm`, `shape__asphericity`, `shape__acylindricity`, `shape__rel_shape_anisotropy`, `shape__principal_length_{1,2,3}`, `shape__n_residues`.
- **Limitations**: CA-only (ignores sidechain atoms).

### `secondary_structure` / `ss_summary`
[src/protein_chisel/tools/secondary_structure.py:22](../src/protein_chisel/tools/secondary_structure.py#L22)

`secondary_structure` returns per-residue DSSP labels (full alphabet `H E L T S B G I -` and reduced `H | E | L`). `ss_summary` rolls them into helix/sheet/loop fractions, run lengths, and `loop_at_motif` (catalytic-residue-in-loop flag).

- **Dependencies**: PyRosetta `core.simple_metrics.metrics.SecondaryStructureMetric` (full and reduced via `set_use_dssp_reduced`, falls back to `set_dssp_reduced` for older bindings).
- **Limitations**: catalytic-residue annotation requires the caller to pass `catalytic_resnos`; no automatic REMARK 666 fallback inside this tool.

### `ligand_environment`
[src/protein_chisel/tools/ligand_environment.py:47](../src/protein_chisel/tools/ligand_environment.py#L47)

Per-ligand metrics: min backbone-ligand distance (`lig_dist`), residues within 5/8 Å, total ligand SASA, ligand SASA relative to free ligand (computes a single-residue pose for the denominator), and per-atom SASA on user-named atoms.

- **Inputs**: `pdb_path`, `target_atoms` tuple, `compute_relative` flag.
- **Outputs**: `list[LigandEnvResult]`. First ligand under prefix `ligand__`; subsequent under `ligand_<i>__`. The pipeline ([pipelines/comprehensive_metrics.py:230](../src/protein_chisel/pipelines/comprehensive_metrics.py#L230)) sets `ligand__n_ligands`.
- **Limitations**: Multi-ligand designs work but the prefix scheme stays per-ligand; aggregation downstream is the caller's responsibility.

### `chemical_interactions` + `interaction_strengths`
[src/protein_chisel/tools/chemical_interactions.py:73](../src/protein_chisel/tools/chemical_interactions.py#L73), [chemical_interactions.py:298](../src/protein_chisel/tools/chemical_interactions.py#L298)

Detects (binary, with energies/distances/angles): hbonds (PyRosetta `HBondSet`), salt bridges (charged-N to carboxylate-O within cutoff), π-π (centroid distance + plane angle, classified `stacked`/`tilted`/`t_shape`), π-cation (aromatic centroid to positive-N). Plus a separate `interaction_strengths()` function that turns binary detections into soft Gaussian-weighted strengths.

- **hbond row**: `donor_res`, `donor_name3`, `donor_h_atom`, `donor_heavy_atom`, `acceptor_res`, `acceptor_name3`, `acceptor_atom`, `energy`. Uses bcov canonical `fix_scorefxn(allow_double_bb=True)` ([utils/pose.py:104](../src/protein_chisel/utils/pose.py#L104)).
- **Binary outputs summary**: `interact__n_hbonds`, `interact__n_salt_bridges`, `interact__n_pi_pi`, `interact__n_pi_cation`, `interact__sum_hbond_energy`.
- **Strength outputs** (`interact_strength__*` prefix): per-type sums, per-type counts, per-residue strength rollup, weighted hbond energy. `INTERACTION_GEOMETRY` declares Gaussian (d0, σ) per type for salt bridges (3.0, 0.5), π-π (4.0, 0.8), π-cation (4.0, 0.8). Hbonds use a softplus on Rosetta hbond_sc energy as the strength proxy. π-π gets an angle factor `0.5 + 0.5·|cos(2θ)|` that peaks at 0° (stacked) and 90° (T-shape) and dips to 0.5 at 45° (tilted).
- **Limitations**: π-π plane-angle classification uses absolute-value cosine. No protein-ligand-specific subset; the result includes intra-protein interactions.
- **Tested**: binary detection in [tests/test_chemistry_tools.py](../tests/test_chemistry_tools.py) (cluster); strength layer in [tests/test_interaction_strengths.py](../tests/test_interaction_strengths.py) (host, no PyRosetta).

### `buns`
[src/protein_chisel/tools/buns.py:72](../src/protein_chisel/tools/buns.py#L72)

Buried unsatisfied polar atoms (donors + acceptors) in a single pass. Uses bcov 2.8 Å probe ([utils/pose.py:225](../src/protein_chisel/utils/pose.py#L225)) for "buried" definition; whitelists `(resno, atom_name)` tuples (e.g. catalytic-residue lone pairs intentionally unsat). `whitelist_from_remark_666` builds a default whitelist of all sidechain donors/acceptors of REMARK 666 catalytic residues.

- **Outputs**: `buns__n_buried_unsat`, `buns__n_buried_polar_total`, `buns__n_whitelisted`, `buns__frac_unsat`.
- **Limitations**: Uses chemical sidechain donor/acceptor tables (`SIDECHAIN_DONORS`, `SIDECHAIN_ACCEPTORS`) hard-coded for the canonical 20 AAs; non-canonical or modified residues are skipped.

### `catres_quality`
[src/protein_chisel/tools/catres_quality.py:56](../src/protein_chisel/tools/catres_quality.py#L56)

Per catalytic residue: `cart_bonded` energy (strain), `fa_dun` (rotamer prob), `bondlen_max_dev` (max sidechain heavy-atom bond-length deviation vs. an idealized A-X-A reference pose). `n_broken_sidechains` flags residues whose deviation exceeds `bondlen_threshold` (default 0.10 Å).

- **sif**: `pyrosetta.sif`. Builds two scorefunctions: `beta_nov16` and `beta_nov16_cart` (for cart_bonded).
- **Limitations**: Treats non-canonical AAs (`X`, `Z`, `B`) as "intact" (returns 0). Reference-pose builder uses single-letter AA so non-standard AA codes silently get 0.

### `theozyme_satisfaction`
[src/protein_chisel/tools/theozyme_satisfaction.py:139](../src/protein_chisel/tools/theozyme_satisfaction.py#L139)

Pure-Python (no PyRosetta) Kabsch alignment of catalytic residues vs. a theozyme reference. Computes `motif_rmsd` (Cα RMSD), `motif_heavy_rmsd`, per-atom max deviation, per-residue Cα + heavy-atom RMSD, and catalytic-atom-to-ligand-atom min distances (always available, even without a reference).

- **Inputs**: `design_pdb`, optional `theozyme_pdb` reference, optional `fixed_atoms_json` (RFdiffusion3 `{pdb: ["A92", ...]}` style), optional explicit `(chain, resno)` iterable.
- **sif**: host (numpy + io/pdb).
- **Limitations**: No iterative-relax variant (one-shot alignment only). No catalytic angle/dihedral/cstfile parsing yet — only RMSD-based metrics. The "iterative-relax variant" is on [docs/future_plans.md](future_plans.md).

### `preorganization`
[src/protein_chisel/tools/preorganization.py:49](../src/protein_chisel/tools/preorganization.py#L49)

Active-site flexibility via repack ensembles. Catalytic residues + ligand are prevent-repacked (NEVER moved); other protein residues get N independent `PackRotamers` trajectories with different seeds. Per-residue centroid variance across the ensemble = preorganization signal.

- **Outputs**: `preorg__n_ensemble`, `preorg__mean_catres_variance` (always 0 since catres are locked), `preorg__mean_near_site_variance`, `preorg__seed_pose_score`.
- **sif**: `pyrosetta.sif`.
- **Limitations**: **Not yet tested.** The "catres locked" choice is intentional (theozyme geometry preserved) but means catres-variance is always zero by construction; the useful signal is `mean_near_site_variance`. No backrub-ensemble alternative implemented.

### `catalytic_pka`
[src/protein_chisel/tools/catalytic_pka.py:57](../src/protein_chisel/tools/catalytic_pka.py#L57)

PROPKA wrapper: per-residue pKa for catalytic residues (REMARK 666 by default, or explicit `(chain, resno)` iterable). Reports `expected_pka_shift = predicted_pka − solution_pka` against canonical solution pKas (`SOLUTION_PKA` in source).

- **Outputs**: `pka__n_catres_evaluated`, plus per-residue `pka__catres__<chain>_<resno>_<name3>__pka` and `__shift`.
- **sif**: `esmc.sif` (PROPKA was added there). Uses `propka.run.single` in-memory rather than parsing the `.pka` file.
- **Limitations**: PROPKA's `MolecularContainer` returns multiple titratable groups per residue; the parser keeps only the residue-titratable group whose label starts with the residue's `name3` (or `N+`/`C-`). PROPKA itself can fail silently on unusual structures; the wrapper catches the exception and re-raises with context.

### `fpocket_run`
[src/protein_chisel/tools/fpocket_run.py:107](../src/protein_chisel/tools/fpocket_run.py#L107)

Runs the `fpocket` binary in a tempdir, parses `<pdb>_info.txt` into `Pocket` objects sorted by druggability. Per-pocket fields: volume, hydrophobicity, polarity, charge, druggability score, n_alpha_spheres, etc.

- **sif**: host or any sif with `fpocket` on `$PATH`.
- **Status**: **Wrapper functional, binary not installed** on the cluster (see [docs/dependencies.md](dependencies.md)). The user is expected to add fpocket to a future sif. Resolution order: explicit `fpocket_exe=` arg → `FPOCKET` env var → `shutil.which("fpocket")`.
- **Outputs**: `fpocket__n_pockets`, `fpocket__largest_pocket_volume`, `fpocket__most_druggable_score`, plus top-5 per-pocket fields under `fpocket__p__*`, `fpocket__p_1__*`, etc.

### `metal3d_score`
[src/protein_chisel/tools/metal3d_score.py:79](../src/protein_chisel/tools/metal3d_score.py#L79)

Currently only extracts HETATM metal atoms (Zn, Fe, Mg, Mn, Cu, Ca, Ni, Co, Cd, Hg, Mo) from the input PDB. The full Metal3D inference path is **stubbed** (see source comment, line 84-90) because the upstream API is notebook-driven.

- **Outputs**: `metal3d__n_actual_metals`, `metal3d__n_predicted_sites` (always 0), per-metal `metal3d__actual__*__max_pred_prob_4A` (empty until inference is wired).
- **sif**: `metal3d.sif` (planned; currently runs on host since only the HETATM scan is active).
- **Status**: **Not tested.** Inference path **stubbed**; see [docs/future_plans.md](future_plans.md).

### `contact_ms`
[src/protein_chisel/tools/contact_ms.py:85](../src/protein_chisel/tools/contact_ms.py#L85)

Distance-weighted contact molecular surface via [py_contact_ms](https://github.com/bcov77/py_contact_ms) (bcov77, NumPy-only). Formula: `area · exp(-0.5 d²)`. Better signal than Rosetta's `ContactMolecularSurface` filter for interface ranking.

- **Inputs**: `pdb_path`, optional `ligand_chain` / `ligand_resname` (defaults to first non-water HETATM).
- **Outputs**: `cms__total`, `cms__n_binder_atoms`, `cms__n_target_atoms`, `per_atom_cms_target` list.
- **sif**: `esmc.sif` (where `py_contact_ms` is installed). PyRosetta-free.
- **Limitations**: Atoms with no known radius (e.g. some metals) are silently filtered before the calc. Only target-side per-atom CMS is returned (binder-side is empty in the result).

### `esmc_logits` / `esmc_score`
[src/protein_chisel/tools/esmc.py:72](../src/protein_chisel/tools/esmc.py#L72), [esmc.py:168](../src/protein_chisel/tools/esmc.py#L168)

Proper masked-LM marginals for ESM-C: `esmc_logits(masked=True)` runs **L forward passes** (one per masked position) and returns `(L, 20)` log-probs over the 20 canonical AAs. `masked=False` is a single-pass diagnostic mode (leaks the true AA at each position — not a marginal). `esmc_score` returns pseudo-perplexity using the masked-LM marginal.

- **Models**: `esmc_300m` (default), `esmc_600m`. The 6B model is API-only (Forge), not downloadable.
- **sif**: `esmc.sif` (uses `from esm.models.esmc import ESMC` and the new `evolutionaryscale` API). GPU recommended for sequences > a few hundred residues.
- **Outputs**: `ESMCLogitsResult` carries `log_probs (L, 20)`, `raw_logits (L+2, vocab)` (zeros in masked mode for API stability), `aa_token_ids (20,)`. `ESMCScoreResult` to_dict → `esmc__pseudo_perplexity`, `esmc__mean_loglik`, `esmc__min_loglik`.
- **Limitations**: `aa_token_ids` is computed via the new tokenizer's `encode(aa, add_special_tokens=False)`; differs from old fair-esm `Alphabet.get_batch_converter` semantics.

### `saprot_logits` / `saprot_score`
[src/protein_chisel/tools/saprot.py:132](../src/protein_chisel/tools/saprot.py#L132), [saprot.py:269](../src/protein_chisel/tools/saprot.py#L269)

SaProt = AA × 3Di structural alphabet model. `run_foldseek_3di` extracts 3Di tokens via `foldseek structureto3didescriptor`. SA-token string interleaves AA (uppercase) + 3Di (lowercase). Logits over the 20-AA marginal are computed by summing across the 21 3Di columns per AA (so output is `(L, 20)` compatible with the fusion code).

- **Models**: `saprot_35m` (default, AF2-trained), `saprot_650m` (PDB), `saprot_1.3b` (AFDB+OMG+NCBI).
- **sif**: `esmc.sif` (foldseek bundled at `/usr/local/bin/foldseek`; transformers + SaProt weights at `/net/databases/huggingface/saprot/`).
- **Limitations**: `<mask>` token erases both AA and 3Di — the documented compromise (see source comment, line 192-198). For "p(aa | structure_full, seq_-i)" you'd want to keep 3Di_i but SaProt has no AA-only mask token.

### `sample_with_ligand_mpnn`
[src/protein_chisel/tools/ligand_mpnn.py:173](../src/protein_chisel/tools/ligand_mpnn.py#L173)

Wraps the modern lab `fused_mpnn` build at `/net/software/lab/fused_mpnn/seth_temp/run.py`. Runs in `universal.sif`. Honors `--repack_everything 0` correctly, accepts `--fixed_residues_multi` JSON of `{pdb: ["A92", ...]}`, accepts `--bias_AA_per_residue_multi` JSON of `{pdb: {<chain><resno>: {AA: bias}}}` (NOT the array-shaped per-position dict from older runners), supports `--enhance plddt_residpo_*`.

- **AA orderings**: `MPNN_AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"` (X is index 20). The PLM fusion uses the 20-AA `PLM_AA_ORDER` ([sampling/plm_fusion.py:40](../src/protein_chisel/sampling/plm_fusion.py#L40)). The wrapper asserts these match.
- **Inputs**: `pdb_path`, `chain="A"`, `fixed_resnos`, `bias_per_residue` (`(L, 20)` array in PLM_AA_ORDER), `protein_resnos` (pose resno per row of `bias_per_residue`), `n_samples`, `LigandMPNNConfig`, `out_dir`.
- **Outputs**: `LigandMPNNResult` with a `CandidateSet` whose rows include parsed header fields (`mpnn_t`, `mpnn_seq_rec`, `mpnn_overall_confidence`, `mpnn_seed`, ...) and a `sampler_params_hash` for downstream provenance.
- **Limitations**: First entry in fused_mpnn's output FASTA is the input header — the wrapper marks it `is_input=True`. `ligand_params` is accepted in the signature for API parity with older runners but is NOT passed to fused_mpnn (the modern runner reads ligand atoms from HETATMs directly).
- **Tested**: helper-level only ([tests/test_ligand_mpnn_unit.py](../tests/test_ligand_mpnn_unit.py), host) — bias-matrix conversion, header parsing, config hashing. End-to-end MPNN execution is untested in CI.
