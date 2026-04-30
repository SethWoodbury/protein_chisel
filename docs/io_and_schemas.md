# I/O and schemas

The artifact contracts that every stage handoff goes through.

[src/protein_chisel/io/schemas.py](../src/protein_chisel/io/schemas.py) defines four typed artifacts plus a Manifest for provenance hashing. Restart logic checks the manifest hash, not just file existence — this is the codex-review-mandated fix for silent stale-reuse bugs.

## Quick reference

| Artifact | Backing file(s) | Required ID columns | Notes |
|---|---|---|---|
| `PoseSet` | JSON sidecar (`<set>.json`) + paths to PDB files | (none — list of `PoseEntry`) | Single PDB inputs become PoseSet of size 1. |
| `PositionTable` | parquet (or TSV fallback) | `pose_id`, `resno`, `chain`, ... (15 required) | Per-residue features for one design. |
| `CandidateSet` | FASTA + parquet sidecar | `id`, `sequence` | Sequence pool with sampler metadata. |
| `MetricTable` | parquet (or TSV fallback) | `sequence_id`, `conformer_index` | One row per (sequence, conformer); columns prefixed by source. |
| `Manifest` | JSON (`_manifest.json`) | — | Provenance: input file SHA-256s, config, tool versions. |

---

## `Manifest` — provenance hashing

[schemas.py:119](../src/protein_chisel/io/schemas.py#L119)

Every stage emits a `_manifest.json` next to its outputs. Stable hash excludes `created_at` and `host` (so manifest hashes are reproducible across runs).

Fields:
- `stage`: e.g. `"comprehensive_metrics"`.
- `inputs`: `{abs_path: sha256_hex}` for every input file. Keyed by **absolute path** so two PDBs with the same basename in different dirs don't collide.
- `config`: dict of CLI args / parameters. Hashed verbatim.
- `tool_versions`: `{"pyrosetta": "...", "esm": "3.2.3", ...}`.
- `package_versions`: auto-collected snapshot of common deps via `_collect_versions()` ([schemas.py:178](../src/protein_chisel/io/schemas.py#L178)) — numpy, pandas, torch, transformers, esm, biotite, biopython, pyrosetta. Cheap, never raises.
- `created_at`, `host`, `python` — informational only, not hashed.

### Stable hash

```python
from protein_chisel.io.schemas import Manifest, manifest_matches

m = Manifest.for_stage(
    stage="classify_positions",
    input_paths=["design.pdb"],
    config={"first_shell_radius": 5.0},
    tool_versions={"pyrosetta": "2024.39"},
)
m.to_json("out/_manifest.json")

# Later: re-derive expected manifest, compare
expected = Manifest.for_stage(stage="classify_positions", input_paths=[...], config={...}, tool_versions={...})
if manifest_matches(expected, "out/_manifest.json"):
    # safe to reuse outputs
    ...
```

### Sample manifest JSON

```json
{
  "stage": "comprehensive_metrics",
  "inputs": {
    "/home/woodbuse/testing_space/align_seth_test/design.pdb": "a8f4d2b3...0e9c"
  },
  "config": {
    "tool_config": {
      "run_position_table": true,
      "run_backbone_sanity": true,
      "salt_bridge_cutoff": 4.0,
      "buns_sasa_cutoff": 1.0,
      "ligand_target_atoms": ["C1", "O1", "O2"]
    },
    "params": ["/home/woodbuse/testing_space/.../params"],
    "metadata": {
      "sequence_id": "design",
      "conformer_index": 0,
      "fold_source": "designed"
    }
  },
  "tool_versions": {"protein_chisel": "0.0.1"},
  "package_versions": {
    "numpy": "1.26.4", "pandas": "2.2.0", "pyrosetta": "2024.39"
  },
  "created_at": "2026-04-28T15:46:12+00:00",
  "host": "g3",
  "python": "3.12.7"
}
```

### Restart-skip semantics

- **`comprehensive_metrics`, `naturalness_metrics`**: full manifest match required. Any change to config / input / tool versions → re-run.
- **`sequence_design_v1`**: file-existence only (no manifest), so reusing a stage with different config silently reuses old artifacts. **Known weakness.**

---

## `PoseSet` and `PoseEntry`

[schemas.py:215](../src/protein_chisel/io/schemas.py#L215)

### `PoseEntry`

| field | type | content |
|---|---|---|
| `path` | `str` | absolute path to the PDB |
| `sequence_id` | `str` | shared across conformers of one sequence (e.g. `"design_001"`) |
| `fold_source` | `str` | `"designed"`, `"AF3_seed1"`, `"Boltz"`, `"RFdiffusion"`, `"AF3_refined"`, ... |
| `conformer_index` | `int` | 0-indexed conformer within (sequence_id, fold_source) |
| `parent_design_id` | `Optional[str]` | original scaffold design |
| `is_apo` | `bool` | True for apo (ligand-free) conformers |
| `chain_id` | `Optional[str]` | main protein chain if not "A" |
| `meta` | `dict[str, Any]` | free-form per-pose annotations |

### `PoseSet` API

- `__len__`, `__iter__`.
- `filter(**kw)` — subset by metadata equality (e.g. `.filter(fold_source="designed")`).
- `by_sequence()` — group entries by `sequence_id`.
- `to_json(path)` / `from_json(path)`.
- `from_single_pdb(path, sequence_id="design", fold_source="designed")` — convenience.

### Sample JSON

```json
{
  "name": "design_plus_af3",
  "entries": [
    {
      "path": "/home/.../align_seth_test/design.pdb",
      "sequence_id": "ub", "fold_source": "designed",
      "conformer_index": 0, "parent_design_id": null,
      "is_apo": false, "chain_id": null, "meta": {}
    },
    {
      "path": "/home/.../align_seth_test/af3_pred.pdb",
      "sequence_id": "ub", "fold_source": "AF3_seed1",
      "conformer_index": 1, "parent_design_id": null,
      "is_apo": true, "chain_id": null, "meta": {}
    },
    {
      "path": "/home/.../align_seth_test/refined.pdb",
      "sequence_id": "ub", "fold_source": "AF3_refined",
      "conformer_index": 2, "parent_design_id": null,
      "is_apo": false, "chain_id": null, "meta": {}
    }
  ]
}
```

---

## `PositionTable`

[schemas.py:317](../src/protein_chisel/io/schemas.py#L317)

Per-residue features for one design. Backed by parquet (or TSV fallback if pyarrow missing).

### Required columns ([schemas.py:297](../src/protein_chisel/io/schemas.py#L297))

`pose_id`, `resno`, `chain`, `name3`, `name1`, `is_protein`, `is_catalytic`, `class`, `sasa`, `dist_ligand`, `dist_catalytic`, `ss`, `ss_reduced`, `in_pocket`, `phi`, `psi`.

Tools may add columns (e.g. `classify_positions` adds `atom_count`, `has_ca`); only the 16 required ones are checked at construction time. `__post_init__` raises `ValueError` if any are missing.

### Sample row

```
pose_id  resno  chain  name3  name1  is_protein  is_catalytic  class       sasa   dist_ligand  dist_catalytic  ss  ss_reduced  in_pocket  phi    psi
design   188    A      HIS    H      True        True          active_site 5.2    2.8          0.0             H   H           True       -65.4  -45.2
design   42     A      MET    M      True        False         buried      3.1    18.7         15.2            E   E           False      -120.5 130.4
design   209    B      YYE    X      False       False         ligand      0.0    NaN          NaN             -   L           False      NaN    NaN
```

---

## `CandidateSet`

[schemas.py:344](../src/protein_chisel/io/schemas.py#L344)

A pool of designed sequences. Two artifacts on disk: a FASTA carrying the sequences (universal format readable by anything) and a parquet sidecar carrying metadata. Joined by candidate `id`.

### Required columns

`id`, `sequence`. Standard sampler metadata: `parent_design_id`, `sampler` (`fused_mpnn` / `iterative_constrained_local_search` / `iterative_mh`), `sampler_params_hash`. Tools may add per-sample fields (LigandMPNN adds `mpnn_t`, `mpnn_seq_rec`, `mpnn_overall_confidence`, `mpnn_seed`, etc.).

### Sample FASTA

```
>design_lmpnn_000
MEEVEEYARLVIEAIEKHRDLIREAIEEEIRIYRETGEETHAKR...
>design_lmpnn_001
MGEVEEYARLVIEAIEKHRDLIREAIEEEIRIYRETGEETHAKR...
```

### Sample parquet schema

```
id                       string   # design_lmpnn_001
sequence                 string
parent_design_id         string   # design
sampler                  string   # fused_mpnn
sampler_params_hash      string   # 12-char hash
is_input                 bool     # True for the seed entry
mpnn_t                   float
mpnn_seq_rec             float
mpnn_overall_confidence  float
plm_fusion_mean_abs_bias float
n_fixed_positions        int
fusion_class_weights     string   # "active_site=0.00|first_shell=0.05|..."
```

### `to_disk(fasta_path, meta_path)` / `from_disk(meta_path)`

Round-trip pair. `to_disk` writes one entry per row in `df.iterrows()`; `from_disk` reads only the parquet sidecar (the FASTA is for downstream tools that don't speak parquet).

---

## `MetricTable`

[schemas.py:374](../src/protein_chisel/io/schemas.py#L374)

One row per `(sequence_id, conformer_index)`. **Columns prefixed by source** so it's clear where a metric came from. Pareto / ranking operates on this.

### Required columns

`sequence_id`, `conformer_index`. Everything else is a metric column.

### Standard prefixes

| Prefix | Source |
|---|---|
| `positions__` | classify_positions counts (n_active_site, n_buried, ...) |
| `backbone__` | backbone_sanity |
| `shape__` | shape_metrics |
| `ss__` | ss_summary |
| `ligand__` | ligand_environment (first ligand); `ligand_<i>__` for additional |
| `interact__` | chemical_interactions |
| `buns__` | buns |
| `catres__` | catres_quality |
| `protparam__` | filters/protparam |
| `protease__` | filters/protease_sites |
| `cms__` | contact_ms |
| `esmc__` | esmc_score |
| `saprot__` | saprot_score |
| `fusion__` | naturalness_metrics fusion bias stats |
| `pka__` | catalytic_pka |
| `theozyme__` | theozyme_satisfaction |
| `preorg__` | preorganization |
| `fpocket__` | fpocket_run |
| `metal3d__` | metal3d_score |
| `rosetta__` | (planned, not yet implemented) |
| `src__<source>__` | per-fold-source preserved values from `aggregate_metric_table` |

### `merge(other, how="outer", on_collision="raise"|"left"|"right")`
[schemas.py:394](../src/protein_chisel/io/schemas.py#L394)

Combine two MetricTables on `(sequence_id, conformer_index)`.

`on_collision="raise"` (default) errors out when the same column exists on both sides — surfaces bugs (codex review feedback). Pass `"left"` or `"right"` to resolve, or rename one side first.

```python
metric_struct = MetricTable.from_parquet("out/comprehensive/metrics.parquet")
metric_natural = MetricTable.from_parquet("out/naturalness/metrics.parquet")
merged = metric_struct.merge(metric_natural)  # raises if any column collides
```

### Sample row (truncated)

| sequence_id | conformer_index | fold_source | positions__n_active_site | shape__rg | ss__helix_frac | interact__n_hbonds | buns__n_buried_unsat | esmc__pseudo_perplexity | fusion__mean_abs_bias |
|---|---|---|---|---|---|---|---|---|---|
| ub | 0 | designed | 6 | 14.7 | 0.43 | 142 | 3 | 4.21 | 0.034 |
| ub | 1 | AF3_seed1 | 0 | 14.5 | 0.41 | 138 | 5 | 4.18 | NaN |

---

## `io/pdb.py` — REMARK 666 + ATOM record parsing

[src/protein_chisel/io/pdb.py](../src/protein_chisel/io/pdb.py) — pure-text helpers, **no PyRosetta**. Heavy operations live in `utils/pose.py`.

### `parse_remark_666(pdb_path, key_by="resno") → dict`

Parses theozyme matcher REMARK 666 lines like:

```
REMARK 666 MATCH TEMPLATE B YYE  209 MATCH MOTIF A HIS  188  1  1
```

Returns `{int_resno: CatalyticResidue}` by default. Pass `key_by="chain_resno"` to get `{(chain, resno): CatalyticResidue}` (avoids collapsing same-resno across chains). Tolerant of malformed lines; stops scanning at first ATOM/HETATM/MODEL/TER record.

### `CatalyticResidue` ([pdb.py:46](../src/protein_chisel/io/pdb.py#L46))

| field | type | content |
|---|---|---|
| `chain` | str | motif (catalytic) chain |
| `name3` | str | motif residue name3 |
| `resno` | int | motif residue number |
| `target_chain` | str | ligand chain |
| `target_name3` | str | ligand name3 |
| `target_resno` | int | ligand residue number |
| `cst_no`, `cst_no_var` | int | constraint indices |

### `write_remark_666(src, dst, catres, drop_existing=True)`

Copies a PDB while replacing/inserting REMARK 666 lines. Useful when re-emitting derived PDBs (e.g. AF3 conformers) so downstream tools always know catalytic positions.

### `parse_catres_spec(["A94-96", "B101"]) → list[ResidueRef]`

Fallback when REMARK 666 is absent. Single residue (`A94`), inclusive range (`A94-96`), or chain-prefixed (`B101`). Adapts process_diffusion3's `parse_ref_catres` pattern.

### Other helpers

- `parse_atom_record(line) → AtomRecord | None` — tolerant fixed-column ATOM/HETATM parser.
- `summarize_pdb(pdb)` — chain composition, ligand list, water flag, elements set.
- `find_ligand(pdb, exclude=("HOH",))` — first non-water HETATM.
- `is_apo(pdb)` — True if no non-water HETATMs.
- `extract_sequence(pdb, chain=None)` — 1-letter sequence from ATOM records.
