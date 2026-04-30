# Filters — sequence-level cheap predicates

Sequence-only filters in `src/protein_chisel/filters/`. All run on host (no PyRosetta). They take a sequence (or list) and return either a boolean + reason, a result dataclass, or a filtered subset.

**SAP score and other structural filters live in `tools/`**, not here, despite the original sketch in [filters/__init__.py](../src/protein_chisel/filters/__init__.py) — the codex review pointed out that the boundary between "filter" and "metric" collapses, and structural filters have been kept under `tools/`.

## Table of filters

| Filter | File | Returns | Tested |
|---|---|---|---|
| `protparam_metrics` | [filters/protparam.py](../src/protein_chisel/filters/protparam.py) | `ProtParamResult` (`protparam__*`) | [tests/test_filters.py](../tests/test_filters.py) (host, biopython only) |
| `find_protease_sites` | [filters/protease_sites.py](../src/protein_chisel/filters/protease_sites.py) | `ProteaseSitesResult` (`protease__*`) | [tests/test_filters.py](../tests/test_filters.py), [test_expression_host.py](../tests/test_expression_host.py) (host) |
| `passes_length_filter` | [filters/length.py](../src/protein_chisel/filters/length.py) | `(bool, reason)` | [tests/test_filters.py](../tests/test_filters.py) (host) |
| `get_host_patterns` | [filters/expression_host.py](../src/protein_chisel/filters/expression_host.py) | `list[(name, regex)]` | [tests/test_expression_host.py](../tests/test_expression_host.py) (host) |

---

## Per-filter detail

### `protparam_metrics`
[src/protein_chisel/filters/protparam.py:65](../src/protein_chisel/filters/protparam.py#L65)

Wraps Biopython's `Bio.SeqUtils.ProtParam.ProteinAnalysis`. Strips non-canonical AAs first (so `MGXJZBAAA` → `MGAAA`); the result `length` reflects the post-strip length.

- **Outputs (`ProtParamResult.to_dict("protparam__")`)**: `length`, `molecular_weight`, `pi`, `instability_index`, `gravy` (grand average of hydropathy), `aromaticity`, `charge_at_pH7` (Biopython, includes HIS), `charge_at_pH7_no_HIS` (custom Henderson-Hasselbalch, matches legacy Rosetta NetCharge filter), `flexibility_mean`, `helix_frac_seq` / `turn_frac_seq` / `sheet_frac_seq` (sequence-derived SS propensities), `extinction_280nm_no_disulfide`, `extinction_280nm_disulfide`.
- **No-HIS variant**: pKas `K=10.5, R=12.5, D=3.65, E=4.25, N-term=9.0, C-term=2.0`. HIS excluded because pKa ~6 is too sensitive to local environment.
- **Limitations**: `flexibility_mean` is `None` for very short sequences (Biopython raises). Extinction defaults to 0.0 if Biopython errors.

### `find_protease_sites`
[src/protein_chisel/filters/protease_sites.py:64](../src/protein_chisel/filters/protease_sites.py#L64)

Default blacklist (in [protease_sites.py:17](../src/protein_chisel/filters/protease_sites.py#L17)): kex2 RR, trypsin `[KR][^P]`, ompT `[KR][KR]`, thrombin `LVPR`, furin `R..R`, caspase `D..D`. Caller can pass `extra_patterns`, set `skip_default=True`, and select a host (`"ecoli"` or `"yeast"`) via `host=`.

- **Output**: `ProteaseSitesResult.hits` is a list of `ProteaseHit(name, pattern, start, end, match)`. `to_dict("protease__")` emits `protease__n_total` and per-name counts (`protease__n_kex2_RR`, etc.).
- **Limitations**: Pure regex, no PSSM-style scoring. The yeast pattern set [filters/expression_host.py:57](../src/protein_chisel/filters/expression_host.py#L57) catches Kex2 + N-glycosylation `N[^P][ST]`; the e-coli set catches OmpT (`KK`/`KR`/`RK`/`RR`) + trypsin. PEST motifs are intentionally omitted (too noisy for a regex).

### `passes_length_filter`
[src/protein_chisel/filters/length.py:19](../src/protein_chisel/filters/length.py#L19)

`LengthFilterConfig` knobs: `min_length`, `max_length`, `forbidden_n_terminal`, `forbidden_c_terminal`, `must_start_with`, `must_end_with`. Empty sequences fail iff terminal constraints exist; otherwise pass.

- **Returns**: `(passed: bool, reason: str)`. Reason is empty string when passed; otherwise a short human-readable description of which check failed.

### `get_host_patterns`
[src/protein_chisel/filters/expression_host.py:90](../src/protein_chisel/filters/expression_host.py#L90)

Combines host-specific protease and PTM patterns with a `GENERAL_FORBIDDEN` host-agnostic list. Hosts: `"ecoli"` (or `"e_coli"`/`"e.coli"`) and `"yeast"` (or `"saccharomyces"`/`"pichia"`). Unknown host → `ValueError`.

- **`GENERAL_FORBIDDEN`** ([expression_host.py:74](../src/protein_chisel/filters/expression_host.py#L74)): internal Met start codons, polyG (≥6), polyN (≥4), polyQ (≥4), `CCC` (cys runs), `PPPP` (pro runs).
- **`E_COLI_PROTEASE_SITES`**: ompT_KK/KR/RK/RR (dibasic), trypsin `[KR][^P]`. PTM list is empty by default in E. coli (Met-aminopeptidase commented out, N-glycosylation off, phosphorylation off).
- **`YEAST_PROTEASE_SITES`**: kex2_KR / kex2_RR (Golgi protease).
- **`YEAST_PTM_SITES`**: N-glycosylation `N[^P][ST]` (yeast hyperglycosylates).
- **Notable**: `intended_kcx` (carbamylated lysine for our PTE designs) is intentionally excluded so the filter doesn't flag intentional KCX positions.
