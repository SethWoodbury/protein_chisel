# Scoring — aggregation, Pareto, diversity

`src/protein_chisel/scoring/` covers three concerns:

1. **Aggregation** — `PoseSet → per-design rollups`, metric-specific (failure → max; descriptive → mean+std+min+max; sequence-only → first; paired → apo/holo delta).
2. **Pareto** — hard constraints first, then ε-Pareto on 3-5 real objectives, then crowding distance for spacing within the front.
3. **Diversity** — Hamming distance over **mutable / pocket positions only** (not full-length, which would be dominated by surface noise).

The codex review's prescriptions on multi-objective ranking are baked into both modules — see [docs/architecture.md](architecture.md) "Multi-objective ranking".

---

## `scoring/aggregate.py`

### `aggregate_metric_table(df, policy=None, group_by="sequence_id", keep_per_source=...)`
[src/protein_chisel/scoring/aggregate.py:88](../src/protein_chisel/scoring/aggregate.py#L88)

Groups by `sequence_id`, applies a metric-specific strategy per column, emits one row per design with suffixed columns (`__mean`, `__std`, `__min`, `__max`, `__any_nonzero`, `__pass_frac`, `__q95`).

**Strategies:**

| Strategy | Output columns | Use for |
|---|---|---|
| `failure` | `<col>__max`, `<col>__any_nonzero` | BUNS, chainbreaks, broken sidechains, protease hits |
| `descriptive` | `<col>__mean`, `<col>__std`, `<col>__min`, `<col>__max` | Rg, helix_frac, hbond counts, ligand SASA |
| `first` | `<col>` (first non-null) | sequence-only metrics (ProtParam, position counts) |
| `vote` | `<col>__pass_frac` | per-conformer threshold pass rate |
| `quantile_95` | `<col>__q95` | tail-aware metric (alternative to `failure__max`) |

The `default_policy()` ([aggregate.py:50](../src/protein_chisel/scoring/aggregate.py#L50)) wires up sensible defaults for the standard metric prefixes (`buns__`, `backbone__`, `catres__`, `protease__` → `failure`; `shape__`, `ss__`, `ligand__`, `interact__` → `descriptive`; `protparam__`, `positions__` → `first`).

### Cross-source agreement (`keep_per_source`)

To preserve "designed-only" alongside "AF3-only" numbers without averaging across sources, pass e.g. `keep_per_source=("designed", "AF3_seed1", "AF3_refined")`. Each fold source emits a `src__<source>__<col>` column. Codex specifically called this out: don't average designed-model and AF3-conformer scores as exchangeable samples.

```python
agg = aggregate_metric_table(
    metric_table_df,
    keep_per_source=("designed", "AF3_seed1"),
)
# Columns produced for shape__rg:
#   shape__rg__mean, shape__rg__std, shape__rg__min, shape__rg__max
#   src__designed__shape__rg
#   src__AF3_seed1__shape__rg
```

### `paired_apo_holo_delta(df, metric_columns, ...)`
[src/protein_chisel/scoring/aggregate.py:180](../src/protein_chisel/scoring/aggregate.py#L180)

For sequences with both apo and holo conformers, emits per-design `delta__<col>` (= holo − apo). Pairs by `is_apo` (preferred) or `fold_source` labels. Returns a DataFrame keyed by `sequence_id`.

```python
delta = paired_apo_holo_delta(
    metric_table_df,
    metric_columns=["shape__rg", "interact__n_hbonds"],
    apo_label="apo", holo_label="holo",
)
# Columns: sequence_id, delta__shape__rg, delta__interact__n_hbonds
```

---

## `scoring/pareto.py`

### `apply_hard_constraints(df, constraints) → (filtered_df, drops_per_constraint)`
[src/protein_chisel/scoring/pareto.py:55](../src/protein_chisel/scoring/pareto.py#L55)

Drop rows that fail any `HardConstraint(column, min_value, max_value)`. NaN values fail by default. The drop counts use the same column name as the constraint description (or column if unset).

```python
constraints = [
    HardConstraint("protparam__pi", min_value=4.0, max_value=8.0),
    HardConstraint("buns__n_buried_unsat__max", max_value=2),
    HardConstraint("protease__n_total", max_value=0),
]
survivors, drops = apply_hard_constraints(metric_table.df, constraints)
# drops -> {"protparam__pi": 4, "buns__n_buried_unsat__max": 2, ...}
```

### `epsilon_pareto_front(df, objectives) → df_subset`
[src/protein_chisel/scoring/pareto.py:89](../src/protein_chisel/scoring/pareto.py#L89)

ε-Pareto: a point P is ε-dominated by Q iff for every objective Q is no worse (binned) and strictly better on at least one (binned). Bin size = `Objective.epsilon`. Direction `"min"` or `"max"`.

```python
objectives = [
    Objective("rosetta__total_score__mean", direction="min", epsilon=1.0),
    Objective("esmc__pseudo_perplexity", direction="min", epsilon=0.1),
    Objective("fpocket__largest_pocket_volume", direction="max", epsilon=10.0),
]
front = epsilon_pareto_front(survivors, objectives)
```

### `crowding_distance(df, objectives) → np.ndarray`
[src/protein_chisel/scoring/pareto.py:136](../src/protein_chisel/scoring/pareto.py#L136)

NSGA-II crowding distance per row. Boundary points get `+inf`. Use to pick well-spread representatives within the Pareto front.

---

## `scoring/diversity.py`

### `hamming_distance(seq_a, seq_b, mask=None)`
[src/protein_chisel/scoring/diversity.py:24](../src/protein_chisel/scoring/diversity.py#L24)

Equal-length only; `ValueError` on mismatch. With a `mask` (length-L bool list), counts only positions where `mask[i]` is True.

### `select_diverse(df, sequence_col, mask, k, min_distance, score_col, score_direction)`
[src/protein_chisel/scoring/diversity.py:51](../src/protein_chisel/scoring/diversity.py#L51)

Greedy selection: sort by `score_col` (`max` = descending, `min` = ascending), then walk and accept each candidate iff its minimum Hamming distance (under `mask`) to all already-chosen is `>= min_distance`. Stop at `k`.

### `mask_from_position_table(pt_df, mutable_classes=...)`
[src/protein_chisel/scoring/diversity.py:97](../src/protein_chisel/scoring/diversity.py#L97)

Builds the position mask from a `PositionTable`. Default `mutable_classes=("buried", "surface", "first_shell", "pocket")` — excludes `active_site` and `ligand`.

```python
from protein_chisel.scoring.diversity import (
    mask_from_position_table, select_diverse,
)

mask = mask_from_position_table(position_table.df)
chosen = select_diverse(
    candidates.df,
    sequence_col="sequence",
    mask=mask,                # ignore active_site / ligand positions
    k=50,
    min_distance=2,           # at least 2 mismatches on mutable positions
    score_col="protparam__pi",
    score_direction="max",
)
```

### `hamming_matrix(sequences, mask=None) → np.ndarray`
[src/protein_chisel/scoring/diversity.py:40](../src/protein_chisel/scoring/diversity.py#L40)

Symmetric `(n, n)` int matrix. Useful for clustering / dendrogram visualization, but `select_diverse` does not need it (greedy walk avoids the O(n²) precompute).

---

## Tests

[tests/test_scoring.py](../tests/test_scoring.py) — host-only, pure numpy/pandas. Covers all three modules with synthetic dataframes (no PyRosetta or PLM dependency).
