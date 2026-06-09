# Pluggable metric registry

The **metric registry** (`protein_chisel.metrics`) is a declarative *catalog* of
every metric the `iterative_design` driver's `stage_*` chain produces. It records,
for each metric: its **role** (filter / objective / diagnostic / transform), the
**stage** that emits it, the exact **TSV columns** it contributes, its capability
**dependencies**, and its **cost** class. The catalog powers `--metrics` / `--filters`
subset selection, verbose logging, and run provenance.

> This is the metric counterpart to the [expert registry](experts.md). Registry
> design mirrors the experts package; the *ranking* objectives are reused from
> `scoring/multi_objective.py` (`DEFAULT_METRIC_SPECS`) — the catalog references
> their labels rather than re-encoding weights/targets.

For the per-column data dictionary (definitions, formulas, ranges, thresholds) see
[metrics_reference.md](metrics_reference.md). This doc is about *selecting* and
*extending* the metric set, not the meaning of each column.

## Three metric layers (don't conflate)

| Layer | Module | Granularity | Job |
|---|---|---|---|
| compute protocol | `scoring/metrics.py` | per-candidate (`Candidate → MetricResult`) | tier-scheduling + caching for `pipelines/` (not the driver) |
| ranking specs | `scoring/multi_objective.py` | column → weight/target | `mo_topsis` ranking (the single source of truth for ranking) |
| **catalog + selection** | **`metrics/`** (this) | per-driver-metric metadata | `--metrics`/`--filters`, logging, provenance |

The catalog **does not compute anything** — computation stays in the driver's
`stage_*` functions; the selection *gates* what runs (which stages execute, which
filters may drop designs, which objectives rank). So with the default selection
(`all`) the pipeline runs the exact same code and the default TSV / filter decisions
/ ranking / PDBs are **byte-identical** (verified by host-unit goldens + a cluster
full-pipeline run).

## Default behavior (unchanged)

`--metrics all --filters all` (the defaults) reproduce today's pipeline exactly,
byte-for-byte. At the default every gate is a no-op, so running with no new flags is
identical to before. Gating only changes behavior when you pass a non-default
selection.

## What gating does

- **`--metrics <subset>`**
  - drops any deselected **objective** metric from the `mo_topsis` ranking basket
    (via `multi_objective.select_specs_by_label`);
  - **skips the whole tunnel stage** if neither `tunnel` nor `pkvf` is selected
    (they share `stage_tunnel_metrics`; deselecting only `pkvf` still computes it but
    drops it from ranking);
  - the opt-in final stages `cms` / `rosetta` / `arpeggio` stay controlled by their
    own flags (`--cms_final` etc.), not by `--metrics`, to preserve their default-off
    semantics.
  - it does **not** suppress an already-computed diagnostic column whose stage still
    runs (deselecting `protparam_aux` / `dfi` has no effect today).
- **`--filters <subset>`** — a deselected **filter** stops dropping designs *and*
  stops influencing the backfill/rescue ordering (its "distance to passing" gap is
  zeroed and its rescue bucket is neutralized), so a deselected filter has zero
  effect on the final top-K.

## Selecting metrics and filters

| Where | Knob | Default |
|---|---|---|
| `run_chisel_design.sh` | `METRICS=` / `FILTERS=` env | `all` / `all` |
| `iterative_design.py` | `--metrics all\|<csv>` / `--filters all\|<csv>` | `all` / `all` |

- `--metrics` selects which metrics are computed/reported (and, where a whole
  stage becomes unused, which stage work can be skipped).
- `--filters` selects which **filter** metrics may drop designs. It must be a
  subset of `--metrics` (you can't gate on a metric you don't compute).
- The shell only emits `--metrics`/`--filters` when they differ from `all`, so the
  default command line is unchanged.

Selections are validated up front (fail-fast, so a typo can't silently no-op):
unknown names, `--filters` names that aren't filter metrics (wrong role), empty
selections, and `--filters` not ⊆ `--metrics` all raise a clear error listing the
available metrics. The default `all` never trips any of these.

Inspect the catalog:

```python
from protein_chisel.metrics import all_metrics, resolve_metrics
for d in all_metrics():
    print(d.name, d.role, d.stage, d.cost, d.objective_labels)
sel = resolve_metrics("fitness,fpocket")          # canonical-order subset
sel.stages(); sel.objective_labels()              # stages to run; ranking axes
```

## The catalogued metrics

Names are what you pass to `--metrics`/`--filters`. (See `metrics/registry.py` for
the exact column lists and `metrics_reference.md` for column meanings.)

| stage | metric | role | deps |
|---|---|---|---|
| `seq_filter` | `length`, `net_charge`, `pi`, `instability`, `gravy`, `aliphatic`, `boman`, `expression` | filter | — |
| `seq_filter` | `protparam_aux` | diagnostic | — |
| `struct_filter` | `cat_his_hbonds`, `sap`, `clash` | filter | `sap`→freesasa |
| `struct_filter` | `ligand_int`, `preorg` | objective | `ligand_int`→ligand |
| `struct_filter` | `dfi` | diagnostic | — |
| `tunnel` | `tunnel` | filter | — |
| `tunnel` | `pkvf` | objective | tunnel_sif |
| `fitness` | `fitness` | objective | — |
| `fpocket` | `fpocket` | filter (final druggability gate) + objective | fpocket |
| `cms` | `cms` | diagnostic (opt-in `--cms_final`) | contact_ms |
| `rosetta` | `rosetta` | diagnostic (opt-in `--rosetta_final`) | rosetta |
| `arpeggio` | `arpeggio` | diagnostic (opt-in) | arpeggio |

A metric whose capability is unavailable at run time is **skipped with a warning**,
never a crash (portability).

## Adding a new metric

1. Compute it in the appropriate `stage_*` (or a new stage) in
   `scripts/iterative_design.py`, emitting its column(s) into the row dict — as
   today. (The catalog describes; it does not compute.)
2. Add a `MetricDescriptor` to `_CATALOG_LIST` in `metrics/registry.py`:
   ```python
   MetricDescriptor(
       name="mymetric", role=ROLE_OBJECTIVE, stage=STAGE_STRUCT_FILTER,
       description="...", columns=("mymetric__score",),
       deps=frozenset({DEP_LIGAND}), cost=COST_CHEAP,
       objective_labels=("my_axis",),   # must exist in DEFAULT_METRIC_SPECS
   )
   ```
   - For a ranking objective, add the `MetricSpec` to
     `multi_objective.DEFAULT_METRIC_SPECS` and reference its `label` here.
   - For a filter, set `role=ROLE_FILTER`, `gates_survivors=True`, and list the
     predicate(s) in `filter_predicates`.
3. The golden test `tests/metrics/test_catalog_vs_real.py` will require the new
   column to appear in the captured canonical column fixture (regenerate it from a
   fresh baseline run) — this keeps the catalog faithful to real output. The
   `tests/metrics/test_registry.py` invariants enforce unique column ownership and
   that every `objective_label` is a real ranking label.

## Provenance

Each run records `metrics_selection`, `filters_selection`, and the resolved
`active_metrics` in `provenance.json` (default `"all"` / `"all"` / full list), so a
design is traceable to the exact metric set that produced it.

## Roadmap (staged)

Landed: the **catalog + selection + gating** (objective-subset ranking, tunnel
stage-skip, and `--filters` predicate gating), all byte-identical at the defaults.
The optional future stage converts the `stage_*` chain into a thin driver that
*iterates the selected catalog entries* (true plug-and-play scoring, incl. suppressing
deselected diagnostic columns), rolled out one stage at a time behind the same
host-unit + cluster byte-identity goldens. Until then, computation delegates to the
existing stages and the default output is byte-identical.
