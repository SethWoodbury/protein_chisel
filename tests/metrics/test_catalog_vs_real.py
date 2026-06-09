"""Golden: the metric catalog must stay faithful to the pipeline's REAL output.

Cross-checks every column in a captured canonical ``chiseled_design_metrics.tsv``
(from a real default GPU run) against the catalog: every *metric* column must be
owned by exactly one :class:`MetricDescriptor`; the remainder must be explicitly
enumerated *framework/bookkeeping* columns (sampler base, selection bookkeeping,
ranking outputs, shipping columns). This catches catalog drift — if a stage starts
emitting a new column, this test fails until the catalog (and column-order golden)
is updated.

The fixture is captured host-side; the comprehensive byte-identical column-ORDER
check is a cluster full-pipeline diff (see docs/dev notes). This test is the cheap,
always-on guard.
"""
from __future__ import annotations

from pathlib import Path

from protein_chisel.metrics import all_metrics
from protein_chisel.metrics.base import STAGE_CMS, STAGE_ROSETTA

_FIXTURE = Path(__file__).parent / "fixtures" / "canonical_tsv_columns.txt"

# Columns that are NOT produced by a catalog metric — pipeline framework /
# bookkeeping. Enumerated explicitly (not a wildcard) so a genuinely new metric
# column can't hide here.
_FRAMEWORK_EXACT = frozenset({
    # sampler base schema (ligand_mpnn.CandidateSet)
    "id", "sequence", "parent_design_id", "sampler", "sampler_params_hash",
    "is_input", "header", "seq_hash", "n_dupes",
    # filter bookkeeping (the survivor/reject framework, not a metric)
    "passed_seq_filter", "fail_reasons", "passed_struct_filter", "struct_fail",
    # cycle + ranking outputs
    "cycle", "mo_topsis", "legacy_rank_score", "mo_topsis_cycle",
    # id-collision + shipping columns (added by reorganize_for_shipping/finalize)
    "source_id", "pdb_path", "seed_pdb", "seed_basename", "run_dir",
})
_FRAMEWORK_PREFIXES = (
    "mpnn_",        # per-design parsed LigandMPNN header fields
    "selection__",  # bucket/priority/gap bookkeeping for backfill+ranking
)


def _load_fixture_columns() -> list[str]:
    return [
        ln.strip()
        for ln in _FIXTURE.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def _column_owner(col: str):
    for d in all_metrics():
        if d.matches_column(col):
            return d.name
    return None


def _is_framework(col: str) -> bool:
    return col in _FRAMEWORK_EXACT or col.startswith(_FRAMEWORK_PREFIXES)


def test_fixture_present_and_nonempty():
    cols = _load_fixture_columns()
    assert len(cols) > 100, "canonical column fixture looks truncated"
    assert cols[0] == "id" and "sequence" in cols


def test_every_real_column_is_metric_or_framework():
    """No real column may be unaccounted-for: each is owned by a catalog metric
    or an explicitly-enumerated framework column."""
    cols = _load_fixture_columns()
    unaccounted = [
        c for c in cols if _column_owner(c) is None and not _is_framework(c)
    ]
    assert not unaccounted, (
        "real TSV columns not owned by any catalog metric and not enumerated as "
        f"framework: {unaccounted}")


def test_no_real_column_is_both_metric_and_framework():
    """Framework enumeration must not shadow a metric column (would mask drift)."""
    cols = _load_fixture_columns()
    both = [c for c in cols if _column_owner(c) is not None and _is_framework(c)]
    assert not both, f"columns classed as BOTH metric and framework: {both}"


def test_default_off_metric_columns_absent_from_default_run():
    """cms/rosetta are default-off, so their columns must NOT appear in the
    captured default run (sanity that the fixture is a real default run)."""
    cols = set(_load_fixture_columns())
    off_cols = [
        c for d in all_metrics() if d.stage in (STAGE_CMS, STAGE_ROSETTA)
        for c in d.columns
    ]
    present = [c for c in off_cols if c in cols]
    assert not present, f"default-off metric columns leaked into default run: {present}"


def test_every_default_on_exact_column_present_in_real_run():
    """Conversely, every default-on metric's exact columns (sans dependency-gated
    ones) should appear in the default run — proving the catalog isn't claiming
    columns the pipeline never emits."""
    cols = set(_load_fixture_columns())
    missing = []
    for d in all_metrics():
        if not d.default_on:
            continue
        for c in d.columns:
            if c not in cols:
                missing.append((d.name, c))
    # In a full default run (ligand+tunnel+pyrosetta+fpocket) there should be no
    # missing default-on columns; dependency-gated metrics (sap/ligand_int/pkvf/
    # fpocket) were all present in the captured baseline.
    assert not missing, f"default-on catalog columns absent from real run: {missing}"
