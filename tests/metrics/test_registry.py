"""Tests for the driver-stage metric catalog + selection (pure metadata)."""
from __future__ import annotations

import pytest

from protein_chisel.metrics import (
    DEP_FPOCKET, DEP_FREESASA, DEP_LIGAND, DEP_TUNNEL_SIF,
    MetricDescriptor, ROLE_DIAGNOSTIC, ROLE_FILTER, ROLE_OBJECTIVE,
    all_metrics, available_metrics, get_metric, resolve_metrics,
)
from protein_chisel.metrics.base import (
    COST_CHEAP, STAGE_FPOCKET, STAGE_SEQ_FILTER,
)
from protein_chisel.scoring.multi_objective import DEFAULT_METRIC_SPECS


# ---- descriptor validation ------------------------------------------------
def test_descriptor_rejects_bad_role():
    with pytest.raises(ValueError):
        MetricDescriptor(name="x", role="bogus", stage="s", description="d")


def test_descriptor_rejects_unknown_dep():
    with pytest.raises(ValueError):
        MetricDescriptor(name="x", role=ROLE_DIAGNOSTIC, stage="s",
                         description="d", deps=frozenset({"warp_drive"}))


def test_descriptor_rejects_bad_cost():
    with pytest.raises(ValueError):
        MetricDescriptor(name="x", role=ROLE_DIAGNOSTIC, stage="s",
                         description="d", cost="instant")


def test_filter_must_gate_survivors():
    # role=filter with gates_survivors=False is a contradiction.
    with pytest.raises(ValueError):
        MetricDescriptor(name="x", role=ROLE_FILTER, stage="s", description="d")


def test_matches_column_exact_and_prefix():
    d = MetricDescriptor(name="dfi", role=ROLE_DIAGNOSTIC, stage="s",
                         description="d", columns=("dfi__mean",),
                         column_prefixes=("dfi__mean__",))
    assert d.matches_column("dfi__mean")
    assert d.matches_column("dfi__mean__primary_sphere")
    assert not d.matches_column("sap_max")
    assert d.all_column_specs() == ("dfi__mean", "dfi__mean__*")


# ---- catalog integrity ----------------------------------------------------
def test_catalog_nonempty_and_named():
    names = available_metrics()
    assert "fitness" in names and "fpocket" in names and "sap" in names
    assert len(names) == len(set(names)), "duplicate metric names"


def test_exact_columns_owned_by_exactly_one_metric():
    """No TSV column may be claimed by two catalog metrics (provenance integrity)."""
    owner: dict[str, str] = {}
    for d in all_metrics():
        for col in d.columns:
            assert col not in owner, (
                f"column {col!r} claimed by both {owner[col]!r} and {d.name!r}")
            owner[col] = d.name


def test_objective_labels_exist_in_ranking_specs():
    """Every objective_label in the catalog must be a real multi_objective label
    (the catalog reuses DEFAULT_METRIC_SPECS, never invents weights/targets)."""
    valid = {s.label for s in DEFAULT_METRIC_SPECS}
    for d in all_metrics():
        for lbl in d.objective_labels:
            assert lbl in valid, (
                f"metric {d.name!r} references unknown ranking label {lbl!r}")


def test_every_ranking_label_is_catalogued():
    """Conversely, every default ranking objective must be owned by some catalog
    metric — so no ranking axis is orphaned from selection/provenance."""
    catalogued = {lbl for d in all_metrics() for lbl in d.objective_labels}
    for s in DEFAULT_METRIC_SPECS:
        assert s.label in catalogued, (
            f"ranking label {s.label!r} ({s.column}) not owned by any catalog metric")


# ---- resolution -----------------------------------------------------------
def test_resolve_all_returns_default_on_in_canonical_order():
    res = resolve_metrics("all")
    expected = [d.name for d in all_metrics() if d.default_on]
    assert res.names() == expected
    # cms/rosetta/arpeggio are default-off
    assert "cms" not in res.names() and "rosetta" not in res.names()


def test_resolve_csv_subset_is_canonical_order_regardless_of_request_order():
    # request out of order; result must follow catalog order
    res = resolve_metrics("fpocket,length,fitness")
    assert res.names() == ["length", "fitness", "fpocket"]


def test_resolve_unknown_name_recorded_not_raised():
    res = resolve_metrics("fitness,does_not_exist")
    assert res.names() == ["fitness"]
    assert res.skipped_unknown == ["does_not_exist"]


def test_capabilities_skip_missing_dep_with_record():
    # No ligand/freesasa/fpocket/tunnel capabilities -> those metrics skipped.
    res = resolve_metrics("all", capabilities=frozenset())
    skipped = {d.name for d, _ in res.skipped_missing_dep}
    assert {"ligand_int", "sap", "fpocket", "pkvf"} <= skipped
    assert "ligand_int" not in res.names()
    # pure-sequence metrics with no deps still selected
    assert "instability" in res.names() and "fitness" in res.names()


def test_capabilities_none_bypasses_dep_check():
    res = resolve_metrics("all", capabilities=None)
    assert "fpocket" in res.names() and "ligand_int" in res.names()


def test_capabilities_present_selects_dep_metric():
    res = resolve_metrics("all", capabilities=frozenset(
        {DEP_LIGAND, DEP_FREESASA, DEP_FPOCKET, DEP_TUNNEL_SIF}))
    assert {"ligand_int", "sap", "fpocket", "pkvf"} <= set(res.names())


def test_role_filter_selects_only_filters():
    res = resolve_metrics("all", role=ROLE_FILTER)
    assert all(d.role == ROLE_FILTER for d in res.selected)
    assert "fitness" not in res.names()      # objective, excluded
    assert "instability" in res.names()      # filter, included


def test_resolved_selection_helpers():
    res = resolve_metrics("all", capabilities=None)
    stages = res.stages()
    assert STAGE_SEQ_FILTER in stages and STAGE_FPOCKET in stages
    # objective labels are deduped + all valid
    labels = res.objective_labels()
    valid = {s.label for s in DEFAULT_METRIC_SPECS}
    assert labels and set(labels) <= valid
    assert len(labels) == len(set(labels))
    # gating filters are exactly the role=filter selected metrics
    assert set(d.name for d in res.gating_filters()) == \
        set(d.name for d in res.selected if d.role == ROLE_FILTER)


def test_get_metric_roundtrip_and_unknown():
    d = get_metric("fpocket")
    assert d.stage == STAGE_FPOCKET and DEP_FPOCKET in d.deps
    with pytest.raises(KeyError):
        get_metric("nope")


def test_fpocket_is_a_filter():
    # fpocket druggability is a real final-topk survivor gate, so it must be
    # selectable via --filters (role=filter, gates_survivors).
    d = get_metric("fpocket")
    assert d.role == ROLE_FILTER and d.gates_survivors
    res = resolve_metrics("all", role=ROLE_FILTER, capabilities=None)
    assert "fpocket" in res.names()


def test_explicit_wrong_role_is_reported_not_silent():
    # A valid metric named under --filters but with the wrong role must be
    # reported (so the driver can fail fast), not silently dropped.
    res = resolve_metrics("fitness", role=ROLE_FILTER)
    assert res.names() == []
    assert res.skipped_wrong_role == ["fitness"]
    assert res.skipped_unknown == []


def test_all_with_role_does_not_report_wrong_role():
    # "all" + a role filter just narrows the catalog — non-matching roles are
    # expected, not user errors, so skipped_wrong_role stays empty.
    res = resolve_metrics("all", role=ROLE_FILTER)
    assert res.skipped_wrong_role == []
    assert res.selected and all(d.role == ROLE_FILTER for d in res.selected)


def test_empty_explicit_selection_resolves_empty():
    # ",,," is an explicit (non-"all") selection of nothing — no unknowns, but
    # an empty selected set the driver treats as an error.
    res = resolve_metrics(",,,")
    assert res.selected == [] and res.skipped_unknown == []
