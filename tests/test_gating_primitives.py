"""Lock the metric-gating byte-identity contract in the driver: with the default
selection the gating primitives are no-ops (so the default run is byte-identical).

Loads scripts/iterative_design.py as a module (registered in sys.modules so its
dataclass decorators resolve) and checks the module-global filter/ranking gates.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "iterative_design.py"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("itdesign_under_test", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # required for @dataclass to resolve
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:               # pragma: no cover - import env issue
        pytest.skip(f"driver not importable in this env: {exc}")
    return mod


def test_gating_globals_default_to_no_gating(driver):
    # Defaults => no gating => byte-identical default run.
    assert driver._RANKING_LABEL_FILTER is None
    assert driver._ACTIVE_FILTERS is None


def test_filter_active_all_on_when_none(driver):
    driver._ACTIVE_FILTERS = None
    assert driver._filter_active("anything") is True
    assert driver._filter_active("sap") is True


def test_filter_active_respects_selection(driver):
    try:
        driver._ACTIVE_FILTERS = frozenset({"sap", "clash"})
        assert driver._filter_active("sap") is True
        assert driver._filter_active("clash") is True
        assert driver._filter_active("boman") is False
        assert driver._filter_active("fpocket") is False
    finally:
        driver._ACTIVE_FILTERS = None      # restore the no-gating default


def test_ranking_label_filter_threads_to_select_specs(driver):
    # The driver delegates objective gating to multi_objective.select_specs_by_label,
    # which is identity for None; confirm the wiring contract.
    from protein_chisel.scoring.multi_objective import (
        DEFAULT_METRIC_SPECS, select_specs_by_label,
    )
    assert select_specs_by_label(DEFAULT_METRIC_SPECS, driver._RANKING_LABEL_FILTER) \
        == DEFAULT_METRIC_SPECS
