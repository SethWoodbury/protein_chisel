"""Tests for the opt-in solubility-band veto helper in scripts/iterative_design.py.

WS-A: the deferred-rescue path could ship a ``passed_seq_filter=False`` design
(e.g. GRAVY=1.05) as rank-0 because rescue re-buckets on fpocket/tunnel/struct only.
``_within_solubility_band`` is the pure predicate that the opt-in
``--ship_solubility_veto`` uses so rescue / final selection can never ship a design
outside the GRAVY + net-charge band. It must mirror ``stage_seq_filter`` exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import iterative_design as v2  # noqa: E402

# The real PTE band (CycleConfig final cycle): gravy [-0.8, 0.3], charge (-18, -4).
BAND = dict(gravy_min=-0.8, gravy_max=0.3, net_charge_min=-18.0, net_charge_max=-4.0)


def _df(rows):
    return pd.DataFrame(rows)


def test_in_band_passes():
    df = _df([{"gravy": 0.0, "net_charge_full_HH": -10.0}])
    assert v2._within_solubility_band(df, **BAND).tolist() == [True]


def test_the_real_bad_design_fails():
    # The actual shipped rank-0: GRAVY=1.05, charge=-8.96 -> must be vetoed.
    df = _df([{"gravy": 1.05, "net_charge_full_HH": -8.96}])
    assert v2._within_solubility_band(df, **BAND).tolist() == [False]


def test_charge_out_of_band_fails_both_sides():
    df = _df([
        {"gravy": 0.0, "net_charge_full_HH": -3.0},    # > -4 (too positive)
        {"gravy": 0.0, "net_charge_full_HH": -20.0},   # < -18 (too negative)
    ])
    assert v2._within_solubility_band(df, **BAND).tolist() == [False, False]


def test_exclusive_charge_inclusive_gravy_boundaries():
    # Mirrors stage_seq_filter: gravy bounds INCLUSIVE, charge bounds EXCLUSIVE.
    df = _df([
        {"gravy": 0.3, "net_charge_full_HH": -10.0},    # gravy == max -> pass
        {"gravy": -0.8, "net_charge_full_HH": -10.0},   # gravy == min -> pass
        {"gravy": 0.0, "net_charge_full_HH": -4.0},     # charge == max -> FAIL (exclusive)
        {"gravy": 0.0, "net_charge_full_HH": -18.0},    # charge == min -> FAIL (exclusive)
    ])
    assert v2._within_solubility_band(df, **BAND).tolist() == [True, True, False, False]


def test_nan_and_missing_value_fail_closed():
    df = _df([
        {"gravy": np.nan, "net_charge_full_HH": -10.0},
        {"gravy": 0.0},  # net_charge_full_HH missing for this row -> NaN
    ])
    assert v2._within_solubility_band(df, **BAND).tolist() == [False, False]


def test_missing_columns_fail_closed():
    df = _df([{"foo": 1}])
    assert v2._within_solubility_band(df, **BAND).tolist() == [False]


def test_empty_df_returns_empty():
    df = pd.DataFrame(columns=["gravy", "net_charge_full_HH"])
    assert v2._within_solubility_band(df, **BAND).tolist() == []


def test_custom_tight_band_minus5_to_minus15():
    # The user's ideal charge window is [-15, -5]; verify the helper honors it.
    band = dict(gravy_min=-2.0, gravy_max=-0.1, net_charge_min=-15.0, net_charge_max=-5.0)
    df = _df([
        {"gravy": -0.5, "net_charge_full_HH": -10.0},   # pass
        {"gravy": -0.5, "net_charge_full_HH": -4.0},    # charge too positive
        {"gravy": 0.2, "net_charge_full_HH": -10.0},    # gravy too high
    ])
    assert v2._within_solubility_band(df, **band).tolist() == [True, False, False]
