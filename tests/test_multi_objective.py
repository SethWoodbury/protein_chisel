"""Tests for protein_chisel.scoring.multi_objective."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from protein_chisel.scoring.multi_objective import Objective, topsis_pareto_rank


def _toy_df() -> pd.DataFrame:
    # 4 designs x 2 objectives. fitness MAX, sap MIN.
    # A is the ideal corner (high fitness, low sap), D is the worst.
    # B and C are tradeoffs: both should be on the Pareto front.
    return pd.DataFrame({
        "id": ["A", "B", "C", "D"],
        "fitness": [-1.5, -1.7, -1.8, -2.0],
        "sap":     [10.0, 12.0, 8.0, 15.0],
    })


def test_pareto_front_membership():
    df = _toy_df()
    out = topsis_pareto_rank(df, [
        Objective("fitness", "max"),
        Objective("sap", "min"),
    ])
    on = dict(zip(out["id"], out["mo_on_pareto"]))
    # A dominates D on both axes; B is dominated by A on both axes; C
    # dominates everything except A on sap. So front = {A, C}.
    assert on["A"] is True or on["A"] == True   # noqa: E712
    assert on["C"] == True
    assert on["D"] == False
    assert on["B"] == False


def test_topsis_ranks_ideal_first():
    df = _toy_df()
    out = topsis_pareto_rank(df, [
        Objective("fitness", "max"),
        Objective("sap", "min"),
    ])
    rank_by_id = dict(zip(out["id"], out["mo_rank"]))
    # D is dominated and farthest from ideal, must be last.
    assert rank_by_id["D"] == 4
    # Front members (A, C) must rank above non-front members (B, D).
    front_ranks = [rank_by_id["A"], rank_by_id["C"]]
    nonfront_ranks = [rank_by_id["B"], rank_by_id["D"]]
    assert max(front_ranks) < min(nonfront_ranks)
    # Among front members, the one whose objectives are uniformly closer to
    # the (per-axis) ideal should win — C beats A because its sap=8 is
    # much closer to the column ideal than A's sap=10.
    topsis_by_id = dict(zip(out["id"], out["mo_topsis"]))
    assert topsis_by_id["C"] > topsis_by_id["A"]


def test_weights_can_break_ties():
    # Two designs, each ideal on a different axis: weighting fitness 10x
    # should pull the high-fitness one to rank #1.
    df = pd.DataFrame({
        "id": ["hi_fit", "lo_sap"],
        "fitness": [-1.0, -2.0],
        "sap":     [20.0, 5.0],
    })
    base = topsis_pareto_rank(df, [
        Objective("fitness", "max", weight=1.0),
        Objective("sap", "min", weight=1.0),
    ])
    weighted = topsis_pareto_rank(df, [
        Objective("fitness", "max", weight=10.0),
        Objective("sap", "min", weight=1.0),
    ])
    # Both should be on the front (incomparable points).
    assert base["mo_on_pareto"].all()
    assert weighted["mo_on_pareto"].all()
    # But under weighted scoring, hi_fit should beat lo_sap.
    w_rank = dict(zip(weighted["id"], weighted["mo_rank"]))
    assert w_rank["hi_fit"] == 1
    assert w_rank["lo_sap"] == 2


def test_nan_rows_go_to_the_bottom():
    df = pd.DataFrame({
        "id": ["good", "missing"],
        "fitness": [-1.0, np.nan],
        "sap":     [10.0, 5.0],
    })
    out = topsis_pareto_rank(df, [
        Objective("fitness", "max"),
        Objective("sap", "min"),
    ])
    rank_by_id = dict(zip(out["id"], out["mo_rank"]))
    assert rank_by_id["good"] == 1
    assert rank_by_id["missing"] == 2
    assert pd.isna(out.loc[out["id"] == "missing", "mo_topsis"].iloc[0])


def test_rejects_bad_direction_and_missing_column():
    df = _toy_df()
    with pytest.raises(KeyError):
        topsis_pareto_rank(df, [Objective("nonexistent", "max")])
    with pytest.raises(ValueError):
        topsis_pareto_rank(df, [Objective("fitness", "sideways")])
    with pytest.raises(ValueError):
        topsis_pareto_rank(df, [])
