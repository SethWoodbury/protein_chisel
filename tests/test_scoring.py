"""Host tests for scoring/{aggregate, pareto, diversity}.

Pure-numpy / pandas; no PyRosetta or PLM dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from protein_chisel.scoring.aggregate import (
    AggregationPolicy,
    aggregate_metric_table,
    default_policy,
    paired_apo_holo_delta,
)
from protein_chisel.scoring.diversity import (
    hamming_distance,
    hamming_matrix,
    mask_from_position_table,
    select_diverse,
)
from protein_chisel.scoring.pareto import (
    HardConstraint,
    Objective,
    apply_hard_constraints,
    crowding_distance,
    epsilon_pareto_front,
)


# ---- aggregate ------------------------------------------------------------


def _sample_metric_df() -> pd.DataFrame:
    return pd.DataFrame({
        "sequence_id": ["d1"] * 3 + ["d2"] * 3,
        "conformer_index": [0, 1, 2] * 2,
        "fold_source": ["designed", "AF3_seed1", "AF3_seed1"] * 2,
        "buns__n_buried_unsat": [2, 5, 3, 0, 0, 1],
        "shape__rg": [15.2, 15.5, 15.3, 12.0, 12.1, 11.9],
        "ss__helix_frac": [0.40, 0.39, 0.41, 0.30, 0.30, 0.29],
        "protparam__pi": [6.5, 6.5, 6.5, 7.2, 7.2, 7.2],
        "ligand__name3": ["YYE"] * 6,
    })


def test_aggregate_failure_metric_uses_max():
    df = _sample_metric_df()
    agg = aggregate_metric_table(df)
    # d1 has unsat counts [2, 5, 3] -> max is 5
    d1 = agg[agg["sequence_id"] == "d1"].iloc[0]
    assert d1["buns__n_buried_unsat__max"] == 5
    assert d1["buns__n_buried_unsat__any_nonzero"] == True
    # d2 has [0, 0, 1] -> max 1
    d2 = agg[agg["sequence_id"] == "d2"].iloc[0]
    assert d2["buns__n_buried_unsat__max"] == 1


def test_aggregate_descriptive_emits_mean_std_min_max():
    df = _sample_metric_df()
    agg = aggregate_metric_table(df)
    cols = set(agg.columns)
    for sub in ("__mean", "__std", "__min", "__max"):
        assert any(c.endswith(sub) for c in cols if "shape__rg" in c)


def test_aggregate_first_for_sequence_metrics():
    df = _sample_metric_df()
    agg = aggregate_metric_table(df)
    d1 = agg[agg["sequence_id"] == "d1"].iloc[0]
    # protparam__pi uses 'first' strategy -> raw column, not __mean
    assert "protparam__pi" in d1.index
    assert d1["protparam__pi"] == 6.5


def test_aggregate_keeps_per_source_means():
    df = _sample_metric_df()
    agg = aggregate_metric_table(
        df, keep_per_source=("designed", "AF3_seed1")
    )
    cols = set(agg.columns)
    assert "src__designed__shape__rg" in cols
    assert "src__AF3_seed1__shape__rg" in cols


def test_aggregate_n_conformers_present():
    df = _sample_metric_df()
    agg = aggregate_metric_table(df)
    assert all(agg["n_conformers"] == 3)


def test_paired_apo_holo_delta():
    df = pd.DataFrame({
        "sequence_id": ["s1", "s1", "s2", "s2"],
        "fold_source": ["apo", "holo", "apo", "holo"],
        "shape__rg": [15.0, 15.5, 12.0, 12.3],
        "interact__n_hbonds": [10, 25, 5, 15],
    })
    deltas = paired_apo_holo_delta(
        df, metric_columns=["shape__rg", "interact__n_hbonds"],
        apo_label="apo", holo_label="holo",
    )
    s1 = deltas[deltas["sequence_id"] == "s1"].iloc[0]
    assert abs(s1["delta__shape__rg"] - 0.5) < 1e-9
    assert s1["delta__interact__n_hbonds"] == 15


# ---- Pareto ---------------------------------------------------------------


def test_hard_constraints_filter_pi_range():
    df = pd.DataFrame({"protparam__pi": [3.5, 5.0, 7.0, 10.0]})
    constraint = HardConstraint("protparam__pi", min_value=4.0, max_value=8.0)
    survived, drops = apply_hard_constraints(df, [constraint])
    assert list(survived["protparam__pi"]) == [5.0, 7.0]
    assert drops["protparam__pi"] == 2


def test_hard_constraints_treat_nan_as_failing():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    survived, _ = apply_hard_constraints(df, [HardConstraint("x", min_value=0.0)])
    assert len(survived) == 2  # NaN dropped


def test_pareto_one_objective_minimization():
    """Single min-objective: front = the single minimum point."""
    df = pd.DataFrame({"score": [5.0, 3.0, 4.0, 1.0, 2.0]})
    front = epsilon_pareto_front(df, [Objective("score", direction="min")])
    assert len(front) == 1
    assert front["score"].iloc[0] == 1.0


def test_pareto_two_objective_classic():
    """Classic 2-D Pareto: x and y to minimize.
    Points (1, 5), (2, 4), (3, 3), (4, 2), (5, 1) are all non-dominated.
    Adding (10, 10) should be dominated by everything.
    """
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
        "y": [5.0, 4.0, 3.0, 2.0, 1.0, 10.0],
    })
    front = epsilon_pareto_front(
        df,
        [Objective("x", direction="min"), Objective("y", direction="min")],
    )
    assert len(front) == 5  # the 5 trade-off points; (10, 10) dropped


def test_pareto_max_direction():
    df = pd.DataFrame({"score": [1.0, 5.0, 3.0]})
    front = epsilon_pareto_front(df, [Objective("score", direction="max")])
    assert len(front) == 1
    assert front["score"].iloc[0] == 5.0


def test_pareto_epsilon_collapses_close_points():
    """ε=1.0 should collapse points within 1.0 of each other."""
    df = pd.DataFrame({"x": [1.0, 1.5, 2.0, 5.0]})
    front = epsilon_pareto_front(
        df, [Objective("x", direction="min", epsilon=1.0)]
    )
    # Both 1.0 and 1.5 fall in the same epsilon bin (1.0). 2.0 is in the
    # next bin (2.0). With ε-dominance, the bin-1 points are tied "best"
    # and 2.0 / 5.0 are dominated.
    assert len(front) == 2
    assert set(front["x"].tolist()) == {1.0, 1.5}


def test_crowding_distance_endpoints_infinite():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    cd = crowding_distance(df, [Objective("x", direction="min")])
    assert cd[0] == np.inf
    assert cd[2] == np.inf
    assert cd[1] == pytest.approx((3.0 - 1.0) / (3.0 - 1.0))


# ---- diversity ------------------------------------------------------------


def test_hamming_distance_basic():
    assert hamming_distance("AAAA", "AAAA") == 0
    assert hamming_distance("AAAA", "AAAB") == 1
    assert hamming_distance("AAAA", "BBBB") == 4


def test_hamming_distance_with_mask():
    # Only positions 0 and 1 count
    mask = [True, True, False, False]
    assert hamming_distance("AAAA", "BBAA", mask=mask) == 2  # both differ
    assert hamming_distance("AAAA", "AABB", mask=mask) == 0  # neither differ


def test_hamming_distance_length_mismatch_raises():
    with pytest.raises(ValueError):
        hamming_distance("AAA", "AAAA")


def test_hamming_matrix_symmetric():
    seqs = ["AAAA", "AAAB", "AABB", "BBBB"]
    m = hamming_matrix(seqs)
    assert m.shape == (4, 4)
    assert np.array_equal(m, m.T)
    assert m[0, 3] == 4
    assert m[1, 2] == 1


def test_select_diverse_min_distance():
    seqs = ["AAAA", "AAAB", "AABB", "ABBB", "BBBB"]
    df = pd.DataFrame({"sequence": seqs, "score": [5, 4, 3, 2, 1]})
    # min_distance=2 forces gaps; with score-desc preference
    chosen = select_diverse(df, "sequence", k=10, min_distance=2,
                            score_col="score", score_direction="max")
    # AAAA chosen first; AAAB rejected (d=1); AABB accepted (d=2);
    # ABBB rejected (d=1 from AABB); BBBB accepted (d=2 from AABB).
    seqs_picked = chosen["sequence"].tolist()
    assert "AAAA" in seqs_picked
    assert "AABB" in seqs_picked
    assert "BBBB" in seqs_picked


def test_select_diverse_caps_at_k():
    seqs = [f"{i:04b}".replace("0", "A").replace("1", "B") for i in range(16)]
    df = pd.DataFrame({"sequence": seqs})
    chosen = select_diverse(df, "sequence", k=4, min_distance=1)
    assert len(chosen) == 4


def test_mask_from_position_table_skips_active_site_and_ligand():
    pt_df = pd.DataFrame({
        "resno": [1, 2, 3, 4, 5],
        "class": ["surface", "active_site", "buried", "ligand", "first_shell"],
        "is_protein": [True, True, True, False, True],
    })
    mask = mask_from_position_table(pt_df)
    # 4 protein rows; ligand row excluded entirely.
    # Default mutable_classes excludes active_site, includes the rest.
    assert mask == [True, False, True, True]
