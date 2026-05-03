"""Tests for pure-python helpers in scripts/iterative_design_v2.py.

We import the script as a module (its sibling-script imports are
defensive sys.path inserts so this works on host pytest as long as
PYTHONPATH includes the repo's src/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import iterative_design_v2 as v2   # noqa: E402


# ----------------------------------------------------------------------
# compute_catalytic_neighbor_omit_dict
# ----------------------------------------------------------------------


def _pt(rows):
    """Build a tiny PositionTable.df from rows of (resno, chain, name1,
    is_protein)."""
    df = pd.DataFrame(rows, columns=["resno", "chain", "name1", "is_protein"])
    return df


def test_omit_dict_catalytic_K_forbids_KR_at_neighbors():
    """Catalytic K at resno 50 -> forbid K/R at 49 and 51 (assuming
    they're protein and not also catalytic)."""
    df = _pt([
        (48, "A", "L", True),
        (49, "A", "A", True),
        (50, "A", "K", True),    # catalytic K
        (51, "A", "A", True),
        (52, "A", "L", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[50],
        chain="A",
    )
    assert out == {"A49": "KR", "A51": "KR"}


def test_omit_dict_catalytic_R_forbids_KR_at_neighbors():
    df = _pt([
        (10, "A", "G", True),
        (11, "A", "R", True),    # catalytic R
        (12, "A", "S", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[11],
        chain="A",
    )
    assert out == {"A10": "KR", "A12": "KR"}


def test_omit_dict_skips_non_K_R_catalytic_residues():
    """HIS/GLU catalytic residues don't trigger neighbor omission —
    they don't form OmpT motifs."""
    df = _pt([
        (60, "A", "L", True),
        (61, "A", "H", True),    # catalytic HIS
        (62, "A", "L", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[61],
        chain="A",
    )
    assert out == {}


def test_omit_dict_skips_neighbor_that_is_also_catalytic():
    """If catalytic residues are adjacent (e.g. K-K active site), don't
    over-constrain the second one."""
    df = _pt([
        (50, "A", "K", True),    # catalytic K
        (51, "A", "K", True),    # also catalytic K
        (52, "A", "L", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[50, 51],
        chain="A",
    )
    # 50's right neighbor (51) is in fixed_set -> skip
    # 51's left neighbor (50) is in fixed_set -> skip
    # 51's right neighbor (52) is NOT fixed -> forbid
    assert out == {"A52": "KR"}


def test_omit_dict_skips_off_chain_neighbors():
    """N-terminal catalytic K (resno 1) has no resno=0 neighbor on the
    chain; only one neighbor gets the omit."""
    df = _pt([
        (1, "A", "K", True),     # catalytic K, N-terminal
        (2, "A", "A", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[1],
        chain="A",
    )
    assert out == {"A2": "KR"}


def test_omit_dict_pte_i1_real_classify_table_yields_156_158_only():
    """Snapshot test on the actual PTE_i1 PositionTable from a previous
    run: catalytic set = (60,64,128,131,132,157), only 157 is K, so
    expected output is {"A156": "KR", "A158": "KR"}."""
    classify_path = Path(
        "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260503-145751/"
        "classify/positions.tsv"
    )
    if not classify_path.is_file():
        pytest.skip("PTE_i1 classify_positions output not available")
    df = pd.read_csv(classify_path, sep="\t")
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=v2.DEFAULT_CATRES,
        chain="A",
    )
    assert out == {"A156": "KR", "A158": "KR"}


def test_omit_dict_custom_forbid_aas():
    """Caller can change the forbidden AA set."""
    df = _pt([
        (10, "A", "K", True),
        (11, "A", "K", True),    # catalytic K
        (12, "A", "L", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[11],
        chain="A",
        forbid_aas="K",  # only forbid K, allow R
    )
    assert out == {"A10": "K", "A12": "K"}


def test_omit_dict_chain_filter():
    """Only neighbors on the same chain as the catalytic residue are
    forbidden."""
    df = _pt([
        (50, "A", "K", True),    # catalytic K on chain A
        (51, "A", "L", True),
        (50, "B", "L", True),    # different chain, same resno (ligand)
        (51, "B", "K", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[50],
        chain="A",
    )
    # Only A51 should appear (not B51, not the off-chain partner)
    assert out == {"A51": "KR"}
