"""Robustness tests for the PDB line parsers in protonate_final.

We construct a few intentionally malformed ATOM lines and verify the
parser extracts reasonable values via either column or whitespace
fallback, never raising.
"""
from __future__ import annotations

import pytest

from protein_chisel.tools.protonate_final import parse_atom_line


# (case_name, raw ATOM line, expected field subset or None for "should be None")
CASES: list[tuple[str, str, dict | None]] = [
    (
        "standard_strict",
        "ATOM   2050  ND1 HIS A 132       3.574   2.127   5.353  1.00  0.00           N  \n",
        {"atom_name": "ND1", "resname": "HIS", "chain": "A", "resno": 132},
    ),
    (
        "rosetta_5char_overflow_HIS_D_chain_A",
        "ATOM    463  N  HIS_DA  60       6.405   5.686   7.346  1.00  0.00           N  \n",
        {"atom_name": "N", "resname": "HIS_D", "chain": "A", "resno": 60},
    ),
    (
        "rosetta_5char_overflow_HIE_chain_A",
        "ATOM    463  N  HIE  A  60       6.405   5.686   7.346  1.00  0.00           N  \n",
        {"atom_name": "N", "resname": "HIE", "chain": "A", "resno": 60},
    ),
    (
        "kcx_3char_normal",
        "ATOM   1273  CX  KCX A 157       1.234   5.678   9.012  1.00  0.00           C  \n",
        {"atom_name": "CX", "resname": "KCX", "chain": "A", "resno": 157},
    ),
    (
        "blank_chain_id",
        "ATOM   1273  N   LYS   157       1.234   5.678   9.012  1.00  0.00           N  \n",
        # blank chain at col 22 -> chain == ' '
        {"atom_name": "N", "resname": "LYS", "resno": 157},
    ),
    (
        "insertion_code",
        "ATOM   1273  N   LYS A 157A      1.234   5.678   9.012  1.00  0.00           N  \n",
        {"atom_name": "N", "resname": "LYS", "chain": "A", "resno": 157, "icode": "A"},
    ),
    (
        "extra_whitespace_columns_off",
        "ATOM    463 N    HIS   A 60         6.4   5.7   7.3  1.00  0.00           N\n",
        # column parse will fail or be weird; whitespace fallback should rescue
        {"resname": "HIS", "resno": 60},
    ),
    (
        "non_atom_line_returns_none",
        "REMARK 666 something here\n",
        None,
    ),
    (
        "truncated_line",
        "ATOM    463\n",
        None,
    ),
]


@pytest.mark.parametrize(
    "line,expected",
    [(c[1], c[2]) for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_parse_atom_line(line: str, expected: dict | None):
    result = parse_atom_line(line)
    if expected is None:
        assert result is None
        return
    assert result is not None
    for key, value in expected.items():
        assert result.get(key) == value, (
            f"field {key!r}: expected {value!r}, got {result.get(key)!r}"
        )
