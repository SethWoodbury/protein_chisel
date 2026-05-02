"""Cluster tests for sidechain_packing_and_scoring/faspr_pack.

FASPR is a tiny C++ binary; tests run on the host directly (no sif).
The cluster install is at /net/software/lab/faspr/bin/FASPR.
"""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.cluster


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"


def test_faspr_binary_resolves():
    from protein_chisel.tools.sidechain_packing_and_scoring.faspr_pack import (
        _find_faspr_binary,
    )

    p = _find_faspr_binary()
    assert p.is_file()
    # Must be executable
    assert p.stat().st_mode & 0o111


def test_faspr_repack_design_pdb(tmp_path: Path):
    from protein_chisel.tools.sidechain_packing_and_scoring.faspr_pack import (
        faspr_pack,
    )

    out = tmp_path / "design_faspr.pdb"
    res = faspr_pack(DESIGN_PDB, out_pdb_path=out)
    assert out.is_file()
    # design.pdb has ~208 residues; FASPR shouldn't drop any
    assert 100 < res.n_residues <= 250
    # FASPR is fast — should be well under 30 s
    assert res.runtime_seconds < 30.0
    # to_dict surfaces metric keys
    d = res.to_dict()
    assert "faspr__n_residues" in d
    assert "faspr__runtime_seconds" in d


def test_faspr_extract_sequence_matches_pdb_length():
    """Sanity: the 1-letter sequence we hand to FASPR matches the PDB."""
    from protein_chisel.tools.sidechain_packing_and_scoring.faspr_pack import (
        _extract_sequence_from_pdb,
    )

    seq = _extract_sequence_from_pdb(DESIGN_PDB)
    # Same as n_residues in the test above (just upper-case 1-letter codes).
    assert 100 < len(seq) <= 250
    # Only canonical 20 letters.
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_faspr_with_fixed_residues_lower_cases_those_positions(tmp_path: Path):
    """Passing fixed_residues should lower-case those positions in the
    sequence string. This is the FASPR convention for "fix this rotamer".
    """
    from protein_chisel.tools.sidechain_packing_and_scoring.faspr_pack import (
        faspr_pack,
    )

    out = tmp_path / "design_fixed.pdb"
    res = faspr_pack(
        DESIGN_PDB, out_pdb_path=out,
        fixed_residues=[10, 20, 30],
    )
    # Positions 10, 20, 30 should be lower-case in the sequence we sent
    s = res.sequence_used
    assert s is not None
    assert s[10].islower(), f"pos 10 not lower-case: {s[10]}"
    assert s[20].islower()
    assert s[30].islower()
    # And the rest of the string is upper-case.
    assert s[0].isupper()
    assert s[5].isupper()
