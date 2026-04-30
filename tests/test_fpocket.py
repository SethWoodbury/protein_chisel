"""Cluster tests for tools/fpocket_run. Runs in esmc.sif (where fpocket
is installed via the vendored external/fpocket submodule build).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.cluster


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"


def test_fpocket_finds_binary():
    from protein_chisel.tools.fpocket_run import find_fpocket_executable

    exe = find_fpocket_executable()
    assert exe.endswith("/fpocket")


def test_fpocket_runs_on_design():
    from protein_chisel.tools.fpocket_run import fpocket_run

    res = fpocket_run(DESIGN_PDB, keep_outputs=False)
    # Designed enzymes have at least one pocket (the active site)
    assert res.n_pockets > 0, "expected at least one pocket on design.pdb"
    assert res.largest_pocket_volume > 0
    # Each pocket has volume + druggability score
    for p in res.pockets:
        assert p.volume >= 0
        assert isinstance(p.druggability_score, float)


def test_fpocket_to_dict_keys():
    from protein_chisel.tools.fpocket_run import fpocket_run

    res = fpocket_run(DESIGN_PDB, keep_outputs=False)
    d = res.to_dict()
    assert "fpocket__n_pockets" in d
    assert "fpocket__largest_pocket_volume" in d
    assert "fpocket__most_druggable_score" in d
    if res.n_pockets > 0:
        # Top pocket gets the flat namespace
        assert "fpocket__p__druggability" in d
        assert "fpocket__p__volume" in d
