"""Slow tests for sidechain_packing_and_scoring/flowpacker_score.

FlowPacker likelihood (Hutchinson trace ODE integration) is GPU-heavy.
Mark as slow so they don't run by default::

    pytest -m slow tests/sidechain_packing_and_scoring/test_flowpacker_score.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.slow


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"


def _gpu_visible() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return False
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return False
    return proc.returncode == 0 and "GPU" in proc.stdout


def test_flowpacker_pack_runs_end_to_end(tmp_path: Path):
    if not _gpu_visible():
        pytest.skip("FlowPacker needs GPU for practical runtime")
    from protein_chisel.tools.sidechain_packing_and_scoring.flowpacker_score import (
        flowpacker_pack,
    )

    res = flowpacker_pack(DESIGN_PDB, out_dir=tmp_path / "out")
    assert res.out_pdb_path.is_file()


def test_flowpacker_score_emits_per_chi_logp(tmp_path: Path):
    if not _gpu_visible():
        pytest.skip("FlowPacker needs GPU for practical runtime")
    from protein_chisel.tools.sidechain_packing_and_scoring.flowpacker_score import (
        flowpacker_score,
    )

    res = flowpacker_score(DESIGN_PDB)
    assert res.n_residues_scored > 100
    # logp_mean is on the order of -2 to -4 per chi for designed proteins
    assert res.logp_mean is not None
    d = res.to_dict()
    for k in ("flowpacker__logp_sum", "flowpacker__logp_mean",
              "flowpacker__logp_mean_chi1"):
        assert k in d
