"""Slow tests for sidechain_packing_and_scoring/pippack_score.

PIPPack inference takes ~5-10 s on GPU. Tests are marked 'slow' so they
don't run by default; invoke with::

    pytest -m slow tests/sidechain_packing_and_scoring/test_pippack_score.py

Skipped on hosts without a CUDA GPU.
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


def test_pippack_pack_runs_end_to_end(tmp_path: Path):
    if not _gpu_visible():
        pytest.skip("PIPPack needs GPU for practical runtime")
    from protein_chisel.tools.sidechain_packing_and_scoring.pippack_score import (
        pippack_pack,
    )

    res = pippack_pack(DESIGN_PDB, out_dir=tmp_path / "out")
    assert res.out_pdb_path.is_file()
    assert res.runtime_seconds > 0


def test_pippack_score_computes_chi_mae(tmp_path: Path):
    if not _gpu_visible():
        pytest.skip("PIPPack needs GPU for practical runtime")
    from protein_chisel.tools.sidechain_packing_and_scoring.pippack_score import (
        pippack_score,
    )

    res = pippack_score(DESIGN_PDB)
    # Sensible numbers for a designed protein
    assert res.n_residues_scored > 100
    assert 0.0 <= res.rotamer_recovery <= 1.0
    assert res.mean_chi_mae >= 0.0
    d = res.to_dict()
    assert "pippack__mean_chi_mae" in d
    assert "pippack__rotamer_recovery" in d
