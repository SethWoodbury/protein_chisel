"""Slow tests for sidechain_packing_and_scoring/opus_rota5_pack.

OPUS-Rota5 inference (3D-Unet + RotaFormer ensemble) is GPU-heavy
(~30-60 s per protein on a4000). Mark as slow::

    pytest -m slow tests/sidechain_packing_and_scoring/test_opus_rota5_pack.py
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


def test_opus_rota5_weights_present():
    """The Google Drive standalone should be downloaded + unzipped."""
    from protein_chisel.paths import (
        OPUS_ROTA5_ROTAFORMER_WEIGHTS,
        OPUS_ROTA5_UNET3D_WEIGHTS,
    )

    for k in (1, 2, 3):
        assert (OPUS_ROTA5_ROTAFORMER_WEIGHTS / f"rota5_{k}.h5").is_file(), (
            f"Missing rota5_{k}.h5 under {OPUS_ROTA5_ROTAFORMER_WEIGHTS}"
        )
        assert (OPUS_ROTA5_UNET3D_WEIGHTS / f"model_{k}.h5").is_file(), (
            f"Missing model_{k}.h5 under {OPUS_ROTA5_UNET3D_WEIGHTS}"
        )


def test_opus_rota5_mkdssp_bin_exists():
    from protein_chisel.paths import MKDSSP_BIN
    assert MKDSSP_BIN.is_file()
    assert MKDSSP_BIN.stat().st_mode & 0o111


def test_opus_rota5_pack_runs_end_to_end(tmp_path: Path):
    if not _gpu_visible():
        pytest.skip("OPUS-Rota5 needs GPU for practical runtime")
    from protein_chisel.tools.sidechain_packing_and_scoring.opus_rota5_pack import (
        opus_rota5_pack,
    )

    out_pdb = tmp_path / "design_opus_rota5.pdb"
    res = opus_rota5_pack(DESIGN_PDB, out_pdb_path=out_pdb)
    assert out_pdb.is_file()
    assert res.runtime_seconds > 0
    # design.pdb has ~208 residues
    assert 100 < res.n_residues < 250
