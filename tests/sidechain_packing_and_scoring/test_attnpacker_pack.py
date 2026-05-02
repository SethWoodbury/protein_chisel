"""Slow tests for sidechain_packing_and_scoring/attnpacker_pack.

AttnPacker inference is GPU-heavy and the weights archive is ~7 GB
(unzip is ~5 min one-time). Mark as slow::

    pytest -m slow tests/sidechain_packing_and_scoring/test_attnpacker_pack.py
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


def test_attnpacker_weights_present():
    """The Zenodo weights tarball should be available."""
    weights_dir = Path("/net/databases/lab/attnpacker")
    has_zip = (weights_dir / "AttnPackerPTM_V2.zip").is_file()
    has_unzipped = (weights_dir / "AttnPackerPTM_V2").is_dir()
    assert has_zip or has_unzipped, (
        f"AttnPacker weights missing under {weights_dir}; download from "
        "https://zenodo.org/records/7713779/files/AttnPackerPTM_V2.zip"
    )


def test_attnpacker_pack_runs_end_to_end(tmp_path: Path):
    if not _gpu_visible():
        pytest.skip("AttnPacker needs GPU for practical runtime")
    from protein_chisel.tools.sidechain_packing_and_scoring.attnpacker_pack import (
        attnpacker_pack,
    )

    out_pdb = tmp_path / "design_attnpacker.pdb"
    res = attnpacker_pack(DESIGN_PDB, out_pdb_path=out_pdb)
    assert out_pdb.is_file()
    assert res.runtime_seconds > 0
    assert res.resource_root is not None
