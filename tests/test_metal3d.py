"""Tests for tools/metal3d_score.

The vendored scripts/run_metal3d.py self-relaunches into metal3d.sif
when invoked OUTSIDE a container — but bails out (assuming Metal3D is
already loaded) when invoked inside any other sif. So the slow
inference test must run from the host, not via esmc_call.

Cheap host tests (find_actual_metals, to_dict): unmarked, run by default.
Slow inference test: marked `slow` (not `cluster`); run with::

    pytest -m slow tests/test_metal3d.py
"""

from __future__ import annotations

from pathlib import Path

import pytest


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"


def test_find_actual_metals_design_pdb():
    """The YYE ligand on design.pdb has 2 Zn atoms (ZN1, ZN2)."""
    from protein_chisel.tools.metal3d_score import find_actual_metals

    metals = find_actual_metals(DESIGN_PDB)
    elements = {m["element"] for m in metals}
    assert "ZN" in elements
    # Two zinc atoms in the YYE ligand
    zn_count = sum(1 for m in metals if m["element"] == "ZN")
    assert zn_count == 2


def _gpu_visible() -> bool:
    """True iff a CUDA GPU is reachable on the current host.

    Metal3D's CNN inference is impractical on CPU (~10× slower); we
    skip the slow test on hosts without a GPU and require it to run
    via sbatch on a GPU partition instead.
    """
    import os
    import subprocess
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return False
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return False
    return proc.returncode == 0 and "GPU" in proc.stdout


@pytest.mark.slow
def test_metal3d_runs_end_to_end(tmp_path: Path):
    """Full Metal3D inference run on the design PDB.

    Skipped on hosts without a GPU — Metal3D inference on CPU is too
    slow for an interactive test (~minutes per protein). Run via sbatch
    on a GPU partition::

        sbatch -p gpu --gres=gpu:a4000:1 --wrap="
            pytest -m slow tests/test_metal3d.py::test_metal3d_runs_end_to_end -v"
    """
    if not _gpu_visible():
        pytest.skip("no GPU detected — Metal3D needs CUDA for practical runtime")

    from protein_chisel.tools.metal3d_score import metal3d_score

    res = metal3d_score(
        DESIGN_PDB,
        out_dir=tmp_path / "metal3d_run",
        keep_outputs=True,
        timeout=900.0,
    )
    # 2 zinc atoms (ZN1, ZN2) on the YYE ligand
    assert res.n_actual_metals == 2
    assert res.n_predicted_sites >= 0
    if res.n_predicted_sites > 0:
        assert 0 <= res.top_site_probability <= 1.0
        # At least one actual Zn should have *some* prediction nearby
        assert any(p > 0 for p in res.actual_metal_max_prob_within_4A.values())


def test_metal3d_source_dir_resolves():
    """The submodule path is exposed and contains Metal3D's source + weights."""
    from protein_chisel.tools.metal3d_score import metal3d_source_dir

    src = metal3d_source_dir()
    assert (src / "Metal3D").is_dir(), f"Metal3D dir missing under {src}"
    assert (src / "Metal3D" / "weights").is_dir()


def test_metal3d_to_dict_keys_exposed():
    """Without running the model, an empty result still serializes correctly."""
    from protein_chisel.tools.metal3d_score import Metal3DResult

    res = Metal3DResult()
    d = res.to_dict()
    assert "metal3d__n_actual_metals" in d
    assert "metal3d__n_predicted_sites" in d
    assert "metal3d__top_site_probability" in d
