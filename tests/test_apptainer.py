"""Tests for utils/apptainer — command building (offline) + one live run."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from protein_chisel.utils.apptainer import (
    ApptainerCall,
    PROTEIN_CHISEL_ROOT,
    esmc_call,
    in_apptainer,
    pyrosetta_call,
)


def test_command_includes_default_bind_and_pythonpath():
    call = ApptainerCall(sif=Path("/x/foo.sif"))
    cmd = call.build_command(["python", "-c", "print(1)"])
    assert cmd[0] == "apptainer"
    assert cmd[1] == "exec"
    # repo bind
    assert any(part.startswith("--bind") or "/code" in part for part in cmd)
    # PYTHONPATH set
    assert any(p.startswith("PYTHONPATH=/code/src") for p in cmd)
    # sif path passed
    assert "/x/foo.sif" in cmd
    # script appended at end
    assert cmd[-3:] == ["python", "-c", "print(1)"]


def test_with_bind_and_env_chains():
    call = (
        ApptainerCall(sif=Path("/x.sif"))
        .with_bind("/data/x")
        .with_env(MY_VAR="hello")
    )
    cmd = call.build_command(["python", "-V"])
    assert any("/data/x:/data/x" in c for c in cmd)
    assert any(c == "MY_VAR=hello" for c in cmd)


def test_nv_flag_added():
    call = ApptainerCall(sif=Path("/x.sif"), nv=True)
    cmd = call.build_command(["python"])
    assert "--nv" in cmd


def test_nv_flag_omitted_by_default():
    call = ApptainerCall(sif=Path("/x.sif"))
    cmd = call.build_command(["python"])
    assert "--nv" not in cmd


def test_esmc_call_binds_hf_caches():
    call = esmc_call()
    cmd = call.build_command(["python"])
    bind_args = [c for c in cmd if "esmc" in c or "saprot" in c]
    assert any("/net/databases/huggingface/esmc" in c for c in bind_args)
    assert any("/net/databases/huggingface/saprot" in c for c in bind_args)


def test_in_apptainer_false_on_host():
    # Tests run on the host
    assert in_apptainer() is False


@pytest.mark.cluster
def test_live_python_inline_in_pyrosetta_sif():
    """Sanity-check that pyrosetta.sif is reachable and PYTHONPATH works.

    Marked `cluster` so default test runs skip it; run with -m cluster.
    """
    call = pyrosetta_call()
    result = call.run_python_inline(
        "from protein_chisel.io.pdb import parse_remark_666; "
        "import sys; sys.exit(0)",
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
