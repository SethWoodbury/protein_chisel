"""Tests for the Stage-2 memory detection + PLM-footprint warning (logging-only)."""
from __future__ import annotations

import logging

import pytest

from protein_chisel.utils import resources as R


# ---- detect_available_mem_mb -------------------------------------------
def test_slurm_mem_per_node(monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "8000")
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    monkeypatch.setattr(R, "_read_cgroup_mem_limit_mb", lambda: None)
    monkeypatch.setattr(R, "_read_proc_mem_available_mb", lambda: None)
    assert R.detect_available_mem_mb() == (8000, "SLURM_MEM_PER_NODE")


def test_slurm_mem_per_cpu_times_cpus(monkeypatch):
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "2000")
    monkeypatch.setattr(R, "detect_n_cpus", lambda: (4, "slurm"))
    monkeypatch.setattr(R, "_read_cgroup_mem_limit_mb", lambda: None)
    monkeypatch.setattr(R, "_read_proc_mem_available_mb", lambda: None)
    assert R.detect_available_mem_mb() == (8000, "SLURM_MEM_PER_CPU")


def test_takes_min_of_slurm_and_cgroup(monkeypatch):
    # SLURM says 16G but cgroup caps at 6G -> the binding cap is 6G.
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16000")
    monkeypatch.setattr(R, "_read_cgroup_mem_limit_mb", lambda: 6000)
    monkeypatch.setattr(R, "_read_proc_mem_available_mb", lambda: 30000)
    mb, src = R.detect_available_mem_mb()
    assert mb == 6000 and src == "cgroup"


def test_falls_back_to_proc(monkeypatch):
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    monkeypatch.setattr(R, "_read_cgroup_mem_limit_mb", lambda: None)
    monkeypatch.setattr(R, "_read_proc_mem_available_mb", lambda: 12000)
    assert R.detect_available_mem_mb() == (12000, "/proc/meminfo")


def test_unknown_when_nothing(monkeypatch):
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    monkeypatch.setattr(R, "_read_cgroup_mem_limit_mb", lambda: None)
    monkeypatch.setattr(R, "_read_proc_mem_available_mb", lambda: None)
    assert R.detect_available_mem_mb() == (0, "unknown")


@pytest.mark.parametrize("content,expect_mb", [
    ("max\n", None),                       # cgroup v2 "no limit"
    (str(1 << 63) + "\n", None),           # v1 huge sentinel "no limit"
    (str(6 * 1024 * 1024 * 1024) + "\n", 6 * 1024),   # 6 GiB -> 6144 MB
    ("0\n", None),
])
def test_cgroup_reader_parsing(tmp_path, monkeypatch, content, expect_mb):
    # Point the v2 path at a temp file by intercepting open() for that path only.
    f = tmp_path / "memory.max"
    f.write_text(content)
    real_open = open

    def fake_open(path, *a, **k):
        if path == "/sys/fs/cgroup/memory.max":
            return real_open(f, *a, **k)
        if path == "/sys/fs/cgroup/memory/memory.limit_in_bytes":
            raise OSError("no v1")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    assert R._read_cgroup_mem_limit_mb() == expect_mb


# ---- estimate_plm_footprint_mb -----------------------------------------
def test_estimate_monotonic_and_dtype_and_gpu():
    big = R.estimate_plm_footprint_mb("esmc_600m", "saprot_1.3b", "fp32", on_gpu=False)
    small = R.estimate_plm_footprint_mb("esmc_300m", "saprot_35m", "fp32", on_gpu=False)
    assert big > small                                   # 1.3b >> 35m
    fp16 = R.estimate_plm_footprint_mb("esmc_600m", "saprot_1.3b", "fp16", on_gpu=False)
    assert fp16 < big                                    # fp16 < fp32
    gpu = R.estimate_plm_footprint_mb("esmc_600m", "saprot_1.3b", "fp32", on_gpu=True)
    assert gpu < big                                     # GPU host peak < CPU
    # anchored ballpark: CPU 1.3b fp32 ~14 GB, GPU ~8 GB
    assert 12000 <= big <= 16000 and 7000 <= gpu <= 9000


# ---- warn_if_plm_mem_tight ---------------------------------------------
def test_warns_when_tight(monkeypatch, caplog):
    # on_gpu pinned so the estimate is machine-independent (CPU factor).
    monkeypatch.setattr(R, "detect_available_mem_mb", lambda: (8000, "SLURM_MEM_PER_NODE"))
    with caplog.at_level(logging.WARNING, logger="protein_chisel.utils.resources"):
        tight = R.warn_if_plm_mem_tight("esmc_600m", "saprot_1.3b", "fp32", on_gpu=False)
    assert tight is True
    msg = caplog.text
    assert "TIGHT" in msg and ("SAPROT_MODEL" in msg or "fp16" in msg)


def test_silent_when_ample(monkeypatch, caplog):
    monkeypatch.setattr(R, "detect_available_mem_mb", lambda: (64000, "SLURM_MEM_PER_NODE"))
    with caplog.at_level(logging.WARNING, logger="protein_chisel.utils.resources"):
        tight = R.warn_if_plm_mem_tight("esmc_600m", "saprot_1.3b", "fp32", on_gpu=False)
    assert tight is False
    assert "TIGHT" not in caplog.text


def test_fp16_relieves_tightness(monkeypatch):
    # budget 11000 (threshold ~9350): CPU fp32 ~14000 tight, fp16 ~8000 not.
    monkeypatch.setattr(R, "detect_available_mem_mb", lambda: (11000, "x"))
    assert R.warn_if_plm_mem_tight("esmc_600m", "saprot_1.3b", "fp32",
                                   log=False, on_gpu=False) is True
    assert R.warn_if_plm_mem_tight("esmc_600m", "saprot_1.3b", "fp16",
                                   log=False, on_gpu=False) is False


def test_undetectable_budget_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(R, "detect_available_mem_mb", lambda: (0, "unknown"))
    with caplog.at_level(logging.WARNING, logger="protein_chisel.utils.resources"):
        assert R.warn_if_plm_mem_tight("esmc_600m", "saprot_1.3b", "fp32",
                                       on_gpu=False) is False
    assert "TIGHT" not in caplog.text
