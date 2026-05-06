"""Auto-detect available CPU/GPU resources for parallelization.

Prefers SLURM env vars (`SLURM_CPUS_PER_TASK`) over OS calls because
slurm allocates a subset of node CPUs to each job. Falls back to
``os.sched_getaffinity`` (Linux), then ``os.cpu_count``.

For GPU: defers to PyTorch when present so we get the actual visible
device count after `CUDA_VISIBLE_DEVICES` filtering. Falls back to
``nvidia-smi`` count or 0.

Threading: when running on CPU (no GPU), this module also configures
PyTorch / numpy thread counts to use the available CPU budget instead
of the (default) one-thread-per-physical-core that often oversubscribes
inside a slurm allocation.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


LOGGER = logging.getLogger("protein_chisel.utils.resources")


@dataclass(frozen=True)
class ResourceInfo:
    n_cpus: int
    n_gpus: int
    has_torch_cuda: bool
    source_cpu: str       # "slurm" | "affinity" | "cpu_count" | "fallback"
    source_gpu: str       # "torch" | "nvidia-smi" | "none"

    def __str__(self) -> str:
        gpu_msg = "no" if self.n_gpus == 0 else f"{self.n_gpus} (cuda={self.has_torch_cuda})"
        return (
            f"ResourceInfo(cpus={self.n_cpus} [{self.source_cpu}], "
            f"gpus={gpu_msg} [{self.source_gpu}])"
        )


def detect_n_cpus() -> tuple[int, str]:
    """Return (n_cpus, source). Slurm-aware."""
    # 1. Slurm allocation
    s = os.environ.get("SLURM_CPUS_PER_TASK")
    if s and s.isdigit():
        return int(s), "slurm"
    # 2. Linux affinity (respects taskset / cgroups)
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0)), "affinity"
        except OSError:
            pass
    # 3. Total OS cores
    n = os.cpu_count()
    if n is not None and n > 0:
        return n, "cpu_count"
    return 1, "fallback"


def detect_n_gpus(*, allow_torch_init: bool = False) -> tuple[int, bool, str]:
    """Return (n_gpus, has_torch_cuda, source).

    Resolution order (fork-safe by default):
      1. ``CUDA_VISIBLE_DEVICES`` env var — counts comma-separated entries.
         Set by both Slurm (when --gres=gpu:N) and manual launches.
      2. Slurm GPU env vars (``SLURM_GPUS_ON_NODE``, ``SLURM_JOB_GPUS``).
      3. ``nvidia-smi -L`` subprocess — does NOT initialize CUDA in the
         parent process.
      4. ``torch.cuda.device_count()`` — LAST RESORT, only when
         ``allow_torch_init=True``. Calling ``torch.cuda.is_available()``
         in a long-lived parent that later forks (e.g. for
         multiprocessing.Pool) is fork-unsafe and is the leading
         hypothesis behind the catastrophic fpocket-collapse-en-masse
         mode observed on GPU nodes (codex deep-dive 2026-05-06).

    ``has_torch_cuda`` is only True when path 4 actually ran and
    succeeded; the env-var paths cannot tell whether torch will work.
    Callers that care about the torch-cuda flag (e.g. the LigandMPNN
    sampler) should set ``allow_torch_init=True`` explicitly, AND only
    do so AFTER all parallel CPU stages are done.
    """
    # 1. CUDA_VISIBLE_DEVICES (most authoritative when set)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() and cvd.strip() != "-1":
        # comma-separated; empty entries skipped
        ids = [s for s in cvd.split(",") if s.strip()]
        if ids:
            return len(ids), False, "CUDA_VISIBLE_DEVICES"
    # 2. Slurm-set GPU env (fallback when CUDA_VISIBLE_DEVICES isn't set)
    for v in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK", "SLURM_JOB_GPUS"):
        s = os.environ.get(v, "").strip()
        if s and s.isdigit() and int(s) > 0:
            return int(s), False, v
    # 3. nvidia-smi -L (subprocess; does not init CUDA in this process)
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL,
            ).decode("utf-8", errors="ignore")
            count = sum(1 for line in out.splitlines() if line.startswith("GPU"))
            if count > 0:
                return count, False, "nvidia-smi"
        except Exception:
            pass
    # 4. torch (opt-in only; risks fork-after-CUDA-init landmine)
    if allow_torch_init:
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.device_count()), True, "torch"
        except ImportError:
            pass
        except Exception:
            pass
    return 0, False, "none"


def detect_resources(
    *, log: bool = True, allow_torch_init: bool = False,
) -> ResourceInfo:
    """One-shot detection of CPU + GPU resources.

    ``allow_torch_init`` defaults to False: GPU count is resolved from
    env vars / nvidia-smi instead of torch.cuda.is_available(). This
    keeps the parent process fork-safe when later stages use
    multiprocessing.Pool. Pass ``allow_torch_init=True`` ONLY when
    every parallel-CPU stage that uses fork() is already done.
    """
    n_cpus, src_cpu = detect_n_cpus()
    n_gpus, has_torch_cuda, src_gpu = detect_n_gpus(
        allow_torch_init=allow_torch_init,
    )
    info = ResourceInfo(
        n_cpus=n_cpus, n_gpus=n_gpus, has_torch_cuda=has_torch_cuda,
        source_cpu=src_cpu, source_gpu=src_gpu,
    )
    if log:
        LOGGER.info("Resources: %s", info)
    return info


def configure_torch_threads(n_cpus: int, *, force: bool = False) -> None:
    """When running on CPU, set PyTorch thread counts to use the full
    allocated CPU budget. Only applies if torch is importable and we
    haven't already configured threads (or `force=True`).

    PyTorch defaults: `torch.set_num_threads = N_physical_cores`
    (whole machine). On a slurm CPU job with cpus_per_task=8, that
    can oversubscribe and slow things down. We pin threads to the
    actual allocation.
    """
    try:
        import torch
    except ImportError:
        return
    cur = torch.get_num_threads()
    if cur == n_cpus and not force:
        return
    torch.set_num_threads(n_cpus)
    # MKL is independent of pytorch's intra-op threads; align them.
    try:
        torch.set_num_interop_threads(max(1, n_cpus // 2))
    except RuntimeError:
        # set_num_interop_threads must be called before parallel work;
        # if torch is already mid-flight it errors. Non-fatal.
        pass
    LOGGER.info(
        "torch threads: was %d -> %d (cpus_per_task=%d)",
        cur, n_cpus, n_cpus,
    )


def pool_workers(
    n_jobs: int,
    *,
    cpu_budget: Optional[int] = None,
    cap: int = 8,
    min_for_pool: int = 3,
) -> int:
    """Decide how many workers to use for a Pool.map over ``n_jobs`` items.

    Args:
        n_jobs: number of work items.
        cpu_budget: total CPUs available (defaults to detect_n_cpus()[0]).
        cap: hard cap on workers; default 8 (Pool overhead grows past this).
        min_for_pool: if ``n_jobs < min_for_pool``, return 1 (serial).

    Returns 1 if a Pool would be wasteful (small workload), else min(n_jobs, cpu_budget, cap).
    """
    if cpu_budget is None:
        cpu_budget, _ = detect_n_cpus()
    if n_jobs < min_for_pool:
        return 1
    return max(1, min(n_jobs, cpu_budget, cap))


__all__ = [
    "ResourceInfo",
    "detect_n_cpus",
    "detect_n_gpus",
    "detect_resources",
    "configure_torch_threads",
    "pool_workers",
]
