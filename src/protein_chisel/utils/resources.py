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


# ---------------------------------------------------------------------------
# Memory detection + PLM footprint warning (read-only; logging-only — never
# changes pipeline behavior). Used by Stage 2 (precompute_plm_artifacts) to warn
# before the big PLMs (SaProt 1.3B) OOM on a too-small --mem.
# ---------------------------------------------------------------------------


def _read_cgroup_mem_limit_mb() -> Optional[int]:
    """cgroup memory cap in MB (v2 ``memory.max`` then v1 ``limit_in_bytes``);
    ``None`` if absent or set to "no limit" (the "max" string / huge sentinels)."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            b = int(raw)
        except ValueError:
            continue
        if b <= 0 or b >= (1 << 62):   # ignore the "unlimited" sentinels
            continue
        return b // (1024 * 1024)
    return None


def _read_proc_mem_available_mb() -> Optional[int]:
    """Node-wide ``MemAvailable`` (MB) from /proc/meminfo; ``None`` if unreadable."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024   # kB -> MB
    except (OSError, ValueError, IndexError):
        pass
    return None


def detect_available_mem_mb() -> tuple[int, str]:
    """Return ``(mem_mb, source)`` — the binding host-memory cap for this process.

    Read-only + fork-safe. Takes the MIN of the SLURM allocation and the cgroup
    limit when both are present (the smaller is the real cap); falls back to
    /proc/meminfo MemAvailable (node-wide, weakest). ``(0, "unknown")`` if nothing
    is detectable.
    """
    candidates: list[tuple[int, str]] = []
    s = os.environ.get("SLURM_MEM_PER_NODE", "").strip()
    if s.isdigit() and int(s) > 0:
        candidates.append((int(s), "SLURM_MEM_PER_NODE"))
    else:
        pc = os.environ.get("SLURM_MEM_PER_CPU", "").strip()
        if pc.isdigit() and int(pc) > 0:
            ncpu, _ = detect_n_cpus()
            candidates.append((int(pc) * ncpu, "SLURM_MEM_PER_CPU"))
    cg = _read_cgroup_mem_limit_mb()
    if cg:
        candidates.append((cg, "cgroup"))
    pm = _read_proc_mem_available_mb()
    if pm:
        candidates.append((pm, "/proc/meminfo"))
    if not candidates:
        return 0, "unknown"
    return min(candidates, key=lambda t: t[0])


# Rough resident-model footprint (MB, float32) per PLM variant. Anchored to the
# measured datapoint (L=202, esmc_600m+saprot_1.3b: ~8 GB GPU host / ~14 GB CPU
# MaxRSS) and the saprot_1.3b ~6 GB VRAM recon. Estimates for a WARNING only.
_PLM_FOOTPRINT_MB = {
    "esmc_300m": 1200, "esmc_600m": 2600,
    "saprot_35m": 700, "saprot_650m": 3000, "saprot_650m_af2": 3000,
    "saprot_1.3b": 6000,
}
_DTYPE_SCALE = {"fp32": 1.0, "fp16": 0.5, "bf16": 0.5}
_PLM_BASE_OVERHEAD_MB = 2000   # torch/python/foldseek + forward activations


def estimate_plm_footprint_mb(
    esmc_model: str, saprot_model: str, dtype: str = "fp32",
    *, on_gpu: Optional[bool] = None,
) -> int:
    """Rough peak HOST-RAM estimate (MB) for Stage-2 precompute.

    Experts run one-at-a-time, so peak ≈ max(expert footprints)×dtype-scale, plus
    base overhead. On GPU the weights live in VRAM (host holds a load copy +
    activations); on CPU the model + activations sit in host RAM (~2× the weights).
    ``on_gpu`` auto-detected when None.
    """
    if on_gpu is None:
        n_gpus, _, _ = detect_n_gpus()
        on_gpu = n_gpus > 0
    scale = _DTYPE_SCALE.get(dtype, 1.0)
    peak_model = max(_PLM_FOOTPRINT_MB.get(esmc_model, 2600),
                     _PLM_FOOTPRINT_MB.get(saprot_model, 6000)) * scale
    factor = 1.0 if on_gpu else 2.0
    return int(peak_model * factor) + _PLM_BASE_OVERHEAD_MB


def warn_if_plm_mem_tight(
    esmc_model: str, saprot_model: str, dtype: str = "fp32",
    *, margin: float = 0.85, log: bool = True, on_gpu: Optional[bool] = None,
) -> bool:
    """LOG a WARNING (never changes behavior) if the configured PLM footprint risks
    exceeding the detected memory budget. Returns True when tight.

    Suggests a smaller ``SAPROT_MODEL`` variant and/or ``--plm_dtype fp16`` /
    ``PLM_DTYPE=fp16``. ``on_gpu`` (auto-detected when None) selects the host-RAM
    factor. Call at Stage-2 startup (before models load).
    """
    budget, src = detect_available_mem_mb()
    est = estimate_plm_footprint_mb(esmc_model, saprot_model, dtype, on_gpu=on_gpu)
    if budget <= 0:
        if log:
            LOGGER.info("PLM memory estimate ~%d MB (esmc=%s saprot=%s dtype=%s); "
                        "budget undetectable.", est, esmc_model, saprot_model, dtype)
        return False
    tight = est > margin * budget
    if not log:
        return tight
    if tight:
        suggest = []
        if saprot_model not in ("saprot_35m", "saprot_650m"):
            suggest.append("a smaller SAPROT_MODEL (e.g. saprot_650m / saprot_35m)")
        if dtype == "fp32":
            suggest.append("PLM_DTYPE=fp16 (--plm_dtype fp16; ~halves PLM memory)")
        LOGGER.warning(
            "PLM memory may be TIGHT: est ~%d MB (esmc=%s, saprot=%s, dtype=%s) vs "
            "budget ~%d MB [%s]. Consider %s. (An OOM/SIGKILL in Stage 2 is likely this.)",
            est, esmc_model, saprot_model, dtype, budget, src,
            " or ".join(suggest) or "more --mem")
    else:
        LOGGER.info("PLM memory OK: est ~%d MB vs budget ~%d MB [%s].", est, budget, src)
    return tight


__all__ = [
    "ResourceInfo",
    "detect_n_cpus",
    "detect_n_gpus",
    "detect_resources",
    "configure_torch_threads",
    "pool_workers",
    "detect_available_mem_mb",
    "estimate_plm_footprint_mb",
    "warn_if_plm_mem_tight",
]
