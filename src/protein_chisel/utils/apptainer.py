"""Subprocess wrapper around `apptainer exec`.

Single source of truth for how we shell into containers. Tools that need
PyRosetta / ESM-C / Rosetta call these helpers instead of building command
lines themselves.

Conventions:
- Always pass --bind for the protein_chisel checkout so the package and
  its src/ tree are importable as `protein_chisel` inside the container.
- Always set PYTHONPATH=/code/src (where /code is the bind target).
- Always pass --nv for GPU images (esmc.sif). Harmless for non-GPU images
  but apptainer 1.4 warns if --nv is used and no GPU is available; tools
  that need only CPU should pass nv=False.
- Use AbsolutePath strings everywhere; relative paths get confused under
  bind mounts.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from protein_chisel.paths import (
    ESMC_SIF,
    METAL3D_SIF,
    PYROSETTA_SIF,
    ROSETTA_SIF,
    UNIVERSAL_SIF,
)


PROTEIN_CHISEL_ROOT = Path(__file__).resolve().parents[3]  # repo root
PROTEIN_CHISEL_SRC = PROTEIN_CHISEL_ROOT / "src"


@dataclass
class ApptainerResult:
    """Result of running a command inside a container."""

    returncode: int
    stdout: str
    stderr: str
    command: list[str]


def _user_site_for_python() -> Optional[str]:
    """Return the host's per-user site-packages dir for the *outer* python.

    The cluster's pyrosetta.sif sets ``PYTHONNOUSERSITE=1`` and ships only a
    minimal interpreter, so things like pytest installed at user-site are
    invisible. We expose them explicitly via PYTHONPATH when a caller asks.
    """
    import sys

    candidate = Path(
        f"/home/{os.environ.get('USER', 'woodbuse')}/.local/lib/"
        f"python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    return str(candidate) if candidate.is_dir() else None


@dataclass
class ApptainerCall:
    """One configured call into a container.

    Construct once, call repeatedly. The default builder injects:
    - bind for the protein_chisel checkout,
    - PYTHONPATH so `import protein_chisel` works,
    - --nv if `nv=True`,
    - host's user-site on PYTHONPATH if `with_user_site=True` (lets pytest
      and other host-installed tools work inside containers that set
      PYTHONNOUSERSITE=1).
    """

    sif: Path
    nv: bool = False
    binds: list[tuple[str, str]] = field(default_factory=list)  # [(host, guest)]
    env: dict[str, str] = field(default_factory=dict)
    repo_bind_target: str = "/code"
    with_user_site: bool = False
    # Paths to PREPEND to the container's existing PYTHONPATH (preserved
    # via APPTAINERENV_PYTHONPATH semantics — we wrap the python invocation
    # with a shell that does the prepend).
    container_pythonpath_keepers: tuple[str, ...] = ()

    def with_bind(self, host: str | Path, guest: Optional[str] = None) -> "ApptainerCall":
        """Add a bind. If `guest` is omitted, use the same path on both sides."""
        host = str(Path(host).resolve())
        guest = guest or host
        return ApptainerCall(
            sif=self.sif,
            nv=self.nv,
            binds=self.binds + [(host, guest)],
            env=dict(self.env),
            repo_bind_target=self.repo_bind_target,
            with_user_site=self.with_user_site,
            container_pythonpath_keepers=self.container_pythonpath_keepers,
        )

    def with_env(self, **kw: str) -> "ApptainerCall":
        merged = dict(self.env)
        merged.update({k: str(v) for k, v in kw.items()})
        return ApptainerCall(
            sif=self.sif,
            nv=self.nv,
            binds=list(self.binds),
            env=merged,
            repo_bind_target=self.repo_bind_target,
            with_user_site=self.with_user_site,
            container_pythonpath_keepers=self.container_pythonpath_keepers,
        )

    def with_pytest(self) -> "ApptainerCall":
        """Enable host user-site on PYTHONPATH so pytest is importable."""
        return ApptainerCall(
            sif=self.sif,
            nv=self.nv,
            binds=list(self.binds),
            env=dict(self.env),
            repo_bind_target=self.repo_bind_target,
            with_user_site=True,
            container_pythonpath_keepers=self.container_pythonpath_keepers,
        )

    def build_command(self, argv: Sequence[str]) -> list[str]:
        cmd = ["apptainer", "exec"]
        if self.nv:
            cmd.append("--nv")

        # Always bind the repo so `import protein_chisel` works.
        repo_host = str(PROTEIN_CHISEL_ROOT.resolve())
        cmd += ["--bind", f"{repo_host}:{self.repo_bind_target}"]
        for host, guest in self.binds:
            cmd += ["--bind", f"{host}:{guest}"]

        # Default PYTHONPATH. Prepend our paths to whatever the container
        # ships with (e.g. pyrosetta.sif sets PYTHONPATH=/pyrosetta) by
        # using APPTAINERENV_PREPEND_PATH semantics — we just pre-build the
        # value and pass it through, including any explicitly-listed
        # `container_pythonpath_keepers` from sub-class configurations.
        pythonpath_parts = [f"{self.repo_bind_target}/src"]
        if self.with_user_site:
            us = _user_site_for_python()
            if us:
                pythonpath_parts.append(us)
        pythonpath_parts.extend(self.container_pythonpath_keepers)
        env = {"PYTHONPATH": ":".join(pythonpath_parts)}
        if self.with_user_site:
            # Some sifs set PYTHONNOUSERSITE=1; clear it so user-site picks up.
            env["PYTHONNOUSERSITE"] = ""
        env.update(self.env)
        for k, v in env.items():
            cmd += ["--env", f"{k}={v}"]

        cmd.append(str(self.sif))
        cmd += list(argv)
        return cmd

    def run(
        self,
        argv: Sequence[str],
        check: bool = True,
        capture_output: bool = True,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ApptainerResult:
        cmd = self.build_command(argv)
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=capture_output,
            input=input,
            timeout=timeout,
        )
        result = ApptainerResult(
            returncode=proc.returncode,
            stdout=proc.stdout if capture_output else "",
            stderr=proc.stderr if capture_output else "",
            command=cmd,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"apptainer exec failed (exit {proc.returncode}):\n"
                f"  cmd: {shlex.join(cmd)}\n"
                f"  stderr: {result.stderr.strip()[-8000:]}"
            )
        return result

    def run_python(
        self,
        script_path: str | Path,
        args: Iterable[str] = (),
        **kw,
    ) -> ApptainerResult:
        """Run `python script_path arg1 arg2 ...` inside the container.

        The container's runscript is `exec python "$@"` for esmc.sif, so we
        pass `python` here for explicitness across other sifs that may
        differ.
        """
        argv = ["python", str(script_path), *map(str, args)]
        return self.run(argv, **kw)

    def run_python_inline(self, code: str, **kw) -> ApptainerResult:
        """Run `python -c '<code>'` inside the container."""
        return self.run(["python", "-c", code], **kw)


# ---------------------------------------------------------------------------
# Pre-configured calls for the standard sifs
# ---------------------------------------------------------------------------


def esmc_call(nv: bool = True) -> ApptainerCall:
    """Default call into esmc.sif (GPU on, HF caches bound).

    Binds the ESM-C and SaProt HF caches by default so `HF_HOME` works
    transparently for either model family.
    """
    return (
        ApptainerCall(sif=ESMC_SIF, nv=nv)
        .with_bind("/net/databases/huggingface/esmc")
        .with_bind("/net/databases/huggingface/saprot")
    )


def pyrosetta_call() -> ApptainerCall:
    # pyrosetta.sif ships PYTHONPATH=/pyrosetta in its %environment. Preserve it.
    return ApptainerCall(
        sif=PYROSETTA_SIF,
        nv=False,
        container_pythonpath_keepers=("/pyrosetta",),
    )


def rosetta_call() -> ApptainerCall:
    return ApptainerCall(sif=ROSETTA_SIF, nv=False)


def metal3d_call(nv: bool = True) -> ApptainerCall:
    return ApptainerCall(sif=METAL3D_SIF, nv=nv)


def universal_call(nv: bool = False) -> ApptainerCall:
    return ApptainerCall(sif=UNIVERSAL_SIF, nv=nv)


def mlfold_call(nv: bool = True) -> ApptainerCall:
    """mlfold.sif — used to run LigandMPNN."""
    from protein_chisel.paths import MLFOLD_SIF
    return ApptainerCall(sif=MLFOLD_SIF, nv=nv)


# ---------------------------------------------------------------------------
# Convenience: detect whether we're already inside a container
# ---------------------------------------------------------------------------


def in_apptainer() -> bool:
    """True if the current process is running inside an apptainer/singularity image."""
    return "APPTAINER_NAME" in os.environ or "SINGULARITY_NAME" in os.environ
