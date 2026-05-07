"""fpocket wrapper — pocket detection and geometry.

fpocket is a CLI tool (https://github.com/Discngine/fpocket). On clusters
without it pre-installed, build the static binary and pass its path via
``--fpocket_exe`` (or the ``FPOCKET`` env var).

This wrapper:
1. Finds the fpocket executable.
2. Runs ``fpocket -f <pdb>`` in a tempdir.
3. Parses the ``<pdb>_info.txt`` output into a list of pockets with
   their per-pocket properties (volume, hydrophobicity, druggability, etc.).

Status note: as of this commit fpocket is NOT installed on the cluster.
The wrapper is functional once you point it at a binary; without one, it
raises an informative ``RuntimeError``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger("protein_chisel.fpocket_run")


@dataclass
class Pocket:
    pocket_idx: int
    score: float = 0.0
    druggability_score: float = 0.0
    n_alpha_spheres: int = 0
    total_sasa: float = 0.0
    polar_sasa: float = 0.0
    apolar_sasa: float = 0.0
    volume: float = 0.0
    mean_local_hydrophobic_density: float = 0.0
    mean_alpha_sphere_radius: float = 0.0
    mean_alpha_sphere_solvent_acc: float = 0.0
    apolar_alpha_sphere_proportion: float = 0.0
    hydrophobicity_score: float = 0.0
    volume_score: float = 0.0
    polarity_score: float = 0.0
    charge_score: float = 0.0
    proportion_polar_atoms: float = 0.0
    alpha_sphere_density: float = 0.0
    cent_of_mass_alpha_sphere_max_dist: float = 0.0
    flexibility: float = 0.0


@dataclass
class FpocketResult:
    pockets: list[Pocket] = field(default_factory=list)
    n_pockets: int = 0
    largest_pocket_volume: float = 0.0
    most_druggable_score: float = 0.0

    def to_dict(self, prefix: str = "fpocket__") -> dict[str, float | int]:
        out: dict[str, float | int] = {
            f"{prefix}n_pockets": self.n_pockets,
            f"{prefix}largest_pocket_volume": self.largest_pocket_volume,
            f"{prefix}most_druggable_score": self.most_druggable_score,
        }
        # Top pocket gets a flat namespace; subsequent get _<i>__.
        for i, p in enumerate(self.pockets[:5]):
            tag = "" if i == 0 else f"_{i}"
            out[f"{prefix}p{tag}__druggability"] = p.druggability_score
            out[f"{prefix}p{tag}__volume"] = p.volume
            out[f"{prefix}p{tag}__hydrophobicity"] = p.hydrophobicity_score
            out[f"{prefix}p{tag}__polarity"] = p.polarity_score
            out[f"{prefix}p{tag}__charge"] = p.charge_score
        return out


def find_fpocket_executable(explicit: Optional[str | Path] = None) -> str:
    """Return a usable fpocket binary path.

    Resolution order:
    1. Explicit argument.
    2. ``FPOCKET`` env var.
    3. ``shutil.which("fpocket")`` on $PATH (e.g. inside esmc.sif which
       has it at /usr/local/bin).
    4. Cluster-wide install at ``/net/software/lab/fpocket/bin/fpocket``.
    """
    if explicit:
        p = str(Path(explicit).resolve())
        if Path(p).is_file():
            return p
    env_path = os.environ.get("FPOCKET")
    if env_path and Path(env_path).is_file():
        return env_path
    found = shutil.which("fpocket")
    if found:
        return found
    # Cluster-wide fallback (built from external/fpocket).
    from protein_chisel.paths import FPOCKET_CLUSTER_BIN
    if FPOCKET_CLUSTER_BIN.is_file():
        return str(FPOCKET_CLUSTER_BIN)
    raise RuntimeError(
        "fpocket binary not found. Looked in $FPOCKET, $PATH, and "
        f"{FPOCKET_CLUSTER_BIN}. Install via "
        "`cd external/fpocket && make` or use the esmc.sif which bundles it."
    )


def fpocket_run(
    pdb_path: str | Path,
    fpocket_exe: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    keep_outputs: bool = False,
    timeout: float = 300.0,
) -> FpocketResult:
    """Run fpocket on a PDB and parse pocket geometry.

    Args:
        pdb_path: input PDB.
        fpocket_exe: optional explicit fpocket path.
        out_dir: where the fpocket workspace lives. Defaults to a
            tempdir; pass a path to keep the raw output.
        keep_outputs: if False, the workspace is wiped after parsing.
    """
    pdb_path = Path(pdb_path).resolve()
    exe = find_fpocket_executable(fpocket_exe)

    workspace = Path(out_dir).resolve() if out_dir else Path(tempfile.mkdtemp(prefix="chisel_fpocket_"))
    workspace.mkdir(parents=True, exist_ok=True)

    # fpocket has an upstream buffer overflow on long input-path strings.
    # Copy into the workspace under a short fixed name and invoke it by
    # RELATIVE path from cwd=workspace. Using an absolute /very/long/... path
    # can abort even when the basename itself is short.
    local_pdb = workspace / "design.pdb"
    local_pdb.write_bytes(pdb_path.read_bytes())

    cmd = [exe, "-f", local_pdb.name]
    LOGGER.info("running fpocket: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(workspace), check=False, capture_output=True,
        text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fpocket failed (exit {proc.returncode}):\nstdout: {proc.stdout[-1000:]}\n"
            f"stderr: {proc.stderr[-1000:]}"
        )

    out_subdir = workspace / f"{local_pdb.stem}_out"
    info_txt = out_subdir / f"{local_pdb.stem}_info.txt"
    pockets = _parse_fpocket_info(info_txt) if info_txt.exists() else []

    if not keep_outputs and not out_dir:
        shutil.rmtree(workspace, ignore_errors=True)

    pockets_sorted = sorted(pockets, key=lambda p: p.druggability_score, reverse=True)
    return FpocketResult(
        pockets=pockets_sorted,
        n_pockets=len(pockets_sorted),
        largest_pocket_volume=max((p.volume for p in pockets_sorted), default=0.0),
        most_druggable_score=max((p.druggability_score for p in pockets_sorted), default=0.0),
    )


_INFO_HEADER_RE = re.compile(r"^Pocket\s+(\d+)\s*:")
_INFO_FIELD_RE = re.compile(r"^\s*([A-Za-z_/ ]+):\s*([+-]?[\d.eE+-]+)")


_FIELD_TO_ATTR = {
    "score": "score",
    "druggability score": "druggability_score",
    "number of alpha spheres": "n_alpha_spheres",
    "total sasa": "total_sasa",
    "polar sasa": "polar_sasa",
    "apolar sasa": "apolar_sasa",
    "volume": "volume",
    "mean local hydrophobic density": "mean_local_hydrophobic_density",
    "mean alpha sphere radius": "mean_alpha_sphere_radius",
    "mean alp. sph. solvent access": "mean_alpha_sphere_solvent_acc",
    "apolar alpha sphere proportion": "apolar_alpha_sphere_proportion",
    "hydrophobicity score": "hydrophobicity_score",
    "volume score": "volume_score",
    "polarity score": "polarity_score",
    "charge score": "charge_score",
    "proportion of polar atoms": "proportion_polar_atoms",
    "alpha sphere density": "alpha_sphere_density",
    "cent. of mass - alpha sphere max dist": "cent_of_mass_alpha_sphere_max_dist",
    "flexibility": "flexibility",
}


def _parse_fpocket_info(info_txt: Path) -> list[Pocket]:
    """Parse fpocket's ``<stem>_info.txt`` into Pocket objects."""
    pockets: list[Pocket] = []
    cur: Optional[Pocket] = None
    with open(info_txt, "r") as fh:
        for line in fh:
            m = _INFO_HEADER_RE.match(line)
            if m:
                if cur is not None:
                    pockets.append(cur)
                cur = Pocket(pocket_idx=int(m.group(1)))
                continue
            if cur is None:
                continue
            m = _INFO_FIELD_RE.match(line)
            if not m:
                continue
            key = m.group(1).strip().lower()
            attr = _FIELD_TO_ATTR.get(key)
            if attr is None:
                continue
            try:
                setattr(cur, attr, float(m.group(2)))
            except ValueError:
                pass
    if cur is not None:
        pockets.append(cur)
    return pockets


__all__ = ["FpocketResult", "Pocket", "find_fpocket_executable", "fpocket_run"]
