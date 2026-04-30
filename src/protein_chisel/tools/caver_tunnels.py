"""CAVER 3 tunnel-detection wrapper.

CAVER (https://www.caver.cz/) finds substrate-access tunnels from a
buried site to the protein surface. Useful for confirming the active
site is *reachable* — a beautiful pocket with no tunnel is a non-
functional design.

Status note: CAVER is a Java tool (``CaverCommandLine_3.x.jar``) requiring
a JRE 1.7+. It's NOT installed on the cluster. This wrapper:

1. Locates the jar via env var ``CAVER_JAR`` or an explicit
   ``caver_jar=`` argument.
2. Locates ``java`` via ``shutil.which("java")``.
3. Generates a minimal CAVER config from the user-supplied starting point
   (e.g. an active-site centroid).
4. Runs CAVER, then parses ``tunnel_characteristics.csv`` for tunnel
   properties: length, throughput, bottleneck radius, curvature.

To enable: download CAVER 3.0.3 (https://www.caver.cz/index.php?sid=199),
extract somewhere (e.g. /net/software/caver/CaverCommandLine_3.0.3.jar),
then ``export CAVER_JAR=/path/to/CaverCommandLine_3.0.3.jar`` before
calling.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


LOGGER = logging.getLogger("protein_chisel.caver_tunnels")


CAVER_DEFAULT_PROBE_RADIUS = 0.9     # Å — minimum tunnel radius
CAVER_DEFAULT_SHELL_RADIUS = 3.0     # Å — outer cluster radius
CAVER_DEFAULT_CLUSTERING_THRESHOLD = 3.5


@dataclass
class Tunnel:
    """One CAVER tunnel record from ``tunnel_characteristics.csv``.

    Note: CAVER does not include ``avg_radius`` in this CSV (it lives in
    ``tunnel_profiles.csv`` per the user guide). We don't parse it.
    """
    tunnel_idx: int                  # CAVER's "Tunnel" column (1-indexed within cluster)
    cluster: int                     # CAVER's "Tunnel cluster" column
    throughput: float = 0.0
    cost: float = 0.0
    length: float = 0.0
    bottleneck_radius: float = 0.0
    curvature: float = 0.0


@dataclass
class CAVERResult:
    tunnels: list[Tunnel] = field(default_factory=list)
    n_tunnels: int = 0
    best_throughput: float = 0.0
    best_bottleneck_radius: float = 0.0

    def to_dict(self, prefix: str = "caver__") -> dict[str, float | int]:
        out: dict[str, float | int] = {
            f"{prefix}n_tunnels": self.n_tunnels,
            f"{prefix}best_throughput": self.best_throughput,
            f"{prefix}best_bottleneck_radius": self.best_bottleneck_radius,
        }
        for i, t in enumerate(self.tunnels[:3]):
            tag = "" if i == 0 else f"_{i}"
            out[f"{prefix}t{tag}__length"] = t.length
            out[f"{prefix}t{tag}__bottleneck_radius"] = t.bottleneck_radius
            out[f"{prefix}t{tag}__throughput"] = t.throughput
            out[f"{prefix}t{tag}__curvature"] = t.curvature
        return out


def find_caver_jar(explicit: Optional[str | Path] = None) -> str:
    if explicit:
        p = str(Path(explicit).resolve())
        if Path(p).is_file():
            return p
    env = os.environ.get("CAVER_JAR")
    if env and Path(env).is_file():
        return env
    raise RuntimeError(
        "CAVER jar not found. Download CAVER 3 from https://www.caver.cz/, "
        "extract, and set CAVER_JAR=/path/to/CaverCommandLine_3.x.jar (or "
        "pass caver_jar= to caver_tunnels)."
    )


def caver_tunnels(
    pdb_path: str | Path,
    starting_point: tuple[float, float, float],
    caver_jar: Optional[str | Path] = None,
    java_bin: Optional[str] = None,
    probe_radius: float = CAVER_DEFAULT_PROBE_RADIUS,
    shell_radius: float = CAVER_DEFAULT_SHELL_RADIUS,
    out_dir: Optional[str | Path] = None,
    keep_outputs: bool = False,
    timeout: float = 1800.0,
) -> CAVERResult:
    """Run CAVER 3 from a starting point and parse tunnel characteristics.

    Args:
        pdb_path: input PDB.
        starting_point: (x, y, z) Å — typically the centroid of the
            ligand or the catalytic atom you want tunnels FROM.
        caver_jar: explicit path to CaverCommandLine jar.
        java_bin: explicit path to ``java`` (defaults to PATH).
        probe_radius: minimum allowed tunnel radius. Lower = finds
            tighter tunnels.
        shell_radius: outer radius for the algorithm's tunnel cluster.
    """
    jar = find_caver_jar(caver_jar)
    java = java_bin or shutil.which("java")
    if not java:
        raise RuntimeError("`java` not found on PATH. Install JRE 1.7+.")

    pdb_path = Path(pdb_path).resolve()
    workspace = Path(out_dir).resolve() if out_dir else Path(tempfile.mkdtemp(prefix="chisel_caver_"))
    workspace.mkdir(parents=True, exist_ok=True)

    # Copy input PDB into ``input_pdb/`` (CAVER expects a directory of PDBs).
    input_dir = workspace / "input_pdb"
    input_dir.mkdir(exist_ok=True)
    (input_dir / pdb_path.name).write_bytes(pdb_path.read_bytes())

    # Minimal CAVER config: starting point + thresholds.
    config_path = workspace / "caver_config.txt"
    sx, sy, sz = starting_point
    config_path.write_text(
        f"starting_point_coordinates {sx} {sy} {sz}\n"
        f"probe_radius {probe_radius}\n"
        f"shell_radius {shell_radius}\n"
        f"clustering_threshold {CAVER_DEFAULT_CLUSTERING_THRESHOLD}\n"
        # Default values for everything else
    )

    cmd = [
        java, "-jar", jar,
        "-pdb", str(input_dir),
        "-conf", str(config_path),
        "-out", str(workspace / "caver_out"),
    ]
    LOGGER.info("running CAVER: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CAVER failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout[-1500:]}\nstderr: {proc.stderr[-1500:]}"
        )

    # CAVER writes <out>/<pdb-stem>/analysis/tunnel_characteristics.csv
    tunnels_csv = (
        workspace / "caver_out" / pdb_path.stem / "analysis" /
        "tunnel_characteristics.csv"
    )
    tunnels = _parse_caver_csv(tunnels_csv) if tunnels_csv.exists() else []

    if not keep_outputs and not out_dir:
        shutil.rmtree(workspace, ignore_errors=True)

    sorted_t = sorted(tunnels, key=lambda t: t.throughput, reverse=True)
    return CAVERResult(
        tunnels=sorted_t,
        n_tunnels=len(sorted_t),
        best_throughput=max((t.throughput for t in sorted_t), default=0.0),
        best_bottleneck_radius=max((t.bottleneck_radius for t in sorted_t), default=0.0),
    )


def starting_point_from_ligand(
    pdb_path: str | Path,
    ligand_chain: Optional[str] = None,
    ligand_resname: Optional[str] = None,
    ligand_resno: Optional[int] = None,
) -> tuple[float, float, float]:
    """Convenience: ligand-centroid as the CAVER starting point.

    Identifies the ligand by full ``(chain, resname, resno)`` so multi-copy
    ligands or cofactors don't get averaged together. When any of the three
    fields is None, falls back to the first non-water HETATM in the PDB.
    """
    from protein_chisel.io.pdb import find_ligand, parse_atom_record

    if ligand_chain is None or ligand_resname is None or ligand_resno is None:
        info = find_ligand(pdb_path)
        if info is None:
            raise RuntimeError(f"no ligand HETATMs in {pdb_path}")
        ligand_chain = ligand_chain or info[0]
        ligand_resname = ligand_resname or info[1]
        ligand_resno = ligand_resno if ligand_resno is not None else info[2]

    coords: list[list[float]] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            atom = parse_atom_record(line)
            if atom is None or atom.record != "HETATM":
                continue
            if (atom.chain, atom.res_name, atom.res_seq) != (ligand_chain, ligand_resname, int(ligand_resno)):
                continue
            if atom.element == "H" or atom.name.startswith("H"):
                continue
            coords.append([atom.x, atom.y, atom.z])
    if not coords:
        raise RuntimeError(
            f"no heavy atoms for ligand ({ligand_chain}, {ligand_resname}, {ligand_resno}) in {pdb_path}"
        )
    cen = np.mean(coords, axis=0)
    return float(cen[0]), float(cen[1]), float(cen[2])


def _parse_caver_csv(csv_path: Path) -> list[Tunnel]:
    """Parse CAVER's tunnel_characteristics.csv.

    Real columns per the CAVER user guide: ``Tunnel cluster``, ``Tunnel``,
    ``Throughput``, ``Cost``, ``Length``, ``Bottleneck radius``, ``Curvature``.
    Older /alternate column names are tolerated as fallbacks.
    """
    import csv

    def _f(row: dict, *keys: str, default: float = 0.0) -> float:
        for k in keys:
            v = row.get(k)
            if v not in (None, ""):
                try:
                    return float(v)
                except ValueError:
                    pass
        return default

    out: list[Tunnel] = []
    with open(csv_path, "r") as fh:
        rdr = csv.DictReader(fh)
        for i, row in enumerate(rdr):
            try:
                out.append(Tunnel(
                    tunnel_idx=int(_f(row, "Tunnel", default=i + 1)),
                    cluster=int(_f(row, "Tunnel cluster", "Cluster", default=0)),
                    throughput=_f(row, "Throughput"),
                    cost=_f(row, "Cost"),
                    length=_f(row, "Length"),
                    bottleneck_radius=_f(row, "Bottleneck radius"),
                    curvature=_f(row, "Curvature"),
                ))
            except (ValueError, KeyError) as e:
                LOGGER.warning("skip CAVER row %d: %s", i, e)
                continue
    return out


__all__ = [
    "CAVERResult",
    "Tunnel",
    "caver_tunnels",
    "find_caver_jar",
    "starting_point_from_ligand",
]
