"""Metal3D wrapper — predict metal-binding sites and compare to actual REMARK 666 metals.

Metal3D (https://github.com/lcbc-epfl/metal-site-prediction, Dürr et al.
2023) predicts metal-binding probability on a 3D voxel grid around a
protein. We expose:

- ``actual_metals``     : HETATM metals in the input PDB (Zn, Fe, Mg, Mn,
                          Cu, Ca, Ni, Co, Cd, Hg, Mo).
- ``predicted_sites``   : clustered probability peaks (x, y, z, p) from
                          Metal3D's ``find_unique_sites``.
- ``actual_metal_max_prob_within_4A`` : per actual metal, the max
  predicted probability within 4 Å — sanity check on "would Metal3D have
  placed a metal where one actually is?"

**Where everything lives:**
- ``external/metal-site-prediction/`` (git submodule): pinned source.
  Read-only on disk; for introspection / patching at design time.
- ``metal3d.sif`` (``/net/software/containers/pipelines/metal3d.sif``):
  runtime image with torch/moleculekit/etc. and weights at
  ``/opt/metal-site-prediction/Metal3D/weights/``. **All inference runs
  here.**
- ``scripts/run_metal3d.py``: self-relaunches into metal3d.sif when
  invoked from outside a container. Handles CPU-safe voxelization,
  KDTree-vectorized probability, batched inference, JSON metadata +
  probe-PDB output.

**GPU strongly recommended.** Metal3D's CNN runs many 32³ voxel cubes
per protein; CPU inference is unworkably slow (~10× longer). The
runner supports ``--device cpu`` but expect minutes-to-hours per PDB.
For real runs, route through sbatch on a GPU partition.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


LOGGER = logging.getLogger("protein_chisel.metal3d_score")


METAL_ELEMENTS = {"ZN", "FE", "MG", "MN", "CU", "CA", "NI", "CO", "CD", "HG", "MO"}


# Located by `paths.py` — vendored at scripts/run_metal3d.py.
def _runner_script() -> Path:
    from protein_chisel.paths import METAL3D_RUNNER_SCRIPT
    return METAL3D_RUNNER_SCRIPT


def metal3d_source_dir() -> Path:
    """Path to the vendored Metal3D source (the git submodule).

    Useful if you want to inspect or patch the upstream code without
    going inside metal3d.sif. The runtime path inside metal3d.sif is
    /opt/metal-site-prediction/Metal3D and is NOT this directory.
    """
    from protein_chisel.paths import METAL3D_SOURCE_DIR
    return METAL3D_SOURCE_DIR


@dataclass
class Metal3DResult:
    actual_metals: list[dict] = field(default_factory=list)
    predicted_sites: list[tuple[float, float, float, float]] = field(default_factory=list)
    actual_metal_max_prob_within_4A: dict[str, float] = field(default_factory=dict)
    n_actual_metals: int = 0
    n_predicted_sites: int = 0
    top_site_probability: float = 0.0
    runner_metadata: dict = field(default_factory=dict)
    raw_results_path: Optional[str] = None

    def to_dict(self, prefix: str = "metal3d__") -> dict[str, float | int]:
        out: dict[str, float | int] = {
            f"{prefix}n_actual_metals": self.n_actual_metals,
            f"{prefix}n_predicted_sites": self.n_predicted_sites,
            f"{prefix}top_site_probability": self.top_site_probability,
        }
        for label, p in self.actual_metal_max_prob_within_4A.items():
            out[f"{prefix}actual__{label}__max_pred_prob_4A"] = p
        return out


def find_actual_metals(pdb_path: str | Path) -> list[dict]:
    """Scan HETATM records for metal atoms (Zn, Fe, Mg, ...)."""
    from protein_chisel.io.pdb import parse_atom_record

    metals: list[dict] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            atom = parse_atom_record(line)
            if atom is None or atom.record != "HETATM":
                continue
            if atom.element.upper() in METAL_ELEMENTS:
                metals.append({
                    "element": atom.element.upper(),
                    "name": atom.name,
                    "chain": atom.chain,
                    "resno": atom.res_seq,
                    "x": atom.x, "y": atom.y, "z": atom.z,
                })
    return metals


def metal3d_score(
    pdb_path: str | Path,
    out_dir: Optional[str | Path] = None,
    metalbinding_only: bool = True,
    pthreshold: float = 0.10,
    cluster_threshold: float = 7.0,
    write_combined_pdbs: bool = False,
    keep_outputs: bool = False,
    timeout: float = 1800.0,
    runner_script: Optional[str | Path] = None,
    extra_runner_args: tuple[str, ...] = (),
) -> Metal3DResult:
    """Run Metal3D on a PDB and compare predicted sites to actual metals.

    The Metal3D inference itself runs via the vendored
    ``scripts/run_metal3d.py``, which auto-relaunches inside
    ``metal3d.sif`` if the current environment lacks the deps. This
    wrapper just shells out to it, parses the JSON, and assembles
    comparison metrics.

    Args:
        pdb_path: input PDB.
        out_dir: directory for retained Metal3D outputs (probe PDB, JSON,
            metadata). Defaults to a tempdir; pass a path to keep outputs.
        metalbinding_only: when True, scan only likely metal-binding
            sidechains (faster). When False, scan all protein residues.
        pthreshold: minimum probability for a predicted site (Metal3D's
            default).
        cluster_threshold: max distance for clustering nearby high-prob
            grid points into one predicted site.
        write_combined_pdbs: if True, write a PDB with predicted probes
            appended (useful for visualization).
        keep_outputs: keep the workspace after parsing.
        timeout: subprocess timeout seconds.
        runner_script: explicit path to run_metal3d.py (default: the
            vendored copy under scripts/).
        extra_runner_args: extra CLI flags to forward to run_metal3d.py.
    """
    pdb_path = Path(pdb_path).resolve()
    actual = find_actual_metals(pdb_path)

    workspace = Path(out_dir).resolve() if out_dir else Path(tempfile.mkdtemp(prefix="chisel_metal3d_"))
    output_dir = workspace / "metal3d_out"
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    script = Path(runner_script) if runner_script else _runner_script()
    if not script.is_file():
        raise FileNotFoundError(f"run_metal3d.py not found at {script}")

    cmd = [
        "python3", str(script),
        "--pdb", str(pdb_path),
        "--output-dir", str(output_dir),
        "--pthreshold", str(float(pthreshold)),
        "--cluster-threshold", str(float(cluster_threshold)),
        "--force",  # overwrite any existing dir
        "--execution-mode", "auto",  # let runner self-relaunch into metal3d.sif if needed
    ]
    if not metalbinding_only:
        cmd.append("--all-protein")
    if write_combined_pdbs:
        cmd.append("--write-combined-pdbs")
    cmd.extend(extra_runner_args)

    LOGGER.info("running metal3d: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_metal3d.py exited {proc.returncode}\n"
            f"stdout (tail): {proc.stdout[-1500:]}\n"
            f"stderr (tail): {proc.stderr[-1500:]}"
        )

    results_path = output_dir / "metal3d_results.json"
    metadata_path = output_dir / "metal3d_run_metadata.json"
    if not results_path.exists():
        raise RuntimeError(
            f"Metal3D produced no results JSON at {results_path}\n"
            f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
        )

    with results_path.open() as fh:
        results = json.load(fh)
    metadata: dict = {}
    if metadata_path.exists():
        with metadata_path.open() as fh:
            metadata = json.load(fh)

    # results is keyed by source PDB filename; we passed exactly one PDB
    # so there's a single entry. Extract its sites.
    predicted_sites: list[tuple[float, float, float, float]] = []
    top_p = 0.0
    for entry in results.values():
        for site in entry.get("sites", []):
            predicted_sites.append(
                (float(site["x"]), float(site["y"]), float(site["z"]), float(site["p"]))
            )
            top_p = max(top_p, float(site["p"]))

    # Per actual metal, max predicted-probability within 4 Å
    per_actual: dict[str, float] = {}
    if predicted_sites and actual:
        pred = np.asarray([(s[0], s[1], s[2]) for s in predicted_sites])
        probs = np.asarray([s[3] for s in predicted_sites])
        for m in actual:
            label = f"{m['chain']}_{m['resno']}_{m['name']}"
            d = np.linalg.norm(pred - np.array([m["x"], m["y"], m["z"]]), axis=-1)
            mask = d <= 4.0
            per_actual[label] = float(probs[mask].max()) if mask.any() else 0.0

    raw_path = str(results_path) if keep_outputs or out_dir else None
    if not keep_outputs and not out_dir:
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)

    return Metal3DResult(
        actual_metals=actual,
        predicted_sites=predicted_sites,
        actual_metal_max_prob_within_4A=per_actual,
        n_actual_metals=len(actual),
        n_predicted_sites=len(predicted_sites),
        top_site_probability=top_p,
        runner_metadata=metadata,
        raw_results_path=raw_path,
    )


def metal3d_score_batch(
    pdb_paths: list[str | Path],
    out_dir: str | Path,
    metalbinding_only: bool = True,
    pthreshold: float = 0.10,
    cluster_threshold: float = 7.0,
    write_combined_pdbs: bool = False,
    batch_size: int = 128,
    timeout_per_pdb: float = 120.0,
    runner_script: Optional[str | Path] = None,
    extra_runner_args: tuple[str, ...] = (),
) -> dict[str, Metal3DResult]:
    """Run Metal3D on many PDBs in a single container invocation.

    Big efficiency win for N>1: container startup (~5s) and model load
    (~2-3s) are paid ONCE rather than N times. For 100 PDBs this can cut
    wall-time by ~25-40% vs. calling ``metal3d_score`` per pose.

    Args:
        pdb_paths: list of input PDBs. The runner accepts a mix of .pdb,
            .cif, .mmcif, and .gz variants.
        out_dir: workspace directory. The runner writes a single
            ``metal3d_results.json`` keyed by source filename, plus one
            probe PDB per input.
        batch_size: model-forward batch size. On a single A4000 (16 GB)
            128 is comfortable for a 32³ voxel cube; larger GPUs can
            push to 256-512 for slightly more throughput.
        timeout_per_pdb: per-PDB timeout multiplier. Total subprocess
            timeout = N * timeout_per_pdb (with a floor of 600 s).

    Returns:
        ``{<input pdb path>: Metal3DResult}``. Result keys are the
        absolute paths of the input PDBs.
    """
    if not pdb_paths:
        return {}
    pdb_paths_resolved = [Path(p).resolve() for p in pdb_paths]

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_dir = out_dir / "metal3d_out"

    script = Path(runner_script) if runner_script else _runner_script()
    if not script.is_file():
        raise FileNotFoundError(f"run_metal3d.py not found at {script}")

    # Write a list-of-paths file the runner accepts via --pdb-list. Reduces
    # CLI length when N is large.
    list_path = out_dir / "metal3d_inputs.txt"
    list_path.write_text("\n".join(str(p) for p in pdb_paths_resolved) + "\n")

    cmd = [
        "python3", str(script),
        "--pdb-list", str(list_path),
        "--output-dir", str(output_dir),
        "--pthreshold", str(float(pthreshold)),
        "--cluster-threshold", str(float(cluster_threshold)),
        "--batch-size", str(int(batch_size)),
        "--force",
        "--execution-mode", "auto",
    ]
    if not metalbinding_only:
        cmd.append("--all-protein")
    if write_combined_pdbs:
        cmd.append("--write-combined-pdbs")
    cmd.extend(extra_runner_args)

    timeout = max(600.0, len(pdb_paths_resolved) * timeout_per_pdb)
    LOGGER.info("metal3d batch (%d pdbs): %s", len(pdb_paths_resolved), " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_metal3d.py exited {proc.returncode}\n"
            f"stdout (tail): {proc.stdout[-1500:]}\n"
            f"stderr (tail): {proc.stderr[-1500:]}"
        )

    results_path = output_dir / "metal3d_results.json"
    if not results_path.exists():
        raise RuntimeError(f"Metal3D produced no results JSON at {results_path}")

    with results_path.open() as fh:
        json_results = json.load(fh)

    # The runner keys results by the input filename's basename. Build a
    # lookup from basename → full path for readback alignment.
    name_to_path = {p.name: str(p) for p in pdb_paths_resolved}
    by_path: dict[str, Metal3DResult] = {}
    for source_name, entry in json_results.items():
        full_path = name_to_path.get(source_name) or entry.get("input") or source_name
        actual = find_actual_metals(full_path) if Path(full_path).is_file() else []

        sites = [
            (float(s["x"]), float(s["y"]), float(s["z"]), float(s["p"]))
            for s in entry.get("sites", [])
        ]
        top_p = max((s[3] for s in sites), default=0.0)
        per_actual: dict[str, float] = {}
        if sites and actual:
            pred = np.asarray([(s[0], s[1], s[2]) for s in sites])
            probs = np.asarray([s[3] for s in sites])
            for m in actual:
                label = f"{m['chain']}_{m['resno']}_{m['name']}"
                d = np.linalg.norm(pred - np.array([m["x"], m["y"], m["z"]]), axis=-1)
                mask = d <= 4.0
                per_actual[label] = float(probs[mask].max()) if mask.any() else 0.0

        by_path[full_path] = Metal3DResult(
            actual_metals=actual,
            predicted_sites=sites,
            actual_metal_max_prob_within_4A=per_actual,
            n_actual_metals=len(actual),
            n_predicted_sites=len(sites),
            top_site_probability=top_p,
            runner_metadata={},
            raw_results_path=str(results_path),
        )

    return by_path


__all__ = [
    "METAL_ELEMENTS",
    "Metal3DResult",
    "find_actual_metals",
    "metal3d_score",
    "metal3d_score_batch",
    "metal3d_source_dir",
]
