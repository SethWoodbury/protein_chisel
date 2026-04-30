#!/usr/bin/env python3
"""Batch Metal3D runner with one-folder output packaging.

Originally authored by Aaron Ruder (aruder2) at
``~aruder2/special_scripts/design_filtering/run_metal3d.py``. Vendored
verbatim into protein_chisel; called via subprocess by
``protein_chisel.tools.metal3d_score``.

This is the single Metal3D wrapper entrypoint. It can run Metal3D directly
when the current Python environment already contains the runtime dependencies,
or relaunch itself inside the default Metal3D apptainer image when needed.

All retained outputs for a run live inside one directory:

* ``metal3d_results.json``
* ``metal3d_run_metadata.json``
* one ``*.probes.pdb`` per processed structure
* optionally one ``*.combined.pdb`` per PDB input
* optionally one ``*.cube`` per processed structure

Two calling conventions are supported:

* direct: ``--pdb`` / ``--pdb-list`` with ``--output-dir``
* notebook-style: positional inputs with ``--output-prefix``

Unlike ``metal3D_characterisation.py``, ``--output-prefix`` here is treated as
the path of the output directory itself so that every retained artifact lands
inside one folder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import warnings
from datetime import datetime
from pathlib import Path

DEFAULT_REPO_ROOT = Path(os.environ.get("METAL3D_REPO", "/opt/metal-site-prediction"))
DEFAULT_CONTAINER = Path("/net/software/containers/pipelines/metal3d.sif")
DEFAULT_BINDS = (
    "/home:/home",
    "/net:/net",
    "/etc/passwd:/etc/passwd",
    "/etc/group:/etc/group",
)

SUPPORTED_SUFFIXES = (
    ".mmcif.gz",
    ".pdb.gz",
    ".cif.gz",
    ".mmcif",
    ".pdb",
    ".cif",
)

np = None
torch = None
Model = None
_vox = None
processStructures = None
create_grid_fromBB = None
find_unique_sites = None
get_all_metalbinding_resids = None
get_all_protein_resids = None
get_bb = None
get_probability_mean = None
write_cubefile = None


def normalize_cli_args(argv: list[str]) -> list[str]:
    """Normalize compatibility flags before argparse sees them.

    ``--metalbinding`` is treated as a boolean flag so a following positional
    input is never consumed as its value. Older ``--metalbinding 0/1`` and
    ``--metalbinding=0/1`` spellings are still accepted.
    """
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--metalbinding":
            if index + 1 < len(argv) and argv[index + 1] in {"0", "1"}:
                normalized.append(
                    "--metalbinding" if argv[index + 1] == "1" else "--all-protein"
                )
                index += 2
                continue
            normalized.append("--metalbinding")
            index += 1
            continue
        if arg.startswith("--metalbinding="):
            value = arg.split("=", 1)[1]
            if value == "1":
                normalized.append("--metalbinding")
            elif value == "0":
                normalized.append("--all-protein")
            else:
                raise SystemExit(
                    "--metalbinding only accepts 0 or 1 when given an explicit value."
                )
            index += 1
            continue
        normalized.append(arg)
        index += 1
    return normalized


def resolve_repo_root(args: argparse.Namespace) -> Path:
    """Resolve the Metal3D repository root for direct execution."""
    raw_value = args.repo_root or os.environ.get("METAL3D_REPO")
    return Path(raw_value).expanduser().resolve() if raw_value else DEFAULT_REPO_ROOT


def resolve_weights_path(args: argparse.Namespace, repo_root: Path) -> Path:
    """Resolve the model weights path after repo-root overrides are known."""
    if args.weights:
        return Path(args.weights).expanduser().resolve()
    return (
        repo_root / "Metal3D" / "weights" / "metal_0.5A_v3_d0.2_16Abox.pth"
    ).resolve()


def runtime_available(repo_root: Path) -> bool:
    """Return True when Metal3D can run directly in the current Python env."""
    metal3d_dir = repo_root / "Metal3D"
    if not metal3d_dir.is_dir():
        return False
    for module_name in ("numpy", "torch", "moleculekit", "scipy"):
        if importlib.util.find_spec(module_name) is None:
            return False
    return True


def should_use_apptainer(args: argparse.Namespace, repo_root: Path) -> bool:
    """Decide whether this invocation should relaunch inside apptainer."""
    if os.environ.get("METAL3D_WRAPPER_IN_APPTAINER") == "1":
        return False
    if os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"):
        return False
    if args.execution_mode == "direct":
        return False
    if args.execution_mode == "apptainer":
        return True
    return not runtime_available(repo_root)


def launch_in_apptainer(args: argparse.Namespace, forwarded_argv: list[str]) -> int:
    """Relaunch this script inside the configured Metal3D apptainer image."""
    container = Path(args.container).expanduser().resolve()
    script_path = Path(__file__).resolve()
    if shutil.which(args.apptainer_bin) is None:
        raise FileNotFoundError(
            f"Apptainer executable was not found on PATH: {args.apptainer_bin}"
        )
    if not container.exists():
        raise FileNotFoundError(f"Container image does not exist: {container}")

    use_nv = args.always_nv or (not args.no_nv and args.device != "cpu")
    command = [args.apptainer_bin, "exec"]
    if use_nv:
        command.append("--nv")
    for bind in (*DEFAULT_BINDS, *args.bind):
        command.extend(["--bind", bind])
    command.extend(
        [str(container), "python3", str(script_path), *forwarded_argv, "--execution-mode", "direct"]
    )

    env = os.environ.copy()
    repo_root_value = args.repo_root or os.environ.get("METAL3D_REPO")
    if repo_root_value:
        env["APPTAINERENV_METAL3D_REPO"] = str(
            Path(repo_root_value).expanduser().resolve()
        )
    env["APPTAINERENV_METAL3D_WRAPPER_IN_APPTAINER"] = "1"

    completed = subprocess.run(command, env=env)
    return completed.returncode


def load_runtime(repo_root: Path) -> None:
    """Import Metal3D runtime modules only after direct execution is chosen."""
    global Model
    global _vox
    global create_grid_fromBB
    global find_unique_sites
    global get_all_metalbinding_resids
    global get_all_protein_resids
    global get_bb
    global get_probability_mean
    global np
    global processStructures
    global torch
    global write_cubefile

    if Model is not None:
        return

    metal3d_dir = repo_root / "Metal3D"
    if not metal3d_dir.is_dir():
        raise FileNotFoundError(f"Metal3D repo not found at {metal3d_dir}")

    os.environ["METAL3D_REPO"] = str(repo_root)
    sys.path.insert(0, str(metal3d_dir))

    # helpers.py imports py3Dmol at module level for interactive viewers.
    if "py3Dmol" not in sys.modules:
        sys.modules["py3Dmol"] = types.ModuleType("py3Dmol")

    import numpy as _np
    import torch as _torch
    from utils import voxelization as _vox_module
    from utils.helpers import (
        create_grid_fromBB as _create_grid_fromBB,
        find_unique_sites as _find_unique_sites,
        get_all_metalbinding_resids as _get_all_metalbinding_resids,
        get_all_protein_resids as _get_all_protein_resids,
        get_bb as _get_bb,
        get_probability_mean as _get_probability_mean,
        write_cubefile as _write_cubefile,
    )
    from utils.model import Model as _Model

    np = _np
    torch = _torch
    Model = _Model
    _vox = _vox_module
    create_grid_fromBB = _create_grid_fromBB
    find_unique_sites = _find_unique_sites
    get_all_metalbinding_resids = _get_all_metalbinding_resids
    get_all_protein_resids = _get_all_protein_resids
    get_bb = _get_bb
    get_probability_mean = _get_probability_mean
    write_cubefile = _write_cubefile

    _patch_voxelization_for_cpu()
    from utils.voxelization import processStructures as _process_structures

    processStructures = _process_structures

    _patch_probability_mean_serial()
    from utils.helpers import get_probability_mean as _patched_probability_mean

    get_probability_mean = _patched_probability_mean

    _patch_create_grid_vectorised()
    from utils.helpers import create_grid_fromBB as _patched_create_grid_fromBB

    create_grid_fromBB = _patched_create_grid_fromBB


def _patch_voxelization_for_cpu() -> None:
    """Replace processStructures with a CPU-safe, fork-free implementation."""
    import time as _time

    def processStructures_cpu_safe(pdb_file, resids, clean=True):
        from moleculekit.molecule import Molecule

        start = _time.time()
        try:
            prot = Molecule(pdb_file)
        except Exception as exc:  # pragma: no cover - input error path
            raise RuntimeError(f"could not read {pdb_file}: {exc}") from exc

        if clean:
            prot.filter("protein and not hydrogen")

        results = []
        for idx in resids:
            try:
                env = (prot.copy(), idx)
            except Exception:
                print(f"ignore {idx}")
                continue
            result = _vox.voxelize_single_notcentered(env)
            if result is not None:
                results.append(result)

        if not results:
            raise RuntimeError(
                "voxelization produced no boxes - check for chain breaks or "
                "missing metal-binding residues"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        voxels = torch.empty(len(results), 8, 32, 32, 32, device=device)
        vox_env, prot_centers_list, prot_n_list, envs = zip(*results)
        for i, voxel in enumerate(vox_env):
            voxels[i] = voxel.to(device)

        print(
            f"-----  Voxelization  -----  "
            f"{_time.time() - start:.3f}s ({len(results)} boxes)"
        )
        return voxels, prot_centers_list, prot_n_list, envs

    _vox.processStructures = processStructures_cpu_safe


def _patch_probability_mean_serial() -> None:
    """Replace the upstream Pool-based helper with a vectorized KDTree path."""
    from scipy.spatial import KDTree
    from utils import helpers as _helpers

    def get_probability_mean_serial(grid, prot_centers, pvalues):
        prot_v = np.asarray(prot_centers)
        pvalues = np.asarray(pvalues, dtype=np.float32)
        tree = KDTree(prot_v)
        dists, idxs = tree.query(grid, k=20, distance_upper_bound=0.25, workers=-1)
        valid = np.isfinite(dists)
        idxs_clipped = np.clip(idxs, 0, len(pvalues) - 1)
        gathered = pvalues[idxs_clipped]
        gathered = np.where(valid, gathered, 0.0)
        counts = valid.sum(axis=1)
        sums = gathered.sum(axis=1)
        return np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    _helpers.get_probability_mean = get_probability_mean_serial


def _patch_create_grid_vectorised() -> None:
    """Replace the upstream Python triple-loop with np.meshgrid."""
    from utils import helpers as _helpers

    def create_grid_fromBB(boundingBox, voxelSize=1):
        xrange = np.arange(boundingBox[0][0], boundingBox[1][0] + 0.5, step=voxelSize)
        yrange = np.arange(boundingBox[0][1], boundingBox[1][1] + 0.5, step=voxelSize)
        zrange = np.arange(boundingBox[0][2], boundingBox[1][2] + 0.5, step=voxelSize)
        xx, yy, zz = np.meshgrid(xrange, yrange, zrange, indexing="ij")
        gridpoints = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(
            np.float64
        )
        return gridpoints, (xrange.shape[0], yrange.shape[0], zrange.shape[0])

    _helpers.create_grid_fromBB = create_grid_fromBB


def eprint(message: str) -> None:
    """Print a progress message to stderr."""
    print(message, file=sys.stderr, flush=True)


def iso_now() -> str:
    """Return the current timestamp in ISO-8601 format."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_seconds(seconds: float) -> str:
    """Format elapsed seconds as H:MM:SS."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def split_structure_suffix(filename: str) -> tuple[str, str]:
    """Split a structure filename into stem and full recognized suffix."""
    lower_name = filename.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lower_name.endswith(suffix):
            return filename[: -len(suffix)], filename[-len(suffix) :]
    return Path(filename).stem, Path(filename).suffix


def output_stem_for_path(path: Path) -> str:
    """Build the retained-output stem for a structure path."""
    stem, _ = split_structure_suffix(path.name)
    return stem


def is_supported_structure(path: Path) -> bool:
    """Return True for supported structure file suffixes."""
    lower_name = path.name.lower()
    return any(lower_name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def is_pdb_input(path: Path) -> bool:
    """Return True when the input is a PDB or gzipped PDB."""
    lower_name = path.name.lower()
    return lower_name.endswith(".pdb") or lower_name.endswith(".pdb.gz")


def remove_existing_path(path: Path) -> None:
    """Remove an existing file or directory."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def collect_structure_files(
    input_paths: list[Path],
    recursive: bool,
) -> tuple[list[Path], list[str]]:
    """Collect supported files from requested inputs and deduplicate them."""
    collected: list[Path] = []
    seen_realpaths: set[str] = set()
    duplicate_paths: list[str] = []

    for input_path in input_paths:
        resolved_input = input_path.expanduser().resolve()
        if resolved_input.is_dir():
            eprint(f"Scanning input: {resolved_input}")
            if recursive:
                iterator = sorted(
                    path for path in resolved_input.rglob("*") if path.is_file()
                )
            else:
                iterator = sorted(
                    path for path in resolved_input.iterdir() if path.is_file()
                )
        elif resolved_input.is_file():
            iterator = [resolved_input]
        else:
            raise FileNotFoundError(f"Input path does not exist: {resolved_input}")

        for path in iterator:
            if not is_supported_structure(path):
                continue
            real_path = str(path.resolve())
            if real_path in seen_realpaths:
                duplicate_paths.append(real_path)
                continue
            seen_realpaths.add(real_path)
            collected.append(path.resolve())

    collected.sort()
    return collected, duplicate_paths


def resolve_output_dir(args: argparse.Namespace, input_paths: list[Path]) -> Path:
    """Resolve the single directory that will hold all retained outputs."""
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    if args.output_prefix:
        return Path(args.output_prefix).expanduser().resolve()
    first_input = input_paths[0].expanduser().resolve()
    base_dir = first_input if first_input.is_dir() else first_input.parent
    return (base_dir / "metal3d_run").resolve()


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    """Validate and create the output directory."""
    if output_dir.exists():
        if force:
            remove_existing_path(output_dir)
        else:
            raise FileExistsError(
                f"Output path already exists: {output_dir}. Use --force to overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=False)


def resolve_device(requested: str) -> torch.device:
    """Resolve the torch device from the CLI request."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_probes(probefile: Path) -> list[dict[str, float]]:
    """Parse a Metal3D probe PDB and return sorted site dictionaries."""
    sites = []
    with probefile.open() as handle:
        for line in handle:
            if not line.startswith(("HETATM", "ATOM")):
                continue
            try:
                x_coord = float(line[30:38])
                y_coord = float(line[38:46])
                z_coord = float(line[46:54])
                probability = float(line[54:60])
            except ValueError:
                continue
            sites.append(
                {
                    "x": x_coord,
                    "y": y_coord,
                    "z": z_coord,
                    "p": probability,
                }
            )
    sites.sort(key=lambda site: site["p"], reverse=True)
    return sites


def split_pdb_body_and_tail(lines: list[str]) -> tuple[list[str], list[str]]:
    """Keep CONECT/MASTER/END records at the end after appended probes."""
    split_index = len(lines)
    while split_index > 0:
        record = lines[split_index - 1][:6].strip()
        if record in {"CONECT", "MASTER", "END", "ENDMDL"} or not lines[
            split_index - 1
        ].strip():
            split_index -= 1
            continue
        break
    return lines[:split_index], lines[split_index:]


def max_atom_serial(lines: list[str]) -> int:
    """Return the maximum ATOM/HETATM serial number present in a PDB."""
    max_serial = 0
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            max_serial = max(max_serial, int(line[6:11]))
        except ValueError:
            continue
    return max_serial


def format_probe_atom_line(
    serial: int,
    residue_number: int,
    site: dict[str, float],
) -> str:
    """Format one predicted metal site as a PDB HETATM record."""
    return (
        f"HETATM{serial:5d}   ZN MZN M{residue_number:4d}    "
        f"{site['x']:8.3f}{site['y']:8.3f}{site['z']:8.3f}"
        f"{site['p']:6.2f}{0.0:6.2f}          ZN\n"
    )


def read_stripped_pdb_lines(source_structure: Path) -> list[str]:
    """Return the same stripped protein-only PDB text Metal3D voxelizes."""
    from moleculekit.molecule import Molecule

    try:
        protein = Molecule(str(source_structure))
    except Exception as exc:
        raise RuntimeError(f"could not read {source_structure}: {exc}") from exc

    protein.filter("protein and not hydrogen", _logger=False)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as handle:
            temp_path = Path(handle.name)
        protein.write(str(temp_path))
        return temp_path.read_text().splitlines(keepends=True)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_combined_pdb(
    source_pdb: Path,
    combined_pdb: Path,
    sites: list[dict[str, float]],
) -> None:
    """Write a protein-only PDB with predicted metal sites appended."""
    if not is_pdb_input(source_pdb):
        raise ValueError(
            f"combined PDB output is only supported for PDB inputs: {source_pdb}"
        )

    stripped_lines = read_stripped_pdb_lines(source_pdb)
    body_lines, tail_lines = split_pdb_body_and_tail(stripped_lines)
    next_serial = max_atom_serial(body_lines) + 1

    with combined_pdb.open("w") as handle:
        handle.writelines(body_lines)
        for residue_number, site in enumerate(sites, start=1):
            handle.write(format_probe_atom_line(next_serial, residue_number, site))
            next_serial += 1
        if tail_lines:
            handle.writelines(tail_lines)
        else:
            handle.write("END\n")


def run_one(
    pdb: Path,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    *,
    metalbinding: bool,
    cluster_threshold: float,
    pthreshold: float,
    batch_size: int,
    write_cube: bool,
    write_combined_pdbs: bool,
) -> dict:
    """Run Metal3D on a single structure and return a summary dict."""
    output_stem = output_stem_for_path(pdb)
    probefile = output_dir / f"{output_stem}.probes.pdb"
    combined_pdb = (
        output_dir / f"{output_stem}.combined.pdb" if write_combined_pdbs else None
    )
    cubefile = output_dir / f"{output_stem}.cube" if write_cube else None

    result = {
        "input": str(pdb),
        "source_name": pdb.name,
        "output_stem": output_stem,
        "success": False,
        "warning": None,
        "error": None,
        "n_candidate_residues": 0,
        "n_sites": 0,
        "max_p": None,
        "max_xyz": None,
        "top_site": None,
        "sites": [],
        "probefile": None,
        "combined_pdbfile": None,
        "cubefile": None,
    }

    residue_ids = (
        get_all_metalbinding_resids(str(pdb))
        if metalbinding
        else get_all_protein_resids(str(pdb))
    )
    result["n_candidate_residues"] = int(len(residue_ids))

    if len(residue_ids) == 0:
        if combined_pdb is not None and is_pdb_input(pdb):
            write_combined_pdb(pdb, combined_pdb, [])
            result["combined_pdbfile"] = str(combined_pdb)
        result["success"] = True
        result["warning"] = "no candidate residues found"
        return result

    voxels, prot_centers, _, _ = processStructures(str(pdb), residue_ids)
    voxels = voxels.to(device)

    outputs = torch.zeros([voxels.size(0), 1, 32, 32, 32], dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for batch_start in range(0, voxels.size(0), batch_size):
            batch_outputs = model(voxels[batch_start : batch_start + batch_size])
            outputs[batch_start : batch_start + batch_size] = batch_outputs.cpu().detach()

    protein_centers = np.vstack(prot_centers)
    output_values = outputs.flatten().numpy()
    bounding_box = get_bb(protein_centers)
    grid, box_shape = create_grid_fromBB(bounding_box)
    probabilities = get_probability_mean(grid, protein_centers, output_values)

    if write_cube and cubefile is not None:
        write_cubefile(
            bounding_box,
            probabilities,
            box_shape,
            outname=str(cubefile),
            gridres=1,
        )
        result["cubefile"] = str(cubefile)

    find_unique_sites(
        probabilities,
        grid,
        writeprobes=True,
        probefile=str(probefile),
        threshold=cluster_threshold,
        p=pthreshold,
    )

    if probefile.exists():
        sites = _read_probes(probefile)
        result["sites"] = sites
        result["n_sites"] = len(sites)
        result["probefile"] = str(probefile)
        if sites:
            result["top_site"] = sites[0]

    if combined_pdb is not None:
        if is_pdb_input(pdb):
            write_combined_pdb(pdb, combined_pdb, result["sites"])
            result["combined_pdbfile"] = str(combined_pdb)
        else:
            result["warning"] = "combined PDB output skipped for non-PDB input"

    if probabilities.size:
        max_index = int(np.argmax(probabilities))
        result["max_p"] = float(np.max(probabilities))
        result["max_xyz"] = [float(value) for value in grid[max_index]]

    result["success"] = True
    return result


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch Metal3D runner with one-folder outputs. In auto mode, the "
            "wrapper relaunches itself inside the Metal3D apptainer image when "
            "the local runtime is unavailable."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Input directories and/or structure files. Directories are scanned "
            "non-recursively unless --recursive is passed."
        ),
    )
    parser.add_argument(
        "--pdb",
        action="append",
        default=[],
        help="Input PDB/CIF path. Pass --pdb multiple times for batches.",
    )
    parser.add_argument(
        "--pdb-list",
        type=str,
        default=None,
        help="Optional file with one input path per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where all retained outputs are written.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Notebook-compatible alias for the output directory path. Unlike "
            "metal3D_characterisation.py, this is treated as a directory name."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan directory inputs recursively.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help=(
            "Metal3D repository root. When launching through apptainer this is "
            "mirrored into METAL3D_REPO inside the container."
        ),
    )
    parser.add_argument(
        "--metalbinding",
        dest="metalbinding",
        action="store_true",
        default=True,
        help=(
            "Scan only likely metal-binding sidechains. Older "
            "--metalbinding=0/1 spellings are also accepted."
        ),
    )
    parser.add_argument(
        "--all-protein",
        dest="metalbinding",
        action="store_false",
        help="Evaluate all protein residues instead of only metal-binding ones.",
    )
    parser.add_argument("--cluster-threshold", type=float, default=7.0)
    parser.add_argument("--pthreshold", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. auto = cuda if available else cpu.",
    )
    parser.add_argument("--status-interval-seconds", type=int, default=30)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-cube", type=int, default=0)
    parser.add_argument("--write-cubes", action="store_true")
    parser.add_argument("--write-combined-pdbs", action="store_true")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional model weights path. Defaults to the selected repo-root.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("auto", "apptainer", "direct"),
        default="auto",
        help=(
            "auto = use local runtime when available else launch apptainer; "
            "apptainer = always launch the container; direct = never relaunch."
        ),
    )
    parser.add_argument(
        "--container",
        type=str,
        default=str(DEFAULT_CONTAINER),
        help="Path to the Metal3D apptainer image.",
    )
    parser.add_argument(
        "--apptainer-bin",
        type=str,
        default="apptainer",
        help="Apptainer executable name or path.",
    )
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        help="Additional apptainer bind mount in SRC:DST form. Pass multiple times.",
    )
    parser.add_argument(
        "--no-nv",
        action="store_true",
        help="Do not pass --nv to apptainer even when device is cuda/auto.",
    )
    parser.add_argument(
        "--always-nv",
        action="store_true",
        help="Always pass --nv to apptainer regardless of the requested device.",
    )
    return parser


def run_direct(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute Metal3D directly in the current Python environment."""
    resolved_repo_root = resolve_repo_root(args)
    load_runtime(resolved_repo_root)

    raw_inputs = [Path(path) for path in args.inputs]
    raw_inputs.extend(Path(path) for path in args.pdb)
    if args.pdb_list:
        with open(args.pdb_list) as handle:
            raw_inputs.extend(Path(line.strip()) for line in handle if line.strip())
    if not raw_inputs:
        parser.error("No inputs provided. Pass positional inputs, --pdb, or --pdb-list.")

    structure_files, duplicate_paths = collect_structure_files(
        raw_inputs,
        recursive=args.recursive,
    )
    if not structure_files:
        raise ValueError("No supported structure files were found in the requested inputs.")

    output_dir = resolve_output_dir(args, raw_inputs)
    write_cube = bool(args.write_cube) or args.write_cubes
    metalbinding = args.metalbinding
    weights_path = resolve_weights_path(args, resolved_repo_root)
    device = resolve_device(args.device)

    eprint(
        f"Resolved {len(structure_files)} structure(s) into output directory: {output_dir}"
    )
    if duplicate_paths:
        eprint(f"Skipped {len(duplicate_paths)} duplicate input path(s).")

    if not weights_path.exists():
        raise FileNotFoundError(f"Metal3D weights not found: {weights_path}")

    if args.dry_run:
        eprint(f"Dry run requested. Resolved weights: {weights_path}")
        return 0

    eprint(f"Loading Metal3D weights from {weights_path} on {device}")
    model = Model().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    prepare_output_dir(output_dir, args.force)

    run_started_at = iso_now()
    overall_start_time = time.monotonic()
    progress_state = {
        "completed": 0,
        "successes": 0,
        "failures": 0,
        "current": "",
    }
    stop_event = threading.Event()
    results: dict[str, dict] = {}
    success_count = 0
    failure_count = 0
    exit_code = 0

    def status_worker(total_structures: int) -> None:
        while not stop_event.wait(args.status_interval_seconds):
            elapsed_seconds = time.monotonic() - overall_start_time
            eprint(
                "[status] "
                f"completed={progress_state['completed']}/{total_structures} "
                f"success={progress_state['successes']} "
                f"failure={progress_state['failures']} "
                f"current={progress_state['current'] or '-'} "
                f"elapsed={format_seconds(elapsed_seconds)}"
            )

    status_thread = None
    try:
        if args.status_interval_seconds > 0:
            status_thread = threading.Thread(
                target=status_worker,
                args=(len(structure_files),),
                daemon=True,
            )
            status_thread.start()

        for index, pdb_path in enumerate(structure_files, start=1):
            progress_state["current"] = pdb_path.name
            structure_start_time = time.monotonic()
            eprint(f"[{index}/{len(structure_files)}] Metal3D on {pdb_path}")
            try:
                result = run_one(
                    pdb_path,
                    output_dir,
                    model,
                    device,
                    metalbinding=metalbinding,
                    cluster_threshold=args.cluster_threshold,
                    pthreshold=args.pthreshold,
                    batch_size=args.batch_size,
                    write_cube=write_cube,
                    write_combined_pdbs=args.write_combined_pdbs,
                )
                success_count += 1
                progress_state["successes"] = success_count
            except Exception as exc:
                failure_count += 1
                progress_state["failures"] = failure_count
                result = {
                    "input": str(pdb_path),
                    "source_name": pdb_path.name,
                    "output_stem": output_stem_for_path(pdb_path),
                    "success": False,
                    "warning": None,
                    "error": str(exc),
                    "n_candidate_residues": 0,
                    "n_sites": 0,
                    "max_p": None,
                    "max_xyz": None,
                    "top_site": None,
                    "sites": [],
                    "probefile": None,
                    "combined_pdbfile": None,
                    "cubefile": None,
                }
                eprint(f"FAILED: {pdb_path} :: {exc}")
                if args.fail_fast:
                    exit_code = 1

            result["elapsed_seconds"] = time.monotonic() - structure_start_time
            result_key = pdb_path.name if pdb_path.name not in results else str(pdb_path)
            results[result_key] = result
            progress_state["completed"] = index
            progress_state["current"] = ""

            if args.fail_fast and not result["success"]:
                break
    finally:
        stop_event.set()
        if status_thread is not None:
            status_thread.join()

    out_json = output_dir / "metal3d_results.json"
    with out_json.open("w") as handle:
        json.dump(results, handle, indent=2)

    total_elapsed_seconds = time.monotonic() - overall_start_time
    metadata = {
        "started_at": run_started_at,
        "finished_at": iso_now(),
        "elapsed_seconds": total_elapsed_seconds,
        "elapsed_hms": format_seconds(total_elapsed_seconds),
        "script": str(Path(__file__).resolve()),
        "success": exit_code == 0,
        "output_dir": str(output_dir),
        "repo_root": str(resolved_repo_root),
        "weights_path": str(weights_path),
        "device": str(device),
        "user_inputs": [str(path.expanduser().resolve()) for path in raw_inputs],
        "structure_count": len(structure_files),
        "duplicate_realpaths_skipped": duplicate_paths,
        "success_count": success_count,
        "failure_count": failure_count,
        "effective_parameters": {
            "recursive": args.recursive,
            "metalbinding_only": metalbinding,
            "batch_size": args.batch_size,
            "pthreshold": args.pthreshold,
            "cluster_threshold": args.cluster_threshold,
            "write_cubes": write_cube,
            "write_combined_pdbs": args.write_combined_pdbs,
            "status_interval_seconds": args.status_interval_seconds,
            "fail_fast": args.fail_fast,
            "force": args.force,
            "dry_run": args.dry_run,
        },
        "retained_outputs": {
            "results_json": str(out_json),
            "metadata_json": str(output_dir / "metal3d_run_metadata.json"),
            "output_dir": str(output_dir),
        },
    }

    metadata_path = output_dir / "metal3d_run_metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    eprint(
        f"Wrote {out_json} ({format_seconds(total_elapsed_seconds)}, "
        f"{len(results)} inputs)"
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = sys.argv[1:] if argv is None else argv
    normalized_argv = normalize_cli_args(raw_argv)
    args = parser.parse_args(normalized_argv)

    if should_use_apptainer(args, resolve_repo_root(args)):
        return launch_in_apptainer(args, normalized_argv)
    return run_direct(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
