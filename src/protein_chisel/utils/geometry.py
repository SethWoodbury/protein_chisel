"""Pose-level geometric helpers.

Only depends on PyRosetta (via utils/pose) and numpy. Keep functions pure:
take a pose and return numbers / dicts / DataFrames.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .pose import _ensure_pyrosetta_imported


def heavy_atom_coords(residue) -> np.ndarray:
    """All heavy-atom xyz coordinates of a residue, shape (N, 3)."""
    coords: list[list[float]] = []
    for i in range(1, residue.natoms() + 1):
        if residue.atom_is_hydrogen(i):
            continue
        xyz = residue.xyz(i)
        coords.append([xyz.x, xyz.y, xyz.z])
    return np.asarray(coords, dtype=float)


def ca_coords(pose, chain_id: Optional[str] = None) -> np.ndarray:
    """CA coordinates of all protein residues (optionally one chain), shape (L, 3)."""
    pts: list[list[float]] = []
    for r in pose.residues:
        if not r.is_protein():
            continue
        if chain_id is not None and pose.pdb_info().chain(r.seqpos()) != chain_id:
            continue
        if not r.has("CA"):
            continue
        xyz = r.xyz("CA")
        pts.append([xyz.x, xyz.y, xyz.z])
    return np.asarray(pts, dtype=float)


def all_protein_heavy_coords(pose, chain_id: Optional[str] = None) -> np.ndarray:
    """Stack of every protein heavy-atom coordinate."""
    out: list[np.ndarray] = []
    for r in pose.residues:
        if not r.is_protein():
            continue
        if chain_id is not None and pose.pdb_info().chain(r.seqpos()) != chain_id:
            continue
        out.append(heavy_atom_coords(r))
    if not out:
        return np.zeros((0, 3))
    return np.vstack(out)


def min_distance_to(pose, target_seqpos: int, query_seqposes: Iterable[int]) -> dict[int, float]:
    """Min heavy-atom distance from each query residue to a target residue.

    Returns ``{query_seqpos: min_d_angstrom}``.
    """
    target_xyz = heavy_atom_coords(pose.residue(target_seqpos))
    out: dict[int, float] = {}
    for q in query_seqposes:
        q_xyz = heavy_atom_coords(pose.residue(q))
        if len(target_xyz) == 0 or len(q_xyz) == 0:
            out[q] = float("nan")
            continue
        d = np.linalg.norm(q_xyz[:, None, :] - target_xyz[None, :, :], axis=-1)
        out[q] = float(d.min())
    return out


def min_distance_to_any(
    pose, target_seqposes: Iterable[int], query_seqposes: Iterable[int]
) -> dict[int, float]:
    """Min heavy-atom distance from each query to ANY of the target residues."""
    target_pts: list[np.ndarray] = []
    for t in target_seqposes:
        target_pts.append(heavy_atom_coords(pose.residue(t)))
    if target_pts:
        target_xyz = np.vstack(target_pts)
    else:
        target_xyz = np.zeros((0, 3))
    out: dict[int, float] = {}
    for q in query_seqposes:
        q_xyz = heavy_atom_coords(pose.residue(q))
        if len(target_xyz) == 0 or len(q_xyz) == 0:
            out[q] = float("nan")
            continue
        d = np.linalg.norm(q_xyz[:, None, :] - target_xyz[None, :, :], axis=-1)
        out[q] = float(d.min())
    return out


def phi_psi(pose) -> dict[int, tuple[float, float]]:
    """Per-residue (phi, psi). Returns dict keyed by seqpos. NaN at termini."""
    out: dict[int, tuple[float, float]] = {}
    for r in pose.residues:
        if not r.is_protein():
            continue
        sp = r.seqpos()
        try:
            phi = pose.phi(sp)
        except Exception:
            phi = float("nan")
        try:
            psi = pose.psi(sp)
        except Exception:
            psi = float("nan")
        out[sp] = (float(phi), float(psi))
    return out


def ca_ca_consecutive_distances(pose, chain_id: Optional[str] = None) -> list[float]:
    """Sequential CA-CA distances along a chain (length L-1).

    Used for chainbreak detection — a clean backbone has all values
    near 3.8 Å. When ``chain_id`` is None, computes per chain and
    concatenates the results — boundaries between chains are NOT included
    (so two-chain inputs don't get a fake chainbreak between them).
    """
    if chain_id is not None:
        coords = ca_coords(pose, chain_id=chain_id)
        if len(coords) < 2:
            return []
        diffs = np.diff(coords, axis=0)
        return [float(d) for d in np.linalg.norm(diffs, axis=1)]
    pdb_info = pose.pdb_info()
    chains: list[str] = []
    seen: set[str] = set()
    for r in pose.residues:
        if not r.is_protein():
            continue
        c = pdb_info.chain(r.seqpos()) if pdb_info else " "
        if c not in seen:
            seen.add(c)
            chains.append(c)
    out: list[float] = []
    for c in chains:
        out.extend(ca_ca_consecutive_distances(pose, chain_id=c))
    return out


def gyration_tensor(coords: np.ndarray) -> np.ndarray:
    """Mass-uniform gyration tensor (3x3). Coords are CA points."""
    if len(coords) == 0:
        return np.zeros((3, 3))
    centered = coords - coords.mean(axis=0)
    return centered.T @ centered / len(coords)


def radius_of_gyration(coords: np.ndarray) -> float:
    """Radius of gyration = sqrt(mean squared distance from centroid)."""
    if len(coords) == 0:
        return 0.0
    centered = coords - coords.mean(axis=0)
    return float(np.sqrt((centered ** 2).sum(axis=1).mean()))


def shape_descriptors(coords: np.ndarray) -> dict[str, float]:
    """Compute Rg + tensor-eigenvalue-derived shape descriptors.

    Returns:
        rg                : sqrt-mean-square radius of gyration (Å)
        rg_norm           : Rg / sqrt(N) — length-normalized
        asphericity       : 0 = sphere, increases with elongation
        acylindricity     : 0 = cylindrical or spherical
        rel_shape_anisotropy : 0 = isotropic, 1 = linear arrangement
        principal_lengths : sorted eigenvalues sqrt'd, [smallest, mid, largest]

    All values are floats; safe (zero-defaulted) for empty input.
    """
    if len(coords) == 0:
        return {
            "rg": 0.0, "rg_norm": 0.0, "asphericity": 0.0,
            "acylindricity": 0.0, "rel_shape_anisotropy": 0.0,
            "principal_length_1": 0.0, "principal_length_2": 0.0, "principal_length_3": 0.0,
        }
    n = len(coords)
    s = gyration_tensor(coords)
    eigvals = np.sort(np.linalg.eigvalsh(s))
    l1, l2, l3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    rg = float(np.sqrt(eigvals.sum()))
    asphericity = l3 - 0.5 * (l1 + l2)
    acylindricity = l2 - l1
    denom = (l1 + l2 + l3) ** 2
    if denom > 0:
        rsa = (asphericity ** 2 + 0.75 * acylindricity ** 2) / denom
    else:
        rsa = 0.0
    return {
        "rg": rg,
        "rg_norm": rg / np.sqrt(n),
        "asphericity": float(asphericity),
        "acylindricity": float(acylindricity),
        "rel_shape_anisotropy": float(rsa),
        "principal_length_1": float(np.sqrt(max(l1, 0))),
        "principal_length_2": float(np.sqrt(max(l2, 0))),
        "principal_length_3": float(np.sqrt(max(l3, 0))),
    }


# ---- DSSP ------------------------------------------------------------------


def dssp(pose, reduced: bool = False) -> dict[int, str]:
    """DSSP per-residue secondary structure.

    Args:
        reduced: if True, the reduced 3-letter alphabet (H/E/L);
            otherwise the full alphabet (H E L T S B G I -).

    Returns dict keyed by protein seqpos. Ligand residues are skipped.
    """
    _ensure_pyrosetta_imported()
    import pyrosetta.rosetta as ros
    metric = ros.core.simple_metrics.metrics.SecondaryStructureMetric()
    # PyRosetta API: set_use_dssp_reduced(bool). Older bindings called this
    # set_dssp_reduced; defend against both.
    if hasattr(metric, "set_use_dssp_reduced"):
        metric.set_use_dssp_reduced(reduced)
    else:
        metric.set_dssp_reduced(reduced)  # legacy path
    s = metric.calculate(pose)  # returns full SS string for protein residues only
    out: dict[int, str] = {}
    j = 0
    for r in pose.residues:
        if not r.is_protein():
            continue
        if j < len(s):
            out[r.seqpos()] = s[j]
        else:
            out[r.seqpos()] = "L"
        j += 1
    return out
