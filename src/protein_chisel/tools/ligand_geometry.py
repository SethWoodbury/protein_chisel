"""Ligand geometry summary — computed once per substrate, cached.

For pocket-accessibility scoring we need a fixed reference point: how
small is the ligand really? Different orientations of the same molecule
project to different cross-sectional radii. The "minimum effective
radius" (smallest projected radius across rotational orientations) is
the relevant number for "can this molecule fit through a tunnel of
clearance X". Larger projected radii are also useful (max, mean) for
characterizing how much steric flexibility we have.

This module computes those numbers from the seed PDB's ligand HETATM
block (treating metals as separate fragments, since they coordinate
not occupy bulk). Optional RDKit conformer-ensemble computation gives
a flexibility-aware estimate; without RDKit we fall back to PCA-axis
projection of the seed pose only (still useful, just less complete).

Public API:

    ligand_geometry_from_pdb(pdb_path, ligand_resname=None)
        Returns dict with n_heavy_atoms, radius_of_gyration, principal
        radii (min/mean/max projected), bounding box diagonal, and a
        per-fragment breakdown if the ligand has metals.

The output is meant to be stored ONCE in manifest.json (the seed-derived
constants don't change across designs — they're scaffold-invariant) and
referenced by score_tunnels for the ligand-fit hard gate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


# Bond-length threshold (Å) for the "if two heavy atoms are within this,
# they're bonded" connectivity heuristic used to fragment the ligand.
# 1.85 Å covers single C-C, C-N, C-O, S-S, etc.; metal-ligand bonds tend
# to be >2 Å so this naturally separates metals.
_BOND_DISTANCE_MAX = 1.85

# Common metals; treated as separate fragments regardless of distance.
_METAL_ELEMENTS = frozenset({
    "ZN", "CA", "MG", "MN", "FE", "CU", "NI", "CO", "CD", "K", "NA",
    "HG", "PB", "PT", "PD", "MO", "W", "CR", "V", "RU", "RH", "AG",
    "AU", "AL",
})


def _read_ligand_atoms(
    pdb_path: str | Path,
    ligand_resname: Optional[str] = None,
) -> list[tuple[str, str, np.ndarray]]:
    """Return [(atom_name, element, coord), ...] for the ligand HETATM block.

    If ``ligand_resname`` is None, auto-detects the largest non-water
    HETATM group.
    """
    # Auto-detect if needed
    target_resname = ligand_resname
    if target_resname is None:
        groups: dict[str, int] = {}
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith("HETATM"):
                    continue
                rn = line[17:20].strip()
                if rn in {"HOH", "WAT", "DOD", "TIP", "TIP3"}:
                    continue
                groups[rn] = groups.get(rn, 0) + 1
        if not groups:
            return []
        target_resname = max(groups.keys(), key=lambda k: groups[k])

    out: list[tuple[str, str, np.ndarray]] = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            if line[17:20].strip() != target_resname:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not element:
                # Best-effort element guess from atom name
                element = "".join(c for c in atom_name if c.isalpha()).upper()[:2]
            # Skip H (we want heavy-atom geometry)
            if element == "H":
                continue
            out.append((atom_name, element, np.array([x, y, z], dtype=np.float64)))
    return out


def _fragment_by_connectivity(
    atoms: list[tuple[str, str, np.ndarray]],
    bond_max: float = _BOND_DISTANCE_MAX,
) -> list[list[int]]:
    """Return fragment indices: list of lists, one per connected component.

    Metal atoms (per ``_METAL_ELEMENTS``) are always isolated (not
    bonded to anything regardless of distance — they coordinate, not
    bond covalently in a way that places them in the same rigid body).
    """
    n = len(atoms)
    if n == 0:
        return []
    # Build adjacency
    coords = np.stack([a[2] for a in atoms])
    elements = [a[1] for a in atoms]
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        if elements[i] in _METAL_ELEMENTS:
            continue
        for j in range(i + 1, n):
            if elements[j] in _METAL_ELEMENTS:
                continue
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= bond_max:
                adj[i].add(j)
                adj[j].add(i)
    # Connected components via DFS
    seen = [False] * n
    fragments: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        comp = []
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    stack.append(v)
        fragments.append(comp)
    return fragments


def _projected_radii(coords: np.ndarray) -> tuple[float, float, float]:
    """Return (min, mean, max) projected radius of a point cloud across
    its three principal axes.

    The "projected radius along axis u" is half the spread of points
    projected onto the plane perpendicular to u — i.e. how wide the
    molecule looks when viewed end-on along that axis. The MIN of these
    is the smallest cross-section the molecule presents (relevant for
    "can it fit through a tunnel of clearance X").
    """
    if len(coords) <= 1:
        return 0.0, 0.0, 0.0
    centered = coords - coords.mean(axis=0)
    cov = centered.T @ centered / max(1, len(centered) - 1)
    _, evecs = np.linalg.eigh(cov)
    # For each principal axis u, the projected radius is the half-width
    # of the perpendicular plane: max norm of (P - origin) projected
    # onto the plane perpendicular to u.
    radii = []
    for k in range(3):
        u = evecs[:, k]
        # Perpendicular component of each point: P - (P·u)u
        proj_along = centered @ u
        perp = centered - np.outer(proj_along, u)
        # Half-width of the cross-section = max distance from axis
        radii.append(float(np.linalg.norm(perp, axis=1).max()))
    return min(radii), float(np.mean(radii)), max(radii)


def ligand_geometry_from_pdb(
    pdb_path: str | Path,
    ligand_resname: Optional[str] = None,
) -> dict:
    """Compute ligand geometry summary from a PDB's HETATM block.

    Args:
        pdb_path: PDB path (typically the seed; the ligand pose is
            scaffold-invariant for our use case).
        ligand_resname: Optional ligand 3-letter code; auto-detected
            if None.

    Returns:
        Dict with keys (some metals-aware):
            ligand_resname           detected/given 3-letter code
            n_heavy_atoms            heavy atom count (excl. metals)
            n_metal_atoms            metals counted separately
            n_fragments              connected components by bond-distance
            radius_of_gyration       Å, over heavy atoms (excl. metals)
            min_projected_radius     Å, smallest cross-sectional radius
                                     (the "can it fit through a tunnel" #)
            mean_projected_radius    Å
            max_projected_radius     Å
            bounding_box_diagonal    Å
            max_atom_to_centroid     Å
            metal_elements           list of metal element symbols
            fragment_atom_counts     list of int (per fragment)
    """
    atoms = _read_ligand_atoms(pdb_path, ligand_resname)
    if not atoms:
        return {
            "ligand_resname": ligand_resname or "",
            "n_heavy_atoms": 0,
            "n_metal_atoms": 0,
            "n_fragments": 0,
            "radius_of_gyration": 0.0,
            "min_projected_radius": 0.0,
            "mean_projected_radius": 0.0,
            "max_projected_radius": 0.0,
            "bounding_box_diagonal": 0.0,
            "max_atom_to_centroid": 0.0,
            "metal_elements": [],
            "fragment_atom_counts": [],
        }

    elements = [a[1] for a in atoms]
    coords = np.stack([a[2] for a in atoms])
    metal_mask = np.array([e in _METAL_ELEMENTS for e in elements])
    n_metals = int(metal_mask.sum())
    metal_elements = sorted({elements[i] for i in range(len(atoms)) if metal_mask[i]})

    # Fragment analysis (metals separate)
    frags = _fragment_by_connectivity(atoms)
    frag_counts = [len(f) for f in frags]

    # Geometry over the largest non-metal fragment (the "main organic
    # backbone"), or over all heavy atoms if there's only one fragment
    nonmetal_idx = np.flatnonzero(~metal_mask)
    if len(nonmetal_idx) == 0:
        organic_coords = coords
    else:
        # Take the largest organic fragment's atoms
        organic_frags = [
            [i for i in f if not metal_mask[i]] for f in frags
        ]
        organic_frags = [f for f in organic_frags if f]
        if organic_frags:
            biggest = max(organic_frags, key=len)
            organic_coords = coords[biggest]
        else:
            organic_coords = coords[nonmetal_idx]

    centroid = organic_coords.mean(axis=0)
    rg = float(np.sqrt(((organic_coords - centroid) ** 2).sum(axis=1).mean()))
    bbox_diag = float(np.linalg.norm(organic_coords.max(axis=0) - organic_coords.min(axis=0)))
    max_atom_to_centroid = float(np.linalg.norm(organic_coords - centroid, axis=1).max())
    rmin, rmean, rmax = _projected_radii(organic_coords)

    # Detected resname (if auto)
    if ligand_resname is None:
        # Re-derive from the PDB
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith("HETATM"):
                    rn = line[17:20].strip()
                    if rn not in {"HOH", "WAT", "DOD", "TIP", "TIP3"}:
                        ligand_resname = rn
                        break

    return {
        "ligand_resname": ligand_resname,
        "n_heavy_atoms": int((~metal_mask).sum()),
        "n_metal_atoms": n_metals,
        "n_fragments": len(frags),
        "radius_of_gyration": rg,
        "min_projected_radius": rmin,
        "mean_projected_radius": rmean,
        "max_projected_radius": rmax,
        "bounding_box_diagonal": bbox_diag,
        "max_atom_to_centroid": max_atom_to_centroid,
        "metal_elements": metal_elements,
        "fragment_atom_counts": frag_counts,
    }
