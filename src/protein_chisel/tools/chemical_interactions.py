"""Chemical interaction detection: hbonds, salt bridges, pi-pi, pi-cation.

All metrics are derived from a single pose; no scorefile dependence.
Output is a list of dicts (one row per interaction) plus aggregate counts.

PyRosetta-bound; run inside pyrosetta.sif.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# Aromatic ring atoms by residue type. The "ring" tuple lists the heavy
# atoms forming the canonical aromatic ring; we use these to compute the
# ring centroid and plane normal.
AROMATIC_RING_ATOMS: dict[str, tuple[str, ...]] = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "NE1", "CE2", "CD2"),  # 5-membered indole ring
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),  # imidazole
}


POSITIVE_SIDECHAIN_ATOMS: dict[str, tuple[str, ...]] = {
    "ARG": ("NH1", "NH2", "NE"),       # guanidinium
    "LYS": ("NZ",),                     # ε-amine
    "HIS": ("NE2", "ND1"),              # protonatable nitrogens
}


NEGATIVE_SIDECHAIN_ATOMS: dict[str, tuple[str, ...]] = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
}


# ---------------------------------------------------------------------------


@dataclass
class InteractionsResult:
    hbonds: list[dict] = field(default_factory=list)
    salt_bridges: list[dict] = field(default_factory=list)
    pi_pi: list[dict] = field(default_factory=list)
    pi_cation: list[dict] = field(default_factory=list)

    def summary(self, prefix: str = "interact__") -> dict[str, int | float]:
        return {
            f"{prefix}n_hbonds": len(self.hbonds),
            f"{prefix}n_salt_bridges": len(self.salt_bridges),
            f"{prefix}n_pi_pi": len(self.pi_pi),
            f"{prefix}n_pi_cation": len(self.pi_cation),
            f"{prefix}sum_hbond_energy": sum(h["energy"] for h in self.hbonds),
        }


def chemical_interactions(
    pdb_path: str | Path,
    params: list[str | Path] = (),
    salt_bridge_cutoff: float = 4.0,            # Å between charged atoms
    pi_pi_centroid_cutoff: float = 6.0,         # Å between ring centroids
    pi_cation_cutoff: float = 6.0,              # Å between ring centroid and cation
) -> InteractionsResult:
    """Detect hbonds, salt bridges, π-π, π-cation interactions in a pose."""
    from protein_chisel.utils.pose import (
        init_pyrosetta, pose_from_file,
        get_hbond_set, hbonds_as_dicts,
    )

    init_pyrosetta(params=list(params))
    pose = pose_from_file(pdb_path)

    res = InteractionsResult()

    # --- hbonds (canonical bcov pattern via utils/pose) -------------------
    hbset = get_hbond_set(pose)
    res.hbonds = hbonds_as_dicts(pose, hbset)

    # --- salt bridges -----------------------------------------------------
    res.salt_bridges = _detect_salt_bridges(pose, cutoff=salt_bridge_cutoff)

    # --- π-π and π-cation -------------------------------------------------
    aromatic_rings = _build_aromatic_rings(pose)
    res.pi_pi = _detect_pi_pi(aromatic_rings, cutoff=pi_pi_centroid_cutoff)
    res.pi_cation = _detect_pi_cation(pose, aromatic_rings, cutoff=pi_cation_cutoff)
    return res


# ---------------------------------------------------------------------------
# Salt bridges
# ---------------------------------------------------------------------------


def _detect_salt_bridges(pose, cutoff: float) -> list[dict]:
    """Salt bridge = positive sidechain N within `cutoff` Å of negative O."""
    pos_atoms: list[tuple[int, str, str, np.ndarray]] = []
    neg_atoms: list[tuple[int, str, str, np.ndarray]] = []

    for r in pose.residues:
        if not r.is_protein():
            continue
        n3 = r.name3()
        atoms_pos = POSITIVE_SIDECHAIN_ATOMS.get(n3, ())
        atoms_neg = NEGATIVE_SIDECHAIN_ATOMS.get(n3, ())
        for atom_name in atoms_pos:
            if r.has(atom_name):
                xyz = r.xyz(atom_name)
                pos_atoms.append((r.seqpos(), n3, atom_name, np.array([xyz.x, xyz.y, xyz.z])))
        for atom_name in atoms_neg:
            if r.has(atom_name):
                xyz = r.xyz(atom_name)
                neg_atoms.append((r.seqpos(), n3, atom_name, np.array([xyz.x, xyz.y, xyz.z])))

    out: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()
    for (psp, pname3, patm, pxyz) in pos_atoms:
        for (nsp, nname3, natm, nxyz) in neg_atoms:
            if psp == nsp:
                continue
            d = float(np.linalg.norm(pxyz - nxyz))
            if d > cutoff:
                continue
            key = tuple(sorted((psp, nsp)))  # type: ignore[assignment]
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            out.append({
                "pos_res": psp,
                "pos_name3": pname3,
                "pos_atom": patm,
                "neg_res": nsp,
                "neg_name3": nname3,
                "neg_atom": natm,
                "distance": d,
            })
    return out


# ---------------------------------------------------------------------------
# Aromatic rings (centroid + normal)
# ---------------------------------------------------------------------------


def _build_aromatic_rings(pose) -> list[dict]:
    """Return list of {resno, name3, centroid, normal} for aromatic rings."""
    out: list[dict] = []
    for r in pose.residues:
        if not r.is_protein():
            continue
        atoms = AROMATIC_RING_ATOMS.get(r.name3())
        if not atoms:
            continue
        coords: list[np.ndarray] = []
        for name in atoms:
            if not r.has(name):
                continue
            xyz = r.xyz(name)
            coords.append(np.array([xyz.x, xyz.y, xyz.z]))
        if len(coords) < 3:
            continue
        coords_arr = np.vstack(coords)
        centroid = coords_arr.mean(axis=0)
        # Plane normal via SVD on centered coords; smallest singular vector.
        centered = coords_arr - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        out.append({
            "resno": r.seqpos(),
            "name3": r.name3(),
            "centroid": centroid,
            "normal": normal,
        })
    return out


def _detect_pi_pi(rings: list[dict], cutoff: float) -> list[dict]:
    """Pairs of aromatic rings within centroid distance < cutoff."""
    out: list[dict] = []
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            a = rings[i]
            b = rings[j]
            d = float(np.linalg.norm(a["centroid"] - b["centroid"]))
            if d > cutoff:
                continue
            # Plane angle (radians, then degrees) — 0° = stacked, 90° = T-shape.
            cos = abs(float(np.dot(a["normal"], b["normal"])))
            cos = max(-1.0, min(1.0, cos))
            angle_deg = float(np.degrees(np.arccos(cos)))
            geom = "stacked" if angle_deg < 30.0 else ("t_shape" if angle_deg > 60.0 else "tilted")
            out.append({
                "res_a": a["resno"],
                "name3_a": a["name3"],
                "res_b": b["resno"],
                "name3_b": b["name3"],
                "centroid_distance": d,
                "plane_angle_deg": angle_deg,
                "geometry": geom,
            })
    return out


def _detect_pi_cation(pose, rings: list[dict], cutoff: float) -> list[dict]:
    """Aromatic ring centroid within `cutoff` Å of a positively charged N."""
    out: list[dict] = []
    cations: list[tuple[int, str, str, np.ndarray]] = []
    for r in pose.residues:
        if not r.is_protein():
            continue
        for atom_name in POSITIVE_SIDECHAIN_ATOMS.get(r.name3(), ()):
            if r.has(atom_name):
                xyz = r.xyz(atom_name)
                cations.append((r.seqpos(), r.name3(), atom_name, np.array([xyz.x, xyz.y, xyz.z])))

    for ring in rings:
        for (csp, cname3, catm, cxyz) in cations:
            if csp == ring["resno"]:
                continue
            d = float(np.linalg.norm(ring["centroid"] - cxyz))
            if d > cutoff:
                continue
            out.append({
                "ring_res": ring["resno"],
                "ring_name3": ring["name3"],
                "cation_res": csp,
                "cation_name3": cname3,
                "cation_atom": catm,
                "distance": d,
            })
    return out


__all__ = ["InteractionsResult", "chemical_interactions"]
