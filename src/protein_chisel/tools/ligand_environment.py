"""Ligand environment metrics.

Per-ligand:
    lig_dist                  : min backbone atom (CA/N/C) to ligand heavy atom
    n_residues_within_5A      : protein residues within 5 Å
    n_residues_within_8A      : protein residues within 8 Å
    ligand_sasa               : full ligand SASA (sum over all atoms)
    ligand_sasa_relative      : ligand_sasa / SASA of the ligand alone (free)
    per_atom_sasa             : dict {atom_name: sasa} for user-specified atoms

Multi-ligand: returns per-ligand result; aggregate at caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class LigandEnvResult:
    ligand_seqpos: int
    ligand_name3: str
    lig_dist: float
    n_residues_within_5A: int
    n_residues_within_8A: int
    ligand_sasa: float
    ligand_sasa_relative: float  # NaN if free-ligand SASA not computed
    per_atom_sasa: dict[str, float] = field(default_factory=dict)

    def to_dict(self, prefix: str = "ligand__") -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {
            f"{prefix}seqpos": self.ligand_seqpos,
            f"{prefix}name3": self.ligand_name3,
            f"{prefix}lig_dist": self.lig_dist,
            f"{prefix}n_residues_within_5A": self.n_residues_within_5A,
            f"{prefix}n_residues_within_8A": self.n_residues_within_8A,
            f"{prefix}sasa": self.ligand_sasa,
            f"{prefix}sasa_relative": self.ligand_sasa_relative,
        }
        for atom, sasa in self.per_atom_sasa.items():
            out[f"{prefix}atom_sasa__{atom}"] = sasa
        return out


def ligand_environment(
    pdb_path: str | Path,
    params: list[str | Path] = (),
    target_atoms: Optional[Iterable[str]] = None,
    compute_relative: bool = True,
) -> list[LigandEnvResult]:
    """Compute ligand-environment metrics for every ligand residue in the pose."""
    from protein_chisel.utils.pose import (
        init_pyrosetta, pose_from_file,
        get_ligand_seqposes, getSASA,
    )
    from protein_chisel.utils.geometry import heavy_atom_coords
    import numpy as np

    init_pyrosetta(params=list(params))
    pose = pose_from_file(pdb_path)

    ligand_seqposes = get_ligand_seqposes(pose)
    if not ligand_seqposes:
        return []

    surf_vol = getSASA(pose, resno=None, probe_radius=1.4)

    out: list[LigandEnvResult] = []
    target_atoms_set = set(target_atoms or ())

    # Pre-compute per-residue heavy atom coords for protein residues
    protein_xyz: dict[int, np.ndarray] = {}
    for r in pose.residues:
        if r.is_protein():
            protein_xyz[r.seqpos()] = heavy_atom_coords(r)

    for lp in ligand_seqposes:
        lig_res = pose.residue(lp)
        lig_xyz = heavy_atom_coords(lig_res)

        # Distance metrics: from ligand heavy atoms to closest protein residue.
        min_d_per_residue: dict[int, float] = {}
        for sp, pxyz in protein_xyz.items():
            if len(pxyz) == 0 or len(lig_xyz) == 0:
                min_d_per_residue[sp] = float("inf")
                continue
            d = np.linalg.norm(pxyz[:, None, :] - lig_xyz[None, :, :], axis=-1)
            min_d_per_residue[sp] = float(d.min())

        if min_d_per_residue:
            lig_dist = float(min(min_d_per_residue.values()))
            n5 = sum(1 for d in min_d_per_residue.values() if d <= 5.0)
            n8 = sum(1 for d in min_d_per_residue.values() if d <= 8.0)
        else:
            lig_dist = float("nan")
            n5 = n8 = 0

        # Ligand SASA (full)
        natom = lig_res.natoms()
        ligand_sasa = sum(surf_vol.surf(lp, a) for a in range(1, natom + 1))

        # Per-atom SASA for requested atoms
        per_atom: dict[str, float] = {}
        if target_atoms_set:
            for a in range(1, natom + 1):
                name = lig_res.atom_name(a).strip()
                if name in target_atoms_set:
                    per_atom[name] = float(surf_vol.surf(lp, a))

        # Relative SASA: ratio to free ligand SASA
        rel = float("nan")
        if compute_relative:
            try:
                free_sasa = _compute_free_ligand_sasa(lig_res)
                if free_sasa > 0:
                    rel = ligand_sasa / free_sasa
            except Exception:
                rel = float("nan")

        out.append(LigandEnvResult(
            ligand_seqpos=lp,
            ligand_name3=lig_res.name3(),
            lig_dist=lig_dist,
            n_residues_within_5A=n5,
            n_residues_within_8A=n8,
            ligand_sasa=float(ligand_sasa),
            ligand_sasa_relative=rel,
            per_atom_sasa=per_atom,
        ))

    return out


def _compute_free_ligand_sasa(lig_res) -> float:
    """SASA of just the ligand residue, by itself in a fresh pose.

    Builds a single-residue pose with the same residue type and gets the
    full SASA. Used for the `ligand_sasa_relative` metric.
    """
    from protein_chisel.utils.pose import getSASA
    import pyrosetta

    # Build a fresh single-residue pose by appending a copy of the ligand
    p = pyrosetta.rosetta.core.pose.Pose()
    p.append_residue_by_jump(lig_res.clone(), 0)
    surf_vol = getSASA(p, resno=None, probe_radius=1.4)
    res = p.residue(1)
    return float(sum(surf_vol.surf(1, a) for a in range(1, res.natoms() + 1)))


__all__ = ["LigandEnvResult", "ligand_environment"]
