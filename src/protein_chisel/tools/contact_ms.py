"""Distance-weighted Contact Molecular Surface via py_contact_ms.

py_contact_ms (bcov77) computes ``contact_ms = area * exp(-0.5 * d**2)``,
giving a smoother interface-quality metric than Rosetta's
ContactMolecularSurface filter and supporting per-atom CMS values.

PyRosetta-FREE: this tool just needs numpy + py_contact_ms. So it can run
in any container that has those (esmc.sif, pyrosetta.sif both work).
We extract atomic coordinates from the PDB directly using io/pdb.

Note: when run in pyrosetta.sif there is no py_contact_ms; tools/run_in_sif
should route to esmc.sif (or any sif that has the package).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass
class ContactMSResult:
    total_cms: float
    n_binder_atoms: int
    n_target_atoms: int
    per_atom_cms_binder: list[float] = field(default_factory=list)
    per_atom_cms_target: list[float] = field(default_factory=list)

    def to_dict(self, prefix: str = "cms__") -> dict[str, float | int]:
        return {
            f"{prefix}total": self.total_cms,
            f"{prefix}n_binder_atoms": self.n_binder_atoms,
            f"{prefix}n_target_atoms": self.n_target_atoms,
        }


def _collect_atoms(
    pdb_path: str | Path,
    chains: Optional[set[str]] = None,
    residue_keys: Optional[set[tuple[str, str, int]]] = None,
    record_filter: Optional[str] = None,  # "ATOM" or "HETATM" or None
    skip_hydrogens: bool = True,
    skip_resnames: Iterable[str] = ("HOH",),
):
    """Extract (xyz, resnames, atom_names) arrays for a subset of atoms.

    Filters by chains, by explicit (chain, name3, resno) keys, and/or by
    record type. Returns three np.ndarray: xyz (N, 3), resnames (N,),
    atom_names (N,).
    """
    from protein_chisel.io.pdb import parse_atom_record

    skip_set = set(skip_resnames)
    xyzs: list[list[float]] = []
    resnames: list[str] = []
    atom_names: list[str] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            atom = parse_atom_record(line)
            if atom is None:
                continue
            if record_filter is not None and atom.record != record_filter:
                continue
            if chains is not None and atom.chain not in chains:
                continue
            if residue_keys is not None and (atom.chain, atom.res_name, atom.res_seq) not in residue_keys:
                continue
            if atom.res_name in skip_set:
                continue
            if skip_hydrogens and (atom.element == "H" or atom.name.startswith("H")):
                continue
            xyzs.append([atom.x, atom.y, atom.z])
            resnames.append(atom.res_name)
            atom_names.append(atom.name)
    return (
        np.asarray(xyzs, dtype=float),
        np.asarray(resnames),
        np.asarray(atom_names),
    )


def contact_ms_protein_ligand(
    pdb_path: str | Path,
    ligand_chain: Optional[str] = None,
    ligand_resname: Optional[str] = None,
) -> ContactMSResult:
    """CMS between the protein and a ligand.

    If both `ligand_chain` and `ligand_resname` are None, uses the first
    HETATM (excluding water).
    """
    import py_contact_ms as pcs

    # Determine ligand identity
    if ligand_chain is None or ligand_resname is None:
        from protein_chisel.io.pdb import find_ligand
        info = find_ligand(pdb_path)
        if info is None:
            return ContactMSResult(0.0, 0, 0)
        lc, lr, lresno = info
        ligand_chain = lc if ligand_chain is None else ligand_chain
        ligand_resname = lr if ligand_resname is None else ligand_resname
    else:
        lresno = None  # not strictly required if we filter by chain+resname

    # Binder = protein chain(s) (not the ligand chain)
    # Target = ligand atoms (HETATM with matching chain/resname)
    binder_xyz, binder_resnames, binder_atomnames = _collect_atoms(
        pdb_path, record_filter="ATOM", skip_hydrogens=True,
    )
    target_xyz, target_resnames, target_atomnames = _collect_atoms(
        pdb_path,
        chains={ligand_chain},
        record_filter="HETATM",
        skip_hydrogens=True,
    )
    target_mask = target_resnames == ligand_resname
    target_xyz = target_xyz[target_mask]
    target_resnames = target_resnames[target_mask]
    target_atomnames = target_atomnames[target_mask]

    if len(binder_xyz) == 0 or len(target_xyz) == 0:
        return ContactMSResult(0.0, len(binder_xyz), len(target_xyz))

    binder_radii = pcs.get_radii_from_names(
        list(binder_resnames), list(binder_atomnames)
    )
    target_radii = pcs.get_radii_from_names(
        list(target_resnames), list(target_atomnames)
    )

    # Atoms with no known radius (e.g. Zn, other metals not in py_contact_ms's
    # table) get radius 0 — they're silently dropped inside the calc. Filter
    # them on our side so n_atoms aligns with per-atom output length.
    binder_keep = binder_radii > 0
    target_keep = target_radii > 0
    binder_xyz = binder_xyz[binder_keep]
    binder_radii = binder_radii[binder_keep]
    target_xyz = target_xyz[target_keep]
    target_radii = target_radii[target_keep]

    if len(binder_xyz) == 0 or len(target_xyz) == 0:
        return ContactMSResult(0.0, len(binder_xyz), len(target_xyz))

    # py_contact_ms.calculate_contact_ms returns:
    #   (cms_total, per_atom_target_cms, calc)
    # The trailing `calc` is the MolecularSurfaceCalculator object; we drop it.
    out = pcs.calculate_contact_ms(
        binder_xyz, binder_radii, target_xyz, target_radii
    )
    total_cms = float(out[0])
    per_atom_target = out[1]

    return ContactMSResult(
        total_cms=total_cms,
        n_binder_atoms=int(len(binder_xyz)),
        n_target_atoms=int(len(target_xyz)),
        per_atom_cms_binder=[],  # not provided by py_contact_ms
        per_atom_cms_target=[float(x) for x in per_atom_target],
    )


__all__ = ["ContactMSResult", "contact_ms_protein_ligand"]
