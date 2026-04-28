"""Buried Unsatisfied (BUNS) hbond donors/acceptors.

Two flavors:

1. Plain BUNS: count protein heavy-atom polar donors/acceptors that are
   buried (low SASA) and have no hbond to anything.
2. Whitelist-aware: skip atoms in the user-supplied (resno, atom_name)
   whitelist — typically catalytic atoms which are *intentionally*
   unsatisfied (e.g. nucleophile lone pairs poised for attack).

This is the modernized port of Brian's
parse_target_buns_recalculate_white.py pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# Hbond donor / acceptor atoms by residue type. Backbone N (donor) and O
# (acceptor) are universal; sidechain donors/acceptors are listed here.
SIDECHAIN_DONORS: dict[str, tuple[str, ...]] = {
    "ARG": ("NE", "NH1", "NH2"),
    "ASN": ("ND2",),
    "GLN": ("NE2",),
    "HIS": ("ND1", "NE2"),
    "LYS": ("NZ",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TRP": ("NE1",),
    "TYR": ("OH",),
    "CYS": ("SG",),  # weak donor
}

SIDECHAIN_ACCEPTORS: dict[str, tuple[str, ...]] = {
    "ASN": ("OD1",),
    "ASP": ("OD1", "OD2"),
    "GLN": ("OE1",),
    "GLU": ("OE1", "OE2"),
    "HIS": ("ND1", "NE2"),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
    "MET": ("SD",),
    "CYS": ("SG",),
}


@dataclass
class BUNSResult:
    n_buried_unsat: int
    n_buried_polar_total: int
    buried_unsat_atoms: list[dict] = field(default_factory=list)
    n_whitelisted: int = 0  # atoms removed by whitelist

    def to_dict(self, prefix: str = "buns__") -> dict[str, int]:
        return {
            f"{prefix}n_buried_unsat": self.n_buried_unsat,
            f"{prefix}n_buried_polar_total": self.n_buried_polar_total,
            f"{prefix}n_whitelisted": self.n_whitelisted,
            # Common shorthand: "fraction unsat" out of buried polars
            f"{prefix}frac_unsat": (
                self.n_buried_unsat / self.n_buried_polar_total
                if self.n_buried_polar_total > 0
                else 0.0
            ),
        }


def buns(
    pdb_path: str | Path,
    params: list[str | Path] = (),
    sasa_buried_cutoff: float = 1.0,
    whitelist: Optional[Iterable[tuple[int, str]]] = None,
    include_backbone: bool = True,
) -> BUNSResult:
    """Compute BUNS — buried polar atoms with no hbond partners.

    Args:
        pdb_path: PDB file.
        params: ligand .params files for PyRosetta init.
        sasa_buried_cutoff: per-atom SASA threshold (Å²) below which an
            atom is considered buried. Default 1.0 (Rosetta convention
            uses ~0.1; 1.0 is permissive but matches bcov practice).
        whitelist: iterable of (resno, atom_name) tuples to ignore (e.g.
            catalytic nucleophile lone pairs that are intentionally unsat).
        include_backbone: if True, also consider backbone N (donor) and
            O (acceptor) atoms.
    """
    from protein_chisel.utils.pose import (
        init_pyrosetta, pose_from_file,
        get_per_atom_sasa, get_hbond_set,
    )

    init_pyrosetta(params=list(params))
    pose = pose_from_file(pdb_path)

    # Per-atom SASA with the bcov 2.8 Å probe — captures contact-region
    # accessibility, which is the right thing for "is this polar buried?"
    surf_vol = get_per_atom_sasa(pose, probe_radius=2.8)

    hbset = get_hbond_set(pose)

    # Build set of (resno, atom_name) that participate in any hbond
    hbonded_atoms: set[tuple[int, str]] = set()
    for i in range(1, hbset.nhbonds() + 1):
        hb = hbset.hbond(i)
        d_res = hb.don_res()
        a_res = hb.acc_res()
        d_atm = pose.residue(d_res).atom_name(hb.don_hatm()).strip()
        a_atm = pose.residue(a_res).atom_name(hb.acc_atm()).strip()
        hbonded_atoms.add((d_res, d_atm))
        hbonded_atoms.add((a_res, a_atm))
        # Donor heavy-atom is the parent atom of the H; record both names
        # in case downstream code keys on either.
        try:
            d_parent = pose.residue(d_res).atom_base(hb.don_hatm())
            d_parent_name = pose.residue(d_res).atom_name(d_parent).strip()
            hbonded_atoms.add((d_res, d_parent_name))
        except Exception:
            pass

    whitelist_set: set[tuple[int, str]] = set()
    if whitelist:
        whitelist_set = {(int(r), str(a).strip()) for r, a in whitelist}

    buried_polars: list[tuple[int, str, float]] = []
    for r in pose.residues:
        if not r.is_protein():
            continue
        n3 = r.name3()
        names: list[str] = []
        if include_backbone:
            if r.has("N"):
                names.append("N")
            if r.has("O"):
                names.append("O")
        names.extend(n for n in SIDECHAIN_DONORS.get(n3, ()) if r.has(n))
        names.extend(n for n in SIDECHAIN_ACCEPTORS.get(n3, ()) if r.has(n))
        for atom_name in set(names):
            try:
                a_idx = r.atom_index(atom_name)
            except RuntimeError:
                continue
            sasa = surf_vol.surf(r.seqpos(), a_idx)
            if sasa <= sasa_buried_cutoff:
                buried_polars.append((r.seqpos(), atom_name, sasa))

    # Now classify: unsat if not in hbonded_atoms, else satisfied.
    unsat: list[dict] = []
    n_white = 0
    for resno, atom_name, sasa in buried_polars:
        if (resno, atom_name) in whitelist_set:
            n_white += 1
            continue
        if (resno, atom_name) not in hbonded_atoms:
            unsat.append({
                "resno": resno,
                "atom_name": atom_name,
                "sasa": float(sasa),
                "name3": pose.residue(resno).name3(),
            })

    n_total_polars_minus_whitelist = len(buried_polars) - n_white

    return BUNSResult(
        n_buried_unsat=len(unsat),
        n_buried_polar_total=n_total_polars_minus_whitelist,
        buried_unsat_atoms=unsat,
        n_whitelisted=n_white,
    )


def whitelist_from_remark_666(pdb_path: str | Path) -> list[tuple[int, str]]:
    """Build a default whitelist of catalytic-residue sidechain donor/acceptor atoms.

    For each REMARK 666 motif residue, lists ALL of that residue's
    sidechain donor + acceptor atoms. This is the safe default — a
    catalytic histidine, glutamate, etc. is allowed to leave its
    catalytic-relevant atoms unsatisfied.

    Callers can refine this with a hand-curated atom list per project.
    """
    from protein_chisel.io.pdb import parse_remark_666

    catres = parse_remark_666(pdb_path)
    out: list[tuple[int, str]] = []
    for resno, c in catres.items():
        n3 = c.name3
        for a in SIDECHAIN_DONORS.get(n3, ()):
            out.append((resno, a))
        for a in SIDECHAIN_ACCEPTORS.get(n3, ()):
            out.append((resno, a))
    return out


__all__ = ["BUNSResult", "buns", "whitelist_from_remark_666"]
