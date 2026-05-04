"""Heavy-atom clash detection for designed PDBs.

Critical concern when ``--ligand_mpnn_use_side_chain_context=0`` is
used: MPNN doesn't see the catalytic sidechain rotamers, so it can
sample bulky residues that the side-chain packer then has to fit
around the catalytic atoms — sometimes producing clashes that the
packer can't resolve.

This module detects clashes between:
1. **Catalytic residue heavy atoms** (positions in ``catalytic_resnos``)
   and **designed residue sidechains** (any other protein residue).
2. **Ligand HETATM atoms** and **designed residue sidechains**.

A clash is defined as two heavy atoms within ``clash_distance`` (default
1.8 Å, generous to allow some rotamer slop) and not connected by a
bond.

Per-design output is the count of clashing atom pairs, plus a list of
the clashing positions for diagnostic logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


LOGGER = logging.getLogger("protein_chisel.structure.clash_check")


# Sidechain atom names per residue (heavy atoms only). Backbone N, CA,
# C, O are excluded — those are common to all and we don't want to
# count CA-CA distances as clashes between adjacent residues.
SIDECHAIN_ATOM_NAMES: dict[str, set[str]] = {
    "ALA": {"CB"},
    "ARG": {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"CB", "CG", "OD1", "ND2"},
    "ASP": {"CB", "CG", "OD1", "OD2"},
    "CYS": {"CB", "SG"},
    "GLN": {"CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"CB", "CG", "CD", "OE1", "OE2"},
    "GLY": set(),
    "HIS": {"CB", "CG", "ND1", "CE1", "NE2", "CD2"},
    "HID": {"CB", "CG", "ND1", "CE1", "NE2", "CD2"},   # HIS protonated on ND1
    "HIE": {"CB", "CG", "ND1", "CE1", "NE2", "CD2"},
    "HIP": {"CB", "CG", "ND1", "CE1", "NE2", "CD2"},
    "HIS_D": {"CB", "CG", "ND1", "CE1", "NE2", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "LYS": {"CB", "CG", "CD", "CE", "NZ"},
    "KCX": {"CB", "CG", "CD", "CE", "NZ", "OQ1", "OQ2"},  # carbamylated Lys
    "MET": {"CB", "CG", "SD", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"CB", "CG", "CD"},
    "SER": {"CB", "OG"},
    "THR": {"CB", "OG1", "CG2"},
    "TRP": {"CB", "CG", "CD1", "NE1", "CE2", "CD2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"CB", "CG1", "CG2"},
}


@dataclass
class ClashCheckResult:
    n_clashes: int = 0
    clashes_to_catalytic: int = 0
    clashes_to_ligand: int = 0
    clash_positions: list[tuple[int, int, float]] = field(default_factory=list)
    # (designed_resno, catalytic_or_ligand_resno_or_-1, distance)
    has_severe_clash: bool = False  # >= 1 clash <= 1.5 A

    def to_dict(self) -> dict:
        return {
            "clash__n_total": self.n_clashes,
            "clash__n_to_catalytic": self.clashes_to_catalytic,
            "clash__n_to_ligand": self.clashes_to_ligand,
            "clash__has_severe": int(self.has_severe_clash),
            "clash__detail": "; ".join(
                f"{a}-{b}({d:.2f})"
                for a, b, d in self.clash_positions[:5]
            ),
        }


def _read_atoms(pdb_path: Path) -> list[dict]:
    """Stdlib ATOM/HETATM parser, format-aware.

    Reads res_name from cols 16-20 — standard PDB has space at 16 plus
    3-char name at 17-19, Rosetta-extended fills all 5 cols (e.g.
    "HIS_D"). line[16:21].strip() returns the right thing for both.
    Chain stays at col 21.
    """
    out = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                d = {
                    "record": line[:6].strip(),
                    "atom_name": line[12:16].strip(),
                    "res_name": line[16:21].strip(),
                    "chain_id": line[21].strip(),
                    "res_seq": int(line[22:26].strip() or 0),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "element": line[76:78].strip(),
                }
            except (ValueError, IndexError):
                continue
            if d["element"] == "H":
                continue
            out.append(d)
    return out


def detect_clashes(
    pdb_path: Path,
    catalytic_resnos: Iterable[int],
    chain: str = "A",
    clash_distance: float = 1.8,
    severe_distance: float = 1.5,
) -> ClashCheckResult:
    """Detect heavy-atom clashes between catalytic + ligand and design.

    Returns a ``ClashCheckResult`` with counts and the worst clashes.
    """
    atoms = _read_atoms(pdb_path)
    if not atoms:
        return ClashCheckResult()

    cat_set = set(int(r) for r in catalytic_resnos)

    # Partition atoms
    catalytic_atoms = [
        a for a in atoms
        if (a["record"] == "ATOM" and a["chain_id"] == chain
            and a["res_seq"] in cat_set)
    ]
    ligand_atoms = [a for a in atoms if a["record"] == "HETATM"]
    designed_atoms = [
        a for a in atoms
        if (a["record"] == "ATOM" and a["chain_id"] == chain
            and a["res_seq"] not in cat_set
            and a["atom_name"] in SIDECHAIN_ATOM_NAMES.get(a["res_name"], set()))
    ]
    if not designed_atoms:
        return ClashCheckResult()

    res = ClashCheckResult()

    # Designed sidechain vs catalytic heavy atoms
    for d in designed_atoms:
        for c in catalytic_atoms:
            # Skip same residue + immediate neighbors (covalent contacts)
            if abs(d["res_seq"] - c["res_seq"]) <= 1:
                continue
            dx = d["x"] - c["x"]; dy = d["y"] - c["y"]; dz = d["z"] - c["z"]
            r2 = dx*dx + dy*dy + dz*dz
            if r2 < clash_distance * clash_distance:
                r = float(np.sqrt(r2))
                res.n_clashes += 1
                res.clashes_to_catalytic += 1
                if r < severe_distance:
                    res.has_severe_clash = True
                if len(res.clash_positions) < 20:
                    res.clash_positions.append((d["res_seq"], c["res_seq"], r))

    # Designed sidechain vs ligand HETATMs
    for d in designed_atoms:
        for L in ligand_atoms:
            dx = d["x"] - L["x"]; dy = d["y"] - L["y"]; dz = d["z"] - L["z"]
            r2 = dx*dx + dy*dy + dz*dz
            if r2 < clash_distance * clash_distance:
                r = float(np.sqrt(r2))
                res.n_clashes += 1
                res.clashes_to_ligand += 1
                if r < severe_distance:
                    res.has_severe_clash = True
                if len(res.clash_positions) < 20:
                    res.clash_positions.append((d["res_seq"], -1, r))

    return res


__all__ = ["ClashCheckResult", "SIDECHAIN_ATOM_NAMES", "detect_clashes"]
