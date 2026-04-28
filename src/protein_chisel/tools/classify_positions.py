"""Classify each residue into a structural category.

Done once per design / per pose. Output is a `PositionTable` whose rows
are consumed by every downstream tool that cares about residue role
(active site / first shell / pocket / buried / surface / ligand).

Inputs:
    pdb_path           : path to a PDB
    catres             : optional explicit dict {resno: CatalyticResidue}.
                         If None, REMARK 666 is parsed from the PDB; if
                         that's empty, you can supply --catres-spec for a
                         user-provided catalytic-residue spec.
    params             : list of paths or dirs to ligand .params files.

Output (PositionTable):
    pose_id, resno, chain, name3, name1, is_protein, is_catalytic, class,
    sasa, dist_ligand, dist_catalytic, ss, ss_reduced, in_pocket, phi, psi
    + extras: name3_3, has_ca, atom_count

`class` is one of:
    active_site      catalytic residue from REMARK 666 (frozen identity)
    first_shell      <= first_shell_radius Å from any ligand heavy atom
    pocket           in_pocket=True (set externally; here we leave False
                     unless caller injected a pocket map)
    buried           SASA < buried_sasa_cutoff
    surface          everything else
    ligand           non-protein residues
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from protein_chisel.io.pdb import (
    CatalyticResidue,
    parse_remark_666,
    parse_catres_spec,
)
from protein_chisel.io.schemas import PositionTable


# ---- Defaults -------------------------------------------------------------


@dataclass
class ClassifyConfig:
    first_shell_radius: float = 5.0       # Å — within this of any ligand atom
    buried_sasa_cutoff: float = 20.0      # Å² — strictly below = buried;
                                           # at-or-above = surface (no separate
                                           # surface_sasa_cutoff: same cutoff
                                           # both sides keeps classes disjoint)
    sasa_probe: float = 1.4               # Å (solvent probe)


# ---- Classifier ----------------------------------------------------------


_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def classify_positions(
    pdb_path: str | Path,
    pose_id: str = "design",
    catres: Optional[dict[int, CatalyticResidue]] = None,
    catres_spec: Optional[list[str]] = None,
    params: list[str | Path] = (),
    pocket_resnos: Optional[set[int]] = None,
    config: Optional[ClassifyConfig] = None,
) -> PositionTable:
    """Build a PositionTable for a single PDB.

    Args:
        pdb_path: PDB file.
        pose_id: identifier carried in the table (defaults to "design").
        catres: explicit catalytic residue dict; overrides REMARK 666.
        catres_spec: fallback list-of-strings (e.g. ["A94-96", "B101"])
            consumed when REMARK 666 is empty and catres is None.
        params: ligand `.params` files for PyRosetta init.
        pocket_resnos: optional set of seqposes flagged as pocket-lining
            (typically from fpocket on the same pose).
        config: ClassifyConfig.
    """
    cfg = config or ClassifyConfig()

    # -- Catalytic residues --------------------------------------------------
    if catres is None:
        catres = parse_remark_666(pdb_path)
    if not catres and catres_spec is not None:
        # Build pseudo entries from the spec — name3 will be filled in later.
        for ref in parse_catres_spec(catres_spec):
            catres[ref.resno] = CatalyticResidue(
                chain=ref.chain, name3="", resno=ref.resno,
                target_chain="?", target_name3="?", target_resno=-1,
                cst_no=0, cst_no_var=0,
            )
    catalytic_resnos: set[int] = set(catres.keys())

    # -- PyRosetta init ------------------------------------------------------
    from protein_chisel.utils.pose import init_pyrosetta, pose_from_file
    from protein_chisel.utils.pose import find_ligand_seqpos, get_ligand_seqposes
    from protein_chisel.utils.pose import get_per_residue_sasa
    from protein_chisel.utils.geometry import (
        dssp, min_distance_to_any, phi_psi,
    )

    init_pyrosetta(params=list(params))
    pose = pose_from_file(pdb_path)

    # -- Per-residue features ------------------------------------------------
    sasa = get_per_residue_sasa(pose, probe_radius=cfg.sasa_probe)
    pp = phi_psi(pose)
    ss_full = dssp(pose, reduced=False)
    ss_red = dssp(pose, reduced=True)

    # Ligand and protein lists
    ligand_seqposes = get_ligand_seqposes(pose)
    protein_seqposes = [r.seqpos() for r in pose.residues if r.is_protein()]

    # Distance maps
    dist_lig = min_distance_to_any(pose, ligand_seqposes, protein_seqposes)
    dist_cat = min_distance_to_any(pose, catalytic_resnos, protein_seqposes)

    pdb_info = pose.pdb_info()
    rows: list[dict] = []
    for r in pose.residues:
        sp = r.seqpos()
        chain = pdb_info.chain(sp) if pdb_info else " "
        name3 = r.name3()
        is_protein = bool(r.is_protein())

        if is_protein:
            cls = _classify_one(
                resno=sp,
                catalytic_resnos=catalytic_resnos,
                d_ligand=dist_lig.get(sp, float("nan")),
                in_pocket=(pocket_resnos is not None and sp in pocket_resnos),
                sasa=sasa.get(sp, 0.0),
                cfg=cfg,
            )
            ss = ss_full.get(sp, "L")
            sr = ss_red.get(sp, "L")
            phi, psi = pp.get(sp, (float("nan"), float("nan")))
            d_l = dist_lig.get(sp, float("nan"))
            d_c = dist_cat.get(sp, float("nan"))
        else:
            cls = "ligand"
            ss = "-"
            sr = "L"
            phi = float("nan")
            psi = float("nan")
            # Distances are meaningless for ligand-self rows.
            # Use NaN to avoid downstream filters mistaking the ligand for a
            # first-shell residue at distance 0.
            d_l = float("nan")
            d_c = float("nan")

        rows.append({
            "pose_id": pose_id,
            "resno": sp,
            "chain": chain or " ",
            "name3": name3,
            "name1": _AA3_TO_1.get(name3, "X"),
            "is_protein": is_protein,
            "is_catalytic": sp in catalytic_resnos,
            "class": cls,
            "sasa": float(sasa.get(sp, 0.0)),
            "dist_ligand": float(d_l) if d_l == d_l else float("nan"),
            "dist_catalytic": float(d_c) if d_c == d_c else float("nan"),
            "ss": ss,
            "ss_reduced": sr,
            "in_pocket": bool(pocket_resnos is not None and sp in pocket_resnos),
            "phi": float(phi) if phi == phi else float("nan"),
            "psi": float(psi) if psi == psi else float("nan"),
            "atom_count": int(r.natoms()),
            "has_ca": bool(r.has("CA")) if hasattr(r, "has") else False,
        })

    df = pd.DataFrame(rows)
    return PositionTable(df=df)


def _classify_one(
    resno: int,
    catalytic_resnos: set[int],
    d_ligand: float,
    in_pocket: bool,
    sasa: float,
    cfg: ClassifyConfig,
) -> str:
    if resno in catalytic_resnos:
        return "active_site"
    if d_ligand <= cfg.first_shell_radius:
        return "first_shell"
    if in_pocket:
        return "pocket"
    if sasa < cfg.buried_sasa_cutoff:
        return "buried"
    return "surface"


__all__ = ["ClassifyConfig", "classify_positions"]
