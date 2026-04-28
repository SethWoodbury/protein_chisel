"""Catalytic-residue rotamer quality and bond-length sanity.

Per catalytic residue (from REMARK 666), report:
- cart_bonded_score    : Cartesian-bonded energy (high = strain)
- fa_dun_score         : rotamer probability score
- bondlen_max_dev      : max sidechain heavy-atom bond-length deviation
                         from an ideal A-X-A reference (catches broken
                         residues from sloppy diffusion / aggressive design)

Modernized port of process_diffusion3.{get_rosetta_scores, sidechain_connectivity}
and bcov ddg_per_res patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CatresQualityRow:
    resno: int
    name3: str
    cart_bonded: float
    fa_dun: float
    bondlen_max_dev: float
    sidechain_intact: bool  # bondlen_max_dev < threshold


@dataclass
class CatresQualityResult:
    per_residue: list[CatresQualityRow] = field(default_factory=list)
    cart_bonded_avg: float = 0.0
    cart_bonded_max: float = 0.0
    fa_dun_avg: float = 0.0
    fa_dun_max: float = 0.0
    bondlen_max_dev: float = 0.0
    n_broken_sidechains: int = 0
    n_residues: int = 0

    def to_dict(self, prefix: str = "catres__") -> dict[str, float | int]:
        return {
            f"{prefix}n_residues": self.n_residues,
            f"{prefix}cart_bonded_avg": self.cart_bonded_avg,
            f"{prefix}cart_bonded_max": self.cart_bonded_max,
            f"{prefix}fa_dun_avg": self.fa_dun_avg,
            f"{prefix}fa_dun_max": self.fa_dun_max,
            f"{prefix}bondlen_max_dev": self.bondlen_max_dev,
            f"{prefix}n_broken_sidechains": self.n_broken_sidechains,
        }


def catres_quality(
    pdb_path: str | Path,
    catres_resnos: Optional[list[int]] = None,
    params: list[str | Path] = (),
    bondlen_threshold: float = 0.10,  # Å — anything > this is suspicious
) -> CatresQualityResult:
    """Score catalytic-residue rotamer + bond quality.

    If `catres_resnos` is None, REMARK 666 from the PDB is used.
    """
    from protein_chisel.utils.pose import (
        init_pyrosetta, pose_from_file, get_default_scorefxn,
    )
    from protein_chisel.io.pdb import parse_remark_666

    init_pyrosetta(params=list(params))
    pose = pose_from_file(pdb_path)

    if catres_resnos is None:
        catres = parse_remark_666(pdb_path)
        catres_resnos = sorted(catres.keys())

    if not catres_resnos:
        return CatresQualityResult()

    sfxn = get_default_scorefxn()
    sfxn(pose)

    # cart_bonded needs cart score function
    import pyrosetta.rosetta as ros
    sfxn_cart = ros.core.scoring.ScoreFunctionFactory.create_score_function("beta_nov16_cart")
    sfxn_cart(pose)

    cb_term = ros.core.scoring.score_type_from_name("cart_bonded")
    fd_term = ros.core.scoring.score_type_from_name("fa_dun")

    rows: list[CatresQualityRow] = []
    for r_no in catres_resnos:
        if r_no < 1 or r_no > pose.size():
            continue
        r = pose.residue(r_no)
        cb = pose.energies().residue_total_energies(r_no).get(cb_term)
        fd = pose.energies().residue_total_energies(r_no).get(fd_term)
        bondlen_dev = _sidechain_bondlen_max_dev(r)
        rows.append(CatresQualityRow(
            resno=r_no,
            name3=r.name3(),
            cart_bonded=float(cb),
            fa_dun=float(fd),
            bondlen_max_dev=float(bondlen_dev),
            sidechain_intact=bondlen_dev < bondlen_threshold,
        ))

    if not rows:
        return CatresQualityResult()

    cb_vals = np.array([r.cart_bonded for r in rows])
    fd_vals = np.array([r.fa_dun for r in rows])
    bl_vals = np.array([r.bondlen_max_dev for r in rows])
    return CatresQualityResult(
        per_residue=rows,
        cart_bonded_avg=float(cb_vals.mean()),
        cart_bonded_max=float(cb_vals.max()),
        fa_dun_avg=float(fd_vals.mean()),
        fa_dun_max=float(fd_vals.max()),
        bondlen_max_dev=float(bl_vals.max()),
        n_broken_sidechains=int(sum(1 for r in rows if not r.sidechain_intact)),
        n_residues=len(rows),
    )


def _sidechain_bondlen_max_dev(residue) -> float:
    """Max sidechain heavy-atom bond-length deviation vs. an ideal reference.

    Ports process_diffusion3.sidechain_connectivity.
    """
    import pyrosetta as pyr
    name1 = residue.name1()
    if name1 in ("X", "Z", "B"):
        return 0.0
    try:
        ref_pose = pyr.pose_from_sequence("A" + name1 + "A")
        ref_res = ref_pose.residue(2)
    except Exception:
        return 0.0
    deviations: list[float] = []
    for an in range(1, residue.natoms() + 1):
        if residue.atom_type(an).element() == "H":
            continue
        try:
            atom_name = residue.atom_name(an).strip()
            if not ref_res.has(atom_name):
                continue
            ref_an = ref_res.atom_index(atom_name)
        except Exception:
            continue
        for nn in residue.bonded_neighbor(an):
            if residue.atom_type(nn).element() == "H":
                continue
            try:
                nn_name = residue.atom_name(nn).strip()
                if not ref_res.has(nn_name):
                    continue
                ref_nn = ref_res.atom_index(nn_name)
            except Exception:
                continue
            d_real = (residue.xyz(an) - residue.xyz(nn)).norm()
            d_ref = (ref_res.xyz(ref_an) - ref_res.xyz(ref_nn)).norm()
            deviations.append(abs(d_real - d_ref))
    return float(max(deviations)) if deviations else 0.0


__all__ = ["CatresQualityRow", "CatresQualityResult", "catres_quality"]
