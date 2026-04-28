"""Backbone sanity checks: chainbreak, rCA_nonadj, term_mindist.

Modernized from process_diffusion3_outputs__REORG.py:
    - chainbreak: max sequential CA-CA distance (clean is ~3.8 Å)
    - rCA_nonadj: minimum CA-CA distance between non-adjacent residues
                  (catches collisions / overlapping chains)
    - term_mindist: min distance from N- or C-terminus CA to any ligand atom
                    (high values mean termini are sequestered away from active site)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BackboneSanityResult:
    chainbreak_max: float        # Å, expect ~3.8
    chainbreak_above_4_5: int    # count of CA-CA gaps > 4.5 Å
    rCA_nonadj_min: float        # Å, expect > ~5
    term_n_mindist_to_lig: float # Å (NaN if no ligand)
    term_c_mindist_to_lig: float # Å
    n_residues: int
    n_chains: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "backbone__chainbreak_max": self.chainbreak_max,
            "backbone__chainbreak_above_4_5": self.chainbreak_above_4_5,
            "backbone__rCA_nonadj_min": self.rCA_nonadj_min,
            "backbone__term_n_mindist_to_lig": self.term_n_mindist_to_lig,
            "backbone__term_c_mindist_to_lig": self.term_c_mindist_to_lig,
            "backbone__n_residues": self.n_residues,
            "backbone__n_chains": self.n_chains,
        }


def backbone_sanity(
    pdb_path: str | Path,
    params: list[str | Path] = (),
) -> BackboneSanityResult:
    """Compute backbone-sanity metrics on a PDB."""
    from protein_chisel.utils.pose import (
        init_pyrosetta, pose_from_file,
        find_ligand_seqpos, get_ligand_seqposes,
    )
    from protein_chisel.utils.geometry import (
        ca_coords, ca_ca_consecutive_distances, heavy_atom_coords,
    )

    init_pyrosetta(params=list(params))
    pose = pose_from_file(pdb_path)

    ca = ca_coords(pose)
    n_res = len(ca)

    # Chainbreak: max sequential CA-CA distance
    seq_dists = ca_ca_consecutive_distances(pose)
    chainbreak_max = float(max(seq_dists)) if seq_dists else 0.0
    chainbreak_above = int(sum(1 for d in seq_dists if d > 4.5))

    # rCA_nonadj: min distance between residues separated by ≥3 in sequence
    rca_nonadj_min = float("inf")
    if n_res >= 4:
        d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        # Mask out |i-j| < 3
        idx = np.arange(n_res)
        mask = np.abs(idx[:, None] - idx[None, :]) >= 3
        d_masked = np.where(mask, d, np.inf)
        rca_nonadj_min = float(d_masked.min())
    if not np.isfinite(rca_nonadj_min):
        rca_nonadj_min = float("nan")

    # Termini → ligand distance
    term_n = float("nan")
    term_c = float("nan")
    ligand_seqposes = get_ligand_seqposes(pose)
    if ligand_seqposes and n_res > 0:
        ligand_xyz: list[np.ndarray] = []
        for lp in ligand_seqposes:
            ligand_xyz.append(heavy_atom_coords(pose.residue(lp)))
        all_lig = np.vstack(ligand_xyz)
        first_ca = ca[0]
        last_ca = ca[-1]
        term_n = float(np.linalg.norm(all_lig - first_ca, axis=1).min())
        term_c = float(np.linalg.norm(all_lig - last_ca, axis=1).min())

    return BackboneSanityResult(
        chainbreak_max=chainbreak_max,
        chainbreak_above_4_5=chainbreak_above,
        rCA_nonadj_min=rca_nonadj_min,
        term_n_mindist_to_lig=term_n,
        term_c_mindist_to_lig=term_c,
        n_residues=n_res,
        n_chains=int(pose.num_chains()),
    )


__all__ = ["BackboneSanityResult", "backbone_sanity"]
