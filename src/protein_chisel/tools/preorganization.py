"""Active-site preorganization via a repack ensemble.

A well-preorganized active site has small geometric variance under
perturbation. We approximate that by running N short PyRosetta
``PackRotamers`` trajectories with the catalytic residues AND the ligand
held *completely fixed* (so we never disturb the QC theozyme geometry).

The variance we report is over **non-catalytic** residue-position drift
near the active site — equivalent in spirit to "how much do the residues
around the active site move when we slightly perturb sidechain
rotamers?" Low variance = preorganized environment.

Run inside pyrosetta.sif.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


LOGGER = logging.getLogger("protein_chisel.preorganization")


@dataclass
class PreorganizationResult:
    n_ensemble: int
    catres_atom_variance: dict[int, float] = field(default_factory=dict)
    near_site_atom_variance: dict[int, float] = field(default_factory=dict)
    mean_catres_variance: float = 0.0
    mean_near_site_variance: float = 0.0
    catres_locked: bool = True       # True if catres were frozen during repack
    seed_pose_score: float = 0.0     # Rosetta total of the input pose

    def to_dict(self, prefix: str = "preorg__") -> dict[str, float | int]:
        return {
            f"{prefix}n_ensemble": self.n_ensemble,
            f"{prefix}mean_catres_variance": self.mean_catres_variance,
            f"{prefix}mean_near_site_variance": self.mean_near_site_variance,
            f"{prefix}catres_locked": self.catres_locked,
            f"{prefix}seed_pose_score": self.seed_pose_score,
        }


def preorganization(
    pdb_path: str | Path,
    n_ensemble: int = 20,
    near_site_radius: float = 8.0,
    params: list[str | Path] = (),
    catres_resnos: Optional[list[int]] = None,
    seed: int = 0,
) -> PreorganizationResult:
    """Run a small repack ensemble and report active-site variance.

    Args:
        pdb_path: design PDB.
        n_ensemble: number of independent repack samples (default 20).
        near_site_radius: Å — residues whose CA is within this of any
            catalytic atom are considered "near-site" and tracked for
            variance.
        params: ligand .params files (forwarded to init_pyrosetta).
        catres_resnos: explicit catres list; defaults to REMARK 666.
        seed: Rosetta-side seed (we add k for k=0..n_ensemble-1 to vary).
    """
    from protein_chisel.utils.pose import (
        init_pyrosetta, pose_from_file, get_default_scorefxn,
        find_ligand_seqpos,
    )
    from protein_chisel.io.pdb import parse_remark_666
    from protein_chisel.utils.geometry import (
        ca_coords, all_protein_heavy_coords, heavy_atom_coords,
    )
    import pyrosetta
    import pyrosetta.rosetta as ros

    init_pyrosetta(params=list(params))
    base_pose = pose_from_file(pdb_path)

    if catres_resnos is None:
        catres = parse_remark_666(pdb_path)
        catres_resnos = sorted(catres.keys())
    catres_set = set(catres_resnos)

    sfxn = get_default_scorefxn()
    seed_score = float(sfxn(base_pose))

    # Identify near-site residues — those whose CA is within `near_site_radius`
    # Å of any catalytic-residue heavy atom.
    catres_xyz: list[np.ndarray] = []
    for rno in catres_resnos:
        catres_xyz.append(heavy_atom_coords(base_pose.residue(rno)))
    if not catres_xyz:
        return PreorganizationResult(n_ensemble=0)
    cat_pts = np.vstack(catres_xyz)

    near_site_resnos: list[int] = []
    for r in base_pose.residues:
        if not r.is_protein():
            continue
        if r.seqpos() in catres_set:
            continue
        if not r.has("CA"):
            continue
        ca = r.xyz("CA")
        ca_arr = np.array([ca.x, ca.y, ca.z])
        if np.linalg.norm(cat_pts - ca_arr, axis=-1).min() <= near_site_radius:
            near_site_resnos.append(r.seqpos())

    LOGGER.info(
        "preorganization: %d catres (locked), %d near-site residues, n=%d",
        len(catres_resnos), len(near_site_resnos), n_ensemble,
    )

    # Build a repack TaskFactory: protein residues NOT in catres are repackable.
    task_factory = ros.core.pack.task.TaskFactory()
    task_factory.push_back(ros.core.pack.task.operation.RestrictToRepacking())

    # PreventRepacking on catres + ligand
    prevent_op = ros.core.pack.task.operation.PreventRepacking()
    prevent_selector = _make_resno_selector(base_pose, catres_resnos)
    plus_ligands = _make_ligand_selector(base_pose)
    or_sel = ros.core.select.residue_selector.OrResidueSelector(prevent_selector, plus_ligands)
    op_apply = ros.core.pack.task.operation.OperateOnResidueSubset(
        ros.core.pack.task.operation.PreventRepackingRLT(), or_sel,
    )
    task_factory.push_back(op_apply)

    # Track per-atom positions across ensemble. For each tracked residue, we
    # record the heavy-atom centroid xyz on each repack — variance of the
    # centroid across the ensemble is our metric.
    per_residue_centroids: dict[int, list[np.ndarray]] = {
        r: [] for r in catres_resnos + near_site_resnos
    }

    for i in range(n_ensemble):
        # Use a different seed per iteration so the random_pack actually moves.
        ros.numeric.random.rg().set_seed(int(seed + i + 1))
        pose_i = base_pose.clone()
        task = task_factory.create_task_and_apply_taskoperations(pose_i)
        packer = ros.protocols.minimization_packing.PackRotamersMover(sfxn, task)
        try:
            packer.apply(pose_i)
        except Exception as e:
            LOGGER.warning("repack iter %d failed: %s", i, e)
            continue
        for rno in per_residue_centroids:
            xyz = heavy_atom_coords(pose_i.residue(rno))
            if len(xyz) == 0:
                continue
            per_residue_centroids[rno].append(xyz.mean(axis=0))

    # Compute per-residue centroid variance (Å²)
    catres_var: dict[int, float] = {}
    near_var: dict[int, float] = {}
    for rno, points in per_residue_centroids.items():
        if len(points) < 2:
            continue
        arr = np.vstack(points)
        # centered → mean squared distance from centroid
        centered = arr - arr.mean(axis=0)
        var = float((centered ** 2).sum(axis=-1).mean())
        if rno in catres_set:
            catres_var[rno] = var
        else:
            near_var[rno] = var

    return PreorganizationResult(
        n_ensemble=n_ensemble,
        catres_atom_variance=catres_var,
        near_site_atom_variance=near_var,
        mean_catres_variance=float(np.mean(list(catres_var.values()))) if catres_var else 0.0,
        mean_near_site_variance=float(np.mean(list(near_var.values()))) if near_var else 0.0,
        catres_locked=True,
        seed_pose_score=seed_score,
    )


def _make_resno_selector(pose, resnos: list[int]):
    import pyrosetta.rosetta as ros
    sel = ros.core.select.residue_selector.ResidueIndexSelector()
    sel.set_index(",".join(str(r) for r in resnos))
    return sel


def _make_ligand_selector(pose):
    """Selector for all ligand residues (and virtual residues), to keep frozen."""
    import pyrosetta.rosetta as ros
    seqs: list[int] = []
    for r in pose.residues:
        if r.is_ligand() or r.is_virtual_residue():
            seqs.append(r.seqpos())
    sel = ros.core.select.residue_selector.ResidueIndexSelector()
    if seqs:
        sel.set_index(",".join(str(s) for s in seqs))
    else:
        # An empty ResidueIndexSelector raises later; emit a never-true selector
        sel.set_index("0")
    return sel


__all__ = ["PreorganizationResult", "preorganization"]
