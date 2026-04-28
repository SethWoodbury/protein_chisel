"""Theozyme-satisfaction metrics — frozen-catres approach.

Philosophy: catalytic residues come from quantum-chemistry-optimized
theozymes (e.g. NEB transition states). LigandMPNN and Rosetta don't
understand the QC physics, so we never relax those residues. Instead we
*measure* whether the design model preserved the QC geometry.

Inputs:
    design_pdb       : current design model (ATOM + HETATM with REMARK 666).
    theozyme_pdb     : optional reference theozyme PDB (the input to the
                       diffusion job). When provided we align to the
                       reference and compute deviations.
    fixed_atoms_json : optional RFdiffusion3-style ``{pdb: ["A92", ...]}``
                       JSON listing the residues whose coordinates were
                       supposed to be held fixed during diffusion.

Outputs (per design):
    motif_rmsd                     : Cα RMSD of all catalytic residues vs. theozyme.
    motif_heavy_rmsd               : heavy-atom RMSD over all catalytic residues.
    catres_max_atom_deviation      : worst per-atom drift Å.
    per_residue_rmsd               : dict {resno: rmsd Å}.
    per_residue_heavy_rmsd         : dict {resno: rmsd Å}.
    distance_to_ligand_atoms       : per (catres atom, ligand atom) pair distances.

If no theozyme reference is given, we skip alignment-based metrics and
return only the per-residue ligand-distance pairs (still useful as a
rough sanity check on rotamer drift since you can compare runs).

Pure-Python (numpy + io/pdb) — no PyRosetta required.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from protein_chisel.io.pdb import (
    AtomRecord, CatalyticResidue, parse_atom_record,
    parse_remark_666,
)


LOGGER = logging.getLogger("protein_chisel.theozyme_satisfaction")


@dataclass
class TheozymeSatisfactionResult:
    motif_rmsd: float = float("nan")          # Cα RMSD vs. theozyme reference
    motif_heavy_rmsd: float = float("nan")    # heavy-atom RMSD across catres
    catres_max_atom_deviation: float = float("nan")
    per_residue_rmsd: dict[int, float] = field(default_factory=dict)
    per_residue_heavy_rmsd: dict[int, float] = field(default_factory=dict)
    catres_to_ligand_distances: dict[str, float] = field(default_factory=dict)
    n_catalytic: int = 0

    def to_dict(self, prefix: str = "theozyme__") -> dict[str, float | int]:
        out: dict[str, float | int] = {
            f"{prefix}motif_rmsd": self.motif_rmsd,
            f"{prefix}motif_heavy_rmsd": self.motif_heavy_rmsd,
            f"{prefix}max_atom_deviation": self.catres_max_atom_deviation,
            f"{prefix}n_catalytic": self.n_catalytic,
        }
        for resno, rmsd in self.per_residue_rmsd.items():
            out[f"{prefix}per_res__{resno}__rmsd"] = rmsd
        for label, dist in self.catres_to_ligand_distances.items():
            out[f"{prefix}d__{label}"] = dist
        return out


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------


def _atoms_for_residues(
    pdb_path: str | Path,
    residues: set[tuple[str, int]],
    record_filter: Optional[str] = None,  # "ATOM" / "HETATM" / None
    skip_hydrogens: bool = True,
) -> dict[tuple[str, int], list[AtomRecord]]:
    """Return ``{(chain, resno): [AtomRecord, ...]}`` for the requested residues."""
    out: dict[tuple[str, int], list[AtomRecord]] = {}
    with open(pdb_path, "r") as fh:
        for line in fh:
            atom = parse_atom_record(line)
            if atom is None:
                continue
            if record_filter is not None and atom.record != record_filter:
                continue
            if skip_hydrogens and (atom.element == "H" or atom.name.startswith("H")):
                continue
            key = (atom.chain, atom.res_seq)
            if key not in residues:
                continue
            out.setdefault(key, []).append(atom)
    return out


def _coord(rec: AtomRecord) -> np.ndarray:
    return np.array([rec.x, rec.y, rec.z], dtype=float)


# ---------------------------------------------------------------------------
# Kabsch alignment
# ---------------------------------------------------------------------------


def _kabsch_align(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (rotation, translation, rmsd) that best aligns P onto Q.

    P and Q are (N, 3); both will be returned to their original shapes.
    Per-point indexing must be aligned beforehand.
    """
    if P.shape != Q.shape or len(P) < 3:
        raise ValueError(f"need matched shapes (≥3): {P.shape} vs {Q.shape}")
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(3); D[2, 2] = d
    R = Vt.T @ D @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    rotated = P @ R.T + t
    rmsd = float(np.sqrt(((rotated - Q) ** 2).sum(axis=-1).mean()))
    return R, t, rmsd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def theozyme_satisfaction(
    design_pdb: str | Path,
    theozyme_pdb: Optional[str | Path] = None,
    fixed_atoms_json: Optional[str | Path] = None,
    explicit_catres: Optional[Iterable[tuple[str, int]]] = None,
) -> TheozymeSatisfactionResult:
    """Score how well a design preserves its theozyme geometry.

    Args:
        design_pdb: design (or AF3-refined) PDB.
        theozyme_pdb: reference theozyme PDB. When provided, motif
            RMSD and per-atom deviations are computed after a Kabsch
            alignment over the catalytic-residue Cα atoms.
        fixed_atoms_json: optional JSON of the form
            ``{<pdb_path>: ["A92", "A136", ...]}`` (RFdiffusion3 style).
            If this references the design PDB, those residues are used
            as the catalytic set; otherwise all entries are merged.
        explicit_catres: optional iterable of (chain, resno) tuples; when
            provided, REMARK 666 is ignored.
    """
    design_pdb = Path(design_pdb)

    # Pick catalytic residue set in priority: explicit -> fixed_atoms -> REMARK 666
    catres: set[tuple[str, int]] = set()
    if explicit_catres is not None:
        catres = {(c, int(r)) for c, r in explicit_catres}
    elif fixed_atoms_json is not None:
        catres = _load_fixed_atoms_json(fixed_atoms_json, design_pdb)
    else:
        for cr in parse_remark_666(design_pdb).values():
            catres.add((cr.chain, cr.resno))

    if not catres:
        LOGGER.warning("no catalytic residues identified for %s", design_pdb)
        return TheozymeSatisfactionResult()

    # Coordinates (heavy atoms) for the catalytic residues in the design
    design_atoms = _atoms_for_residues(design_pdb, catres, record_filter="ATOM")

    # If we have a theozyme reference, align and compute RMSDs
    motif_rmsd = float("nan")
    motif_heavy_rmsd = float("nan")
    max_dev = float("nan")
    per_res_rmsd: dict[int, float] = {}
    per_res_heavy_rmsd: dict[int, float] = {}

    if theozyme_pdb is not None:
        theozyme_pdb = Path(theozyme_pdb)
        ref_atoms = _atoms_for_residues(theozyme_pdb, catres, record_filter="ATOM")
        ca_design, ca_ref, paired_residues = _matched_ca_pairs(design_atoms, ref_atoms)
        if len(ca_design) >= 3:
            R, t, motif_rmsd = _kabsch_align(np.asarray(ca_design), np.asarray(ca_ref))
            # Per-residue RMSDs (Cα and heavy atoms) AFTER alignment
            heavy_pairs = _matched_heavy_pairs(design_atoms, ref_atoms)
            d_pts = np.array([R @ np.array(p) + t for p, _ in heavy_pairs])
            r_pts = np.array([np.array(q) for _, q in heavy_pairs])
            if len(d_pts) > 0:
                deviations = np.linalg.norm(d_pts - r_pts, axis=-1)
                max_dev = float(deviations.max())
                motif_heavy_rmsd = float(np.sqrt((deviations ** 2).mean()))
            # Per-residue
            for (c, rno) in paired_residues:
                # Cα-only pair for this residue
                d_ca = _ca_xyz(design_atoms.get((c, rno), []))
                r_ca = _ca_xyz(ref_atoms.get((c, rno), []))
                if d_ca is not None and r_ca is not None:
                    rotated = R @ d_ca + t
                    per_res_rmsd[rno] = float(np.linalg.norm(rotated - r_ca))
                # heavy-atom RMSD per residue
                d_heavy, r_heavy = _matched_heavy_one_residue(
                    design_atoms.get((c, rno), []),
                    ref_atoms.get((c, rno), []),
                )
                if d_heavy is not None and r_heavy is not None:
                    rotated = (np.asarray(d_heavy) @ R.T) + t
                    diff = rotated - np.asarray(r_heavy)
                    per_res_heavy_rmsd[rno] = float(np.sqrt((diff ** 2).sum(axis=-1).mean()))

    # Catres → ligand distances (always computable)
    catres_to_lig_distances = _catres_ligand_distances(design_pdb, catres)

    return TheozymeSatisfactionResult(
        motif_rmsd=motif_rmsd,
        motif_heavy_rmsd=motif_heavy_rmsd,
        catres_max_atom_deviation=max_dev,
        per_residue_rmsd=per_res_rmsd,
        per_residue_heavy_rmsd=per_res_heavy_rmsd,
        catres_to_ligand_distances=catres_to_lig_distances,
        n_catalytic=len(catres),
    )


def _load_fixed_atoms_json(
    fixed_atoms_json: str | Path, design_pdb: Path
) -> set[tuple[str, int]]:
    payload = json.loads(Path(fixed_atoms_json).read_text())
    out: set[tuple[str, int]] = set()
    # Try matching the design path; otherwise merge all entries.
    keys_to_consume = []
    for k in payload:
        if Path(k).resolve() == design_pdb.resolve() or Path(k).name == design_pdb.name:
            keys_to_consume = [k]
            break
    if not keys_to_consume:
        keys_to_consume = list(payload.keys())
    for k in keys_to_consume:
        for label in payload[k]:
            label = label.strip()
            if not label:
                continue
            chain = label[0]
            try:
                resno = int(label[1:])
            except ValueError:
                continue
            out.add((chain, resno))
    return out


def _matched_ca_pairs(
    design_atoms: dict, ref_atoms: dict
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[str, int]]]:
    d_ca: list[np.ndarray] = []
    r_ca: list[np.ndarray] = []
    paired: list[tuple[str, int]] = []
    for key in design_atoms:
        if key not in ref_atoms:
            continue
        d = _ca_xyz(design_atoms[key])
        r = _ca_xyz(ref_atoms[key])
        if d is None or r is None:
            continue
        d_ca.append(d)
        r_ca.append(r)
        paired.append(key)
    return d_ca, r_ca, paired


def _ca_xyz(records: list[AtomRecord]) -> Optional[np.ndarray]:
    for r in records:
        if r.name == "CA":
            return _coord(r)
    return None


def _matched_heavy_pairs(
    design_atoms: dict, ref_atoms: dict
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pairs of (design_xyz, ref_xyz) for atoms that exist in both, by atom name."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for key, d_records in design_atoms.items():
        r_records = ref_atoms.get(key, [])
        r_by_name = {r.name: r for r in r_records}
        for d in d_records:
            r = r_by_name.get(d.name)
            if r is None:
                continue
            out.append((_coord(d), _coord(r)))
    return out


def _matched_heavy_one_residue(
    design_records: list[AtomRecord], ref_records: list[AtomRecord]
) -> tuple[Optional[list[np.ndarray]], Optional[list[np.ndarray]]]:
    if not design_records or not ref_records:
        return None, None
    r_by_name = {r.name: r for r in ref_records}
    d_pts: list[np.ndarray] = []
    r_pts: list[np.ndarray] = []
    for d in design_records:
        r = r_by_name.get(d.name)
        if r is None:
            continue
        d_pts.append(_coord(d))
        r_pts.append(_coord(r))
    if not d_pts:
        return None, None
    return d_pts, r_pts


def _catres_ligand_distances(
    pdb_path: str | Path, catres: set[tuple[str, int]]
) -> dict[str, float]:
    """For each catalytic atom, the min distance to any ligand heavy atom."""
    catres_atoms = _atoms_for_residues(pdb_path, catres, record_filter="ATOM")
    # Ligand: every non-water HETATM
    ligand_atoms: list[AtomRecord] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            atom = parse_atom_record(line)
            if atom is None or atom.record != "HETATM":
                continue
            if atom.res_name == "HOH":
                continue
            if atom.element == "H" or atom.name.startswith("H"):
                continue
            ligand_atoms.append(atom)
    if not ligand_atoms:
        return {}
    L_xyz = np.array([_coord(a) for a in ligand_atoms])
    out: dict[str, float] = {}
    for (chain, resno), records in catres_atoms.items():
        for d in records:
            label = f"{chain}{resno}_{d.name}"
            d_xyz = _coord(d)
            dists = np.linalg.norm(L_xyz - d_xyz, axis=-1)
            out[label] = float(dists.min())
    return out


__all__ = ["TheozymeSatisfactionResult", "theozyme_satisfaction"]
