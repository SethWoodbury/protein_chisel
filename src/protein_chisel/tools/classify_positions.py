"""Classify each residue into a structural-functional category.

Done once per design / per pose. Output is a `PositionTable` whose rows
are consumed by every downstream tool that cares about residue role.

Two-pass directional taxonomy (matches Tawfik / Markin / Warshel
preorganization literature):

    primary_sphere    catalytic OR sidechain ≤ 4.5 Å of ligand/cofactor/
                      metal/μ-OH OR backbone polar contact (with
                      directional gate or explicit H-bond geometry)
    secondary_sphere  contacts a primary_sphere residue (sidechain or
                      backbone polar) OR sidechain ≤ 6.0 Å of ligand
                      AND pointing inward (θ ≤ 70° centroid) AND
                      interior (sidechain SASA fraction < 0.40)
    nearby_surface    CA ≤ 10.0 Å of any ligand atom (failed primary +
                      secondary)
    distal_buried     CA > 10.0 Å AND sidechain SASA fraction < 0.20
    distal_surface    CA > 10.0 Å AND sidechain SASA fraction ≥ 0.20
    ligand            non-protein

Continuous metrics are recorded as separate columns so we can re-bin
offline or apply gradient PLM weights without re-classifying. Soft
sigmoid scores per tier let downstream tools blend smoothly across
boundaries.

See docs/plans/directional_classification_plan.md for full design notes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
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
from protein_chisel.utils.sidechain_geometry import (
    BACKBONE_ATOMS,
    functional_atom_position,
    max_sasa_for,
    orientation_angle_deg,
    phantom_cb,
    ring_centroid,
    sidechain_atom_names,
    sidechain_centroid,
    sigmoid,
)


LOGGER = logging.getLogger("protein_chisel.tools.classify_positions")


# Schema bumped 2026-05-04 with the directional-taxonomy rewrite.
POSITION_TABLE_SCHEMA_VERSION: int = 2


# ---- Class names + legacy remap ------------------------------------------


# New 6-class taxonomy (literature-aligned).
NEW_CLASSES: tuple[str, ...] = (
    "primary_sphere",
    "secondary_sphere",
    "nearby_surface",
    "distal_buried",
    "distal_surface",
    "ligand",
)


# Legacy → new remap. Used by FusionConfig and other consumers that still
# pass legacy class strings (deprecated; emits a warning).
LEGACY_CLASS_REMAP: dict[str, str] = {
    "active_site": "primary_sphere",
    "first_shell": "primary_sphere",
    "pocket":      "secondary_sphere",
    "buried":      "distal_buried",
    "surface":     "distal_surface",
    # `ligand` is identical in both vocabularies.
}


def remap_legacy_class(name: str) -> str:
    """Map a legacy class string to the new vocabulary; pass through if new."""
    if name in NEW_CLASSES:
        return name
    new = LEGACY_CLASS_REMAP.get(name)
    if new is None:
        # Unknown — don't silently rewrite.
        return name
    return new


# ---- Config -------------------------------------------------------------


@dataclass
class ClassifyConfig:
    """Tunable parameters for the directional classifier."""

    # Tier-distance cutoffs (Å, heavy-atom min unless noted).
    primary_distance: float = 4.5            # Richter 2011 enzdes
    secondary_distance: float = 6.0          # Khersonsky/Tawfik HG3
    nearby_ca_distance: float = 10.0         # Richter 2011 design shell

    # Orientation gates (deg).
    orient_inward_max_deg: float = 70.0      # Chien & Huang EXIA 2012
    orient_outward_min_deg: float = 110.0    # diagnostic only
    orient_backbone_gate_deg: float = 110.0  # for backbone-polar primary path

    # Sidechain SASA fractions (Tien 2013 max-SASA reference).
    secondary_sasa_max_fraction: float = 0.40
    distal_buried_sasa_max_fraction: float = 0.20

    # SASA probe radius (unchanged).
    sasa_probe: float = 1.4

    # Sigmoid half-widths for soft membership scores (Å, deg).
    primary_halfwidth: float = 0.5
    secondary_halfwidth: float = 0.7
    nearby_halfwidth: float = 1.0
    theta_halfwidth: float = 10.0

    # Sanity gate: warn if catalytic residues are >this from ligand
    # (signals a poorly-placed ligand pose).
    poor_ligand_pose_warning_threshold: float = 5.0

    # Disulfide partner exclusion: SG-SG ≤ this Å treated as bonded.
    disulfide_distance: float = 2.3

    # Multi-primary coordination boost on secondary_score (1×, 1.3×, ...).
    secondary_coord_boost_per_extra: float = 0.3
    secondary_coord_boost_max: float = 1.5

    # Preset name (persisted into parquet metadata for deterministic
    # cross-user re-classification).
    preset_name: str = "pte_v1"


# ---- 3-letter to 1-letter map (covers HIS_D, KCX, etc.) ------------------


_AA3_TO_1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
    "HID": "H", "HIE": "H", "HIP": "H", "HIS_D": "H",
    "KCX": "K",
}


# Standard catalytic-metal element list (used to expand the ligand-atom
# set so His coordinating Zn registers as primary even when the substrate
# isn't immediately adjacent).
_METAL_ELEMENTS: frozenset[str] = frozenset({
    "ZN", "MN", "FE", "MG", "CA", "CO", "NI", "CU", "CD", "K", "NA",
})


# ---- Helpers ------------------------------------------------------------


def _residue_coords_by_atom(residue) -> dict[str, np.ndarray]:
    """Build {atom_name: xyz} for one PyRosetta residue (heavy atoms only)."""
    out: dict[str, np.ndarray] = {}
    for i in range(1, residue.natoms() + 1):
        atom = residue.atom(i)
        # Skip hydrogens.
        if residue.atom_is_hydrogen(i):
            continue
        name = residue.atom_name(i).strip()
        xyz = atom.xyz()
        out[name] = np.array([xyz.x, xyz.y, xyz.z], dtype=float)
    return out


def _backbone_polar_distances(
    coords: dict[str, np.ndarray], ligand_atom_pts: np.ndarray
) -> tuple[float, float]:
    """Return (d_backbone_N_lig, d_backbone_O_lig)."""
    out = []
    for name in ("N", "O"):
        if name not in coords or len(ligand_atom_pts) == 0:
            out.append(float("nan"))
            continue
        d = np.linalg.norm(ligand_atom_pts - coords[name][None, :], axis=-1)
        out.append(float(d.min()))
    return out[0], out[1]


def _sidechain_min_dist(
    sc_atoms: list[np.ndarray], target_pts: np.ndarray
) -> float:
    """Min distance from any sidechain atom to any target atom. NaN if either empty."""
    if not sc_atoms or len(target_pts) == 0:
        return float("nan")
    sc = np.stack(sc_atoms, axis=0)
    d = np.linalg.norm(sc[:, None, :] - target_pts[None, :, :], axis=-1)
    return float(d.min())


def _explicit_backbone_hbond_to_ligand(
    bb_coords: dict[str, np.ndarray],
    ligand_atom_pts: np.ndarray,
    ligand_polar_mask: np.ndarray,
) -> bool:
    """Approximate explicit H-bond geometry check on backbone N and O.

    For backbone N (donor): ligand polar atom should be in the direction
    NH would point. We approximate the NH direction as bisector of
    -(C_prev → N) and -(CA → N), which points away from the alpha-C.
    Without C_prev we approximate as -(CA → N) only.

    For backbone O (acceptor): ligand polar atom should lie within ~90°
    of the C=O extension (lone-pair direction). C=O direction = O - C.
    Geometry: ligand atom in the half-space pointing away from C from O.

    Returns True if either backbone polar atom satisfies a credible
    H-bond geometry to a ligand polar atom. Distance threshold 3.5 Å
    (heavy-atom donor-acceptor).
    """
    if len(ligand_atom_pts) == 0:
        return False
    polar_pts = ligand_atom_pts[ligand_polar_mask] if ligand_polar_mask.any() else ligand_atom_pts
    if len(polar_pts) == 0:
        return False
    HBOND_MAX = 3.5

    # Donor side (N).
    if "N" in bb_coords and "CA" in bb_coords:
        N = bb_coords["N"]; CA = bb_coords["CA"]
        nh_dir = N - CA
        nl = np.linalg.norm(nh_dir)
        if nl > 1e-9:
            nh_dir = nh_dir / nl
            for lp in polar_pts:
                v = lp - N
                d = float(np.linalg.norm(v))
                if d > HBOND_MAX or d < 1e-9:
                    continue
                cos = float(np.dot(nh_dir, v / d))
                # NH within ~60° of the donor-acceptor vector
                if cos > 0.5:
                    return True

    # Acceptor side (O), C=O lone-pair direction.
    if "O" in bb_coords and "C" in bb_coords:
        O = bb_coords["O"]; C = bb_coords["C"]
        co_dir = O - C
        nl = np.linalg.norm(co_dir)
        if nl > 1e-9:
            co_dir = co_dir / nl
            for lp in polar_pts:
                v = lp - O
                d = float(np.linalg.norm(v))
                if d > HBOND_MAX or d < 1e-9:
                    continue
                cos = float(np.dot(co_dir, v / d))
                # ligand on the C=O side (within ~90°)
                if cos > 0.0:
                    return True

    return False


def _freesasa_per_residue(
    pdb_path: str | Path, probe_radius: float = 1.4,
) -> dict[int, float]:
    """Per-residue SASA via freesasa, keyed by pose seqpos (1-indexed).

    Used as a fallback when Rosetta's DAlphaBall-backed SASA isn't
    available. Returns total residue SASA (Å²); mirrors what Rosetta's
    `get_per_residue_sasa` returns. The seqpos enumeration walks the
    protein chains in the order freesasa parsed them — for single-chain
    poses this matches PyRosetta directly.
    """
    import freesasa as _fs
    structure = _fs.Structure(str(pdb_path))
    params = _fs.Parameters({"algorithm": _fs.ShrakeRupley,
                              "probe-radius": float(probe_radius),
                              "n-points": 100})
    result = _fs.calc(structure, params)
    out: dict[int, float] = {}
    seqpos = 0
    last_chain_resno: tuple[Optional[str], Optional[int]] = (None, None)
    n_atoms = structure.nAtoms()
    for i in range(n_atoms):
        chain = structure.chainLabel(i)
        try:
            resno = int(structure.residueNumber(i))
        except (ValueError, TypeError):
            continue
        key = (chain, resno)
        if key != last_chain_resno:
            seqpos += 1
            last_chain_resno = key
            out[seqpos] = 0.0
        out[seqpos] += float(result.atomArea(i))
    return out


# ---- Public entrypoint --------------------------------------------------


def classify_positions(
    pdb_path: str | Path,
    pose_id: str = "design",
    catres: Optional[dict[int, CatalyticResidue]] = None,
    catres_spec: Optional[list[str]] = None,
    params: list[str | Path] = (),
    pocket_resnos: Optional[set[int]] = None,
    config: Optional[ClassifyConfig] = None,
) -> PositionTable:
    """Build a PositionTable for a single PDB using the directional taxonomy.

    Args:
        pdb_path: PDB file.
        pose_id: identifier carried in the table.
        catres: explicit catalytic residue dict; overrides REMARK 666.
        catres_spec: fallback list-of-strings (e.g. ["A94-96", "B101"]).
        params: ligand .params files for PyRosetta.
        pocket_resnos: optional fpocket-derived pocket-lining set.
        config: ClassifyConfig.
    """
    cfg = config or ClassifyConfig()

    # ---- Catalytic residues -----------------------------------------------
    if catres is None:
        catres = parse_remark_666(pdb_path)
    if not catres and catres_spec is not None:
        for ref in parse_catres_spec(catres_spec):
            catres[ref.resno] = CatalyticResidue(
                chain=ref.chain, name3="", resno=ref.resno,
                target_chain="?", target_name3="?", target_resno=-1,
                cst_no=0, cst_no_var=0,
            )
    catalytic_resnos: set[int] = set(catres.keys())
    catalytic_chain_resnos: set[tuple[str, int]] = {
        (cr.chain, cr.resno) for cr in catres.values()
    }
    if not catalytic_resnos:
        LOGGER.warning(
            "classify_positions: no catalytic residues; all → distal_*"
        )

    # ---- PyRosetta init -------------------------------------------------
    from protein_chisel.utils.pose import (
        init_pyrosetta, pose_from_file,
        get_ligand_seqposes, get_per_residue_sasa,
    )
    from protein_chisel.utils.geometry import dssp, phi_psi
    init_pyrosetta(params=list(params))
    pose = pose_from_file(pdb_path)
    pdb_info = pose.pdb_info()

    # ---- Per-residue features (inherited) -------------------------------
    # SASA: Rosetta's get_surf_vol depends on the DAlphaBall binary, which
    # fails in some apptainer containers (libgfortran.so.5 missing). Fall
    # back to freesasa (Shrake-Rupley) on failure — accurate enough for
    # the buried/surface classification.
    try:
        sasa = get_per_residue_sasa(pose, probe_radius=cfg.sasa_probe)
    except Exception as exc:
        LOGGER.warning("Rosetta SASA failed (%s); falling back to freesasa.", exc)
        sasa = _freesasa_per_residue(pdb_path, probe_radius=cfg.sasa_probe)
    pp = phi_psi(pose)
    ss_full = dssp(pose, reduced=False)
    ss_red = dssp(pose, reduced=True)

    # ---- Atom maps + ligand atom set ------------------------------------
    coords_by_seqpos: dict[int, dict[str, np.ndarray]] = {}
    for r in pose.residues:
        coords_by_seqpos[r.seqpos()] = _residue_coords_by_atom(r)

    ligand_seqposes = get_ligand_seqposes(pose)
    # Ligand-atom set: HETATMs + catalytic metals (any residue with an
    # atom whose element is in _METAL_ELEMENTS — covers Zn, Mn, etc.).
    lig_atom_pts: list[np.ndarray] = []
    lig_polar_flags: list[bool] = []
    for lsp in ligand_seqposes:
        for name, xyz in coords_by_seqpos.get(lsp, {}).items():
            lig_atom_pts.append(xyz)
            # Polar atom heuristic: starts with N, O, S, F, Cl
            lig_polar_flags.append(name[0] in ("N", "O", "S", "F"))
    # Plus metals from any residue (the binuclear Zn/Mn for PTE may be
    # parsed as separate residues with single metal atoms).
    for r in pose.residues:
        if r.is_protein():
            continue
        for i in range(1, r.natoms() + 1):
            elem = r.atom_type(i).element().strip().upper()
            if elem in _METAL_ELEMENTS:
                xyz = r.atom(i).xyz()
                lig_atom_pts.append(np.array([xyz.x, xyz.y, xyz.z]))
                lig_polar_flags.append(True)
    if not lig_atom_pts:
        LOGGER.warning("classify_positions: no ligand atoms; everything → distal_*")
        ligand_atom_pts = np.zeros((0, 3))
        ligand_polar_mask = np.zeros((0,), dtype=bool)
        ligand_centroid = None
    else:
        ligand_atom_pts = np.stack(lig_atom_pts, axis=0)
        ligand_polar_mask = np.array(lig_polar_flags, dtype=bool)
        ligand_centroid = ligand_atom_pts.mean(axis=0)

    # ---- Catalytic sidechain atoms (for d_sidechain_catshell) ----------
    cat_sidechain_pts: list[np.ndarray] = []
    for cat_resno in catalytic_resnos:
        if cat_resno not in coords_by_seqpos:
            continue
        cres = pose.residue(cat_resno)
        if not cres.is_protein():
            continue
        sc_names = sidechain_atom_names(cres.name3())
        for name in sc_names:
            if name in coords_by_seqpos[cat_resno]:
                cat_sidechain_pts.append(coords_by_seqpos[cat_resno][name])
    cat_sc_arr = (
        np.stack(cat_sidechain_pts, axis=0)
        if cat_sidechain_pts else np.zeros((0, 3))
    )

    # ---- Disulfide partners (for secondary path A exclusion) ------------
    cys_seqposes = [
        r.seqpos() for r in pose.residues
        if r.is_protein() and r.name3().upper() == "CYS"
    ]
    disulfide_partners: dict[int, int] = {}
    for i, sp_i in enumerate(cys_seqposes):
        sg_i = coords_by_seqpos[sp_i].get("SG")
        if sg_i is None:
            continue
        for sp_j in cys_seqposes[i+1:]:
            sg_j = coords_by_seqpos[sp_j].get("SG")
            if sg_j is None:
                continue
            d = float(np.linalg.norm(sg_i - sg_j))
            if d <= cfg.disulfide_distance:
                disulfide_partners[sp_i] = sp_j
                disulfide_partners[sp_j] = sp_i

    # ---- Per-residue continuous metrics ---------------------------------
    rows: list[dict] = []
    # Sanity gate: catalytic residues should be close to the ligand.
    cat_d_check: list[float] = []

    # Pre-compute every residue's metrics; classification (passes 1+2)
    # uses these accumulated values.
    metrics_by_seqpos: dict[int, dict] = {}
    for r in pose.residues:
        sp = r.seqpos()
        chain = pdb_info.chain(sp) if pdb_info else " "
        name3 = r.name3()
        coords = coords_by_seqpos.get(sp, {})
        is_protein = bool(r.is_protein())

        if not is_protein:
            metrics_by_seqpos[sp] = {
                "is_protein": False, "chain": chain, "name3": name3,
            }
            continue

        # Sidechain atoms present in this residue.
        sc_names = sidechain_atom_names(name3)
        sc_atoms = [coords[n] for n in sc_names if n in coords]

        # Backbone polar atoms.
        d_bb_N, d_bb_O = _backbone_polar_distances(coords, ligand_atom_pts)

        # Sidechain → ligand min dist.
        d_sc_lig = _sidechain_min_dist(sc_atoms, ligand_atom_pts)

        # Sidechain → catalytic-shell sidechain min dist.
        d_sc_cat = _sidechain_min_dist(sc_atoms, cat_sc_arr)

        # CA → min ligand atom.
        ca = coords.get("CA")
        if ca is not None and len(ligand_atom_pts) > 0:
            d_ca_lig = float(np.linalg.norm(
                ligand_atom_pts - ca[None, :], axis=-1
            ).min())
        else:
            d_ca_lig = float("nan")

        # Phantom CB for Gly, otherwise actual CB or sidechain centroid.
        if name3.upper() == "GLY":
            phantom = phantom_cb(coords["N"], ca, coords["C"]) \
                if all(k in coords for k in ("N", "C")) and ca is not None else None
            cb_atom = phantom
            sc_centroid_pt = phantom
        else:
            cb_atom = coords.get("CB")
            sc_centroid_pt = sidechain_centroid(coords, name3)

        # Functional atom (chemically meaningful).
        func_atom_pt = functional_atom_position(
            coords, name3, ligand_centroid=ligand_centroid,
        )

        # Ring centroid (aromatic only).
        ring_pt = ring_centroid(coords, name3)

        # Orientation angles.
        if ca is not None and ligand_centroid is not None:
            theta_cb = (
                orientation_angle_deg(ca, cb_atom, ligand_centroid)
                if cb_atom is not None else float("nan")
            )
            theta_centroid = (
                orientation_angle_deg(ca, sc_centroid_pt, ligand_centroid)
                if sc_centroid_pt is not None else float("nan")
            )
            theta_func = (
                orientation_angle_deg(ca, func_atom_pt, ligand_centroid)
                if func_atom_pt is not None else float("nan")
            )
            theta_aromatic = (
                orientation_angle_deg(ca, ring_pt, ligand_centroid)
                if ring_pt is not None else float("nan")
            )
        else:
            theta_cb = theta_centroid = theta_func = theta_aromatic = float("nan")

        # Sidechain SASA fraction. Use total residue SASA as proxy when we
        # don't have per-atom split (PyRosetta returns per-residue total).
        # This overstates fraction for residues with significant backbone
        # exposure but matches what's commonly available in Rosetta. Use
        # max-SASA from Tien 2013 reference table.
        max_sasa = max_sasa_for(name3)
        residue_sasa = float(sasa.get(sp, 0.0))
        sasa_sc_frac = min(1.0, residue_sasa / max_sasa) if max_sasa > 0 else 0.0

        # Backbone polar proximity & contact (with directional gate).
        bb_proximity = (
            (not np.isnan(d_bb_N) and d_bb_N <= cfg.primary_distance)
            or (not np.isnan(d_bb_O) and d_bb_O <= cfg.primary_distance)
        )
        # Directional gate: θ_centroid <= orient_backbone_gate_deg
        # OR explicit H-bond geometry.
        is_seq_neighbor_of_cat = any(
            abs(sp - cat_sp) <= 1 and chain == cat_chain
            for cat_chain, cat_sp in catalytic_chain_resnos
        )
        if bb_proximity and not is_seq_neighbor_of_cat:
            theta_pass = (
                not np.isnan(theta_centroid)
                and theta_centroid <= cfg.orient_backbone_gate_deg
            )
            hbond_pass = _explicit_backbone_hbond_to_ligand(
                coords, ligand_atom_pts, ligand_polar_mask,
            )
            bb_polar_contact = theta_pass or hbond_pass
            is_explicit_hbond = hbond_pass
        else:
            bb_polar_contact = False
            is_explicit_hbond = False

        # Track for sanity gate.
        if sp in catalytic_resnos and not np.isnan(d_sc_lig):
            cat_d_check.append(d_sc_lig)

        metrics_by_seqpos[sp] = {
            "is_protein": True,
            "chain": chain,
            "name3": name3,
            "name1": _AA3_TO_1.get(name3, "X"),
            "is_catalytic": sp in catalytic_resnos,
            "sasa": residue_sasa,
            "sasa_sc_fraction": sasa_sc_frac,
            "d_sidechain_lig": d_sc_lig,
            "d_sidechain_catshell": d_sc_cat,
            "d_backbone_N_lig": d_bb_N,
            "d_backbone_O_lig": d_bb_O,
            "d_ca_ligand": d_ca_lig,
            "theta_orient_cb_deg": theta_cb,
            "theta_orient_centroid_deg": theta_centroid,
            "theta_orient_functional_deg": theta_func,
            "theta_orient_aromatic_deg": theta_aromatic,
            "is_backbone_polar_proximity": bb_proximity,
            "is_backbone_polar_contact": bb_polar_contact,
            "is_explicit_hbond_to_ligand": is_explicit_hbond,
            "is_seq_neighbor_of_catalytic": is_seq_neighbor_of_cat,
            "is_disulfide_partner_of_primary": False,  # set in pass 2
            "_sc_atoms": sc_atoms,                    # internal, dropped before save
            "_disulfide_partner": disulfide_partners.get(sp),
        }

    # Sanity warning.
    if cat_d_check and min(cat_d_check) > cfg.poor_ligand_pose_warning_threshold:
        LOGGER.warning(
            "classify_positions: catalytic residue d_sidechain_lig min=%.2f > %.1f; "
            "ligand may be poorly placed",
            min(cat_d_check), cfg.poor_ligand_pose_warning_threshold,
        )

    # ---- Pass 1: primary_sphere -----------------------------------------
    primary_seqposes: set[int] = set()
    for sp, m in metrics_by_seqpos.items():
        if not m.get("is_protein"):
            continue
        if m["is_catalytic"]:
            primary_seqposes.add(sp)
            continue
        if not np.isnan(m["d_sidechain_lig"]) and m["d_sidechain_lig"] <= cfg.primary_distance:
            primary_seqposes.add(sp)
            continue
        if m["is_backbone_polar_contact"]:
            primary_seqposes.add(sp)
            continue

    # Build primary residues' sidechain + backbone polar atoms for pass 2.
    primary_atom_pts: dict[int, np.ndarray] = {}
    for sp in primary_seqposes:
        m = metrics_by_seqpos.get(sp)
        if not m:
            continue
        coords = coords_by_seqpos[sp]
        pts: list[np.ndarray] = list(m["_sc_atoms"])
        # Backbone polar atoms (N, O) of primary residues — codex r2 #1.
        for name in ("N", "O"):
            if name in coords:
                pts.append(coords[name])
        primary_atom_pts[sp] = (
            np.stack(pts, axis=0) if pts else np.zeros((0, 3))
        )

    # ---- Pass 2: secondary_sphere ---------------------------------------
    secondary_seqposes: set[int] = set()
    secondary_coord_count: dict[int, int] = {}
    for sp, m in metrics_by_seqpos.items():
        if not m.get("is_protein") or sp in primary_seqposes:
            continue
        sc_atoms = m["_sc_atoms"]
        # Path A: contacts a primary residue's sidechain or backbone polar.
        coord_count = 0
        if sc_atoms:
            sc_arr = np.stack(sc_atoms, axis=0)
            for prim_sp, prim_pts in primary_atom_pts.items():
                # Skip self; skip disulfide partner (codex r2 #8).
                if prim_sp == m.get("_disulfide_partner"):
                    metrics_by_seqpos[sp]["is_disulfide_partner_of_primary"] = True
                    continue
                if len(prim_pts) == 0:
                    continue
                d = np.linalg.norm(
                    sc_arr[:, None, :] - prim_pts[None, :, :], axis=-1
                )
                if float(d.min()) <= cfg.primary_distance:
                    coord_count += 1
        if coord_count >= 1:
            secondary_seqposes.add(sp)
            secondary_coord_count[sp] = coord_count
            continue

        # Path B: near-pocket inward-pointing interior residue.
        path_b = (
            not np.isnan(m["d_sidechain_lig"])
            and m["d_sidechain_lig"] <= cfg.secondary_distance
            and not np.isnan(m["theta_orient_centroid_deg"])
            and m["theta_orient_centroid_deg"] <= cfg.orient_inward_max_deg
            and m["sasa_sc_fraction"] < cfg.secondary_sasa_max_fraction
        )
        if path_b:
            secondary_seqposes.add(sp)
            secondary_coord_count[sp] = 0  # path B has no coordination count

    # ---- Build final rows -----------------------------------------------
    for sp, m in metrics_by_seqpos.items():
        chain = m.get("chain", " ")
        name3 = m.get("name3", "?")
        if not m.get("is_protein"):
            cls = "ligand"
            ss = "-"; sr = "L"
            phi = float("nan"); psi = float("nan")
            row = {
                "pose_id": pose_id, "resno": sp, "chain": chain or " ",
                "name3": name3, "name1": _AA3_TO_1.get(name3, "X"),
                "is_protein": False, "is_catalytic": False,
                "class": cls, "class_legacy": cls,
                "sasa": 0.0, "sasa_sc_fraction": 0.0,
                "dist_ligand": float("nan"), "dist_catalytic": float("nan"),
                "d_sidechain_lig": float("nan"), "d_sidechain_catshell": float("nan"),
                "d_sidechain_primary": float("nan"),
                "d_backbone_N_lig": float("nan"), "d_backbone_O_lig": float("nan"),
                "d_ca_ligand": float("nan"),
                "theta_orient_cb_deg": float("nan"),
                "theta_orient_centroid_deg": float("nan"),
                "theta_orient_functional_deg": float("nan"),
                "theta_orient_aromatic_deg": float("nan"),
                "is_backbone_polar_proximity": False,
                "is_backbone_polar_contact": False,
                "is_explicit_hbond_to_ligand": False,
                "is_seq_neighbor_of_catalytic": False,
                "is_disulfide_partner_of_primary": False,
                "secondary_coordination_count": 0,
                "primary_score": 0.0, "secondary_score": 0.0, "nearby_score": 0.0,
                "ss": ss, "ss_reduced": sr,
                "in_pocket": False,
                "phi": phi, "psi": psi,
                "atom_count": 0, "has_ca": False,
            }
            rows.append(row)
            continue

        # Discrete class.
        if sp in primary_seqposes:
            cls = "primary_sphere"
        elif sp in secondary_seqposes:
            cls = "secondary_sphere"
        elif (not np.isnan(m["d_ca_ligand"])
              and m["d_ca_ligand"] <= cfg.nearby_ca_distance):
            cls = "nearby_surface"
        elif m["sasa_sc_fraction"] < cfg.distal_buried_sasa_max_fraction:
            cls = "distal_buried"
        else:
            cls = "distal_surface"

        # d_sidechain_primary (computed for diagnostics).
        sc_atoms = m["_sc_atoms"]
        d_sc_prim = float("nan")
        if sc_atoms and primary_atom_pts and sp not in primary_seqposes:
            sc_arr = np.stack(sc_atoms, axis=0)
            min_d = float("inf")
            for prim_sp, prim_pts in primary_atom_pts.items():
                if prim_sp == m.get("_disulfide_partner") or len(prim_pts) == 0:
                    continue
                dd = float(np.linalg.norm(
                    sc_arr[:, None, :] - prim_pts[None, :, :], axis=-1
                ).min())
                if dd < min_d:
                    min_d = dd
            d_sc_prim = float(min_d) if min_d < float("inf") else float("nan")

        # Soft membership scores.
        primary_score = 0.0
        secondary_score = 0.0
        nearby_score = 0.0
        if not np.isnan(m["d_sidechain_lig"]):
            primary_score = float(sigmoid(
                (cfg.primary_distance - m["d_sidechain_lig"]) / cfg.primary_halfwidth
            ))
            sec_dist_score = float(sigmoid(
                (cfg.secondary_distance - m["d_sidechain_lig"]) / cfg.secondary_halfwidth
            ))
            theta_score = (
                float(sigmoid(
                    (cfg.orient_inward_max_deg - m["theta_orient_centroid_deg"])
                    / cfg.theta_halfwidth
                ))
                if not np.isnan(m["theta_orient_centroid_deg"]) else 0.0
            )
            secondary_score = (
                sec_dist_score * theta_score
                * max(0.0, 1.0 - primary_score)
            )
            # Multi-primary boost on path-A coordinates.
            n_coord = secondary_coord_count.get(sp, 0)
            if n_coord >= 1:
                boost = min(
                    cfg.secondary_coord_boost_max,
                    1.0 + cfg.secondary_coord_boost_per_extra * (n_coord - 1),
                )
                secondary_score = min(1.0, secondary_score * boost)
        if not np.isnan(m["d_ca_ligand"]):
            nearby_score = float(sigmoid(
                (cfg.nearby_ca_distance - m["d_ca_ligand"]) / cfg.nearby_halfwidth
            )) * max(0.0, 1.0 - primary_score - secondary_score)

        # Legacy class for back-compat: emit the old vocabulary mapping.
        # primary_sphere → first_shell, secondary_sphere → first_shell or
        # buried (based on which path triggered), nearby_surface → surface,
        # distal_buried → buried, distal_surface → surface.
        # Most consumers just use legacy strings as set-membership filters
        # so this mapping preserves their semantics.
        if cls == "primary_sphere" and m["is_catalytic"]:
            class_legacy = "active_site"
        elif cls == "primary_sphere":
            class_legacy = "first_shell"
        elif cls == "secondary_sphere":
            # Path B residues map to first_shell (close, inward); path A
            # residues map to buried (preorganization).
            class_legacy = "first_shell" if (
                not np.isnan(m["d_sidechain_lig"])
                and m["d_sidechain_lig"] <= cfg.secondary_distance
            ) else "buried"
        elif cls == "nearby_surface":
            class_legacy = "surface"
        elif cls == "distal_buried":
            class_legacy = "buried"
        else:
            class_legacy = "surface"

        ss = ss_full.get(sp, "L")
        sr = ss_red.get(sp, "L")
        phi, psi = pp.get(sp, (float("nan"), float("nan")))

        # is_disulfide_partner_of_primary may have been flipped during pass 2.
        is_dsf = bool(metrics_by_seqpos[sp].get("is_disulfide_partner_of_primary", False))

        row = {
            "pose_id": pose_id, "resno": sp, "chain": chain or " ",
            "name3": name3, "name1": _AA3_TO_1.get(name3, "X"),
            "is_protein": True, "is_catalytic": bool(m["is_catalytic"]),
            "class": cls, "class_legacy": class_legacy,
            "sasa": m["sasa"], "sasa_sc_fraction": m["sasa_sc_fraction"],
            "dist_ligand": m["d_sidechain_lig"]
                if not np.isnan(m["d_sidechain_lig"]) else float("nan"),
            "dist_catalytic": m["d_sidechain_catshell"]
                if not np.isnan(m["d_sidechain_catshell"]) else float("nan"),
            "d_sidechain_lig": m["d_sidechain_lig"],
            "d_sidechain_catshell": m["d_sidechain_catshell"],
            "d_sidechain_primary": d_sc_prim,
            "d_backbone_N_lig": m["d_backbone_N_lig"],
            "d_backbone_O_lig": m["d_backbone_O_lig"],
            "d_ca_ligand": m["d_ca_ligand"],
            "theta_orient_cb_deg": m["theta_orient_cb_deg"],
            "theta_orient_centroid_deg": m["theta_orient_centroid_deg"],
            "theta_orient_functional_deg": m["theta_orient_functional_deg"],
            "theta_orient_aromatic_deg": m["theta_orient_aromatic_deg"],
            "is_backbone_polar_proximity": m["is_backbone_polar_proximity"],
            "is_backbone_polar_contact": m["is_backbone_polar_contact"],
            "is_explicit_hbond_to_ligand": m["is_explicit_hbond_to_ligand"],
            "is_seq_neighbor_of_catalytic": m["is_seq_neighbor_of_catalytic"],
            "is_disulfide_partner_of_primary": is_dsf,
            "secondary_coordination_count": int(secondary_coord_count.get(sp, 0)),
            "primary_score": primary_score,
            "secondary_score": secondary_score,
            "nearby_score": nearby_score,
            "ss": ss, "ss_reduced": sr,
            "in_pocket": bool(pocket_resnos is not None and sp in pocket_resnos),
            "phi": float(phi) if phi == phi else float("nan"),
            "psi": float(psi) if psi == psi else float("nan"),
            "atom_count": int(pose.residue(sp).natoms()),
            "has_ca": bool(pose.residue(sp).has("CA"))
                if hasattr(pose.residue(sp), "has") else False,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Diagnostics: per-class counts.
    LOGGER.info(
        "classify_positions: classes = %s",
        df[df["is_protein"]]["class"].value_counts().to_dict(),
    )

    # Stash config snapshot in df.attrs (parquet preserves via pyarrow if
    # supported; fallback at to_parquet adds a sidecar).
    df.attrs["classify_config"] = json.dumps(asdict(cfg))
    df.attrs["schema_version"] = POSITION_TABLE_SCHEMA_VERSION

    return PositionTable(df=df)


__all__ = [
    "POSITION_TABLE_SCHEMA_VERSION",
    "NEW_CLASSES",
    "LEGACY_CLASS_REMAP",
    "ClassifyConfig",
    "classify_positions",
    "remap_legacy_class",
]
