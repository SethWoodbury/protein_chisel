"""Chemical interaction detection + strength-weighted scoring.

Detects (binary):
- hbonds (PyRosetta HBondSet, with energies in kcal/mol).
- salt bridges (positive sidechain N within cutoff of negative carboxylate O).
- π-π (aromatic ring centroid distance + plane angle: stacked / tilted / t_shape).
- π-cation (aromatic ring centroid within cutoff of cation atom).

Strength layer (``interaction_strengths()``):
- Per detected interaction, compute a soft strength using
  ``exp(-(d - d0)^2 / (2 σ^2))`` with type-specific d0 and σ
  (chosen so a typical optimal interaction scores ≈ 1.0 and weak
  interactions decay smoothly to ~0).
- Aggregate to per-residue and per-type rollups, suitable as
  Pareto objectives, MH energy components, or ML features.

PyRosetta-bound; run inside pyrosetta.sif.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# Aromatic ring atoms by residue type. The "ring" tuple lists the heavy
# atoms forming the canonical aromatic ring; we use these to compute the
# ring centroid and plane normal.
AROMATIC_RING_ATOMS: dict[str, tuple[str, ...]] = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "NE1", "CE2", "CD2"),  # 5-membered indole ring
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),  # imidazole
}


POSITIVE_SIDECHAIN_ATOMS: dict[str, tuple[str, ...]] = {
    "ARG": ("NH1", "NH2", "NE"),       # guanidinium
    "LYS": ("NZ",),                     # ε-amine
    "HIS": ("NE2", "ND1"),              # protonatable nitrogens
}


NEGATIVE_SIDECHAIN_ATOMS: dict[str, tuple[str, ...]] = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
}


# ---------------------------------------------------------------------------


@dataclass
class InteractionsResult:
    hbonds: list[dict] = field(default_factory=list)
    salt_bridges: list[dict] = field(default_factory=list)
    pi_pi: list[dict] = field(default_factory=list)
    pi_cation: list[dict] = field(default_factory=list)

    def summary(self, prefix: str = "interact__") -> dict[str, int | float]:
        return {
            f"{prefix}n_hbonds": len(self.hbonds),
            f"{prefix}n_salt_bridges": len(self.salt_bridges),
            f"{prefix}n_pi_pi": len(self.pi_pi),
            f"{prefix}n_pi_cation": len(self.pi_cation),
            f"{prefix}sum_hbond_energy": sum(h["energy"] for h in self.hbonds),
        }


def chemical_interactions(
    pdb_path: str | Path,
    params: list[str | Path] = (),
    salt_bridge_cutoff: float = 4.0,            # Å between charged atoms
    pi_pi_centroid_cutoff: float = 6.0,         # Å between ring centroids
    pi_cation_cutoff: float = 6.0,              # Å between ring centroid and cation
) -> InteractionsResult:
    """Detect hbonds, salt bridges, π-π, π-cation interactions in a pose."""
    from protein_chisel.utils.pose import (
        init_pyrosetta, pose_from_file,
        get_hbond_set, hbonds_as_dicts,
    )

    init_pyrosetta(params=list(params))
    pose = pose_from_file(pdb_path)

    res = InteractionsResult()

    # --- hbonds (canonical bcov pattern via utils/pose) -------------------
    hbset = get_hbond_set(pose)
    res.hbonds = hbonds_as_dicts(pose, hbset)

    # --- salt bridges -----------------------------------------------------
    res.salt_bridges = _detect_salt_bridges(pose, cutoff=salt_bridge_cutoff)

    # --- π-π and π-cation -------------------------------------------------
    aromatic_rings = _build_aromatic_rings(pose)
    res.pi_pi = _detect_pi_pi(aromatic_rings, cutoff=pi_pi_centroid_cutoff)
    res.pi_cation = _detect_pi_cation(pose, aromatic_rings, cutoff=pi_cation_cutoff)
    return res


# ---------------------------------------------------------------------------
# Salt bridges
# ---------------------------------------------------------------------------


def _detect_salt_bridges(pose, cutoff: float) -> list[dict]:
    """Salt bridge = positive sidechain N within `cutoff` Å of negative O."""
    pos_atoms: list[tuple[int, str, str, np.ndarray]] = []
    neg_atoms: list[tuple[int, str, str, np.ndarray]] = []

    for r in pose.residues:
        if not r.is_protein():
            continue
        n3 = r.name3()
        atoms_pos = POSITIVE_SIDECHAIN_ATOMS.get(n3, ())
        atoms_neg = NEGATIVE_SIDECHAIN_ATOMS.get(n3, ())
        for atom_name in atoms_pos:
            if r.has(atom_name):
                xyz = r.xyz(atom_name)
                pos_atoms.append((r.seqpos(), n3, atom_name, np.array([xyz.x, xyz.y, xyz.z])))
        for atom_name in atoms_neg:
            if r.has(atom_name):
                xyz = r.xyz(atom_name)
                neg_atoms.append((r.seqpos(), n3, atom_name, np.array([xyz.x, xyz.y, xyz.z])))

    out: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()
    for (psp, pname3, patm, pxyz) in pos_atoms:
        for (nsp, nname3, natm, nxyz) in neg_atoms:
            if psp == nsp:
                continue
            d = float(np.linalg.norm(pxyz - nxyz))
            if d > cutoff:
                continue
            key = tuple(sorted((psp, nsp)))  # type: ignore[assignment]
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            out.append({
                "pos_res": psp,
                "pos_name3": pname3,
                "pos_atom": patm,
                "neg_res": nsp,
                "neg_name3": nname3,
                "neg_atom": natm,
                "distance": d,
            })
    return out


# ---------------------------------------------------------------------------
# Aromatic rings (centroid + normal)
# ---------------------------------------------------------------------------


def _build_aromatic_rings(pose) -> list[dict]:
    """Return list of {resno, name3, centroid, normal} for aromatic rings."""
    out: list[dict] = []
    for r in pose.residues:
        if not r.is_protein():
            continue
        atoms = AROMATIC_RING_ATOMS.get(r.name3())
        if not atoms:
            continue
        coords: list[np.ndarray] = []
        for name in atoms:
            if not r.has(name):
                continue
            xyz = r.xyz(name)
            coords.append(np.array([xyz.x, xyz.y, xyz.z]))
        if len(coords) < 3:
            continue
        coords_arr = np.vstack(coords)
        centroid = coords_arr.mean(axis=0)
        # Plane normal via SVD on centered coords; smallest singular vector.
        centered = coords_arr - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        out.append({
            "resno": r.seqpos(),
            "name3": r.name3(),
            "centroid": centroid,
            "normal": normal,
        })
    return out


def _detect_pi_pi(rings: list[dict], cutoff: float) -> list[dict]:
    """Pairs of aromatic rings within centroid distance < cutoff."""
    out: list[dict] = []
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            a = rings[i]
            b = rings[j]
            d = float(np.linalg.norm(a["centroid"] - b["centroid"]))
            if d > cutoff:
                continue
            # Plane angle (radians, then degrees) — 0° = stacked, 90° = T-shape.
            cos = abs(float(np.dot(a["normal"], b["normal"])))
            cos = max(-1.0, min(1.0, cos))
            angle_deg = float(np.degrees(np.arccos(cos)))
            geom = "stacked" if angle_deg < 30.0 else ("t_shape" if angle_deg > 60.0 else "tilted")
            out.append({
                "res_a": a["resno"],
                "name3_a": a["name3"],
                "res_b": b["resno"],
                "name3_b": b["name3"],
                "centroid_distance": d,
                "plane_angle_deg": angle_deg,
                "geometry": geom,
            })
    return out


def _detect_pi_cation(pose, rings: list[dict], cutoff: float) -> list[dict]:
    """Aromatic ring centroid within `cutoff` Å of a positively charged N."""
    out: list[dict] = []
    cations: list[tuple[int, str, str, np.ndarray]] = []
    for r in pose.residues:
        if not r.is_protein():
            continue
        for atom_name in POSITIVE_SIDECHAIN_ATOMS.get(r.name3(), ()):
            if r.has(atom_name):
                xyz = r.xyz(atom_name)
                cations.append((r.seqpos(), r.name3(), atom_name, np.array([xyz.x, xyz.y, xyz.z])))

    for ring in rings:
        for (csp, cname3, catm, cxyz) in cations:
            if csp == ring["resno"]:
                continue
            d = float(np.linalg.norm(ring["centroid"] - cxyz))
            if d > cutoff:
                continue
            out.append({
                "ring_res": ring["resno"],
                "ring_name3": ring["name3"],
                "cation_res": csp,
                "cation_name3": cname3,
                "cation_atom": catm,
                "distance": d,
            })
    return out


# ---------------------------------------------------------------------------
# Strength-weighted layer (no extra deps)
# ---------------------------------------------------------------------------


# Canonical (mean, sigma) per interaction type, in Å, picked so a typical
# optimal interaction scores ≈ 1.0 (Gaussian centered on d0). The "strength"
# of a detected interaction at distance d is ``exp(-(d - d0)^2 / (2 σ^2))``.
# σ values are deliberately permissive (~ half the canonical cutoff) so the
# decay is gentle rather than a step.
#
# Note: hbond strength uses an *energy* proxy (Rosetta hbond_sc) rather than
# distance, so there is no hbond entry here.
INTERACTION_GEOMETRY = {
    "salt_bridge":   {"d0": 3.0, "sigma": 0.50},  # N...O distance
    "pi_pi":         {"d0": 4.0, "sigma": 0.80},  # ring centroid distance
    "pi_cation":     {"d0": 4.0, "sigma": 0.80},  # centroid...cation
}


@dataclass
class InteractionStrengthResult:
    """Strength-weighted aggregation of an InteractionsResult."""

    by_type_strength_sum: dict[str, float] = field(default_factory=dict)
    by_type_count: dict[str, int] = field(default_factory=dict)
    per_residue_strength: dict[int, float] = field(default_factory=dict)
    per_residue_strength_by_type: dict[int, dict[str, float]] = field(default_factory=dict)
    weighted_hbond_energy: float = 0.0  # Σ -energy_i × strength_i
    rows: list[dict] = field(default_factory=list)  # per-interaction with strength

    def to_dict(self, prefix: str = "interact_strength__") -> dict[str, float | int]:
        out: dict[str, float | int] = {}
        for typ, s in self.by_type_strength_sum.items():
            out[f"{prefix}{typ}__sum"] = s
        for typ, n in self.by_type_count.items():
            out[f"{prefix}{typ}__count"] = n
        out[f"{prefix}weighted_hbond_energy"] = self.weighted_hbond_energy
        return out


def _gauss_strength(d: float, d0: float, sigma: float) -> float:
    if not np.isfinite(d):
        return 0.0
    z = (d - d0) / max(sigma, 1e-6)
    return float(np.exp(-0.5 * z * z))


def interaction_strengths(
    interactions: "InteractionsResult",
    geometry: Optional[dict] = None,
    pi_pi_angle_factor: bool = True,
) -> InteractionStrengthResult:
    """Compute strength-weighted scores from binary detection results.

    Args:
        interactions: result from ``chemical_interactions(pdb)``.
        geometry: per-interaction-type overrides. Deep-merged with the
            default ``INTERACTION_GEOMETRY``: ``{"salt_bridge": {"sigma": 0.7}}``
            keeps the default ``d0`` and only overrides ``sigma``.
        pi_pi_angle_factor: when True, multiplies π-π strength by an
            angle factor that down-weights "tilted" geometries (peaks at
            stacked 0° and t-shape 90°, half-strength at 45°).
    """
    # Deep merge so callers can partially override per-type parameters
    # without dropping the rest of that type's defaults.
    geom: dict[str, dict[str, float]] = {
        k: dict(v) for k, v in INTERACTION_GEOMETRY.items()
    }
    if geometry:
        for typ, overrides in geometry.items():
            geom.setdefault(typ, {}).update(overrides)

    # Always include hbond in the type tallies even though it doesn't have
    # a geometry entry (it uses an energy proxy).
    type_keys = list(geom.keys()) + ["hbond"]
    by_type_sum: dict[str, float] = {k: 0.0 for k in type_keys}
    by_type_count: dict[str, int] = {k: 0 for k in type_keys}
    per_residue_sum: dict[int, float] = {}
    per_residue_by_type: dict[int, dict[str, float]] = {}
    rows: list[dict] = []
    weighted_hbond_e = 0.0

    def _add_per_residue(resno: int, typ: str, strength: float) -> None:
        per_residue_sum[resno] = per_residue_sum.get(resno, 0.0) + strength
        per_residue_by_type.setdefault(resno, {}).setdefault(typ, 0.0)
        per_residue_by_type[resno][typ] += strength

    # Hbonds — distance from acceptor xyz to donor heavy atom xyz isn't
    # available in the dict (we have indices but not coords); use the
    # Rosetta hbond energy as a proxy when present, mapped through a
    # softplus so very negative energies score ~1 and 0 → 0.
    for h in interactions.hbonds:
        # Map Rosetta hbond energy (kcal/mol, typically -2..0) to (0, 1].
        # 1 / (1 + exp(2 + 4*e)) — at e = -2: ≈ 0.5; at e = -3: ≈ 0.99.
        e = float(h.get("energy", 0.0))
        # Guard NaN/non-finite (would poison the entire aggregate sum).
        # Also clamp positive energies (repulsive hbonds — unphysical for
        # detected hbonds but seen in mis-parsed input) to zero strength.
        if not np.isfinite(e) or e >= 0.0:
            s = 0.0
        else:
            s = float(1.0 / (1.0 + np.exp(2.0 + 4.0 * e)))
        by_type_sum["hbond"] += s
        by_type_count["hbond"] += 1
        if np.isfinite(e):
            weighted_hbond_e += -e * s  # bigger |energy| → bigger weighted total
        for resno in (h["donor_res"], h["acceptor_res"]):
            _add_per_residue(resno, "hbond", s)
        rows.append({"type": "hbond", "strength": s, **h})

    # Salt bridges
    for sb in interactions.salt_bridges:
        d = float(sb["distance"])
        s = _gauss_strength(d, **geom["salt_bridge"])
        by_type_sum["salt_bridge"] += s
        by_type_count["salt_bridge"] += 1
        for resno in (sb["pos_res"], sb["neg_res"]):
            _add_per_residue(resno, "salt_bridge", s)
        rows.append({"type": "salt_bridge", "strength": s, **sb})

    # π-π
    for pp in interactions.pi_pi:
        d = float(pp["centroid_distance"])
        ang = float(pp.get("plane_angle_deg", 0.0))
        s = _gauss_strength(d, **geom["pi_pi"])
        if pi_pi_angle_factor:
            # Aromatic stacking peaks at 0° (face-to-face) and 90° (T-shape);
            # the bad geometry is the in-between 45° tilted case. Use
            # 0.5 + 0.5·|cos(2θ)|: 1 at 0°/90°, 0.5 at 45°. Guard NaN angles.
            if np.isfinite(ang):
                ang_factor = 0.5 + 0.5 * abs(np.cos(2.0 * np.deg2rad(ang)))
                s *= float(ang_factor)
            else:
                s = 0.0
        by_type_sum["pi_pi"] += s
        by_type_count["pi_pi"] += 1
        for resno in (pp["res_a"], pp["res_b"]):
            _add_per_residue(resno, "pi_pi", s)
        rows.append({"type": "pi_pi", "strength": s, **pp})

    # π-cation
    for pc in interactions.pi_cation:
        d = float(pc["distance"])
        s = _gauss_strength(d, **geom["pi_cation"])
        by_type_sum["pi_cation"] += s
        by_type_count["pi_cation"] += 1
        for resno in (pc["ring_res"], pc["cation_res"]):
            _add_per_residue(resno, "pi_cation", s)
        rows.append({"type": "pi_cation", "strength": s, **pc})

    return InteractionStrengthResult(
        by_type_strength_sum=by_type_sum,
        by_type_count=by_type_count,
        per_residue_strength=per_residue_sum,
        per_residue_strength_by_type=per_residue_by_type,
        weighted_hbond_energy=weighted_hbond_e,
        rows=rows,
    )


__all__ = [
    "INTERACTION_GEOMETRY",
    "InteractionsResult",
    "InteractionStrengthResult",
    "chemical_interactions",
    "interaction_strengths",
]
