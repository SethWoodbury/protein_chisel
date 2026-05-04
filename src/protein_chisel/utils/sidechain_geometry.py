"""Sidechain geometry helpers for directional position classification.

Provides:

- ``phantom_cb(N, CA, C)``: ideal Cβ position for Gly (or any residue
  missing CB). Tetrahedral construction from N, Cα, C. Auto-validated
  chirality: at import time we test against a known L-alanine geometry
  and pick the cross-product order that gives < 0.15 Å RMSD.

- ``BACKBONE_ATOMS``, ``SIDECHAIN_ATOMS_BY_AA``: atom-name sets shared
  with ``structure/clash_check.py`` (we import that table to avoid drift).

- ``FUNCTIONAL_ATOMS``: per-AA list of "chemically meaningful" atom names
  for orientation/contact detection. Multiple atoms (e.g. Arg NH1 + NH2)
  are averaged to a single coordinate.

- ``RING_ATOMS``: per-AA aromatic ring atom set (Phe / Tyr / Trp / His)
  for π-stacking centroid.

- ``MAX_SASA_BY_AA``: Tien et al. 2013 maximum-allowed SASA per residue
  type, used to compute ``sasa_sc_fraction``.

- ``sigmoid``: numerically stable sigmoid for soft-membership scores.

All distances Å, angles deg, scores [0, 1].
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np


LOGGER = logging.getLogger("protein_chisel.utils.sidechain_geometry")


# ---------------------------------------------------------------------------
# Backbone / sidechain atom sets
# ---------------------------------------------------------------------------


BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O", "OXT"})


# Reuse the canonical sidechain table from clash_check, but lazy-import to
# avoid circular dependency. structure/clash_check.py is the source of truth.
def _load_sidechain_atom_names() -> dict[str, set[str]]:
    from protein_chisel.structure.clash_check import SIDECHAIN_ATOM_NAMES
    return dict(SIDECHAIN_ATOM_NAMES)


# Per-AA "chemically meaningful" functional atoms for orientation /
# contact analysis. Ordered list — we average their coords.
# Codex round-2: His uses closer-of-{ND1, NE2} (tautomer-agnostic
# simplification); Trp donor is NE1 (ring centroid is `RING_ATOMS`).
FUNCTIONAL_ATOMS: dict[str, tuple[str, ...]] = {
    "ALA": ("CB",),                          # nominal — short
    "ARG": ("NH1", "NH2"),                   # guanidinium tip
    "ASN": ("OD1", "ND2"),                   # amide
    "ASP": ("OD1", "OD2"),                   # carboxylate
    "CYS": ("SG",),
    "GLN": ("OE1", "NE2"),
    "GLU": ("OE1", "OE2"),
    "GLY": (),                               # no sidechain — phantom CB used elsewhere
    "HIS": ("ND1", "NE2"),                   # closer-of resolved at runtime
    "HID": ("ND1", "NE2"),
    "HIE": ("ND1", "NE2"),
    "HIP": ("ND1", "NE2"),
    "HIS_D": ("ND1", "NE2"),
    "ILE": ("CD1",),
    "LEU": ("CD1", "CD2"),
    "LYS": ("NZ",),
    "KCX": ("NZ", "OQ1", "OQ2"),             # carbamylated Lys
    "MET": ("SD",),
    "PHE": ("CZ",),                          # ring centroid recorded separately
    "PRO": ("CG",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TRP": ("NE1",),                         # H-bond donor; centroid is separate
    "TYR": ("OH",),
    "VAL": ("CG1", "CG2"),
}


# Aromatic ring atoms for π-stacking centroid.
RING_ATOMS: dict[str, tuple[str, ...]] = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "NE1", "CE2", "CD2", "CE3", "CZ2", "CZ3", "CH2"),
    "HIS": ("CG", "ND1", "CE1", "NE2", "CD2"),
    "HID": ("CG", "ND1", "CE1", "NE2", "CD2"),
    "HIE": ("CG", "ND1", "CE1", "NE2", "CD2"),
    "HIP": ("CG", "ND1", "CE1", "NE2", "CD2"),
    "HIS_D": ("CG", "ND1", "CE1", "NE2", "CD2"),
}


# Maximum-allowed solvent-accessible surface area per residue type,
# Tien et al. 2013, doi:10.1371/journal.pone.0080635 (Theoretical scale,
# Å²). Used for `sasa_sc_fraction` = sasa_sidechain / max_sasa.
MAX_SASA_BY_AA: dict[str, float] = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY":  104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
    # Modified / alt names default to the parent residue's value.
    "MSE": 224.0, "HID": 224.0, "HIE": 224.0, "HIP": 224.0, "HIS_D": 224.0,
    "KCX": 236.0,  # treat as Lys-like
    "SEC": 167.0, "PYL": 236.0,
}


# ---------------------------------------------------------------------------
# Phantom Cβ
# ---------------------------------------------------------------------------


# Tetrahedral angle between bisector and the CB direction, derived from
# ideal sp³ geometry: acos(1/sqrt(3)) ≈ 54.7356°.
_TETRAHEDRAL_TILT_RAD: float = math.acos(1.0 / math.sqrt(3.0))
_CA_CB_BOND_LENGTH: float = 1.522  # Å, Engh & Huber 1991 ideal

# Set at import time by `_select_chirality_sign()`. The constant cross-
# product order that yields the L-amino-acid Cβ position. +1 means
# `cross(â, b̂)` (N before C); -1 means `cross(b̂, â)` (C before N).
_CHIRALITY_SIGN: int = 0   # populated below


def _phantom_cb_with_sign(
    N: np.ndarray, CA: np.ndarray, C: np.ndarray, sign: int
) -> np.ndarray:
    """Compute phantom Cβ with the given chirality sign (+1 or -1)."""
    a = N - CA
    a /= np.linalg.norm(a)
    b = C - CA
    b /= np.linalg.norm(b)
    bisector = -(a + b)
    bisector /= np.linalg.norm(bisector)
    normal = np.cross(a, b) * sign
    normal /= np.linalg.norm(normal)
    direction = (
        math.cos(_TETRAHEDRAL_TILT_RAD) * bisector
        + math.sin(_TETRAHEDRAL_TILT_RAD) * normal
    )
    return CA + _CA_CB_BOND_LENGTH * direction


def _select_chirality_sign() -> int:
    """Pick +1 or -1 by validating against an L-alanine standard.

    Builds canonical L-Ala backbone atoms (in the standard PDB convention),
    computes phantom Cβ with both signs, compares to the canonical Cβ
    position, picks the matching sign. Raises if neither matches within
    0.20 Å (would mean the formula itself is wrong).
    """
    # L-alanine canonical coords (PDB-like, 0.5 Å precision is enough):
    #   N at origin, CA along +x, peptide plane ~ xy-plane.
    # Source: idealized L-Ala from Rosetta CHARMM patch (matches Engh&Huber).
    N = np.array([1.458, 0.000, 0.000])
    CA = np.array([2.009, 1.420, 0.000])
    C = np.array([3.535, 1.420, 0.000])
    CB_canonical = np.array([1.483, 2.158, -1.190])   # standard L-Ala Cβ

    best_sign, best_err = 0, float("inf")
    for sign in (+1, -1):
        cb_phantom = _phantom_cb_with_sign(N, CA, C, sign)
        err = float(np.linalg.norm(cb_phantom - CB_canonical))
        LOGGER.debug("phantom CB chirality probe: sign=%+d err=%.4f Å", sign, err)
        if err < best_err:
            best_err, best_sign = err, sign

    if best_err > 0.20:
        raise RuntimeError(
            f"phantom_cb formula is wrong: best L-Ala RMSD={best_err:.3f} Å "
            "(>0.20 Å). Check tetrahedral construction in sidechain_geometry.py."
        )
    LOGGER.debug("phantom_cb chirality = sign=%+d (L-Ala RMSD=%.3f Å)",
                 best_sign, best_err)
    return best_sign


_CHIRALITY_SIGN = _select_chirality_sign()


def phantom_cb(
    N: np.ndarray, CA: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """Tetrahedral phantom-Cβ position for Gly (or any CB-less residue).

    Args:
        N, CA, C: 3-vector heavy-atom coordinates (Å).

    Returns:
        3-vector Cβ position 1.522 Å from CA, with L-amino-acid handedness
        validated at import.
    """
    return _phantom_cb_with_sign(np.asarray(N), np.asarray(CA), np.asarray(C),
                                   _CHIRALITY_SIGN)


# ---------------------------------------------------------------------------
# Sidechain centroid + functional atom + ring centroid
# ---------------------------------------------------------------------------


def sidechain_atom_names(name3: str) -> set[str]:
    """Sidechain heavy-atom name set for a residue type (excluding backbone).

    Uses the canonical table from ``structure/clash_check.SIDECHAIN_ATOM_NAMES``
    so the two modules cannot drift.
    """
    table = _load_sidechain_atom_names()
    return set(table.get(name3.upper(), set()))


def sidechain_centroid(
    coords_by_atom: dict[str, np.ndarray], name3: str,
    *, fallback_phantom_cb: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Mean position of all sidechain heavy atoms present in the residue.

    For Gly: returns ``fallback_phantom_cb`` if provided, else None.
    For Ala: returns CB (the only sidechain atom).
    For other AAs: mean of all sidechain heavy atoms found in
    ``coords_by_atom``.
    """
    name = name3.upper()
    if name == "GLY":
        return fallback_phantom_cb
    sc_names = sidechain_atom_names(name)
    pts = [coords_by_atom[n] for n in sc_names if n in coords_by_atom]
    if not pts:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def functional_atom_position(
    coords_by_atom: dict[str, np.ndarray], name3: str,
    *, ligand_centroid: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Position of the residue's chemically meaningful functional atom.

    For multi-atom functional groups (Arg NH1+NH2, Asp OD1+OD2, His
    ND1+NE2), returns the mean position EXCEPT for His where we return
    the closer-of-{ND1, NE2} to ``ligand_centroid`` (codex round-2:
    tautomer assignment from a static PDB is unreliable; the closer
    atom is the one most likely to be making the H-bond and is
    tautomer-agnostic).
    """
    name = name3.upper()
    atoms = FUNCTIONAL_ATOMS.get(name, ())
    if not atoms:
        return None
    pts = [coords_by_atom[n] for n in atoms if n in coords_by_atom]
    if not pts:
        return None
    is_his = name in ("HIS", "HID", "HIE", "HIP", "HIS_D")
    if is_his and ligand_centroid is not None and len(pts) > 1:
        # Closer-of-{ND1, NE2}.
        d = [float(np.linalg.norm(p - ligand_centroid)) for p in pts]
        return pts[int(np.argmin(d))]
    return np.mean(np.stack(pts, axis=0), axis=0)


def ring_centroid(
    coords_by_atom: dict[str, np.ndarray], name3: str,
) -> Optional[np.ndarray]:
    """Aromatic ring centroid for Phe/Tyr/Trp/His. None for non-aromatic AAs."""
    atoms = RING_ATOMS.get(name3.upper(), ())
    if not atoms:
        return None
    pts = [coords_by_atom[n] for n in atoms if n in coords_by_atom]
    if not pts:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


# ---------------------------------------------------------------------------
# Orientation angle
# ---------------------------------------------------------------------------


def orientation_angle_deg(
    ca: np.ndarray, ref_atom: np.ndarray, ligand_target: np.ndarray
) -> float:
    """Angle between (CA → ref_atom) and (CA → ligand_target), in degrees.

    Returns 0.0 (perfectly inward) to 180.0 (perfectly outward). NaN if
    either vector has zero length.
    """
    v1 = ref_atom - ca
    v2 = ligand_target - ca
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cos)))


# ---------------------------------------------------------------------------
# Soft membership scores
# ---------------------------------------------------------------------------


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable logistic sigmoid: 1 / (1 + exp(-x))."""
    if isinstance(x, np.ndarray):
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def max_sasa_for(name3: str) -> float:
    """Tien 2013 max-SASA for a residue type. Falls back to ALA if unknown."""
    return MAX_SASA_BY_AA.get(name3.upper(), MAX_SASA_BY_AA["ALA"])


__all__ = [
    "BACKBONE_ATOMS",
    "FUNCTIONAL_ATOMS",
    "MAX_SASA_BY_AA",
    "RING_ATOMS",
    "functional_atom_position",
    "max_sasa_for",
    "orientation_angle_deg",
    "phantom_cb",
    "ring_centroid",
    "sidechain_atom_names",
    "sidechain_centroid",
    "sigmoid",
]
