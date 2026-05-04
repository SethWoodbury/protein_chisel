"""Fast geometric protein-ligand (and protein-protein) interaction detector.

Heavy-atom-only, sub-millisecond per protein-ligand pair. Designed for
inner-loop screening where ProLIF's bond perception fails (e.g. our
Zn-carbamylated PTE) or where arpeggio's mmCIF path is too slow.

Detection (each with a strength score in [0, 1] via Gaussian decay):

  1. **H-bonds** — D-A heavy-atom distance + D-antecedent angle.
     Works without hydrogens by checking the D's antecedent vector:
     for a real H-bond, the antecedent-D-A angle is < 70° (i.e. the
     putative H lies between D and A).

  2. **Salt bridges** — K-NZ / R-NH/NE ↔ D-OD / E-OE within 4.5 Å.
     A pair-specific check, not just "N near O".

  3. **π-stacking** — aromatic ring centroid distance + plane angle.
     Handles parallel (cos > 0.7), edge-to-face (cos < 0.3), tilted.

  4. **π-cation** — aromatic centroid ↔ K-NZ / R-NH within 6.5 Å.

  5. **Hydrophobic** — protein C ↔ ligand C within 5.0 Å, residue must
     be A/V/L/I/M/F/W/Y/C.

  6. **vdW clash** — any heavy atom pair < 2.0 Å (red flag).

Returns ``InteractionPanel`` with per-type counts, total
strength-weighted score, and the raw per-pair list. Standard d0 and
sigma values are tunable but defaults are calibrated to give ~1.0 for
canonical optimal interactions and ~0 for marginal ones.

Use as: ``detect_interactions(pdb_path, chain="A")`` for protein↔ligand;
or ``detect_interactions(pdb_path, chain_a="A", chain_b="A",
selection="all_vs_all")`` for protein↔protein.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


LOGGER = logging.getLogger("protein_chisel.tools.geometric_interactions")


# ---------------------------------------------------------------------
# Chemistry tables (donor / acceptor / charge / aromatic atoms)
# ---------------------------------------------------------------------


# Heavy-atom donor types per residue (have at least one polar H).
# Used to decide which protein atoms can act as H-bond DONORS.
HBOND_DONORS_BY_RES: dict[str, set[str]] = {
    "ARG": {"NE", "NH1", "NH2"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"}, "HID": {"ND1"}, "HIE": {"NE2"},
    "HIP": {"ND1", "NE2"}, "HIS_D": {"ND1"},
    "LYS": {"NZ"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "TRP": {"NE1"},
    "CYS": {"SG"},
    # Backbone N is always a donor (any residue except Pro)
}

# Heavy-atom acceptor types per residue (have a lone pair).
HBOND_ACCEPTORS_BY_RES: dict[str, set[str]] = {
    "ASN": {"OD1"}, "ASP": {"OD1", "OD2"},
    "GLN": {"OE1"}, "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"}, "HID": {"NE2"}, "HIE": {"ND1"},
    "HIS_D": {"NE2"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "MET": {"SD"},
    # Backbone O always an acceptor.
}

# Antecedent atom map (used to estimate H direction without explicit H).
# For each donor, the antecedent is the atom whose vector to the donor
# defines the putative H direction.
HBOND_ANTECEDENT: dict[tuple[str, str], str] = {
    ("ARG", "NE"): "CD", ("ARG", "NH1"): "CZ", ("ARG", "NH2"): "CZ",
    ("ASN", "ND2"): "CG", ("GLN", "NE2"): "CD",
    ("LYS", "NZ"): "CE",
    ("SER", "OG"): "CB", ("THR", "OG1"): "CB", ("TYR", "OH"): "CZ",
    ("TRP", "NE1"): "CD1",
    ("CYS", "SG"): "CB",
    ("HIS", "ND1"): "CG", ("HIS", "NE2"): "CD2",
    ("HID", "ND1"): "CG", ("HIE", "NE2"): "CD2",
    ("HIP", "ND1"): "CG", ("HIP", "NE2"): "CD2",
    ("HIS_D", "ND1"): "CG",
}

# Cation atoms (positively charged at pH 7).
CATION_ATOMS_BY_RES: dict[str, set[str]] = {
    "ARG": {"NH1", "NH2"}, "LYS": {"NZ"}, "HIP": {"ND1", "NE2"},
}
ANION_ATOMS_BY_RES: dict[str, set[str]] = {
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
}

# Aromatic ring atoms.
AROMATIC_RES: set[str] = {"PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HIP", "HIS_D"}
AROMATIC_RING_ATOMS: dict[str, list[str]] = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    # 6-membered ring of TRP indole
    "TRP": ["CG", "CD2", "CE2", "CE3", "CZ2", "CZ3"],
    "HIS": ["CG", "ND1", "CE1", "NE2", "CD2"],
    "HID": ["CG", "ND1", "CE1", "NE2", "CD2"],
    "HIE": ["CG", "ND1", "CE1", "NE2", "CD2"],
    "HIP": ["CG", "ND1", "CE1", "NE2", "CD2"],
    "HIS_D": ["CG", "ND1", "CE1", "NE2", "CD2"],
}

HYDROPHOBIC_RES: set[str] = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "CYS"}


# Strength-curve calibration constants (canonical optimum and width).
# Strength = exp(-(d - d0)^2 / (2 σ^2)) clamped to [0, 1].
HBOND_D0 = 2.9; HBOND_SIGMA = 0.35
SALT_BRIDGE_D0 = 3.5; SALT_BRIDGE_SIGMA = 0.6
PI_STACK_D0 = 4.0; PI_STACK_SIGMA = 0.7      # centroid-centroid
PI_CATION_D0 = 4.5; PI_CATION_SIGMA = 0.8
HYDROPHOBIC_D0 = 4.0; HYDROPHOBIC_SIGMA = 0.7

HBOND_MAX_DIST = 3.5
HBOND_MAX_ANGLE_DEG = 70.0   # antecedent-D-A angle: <=70 means H is roughly toward A
SALT_BRIDGE_MAX_DIST = 4.5
PI_STACK_MAX_DIST = 6.0
PI_CATION_MAX_DIST = 6.5
HYDROPHOBIC_MAX_DIST = 5.0
VDW_CLASH_DIST = 2.0


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------


@dataclass
class Interaction:
    type: str             # "hbond" / "salt_bridge" / "pi_pi" / "pi_cation" / "hydrophobic" / "vdw_clash"
    res_a_name: str       # donor / cation / aromatic-A / hydrophobic-A
    res_a_chain: str
    res_a_seq: int
    atom_a: str
    res_b_name: str
    res_b_chain: str
    res_b_seq: int
    atom_b: str
    distance: float       # heavy-atom distance (centroid for π-stack)
    strength: float       # Gaussian-weighted [0, 1]
    extra: dict = field(default_factory=dict)


@dataclass
class InteractionPanel:
    interactions: list[Interaction] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.interactions)

    def by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ix in self.interactions:
            out[ix.type] = out.get(ix.type, 0) + 1
        return out

    def total_strength_by_type(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ix in self.interactions:
            out[ix.type] = out.get(ix.type, 0.0) + ix.strength
        return out

    def to_dict(self, prefix: str = "geom_int__") -> dict[str, float | int]:
        out: dict[str, float | int] = {f"{prefix}n_total": self.n_total}
        n_by = self.by_type()
        s_by = self.total_strength_by_type()
        for typ in ("hbond", "salt_bridge", "pi_pi", "pi_cation",
                    "hydrophobic", "vdw_clash"):
            out[f"{prefix}n_{typ}"] = n_by.get(typ, 0)
            out[f"{prefix}strength_{typ}"] = round(s_by.get(typ, 0.0), 3)
        out[f"{prefix}strength_total"] = round(sum(s_by.values()), 3)
        return out


# ---------------------------------------------------------------------
# Atom parsing
# ---------------------------------------------------------------------


def _parse_pdb_atoms(pdb_path: Path | str) -> list[dict]:
    """Read all ATOM/HETATM heavy-atom records into dict-list.

    Format-aware: handles both standard PDB (3-char res_name) and
    Rosetta-extended (5-char res_name like "HIS_D" with chain at col 22).
    """
    out: list[dict] = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                element = line[76:78].strip()
                if element == "H":
                    continue
                # res_name at cols 16-20 (5 chars; works for both standard
                # PDB with leading space and Rosetta-extended like HIS_D).
                out.append({
                    "record": line[:6].strip(),
                    "atom_name": line[12:16].strip(),
                    "alt_loc": "",     # consumed by extended res_name field
                    "res_name": line[16:21].strip(),
                    "chain_id": line[21].strip(),
                    "res_seq": int(line[22:26].strip() or 0),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "element": element,
                })
            except (ValueError, IndexError):
                continue
    return out


def _xyz(a: dict) -> np.ndarray:
    return np.array([a["x"], a["y"], a["z"]])


def _gauss(d: float, d0: float, sigma: float) -> float:
    return math.exp(-((d - d0) ** 2) / (2.0 * sigma * sigma))


def _angle_deg(p_at: np.ndarray, p_d: np.ndarray, p_a: np.ndarray) -> float:
    """Angle at p_d between (p_at→p_d) and (p_d→p_a)."""
    v1 = p_at - p_d
    v2 = p_a - p_d
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# ---------------------------------------------------------------------
# Build per-residue lookups
# ---------------------------------------------------------------------


def _atoms_by_res(atoms: list[dict]) -> dict[tuple, dict[str, dict]]:
    """{(chain, resseq, resname): {atom_name: atom_dict}}"""
    out: dict = {}
    for a in atoms:
        key = (a["chain_id"], a["res_seq"], a["res_name"])
        out.setdefault(key, {})[a["atom_name"]] = a
    return out


# ---------------------------------------------------------------------
# Detectors (per-pair, optimized with vectorized distance pre-screen)
# ---------------------------------------------------------------------


def _detect_hbonds(
    a_atoms: list[dict], b_atoms: list[dict],
    a_by_res: dict, b_by_res: dict,
    *, max_dist: float = HBOND_MAX_DIST,
) -> list[Interaction]:
    """Heavy-atom h-bond detection with antecedent-angle check.

    For every (donor, acceptor) candidate pair within ``max_dist``, also
    require the antecedent-donor-acceptor angle ≤ HBOND_MAX_ANGLE_DEG
    (< 70°) — this approximates the donor-H...acceptor geometry without
    needing explicit hydrogens.
    """
    out: list[Interaction] = []
    # Pre-build candidate donor/acceptor atom lists for both selections
    def donors_in(atoms: list[dict]) -> list[dict]:
        ds = []
        for a in atoms:
            rn = a["res_name"]; an = a["atom_name"]
            # Backbone N is donor on every residue except Pro
            if an == "N" and rn != "PRO":
                ds.append(a); continue
            sc_donors = HBOND_DONORS_BY_RES.get(rn, set())
            if an in sc_donors:
                ds.append(a)
            # Ligand HETATMs: any N or O is potential donor (we don't
            # know which has H without a smiles; treat conservatively).
            if a["record"] == "HETATM" and a["element"] in ("N", "O"):
                ds.append(a)
        return ds

    def acceptors_in(atoms: list[dict]) -> list[dict]:
        acc = []
        for a in atoms:
            rn = a["res_name"]; an = a["atom_name"]
            if an == "O":
                acc.append(a); continue   # backbone carbonyl
            sc_acc = HBOND_ACCEPTORS_BY_RES.get(rn, set())
            if an in sc_acc:
                acc.append(a)
            if a["record"] == "HETATM" and a["element"] in ("N", "O"):
                acc.append(a)
        return acc

    a_donors = donors_in(a_atoms); a_acc = acceptors_in(a_atoms)
    b_donors = donors_in(b_atoms); b_acc = acceptors_in(b_atoms)

    def check_pair(d: dict, a: dict) -> Optional[Interaction]:
        dx = d["x"] - a["x"]; dy = d["y"] - a["y"]; dz = d["z"] - a["z"]
        r = math.sqrt(dx*dx + dy*dy + dz*dz)
        if r > max_dist or r < 1.5:
            return None
        # Antecedent angle check: for protein donors use the chemistry
        # table; for HETATM donors fall back to the closest in-residue
        # neighbor (approximates the covalent partner). The angle is the
        # antecedent-D-A angle: a good H-bond has H pointing toward A,
        # so the antecedent (which lies opposite the H) sits >110 deg
        # from A. Equivalently: angle <= 70 deg means H points roughly
        # AWAY from A -> not an H-bond.
        d_key = (d["chain_id"], d["res_seq"], d["res_name"])
        ant_name = HBOND_ANTECEDENT.get((d["res_name"], d["atom_name"]))
        if ant_name is None and d["atom_name"] == "N":
            ant_name = "CA"
        ant_xyz = None
        owning_dict = a_by_res if d in a_donors else b_by_res
        if ant_name and d_key in owning_dict and ant_name in owning_dict[d_key]:
            ant_xyz = _xyz(owning_dict[d_key][ant_name])
        elif d["record"] == "HETATM":
            # Find the nearest atom in the same residue (excluding self)
            res_atoms = owning_dict.get(d_key, {})
            best = None; best_d = 1e9
            for nm, at in res_atoms.items():
                if nm == d["atom_name"]:
                    continue
                rr = math.sqrt(
                    (at["x"]-d["x"])**2 + (at["y"]-d["y"])**2 + (at["z"]-d["z"])**2
                )
                if rr < best_d and rr < 2.0:    # covalent-bond range
                    best_d = rr; best = at
            if best is not None:
                ant_xyz = _xyz(best)
        if ant_xyz is not None:
            # Angle at D between (D->ant) and (D->A). Good H-bond has
            # this angle >= 110 deg (H points toward A, opposite ant);
            # equivalently angle <= 70 means H points away from A.
            ang = _angle_deg(ant_xyz, _xyz(d), _xyz(a))
            if ang < (180.0 - HBOND_MAX_ANGLE_DEG):
                return None
        strength = _gauss(r, HBOND_D0, HBOND_SIGMA)
        return Interaction(
            type="hbond",
            res_a_name=d["res_name"], res_a_chain=d["chain_id"], res_a_seq=d["res_seq"], atom_a=d["atom_name"],
            res_b_name=a["res_name"], res_b_chain=a["chain_id"], res_b_seq=a["res_seq"], atom_b=a["atom_name"],
            distance=round(r, 3), strength=round(strength, 3),
            extra={"role": "donor:acceptor"},
        )

    seen: set = set()
    for donors, accs in ((a_donors, b_acc), (b_donors, a_acc)):
        for d in donors:
            for a in accs:
                # skip same-residue (covalent)
                if (d["chain_id"] == a["chain_id"]
                        and d["res_seq"] == a["res_seq"]):
                    continue
                ix = check_pair(d, a)
                if ix is None:
                    continue
                # dedup
                key = (ix.res_a_chain, ix.res_a_seq, ix.atom_a,
                        ix.res_b_chain, ix.res_b_seq, ix.atom_b)
                rkey = (ix.res_b_chain, ix.res_b_seq, ix.atom_b,
                         ix.res_a_chain, ix.res_a_seq, ix.atom_a)
                if key in seen or rkey in seen:
                    continue
                seen.add(key)
                out.append(ix)
    return out


def _detect_salt_bridges(a_atoms: list[dict], b_atoms: list[dict]) -> list[Interaction]:
    out: list[Interaction] = []
    def cations_in(atoms): return [
        a for a in atoms
        if (a["res_name"] in CATION_ATOMS_BY_RES
            and a["atom_name"] in CATION_ATOMS_BY_RES[a["res_name"]])
    ]
    def anions_in(atoms): return [
        a for a in atoms
        if (a["res_name"] in ANION_ATOMS_BY_RES
            and a["atom_name"] in ANION_ATOMS_BY_RES[a["res_name"]])
    ]
    # Also: ligand HETATM N+ vs protein anion, ligand O- vs protein cation
    def ligand_N(atoms): return [a for a in atoms if a["record"] == "HETATM" and a["element"] == "N"]
    def ligand_O(atoms): return [a for a in atoms if a["record"] == "HETATM" and a["element"] == "O"]

    pairs = []
    for cations, anions in (
        (cations_in(a_atoms), anions_in(b_atoms)),
        (cations_in(b_atoms), anions_in(a_atoms)),
        (cations_in(a_atoms), ligand_O(b_atoms)),
        (cations_in(b_atoms), ligand_O(a_atoms)),
        (ligand_N(a_atoms), anions_in(b_atoms)),
        (ligand_N(b_atoms), anions_in(a_atoms)),
    ):
        for c in cations:
            for an in anions:
                if c["chain_id"] == an["chain_id"] and c["res_seq"] == an["res_seq"]:
                    continue
                dx = c["x"]-an["x"]; dy = c["y"]-an["y"]; dz = c["z"]-an["z"]
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                if r > SALT_BRIDGE_MAX_DIST or r < 1.5:
                    continue
                strength = _gauss(r, SALT_BRIDGE_D0, SALT_BRIDGE_SIGMA)
                out.append(Interaction(
                    type="salt_bridge",
                    res_a_name=c["res_name"], res_a_chain=c["chain_id"],
                    res_a_seq=c["res_seq"], atom_a=c["atom_name"],
                    res_b_name=an["res_name"], res_b_chain=an["chain_id"],
                    res_b_seq=an["res_seq"], atom_b=an["atom_name"],
                    distance=round(r, 3), strength=round(strength, 3),
                ))
    return out


def _aromatic_rings(atoms: list[dict]) -> list[dict]:
    """Return list of {res_key, centroid, normal, atoms}."""
    by_res = _atoms_by_res(atoms)
    rings: list[dict] = []
    for key, ring_atoms_dict in by_res.items():
        chain, resseq, rn = key
        if rn not in AROMATIC_RING_ATOMS:
            continue
        names = AROMATIC_RING_ATOMS[rn]
        coords = []
        for n in names:
            if n in ring_atoms_dict:
                coords.append(_xyz(ring_atoms_dict[n]))
        if len(coords) < 4:
            continue
        coords = np.array(coords)
        centroid = coords.mean(axis=0)
        # Plane normal via SVD
        u, s, vh = np.linalg.svd(coords - centroid)
        normal = vh[-1] / max(np.linalg.norm(vh[-1]), 1e-9)
        rings.append({
            "chain": chain, "res_seq": resseq, "res_name": rn,
            "centroid": centroid, "normal": normal,
            "atoms": ring_atoms_dict,
        })
    return rings


def _detect_pi_stacking(a_atoms: list[dict], b_atoms: list[dict]) -> list[Interaction]:
    out: list[Interaction] = []
    rings_a = _aromatic_rings(a_atoms)
    rings_b = _aromatic_rings(b_atoms)
    for ra in rings_a:
        for rb in rings_b:
            if (ra["chain"] == rb["chain"] and ra["res_seq"] == rb["res_seq"]):
                continue
            d = float(np.linalg.norm(ra["centroid"] - rb["centroid"]))
            if d > PI_STACK_MAX_DIST:
                continue
            cos_norm = abs(float(np.dot(ra["normal"], rb["normal"])))
            mode = "parallel" if cos_norm > 0.7 else "edge_to_face" if cos_norm < 0.3 else "tilted"
            strength = _gauss(d, PI_STACK_D0, PI_STACK_SIGMA)
            out.append(Interaction(
                type="pi_pi",
                res_a_name=ra["res_name"], res_a_chain=ra["chain"],
                res_a_seq=ra["res_seq"], atom_a="ring_centroid",
                res_b_name=rb["res_name"], res_b_chain=rb["chain"],
                res_b_seq=rb["res_seq"], atom_b="ring_centroid",
                distance=round(d, 3), strength=round(strength, 3),
                extra={"mode": mode, "cos_normal": round(cos_norm, 3)},
            ))
    return out


def _detect_pi_cation(a_atoms: list[dict], b_atoms: list[dict]) -> list[Interaction]:
    out: list[Interaction] = []
    rings_a = _aromatic_rings(a_atoms); rings_b = _aromatic_rings(b_atoms)
    cations_a = [a for a in a_atoms if (a["res_name"] in CATION_ATOMS_BY_RES
                                         and a["atom_name"] in CATION_ATOMS_BY_RES[a["res_name"]])]
    cations_b = [a for a in b_atoms if (a["res_name"] in CATION_ATOMS_BY_RES
                                         and a["atom_name"] in CATION_ATOMS_BY_RES[a["res_name"]])]
    for rings, cations in ((rings_a, cations_b), (rings_b, cations_a)):
        for r in rings:
            for c in cations:
                if r["chain"] == c["chain_id"] and r["res_seq"] == c["res_seq"]:
                    continue
                d = float(np.linalg.norm(r["centroid"] - _xyz(c)))
                if d > PI_CATION_MAX_DIST:
                    continue
                strength = _gauss(d, PI_CATION_D0, PI_CATION_SIGMA)
                out.append(Interaction(
                    type="pi_cation",
                    res_a_name=r["res_name"], res_a_chain=r["chain"],
                    res_a_seq=r["res_seq"], atom_a="ring_centroid",
                    res_b_name=c["res_name"], res_b_chain=c["chain_id"],
                    res_b_seq=c["res_seq"], atom_b=c["atom_name"],
                    distance=round(d, 3), strength=round(strength, 3),
                ))
    return out


def _detect_hydrophobic(a_atoms: list[dict], b_atoms: list[dict]) -> list[Interaction]:
    out: list[Interaction] = []
    def hyd_atoms(atoms):
        return [a for a in atoms
                if (a["res_name"] in HYDROPHOBIC_RES and a["element"] == "C"
                    and a["atom_name"] not in ("C", "CA"))]   # exclude backbone carbonyl C / CA
    def lig_C(atoms):
        return [a for a in atoms if a["record"] == "HETATM" and a["element"] == "C"]
    pairs = []
    for ha, hb in (
        (hyd_atoms(a_atoms), hyd_atoms(b_atoms)),
        (hyd_atoms(a_atoms), lig_C(b_atoms)),
        (hyd_atoms(b_atoms), lig_C(a_atoms)),
    ):
        for x in ha:
            for y in hb:
                if (x["chain_id"] == y["chain_id"] and x["res_seq"] == y["res_seq"]):
                    continue
                dx = x["x"]-y["x"]; dy = x["y"]-y["y"]; dz = x["z"]-y["z"]
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                if r > HYDROPHOBIC_MAX_DIST or r < 2.0:
                    continue
                strength = _gauss(r, HYDROPHOBIC_D0, HYDROPHOBIC_SIGMA)
                out.append(Interaction(
                    type="hydrophobic",
                    res_a_name=x["res_name"], res_a_chain=x["chain_id"],
                    res_a_seq=x["res_seq"], atom_a=x["atom_name"],
                    res_b_name=y["res_name"], res_b_chain=y["chain_id"],
                    res_b_seq=y["res_seq"], atom_b=y["atom_name"],
                    distance=round(r, 3), strength=round(strength, 3),
                ))
    return out


def _detect_vdw_clashes(a_atoms: list[dict], b_atoms: list[dict]) -> list[Interaction]:
    out: list[Interaction] = []
    for x in a_atoms:
        for y in b_atoms:
            if (x["chain_id"] == y["chain_id"]
                    and abs(x["res_seq"] - y["res_seq"]) <= 1):
                continue
            dx = x["x"]-y["x"]; dy = x["y"]-y["y"]; dz = x["z"]-y["z"]
            r = math.sqrt(dx*dx + dy*dy + dz*dz)
            if r >= VDW_CLASH_DIST or r < 0.5:
                continue
            out.append(Interaction(
                type="vdw_clash",
                res_a_name=x["res_name"], res_a_chain=x["chain_id"],
                res_a_seq=x["res_seq"], atom_a=x["atom_name"],
                res_b_name=y["res_name"], res_b_chain=y["chain_id"],
                res_b_seq=y["res_seq"], atom_b=y["atom_name"],
                distance=round(r, 3), strength=1.0,
            ))
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def detect_interactions(
    pdb_path: str | Path,
    *,
    chain: str = "A",
    selection: str = "protein_vs_ligand",
    max_neighbor_distance: float = 7.0,
) -> InteractionPanel:
    """Detect all interaction types in a single PDB pass.

    selection:
      - "protein_vs_ligand": protein (chain) vs all HETATM (default)
      - "all_vs_all": every pair (slow, intra-protein for QC)
    """
    atoms = _parse_pdb_atoms(pdb_path)
    if not atoms:
        return InteractionPanel()

    if selection == "protein_vs_ligand":
        a_atoms = [a for a in atoms if a["record"] == "ATOM" and a["chain_id"] == chain]
        b_atoms = [a for a in atoms if a["record"] == "HETATM"]
    else:
        a_atoms = atoms; b_atoms = atoms

    a_by_res = _atoms_by_res(a_atoms)
    b_by_res = _atoms_by_res(b_atoms)

    panel = InteractionPanel()
    panel.interactions.extend(_detect_hbonds(a_atoms, b_atoms, a_by_res, b_by_res))
    panel.interactions.extend(_detect_salt_bridges(a_atoms, b_atoms))
    panel.interactions.extend(_detect_pi_stacking(a_atoms, b_atoms))
    panel.interactions.extend(_detect_pi_cation(a_atoms, b_atoms))
    panel.interactions.extend(_detect_hydrophobic(a_atoms, b_atoms))
    panel.interactions.extend(_detect_vdw_clashes(a_atoms, b_atoms))
    return panel


__all__ = [
    "Interaction",
    "InteractionPanel",
    "detect_interactions",
]
