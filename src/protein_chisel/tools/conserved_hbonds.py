"""Find designable (unfixed) **sidechain** H-bonds to the ligand or fixed residues.

These are candidates to *conserve* during LigandMPNN design: a designable residue
whose sidechain makes a functionally meaningful H-bond to the ligand or to a fixed
(catalytic / user-fixed) residue can be probabilistically pinned into the fixed-
residue list.

Rules:
  * The DESIGNABLE residue must participate via its **sidechain** — a backbone N/O
    H-bond doesn't count (the sidechain is still freely designable).
  * The partner may be the **ligand** or a **fixed residue**, using sidechain OR
    backbone.
  * **Histidine is tautomer-agnostic**: both ND1 and NE2 are treated as donor AND
    acceptor (a non-catalytic His may be either tautomer, or HIP with both).
  * Each bond is binned by strength {super_strong … super_weak}.
  * Candidates whose current sidechain **clashes** with anything that won't be
    redesigned (fixed/all backbone, ligand, fixed sidechains) are flagged; clashes
    against other DESIGNABLE sidechains are ignored (those get redesigned).

Heavy-atom only (no reliance on explicit/relaxed H): reuses the H-bond geometry in
``geometric_interactions`` (donor···acceptor distance + antecedent angle), the same
way ``scoring/preorganization`` does.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from protein_chisel.structure.clash_check import SIDECHAIN_ATOM_NAMES
from protein_chisel.tools.geometric_interactions import (
    _atoms_by_res,
    _detect_hbonds,
    _parse_pdb_atoms,
)
from protein_chisel.utils.sidechain_geometry import BACKBONE_ATOMS

LOGGER = logging.getLogger("protein_chisel.tools.conserved_hbonds")

# His-variant resnames normalized to "HIS" so both ND1/NE2 act as donor+acceptor.
_HIS_VARIANTS = {"HID", "HIE", "HIP", "HIS_D", "HIS_E", "HISD", "HISE", "HSD", "HSE", "HSP"}

# Strength bins on the geometric_interactions Gaussian strength (distance-based,
# d0=2.9 Å, sigma=0.35). e.g. 2.72 Å -> ~0.88 (super_strong); 3.9 Å -> ~0.02.
_STRENGTH_BINS: tuple[tuple[float, str], ...] = (
    (0.85, "super_strong"),
    (0.65, "strong"),
    (0.45, "moderate"),
    (0.25, "weak"),
    (0.0, "super_weak"),
)

DEFAULT_MAX_DIST = 3.9
DEFAULT_MAX_ANGLE_DEG = 90.0  # generous (stock detector uses 70)
DEFAULT_CLASH_DIST = 1.8


def strength_bin(strength: float) -> str:
    for thr, label in _STRENGTH_BINS:
        if strength >= thr:
            return label
    return "super_weak"


@dataclass
class ConservableHbond:
    resno: int
    resname: str
    sidechain_atom: str
    partner_kind: str          # "ligand" | "catalytic" | "user_fixed"
    partner_resno: int         # -1 for ligand
    partner_resname: str
    partner_atom: str
    distance: float
    strength: float
    strength_bin: str
    hypothesized_donor: str    # "RES<seq>/<atom>"
    hypothesized_acceptor: str
    clashes: bool = False
    clash_with: str = ""       # "RES<seq>/<atom>" or "LIG/<atom>" of the clash partner


def _norm_his(atoms: list[dict]) -> list[dict]:
    """Shallow-copy atoms, normalizing His-variant resnames to 'HIS' so the
    donor/acceptor tables treat both ND1 and NE2 (tautomer-agnostic)."""
    out = []
    for a in atoms:
        if a["res_name"] in _HIS_VARIANTS:
            a = dict(a)
            a["res_name"] = "HIS"
        out.append(a)
    return out


def find_conservable_sidechain_hbonds(
    pdb_path: str | Path,
    *,
    designable_resnos: Iterable[int],
    catalytic_resnos: Iterable[int] = (),
    user_fixed_resnos: Iterable[int] = (),
    include_ligand: bool = True,
    chain: str = "A",
    max_dist: float = DEFAULT_MAX_DIST,
    max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG,
    clash_dist: float = DEFAULT_CLASH_DIST,
) -> list[ConservableHbond]:
    """Return per-bond records for designable sidechain H-bonds to anchors
    (ligand / catalytic / user-fixed). One designable residue may yield several
    records; de-dupe to residues at the call site via ``{r.resno for r in ...}``.
    """
    atoms = _parse_pdb_atoms(pdb_path)
    designable = {int(r) for r in designable_resnos}
    catalytic = {int(r) for r in catalytic_resnos}
    user_fixed = {int(r) for r in user_fixed_resnos}
    anchor_resnos = catalytic | user_fixed

    des_atoms = [a for a in atoms
                 if a["record"] == "ATOM" and a["chain_id"] == chain
                 and a["res_seq"] in designable]
    anc_atoms = [a for a in atoms
                 if a["record"] == "ATOM" and a["chain_id"] == chain
                 and a["res_seq"] in anchor_resnos]
    lig_keys = {(a["chain_id"], a["res_seq"]) for a in atoms if a["record"] == "HETATM"}
    if include_ligand:
        anc_atoms += [a for a in atoms if a["record"] == "HETATM"]
    if not des_atoms or not anc_atoms:
        return []

    des_n, anc_n = _norm_his(des_atoms), _norm_his(anc_atoms)
    hbonds = _detect_hbonds(
        des_n, anc_n, _atoms_by_res(des_n), _atoms_by_res(anc_n),
        max_dist=max_dist, max_angle_deg=max_angle_deg,
    )

    records: list[ConservableHbond] = []
    for ix in hbonds:
        a_is_des = ix.res_a_chain == chain and ix.res_a_seq in designable
        b_is_des = ix.res_b_chain == chain and ix.res_b_seq in designable
        if a_is_des == b_is_des:
            continue  # exactly one side should be designable (sets are disjoint)
        if a_is_des:
            d_seq, d_name, d_atom = ix.res_a_seq, ix.res_a_name, ix.atom_a
            p_chain, p_seq, p_name, p_atom = ix.res_b_chain, ix.res_b_seq, ix.res_b_name, ix.atom_b
        else:
            d_seq, d_name, d_atom = ix.res_b_seq, ix.res_b_name, ix.atom_b
            p_chain, p_seq, p_name, p_atom = ix.res_a_chain, ix.res_a_seq, ix.res_a_name, ix.atom_a

        if d_atom in BACKBONE_ATOMS:
            continue  # designable side must use its sidechain

        if (p_chain, p_seq) in lig_keys:
            kind, p_out = "ligand", -1
        elif p_seq in catalytic:
            kind, p_out = "catalytic", p_seq
        elif p_seq in user_fixed:
            kind, p_out = "user_fixed", p_seq
        else:
            continue

        records.append(ConservableHbond(
            resno=d_seq, resname=d_name, sidechain_atom=d_atom,
            partner_kind=kind, partner_resno=p_out, partner_resname=p_name, partner_atom=p_atom,
            distance=ix.distance, strength=ix.strength, strength_bin=strength_bin(ix.strength),
            hypothesized_donor=f"{ix.res_a_name}{ix.res_a_seq}/{ix.atom_a}",
            hypothesized_acceptor=f"{ix.res_b_name}{ix.res_b_seq}/{ix.atom_b}",
        ))

    _annotate_clashes(records, atoms, designable, chain, clash_dist)
    return records


def _annotate_clashes(records, atoms, designable, chain, clash_dist):
    """Flag candidates whose sidechain clashes with anything that won't be
    redesigned: all backbone (any residue) + ligand + fixed sidechains. Clashes
    against other DESIGNABLE sidechains are ignored."""
    if not records:
        return
    cand_resnos = {r.resno for r in records}

    nonredesign = []
    for a in atoms:
        if a["record"] == "HETATM" or a["chain_id"] != chain:
            nonredesign.append(a)
            continue
        designable_sc = a["res_seq"] in designable and a["atom_name"] not in BACKBONE_ATOMS
        if not designable_sc:  # keep all backbone + fixed sidechains
            nonredesign.append(a)

    cand_sc: dict[int, list[dict]] = {}
    for a in atoms:
        if (a["record"] == "ATOM" and a["chain_id"] == chain
                and a["res_seq"] in cand_resnos
                and a["atom_name"] in SIDECHAIN_ATOM_NAMES.get(a["res_name"], set())):
            cand_sc.setdefault(a["res_seq"], []).append(a)

    cd2 = clash_dist * clash_dist
    clash_with: dict[int, str] = {}
    for resno, scs in cand_sc.items():
        for s in scs:
            for p in nonredesign:
                if (p["record"] == "ATOM" and p["chain_id"] == chain
                        and abs(p["res_seq"] - resno) <= 1):
                    continue  # self + covalent neighbors
                dx = s["x"] - p["x"]; dy = s["y"] - p["y"]; dz = s["z"] - p["z"]
                if dx * dx + dy * dy + dz * dz < cd2:
                    clash_with[resno] = (
                        f"LIG/{p['atom_name']}" if p["record"] == "HETATM"
                        else f"{p['res_name']}{p['res_seq']}/{p['atom_name']}")
                    break
            if resno in clash_with:
                break

    for r in records:
        if r.resno in clash_with:
            r.clashes = True
            r.clash_with = clash_with[r.resno]


# ---------------------------------------------------------------------------
# Selection / rolling helpers — shared by scripts/chisel_ligandMPNN.py and
# scripts/iterative_design.py so both pick + roll candidates identically.
# ---------------------------------------------------------------------------
def normalize_probability(value: float) -> float:
    """A 0-1 probability, or a percentage in (1, 100] (auto-converted, logged).

    Raises ``ValueError`` on anything else; callers map it to their own error
    type (``SystemExit`` for a CLI, ``argparse`` error for a driver).
    """
    v = float(value)
    if v < 0:
        raise ValueError(f"must be >= 0 (got {v})")
    if v <= 1:
        return v
    if v <= 100:
        LOGGER.warning("probability %g > 1; interpreting as a percentage -> %g",
                       v, v / 100.0)
        return v / 100.0
    raise ValueError(f"must be 0-1 or a percentage <=100 (got {v})")


def roll_conserved(items: Iterable, prob: float, rng) -> set:
    """Independently keep each item with probability ``prob`` using the supplied
    ``random.Random``. Type-agnostic (resno ints or ``"A157"`` labels). Items are
    rolled in iteration order, so pass an ordered sequence for reproducibility.
    """
    return {x for x in items if rng.random() < prob}


def select_conservable_resnos(
    records: Iterable[ConservableHbond],
    *,
    keep_clashing: bool = False,
) -> tuple[list[int], list[tuple[int, str]]]:
    """Collapse per-bond records to a sorted list of candidate residue numbers,
    excluding residues whose sidechain clashes with fixed backbone/ligand unless
    ``keep_clashing``. Returns ``(candidates, excluded)`` where ``excluded`` is
    ``[(resno, clash_with), ...]`` for logging.
    """
    by_res: dict[int, list[ConservableHbond]] = {}
    for r in records:
        by_res.setdefault(r.resno, []).append(r)
    candidates: list[int] = []
    excluded: list[tuple[int, str]] = []
    for resno, rs in sorted(by_res.items()):
        if rs[0].clashes and not keep_clashing:
            excluded.append((resno, rs[0].clash_with))
            continue
        candidates.append(resno)
    return candidates, excluded
