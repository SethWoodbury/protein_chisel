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
import random as _random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

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


# ---------------------------------------------------------------------------
# Active-site interaction-network growth (recursive shells from the active site)
# ---------------------------------------------------------------------------
SUPPORTED_INTERACTION_TYPES = ("hbond",)


@dataclass
class InteractionShell:
    depth: int                       # 1 = directly bonding the active site
    anchors: list[int]               # what this shell bonded to (prev shell / active site)
    candidates: list[int]            # detected, clash-filtered designable residues
    rolled: list[int]                # the subset pinned this shell
    excluded: list[tuple[int, str]]  # (resno, clash_with) dropped for clashing


@dataclass
class InteractionNetworkResult:
    rolled: set[int]                 # union of all shells' pinned residues (the network)
    shells: list[InteractionShell] = field(default_factory=list)
    records: list = field(default_factory=list)   # all detected records (for logging)


def _detect_network_records(
    pdb_path,
    *,
    designable_resnos,
    anchor_resnos,
    include_ligand,
    interaction_types,
    chain,
    max_dist,
    max_angle_deg,
    clash_dist,
):
    """Detect conservable designable-residue interactions to the current anchors.

    For ``interaction_types == ("hbond",)`` this is the EXACT existing
    sidechain-H-bond detector (so the single-shell hbond case is byte-identical).
    Other interaction types are a planned extension via
    ``geometric_interactions.detect_interactions`` (NotImplementedError for now).
    """
    types = tuple(interaction_types)
    if set(types) == {"hbond"}:   # set-based so ("hbond","hbond") also works
        return find_conservable_sidechain_hbonds(
            pdb_path,
            designable_resnos=designable_resnos,
            catalytic_resnos=anchor_resnos,
            user_fixed_resnos=(),
            include_ligand=include_ligand,
            chain=chain,
            max_dist=max_dist,
            max_angle_deg=max_angle_deg,
            clash_dist=clash_dist,
        )
    bad = [t for t in types if t not in SUPPORTED_INTERACTION_TYPES]
    raise NotImplementedError(
        f"interaction_types {bad} not yet supported; only {SUPPORTED_INTERACTION_TYPES} "
        "are wired. Other types (salt_bridge / pi_pi / pi_cation / hydrophobic) are a "
        "planned extension via geometric_interactions.detect_interactions."
    )


def build_interaction_network(
    pdb_path: str | Path,
    *,
    designable_resnos: Iterable[int],
    catalytic_resnos: Iterable[int] = (),
    user_fixed_resnos: Iterable[int] = (),
    include_ligand: bool = True,
    interaction_types: Sequence[str] = ("hbond",),
    chain: str = "A",
    depth: int = 1,
    prob: float = 1.0,
    shell_decay: float = 1.0,
    keep_clashing: bool = False,
    seed: str = "0",
    max_dist: float = DEFAULT_MAX_DIST,
    max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG,
    clash_dist: float = DEFAULT_CLASH_DIST,
) -> InteractionNetworkResult:
    """Grow a conserved-interaction network outward from the active site.

    Shell 0 = the active-site anchors (catalytic + user-fixed residues, plus the
    ligand). Shell *d* = designable residues that interact (of ``interaction_types``)
    with a residue pinned at shell *d-1*; those get probabilistically pinned and
    become the anchors for shell *d+1*. BFS to ``depth``. This is "fix H-bonds to
    catalytic residues, then H-bonds to those, ..." building a network around the
    active site.

    ``depth=1`` + ``interaction_types=("hbond",)`` reduces EXACTLY to the legacy
    single-shell conserved-H-bond fixing (shell 1 uses the same detector,
    selection, and ``random.Random(seed)`` roll), so the default is byte-identical.

    Per-shell roll probability is ``prob * shell_decay**(depth-1)`` (outer shells
    optionally rolled less aggressively). Each shell's roll is seeded
    ``seed`` (shell 1) / ``f"{seed}:shell{d}"`` (deeper) for reproducibility.
    """
    designable = {int(r) for r in designable_resnos}
    anchor0 = {int(r) for r in catalytic_resnos} | {int(r) for r in user_fixed_resnos}
    avail = designable - anchor0
    frontier = anchor0
    ligand_this = include_ligand
    rolled_all: set[int] = set()
    shells: list[InteractionShell] = []
    records_all: list = []
    for d in range(1, int(depth) + 1):
        # The ligand is a shell-0 anchor too: shell 1 can run with an empty
        # residue frontier as long as the ligand is active (ligand-only anchors,
        # e.g. CONSERVE_ANCHORS=ligand). Deeper shells require a residue frontier.
        if not avail or (not frontier and not ligand_this):
            break
        recs = _detect_network_records(
            pdb_path,
            designable_resnos=sorted(avail),
            anchor_resnos=frontier,
            include_ligand=ligand_this,
            interaction_types=interaction_types,
            chain=chain,
            max_dist=max_dist,
            max_angle_deg=max_angle_deg,
            clash_dist=clash_dist,
        )
        candidates, excluded = select_conservable_resnos(
            recs, keep_clashing=keep_clashing)
        if not candidates and not excluded:
            break  # nothing interacts with this shell's anchors -> network stops
        records_all.extend(recs)
        # Clamp to a valid probability (shell_decay>1 could otherwise exceed 1).
        shell_prob = min(1.0, max(0.0, prob * (shell_decay ** (d - 1))))
        shell_seed = seed if d == 1 else f"{seed}:shell{d}"
        rolled = roll_conserved(candidates, shell_prob, _random.Random(shell_seed))
        shells.append(InteractionShell(
            depth=d, anchors=sorted(frontier), candidates=list(candidates),
            rolled=sorted(rolled), excluded=excluded,
        ))
        rolled_all |= rolled
        if not rolled:
            break  # rolled nothing -> no frontier for the next shell
        # Next shell bonds to THIS shell's new residues; ligand is shell-0 only.
        frontier = set(rolled)
        avail = avail - rolled
        ligand_this = False
    return InteractionNetworkResult(
        rolled=rolled_all, shells=shells, records=records_all)
