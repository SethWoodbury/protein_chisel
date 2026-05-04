"""Geometric preorganization metric around catalytic residues.

Quantifies the *interactome* in the active-site region: how rich and
well-connected the network of stabilizing interactions is among the
catalytic residues plus their first- and second-shell neighbors.

Shells are defined by CA-CA distance from any catalytic residue:

* **first shell** : non-catalytic residues whose CA is within
  ``first_shell_radius`` Å of any catalytic CA (default 5.0 Å).
* **second shell**: non-catalytic, non-first-shell residues whose CA is
  within ``second_shell_radius`` Å of any catalytic CA (default 7.0 Å).

We then run :func:`tools.geometric_interactions.detect_interactions`
restricted to atoms in the union (catalytic + first + second shell), and
aggregate the resulting interactions:

* interactions FROM a shell residue TO a catalytic residue
  (preorg__n_hbonds_to_cat / n_salt_bridges_to_cat / n_pi_to_cat)
* interactions WITHIN the shells (preorg__n_hbonds_within_shells)
* total Gaussian-strength score (preorg__strength_total)
* density = interactions / (n_first + n_second)
  (preorg__interactome_density)

Pure heavy-atom geometry; runs in any sif (no PyRosetta, no py_contact_ms).
Inner-loop budget: ~0.3-0.7 s/design even for >10 catalytic residues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from protein_chisel.tools.geometric_interactions import (
    InteractionPanel,
    _atoms_by_res,
    _detect_hbonds,
    _detect_pi_cation,
    _detect_pi_stacking,
    _detect_salt_bridges,
    _parse_pdb_atoms,
)


def _ca_xyz_by_res(atoms: list[dict], chain: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for a in atoms:
        if a["record"] != "ATOM" or a["chain_id"] != chain or a["atom_name"] != "CA":
            continue
        out[a["res_seq"]] = np.array([a["x"], a["y"], a["z"]])
    return out


def _classify_shells(
    cas: dict[int, np.ndarray],
    catalytic_resnos: set[int],
    first_radius: float,
    second_radius: float,
) -> tuple[set[int], set[int]]:
    """Return (first_shell, second_shell) resno sets.

    Membership is by min CA-CA distance to any catalytic CA. Catalytic
    residues themselves are excluded from both shells.
    """
    cat_pts = np.array(
        [cas[r] for r in catalytic_resnos if r in cas], dtype=float
    )
    first: set[int] = set()
    second: set[int] = set()
    if len(cat_pts) == 0:
        return first, second
    for r, ca in cas.items():
        if r in catalytic_resnos:
            continue
        d = float(np.linalg.norm(cat_pts - ca, axis=-1).min())
        if d <= first_radius:
            first.add(r)
        elif d <= second_radius:
            second.add(r)
    return first, second


def preorganization_score(
    pdb: str | Path,
    catalytic_resnos: Iterable[int],
    chain: str = "A",
    first_shell_radius: float = 5.0,
    second_shell_radius: float = 7.0,
) -> dict[str, float | int]:
    """Compute the 6 preorganization metrics for ``pdb``.

    Args:
        pdb: path to PDB.
        catalytic_resnos: residue numbers of the catalytic residues
            (chain ``chain``).
        chain: protein chain id. HETATM ligands are ignored.
        first_shell_radius: Å (CA-CA, default 5.0).
        second_shell_radius: Å (CA-CA, default 7.0).
    """
    cat_set = set(int(r) for r in catalytic_resnos)
    atoms = _parse_pdb_atoms(pdb)

    # Restrict to the protein chain — preorganization is intra-protein.
    atoms = [a for a in atoms if a["record"] == "ATOM" and a["chain_id"] == chain]
    if not atoms:
        return _empty_result()

    cas = _ca_xyz_by_res(atoms, chain)
    first, second = _classify_shells(
        cas, cat_set, first_shell_radius, second_shell_radius,
    )
    keep_resnos = cat_set | first | second
    sub_atoms = [a for a in atoms if a["res_seq"] in keep_resnos]
    if not sub_atoms:
        return _empty_result(n_first=len(first), n_second=len(second))

    by_res = _atoms_by_res(sub_atoms)
    panel = InteractionPanel()
    panel.interactions.extend(
        _detect_hbonds(sub_atoms, sub_atoms, by_res, by_res)
    )
    panel.interactions.extend(_detect_salt_bridges(sub_atoms, sub_atoms))
    panel.interactions.extend(_detect_pi_stacking(sub_atoms, sub_atoms))
    panel.interactions.extend(_detect_pi_cation(sub_atoms, sub_atoms))

    n_hb_to_cat = 0
    n_sb_to_cat = 0
    n_pi_to_cat = 0
    n_hb_within_shells = 0
    strength_total = 0.0
    seen: set[tuple] = set()
    for ix in panel.interactions:
        # all_vs_all detectors emit each pair twice (a->b and b->a swap);
        # canonicalize on (atom-key-pair) to dedup.
        ka = (ix.res_a_chain, ix.res_a_seq, ix.atom_a)
        kb = (ix.res_b_chain, ix.res_b_seq, ix.atom_b)
        key = tuple(sorted([ka, kb])) + (ix.type,)
        if key in seen:
            continue
        seen.add(key)
        a_in_cat = ix.res_a_seq in cat_set
        b_in_cat = ix.res_b_seq in cat_set
        a_in_shell = ix.res_a_seq in first or ix.res_a_seq in second
        b_in_shell = ix.res_b_seq in first or ix.res_b_seq in second
        strength_total += ix.strength
        if a_in_cat ^ b_in_cat:  # exactly one side catalytic
            if ix.type == "hbond":
                n_hb_to_cat += 1
            elif ix.type == "salt_bridge":
                n_sb_to_cat += 1
            elif ix.type in ("pi_pi", "pi_cation"):
                n_pi_to_cat += 1
        elif a_in_shell and b_in_shell and not (a_in_cat or b_in_cat):
            if ix.type == "hbond":
                n_hb_within_shells += 1

    n_shell = len(first) + len(second)
    density = (
        (n_hb_to_cat + n_sb_to_cat + n_pi_to_cat + n_hb_within_shells) / n_shell
        if n_shell > 0 else 0.0
    )
    return {
        "preorg__n_hbonds_to_cat": int(n_hb_to_cat),
        "preorg__n_salt_bridges_to_cat": int(n_sb_to_cat),
        "preorg__n_pi_to_cat": int(n_pi_to_cat),
        "preorg__n_hbonds_within_shells": int(n_hb_within_shells),
        "preorg__strength_total": round(float(strength_total), 3),
        "preorg__interactome_density": round(float(density), 3),
        "preorg__n_first_shell": int(len(first)),
        "preorg__n_second_shell": int(len(second)),
    }


def _empty_result(n_first: int = 0, n_second: int = 0) -> dict[str, float | int]:
    return {
        "preorg__n_hbonds_to_cat": 0,
        "preorg__n_salt_bridges_to_cat": 0,
        "preorg__n_pi_to_cat": 0,
        "preorg__n_hbonds_within_shells": 0,
        "preorg__strength_total": 0.0,
        "preorg__interactome_density": 0.0,
        "preorg__n_first_shell": int(n_first),
        "preorg__n_second_shell": int(n_second),
    }


__all__ = ["preorganization_score"]
