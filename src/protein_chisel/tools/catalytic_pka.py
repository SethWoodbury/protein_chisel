"""PROPKA wrapper — pKa predictions, focused on catalytic residues.

PROPKA 3.x is installed in esmc.sif (and most chemistry-flavored sifs).
The Python API at ``propka.run.single`` runs PROPKA on a PDB and writes
a ``.pka`` file alongside; we parse that and return per-residue pKa values.

For catalytic residues (REMARK 666), we report:
    expected_pka_shift   = predicted_pka - solution_pka  (signed)
A positive shift on Asp/Glu means the residue is harder to deprotonate
than in solution; on His it means easier to protonate.

Run inside esmc.sif (where propka was added).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


LOGGER = logging.getLogger("protein_chisel.catalytic_pka")


# Canonical solution pKas (model values from PROPKA's training).
SOLUTION_PKA = {
    "ASP": 3.80, "GLU": 4.50, "HIS": 6.50, "CYS": 9.00, "TYR": 10.00,
    "LYS": 10.50, "ARG": 12.50, "N+": 8.00, "C-": 3.20,
}


@dataclass
class CatalyticPkaResult:
    per_residue_pka: dict[tuple[str, int, str], float] = field(default_factory=dict)
    catres_pka: dict[tuple[str, int, str], float] = field(default_factory=dict)
    catres_pka_shift: dict[tuple[str, int, str], float] = field(default_factory=dict)
    n_catres_evaluated: int = 0
    raw_pka_path: Optional[str] = None

    def to_dict(self, prefix: str = "pka__") -> dict[str, float | int]:
        out: dict[str, float | int] = {
            f"{prefix}n_catres_evaluated": self.n_catres_evaluated,
        }
        # Catalytic residues by chain+resno (label e.g. "A_64_LYS")
        for (chain, resno, name3), pka in self.catres_pka.items():
            out[f"{prefix}catres__{chain}_{resno}_{name3}__pka"] = pka
        for (chain, resno, name3), shift in self.catres_pka_shift.items():
            out[f"{prefix}catres__{chain}_{resno}_{name3}__shift"] = shift
        return out


def catalytic_pka(
    pdb_path: str | Path,
    catres: Optional[Iterable[tuple[str, int]]] = None,
) -> CatalyticPkaResult:
    """Run PROPKA on the input PDB and return pKa metrics.

    Args:
        pdb_path: input PDB.
        catres: optional iterable of (chain, resno) to report on. Defaults
            to REMARK 666 catalytic residues; if those are missing we
            return per-residue pKa for the whole protein with empty
            catres_pka.
    """
    pdb_path = Path(pdb_path).resolve()

    if catres is None:
        from protein_chisel.io.pdb import parse_remark_666

        cr = parse_remark_666(pdb_path, key_by="chain_resno")
        catres = list(cr.keys())
    catres_set = {(c, int(r)) for c, r in catres or ()}

    # Use MolecularContainer in-memory rather than relying on the .pka
    # file (write_pka rules differ across PROPKA versions).
    try:
        from propka.run import single
        mol = single(str(pdb_path), write_pka=False)
    except Exception as e:
        raise RuntimeError(f"PROPKA failed on {pdb_path}: {e}") from e

    per_residue = _walk_propka_groups(mol)

    catres_pka: dict[tuple[str, int, str], float] = {}
    catres_shift: dict[tuple[str, int, str], float] = {}
    for (chain, resno, name3), pka in per_residue.items():
        if (chain, resno) in catres_set:
            catres_pka[(chain, resno, name3)] = pka
            sol = SOLUTION_PKA.get(name3.upper())
            if sol is not None:
                catres_shift[(chain, resno, name3)] = pka - sol

    return CatalyticPkaResult(
        per_residue_pka=per_residue,
        catres_pka=catres_pka,
        catres_pka_shift=catres_shift,
        n_catres_evaluated=len(catres_pka),
        raw_pka_path=None,
    )


def _walk_propka_groups(mol) -> dict[tuple[str, int, str], float]:
    """Walk a PROPKA MolecularContainer and pull pKa per (chain, resno, name3).

    PROPKA reports multiple groups per residue (e.g. ASP has 'COO' and the
    sidechain). We pick the residue-level group whose label starts with
    the residue name3 (e.g. label='ASP   3 A'), which is PROPKA's
    canonical "this residue's titratable group" entry.
    """
    out: dict[tuple[str, int, str], float] = {}
    if not hasattr(mol, "conformations"):
        return out
    # PROPKA averages across conformations into a single result; we use
    # the first conformation since it carries the .pka_value attributes.
    confs = mol.conformations
    if not confs:
        return out
    # Take the alphabetically-first conformation (typically "1A").
    name = sorted(confs.keys())[0]
    conf = confs[name]
    if not hasattr(conf, "groups"):
        return out
    for g in conf.groups:
        pka = getattr(g, "pka_value", None)
        if pka is None or pka == 0.0:
            continue
        atom = getattr(g, "atom", None)
        if atom is None:
            continue
        chain = getattr(atom, "chain_id", None)
        resno = getattr(atom, "res_num", None)
        name3 = getattr(atom, "res_name", None)
        if chain is None or resno is None or name3 is None:
            continue
        # Only keep residue-titratable groups: label starts with name3 or "N+"/"C-".
        label = getattr(g, "label", "") or ""
        if not (label.startswith(name3) or label.startswith("N+") or label.startswith("C-")):
            continue
        out[(str(chain), int(resno), str(name3))] = float(pka)
    return out


# ---------------------------------------------------------------------------
# PROPKA output parser
# ---------------------------------------------------------------------------


# A typical line in the SUMMARY section of a .pka file looks like:
#       ASP   3 A      3.66        3.80
# columns: name3, resno, chain, pka, model_pka
_PROPKA_SUMMARY_RE = re.compile(
    r"^\s*([A-Z][A-Z][A-Z\+\-])\s+(\d+)\s+([A-Za-z])\s+([\d\.\-]+)\s+([\d\.\-]+)"
)


def _parse_propka(pka_path: Path) -> dict[tuple[str, int, str], float]:
    """Parse PROPKA's ``.pka`` summary into a {(chain, resno, name3): pka} dict."""
    out: dict[tuple[str, int, str], float] = {}
    in_summary = False
    with open(pka_path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("SUMMARY OF THIS PREDICTION"):
                in_summary = True
                continue
            if not in_summary:
                continue
            if stripped.startswith("Free energy of") or stripped.startswith("---"):
                break
            m = _PROPKA_SUMMARY_RE.match(line)
            if not m:
                continue
            name3 = m.group(1)
            resno = int(m.group(2))
            chain = m.group(3)
            pka = float(m.group(4))
            out[(chain, resno, name3)] = pka
    return out


__all__ = ["CatalyticPkaResult", "catalytic_pka"]
