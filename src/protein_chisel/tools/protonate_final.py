"""Final-output protonation + PTM annotation + REMARK 668 emission.

This module is the *clean-up* counterpart to ``pdb_restoration.py``. It runs
ONCE on the final top-K PDBs (off the hot path) to produce downstream-clean
files with:

    * full hydrogens on every residue (placed by Rosetta from ideal geometry)
    * standard 3-character residue names (``HIS`` everywhere; not ``HIS_D``)
    * REMARK 666 catalytic-motif lines preserved verbatim from the seed
    * a NEW REMARK 668 block that documents BOTH protonation/tautomer state
      AND post-translational modifications of each catalytic residue,
      indexed to the REMARK 666 motif index
    * ligand HETATM block (with its hydrogens) forced to match the seed
    * clean atom serial numbering, single TER between protein and ligand,
      no CONECT or score-table junk

PTM tracking:
    Some designs intend a catalytic residue to carry a PTM that the seed
    PDB doesn't express. Example: PTE_i1 has LYS 157 as a standard LYS in
    the AF3 seed, but the catalytic mechanism requires it to be KCX
    (carbamylated). REMARK 668 records both the *resolved-from-coords*
    state (what's actually there in the PDB) and the *intended PTM*
    (what the design pipeline knows it should be). PTM is supplied via
    one of three channels (in trust order):
        1. Explicit ``ptm_map={"A:157": "KCX", ...}`` from CLI/API
        2. Auto-detected from seed atom inventory (e.g. CX+OQ1+OQ2 -> KCX)
        3. Auto-detected from the seed residue name itself (KCX/SEP/etc.)

Why two modules:
    ``pdb_restoration.py`` runs INSIDE the design loop on every cycle, so
    it has to be fast and stdlib-only. It uses Rosetta's 5-character
    tautomer labels (``HIS_D`` / ``HIE`` / ``HIP``) so subsequent Rosetta-
    aware steps still know the tautomer. That is correct for the inner
    loop but breaks downstream tools that expect strict PDB column format.

    ``protonate_final.py`` runs ONCE at the end (only on top-K) and is
    PyRosetta-backed. It can afford the model load.

REMARK 668 format (paired with REMARK 666):

    REMARK 668 ---------------------------------------------------------------
    REMARK 668 PROTONATION STATE + PTM OF CATALYTIC RESIDUES (PAIRED W/ R 666)
    REMARK 668 ---------------------------------------------------------------
    REMARK 668   <documentation block: STATE / PTM / ROSETTA_PATCH / H_ATOMS>
    REMARK 668 IDX CHN RESN  RESI STATE PTM H_ATOMS                ROSETTA_PATCH
    REMARK 668   1   A HIS    132   HID  -  HD1                    HIS_D
    REMARK 668   3   A LYS    157   LYS KCX -                      LYS
    ...

The ``IDX`` field is the integer that appears as the trailing motif-index
column of the corresponding ``REMARK 666 ... MATCH MOTIF ... <IDX>`` line.
This makes the pairing unambiguous and machine-parseable.

Public entry points:
    protonate_pdb_with_pyrosetta(input_pdb, ligand_params, out_pdb)
    write_clean_final_pdb(rosetta_pdb, seed_pdb, out_pdb, ligand_resname=None,
                           ptm_map=None)
    protonate_final_topk(topk_dir, seed_pdb, ligand_params, ptm_map=None,
                         out_dir=None)
        ^ end-to-end driver. Accepts a directory of design PDBs and writes
          the cleaned versions next to them (or into ``out_dir`` if given).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Mapping from Rosetta variant patch name -> standard biochemistry tautomer
# code emitted in the REMARK 668 STATE column.
_ROSETTA_TO_STATE: dict[str, str] = {
    # histidine
    "HIS": "HIE",   # default: epsilon-protonated unless we see otherwise
    "HIS_D": "HID",
    "HIS_E": "HIE",
    "HIE": "HIE",
    "HID": "HID",
    "HIP": "HIP",
    # aspartate
    "ASP": "ASP",
    "ASH": "ASH",
    "ASP_P1": "ASH",
    "ASP_P2": "ASH",
    # glutamate
    "GLU": "GLU",
    "GLH": "GLH",
    "GLU_P1": "GLH",
    "GLU_P2": "GLH",
    # lysine
    "LYS": "LYS",
    "LYN": "LYN",
    "KCX": "KCX",
    # cysteine
    "CYS": "CYS",
    "CYM": "CYM",
    "CYX": "CYX",
    # tyrosine
    "TYR": "TYR",
    "TYM": "TYM",
}

# Side-chain protonating-hydrogen atoms keyed by 3-letter resname. We list
# only the H atoms that disambiguate protonation state; backbone H is left
# implicit. Used both to detect state from a hydrated PDB and to emit the
# H_ATOMS column of REMARK 668.
_SIDECHAIN_PROTON_ATOMS: dict[str, frozenset[str]] = {
    "HIS": frozenset({"HD1", "HE2"}),
    "ASP": frozenset({"HD2", "HD1"}),  # OD1 / OD2 protonation in ASH
    "GLU": frozenset({"HE2", "HE1"}),  # OE1 / OE2 protonation in GLH
    "LYS": frozenset({"HZ1", "HZ2", "HZ3"}),
    "CYS": frozenset({"HG"}),
    "TYR": frozenset({"HH"}),
    "KCX": frozenset({"HQ"}),  # carbamate H if present (rare in Rosetta out)
}

# Standard 20 AA + common Rosetta variants we will rename to 3-char form on
# write. Anything else that has a 5-char Rosetta label is an unexpected case
# and we'll keep the 5-char label rather than silently lose info.
#
# KCX is intentionally NOT mapped: it is already a 3-char code in standard
# PDB format (no column overflow) and Rosetta dumps the carbamate atoms
# (CX, OQ1, OQ2 [, HQ]) under the KCX resname. Renaming to LYS would leave
# those atoms orphaned under a "LYS" residue — most parsers reject that.
# Downstream tools should treat KCX as a non-canonical residue using
# REMARK 666 + REMARK 668 metadata.
_VARIANT_TO_STANDARD: dict[str, str] = {
    "HIS_D": "HIS",
    "HIS_E": "HIS",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "ASH": "ASP",
    "ASP_P1": "ASP",
    "ASP_P2": "ASP",
    "GLH": "GLU",
    "GLU_P1": "GLU",
    "GLU_P2": "GLU",
    "LYN": "LYS",
    "CYM": "CYS",
    "CYX": "CYS",
    "TYM": "TYR",
}

# ----------------------------------------------------------------------------
# Post-translational-modification (PTM) registry
# ----------------------------------------------------------------------------
#
# Maps PDB-component PTM codes to a (parent_resname, marker_atoms, blurb)
# tuple. ``parent_resname`` is the canonical 3-letter unmodified residue
# code; ``marker_atoms`` is the set of side-chain atom names that, when
# all present on a residue, identify the PTM uniquely; ``blurb`` is a
# short one-line description for human-readable REMARK output.
#
# To extend: copy a row, look up the wwPDB Chemical Component Dictionary
# entry to get the marker atom set, drop it in. The detector treats the
# first PTM whose marker_atoms is a subset of the residue's atom inventory
# as the match (so the most specific fingerprint wins — list more-
# specific PTMs first).
PTMS: dict[str, dict] = {
    "KCX": {
        "parent": "LYS",
        "atoms": frozenset({"CX", "OQ1", "OQ2"}),
        "blurb": "carbamylated lysine (Zn-coord; ureido)",
    },
    "MLY": {
        "parent": "LYS",
        "atoms": frozenset({"CH1", "CH2"}),
        "blurb": "N6,N6-dimethyl-L-lysine",
    },
    "M3L": {
        "parent": "LYS",
        "atoms": frozenset({"CM1", "CM2", "CM3"}),
        "blurb": "N6,N6,N6-trimethyl-L-lysine",
    },
    "ALY": {
        "parent": "LYS",
        "atoms": frozenset({"CH3", "OH"}),
        "blurb": "N6-acetyl-L-lysine",
    },
    # Phospho-residues: just rely on the phosphorus atom. The 2020 wwPDB
    # remediation renamed phosphate oxygens from O1P/O2P/O3P to
    # OP1/OP2/OP3, so depending on a specific oxygen-name set is fragile.
    # A P atom on a Ser/Thr/Tyr side chain is itself sufficient.
    "SEP": {
        "parent": "SER",
        "atoms": frozenset({"P"}),
        "blurb": "phospho-L-serine",
    },
    "TPO": {
        "parent": "THR",
        "atoms": frozenset({"P"}),
        "blurb": "phospho-L-threonine",
    },
    "PTR": {
        "parent": "TYR",
        "atoms": frozenset({"P"}),
        "blurb": "phospho-L-tyrosine",
    },
    "HYP": {
        "parent": "PRO",
        "atoms": frozenset({"OD1"}),
        "blurb": "4-hydroxy-L-proline",
    },
    "CSO": {
        "parent": "CYS",
        "atoms": frozenset({"OD"}),
        "blurb": "S-hydroxy-L-cysteine (oxidized cys)",
    },
    "CME": {
        "parent": "CYS",
        "atoms": frozenset({"CE", "SD"}),
        "blurb": "S,S-(2-hydroxyethyl)thiocysteine",
    },
    "5HP": {
        "parent": "GLU",
        "atoms": frozenset({"OE"}),
        "blurb": "pyroglutamate (5-oxoproline)",
    },
    "PCA": {
        "parent": "GLN",
        "atoms": frozenset(),  # cyclized; detected by N-terminal context
        "blurb": "pyroglutamate (pyrrolidone carboxylic acid)",
    },
}

# Reverse: parent residue -> set of PTMs that derive from it. Useful for
# narrowing the auto-detect search.
_PARENT_TO_PTMS: dict[str, list[str]] = {}
for _ptm_code, _info in PTMS.items():
    _PARENT_TO_PTMS.setdefault(_info["parent"], []).append(_ptm_code)
del _ptm_code, _info


# ----------------------------------------------------------------------------
# Lightweight PDB line helpers (column-aware, 5-char-resname-aware)
# ----------------------------------------------------------------------------


def _is_atom_line(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM")


# Whitespace-split fallback for ATOM/HETATM records that look corrupted
# (e.g. column drift from a non-PDB-strict writer). We try the column
# parse first; if that yields nonsense, we fall back to splitting tokens
# and matching by position. PDB record token order is:
#   ATOM <serial> <atom_name> [altloc]<resname> <chain><resno>[icode] x y z occ b ... <element>
# but with chain ID frequently glued to atom_name or resname after column
# corruption. Best-effort, never raises.
def _whitespace_parse(line: str) -> dict | None:
    """Return ``{"atom_name", "resname", "chain", "resno", "element"}``
    via whitespace tokenization, or None if it can't be reasonably
    interpreted. Used when column parsing yields garbage.

    Expected layouts (we try both):
        [ATOM, serial, atom, resname,        resno, x, y, z, ...]
        [ATOM, serial, atom, resname, chain, resno, x, y, z, ...]
        [ATOM, serial, atom, resname_with_chain_glued, resno, x, y, z, ...]

    We disambiguate by walking parts looking for the first token that
    parses as an integer (which is resno).
    """
    parts = line.split()
    if len(parts) < 6 or parts[0] not in {"ATOM", "HETATM"}:
        return None

    # Find resno: first token after parts[2] (atom_name) that parses as int.
    # Allow trailing insertion-code letter (e.g. "132A").
    resno = None
    resno_idx = None
    icode = ""
    for i in range(3, min(len(parts), 7)):
        tok = parts[i]
        m = re.match(r"^(-?\d+)([A-Za-z]?)$", tok)
        if m:
            resno = int(m.group(1))
            icode = m.group(2)
            resno_idx = i
            break
    if resno is None:
        return None

    atom_name = parts[2]
    # Tokens between parts[3] and parts[resno_idx] hold resname [+ chain]
    middle = parts[3:resno_idx]
    chain = ""
    resname = "UNK"

    if len(middle) == 1:
        # Either glued ("HIS_DA") or no chain
        token = middle[0]
        if len(token) > 4:
            # Try plausible chain split (last char is chain if alphabetic and
            # the prefix is a known resname/variant/PTM)
            for split_at in (5, 4, 3):
                if split_at >= len(token):
                    continue
                head, tail = token[:split_at], token[split_at:]
                if len(tail) != 1 or not tail.isalpha():
                    continue
                if (head in _VARIANT_TO_STANDARD or head in PTMS
                        or (len(head) == 3 and head.isalpha())):
                    resname = head
                    chain = tail
                    break
            else:
                resname = token  # leave as-is; downstream may still cope
        else:
            resname = token
            chain = ""
    elif len(middle) == 2:
        # resname, chain
        resname, chain = middle[0], middle[1]
    elif len(middle) >= 3:
        # Unusual: take first as resname, last as chain
        resname = middle[0]
        chain = middle[-1] if len(middle[-1]) == 1 and middle[-1].isalpha() else ""

    return {
        "atom_name": atom_name,
        "resname": resname,
        "chain": chain,
        "resno": resno,
        "icode": icode,
        "element": "",
        "raw_resname_field": "",
    }


def parse_atom_line(line: str) -> dict | None:
    """Robust column-then-whitespace parser for ATOM/HETATM records.

    Returns a dict with keys ``atom_name``, ``resname``, ``chain``,
    ``resno``, ``icode``, ``element``, ``raw_resname_field`` (the literal
    text in cols 17-21 useful for round-trip writes).

    Returns None for non-ATOM lines and for lines that defeat both
    column and whitespace parsing.

    Robustness:
        - Column parse first (PDB strict).
        - If the parsed resno is non-integer or chain ID looks like a
          letter that's part of a 5-char overflow (e.g. "A" right after
          "HIS_D"), trust the column parse for resname and resno.
        - If column parse raises, fall back to whitespace tokenization.
    """
    if not _is_atom_line(line) or len(line) < 26:
        return None
    raw_resname_field = line[16:21] if len(line) >= 21 else line[16:].rstrip("\n")
    five = raw_resname_field.strip()
    # Resname: prefer 5-char rosetta label if it matches our registry,
    # otherwise the 3-char standard slot.
    if five in _VARIANT_TO_STANDARD or five in PTMS or five in {"HIS_D", "HIS_E"}:
        resname = five
    else:
        resname = line[17:20].strip()

    # Chain: col 22. If the 5-char label overflowed it, the col 22 char
    # is the chain ID itself (e.g. for "HIS_DA" the 'A' at col 22 is
    # chain, not part of the label).
    chain = line[21:22] if len(line) >= 22 else " "
    if len(five) == 5 and chain.strip() == "":
        # Some writers eat the chain when overflowing; treat as blank.
        chain = ""

    icode = line[26:27] if len(line) >= 27 else " "

    # Resno: try column [22:26]; fall back to whitespace.
    try:
        resno = int(line[22:26])
    except (ValueError, TypeError):
        ws = _whitespace_parse(line)
        if ws is None:
            return None
        return {**ws, "icode": "", "raw_resname_field": raw_resname_field}

    atom_name = line[12:16].strip() if len(line) >= 16 else ""
    element = ""
    if len(line) >= 78:
        element = line[76:78].strip()

    # Extra sanity: resname must be alphabetic (and ≤5 chars).
    if not resname or not re.match(r"^[A-Z0-9_]{1,5}$", resname):
        ws = _whitespace_parse(line)
        if ws is not None:
            return {**ws, "icode": "", "raw_resname_field": raw_resname_field}

    return {
        "atom_name": atom_name,
        "resname": resname,
        "chain": chain,
        "resno": resno,
        "icode": icode if icode.strip() else "",
        "element": element,
        "raw_resname_field": raw_resname_field,
    }


def _resname_from_line(line: str) -> str:
    """Return the residue name, handling Rosetta 5-char labels (cols 17-21)."""
    parsed = parse_atom_line(line)
    if parsed is None:
        return ""
    return parsed["resname"]


def _resno_from_line(line: str) -> int:
    parsed = parse_atom_line(line)
    if parsed is None:
        raise ValueError(f"could not parse ATOM line: {line!r}")
    return parsed["resno"]


def _chain_from_line(line: str) -> str:
    parsed = parse_atom_line(line)
    if parsed is None:
        return " "
    return parsed["chain"]


def _atom_name_from_line(line: str) -> str:
    parsed = parse_atom_line(line)
    if parsed is None:
        return ""
    return parsed["atom_name"]


def _element_from_line(line: str) -> str:
    parsed = parse_atom_line(line)
    if parsed is None:
        return ""
    return parsed["element"]


def _looks_like_hydrogen(line: str) -> bool:
    """True for any H atom line (ATOM or HETATM)."""
    elem = _element_from_line(line)
    if elem == "H":
        return True
    if not elem:
        atom = _atom_name_from_line(line)
        # Atom names starting with H or like "1H", "2H", etc.
        if atom.startswith("H"):
            return True
        if atom[:1].isdigit() and "H" in atom:
            return True
    return False


def _rewrite_resname_to_standard(line: str, new_resname: str) -> str:
    """Rewrite an ATOM/HETATM line so cols 17-22 hold:
        col 17:    alt-loc indicator (blank)
        col 18-20: 3-char residue name
        col 21:    blank
        col 22:    chain ID (preserved from input col 22 = Python index 21)

    Works for both already-standard 3-char input and overflow 5-char input
    (HIS_D/HIE/HIP/etc.) where the chain ID lives at col 22 on disk
    regardless of the resname width.
    """
    if len(line) < 22:
        return line
    return (
        line[:16]                # cols 1-16: atom name region
        + " "                     # col 17:    alt-loc blank
        + new_resname.ljust(3)    # cols 18-20: 3-char resname
        + " "                     # col 21:    blank
        + line[21:]               # col 22 onward: chain ID + rest
    )


# ----------------------------------------------------------------------------
# REMARK 666 parsing
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class Remark666Entry:
    """One catalytic-motif anchor described by a REMARK 666 line."""

    raw: str               # original REMARK 666 line (with newline)
    template_chain: str    # chain of the template (ligand)
    template_resname: str  # resname of the template (e.g. "YYE")
    template_resno: int
    motif_chain: str       # chain of the catalytic residue (e.g. "A")
    motif_resname: str     # resname of the catalytic residue (e.g. "HIS")
    motif_resno: int
    motif_index: int       # the trailing integer that REMARK 668 will pair to
    matched: int           # second trailing integer (Rosetta sets this to 1)


# Pattern: REMARK 666 MATCH TEMPLATE <ch> <res> <no> MATCH MOTIF <ch> <res> <no> <idx> <matched>
_RE_REMARK_666 = re.compile(
    r"^REMARK\s+666\s+MATCH\s+TEMPLATE\s+(\S)\s+(\S+)\s+(\d+)"
    r"\s+MATCH\s+MOTIF\s+(\S)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
)


def parse_remark_666(pdb_path: str | Path) -> list[Remark666Entry]:
    """Extract every REMARK 666 motif entry from a PDB."""
    entries: list[Remark666Entry] = []
    with open(pdb_path) as fh:
        for line in fh:
            stripped = line.rstrip()
            m = _RE_REMARK_666.match(stripped)
            if not m:
                continue
            entries.append(
                Remark666Entry(
                    raw=line if line.endswith("\n") else line + "\n",
                    template_chain=m.group(1),
                    template_resname=m.group(2),
                    template_resno=int(m.group(3)),
                    motif_chain=m.group(4),
                    motif_resname=m.group(5),
                    motif_resno=int(m.group(6)),
                    motif_index=int(m.group(7)),
                    matched=int(m.group(8)),
                )
            )
    return entries


# ----------------------------------------------------------------------------
# Protonation-state detection from a hydrated PDB
# ----------------------------------------------------------------------------


def _collect_residue_atom_inventory(
    pdb_path: str | Path,
) -> dict[tuple[str, int], dict]:
    """Return {(chain, resno) -> {"resname": str, "atoms": set[str], "lines": [str]}}.

    ``resname`` is whatever the line wrote (could be 3 or 5 char). ``atoms``
    is the set of atom-name strings present.
    """
    out: dict[tuple[str, int], dict] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not _is_atom_line(line):
                continue
            key = (_chain_from_line(line), _resno_from_line(line))
            entry = out.setdefault(
                key,
                {"resname": _resname_from_line(line), "atoms": set(), "lines": []},
            )
            entry["atoms"].add(_atom_name_from_line(line))
            entry["lines"].append(line)
    return out


def detect_protonation_state(
    resname_at_input: str,
    atoms_present: Iterable[str],
) -> tuple[str, list[str]]:
    """Determine the biochemistry state code + which H atoms identify it.

    Args:
        resname_at_input: residue name as written in the input PDB (may be a
            Rosetta 5-char label like ``HIS_D``).
        atoms_present: atom names on this residue.

    Returns:
        ``(STATE, [H atoms responsible])``. STATE is one of HID/HIE/HIP for
        HIS, ASP/ASH for ASP, GLU/GLH for GLU, LYS/LYN/KCX for LYS, CYS/CYM/
        CYX for CYS, TYR/TYM for TYR. Falls back to the standard 3-letter
        name for everything else.
    """
    atoms = set(atoms_present)
    standard = _VARIANT_TO_STANDARD.get(resname_at_input, resname_at_input)
    standard = standard.strip()

    if standard == "HIS":
        # Trust an explicit 5-char label first.
        if resname_at_input in {"HIS_D", "HID"}:
            return "HID", ["HD1"] if "HD1" in atoms else []
        if resname_at_input in {"HIS_E", "HIE"}:
            return "HIE", ["HE2"] if "HE2" in atoms else []
        if resname_at_input == "HIP":
            present = [h for h in ("HD1", "HE2") if h in atoms]
            return "HIP", present
        # No explicit label: infer from H atoms.
        has_hd1 = "HD1" in atoms
        has_he2 = "HE2" in atoms
        if has_hd1 and has_he2:
            return "HIP", ["HD1", "HE2"]
        if has_hd1:
            return "HID", ["HD1"]
        if has_he2:
            return "HIE", ["HE2"]
        # No protonating H found at all (stripped or never added) — leave as HIE
        # (default Rosetta tautomer) and report empty H_ATOMS.
        return "HIE", []

    if standard == "ASP":
        if resname_at_input in {"ASH", "ASP_P1", "ASP_P2"} or "HD2" in atoms or "HD1" in atoms:
            return "ASH", [h for h in ("HD2", "HD1") if h in atoms]
        return "ASP", []

    if standard == "GLU":
        if resname_at_input in {"GLH", "GLU_P1", "GLU_P2"} or "HE2" in atoms or "HE1" in atoms:
            return "GLH", [h for h in ("HE2", "HE1") if h in atoms]
        return "GLU", []

    if standard == "LYS":
        if resname_at_input == "KCX" or {"CX", "OQ1", "OQ2"}.issubset(atoms):
            return "KCX", [a for a in ("HQ",) if a in atoms]
        if resname_at_input == "LYN":
            return "LYN", [h for h in ("HZ1", "HZ2") if h in atoms]
        # Default: protonated NH3+
        present = [h for h in ("HZ1", "HZ2", "HZ3") if h in atoms]
        return "LYS", present

    if standard == "CYS":
        if resname_at_input == "CYX":
            return "CYX", []
        if resname_at_input == "CYM" or ("HG" not in atoms and "SG" in atoms):
            return "CYM", []
        return "CYS", ["HG"] if "HG" in atoms else []

    if standard == "TYR":
        if resname_at_input == "TYM" or ("HH" not in atoms and "OH" in atoms):
            return "TYM", []
        return "TYR", ["HH"] if "HH" in atoms else []

    # Fallthrough: report the standard resname unchanged.
    return standard, []


# ----------------------------------------------------------------------------
# REMARK 668 emission
# ----------------------------------------------------------------------------


_REMARK_668_HEADER: tuple[str, ...] = (
    "REMARK 668 -----------------------------------------------------------------\n",
    "REMARK 668 PROTONATION STATE + PTM OF CATALYTIC RESIDUES (PAIRED W/ R 666)\n",
    "REMARK 668 -----------------------------------------------------------------\n",
    "REMARK 668 Each entry below documents the protonation/tautomer state AND\n",
    "REMARK 668 any post-translational modification of one catalytic residue\n",
    "REMARK 668 listed in REMARK 666. The IDX field matches the trailing motif\n",
    "REMARK 668 index of the corresponding REMARK 666 MATCH MOTIF line;\n",
    "REMARK 668 CHN/RESN/RESI duplicate that line's chain, resname, resno for\n",
    "REMARK 668 cross-check. STATE uses standard biochemistry conventions:\n",
    "REMARK 668   HIS: HID (ND1-H, delta), HIE (NE2-H, epsilon), HIP (both)\n",
    "REMARK 668   ASP: ASP (deprot), ASH (OD2-H)\n",
    "REMARK 668   GLU: GLU (deprot), GLH (OE2-H)\n",
    "REMARK 668   LYS: LYS (NH3+), LYN (NH2), KCX (carbamylated)\n",
    "REMARK 668   CYS: CYS (thiol), CYM (thiolate), CYX (disulfide)\n",
    "REMARK 668   TYR: TYR (phenol), TYM (phenolate)\n",
    "REMARK 668 PTM is the wwPDB Chemical Component Dictionary 3-letter code\n",
    "REMARK 668 for any post-translational modification known to be intended\n",
    "REMARK 668 at this position (e.g. KCX, SEP, TPO, PTR, MLY, M3L, ALY, HYP).\n",
    "REMARK 668 \"-\" means no PTM declared. PTM is set from explicit user/CLI\n",
    "REMARK 668 declaration, the seed PDB resname, or auto-detect from the\n",
    "REMARK 668 seed atom inventory (e.g. CX/OQ1/OQ2 -> KCX). RESN may show the\n",
    "REMARK 668 unmodified residue (e.g. LYS) even when PTM=KCX, because the\n",
    "REMARK 668 reference structure stored the residue in unmodified form;\n",
    "REMARK 668 PTM tells downstream consumers what modification to apply.\n",
    "REMARK 668 ROSETTA_PATCH is the Rosetta variant_type tag actually present\n",
    "REMARK 668 in the dumped pose (HIS_D, KCX, etc.). H_ATOMS lists the side-\n",
    "REMARK 668 chain protonating hydrogens present; \"-\" means none observed.\n",
    "REMARK 668 -----------------------------------------------------------------\n",
    # Header columns must align EXACTLY with the data-line format string in
    # format_remark_668_line(). Data line layout (after "REMARK 668 " prefix):
    #   col 12-14: IDX   (>3d), col 15: space
    #   col 16-18: CHN   (>3s), col 19: space
    #   col 20-23: RESN  (<4s), col 24: space
    #   col 25-29: RESI  (>5d), col 30: space
    #   col 31-35: STATE (>5s), col 36: space
    #   col 37-39: PTM   (<3s), col 40: space
    #   col 41-63: H_ATOMS (<23s), col 64: space
    #   col 65-72: ROSETTA_PATCH (<8s)
    "REMARK 668 IDX CHN RESN  RESI STATE PTM H_ATOMS                 ROSETTA_PATCH\n",
)
_REMARK_668_FOOTER: tuple[str, ...] = (
    "REMARK 668 -----------------------------------------------------------------\n",
)


def format_remark_668_line(
    idx: int,
    chain: str,
    resname: str,
    resno: int,
    state: str,
    h_atoms: Sequence[str],
    rosetta_patch: str,
    ptm: str = "-",
) -> str:
    """Build one fixed-column REMARK 668 data row.

    Args:
        idx: REMARK 666 motif index (1-based).
        chain: chain ID (1 char).
        resname: standard 3-char residue name (HIS, LYS, ...).
        resno: residue sequence number.
        state: biochemistry tautomer code (HID, HIE, HIP, LYS, GLU, ...).
        h_atoms: list of side-chain H atom names that identify the state.
        rosetta_patch: Rosetta variant_type tag actually in the pose.
        ptm: 3-letter wwPDB CCD code for declared/detected PTM,
            or "-" if none.
    """
    h_str = ",".join(h_atoms) if h_atoms else "-"
    return (
        f"REMARK 668 "                # 11 chars (cols 1-11)
        f"{idx:>3d} "                  # IDX  (12-14) + space
        f"{chain:>3s} "                # CHN  (16-18) + space
        f"{resname:<4s} "              # RESN (20-23) + space
        f"{resno:>5d} "                # RESI (25-29) + space
        f"{state:>5s} "                # STATE (31-35) + space
        f"{ptm:<3s} "                  # PTM  (37-39) + space
        f"{h_str:<23s} "               # H_ATOMS (41-63) + space
        f"{rosetta_patch:<8s}"         # ROSETTA_PATCH (65-72)
        "\n"
    )


def parse_ptm_map(spec: str | None) -> dict[tuple[str, int], str]:
    """Parse a CLI/API PTM spec into a ``{(chain, resno) -> code}`` map.

    Accepted formats (mix-and-match in a comma list):
        "A:157=KCX"           # chain:resno=code
        "A157=KCX"            # chain<resno>=code (no colon)
        "157=KCX"             # default chain (treated as 'A')
    Whitespace and trailing semicolons are tolerated.
    """
    out: dict[tuple[str, int], str] = {}
    if not spec:
        return out
    parts = re.split(r"[,;\s]+", spec.strip())
    for part in parts:
        if not part:
            continue
        try:
            key, code = part.split("=", 1)
        except ValueError:
            LOGGER.warning("parse_ptm_map: skipping malformed entry %r", part)
            continue
        code = code.strip().upper()
        key = key.strip()
        m = re.match(r"^([A-Za-z])?:?(\d+)$", key)
        if not m:
            LOGGER.warning("parse_ptm_map: skipping malformed key %r", key)
            continue
        chain = m.group(1) or "A"
        resno = int(m.group(2))
        if code not in PTMS and code != "-":
            LOGGER.warning(
                "parse_ptm_map: PTM code %r at %s%d is not in known PTMS "
                "registry; emitting it but downstream tools may not recognize",
                code, chain, resno,
            )
        out[(chain, resno)] = code
    return out


def detect_ptms_from_inventory(
    inventory: dict[tuple[str, int], dict],
) -> dict[tuple[str, int], str]:
    """Auto-detect PTMs from a residue-atom inventory.

    For each residue, if the side-chain atom set contains the marker
    atoms of a known PTM (and the resname matches the PTM's parent or
    is the PTM code itself), record the PTM code. The first matching
    PTM in registration order wins per residue (so list narrower
    fingerprints first in PTMS).

    Args:
        inventory: ``{(chain, resno) -> {"resname": str, "atoms": set, ...}}``
            as produced by ``_collect_residue_atom_inventory``.

    Returns:
        ``{(chain, resno) -> ptm_code}``. Residues with no detectable PTM
        are absent from the dict.
    """
    out: dict[tuple[str, int], str] = {}
    for key, info in inventory.items():
        resname = info["resname"].strip()
        # Direct hit: resname IS a PTM code
        if resname in PTMS:
            out[key] = resname
            continue
        # Marker-atom hit: resname is the parent and all PTM marker atoms present
        candidates = _PARENT_TO_PTMS.get(resname, [])
        # Also handle the case where resname is a Rosetta variant of the parent
        std = _VARIANT_TO_STANDARD.get(resname, resname)
        if std != resname:
            candidates = _PARENT_TO_PTMS.get(std, [])
        atoms = info["atoms"]
        for ptm_code in candidates:
            markers = PTMS[ptm_code]["atoms"]
            if markers and markers.issubset(atoms):
                out[key] = ptm_code
                break
    return out


def resolve_ptm_map(
    seed_pdb: str | Path,
    explicit_ptm_map: Optional[dict[tuple[str, int], str]] = None,
) -> dict[tuple[str, int], str]:
    """Resolve the final PTM map for a design from all available sources.

    Trust order (later wins):
        1. Auto-detect from the seed PDB's atom inventory + resnames.
        2. Explicit ``explicit_ptm_map`` from CLI/API. Use "-" as the
           value to FORCE no-PTM for a residue (overrides auto-detect).

    Args:
        seed_pdb: Reference PDB to scan for PTM markers.
        explicit_ptm_map: User-supplied overrides.

    Returns:
        Final ``{(chain, resno) -> ptm_code}`` map (without "-" entries).
    """
    inventory = _collect_residue_atom_inventory(seed_pdb)
    final: dict[tuple[str, int], str] = dict(detect_ptms_from_inventory(inventory))
    if explicit_ptm_map:
        for key, code in explicit_ptm_map.items():
            if code == "-":
                final.pop(key, None)
            else:
                final[key] = code
    return final


def build_remark_668_block(
    rosetta_pdb: str | Path,
    seed_pdb: str | Path,
    ptm_map: Optional[dict[tuple[str, int], str]] = None,
) -> list[str]:
    """Build the full REMARK 668 block for a hydrated design PDB.

    Reads REMARK 666 entries from ``seed_pdb`` and per-residue protonation
    info from ``rosetta_pdb`` (the freshly hydrated, Rosetta-output PDB).

    Args:
        rosetta_pdb: PyRosetta-hydrated dump (source of STATE / H_ATOMS).
        seed_pdb: Original seed PDB (source of REMARK 666 + auto-detected PTMs).
        ptm_map: User-supplied overrides as a ``{(chain, resno) -> code}``
            map. Use "-" as the value to FORCE no-PTM annotation for a
            residue. ``None`` defaults to auto-detect-only from seed.
    """
    motif_entries = parse_remark_666(seed_pdb)
    if not motif_entries:
        return []

    inventory = _collect_residue_atom_inventory(rosetta_pdb)
    resolved_ptm_map = resolve_ptm_map(seed_pdb, ptm_map)

    lines: list[str] = list(_REMARK_668_HEADER)
    for entry in motif_entries:
        key = (entry.motif_chain, entry.motif_resno)
        residue = inventory.get(key)
        ptm_code = resolved_ptm_map.get(key, "-")
        if residue is None:
            # Catalytic residue missing from output — emit a placeholder so
            # the pairing is still complete and the absence is visible.
            lines.append(
                format_remark_668_line(
                    idx=entry.motif_index,
                    chain=entry.motif_chain,
                    resname=entry.motif_resname,
                    resno=entry.motif_resno,
                    state="???",
                    h_atoms=[],
                    rosetta_patch=entry.motif_resname,
                    ptm=ptm_code,
                )
            )
            continue
        rosetta_patch = residue["resname"].strip()
        state, h_atoms = detect_protonation_state(rosetta_patch, residue["atoms"])
        lines.append(
            format_remark_668_line(
                idx=entry.motif_index,
                chain=entry.motif_chain,
                resname=_VARIANT_TO_STANDARD.get(rosetta_patch, rosetta_patch),
                resno=entry.motif_resno,
                state=state,
                h_atoms=h_atoms,
                rosetta_patch=rosetta_patch if rosetta_patch else entry.motif_resname,
                ptm=ptm_code,
            )
        )
    lines.extend(_REMARK_668_FOOTER)
    return lines


# ----------------------------------------------------------------------------
# Final-PDB writer (combines hydrated coords + REMARK header + ligand)
# ----------------------------------------------------------------------------


def _extract_seed_header_lines(seed_pdb: str | Path) -> list[str]:
    """REMARK 666, HETNAM, LINK from the seed (these flow downstream)."""
    keep_prefixes = ("REMARK 666", "HETNAM", "LINK")
    out: list[str] = []
    with open(seed_pdb) as fh:
        for line in fh:
            if _is_atom_line(line):
                break
            for p in keep_prefixes:
                if line.startswith(p):
                    out.append(line if line.endswith("\n") else line + "\n")
                    break
    return out


def _extract_seed_ligand_atoms(
    seed_pdb: str | Path,
    ligand_resname: Optional[str] = None,
) -> list[str]:
    """Return all HETATM lines (incl. hydrogens) for the ligand from seed.

    If ``ligand_resname`` is given, take only HETATM lines matching it.
    Otherwise: identify the ligand as the largest non-water HETATM residue.
    """
    if ligand_resname is None:
        # Auto-detect: largest non-water HETATM group.
        groups: dict[tuple[str, int, str], list[str]] = {}
        with open(seed_pdb) as fh:
            for line in fh:
                if not line.startswith("HETATM"):
                    continue
                resname = line[17:20].strip()
                if resname in {"HOH", "WAT", "DOD", "TIP", "TIP3"}:
                    continue
                key = (line[21:22], int(line[22:26]), resname)
                groups.setdefault(key, []).append(line)
        if not groups:
            return []
        best = max(groups.values(), key=len)
        return [ln if ln.endswith("\n") else ln + "\n" for ln in best]

    out: list[str] = []
    with open(seed_pdb) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            if line[17:20].strip() == ligand_resname:
                out.append(line if line.endswith("\n") else line + "\n")
    return out


def _renumber_serial(line: str, serial: int) -> str:
    """Rewrite cols 7-11 with a new atom serial number."""
    return line[:6] + f"{serial:>5d}" + line[11:]


def write_clean_final_pdb(
    rosetta_pdb: str | Path,
    seed_pdb: str | Path,
    out_pdb: str | Path,
    ligand_resname: Optional[str] = None,
    ptm_map: Optional[dict[tuple[str, int], str]] = None,
) -> dict:
    """Combine a Rosetta-hydrated PDB with the seed's REMARK 666 + ligand.

    Output file structure:
        REMARK 666 ... (from seed)
        HETNAM / LINK / REMARK PDBinfo-LABEL (from seed)
        REMARK 668 ... (computed here, includes PTM column)
        ATOM   ... (from rosetta_pdb, with HIS_D/etc. -> HIS, atoms renumbered)
        TER
        HETATM ... (ligand, from seed; preserves seed hydrogens exactly)
        END

    Args:
        rosetta_pdb: A PDB hydrated by PyRosetta. Side-chain tautomers are
            embedded in residue names (HIS_D etc.) which we will normalize.
        seed_pdb: The original seed PDB. Source of REMARK 666 / HETNAM /
            LINK / ligand HETATM coordinates.
        out_pdb: Where to write the final cleaned PDB.
        ligand_resname: Optional 3-letter ligand code. ``None`` = auto-detect
            (largest non-water HETATM group in seed).
        ptm_map: User-supplied PTM declarations as ``{(chain, resno) -> code}``.
            Use "-" as the value to FORCE no-PTM annotation for a residue
            (overrides auto-detect). ``None`` = auto-detect-only from seed.

    Returns:
        Stats dict with counts of remark lines written, atoms renumbered,
        ligand atoms inserted, and HIS/variant relabels.
    """
    rosetta_pdb = Path(rosetta_pdb)
    seed_pdb = Path(seed_pdb)
    out_pdb = Path(out_pdb)

    stats = {
        "remark_666_in": 0,
        "remark_668_lines": 0,
        "atoms_protein": 0,
        "atoms_ligand": 0,
        "variants_normalized": 0,
        "score_lines_dropped": 0,
        "conect_lines_dropped": 0,
        "ptms_declared": 0,
    }

    # 1. Pull seed-derived header lines (REMARK 666 etc.) -------------------
    header_lines = _extract_seed_header_lines(seed_pdb)
    stats["remark_666_in"] = sum(1 for ln in header_lines if ln.startswith("REMARK 666"))

    # 2. Compute REMARK 668 ------------------------------------------------
    remark_668 = build_remark_668_block(rosetta_pdb, seed_pdb, ptm_map=ptm_map)
    # Count data rows (skip header/footer/separator/documentation lines).
    # Data rows start with "REMARK 668 " followed by a digit (the IDX field).
    data_row_pattern = re.compile(r"^REMARK 668\s+\d+\s")
    data_rows = [ln for ln in remark_668 if data_row_pattern.match(ln)]
    # PTM field is at fixed cols 37-39 (Python slice 36:39) per the format
    # string in format_remark_668_line. "-" means no PTM declared.
    stats["ptms_declared"] = sum(
        1 for ln in data_rows if ln[36:39].strip() not in ("", "-")
    )
    # Count any REMARK 668 line that is NOT a separator (cosmetic dashes).
    # Note: str.lstrip(chars) takes a character SET, not a prefix — a previous
    # version used it as a prefix-strip and worked only by accident.
    stats["remark_668_lines"] = sum(
        1 for ln in remark_668
        if ln.startswith("REMARK 668 ")
        and not ln[11:].lstrip().startswith("---")
    )

    # 3. Walk rosetta_pdb and rewrite ATOM lines to standard 3-char names. -
    protein_atom_lines: list[str] = []
    in_score_table = False
    with open(rosetta_pdb) as fh:
        for line in fh:
            # Drop Rosetta score-table junk
            if line.startswith("# All scores below") or line.startswith("#BEGIN_POSE_ENERGIES_TABLE"):
                in_score_table = True
                stats["score_lines_dropped"] += 1
                continue
            if line.startswith("#END_POSE_ENERGIES_TABLE"):
                in_score_table = False
                stats["score_lines_dropped"] += 1
                continue
            if in_score_table:
                stats["score_lines_dropped"] += 1
                continue
            if line.startswith("CONECT"):
                stats["conect_lines_dropped"] += 1
                continue
            if not _is_atom_line(line):
                continue  # skip rosetta's REMARKs / HETATM / TER / END; we rebuild
            # We only take ATOM lines (protein); ligand HETATM comes from seed.
            if line.startswith("HETATM"):
                continue
            resname = _resname_from_line(line)
            standard = _VARIANT_TO_STANDARD.get(resname)
            if standard is not None:
                line = _rewrite_resname_to_standard(line, standard)
                stats["variants_normalized"] += 1
            elif len(resname) > 3:
                # Unknown 5-char label: keep as-is but log so we can extend
                # _VARIANT_TO_STANDARD if it shows up routinely.
                LOGGER.warning(
                    "protonate_final: unknown 5-char residue label %r at "
                    "%s%s; left in 5-char form",
                    resname, _chain_from_line(line), _resno_from_line(line),
                )
            protein_atom_lines.append(line)

    # 4. Pull ligand HETATM block (with H) from seed -----------------------
    ligand_lines = _extract_seed_ligand_atoms(seed_pdb, ligand_resname)

    # 5. Renumber serials end-to-end ---------------------------------------
    output_lines: list[str] = []
    output_lines.extend(header_lines)
    output_lines.extend(remark_668)

    serial = 1
    for line in protein_atom_lines:
        output_lines.append(_renumber_serial(line, serial))
        serial += 1
        stats["atoms_protein"] += 1

    if protein_atom_lines and ligand_lines:
        output_lines.append("TER\n")

    for line in ligand_lines:
        output_lines.append(_renumber_serial(line, serial))
        serial += 1
        stats["atoms_ligand"] += 1

    if protein_atom_lines or ligand_lines:
        output_lines.append("TER\n")
    output_lines.append("END\n")

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdb, "w") as fh:
        fh.writelines(output_lines)

    LOGGER.info(
        "write_clean_final_pdb: %s -> %s | REMARK 666=%d, REMARK 668=%d, "
        "protein atoms=%d, ligand atoms=%d, variants normalized=%d",
        rosetta_pdb.name, out_pdb.name,
        stats["remark_666_in"], stats["remark_668_lines"],
        stats["atoms_protein"], stats["atoms_ligand"],
        stats["variants_normalized"],
    )
    return stats


# ----------------------------------------------------------------------------
# PyRosetta hydration + driver entry point
# ----------------------------------------------------------------------------


def normalize_pdb_for_pyrosetta(
    input_pdb: str | Path,
    out_pdb: str | Path,
) -> dict[tuple[str, int], str]:
    """Rewrite an input PDB so PyRosetta's PDB reader will load every residue.

    The MPNN-restored PDBs use Rosetta 5-character residue labels (HIS_D,
    HIE, HIP, KCX) packed into PDB columns 17-21. That overflows the
    standard 3-char residue-name field (cols 18-20) and writes into the
    chain-ID column (col 22). Strict PDB readers — including PyRosetta's
    — silently drop residues whose chain ID can't be parsed.

    We rewrite each affected ATOM/HETATM line so:
      - cols 17-19 hold a standard 3-char residue name (HIS, LYS, ...)
      - col 22 holds the original chain ID
      - col 21 is blank
    and we return a map ``{(chain, resno) -> original_5char_label}`` so
    the caller can re-apply Rosetta variant types after pose construction.

    Args:
        input_pdb: PDB path with possibly malformed 5-char resnames.
        out_pdb: Where to write the normalized copy.

    Returns:
        Dict ``{(chain, resno) -> 5char_label}`` for every residue that
        had a non-standard label and needs a variant_type re-applied
        post-load.
    """
    input_pdb = Path(input_pdb)
    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    variant_map: dict[tuple[str, int], str] = {}
    with open(input_pdb) as fin, open(out_pdb, "w") as fout:
        for line in fin:
            if not _is_atom_line(line):
                fout.write(line)
                continue
            five_char = line[16:21].strip()
            standard = _VARIANT_TO_STANDARD.get(five_char)
            if standard is None:
                # Already standard 3-char; pass through
                fout.write(line)
                continue
            # Layout of malformed input (1-indexed cols):
            #   col: 1234567890123456789012345
            #        ATOM    463  N  HIS_DA  60
            # cols 17-21 hold "HIS_D" (the Rosetta 5-char label),
            # col 22 holds the original chain ID 'A'.
            #
            # PDB-strict layout for output:
            #   cols 13-16 atom name, col 17 alt-loc (blank),
            #   cols 18-20 resname (3-char), col 21 blank, col 22 chain.
            chain = line[21:22]
            resno = int(line[22:26])
            variant_map[(chain, resno)] = five_char
            new_line = (
                line[:16]                # cols 1-16: atom name region
                + " "                     # col 17: alt-loc blank
                + standard.ljust(3)       # cols 18-20: 3-char resname
                + " "                     # col 21: blank
                + line[21:]               # col 22 onward: chain ID + rest
            )
            fout.write(new_line)

    return variant_map


_PYROSETTA_INITED: bool = False
_PYROSETTA_INIT_PARAMS: tuple = ()


def _ensure_pyrosetta_inited(
    ligand_params: Optional[Iterable[str | Path]] = None,
    extra_init_flags: Optional[Iterable[str]] = None,
) -> None:
    """Initialize PyRosetta once per process.

    PyRosetta is a singleton — the first call to ``init`` controls the
    options for the entire session. Subsequent calls are no-ops, so for
    a directory of PDBs sharing the same ligand params we want to init
    once with the right ``-extra_res_fa`` set rather than per PDB.

    If a later caller passes a DIFFERENT ``ligand_params`` set, that
    second-call ligand params are silently ignored by Rosetta — we log
    a loud warning so the user can spot the misuse.
    """
    global _PYROSETTA_INITED, _PYROSETTA_INIT_PARAMS
    params_paths = tuple(
        sorted(str(Path(p).resolve()) for p in (ligand_params or ()))
    )
    if _PYROSETTA_INITED:
        if params_paths != _PYROSETTA_INIT_PARAMS:
            LOGGER.warning(
                "PyRosetta is already initialized with ligand_params=%r; "
                "the new request for %r is being IGNORED. "
                "If you need different ligand params, run in a fresh Python "
                "process (or restructure to init once with the union set).",
                _PYROSETTA_INIT_PARAMS, params_paths,
            )
        return
    import pyrosetta  # type: ignore[import-not-found]

    flags: list[str] = ["-mute all", "-out:level 0"]
    if params_paths:
        flags.append("-extra_res_fa " + " ".join(params_paths))
    flags.append("-ignore_unrecognized_res")
    flags.append("-load_PDB_components false")
    if extra_init_flags:
        flags.extend(extra_init_flags)
    pyrosetta.init(" ".join(flags), silent=True)
    _PYROSETTA_INITED = True
    _PYROSETTA_INIT_PARAMS = params_paths


def protonate_pdb_with_pyrosetta(
    input_pdb: str | Path,
    out_pdb: str | Path,
    ligand_params: Optional[Iterable[str | Path]] = None,
    extra_init_flags: Optional[Iterable[str]] = None,
    variant_map: Optional[dict[tuple[str, int], str]] = None,
) -> None:
    """Load a (normalized) heavy-atom PDB into a PyRosetta pose and dump back.

    PyRosetta's pose construction places ideal hydrogens on every residue
    based on the residue type. To respect catalytic tautomers we apply
    Rosetta variant types AFTER pose construction using ``variant_map``.

    Args:
        input_pdb: PDB normalized to standard 3-char residue names (call
            ``normalize_pdb_for_pyrosetta`` first).
        out_pdb: Where to write the hydrated PDB.
        ligand_params: Iterable of ``.params`` files for the ligand.
        extra_init_flags: Extra strings to append to PyRosetta's init().
        variant_map: ``{(chain, resno) -> rosetta_variant_label}`` to
            re-apply after loading. Labels: ``HIS_D``, ``HIS_E`` (= HIE),
            ``HIP``, ``KCX``, ``ASH``, ``GLH``, ``LYN``, ``CYM``, ``CYX``,
            ``TYM``. Unknown labels are skipped with a warning.
    """
    input_pdb = str(input_pdb)
    out_pdb = str(out_pdb)
    Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)

    _ensure_pyrosetta_inited(ligand_params=ligand_params,
                              extra_init_flags=extra_init_flags)
    import pyrosetta  # type: ignore[import-not-found]

    pose = pyrosetta.pose_from_pdb(input_pdb)

    # Re-apply Rosetta variant types from the variant_map so tautomers
    # carry over from the (normalized) input. PyRosetta enum names:
    #   HIS_D       -> SIDECHAIN_CONJUGATION? no — the actual variant for
    #                  HIS_D is just selecting the "HIS_D" residue type
    #                  from the residue type set, not a variant_type per
    #                  se. Use mutate_residue / replace_residue.
    # The cleanest API: replace the residue with one of a different name
    # via core.pose.replace_pose_residue_copying_existing_coordinates.
    if variant_map:
        from pyrosetta.rosetta.core.pose import (  # type: ignore
            replace_pose_residue_copying_existing_coordinates,
        )
        rts = pose.residue_type_set_for_pose()
        for (chain, resno), variant_label in variant_map.items():
            try:
                pose_resno = pose.pdb_info().pdb2pose(chain, resno)
                if pose_resno == 0:
                    LOGGER.warning(
                        "variant remap: residue %s%d not found in pose",
                        chain, resno,
                    )
                    continue
                # Map our 5-char label -> Rosetta residue type name. For
                # canonical AAs Rosetta uses the same labels (HIS_D, KCX,
                # ASN_p ...) but mostly "HIS_D" works directly.
                target_name = variant_label  # e.g. "HIS_D"
                if not rts.has_name(target_name):
                    LOGGER.warning(
                        "variant remap: residue type %r unknown; "
                        "leaving %s%d as default tautomer",
                        target_name, chain, resno,
                    )
                    continue
                new_rt = rts.name_map(target_name)
                replace_pose_residue_copying_existing_coordinates(
                    pose, pose_resno, new_rt,
                )
            except Exception as exc:  # don't block whole pose dump
                LOGGER.warning(
                    "variant remap failed for %s%d (%s): %s",
                    chain, resno, variant_label, exc,
                )

    pose.dump_pdb(out_pdb)


def protonate_final_topk(
    topk_dir: str | Path,
    seed_pdb: str | Path,
    ligand_params: Iterable[str | Path],
    out_dir: Optional[str | Path] = None,
    ligand_resname: Optional[str] = None,
    keep_intermediate: bool = False,
    ptm_map: Optional[dict[tuple[str, int], str] | str] = None,
) -> dict:
    """End-to-end driver for the post-design protonation cleanup.

    For every ``*.pdb`` in ``topk_dir``:
        1. Run PyRosetta hydrate -> intermediate ``.rosetta.pdb`` (full H,
           Rosetta tautomer labels).
        2. Combine with seed REMARK 666 + ligand block via
           ``write_clean_final_pdb`` -> final ``*.protonated.pdb``.

    Args:
        topk_dir: Directory of MPNN-restored design PDBs (HIS_D-style labels,
            REMARK 666 already present).
        seed_pdb: Source for REMARK 666 + ligand HETATM block.
        ligand_params: Iterable of .params files for the ligand.
        out_dir: Where to write final PDBs. ``None`` = same as topk_dir.
        ligand_resname: Optional ligand 3-letter code; auto-detected if not
            given.
        keep_intermediate: If False (default), delete the .rosetta.pdb
            intermediates after writing the cleaned outputs.
        ptm_map: User-supplied PTM declarations. Either a string spec
            (e.g. ``"A:157=KCX,A:200=SEP"``) or a pre-parsed dict
            ``{(chain, resno) -> code}``. Use ``"-"`` as the value to
            FORCE no-PTM annotation for a residue (overrides auto-detect).
            ``None`` = auto-detect-only from seed.

    Returns:
        Stats dict aggregated across all PDBs processed.
    """
    if isinstance(ptm_map, str):
        ptm_map = parse_ptm_map(ptm_map)
    topk_dir = Path(topk_dir)
    out_dir = Path(out_dir) if out_dir is not None else topk_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pdbs = sorted(p for p in topk_dir.iterdir() if p.suffix == ".pdb"
                   and ".rosetta.pdb" not in p.name
                   and ".protonated.pdb" not in p.name)
    LOGGER.info("protonate_final_topk: %d PDBs in %s", len(pdbs), topk_dir)

    summary = {
        "pdbs_processed": 0,
        "pdbs_failed": 0,
        "atoms_protein_total": 0,
        "atoms_ligand_total": 0,
        "remark_666_total": 0,
        "remark_668_total": 0,
        "variants_normalized_total": 0,
    }
    failures: list[tuple[str, str]] = []

    for pdb in pdbs:
        stem = pdb.stem
        normalized = out_dir / f"{stem}.norm.pdb"
        rosetta_intermediate = out_dir / f"{stem}.rosetta.pdb"
        final_pdb = out_dir / f"{stem}.protonated.pdb"
        try:
            variant_map = normalize_pdb_for_pyrosetta(
                input_pdb=pdb,
                out_pdb=normalized,
            )
            protonate_pdb_with_pyrosetta(
                input_pdb=normalized,
                out_pdb=rosetta_intermediate,
                ligand_params=ligand_params,
                variant_map=variant_map,
            )
            stats = write_clean_final_pdb(
                rosetta_pdb=rosetta_intermediate,
                seed_pdb=seed_pdb,
                out_pdb=final_pdb,
                ligand_resname=ligand_resname,
                ptm_map=ptm_map,
            )
            stats["variants_remapped"] = len(variant_map)
        except Exception as exc:
            failures.append((pdb.name, str(exc)))
            summary["pdbs_failed"] += 1
            LOGGER.error("protonate_final_topk: %s failed: %s", pdb.name, exc)
            continue

        summary["pdbs_processed"] += 1
        summary["atoms_protein_total"] += stats["atoms_protein"]
        summary["atoms_ligand_total"] += stats["atoms_ligand"]
        summary["remark_666_total"] += stats["remark_666_in"]
        summary["remark_668_total"] += stats["remark_668_lines"]
        summary["variants_normalized_total"] += stats["variants_normalized"]

        if not keep_intermediate:
            for tmp in (normalized, rosetta_intermediate):
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass

    LOGGER.info(
        "protonate_final_topk: done. processed=%d failed=%d",
        summary["pdbs_processed"], summary["pdbs_failed"],
    )
    if failures:
        for name, err in failures[:5]:
            LOGGER.warning("  failure %s: %s", name, err)
    summary["failures"] = failures
    return summary
