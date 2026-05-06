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

REMARK code split (so REMARK 666/668 are PURE DATA, grep-friendly):

    REMARK 665   = column-key + format hint for REMARK 666 lines
    REMARK 666   = data only (Rosetta enzyme-matcher anchor lines)
    REMARK 667   = column-key + STATE/PTM legend for REMARK 668 lines
    REMARK 668   = data only (one row per catalytic residue)

Example output:

    REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors.
    REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI> MATCH MOTIF <mCH mRESN mRESI IDX VAR>
    REMARK 666 MATCH TEMPLATE B YYE  211 MATCH MOTIF A HIS   97  1  1
    REMARK 666 MATCH TEMPLATE B YYE  211 MATCH MOTIF A LYS   19  3  1
    ...
    REMARK 667 REMARK 668 = catres protonation/PTM (paired by IDX with REMARK 666).
    REMARK 667 STATE: HID/HIE/HIP (HIS); ASP/ASH; GLU/GLH; LYS/LYN/KCX; CYS/CYM/CYX; TYR/TYM.
    REMARK 667 PTM = wwPDB CCD code (KCX,SEP,TPO,...) or '-' if none; ANNOTATION-ONLY.
    REMARK 667 fmt: REMARK 668 <IDX CH RESN RESI STATE PTM H_ATOMS ROSETTA_PATCH>
    REMARK 668   1   A HIS     97   HID -   HD1                     HIS
    REMARK 668   3   A LYS     19   LYS KCX -                       LYS
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
    """Extract every REMARK 666 motif entry from a PDB.

    Tries strict regex first; falls back to whitespace-split if the regex
    doesn't match (some pipeline tools emit subtly different REMARK 666
    formatting that the strict pattern misses). Same field semantics as
    the established lab utility ``get_matcher_residues`` in
    ``/net/software/lab/scripts/enzyme_design/SETH_TEMP_UTILS/``: tokens
    after "REMARK 666 MATCH TEMPLATE" are
        [tpl_chain, tpl_resname, tpl_resno,
         "MATCH", "MOTIF",
         motif_chain, motif_resname, motif_resno,
         motif_index, matched_int]
    """
    entries: list[Remark666Entry] = []
    with open(pdb_path) as fh:
        for line in fh:
            if "ATOM" in line[:6]:  # stop at coordinate block
                break
            if not line.startswith("REMARK 666"):
                continue
            stripped = line.rstrip()
            m = _RE_REMARK_666.match(stripped)
            if m:
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
                continue
            # Whitespace-split fallback. Mirrors get_matcher_residues
            # exactly (tokens 4..13).
            try:
                lspl = stripped.split()
                entries.append(
                    Remark666Entry(
                        raw=line if line.endswith("\n") else line + "\n",
                        template_chain=lspl[4],
                        template_resname=lspl[5],
                        template_resno=int(lspl[6]),
                        motif_chain=lspl[9],
                        motif_resname=lspl[10],
                        motif_resno=int(lspl[11]),
                        motif_index=int(lspl[12]),
                        matched=int(lspl[13]) if len(lspl) > 13 else 1,
                    )
                )
            except (IndexError, ValueError) as exc:
                LOGGER.warning(
                    "parse_remark_666: could not parse line %r (%s); "
                    "skipping", stripped, exc,
                )
    return entries


def get_matcher_residues(filename: str | Path) -> dict[int, dict]:
    """Drop-in compatible with the lab's get_matcher_residues helper.

    Returns ``{motif_resno: {target_name, target_chain, target_resno,
    chain, name3, cst_no, cst_no_var}}``. Field names match the
    ``/net/software/lab/scripts/enzyme_design/SETH_TEMP_UTILS/
    process_diffusion3_outputs__REORG.py`` convention so existing
    downstream tooling that uses that helper can swap modules.

    Internally we go through :func:`parse_remark_666` so the same robust
    regex+whitespace fallback applies.
    """
    matches: dict[int, dict] = {}
    for entry in parse_remark_666(filename):
        matches[entry.motif_resno] = {
            "target_name": entry.template_resname,
            "target_chain": entry.template_chain,
            "target_resno": entry.template_resno,
            "chain": entry.motif_chain,
            "name3": entry.motif_resname,
            "cst_no": entry.motif_index,
            "cst_no_var": entry.matched,
        }
    LOGGER.debug("get_matcher_residues: %d entries from %s",
                  len(matches), filename)
    return matches


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


# Aesthetic / column-header lines for REMARK 666 + REMARK 668 are emitted
# under different REMARK codes (665 and 667 respectively) so that REMARK 666
# and REMARK 668 lines are PURELY DATA and can be `grep`'d without false
# hits on the documentation block.
#
# REMARK 665: column key for REMARK 666 (Rosetta enzyme-matcher motifs).
# REMARK 667: column key + protonation-code legend for REMARK 668.
#
# Lines are kept short (<= 80 PDB columns) and dense.
_REMARK_665_HEADER: tuple[str, ...] = (
    "REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors.\n",
    "REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI> MATCH MOTIF <mCH mRESN mRESI IDX VAR>\n",
)

_REMARK_667_HEADER: tuple[str, ...] = (
    "REMARK 667 REMARK 668 = catres protonation/PTM (paired by IDX with REMARK 666).\n",
    "REMARK 667 STATE: HID/HIE/HIP (HIS); ASP/ASH; GLU/GLH; LYS/LYN/KCX; CYS/CYM/CYX; TYR/TYM.\n",
    "REMARK 667 PTM = wwPDB CCD code (KCX,SEP,TPO,PTR,MLY,M3L,ALY,HYP,...) or '-' if none;\n",
    "REMARK 667 PTM is ANNOTATION-ONLY -- RESN keeps unmodified form; ROSETTA_PATCH = pose variant.\n",
    "REMARK 667 fmt: REMARK 668 <IDX CH RESN RESI STATE PTM H_ATOMS ROSETTA_PATCH>\n",
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


@dataclasses.dataclass
class PtmSpec:
    """One parsed PTM declaration.

    Two addressing modes (exactly one set):

        * ``resno`` — explicit residue number on ``chain``. Format
          ``"A:157=KCX"``. Use when the residue is known by sequence
          position.

        * ``motif_idx`` — REMARK 666 motif index. Format
          ``"A/LYS/3:KCX"``. Resolved against the seed PDB's REMARK 666
          entries: the motif with trailing index 3 (5th column from the
          end of the MATCH MOTIF line) on chain A is the catalytic
          residue this declaration applies to. Use this for design
          campaigns where the catalytic residue has a stable motif
          index but its sequence position varies between scaffolds.
          ``expected_resname`` is a sanity-check hint (e.g. "LYS") —
          a warning is logged if the seed's motif residue at that
          index is NOT this residue.

    PTM declarations are ANNOTATION ONLY: they appear in REMARK 668 of
    the output PDB but do NOT modify the residue's coordinates,
    resname (still LYS, not KCX), Rosetta variant type, sequence
    reading, or protonation. The PTM column in REMARK 668 tells
    downstream consumers (docking, MD setup) what modification the
    design intent calls for.
    """
    chain: str
    code: str
    resno: int | None = None
    motif_idx: int | None = None
    expected_resname: str | None = None


def parse_ptm_map(
    spec: str | Iterable[str] | None,
) -> list[PtmSpec]:
    """Parse a CLI/API PTM spec string (or iterable of strings) into specs.

    Accepted entry formats (mix-and-match in a comma- or whitespace-
    separated list, or pass a Python list of single-spec strings):

        * Motif-index form (preferred for catalytic residues):
            ``"A/LYS/3:KCX"``
              chain ``A``, expected_resname ``LYS``, REMARK 666 motif
              index 3, PTM code ``KCX``.

        * Explicit-residue form (use for non-motif residues):
            ``"A:157=KCX"``     chain:resno=code
            ``"A157=KCX"``      chain<resno>=code (no colon)
            ``"157=KCX"``       default chain ('A')

    Use ``"-"`` as the code to FORCE no-PTM annotation at that residue
    (overrides any auto-detect from the seed atom inventory).

    Returns:
        List of :class:`PtmSpec`. Empty list if ``spec`` is empty/None.
    """
    out: list[PtmSpec] = []
    if not spec:
        return out
    if isinstance(spec, str):
        # Allow comma, semicolon, or whitespace separators.
        parts = [p for p in re.split(r"[,;\s]+", spec.strip()) if p]
    else:
        parts = [str(p).strip() for p in spec if str(p).strip()]

    for part in parts:
        # Motif-index form: "A/LYS/3:KCX"
        m = re.match(r"^([A-Za-z])\s*/\s*([A-Za-z0-9]{1,5})\s*/\s*(\d+)\s*:\s*(\S+)$", part)
        if m:
            chain = m.group(1)
            expected_resname = m.group(2).upper()
            motif_idx = int(m.group(3))
            code = m.group(4).strip().upper()
            if code not in PTMS and code != "-":
                LOGGER.warning(
                    "parse_ptm_map: PTM code %r at motif index %d (chain %s, "
                    "expected resname %s) is not in PTMS registry; emitting "
                    "the annotation but downstream tools may not recognize it",
                    code, motif_idx, chain, expected_resname,
                )
            out.append(PtmSpec(
                chain=chain, code=code,
                motif_idx=motif_idx, expected_resname=expected_resname,
            ))
            continue

        # Explicit-residue form: "A:157=KCX" / "A157=KCX" / "157=KCX"
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
                "parse_ptm_map: PTM code %r at %s%d is not in PTMS registry; "
                "emitting the annotation but downstream tools may not recognize",
                code, chain, resno,
            )
        out.append(PtmSpec(chain=chain, code=code, resno=resno))
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


def _resolve_motif_specs_to_keys(
    seed_pdb: str | Path,
    specs: Iterable[PtmSpec],
) -> list[tuple[tuple[str, int], str]]:
    """Resolve PtmSpec entries to ``[((chain, resno), code), ...]`` pairs.

    For ``motif_idx`` specs, look up the corresponding REMARK 666 entry
    in ``seed_pdb`` and use its ``motif_chain`` and ``motif_resno``.
    Validates that the chain matches and warns if the expected resname
    doesn't match the seed's recorded motif residue.

    Specs that fail to resolve (no matching motif index, etc.) are
    skipped with a warning.
    """
    motif_entries = parse_remark_666(seed_pdb)
    by_idx: dict[int, Remark666Entry] = {e.motif_idx if False else e.motif_index: e for e in motif_entries}
    out: list[tuple[tuple[str, int], str]] = []
    for spec in specs:
        if spec.resno is not None:
            out.append(((spec.chain, spec.resno), spec.code))
            continue
        if spec.motif_idx is None:
            LOGGER.warning("PtmSpec has neither resno nor motif_idx: %r", spec)
            continue
        entry = by_idx.get(spec.motif_idx)
        if entry is None:
            LOGGER.warning(
                "PtmSpec %r references REMARK 666 motif index %d which is "
                "not present in seed PDB; skipping. Available indices: %s",
                spec, spec.motif_idx, sorted(by_idx.keys()),
            )
            continue
        if entry.motif_chain != spec.chain:
            LOGGER.warning(
                "PtmSpec %r expected chain %r at motif index %d but seed "
                "has chain %r; using the seed's chain (%r) anyway",
                spec, spec.chain, spec.motif_idx,
                entry.motif_chain, entry.motif_chain,
            )
        if (spec.expected_resname is not None
                and entry.motif_resname != spec.expected_resname):
            LOGGER.warning(
                "PtmSpec %r expected resname %r at chain %s motif index %d "
                "but seed has %r; the PTM annotation will still be applied "
                "(no residue handling change)",
                spec, spec.expected_resname, entry.motif_chain,
                spec.motif_idx, entry.motif_resname,
            )
        out.append(((entry.motif_chain, entry.motif_resno), spec.code))
    return out


def resolve_ptm_map(
    seed_pdb: str | Path,
    explicit_ptm: Optional[
        dict[tuple[str, int], str] | Iterable[PtmSpec] | str
    ] = None,
) -> dict[tuple[str, int], str]:
    """Resolve the final PTM map for a design from all available sources.

    Trust order (later wins):
        1. Auto-detect from the seed PDB's atom inventory + resnames.
        2. Explicit user/CLI declarations from ``explicit_ptm``. Use
           ``"-"`` as the code to FORCE no-PTM at that residue
           (overrides auto-detect).

    Args:
        seed_pdb: Reference PDB to scan for PTM markers and (when
            spec uses motif-index form) REMARK 666 entries.
        explicit_ptm: User-supplied overrides. Accepts:
            * a string CLI spec (e.g. ``"A/LYS/3:KCX,A:200=SEP"``)
            * a list of :class:`PtmSpec`
            * a pre-resolved ``{(chain, resno) -> code}`` dict

    Returns:
        Final ``{(chain, resno) -> ptm_code}`` map (with "-" entries
        removed).
    """
    inventory = _collect_residue_atom_inventory(seed_pdb)
    final: dict[tuple[str, int], str] = dict(detect_ptms_from_inventory(inventory))

    if explicit_ptm is None:
        return final

    # Coerce input to list[PtmSpec]
    specs: list[PtmSpec]
    if isinstance(explicit_ptm, str):
        specs = parse_ptm_map(explicit_ptm)
    elif isinstance(explicit_ptm, dict):
        specs = [
            PtmSpec(chain=chain, code=code, resno=resno)
            for (chain, resno), code in explicit_ptm.items()
        ]
    else:
        specs = list(explicit_ptm)

    for key, code in _resolve_motif_specs_to_keys(seed_pdb, specs):
        if code == "-":
            final.pop(key, None)
        else:
            final[key] = code
    return final


def build_remark_668_block(
    rosetta_pdb: str | Path,
    seed_pdb: str | Path,
    ptm_map: Optional[
        dict[tuple[str, int], str] | Iterable[PtmSpec] | str
    ] = None,
) -> list[str]:
    """Build the full REMARK 668 block for a hydrated design PDB.

    Reads REMARK 666 entries from ``seed_pdb`` and per-residue protonation
    info from ``rosetta_pdb`` (the freshly hydrated, Rosetta-output PDB).

    Args:
        rosetta_pdb: PyRosetta-hydrated dump (source of STATE / H_ATOMS).
        seed_pdb: Original seed PDB (source of REMARK 666 + auto-detected PTMs).
        ptm_map: User-supplied PTM declarations. Accepts:
            * a string spec (e.g. ``"A/LYS/3:KCX,A:200=SEP"``)
            * a list of :class:`PtmSpec`
            * a pre-resolved ``{(chain, resno) -> code}`` dict
            Use ``"-"`` as the code to FORCE no-PTM annotation
            (overrides auto-detect from seed atoms). ``None`` defaults
            to auto-detect-only from seed atom inventory.
    """
    motif_entries = parse_remark_666(seed_pdb)
    if not motif_entries:
        return []

    inventory = _collect_residue_atom_inventory(rosetta_pdb)
    resolved_ptm_map = resolve_ptm_map(seed_pdb, ptm_map)

    # REMARK 667 is the documentation/column-header block for REMARK 668.
    # REMARK 668 lines are PURE DATA (one per catalytic residue).
    lines: list[str] = list(_REMARK_667_HEADER)
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
    return lines


# ----------------------------------------------------------------------------
# Final-PDB writer (combines hydrated coords + REMARK header + ligand)
# ----------------------------------------------------------------------------


def _extract_seed_header_lines(seed_pdb: str | Path) -> list[str]:
    """Header block to flow downstream: REMARK 665 (column key for 666),
    REMARK 666 (verbatim from seed), HETNAM, LINK.
    """
    keep_prefixes = ("REMARK 666", "HETNAM", "LINK")
    seed_lines: list[str] = []
    with open(seed_pdb) as fh:
        for line in fh:
            if _is_atom_line(line):
                break
            for p in keep_prefixes:
                if line.startswith(p):
                    seed_lines.append(line if line.endswith("\n") else line + "\n")
                    break

    # Inject REMARK 665 column-key header BEFORE the first REMARK 666
    # (and only if REMARK 666 lines were found; HETNAM/LINK don't need it).
    out: list[str] = []
    inserted_665 = False
    for line in seed_lines:
        if not inserted_665 and line.startswith("REMARK 666"):
            out.extend(_REMARK_665_HEADER)
            inserted_665 = True
        out.append(line)
    # If no REMARK 666 was present, return whatever we had (HETNAM/LINK only)
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
    ptm_map: Optional[
        dict[tuple[str, int], str] | Iterable[PtmSpec] | str
    ] = None,
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
    ptm_map: Optional[
        dict[tuple[str, int], str] | Iterable[PtmSpec] | str
    ] = None,
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
    # Normalize ptm_map to a list[PtmSpec] for consistent downstream
    # handling. Strings get parsed; pre-built dicts stay as dicts (still
    # accepted by resolve_ptm_map).
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


# ----------------------------------------------------------------------------
# Shipping-layout reorganization
# ----------------------------------------------------------------------------


def reorganize_for_shipping(
    run_dir: str | Path,
    *,
    strip_intermediates: bool = True,
    pdb_subdir_name: str = "designs",
    minimal: bool = False,
) -> dict:
    """Build a tidy "ready-to-ship" layout in ``run_dir``.

    Production runs accumulate a heavy file tree:

        run_dir/
        ├── _seed_fpocket_workspace/
        ├── cycle_00/  cycle_01/  cycle_02/
        ├── final_topk/
        │   ├── topk.tsv  topk.fasta  all_survivors.tsv
        │   ├── topk_pdbs/                  (heavy-atom restored)
        │   └── topk_pdbs_protonated/       (full-H, REMARK 668)
        ├── fusion_runtime/
        ├── manifest.json
        ├── seed_tunnel_residues.tsv
        └── cycle_metrics.tsv

    The shipping layout collapses this to:

        run_dir/
        ├── designs/
        │   └── <id>.pdb         (renamed from <id>.protonated.pdb;
        │                          full-H, REMARK 665+666+667+668 + ligand)
        ├── designs.tsv          (all top-K rows + new pdb_path column)
        ├── designs.fasta        (one entry per design)
        ├── cycle_metrics.tsv    (per-cycle quality dynamics)
        └── manifest.json        (run config + counts + outputs map)

    With ``strip_intermediates=False``, the heavy subtrees are KEPT alongside
    the clean files (useful for deep diagnostics).

    Args:
        run_dir: Top-level run directory created by iterative_design_v2.
        strip_intermediates: If True (default), remove cycle_NN/, the
            unprotonated final_topk/topk_pdbs/, the seed fpocket workspace,
            fusion_runtime/, seed_tunnel_residues.tsv, and the dual final_topk
            wrapper. The final_topk/topk_pdbs_protonated/ contents become
            the new run_dir/designs/.
        pdb_subdir_name: Name of the per-design PDB directory.

    Returns:
        Stats dict with counts of files moved / removed.
    """
    import json as _json
    import shutil as _shutil

    run_dir = Path(run_dir)
    stats = {
        "designs_moved": 0,
        "subdirs_removed": [],
        "files_removed": [],
        "designs_tsv_rows": 0,
    }

    final_dir = run_dir / "final_topk"
    proto_dir = final_dir / "topk_pdbs_protonated"
    raw_dir = final_dir / "topk_pdbs"
    topk_tsv = final_dir / "topk.tsv"
    topk_fasta = final_dir / "topk.fasta"

    if not proto_dir.is_dir():
        LOGGER.warning("reorganize_for_shipping: %s missing; nothing to do",
                        proto_dir)
        return stats

    # 1. Move/rename protonated PDBs.
    # In MINIMAL mode, PDBs go directly under run_dir/ (flat layout).
    # In standard mode, they go under run_dir/<pdb_subdir_name>/.
    # ALSO: rename the LigandMPNN-internal "_lmpnn_<NNN>" suffix to
    # "_chisel_<NNN>" so the brand reflects this codebase (protein_chisel)
    # rather than the underlying sampler.
    if minimal:
        designs_dir = run_dir
    else:
        designs_dir = run_dir / pdb_subdir_name
        designs_dir.mkdir(exist_ok=True)

    def _rename_id(s: str) -> str:
        """Rename '_lmpnn_NNN' -> '_chisel_NNN' in a string. Idempotent."""
        return s.replace("_lmpnn_", "_chisel_")

    pdb_id_to_path: dict[str, Path] = {}      # NEW (chisel) id -> path
    pdb_id_old_to_new: dict[str, str] = {}    # OLD lmpnn id -> NEW chisel id
    for src in proto_dir.iterdir():
        if not (src.is_file() and src.name.endswith(".protonated.pdb")):
            continue
        # strip .protonated suffix; e.g. "FOO_lmpnn_004.protonated.pdb"
        # -> "FOO_lmpnn_004.pdb" -> rename to "FOO_chisel_004.pdb"
        new_name = src.name[: -len(".protonated.pdb")] + ".pdb"
        new_name_renamed = _rename_id(new_name)
        dst = designs_dir / new_name_renamed
        _shutil.move(str(src), str(dst))
        old_stem = new_name[: -len(".pdb")]   # FOO_lmpnn_004
        new_stem = dst.stem                    # FOO_chisel_004
        pdb_id_to_path[new_stem] = dst
        pdb_id_old_to_new[old_stem] = new_stem
        stats["designs_moved"] += 1

    # 2. Build the metrics TSV.
    # Filename is 'chiseled_design_metrics.tsv' (verbose-but-clear) — easy
    # to glob across many runs in jupyterhub. The legacy 'designs.tsv' name
    # is no longer written.
    metrics_filename = "chiseled_design_metrics.tsv"
    if topk_tsv.is_file():
        try:
            import pandas as _pd
            df = _pd.read_csv(topk_tsv, sep="\t")
            # Rename the id column from "_lmpnn_NNN" to "_chisel_NNN" so it
            # matches the on-disk PDB names.
            if "id" in df.columns:
                df["id"] = df["id"].astype(str).map(_rename_id)
                df["pdb_path"] = df["id"].map(
                    lambda i: str(pdb_id_to_path[i]) if i in pdb_id_to_path else ""
                )
            # Add a couple of run-identifier columns that make multi-run
            # concat in a notebook trivial: filter / groupby on these.
            try:
                with open(run_dir / "manifest.json") as _f:
                    _m = _json.load(_f)
                seed = _m.get("seed_pdb", "")
                df["seed_pdb"] = seed
                df["seed_basename"] = Path(seed).stem if seed else ""
                df["run_dir"] = str(run_dir)
            except Exception:
                pass
            df.to_csv(run_dir / metrics_filename, sep="\t", index=False)
            stats["designs_tsv_rows"] = len(df)
        except Exception as exc:
            LOGGER.warning("reorganize_for_shipping: could not write "
                            "%s (%s); falling back to copying topk.tsv",
                            metrics_filename, exc)
            _shutil.copy2(topk_tsv, run_dir / metrics_filename)

    # 3. Promote topk.fasta — but RENAME ids inside it too, and only in
    # standard layout (minimal layout drops fasta entirely below).
    if topk_fasta.is_file() and not minimal:
        try:
            with open(topk_fasta) as _src, open(run_dir / "designs.fasta", "w") as _dst:
                for ln in _src:
                    if ln.startswith(">"):
                        _dst.write(_rename_id(ln))
                    else:
                        _dst.write(ln)
        except Exception:
            _shutil.copy2(topk_fasta, run_dir / "designs.fasta")

    # 4. Remove heavy intermediates (default for shipping layout)
    if strip_intermediates:
        # cycle_NN/ heavy subdirs
        for sub in run_dir.iterdir():
            if sub.is_dir() and sub.name.startswith("cycle_"):
                _shutil.rmtree(sub, ignore_errors=True)
                stats["subdirs_removed"].append(sub.name)
        # other transient dirs
        for transient in (
            "_seed_fpocket_workspace", "fusion_runtime",
        ):
            t = run_dir / transient
            if t.exists():
                _shutil.rmtree(t, ignore_errors=True)
                stats["subdirs_removed"].append(transient)
        # transient files
        for fname in ("seed_tunnel_residues.tsv",):
            f = run_dir / fname
            if f.exists():
                f.unlink()
                stats["files_removed"].append(fname)
        # entire final_topk/ subtree (we promoted what we needed)
        if final_dir.is_dir():
            _shutil.rmtree(final_dir, ignore_errors=True)
            stats["subdirs_removed"].append("final_topk")

    # 5. Update manifest.json with new output paths
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            with open(manifest_path) as f:
                m = _json.load(f)
            m.setdefault("outputs", {})
            m["outputs"]["designs_dir"] = str(designs_dir)
            m["outputs"]["designs_tsv"] = str(run_dir / "chiseled_design_metrics.tsv")
            m["outputs"]["designs_fasta"] = str(run_dir / "designs.fasta")
            m["shipping_layout"] = True
            m["intermediates_stripped"] = bool(strip_intermediates)
            with open(manifest_path, "w") as f:
                _json.dump(m, f, indent=2)
        except Exception as exc:
            LOGGER.warning("reorganize_for_shipping: manifest update failed: %s",
                            exc)

    # 6. Minimal layout: collapse to a SINGLE FLAT directory containing
    # ONLY the PDBs and chiseled_design_metrics.tsv (with embedded
    # RUN_META JSON header on the first line). Drops every aux file.
    # PDBs are already in run_dir/ at this point because we used
    # designs_dir = run_dir for minimal mode (step 1).
    if minimal:
        # Collect all meta JSON contents into a single dict
        meta_payload: dict = {}
        for fname in (
            "manifest.json", "cycle_metrics.json",
            "throat_blocker_telemetry.json", "protonation_summary.json",
        ):
            p = run_dir / fname
            if p.is_file():
                try:
                    meta_payload[p.stem] = _json.loads(p.read_text())
                except Exception as exc:
                    LOGGER.warning("minimal_layout: could not read %s: %s",
                                    p.name, exc)

        # Prepend meta as a `# RUN_META: <single-line-json>` comment at
        # the top of chiseled_design_metrics.tsv (pandas-friendly with
        # comment='#'). One line keeps it grep- and column-tool-friendly.
        metrics_tsv = run_dir / metrics_filename
        if metrics_tsv.is_file() and meta_payload:
            try:
                meta_blob = _json.dumps(meta_payload, default=str,
                                         separators=(",", ":"))
                body = metrics_tsv.read_text()
                with open(metrics_tsv, "w") as f:
                    f.write(f"# RUN_META: {meta_blob}\n")
                    f.write(body)
                LOGGER.info(
                    "minimal_layout: prepended %d-byte RUN_META JSON "
                    "header to %s (read with "
                    "pd.read_csv(path, sep='\\t', comment='#'))",
                    len(meta_blob), metrics_filename,
                )
            except Exception as exc:
                LOGGER.warning("minimal_layout: could not write meta header: %s",
                                exc)

        # Strip every aux file. Reproducibility is preserved via the
        # embedded RUN_META.
        for fname in (
            "manifest.json", "cycle_metrics.tsv", "cycle_metrics.json",
            "throat_blocker_telemetry.json", "protonation_summary.json",
            "designs.fasta",
        ):
            p = run_dir / fname
            if p.is_file():
                try:
                    p.unlink()
                    stats["files_removed"].append(fname)
                except Exception:
                    pass
        LOGGER.info(
            "minimal_layout: final tree = flat run_dir/ with N PDBs + "
            "chiseled_design_metrics.tsv (RUN_META embedded as "
            "first-line comment); no subdirectories, no aux JSON files"
        )

    LOGGER.info(
        "reorganize_for_shipping: %d designs -> %s; designs.tsv rows=%d; "
        "stripped %d subdirs (%s) %d files; minimal=%s",
        stats["designs_moved"], designs_dir, stats["designs_tsv_rows"],
        len(stats["subdirs_removed"]), ",".join(stats["subdirs_removed"]),
        len(stats["files_removed"]), minimal,
    )
    return stats
