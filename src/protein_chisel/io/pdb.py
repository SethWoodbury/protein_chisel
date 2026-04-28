"""PDB I/O — pure-text helpers (no PyRosetta dependency).

Heavy operations (load into a Pose, repack, score) live in `utils/pose`,
which does require PyRosetta. The routines here only require a PDB file as
text: they parse REMARK 666 catalytic-residue lines, write them back,
extract chain composition, find ligand residues, etc. — all without
needing to instantiate a pose.

REMARK 666 format (theozyme matcher output):

    REMARK 666 MATCH TEMPLATE B YYE  209 MATCH MOTIF A HIS  188  1  1

Tokens (whitespace-split):
    [0] REMARK
    [1] 666
    [2] MATCH
    [3] TEMPLATE
    [4] target_chain     (B)
    [5] target_name3     (YYE)
    [6] target_resno     (209)
    [7] MATCH
    [8] MOTIF
    [9] motif_chain      (A)
    [10] motif_name3     (HIS)
    [11] motif_resno     (188)
    [12] cst_no          (1)
    [13] cst_no_var      (1)

The catalytic residue is the MOTIF entry; the TEMPLATE entry tells you
what ligand atom group it pairs with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# REMARK 666 parsing
# ---------------------------------------------------------------------------


@dataclass
class CatalyticResidue:
    """One MOTIF residue from a REMARK 666 line."""

    chain: str           # motif (catalytic) chain
    name3: str           # motif residue name3
    resno: int           # motif residue number
    target_chain: str    # ligand chain
    target_name3: str    # ligand name3
    target_resno: int    # ligand residue number
    cst_no: int
    cst_no_var: int

    def to_remark_line(self) -> str:
        """Serialize back to a REMARK 666 line.

        PDB lines are 80 characters of content + a single newline (so 81
        bytes total on disk). We pad the content to exactly 80 columns,
        then append the newline.
        """
        body = (
            f"REMARK 666 MATCH TEMPLATE {self.target_chain} {self.target_name3:>3} "
            f"{self.target_resno:>4} MATCH MOTIF {self.chain} {self.name3:>3} "
            f"{self.resno:>4}  {self.cst_no}  {self.cst_no_var}"
        )
        # Truncate (shouldn't happen with sane inputs) or pad to 80.
        return f"{body[:80]:<80}\n"


def parse_remark_666(
    pdb_path: str | Path,
    key_by: str = "resno",
) -> dict:
    """Parse all REMARK 666 lines from a PDB.

    Args:
        key_by: how to key the returned dict.
            * ``"resno"`` (default, back-compat): ``{int_resno: CatalyticResidue}``.
                Collapses entries with the same residue number across chains
                or insertion codes.
            * ``"chain_resno"``: ``{(chain: str, resno: int): CatalyticResidue}``.
                Always unique; use this for multi-chain inputs.

    Tolerant: skips malformed lines (comment lines, truncated records).
    Stops scanning at the first ATOM/HETATM/MODEL/TER record.
    """
    out: dict = {}
    with open(pdb_path, "r") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM", "MODEL", "TER")):
                # Stop scanning headers; REMARK lines should all precede coordinates.
                break
            if not line.startswith("REMARK 666"):
                continue
            parts = line.split()
            # Need at least 14 tokens for a complete record.
            if len(parts) < 14 or parts[2] != "MATCH" or parts[3] != "TEMPLATE":
                continue
            try:
                catres = CatalyticResidue(
                    chain=parts[9],
                    name3=parts[10],
                    resno=int(parts[11]),
                    target_chain=parts[4],
                    target_name3=parts[5],
                    target_resno=int(parts[6]),
                    cst_no=int(parts[12]),
                    cst_no_var=int(parts[13]),
                )
            except (ValueError, IndexError):
                continue
            if key_by == "chain_resno":
                out[(catres.chain, catres.resno)] = catres
            else:
                out[catres.resno] = catres
    return out


def write_remark_666(
    src_pdb: str | Path,
    dst_pdb: str | Path,
    catres: dict[int, CatalyticResidue],
    drop_existing: bool = True,
) -> None:
    """Copy `src_pdb` to `dst_pdb`, replacing/inserting REMARK 666 lines.

    If `drop_existing` is True, removes any existing REMARK 666 lines from
    `src_pdb` first; else they are kept.
    """
    src = Path(src_pdb)
    dst = Path(dst_pdb)
    dst.parent.mkdir(parents=True, exist_ok=True)

    new_remarks = "".join(c.to_remark_line() for c in catres.values())

    with open(src, "r") as fh, open(dst, "w") as out:
        wrote_remarks = False
        for line in fh:
            if drop_existing and line.startswith("REMARK 666"):
                continue
            # Insert new remarks just before the first ATOM/HETATM/MODEL line.
            if not wrote_remarks and line.startswith(("ATOM", "HETATM", "MODEL")):
                out.write(new_remarks)
                wrote_remarks = True
            out.write(line)
        if not wrote_remarks:
            out.write(new_remarks)


# ---------------------------------------------------------------------------
# Catalytic-residue spec parser (fallback when REMARK 666 is absent)
# ---------------------------------------------------------------------------


@dataclass
class ResidueRef:
    """A `(chain, resno)` pair without ligand context."""

    chain: str
    resno: int

    def __str__(self) -> str:
        return f"{self.chain}{self.resno}"


_RE_RESREF = re.compile(r"^([A-Za-z])(\d+)$")
_RE_RESRANGE = re.compile(r"^([A-Za-z])(\d+)-(\d+)$")


def parse_catres_spec(items: list[str]) -> list[ResidueRef]:
    """Parse user-supplied catalytic-residue identifiers.

    Each item is one of:
        ``A94``         single residue
        ``A94-96``      inclusive range (chain A residues 94, 95, 96)
        ``B101``        residue on chain B

    Adapted from process_diffusion3.parse_ref_catres.
    """
    out: list[ResidueRef] = []
    for raw in items:
        s = raw.strip()
        if not s:
            continue
        m = _RE_RESRANGE.match(s)
        if m:
            ch = m.group(1)
            start = int(m.group(2))
            end = int(m.group(3))
            if end < start:
                raise ValueError(f"range end < start: {raw!r}")
            for r in range(start, end + 1):
                out.append(ResidueRef(chain=ch, resno=r))
            continue
        m = _RE_RESREF.match(s)
        if m:
            out.append(ResidueRef(chain=m.group(1), resno=int(m.group(2))))
            continue
        raise ValueError(f"unrecognized catres spec: {raw!r}")
    return out


# ---------------------------------------------------------------------------
# Light-touch ATOM/HETATM scanning (no PyRosetta)
# ---------------------------------------------------------------------------


@dataclass
class AtomRecord:
    """Minimal ATOM/HETATM record parsed from the PDB columns."""

    record: str        # "ATOM" or "HETATM"
    serial: int
    name: str          # column 13-16
    alt_loc: str       # column 17
    res_name: str      # column 18-20 (3 chars, may include 4th col for KCX-like)
    chain: str         # column 22
    res_seq: int       # column 23-26
    i_code: str        # column 27
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    element: str       # column 77-78


def parse_atom_record(line: str) -> Optional[AtomRecord]:
    """Parse one ATOM/HETATM line. Returns None for non-ATOM lines.

    Uses fixed PDB columns (PDB format spec), tolerant of short lines.
    """
    if not line.startswith(("ATOM  ", "HETATM")):
        return None
    try:
        return AtomRecord(
            record=line[0:6].strip(),
            serial=int(line[6:11].strip()),
            name=line[12:16].strip(),
            alt_loc=line[16:17].strip(),
            res_name=line[17:20].strip(),
            chain=line[21:22].strip() or " ",
            res_seq=int(line[22:26].strip()),
            i_code=line[26:27].strip(),
            x=float(line[30:38].strip()),
            y=float(line[38:46].strip()),
            z=float(line[46:54].strip()),
            occupancy=float(line[54:60].strip() or "0"),
            b_factor=float(line[60:66].strip() or "0"),
            element=line[76:78].strip() if len(line) >= 78 else "",
        )
    except ValueError:
        return None


@dataclass
class PdbSummary:
    """Quick summary of a PDB without instantiating a pose.

    Useful for sanity checks and for routing logic (e.g. is this PDB apo?).
    """

    n_atom: int = 0
    n_hetatm: int = 0
    chains: set[str] = field(default_factory=set)
    protein_chains: set[str] = field(default_factory=set)
    ligand_residues: list[tuple[str, str, int]] = field(default_factory=list)
    # ^ list of (chain, name3, resno); excludes water
    has_water: bool = False
    elements: set[str] = field(default_factory=set)


# Standard amino acid 3-letter codes (canonical 20). Anything else in an
# ATOM record (but not HETATM) is unusual; anything else in HETATM is
# either water or a ligand.
_CANONICAL_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def summarize_pdb(pdb_path: str | Path) -> PdbSummary:
    """Cheap one-pass summary of a PDB file."""
    summary = PdbSummary()
    seen_ligands: set[tuple[str, str, int]] = set()
    with open(pdb_path, "r") as fh:
        for line in fh:
            atom = parse_atom_record(line)
            if atom is None:
                continue
            if atom.record == "ATOM":
                summary.n_atom += 1
                summary.protein_chains.add(atom.chain)
            else:  # HETATM
                summary.n_hetatm += 1
                if atom.res_name == "HOH":
                    summary.has_water = True
                    continue
                key = (atom.chain, atom.res_name, atom.res_seq)
                if key not in seen_ligands:
                    seen_ligands.add(key)
                    summary.ligand_residues.append(key)
            summary.chains.add(atom.chain)
            if atom.element:
                summary.elements.add(atom.element)
    return summary


def find_ligand(
    pdb_path: str | Path, exclude: tuple[str, ...] = ("HOH",)
) -> Optional[tuple[str, str, int]]:
    """Return (chain, name3, resno) of the *first* ligand HETATM residue.

    Returns None if no ligand HETATMs are present (apo structure).
    """
    seen: set[tuple[str, str, int]] = set()
    with open(pdb_path, "r") as fh:
        for line in fh:
            atom = parse_atom_record(line)
            if atom is None or atom.record != "HETATM":
                continue
            if atom.res_name in exclude:
                continue
            key = (atom.chain, atom.res_name, atom.res_seq)
            if key not in seen:
                seen.add(key)
                return key
    return None


def is_apo(pdb_path: str | Path, exclude: tuple[str, ...] = ("HOH",)) -> bool:
    """True if the PDB has no non-water HETATM records."""
    return find_ligand(pdb_path, exclude=exclude) is None


def extract_sequence(pdb_path: str | Path, chain: Optional[str] = None) -> str:
    """Read the AA sequence from ATOM records.

    Returns 1-letter codes; unknown residues become 'X'. Skips alt locs
    other than blank/A. If `chain` is None, uses the first protein chain.
    """
    aa3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M", "SEC": "U", "PYL": "O",
    }
    out_chars: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    target_chain = chain
    with open(pdb_path, "r") as fh:
        for line in fh:
            atom = parse_atom_record(line)
            if atom is None or atom.record != "ATOM":
                continue
            if atom.alt_loc not in ("", "A"):
                continue
            if target_chain is None:
                target_chain = atom.chain
            if atom.chain != target_chain:
                continue
            key = (atom.chain, atom.res_seq, atom.i_code)
            if key in seen:
                continue
            seen.add(key)
            out_chars.append(aa3to1.get(atom.res_name, "X"))
    return "".join(out_chars)
