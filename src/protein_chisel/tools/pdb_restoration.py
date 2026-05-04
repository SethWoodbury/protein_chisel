"""Restore protonation, REMARK 666, HETNAM, LINK, and (optionally) hydrogens
on heavy-atom-only PDBs produced by LigandMPNN's side-chain packer.

This is the inner-loop equivalent of the upgraded_fastMPNNdesign
``pdb_restoration`` module, but stripped to pure stdlib (+ optional numpy
elsewhere in the package). It runs INSIDE the design loop (no PyRosetta,
no Bio dependency) so it can sit on the hot path between LigandMPNN
sampling and downstream filters.

What MPNN drops that we restore:
    1. REMARK 666 / HETNAM / LINK / REMARK PDBinfo-LABEL header lines.
       These are pipeline metadata: catalytic motifs, ligand 3-letter
       names, and explicit covalent bonds. Any tool that reads them
       (Rosetta, theozyme satisfaction, AF3 templating) needs them
       carried through unchanged.
    2. Rosetta-style 5-character residue protonation labels:
            HIS_D = HIS protonated on ND1  (delta tautomer)
            HIE   = HIS protonated on NE2  (epsilon tautomer)
            HIP   = HIS doubly protonated  (cation form)
       MPNN always emits "HIS"; without this restoration step every HIS
       in the design reads as the default tautomer.
    3. Special non-canonical labels (KCX = carbamylated lysine).
       Detected by presence of the carbamate atoms (CX/OQ1/OQ2/NZ).
    4. Optionally: heavy-atom catalytic hydrogens from the input PDB
       (HD1/HE2 on HIS, HZ on KCX-bearing LYS, etc.). MPNN emits no H,
       so if you want explicit protons on the catalytic residues without
       running Rosetta first, this puts them back from the seed.

Functions
---------
extract_remark_lines(input_pdb)
    Pull REMARK 666 / HETNAM / LINK / REMARK PDBinfo-LABEL header lines.
build_his_tautomer_map(input_pdb)
    {(chain, resno) -> "HIS"|"HIS_D"|"HIE"|"HIP"} from input H atoms.
detect_kcx_residues(input_pdb)
    {(chain, resno) -> True} where the input has KCX carbamate atoms.
restore_pdb_features(mpnn_pdb, ref_pdb, out_pdb, ...)
    Write a PDB equal to mpnn_pdb but with REMARK header restored,
    HIS residues relabeled per ref_pdb's tautomer state, KCX residues
    relabeled (and OQ1/OQ2/CX inserted), and optional catalytic
    hydrogens copied from ref.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------- #

# Header REMARK / annotation prefixes we always preserve from the seed.
_HEADER_PREFIXES: tuple[str, ...] = (
    "REMARK 666",
    "REMARK PDBinfo-LABEL",
    "HETNAM",
    "LINK",
)

# 3-letter residue codes that protein_chisel treats as "a histidine variant".
_HIS_VARIANTS: frozenset[str] = frozenset({"HIS", "HIS_D", "HIE", "HID", "HIP"})

# Carbamylated-lysine atom set. KCX is LYS with a carbamate (-NH-CO-O(-)) on
# the side chain epsilon nitrogen. Definitive markers are the carbamate
# carbon (CX) and its two carboxyl oxygens (OQ1, OQ2).
_KCX_MARKERS: frozenset[str] = frozenset({"CX", "OQ1", "OQ2"})

# Catalytic-residue hydrogens we will optionally copy from the seed. Order
# matters: by N-H atom name we can tell which tautomer they belong to.
_HIS_PROTON_NAMES: frozenset[str] = frozenset({"HD1", "HE2"})


# ---------------------------------------------------------------------- #
# Header / REMARK extraction
# ---------------------------------------------------------------------- #


def extract_remark_lines(
    input_pdb: str | Path,
    prefixes: Iterable[str] = _HEADER_PREFIXES,
) -> list[str]:
    """Return the non-coordinate header block from ``input_pdb``.

    Lines are returned with their trailing newlines intact. Stops scanning
    on the first ATOM/HETATM record (header lines must precede coordinates).

    Args:
        input_pdb: Path to a PDB file.
        prefixes: Line-start prefixes to keep. Defaults to REMARK 666,
            REMARK PDBinfo-LABEL, HETNAM, LINK.

    Returns:
        list[str] of preserved header lines, in source order.
    """
    prefixes = tuple(prefixes)
    out: list[str] = []
    with open(input_pdb, "r") as fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                break
            for p in prefixes:
                if line.startswith(p):
                    out.append(line)
                    break
    return out


# ---------------------------------------------------------------------- #
# Atom-line parsing helpers (PDB-format-aware, 5-char-resname-aware)
# ---------------------------------------------------------------------- #


def _resname_from_line(line: str) -> str:
    """Extract residue name handling both 3-char and 5-char Rosetta labels.

    Rosetta writes 5-char names (HIS_D, HIS_E, KCX_*) using columns 17-21
    rather than 17-19. We detect the 5-char form by sniffing column 16
    (alt-loc) for a non-blank that, combined with cols 17-19, forms a
    known 5-char name.
    """
    if len(line) < 21:
        return ""
    five = line[16:21].strip()
    if five in {"HIS_D", "HIS_E"}:
        return five
    return line[17:20].strip()


def _resno_from_line(line: str) -> int:
    return int(line[22:26])


def _chain_from_line(line: str) -> str:
    return line[21:22]


def _atom_name_from_line(line: str) -> str:
    return line[12:16].strip()


def _is_atom_line(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM")


# ---------------------------------------------------------------------- #
# Tautomer + KCX detection from input PDB
# ---------------------------------------------------------------------- #


def build_his_tautomer_map(
    input_pdb: str | Path,
) -> dict[tuple[str, int], str]:
    """Map HIS residues to their tautomer state from the input PDB.

    Reads atom records and determines tautomer by which N-H atom is
    present:

        HD1 only           -> "HIS_D"  (delta-N protonated)
        HE2 only           -> "HIE"    (epsilon-N protonated)
        both HD1 and HE2   -> "HIP"    (doubly protonated cation)
        neither H present  -> "HIS"    (no info; leave default)

    If the input itself uses 5-char labels (HIS_D / HIS_E), that label is
    trusted directly even when no hydrogens are present (heavy-atom-only
    Rosetta output retains the label).

    Args:
        input_pdb: Path to reference PDB.

    Returns:
        {(chain, resno) -> tautomer-string} for every HIS-variant residue
        seen in the file.
    """
    his_atoms: dict[tuple[str, int], set[str]] = {}
    his_label: dict[tuple[str, int], str] = {}

    with open(input_pdb, "r") as fh:
        for line in fh:
            if not _is_atom_line(line):
                continue
            resname = _resname_from_line(line)
            if resname not in _HIS_VARIANTS:
                continue
            key = (_chain_from_line(line), _resno_from_line(line))
            his_atoms.setdefault(key, set()).add(_atom_name_from_line(line))
            # Trust an explicit 5-char label if it disagrees with HIS.
            if resname != "HIS" and his_label.get(key, "HIS") == "HIS":
                his_label[key] = resname

    out: dict[tuple[str, int], str] = {}
    for key, atom_names in his_atoms.items():
        explicit = his_label.get(key)
        if explicit and explicit != "HIS":
            # Normalize HID -> HIS_D (Amber-style -> Rosetta-style).
            out[key] = "HIS_D" if explicit == "HID" else explicit
            continue
        has_hd1 = "HD1" in atom_names
        has_he2 = "HE2" in atom_names
        if has_hd1 and has_he2:
            out[key] = "HIP"
        elif has_hd1:
            out[key] = "HIS_D"
        elif has_he2:
            out[key] = "HIE"
        else:
            out[key] = "HIS"
    return out


def detect_kcx_residues(
    input_pdb: str | Path,
) -> dict[tuple[str, int], bool]:
    """Return {(chain, resno) -> True} for residues that look like KCX.

    A residue is flagged as KCX if either:
      * its 3-letter resname is "KCX", or
      * it carries the full carbamate atom set CX, OQ1, OQ2 (Rosetta's
        atom names for the carbamylated lysine cap).
    """
    by_res: dict[tuple[str, int], dict[str, str]] = {}
    explicit: set[tuple[str, int]] = set()

    with open(input_pdb, "r") as fh:
        for line in fh:
            if not _is_atom_line(line):
                continue
            resname = _resname_from_line(line)
            key = (_chain_from_line(line), _resno_from_line(line))
            atom = _atom_name_from_line(line)
            by_res.setdefault(key, {})[atom] = resname
            if resname == "KCX":
                explicit.add(key)

    out: dict[tuple[str, int], bool] = {k: True for k in explicit}
    for key, atoms in by_res.items():
        if _KCX_MARKERS.issubset(atoms.keys()):
            out[key] = True
    return out


def collect_catalytic_hydrogens(
    input_pdb: str | Path,
    keys: Iterable[tuple[str, int]],
    atom_names: Iterable[str] = _HIS_PROTON_NAMES,
) -> dict[tuple[str, int], list[str]]:
    """Collect specific hydrogen ATOM lines from the input PDB.

    Used to copy HIS HD1/HE2 (etc.) onto the MPNN heavy-atom output so
    Rosetta etc. don't have to guess the tautomer.

    Args:
        input_pdb: Reference PDB path.
        keys: residues to grab H atoms for.
        atom_names: atom-name whitelist to copy (default HD1, HE2).

    Returns:
        {(chain, resno) -> [raw atom lines]}. Lines retain their newlines.
    """
    keys = set(keys)
    atom_names = set(atom_names)
    out: dict[tuple[str, int], list[str]] = {k: [] for k in keys}
    with open(input_pdb, "r") as fh:
        for line in fh:
            if not _is_atom_line(line):
                continue
            key = (_chain_from_line(line), _resno_from_line(line))
            if key not in keys:
                continue
            if _atom_name_from_line(line) not in atom_names:
                continue
            out[key].append(line)
    return out


# ---------------------------------------------------------------------- #
# Atom-line rewriting (resname swap + KCX cap)
# ---------------------------------------------------------------------- #


def _rewrite_atom_resname(line: str, new_resname: str) -> str:
    """Return ``line`` with its 3-letter / 5-letter residue name replaced.

    For 5-char labels (HIS_D / HIS_E) we overwrite columns 17-21 (the
    alt-loc + 3-char-resname slot, since Rosetta stores 5-char names
    spanning that block). For ordinary 3-char names we replace cols 17-19.
    """
    if len(line) < 21:
        return line
    if len(new_resname) > 3:
        # 5-char form: cols 17-21, no alt-loc gap.
        new_label = new_resname.ljust(5)
        return line[:16] + new_label + line[21:]
    # 3-char form: keep alt-loc (col 16) intact.
    return line[:17] + new_resname.ljust(3) + line[20:]


def _looks_like_kcx_capable(resname: str) -> bool:
    """True if MPNN-output resname can plausibly be the KCX backbone (LYS)."""
    return resname in {"LYS", "KCX"}


# ---------------------------------------------------------------------- #
# Main restoration
# ---------------------------------------------------------------------- #


def restore_pdb_features(
    mpnn_pdb: str | Path,
    ref_pdb: str | Path,
    out_pdb: str | Path,
    *,
    his_tautomers: bool = True,
    kcx: bool = True,
    catalytic_hydrogens: bool = True,
    catalytic_resnos: Optional[Iterable[int]] = None,
    chain: str = "A",
    drop_mpnn_remarks: bool = True,
) -> dict:
    """Restore protonation, REMARK header, optional H atoms onto an MPNN PDB.

    Args:
        mpnn_pdb: Input PDB from LigandMPNN (heavy atoms only, all HIS,
            no REMARK 666).
        ref_pdb: Reference / seed PDB carrying tautomer labels (or HD1/HE2
            atoms), KCX atoms, REMARK 666, HETNAM, LINK.
        out_pdb: Path to write the restored PDB to.
        his_tautomers: Relabel HIS residues to their HIS_D / HIE / HIP
            variant per ref_pdb. Heavy-atom output, MPNN doesn't change
            HIS to anything else when fixed, so safe to apply universally.
        kcx: Relabel residues that were KCX in ref to KCX, and inject
            their CX / OQ1 / OQ2 atoms from the ref. Only acts on
            residues whose MPNN resname is LYS or already KCX.
        catalytic_hydrogens: Copy the catalytic HD1 / HE2 / KCX-cap H
            atoms from ref onto the corresponding residues in out_pdb.
            Skipped automatically if ref has no hydrogens at all.
        catalytic_resnos: Restrict tautomer/KCX/H restoration to these
            residue numbers (typical: REMARK 666 catalytic residues).
            None means "all HIS / KCX residues found in ref".
        chain: Chain to operate on (default "A"; PTE_i1 monomer).
        drop_mpnn_remarks: If True, discard MPNN's own REMARK lines so the
            ref's REMARK 666 etc. are the only headers in the output.

    Returns:
        Stats dict with counts for header lines restored, HIS relabels,
        KCX relabels, KCX caps inserted, hydrogens copied.
    """
    stats = {
        "header_lines_restored": 0,
        "his_relabeled": 0,
        "kcx_relabeled": 0,
        "kcx_atoms_inserted": 0,
        "hydrogens_copied": 0,
    }

    ref_pdb = Path(ref_pdb)
    mpnn_pdb = Path(mpnn_pdb)
    out_pdb = Path(out_pdb)

    # ---- 1. REMARK / HETNAM / LINK header from ref ------------------- #
    header_lines = extract_remark_lines(ref_pdb)
    stats["header_lines_restored"] = len(header_lines)

    # ---- 2. Tautomer + KCX maps from ref ----------------------------- #
    his_map = build_his_tautomer_map(ref_pdb) if his_tautomers else {}
    kcx_map = detect_kcx_residues(ref_pdb) if kcx else {}

    if catalytic_resnos is not None:
        wanted = {(chain, int(r)) for r in catalytic_resnos}
        his_map = {k: v for k, v in his_map.items() if k in wanted}
        kcx_map = {k: True for k in kcx_map if k in wanted}

    # ---- 3. KCX atom block from ref (CX, OQ1, OQ2; also NZ override) - #
    kcx_atom_lines: dict[tuple[str, int], list[str]] = {}
    if kcx_map:
        wanted_kcx = set(kcx_map.keys())
        cap_atoms = {"CX", "OQ1", "OQ2"}
        with open(ref_pdb, "r") as fh:
            for line in fh:
                if not _is_atom_line(line):
                    continue
                key = (_chain_from_line(line), _resno_from_line(line))
                if key not in wanted_kcx:
                    continue
                if _atom_name_from_line(line) in cap_atoms:
                    kcx_atom_lines.setdefault(key, []).append(line)

    # ---- 4. Catalytic-H lines from ref (HD1 / HE2 + KCX HZ if any) --- #
    h_lines: dict[tuple[str, int], list[str]] = {}
    if catalytic_hydrogens and (his_map or kcx_map):
        keys = set(his_map.keys()) | set(kcx_map.keys())
        h_lines = collect_catalytic_hydrogens(
            ref_pdb, keys, atom_names=_HIS_PROTON_NAMES,
        )

    # ---- 5. Walk MPNN PDB and rewrite -------------------------------- #
    output: list[str] = list(header_lines)

    # Track per-residue insertion of KCX caps and catalytic Hs (insert
    # after the last heavy-atom record for that residue).
    kcx_inserted: set[tuple[str, int]] = set()
    h_inserted: set[tuple[str, int]] = set()

    with open(mpnn_pdb, "r") as fh:
        mpnn_iter = list(fh)

    # Build resno -> last-line-index map so we know where to drop the
    # extra atoms.
    last_atom_idx: dict[tuple[str, int], int] = {}
    for idx, line in enumerate(mpnn_iter):
        if _is_atom_line(line):
            key = (_chain_from_line(line), _resno_from_line(line))
            last_atom_idx[key] = idx

    for idx, line in enumerate(mpnn_iter):
        if drop_mpnn_remarks and line.startswith("REMARK"):
            continue

        if not _is_atom_line(line):
            output.append(line)
            continue

        key = (_chain_from_line(line), _resno_from_line(line))
        resname = _resname_from_line(line)

        # KCX relabel + cap injection: only if MPNN didn't mutate it.
        if key in kcx_map and _looks_like_kcx_capable(resname):
            new_line = _rewrite_atom_resname(line, "KCX")
            if new_line != line:
                stats["kcx_relabeled"] += 1
            output.append(new_line)
            if idx == last_atom_idx[key] and key not in kcx_inserted:
                for cap in kcx_atom_lines.get(key, []):
                    cap_relabeled = _rewrite_atom_resname(cap, "KCX")
                    output.append(cap_relabeled)
                    stats["kcx_atoms_inserted"] += 1
                kcx_inserted.add(key)
            # Optional catalytic H tail
            if (
                idx == last_atom_idx[key]
                and key not in h_inserted
                and h_lines.get(key)
            ):
                for hline in h_lines[key]:
                    output.append(_rewrite_atom_resname(hline, "KCX"))
                    stats["hydrogens_copied"] += 1
                h_inserted.add(key)
            continue

        # HIS tautomer relabel: only if MPNN kept it as HIS.
        if key in his_map and resname in _HIS_VARIANTS:
            target = his_map[key]
            if target == "HIS":
                output.append(line)
            else:
                new_line = _rewrite_atom_resname(line, target)
                if new_line != line:
                    stats["his_relabeled"] += 1
                output.append(new_line)
            if (
                idx == last_atom_idx[key]
                and key not in h_inserted
                and h_lines.get(key)
            ):
                for hline in h_lines[key]:
                    target_label = his_map.get(key, "HIS")
                    output.append(_rewrite_atom_resname(hline, target_label))
                    stats["hydrogens_copied"] += 1
                h_inserted.add(key)
            continue

        # Default: pass through.
        output.append(line)

    # ---- 6. Write ---------------------------------------------------- #
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdb, "w") as fh:
        fh.writelines(output)

    LOGGER.debug(
        "pdb_restoration: %s -> %s | header=%d his=%d kcx=%d kcx_atoms=%d H=%d",
        mpnn_pdb.name, out_pdb.name,
        stats["header_lines_restored"], stats["his_relabeled"],
        stats["kcx_relabeled"], stats["kcx_atoms_inserted"],
        stats["hydrogens_copied"],
    )
    return stats


# ---------------------------------------------------------------------- #
# Driver-facing helper (matches the v2 pipeline call signature)
# ---------------------------------------------------------------------- #


def restore_sample_dir(
    *,
    sample_dir: str | Path,
    ref_pdb: str | Path,
    out_pdb_dir: str | Path,
    pdb_basename: str,
    candidate_ids: list[str],
    chain: str = "A",
    catalytic_resnos: Optional[Iterable[int]] = None,
    catalytic_hydrogens: bool = True,
) -> dict[str, Path]:
    """Restore every packed MPNN PDB in ``sample_dir/packed/`` and copy the
    result into ``out_pdb_dir`` keyed by candidate id.

    Mirrors the in-driver ``stage_restore_pdbs`` signature in
    ``scripts/iterative_design_v2.py`` so it can drop in directly.

    Args:
        sample_dir: LigandMPNN output dir (must contain a ``packed/``
            subdir produced by ``pack_side_chains=1``).
        ref_pdb: Seed PDB to pull header / tautomers / KCX from.
        out_pdb_dir: Where to write restored PDBs.
        pdb_basename: stem of the seed PDB (used to find packed files).
        candidate_ids: list of MPNN candidate ids of the form
            ``f"{pdb_basename}_lmpnn_{idx}"``. ``idx == 0`` = WT input
            row and is skipped (no PDB).
        chain: Chain to operate on.
        catalytic_resnos: Catalytic residues (1-indexed) to target;
            None means restore all HIS / KCX variants.
        catalytic_hydrogens: Copy catalytic HD1/HE2 from ref into output.

    Returns:
        ``{candidate_id: output_pdb_path}`` for every packed PDB found.
    """
    sample_dir = Path(sample_dir)
    out_pdb_dir = Path(out_pdb_dir)
    out_pdb_dir.mkdir(parents=True, exist_ok=True)

    packed_dir = sample_dir / "packed"
    if not packed_dir.is_dir():
        raise FileNotFoundError(
            f"No packed/ subdir under {sample_dir} -- did fused_mpnn run "
            "with pack_side_chains=1?"
        )

    pat = re.compile(rf"^{re.escape(pdb_basename)}_lmpnn_(\d+)$")

    out_map: dict[str, Path] = {}
    for cid in candidate_ids:
        m = pat.match(cid)
        if not m:
            continue
        idx = int(m.group(1))
        if idx == 0:
            continue  # WT input row, no PDB to restore
        src = packed_dir / f"{pdb_basename}_packed_{idx}_1.pdb"
        if not src.is_file():
            LOGGER.warning("restore_sample_dir: missing packed PDB %s", src.name)
            continue
        dst = out_pdb_dir / f"{cid}.pdb"
        try:
            restore_pdb_features(
                mpnn_pdb=src,
                ref_pdb=ref_pdb,
                out_pdb=dst,
                his_tautomers=True,
                kcx=True,
                catalytic_hydrogens=catalytic_hydrogens,
                catalytic_resnos=catalytic_resnos,
                chain=chain,
                drop_mpnn_remarks=True,
            )
        except Exception as exc:  # don't kill the whole batch
            LOGGER.error(
                "restore_sample_dir: %s failed (%s); copying raw + header",
                src.name, exc,
            )
            head = extract_remark_lines(ref_pdb)
            with open(src) as fin, open(dst, "w") as fout:
                fout.writelines(head)
                for line in fin:
                    if line.startswith("REMARK"):
                        continue
                    fout.write(line)
        out_map[cid] = dst

    LOGGER.info("restore_sample_dir: restored %d PDBs -> %s",
                 len(out_map), out_pdb_dir)
    return out_map
