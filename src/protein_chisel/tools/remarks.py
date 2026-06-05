"""Transfer, merge, and record PDB REMARK header lines (pure-Python, no PyRosetta).

Shared by the design pipelines so they can rescue REMARK metadata from an
input/seed PDB onto their outputs and stamp provenance:

  * ``scripts/chisel_ligandMPNN.py``  (DESIGN_PATH stage ``chisel_ligandmpnn``)
  * ``scripts/iterative_design.py``   (stage ``iterative_design``, on adoption)

Core operations:
  - :func:`reorganize_pdb_remarks` — rescue REMARK lines from an input/seed PDB
    onto an output PDB, drop PyRosetta dump artifacts (``REMARK 0``), and rewrite
    the header in a canonical order; optionally append a
    ``REMARK DESIGN_PATH <stage> <kind> <path>`` provenance line. The ``<stage>``
    tag is a **caller argument** so each pipeline records its own label.
  - :func:`transfer_remarks_to_dir` — apply the above to every design PDB in a dir.
  - :func:`replace_remark_block` — strip a numbered-REMARK block (e.g. 667/668)
    and insert a freshly-built one after ``REMARK 666``.

Ported from ``special_scripts/.../idealize_rfdiffusion3_geometry__MAIN.py``
``::reorganize_output_pdb_remarks`` (same logic; DESIGN_PATH stage parameterized).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

LOGGER = logging.getLogger(__name__)

_BODY_RECORDS = (
    "ATOM", "HETATM", "TER", "END", "ENDMDL", "MODEL", "CONECT", "MASTER",
)
_INTERMEDIATE_SUFFIXES = (".protonated.pdb", ".rosetta.pdb", ".norm.pdb")


def is_intermediate(name: str) -> bool:
    """True for protonation intermediates that batch ops should skip."""
    return name.endswith(_INTERMEDIATE_SUFFIXES)


def design_path_line(stage: str, kind: str, path: str) -> str:
    """``REMARK DESIGN_PATH <stage> <kind> <normpath(path)>`` (newline-terminated)."""
    return f"REMARK DESIGN_PATH {stage} {kind} {os.path.normpath(path)}\n"


def normalize_design_path_line(line: str) -> str:
    """Normalize the path token of a DESIGN_PATH line (collapse ``//``, strip
    column-80 padding). Non-DESIGN_PATH / malformed lines pass through unchanged.
    """
    if not line.startswith("REMARK DESIGN_PATH"):
        return line
    body = line.rstrip("\n")
    nl = line[len(body):]
    tokens = body.split(None, 4)  # at most 5 fields: REMARK DESIGN_PATH stage kind path
    if len(tokens) < 5:
        return line
    return f"{' '.join(tokens[:4])} {os.path.normpath(tokens[4].rstrip())}{nl}"


def _parse(lines, capture_body, include_all_header_lines):
    numbered: dict = {}
    grouped: dict = {"QCB": [], "rfd3_property": [], "DESIGN_PATH": [], "_misc": []}
    header_kept, body_lines = [], []
    in_body = False
    for line in lines:
        if not in_body and any(line.startswith(rec) for rec in _BODY_RECORDS):
            in_body = True
        if in_body:
            if capture_body:
                body_lines.append(line)
            continue
        if line.startswith("HEADER") or line.startswith("EXPDTA"):
            if include_all_header_lines:
                header_kept.append(line)
            continue
        if line.startswith("REMARK"):
            tokens = line.split()
            if len(tokens) < 2:
                continue
            tag = tokens[1]
            if tag.isdigit():
                rem_num = int(tag)
                if rem_num == 0:  # PyRosetta dump_pdb artifact for named REMARKs
                    continue
                if rem_num == 220 and not include_all_header_lines:
                    continue
                numbered.setdefault(rem_num, []).append(line)
            elif tag in ("QCB", "rfd3_property", "DESIGN_PATH"):
                grouped[tag].append(line)
            else:
                grouped["_misc"].append(line)
            continue
        if include_all_header_lines:
            header_kept.append(line)
    return numbered, grouped, header_kept, body_lines


def _merge(input_list, output_list):
    """Input lines authoritative; keep output-only lines that aren't a
    byte-identical or column-80-truncation match of an input line (PyRosetta
    cuts long REMARKs at col 80)."""
    in_stripped = [s.rstrip() for s in input_list]
    in_set = set(in_stripped)

    def _is_trunc(s):
        return bool(s) and any(i.startswith(s) and len(i) > len(s) for i in in_stripped)

    surviving = [o for o in output_list
                 if o.rstrip() not in in_set and not _is_trunc(o.rstrip())]
    return list(input_list) + surviving


def _dedupe(lines, *, fold_prefixes=False):
    """Order-preserving de-duplication for the final REMARK header.

    Always drops a line byte-identical (ignoring trailing whitespace) to one
    already kept. When ``fold_prefixes`` is set, also collapses prefix
    duplicates — a shorter line that is a strict prefix of a longer one — into
    the longer, more complete variant (kept in the shorter line's position).
    That folds PyRosetta col-80 truncations *and* legend lines that drifted by a
    trailing edit (e.g. ``…anchors`` vs ``…anchors.``) whichever order they
    arrive in. Distinct fixed-width data lines (REMARK 666/668) are equal-length
    and so never strict-prefix one another, so folding is safe for them; it is
    left OFF for grouped/provenance lines (e.g. DESIGN_PATH) where one path may
    legitimately nest under another.
    """
    kept: list[str] = []
    kept_s: list[str] = []
    for line in lines:
        s = line.rstrip()
        if not s or s in kept_s:
            continue
        if fold_prefixes:
            if any(k.startswith(s) and len(k) > len(s) for k in kept_s):
                continue  # shorter truncation of a line already kept
            longer = next((i for i, k in enumerate(kept_s)
                           if s.startswith(k) and len(s) > len(k)), None)
            if longer is not None:
                kept[longer], kept_s[longer] = line, s  # supersede the kept prefix
                continue
        kept.append(line)
        kept_s.append(s)
    return kept


def reorganize_pdb_remarks(
    output_pdb: str | Path,
    input_pdb: Optional[str | Path] = None,
    *,
    design_path_stage: Optional[str] = None,
    design_path_kind: str = "output",
    include_all_header_lines: bool = False,
    keep_output_numbered: Iterable[int] = (),
    verbose: bool = False,
) -> None:
    """Rewrite ``output_pdb``'s header into a canonical REMARK layout, rescuing
    any REMARK lines present in ``input_pdb`` (REMARK 665/666, QCB,
    rfd3_property, DESIGN_PATH, misc) — input lines win on conflict.

    ``keep_output_numbered`` lists numbered-REMARK kinds (e.g. ``(667, 668)``)
    for which ``output_pdb``'s own lines are authoritative: the input's lines for
    those numbers are NOT merged in. Use it when ``output_pdb`` already carries a
    freshly-rebuilt block (e.g. a protonation REMARK 668 reflecting the actual
    pose) that must not be overwritten by a stale copy from ``input_pdb``.

    When ``design_path_stage`` is given, append a
    ``REMARK DESIGN_PATH <design_path_stage> <design_path_kind> <output_pdb>``
    line at the bottom of the DESIGN_PATH group. Rewrites ``output_pdb`` in place.
    """
    output_pdb = str(output_pdb)
    with open(output_pdb, "r") as fh:
        out_lines = fh.readlines()
    numbered, grouped, header_kept, body_lines = _parse(
        out_lines, capture_body=True, include_all_header_lines=include_all_header_lines)

    if input_pdb is not None:
        try:
            with open(str(input_pdb), "r") as fh:
                in_lines = fh.readlines()
        except OSError as exc:
            if verbose:
                LOGGER.warning("could not read input PDB for REMARK transfer (%s): %s",
                               input_pdb, exc)
            in_lines = None
        if in_lines is not None:
            in_numbered, in_grouped, _h, _b = _parse(
                in_lines, capture_body=False,
                include_all_header_lines=include_all_header_lines)
            keep_out = {int(n) for n in keep_output_numbered}
            for num, lst in in_numbered.items():
                if num in keep_out:
                    continue  # output's lines for this REMARK number are authoritative
                numbered[num] = _merge(lst, numbered.get(num, []))
            for grp, lst in in_grouped.items():
                grouped[grp] = _merge(lst, grouped[grp])

    if design_path_stage is not None:
        grouped["DESIGN_PATH"].append(
            design_path_line(design_path_stage, design_path_kind, output_pdb))
    grouped["DESIGN_PATH"] = [normalize_design_path_line(l) for l in grouped["DESIGN_PATH"]]

    new_lines = list(header_kept)
    for n in sorted(numbered.keys()):
        new_lines.extend(_dedupe(numbered[n], fold_prefixes=True))
    for grp in ("QCB", "rfd3_property", "DESIGN_PATH", "_misc"):
        new_lines.extend(_dedupe(grouped[grp]))
    new_lines.extend(body_lines)
    with open(output_pdb, "w") as fh:
        fh.writelines(new_lines)


def transfer_remarks_to_dir(
    directory: str | Path,
    input_pdb: str | Path,
    *,
    transfer_input: bool = True,
    design_path_stage: Optional[str] = None,
    design_path_kind: str = "output",
    keep_output_numbered: Iterable[int] = (),
) -> int:
    """Apply :func:`reorganize_pdb_remarks` to every non-intermediate ``*.pdb`` in
    ``directory``. Returns the number of PDBs processed."""
    base = Path(directory)
    pdbs = [p for p in sorted(base.glob("*.pdb")) if not is_intermediate(p.name)]
    for p in pdbs:
        reorganize_pdb_remarks(
            p,
            input_pdb if transfer_input else None,
            design_path_stage=design_path_stage,
            design_path_kind=design_path_kind,
            keep_output_numbered=keep_output_numbered,
        )
    return len(pdbs)


def replace_remark_block(
    pdb_path: str | Path,
    new_lines: Iterable[str],
    *,
    drop_prefixes: Iterable[str] = ("REMARK 667", "REMARK 668"),
    after_prefix: str = "REMARK 666",
) -> bool:
    """Strip lines matching ``drop_prefixes`` and insert ``new_lines`` right after
    the last ``after_prefix`` line (else before the first coordinate record).
    Returns False (no-op) if ``new_lines`` is empty."""
    block = [b if b.endswith("\n") else b + "\n" for b in new_lines]
    if not block:
        return False
    pdb_path = Path(pdb_path)
    drop = tuple(drop_prefixes)
    lines = pdb_path.read_text().splitlines(keepends=True)
    kept = [l for l in lines if not any(l.startswith(d) for d in drop)]

    last_after, first_body = -1, len(kept)
    for i, l in enumerate(kept):
        if l.startswith(after_prefix):
            last_after = i
        if first_body == len(kept) and any(l.startswith(r) for r in _BODY_RECORDS):
            first_body = i
    insert_at = last_after + 1 if last_after >= 0 else first_body

    pdb_path.write_text("".join(kept[:insert_at] + block + kept[insert_at:]))
    return True
