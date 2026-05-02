"""FASPR side-chain packing wrapper (Huang 2020, MIT).

FASPR (https://github.com/tommyhuangthu/FASPR) is a tiny C++ classical
side-chain packer built around the Dunbrack 2010 backbone-dependent
rotamer library. It runs in well under a second on CPU for a typical
200aa protein -- ideal for use as a fast sanity baseline / reference
when comparing modern neural packers against a classical one.

Usage::

    from protein_chisel.tools.sidechain_packing_and_scoring.faspr_pack \
        import faspr_pack
    res = faspr_pack(
        pdb_path="design.pdb",
        out_pdb_path="design_repacked.pdb",
        sequence=None,            # default: keep input sequence
        fixed_residues=None,      # default: repack everything
    )
    res.to_dict()                 # filter-friendly metric dict

Notes:
- The binary lives at /net/software/lab/faspr/bin/FASPR with the Dunbrack
  rotamer library at /net/software/lab/faspr/bin/dun2010bbdep.bin
  (FASPR hard-codes that filename; the wrapper runs FASPR with cwd set
  to that directory so the lookup succeeds).
- Inputs MUST be a backbone-only or full-atom PDB with N/CA/C/O present
  for every residue. If side-chain atoms are included, FASPR ignores
  them and re-builds them. Multi-chain is supported.
- Lower-case letters in the sequence string fix that residue's rotamer
  (per FASPR's CLI convention). We expose this via `fixed_residues`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


LOGGER = logging.getLogger("protein_chisel.faspr_pack")


@dataclass
class FasprResult:
    """Result of a FASPR run."""
    out_pdb_path: Path
    n_residues: int = 0          # residues in the repacked output
    runtime_seconds: float = 0.0
    sequence_used: Optional[str] = None  # the seq we actually packed
    stdout_tail: str = ""        # last ~500 chars of FASPR stdout

    def to_dict(self, prefix: str = "faspr__") -> dict[str, float | int | str]:
        return {
            f"{prefix}n_residues": self.n_residues,
            f"{prefix}runtime_seconds": float(self.runtime_seconds),
        }


def _find_faspr_binary(explicit: Optional[str | Path] = None) -> Path:
    """Resolve a usable FASPR binary path.

    Order: explicit arg -> $FASPR -> $PATH (`shutil.which`) -> cluster
    install at /net/software/lab/faspr/bin/FASPR.
    """
    import os

    if explicit:
        p = Path(explicit).resolve()
        if p.is_file():
            return p
    env = os.environ.get("FASPR")
    if env and Path(env).is_file():
        return Path(env)
    found = shutil.which("FASPR")
    if found:
        return Path(found)
    from protein_chisel.paths import FASPR_CLUSTER_BIN
    if FASPR_CLUSTER_BIN.is_file():
        return FASPR_CLUSTER_BIN
    raise RuntimeError(
        f"FASPR binary not found. Looked in $FASPR, $PATH, and "
        f"{FASPR_CLUSTER_BIN}. Build via `cd external/faspr && bash build.sh` "
        "or use the cluster install."
    )


def _extract_sequence_from_pdb(pdb_path: Path) -> str:
    """Read the chain-A 1-letter sequence from a PDB.

    FASPR expects a single string of upper-case (repack) / lower-case
    (fix) letters covering every standard residue. Multi-chain handling
    is done by FASPR itself when the sequence string spans chains in
    PDB residue order.
    """
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    seen: set[tuple[str, int, str]] = set()  # (chain, resseq, icode)
    seq_chars: list[str] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            resname = line[17:20].strip()
            if resname not in three_to_one:
                continue
            chain = line[21]
            resseq = int(line[22:26])
            icode = line[26]
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            seq_chars.append(three_to_one[resname])
    return "".join(seq_chars)


def faspr_pack(
    pdb_path: str | Path,
    out_pdb_path: Optional[str | Path] = None,
    sequence: Optional[str] = None,
    fixed_residues: Optional[Iterable[int]] = None,
    faspr_exe: Optional[str | Path] = None,
    timeout: float = 60.0,
) -> FasprResult:
    """Repack side chains on a backbone with FASPR.

    Args:
        pdb_path: input PDB.
        out_pdb_path: where to write the repacked PDB. Defaults to
            ``<input_stem>_faspr.pdb`` next to the input.
        sequence: optional explicit 1-letter sequence to pack. If None,
            the input PDB's sequence is used. Pass a different sequence
            here to introduce mutations -- see FASPR README.
        fixed_residues: 0-indexed positions that should NOT be repacked
            (FASPR convention: lower-case in the sequence string).
        faspr_exe: optional explicit FASPR binary path.
        timeout: seconds before we kill the subprocess.

    Returns:
        FasprResult with the output path + a few summary metrics.
    """
    import time

    pdb_path = Path(pdb_path).resolve()
    exe = _find_faspr_binary(faspr_exe)
    if out_pdb_path is None:
        out_pdb_path = pdb_path.with_name(f"{pdb_path.stem}_faspr.pdb")
    out_pdb_path = Path(out_pdb_path).resolve()
    out_pdb_path.parent.mkdir(parents=True, exist_ok=True)

    seq = sequence or _extract_sequence_from_pdb(pdb_path)
    if fixed_residues:
        chars = list(seq)
        for i in fixed_residues:
            if 0 <= i < len(chars):
                chars[i] = chars[i].lower()
        seq = "".join(chars)

    # FASPR hard-codes 'dun2010bbdep.bin' relative to the binary. Run
    # with cwd=<binary_dir> so it finds the rotamer library, and use
    # absolute paths for input/output.
    workdir = tempfile.mkdtemp(prefix="chisel_faspr_")
    try:
        seq_file = Path(workdir) / "seq.txt"
        seq_file.write_text(seq + "\n")
        cmd = [
            str(exe),
            "-i", str(pdb_path),
            "-o", str(out_pdb_path),
            "-s", str(seq_file),
        ]
        LOGGER.info("running FASPR: %s", " ".join(cmd))
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(exe.parent),  # critical: FASPR finds dun2010bbdep.bin here
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        runtime = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(
                f"FASPR failed (exit {proc.returncode}):\n"
                f"  stdout: {proc.stdout[-1000:]}\n"
                f"  stderr: {proc.stderr[-1000:]}"
            )
        if not out_pdb_path.is_file():
            raise RuntimeError(
                f"FASPR exited 0 but did not write {out_pdb_path}"
            )
        # n_residues from the output PDB (CA atoms)
        n_residues = sum(
            1 for line in out_pdb_path.read_text().splitlines()
            if line.startswith("ATOM  ") and line[12:16].strip() == "CA"
        )
        return FasprResult(
            out_pdb_path=out_pdb_path,
            n_residues=n_residues,
            runtime_seconds=runtime,
            sequence_used=seq,
            stdout_tail=proc.stdout[-500:],
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


__all__ = ["FasprResult", "faspr_pack"]
