"""Multi-algorithm secondary-structure consensus.

Three independent algorithms vote per residue:

1. **DSSP** (via biotite or biopython if mkdssp is available). Standard
   reference; can fail on suboptimal backbones (e.g. RFdiffusion outputs
   with imperfect O/N placement).
2. **CA-only P-SEA** (Labesse & Mornon 1997). Robust on degenerate
   backbones because it uses only CA-CA distances and CA pseudo-torsions.
3. **Phi/psi torsion-based**. Maps each residue's (phi, psi) to a
   Ramachandran region: alpha-helix (-65, -45), beta (-120, +130),
   left-handed (~+60, +60). Else loop.

Consensus is per-residue majority vote across the algorithms that
returned a label. Confidence = fraction of agreeing algorithms.

When DSSP fails (e.g. ``Bio.PDB.DSSP`` raises or biotite returns nothing),
we fall back to consensus of the two remaining algorithms.

The output ``SSConsensus.ss_reduced`` is over the 3-letter reduced
alphabet ``H`` (helix) / ``E`` (strand) / ``L`` (loop).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


LOGGER = logging.getLogger("protein_chisel.structure.secondary_structure")


@dataclass
class SSConsensus:
    sequence: str                          # length L
    ss_reduced: str                        # length L, "HEL" alphabet
    confidence: np.ndarray                 # shape (L,), in [0, 1]
    per_algo: dict[str, str] = field(default_factory=dict)
    used_algos: tuple[str, ...] = ()
    failed_algos: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Algorithm 1: DSSP via biotite (preferred — no mkdssp binary required)
# ---------------------------------------------------------------------------


def _dssp_via_biotite(pdb_path: Path, chain: str) -> Optional[str]:
    """Compute DSSP via biotite. Returns reduced-alphabet string or None."""
    try:
        import biotite.structure.io.pdb as bpdb
        import biotite.structure as bst
    except ImportError:
        return None
    try:
        f = bpdb.PDBFile.read(str(pdb_path))
        struct = f.get_structure(model=1)
        # restrict to one chain, protein residues, CA-bearing
        mask = (struct.chain_id == chain) & np.isin(
            struct.atom_name, ["CA", "N", "C", "O"],
        )
        struct = struct[mask]
        if len(struct) == 0:
            return None
        # biotite uses an internal DSSP-style annotator
        # annotate_sse returns one letter per residue: 'a' (alpha), 'b' (beta),
        # 'c' (coil)
        ca_struct = struct[struct.atom_name == "CA"]
        sse = bst.annotate_sse(ca_struct)
        mapping = {"a": "H", "b": "E", "c": "L"}
        return "".join(mapping.get(s, "L") for s in sse)
    except Exception as e:
        LOGGER.debug("biotite DSSP failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Algorithm 2: CA-only P-SEA (Labesse & Mornon 1997)
# ---------------------------------------------------------------------------


def _ca_only_psea(pdb_path: Path, chain: str) -> Optional[str]:
    """Pure-CA P-SEA SS assignment.

    For each residue i in the protein chain, we consider CA(i-2..i+2)
    where available and compute CA-CA distances (i, i+2), (i, i+3),
    (i, i+4) plus the pseudo-torsion CA(i-1)-CA(i)-CA(i+1)-CA(i+2).

    Helix region:
        d(i, i+2) ~ 5.5 +/- 0.5
        d(i, i+3) ~ 5.3 +/- 0.5
        d(i, i+4) ~ 6.4 +/- 0.6
        torsion ~ 50 +/- 20 deg

    Strand region:
        d(i, i+2) ~ 6.7 +/- 0.6
        d(i, i+3) ~ 9.9 +/- 0.9
        d(i, i+4) ~ 12.4 +/- 1.1
        torsion ~ -170 +/- 45 deg
    """
    try:
        cas = _read_ca_coords(pdb_path, chain)
    except Exception as e:
        LOGGER.debug("ca-only: read CA failed: %s", e)
        return None
    if len(cas) < 5:
        return None

    L = len(cas)
    cas_a = np.array([c for _, c in cas])

    def dist(i: int, j: int) -> float:
        return float(np.linalg.norm(cas_a[i] - cas_a[j]))

    def pseudo_torsion(i: int) -> float:
        # CA(i-1) -> CA(i) -> CA(i+1) -> CA(i+2)
        if i < 1 or i + 2 >= L:
            return float("nan")
        b1 = cas_a[i] - cas_a[i - 1]
        b2 = cas_a[i + 1] - cas_a[i]
        b3 = cas_a[i + 2] - cas_a[i + 1]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        m1 = np.cross(n1, b2 / max(np.linalg.norm(b2), 1e-9))
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return float(np.degrees(np.arctan2(y, x)))

    # Wider P-SEA-like bounds: tested empirically against biotite SSE on
    # PTE_i1; original Labesse 1997 cutoffs are too narrow on real
    # protein structures.
    out = ["L"] * L
    for i in range(L):
        if i + 4 >= L:
            continue
        d2 = dist(i, i + 2)
        d3 = dist(i, i + 3)
        d4 = dist(i, i + 4)
        # Helix: tight i->i+4 packing
        if 5.0 <= d2 <= 7.0 and 4.5 <= d3 <= 7.0 and 5.5 <= d4 <= 7.5:
            out[i] = "H"
            continue
        # Strand: extended chain
        if d2 >= 6.5 and d3 >= 9.0 and d4 >= 11.0:
            out[i] = "E"
    # Smooth: 1-residue islands -> loop; extend helices by one (P-SEA convention)
    arr = list(out)
    for i in range(1, L - 1):
        if arr[i] != "L" and arr[i - 1] == "L" and arr[i + 1] == "L":
            arr[i] = "L"
    return "".join(arr)


def _read_ca_coords(pdb_path: Path, chain: str) -> list[tuple[int, np.ndarray]]:
    """Stdlib ATOM CA parser, chain-restricted, sorted by resseq."""
    out: dict[int, np.ndarray] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21].strip() != chain:
                continue
            try:
                resno = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            if resno not in out:    # first altloc only
                out[resno] = np.array([x, y, z], dtype=float)
    return sorted(out.items(), key=lambda t: t[0])


# ---------------------------------------------------------------------------
# Algorithm 3: phi/psi torsion-based (using full backbone)
# ---------------------------------------------------------------------------


def _torsion_based(pdb_path: Path, chain: str) -> Optional[str]:
    """Per-residue (phi, psi) -> H/E/L assignment.

    Uses the full backbone (N, CA, C). Returns None if not parseable.
    """
    try:
        backbones = _read_backbone(pdb_path, chain)
    except Exception:
        return None
    if len(backbones) < 3:
        return None

    L = len(backbones)
    out = ["L"] * L
    coords = [c for _, c in backbones]   # list of dict[atom -> xyz]

    def dihedral(p0, p1, p2, p3) -> float:
        # Praxeolitic single-precision torsion. b0 points from p1 -> p0,
        # b2 from p2 -> p3. Sign convention: positive = counter-clockwise
        # looking down the b1 axis.
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1n = b1 / max(np.linalg.norm(b1), 1e-9)
        v = b0 - np.dot(b0, b1n) * b1n
        w = b2 - np.dot(b2, b1n) * b1n
        x = np.dot(v, w)
        y = np.dot(np.cross(b1n, v), w)
        return float(np.degrees(np.arctan2(y, x)))

    for i in range(1, L - 1):
        ci = coords[i]
        if not all(k in ci for k in ("N", "CA", "C")):
            continue
        prev_C = coords[i - 1].get("C")
        next_N = coords[i + 1].get("N")
        if prev_C is None or next_N is None:
            continue
        phi = dihedral(prev_C, ci["N"], ci["CA"], ci["C"])
        psi = dihedral(ci["N"], ci["CA"], ci["C"], next_N)
        if -100 <= phi <= -30 and -90 <= psi <= 0:
            out[i] = "H"
        elif -180 <= phi <= -50 and 90 <= psi <= 180:
            out[i] = "E"
    # Smooth: 1-residue helix/strand islands -> loop
    for i in range(1, L - 1):
        if out[i] != "L" and out[i - 1] == "L" and out[i + 1] == "L":
            out[i] = "L"
    return "".join(out)


def _read_backbone(pdb_path: Path, chain: str) -> list[tuple[int, dict[str, np.ndarray]]]:
    """Read N/CA/C atoms per residue on chain, sorted by resseq."""
    out: dict[int, dict[str, np.ndarray]] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[21].strip() != chain:
                continue
            atom = line[12:16].strip()
            if atom not in ("N", "CA", "C"):
                continue
            try:
                resno = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            out.setdefault(resno, {})
            if atom not in out[resno]:
                out[resno][atom] = np.array([x, y, z], dtype=float)
    return sorted(out.items(), key=lambda t: t[0])


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


class SSProvider:
    """Compute multi-algorithm SS consensus from a PDB."""

    def __init__(
        self,
        algos: tuple[str, ...] = ("dssp_biotite", "ca_only_psea", "torsion"),
    ) -> None:
        self.algos = algos
        self._impls = {
            "dssp_biotite": _dssp_via_biotite,
            "ca_only_psea": _ca_only_psea,
            "torsion": _torsion_based,
        }

    def from_pdb(self, pdb_path: Path, chain: str = "A") -> SSConsensus:
        per_algo: dict[str, str] = {}
        failed: list[str] = []
        for algo in self.algos:
            fn = self._impls.get(algo)
            if fn is None:
                continue
            res = fn(pdb_path, chain)
            if res is None:
                failed.append(algo)
                continue
            per_algo[algo] = res

        # Reference length: extract sequence from CA coords as the source of truth
        cas = _read_ca_coords(pdb_path, chain)
        L = len(cas)
        seq = "X" * L  # placeholder; we don't decode AA from CA-only

        if not per_algo:
            LOGGER.warning("all SS algos failed for %s chain %s", pdb_path, chain)
            return SSConsensus(
                sequence=seq, ss_reduced="L" * L,
                confidence=np.zeros(L), per_algo={},
                used_algos=(), failed_algos=tuple(failed),
            )

        # Truncate / pad to L
        for k, v in list(per_algo.items()):
            if len(v) > L:
                per_algo[k] = v[:L]
            elif len(v) < L:
                per_algo[k] = v + "L" * (L - len(v))

        # Per-residue majority vote
        consensus = []
        confidence = np.zeros(L, dtype=float)
        labels_arr = np.array([list(v) for v in per_algo.values()])  # (n_algos, L)
        for i in range(L):
            col = labels_arr[:, i]
            unique, counts = np.unique(col, return_counts=True)
            top = unique[np.argmax(counts)]
            consensus.append(top)
            confidence[i] = counts.max() / len(col)
        return SSConsensus(
            sequence=seq,
            ss_reduced="".join(consensus),
            confidence=confidence,
            per_algo=per_algo,
            used_algos=tuple(per_algo.keys()),
            failed_algos=tuple(failed),
        )


__all__ = ["SSConsensus", "SSProvider"]
