"""Dynamic Flexibility Index (DFI) — GNM-based, CA-only.

Implements the simple Gaussian Network Model variant of the Dynamic
Flexibility Index (Bahadur & Ozkan, *PLoS Comp Biol* 2013,
doi:10.1371/journal.pcbi.1003217). Builds a Kirchhoff connectivity
matrix from CA coordinates, computes its pseudo-inverse, and reports
per-residue mean-square fluctuation (the diagonal of Γ⁺).

The "DFI score" we report per residue is just `Γ⁺_ii`, normalized so
that the mean across all residues = 1.0. Values > 1 = more flexible
than the protein average; values < 1 = more rigid.

For enzyme design we particularly care about:
  - DFI mean across **primary_sphere** residues (catalytic loop /
    substrate-binding shell). High = flexible enough for turnover.
  - DFI mean across **distal_buried** residues (framework). Low =
    stable scaffold. High framework DFI = unstable, often misfolds.

This is a sequence-only-of-the-CA-trace calculation; it doesn't need
the ligand or any sidechain detail. Cost: dominated by the SVD pseudo-
inverse, ~O(L³). For L≈200 expect 30–100 ms per design.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


LOGGER = logging.getLogger("protein_chisel.scoring.dfi")


@dataclass
class DfiResult:
    """Per-residue DFI scores + per-class summaries."""
    dfi: np.ndarray                 # shape (L,), mean-normalized
    resnos: list[int]               # CA-resno per row, length L
    cutoff_used: float              # Å radius used for Kirchhoff
    elapsed_ms: float
    # Per-class summary if classes provided.
    per_class_mean: dict[str, float] = None      # type: ignore[assignment]
    per_class_std: dict[str, float] = None       # type: ignore[assignment]

    def to_dict(self, prefix: str = "dfi__") -> dict[str, float]:
        out: dict[str, float] = {
            f"{prefix}mean":     float(np.nanmean(self.dfi)),
            f"{prefix}std":      float(np.nanstd(self.dfi)),
            f"{prefix}max":      float(np.nanmax(self.dfi)),
            f"{prefix}min":      float(np.nanmin(self.dfi)),
            f"{prefix}elapsed_ms": float(self.elapsed_ms),
        }
        if self.per_class_mean:
            for cls, v in self.per_class_mean.items():
                out[f"{prefix}mean__{cls}"] = float(v)
            for cls, v in (self.per_class_std or {}).items():
                out[f"{prefix}std__{cls}"] = float(v)
        return out


def _read_ca_coords(pdb_path: Path | str, chain: str = "A") -> tuple[np.ndarray, list[int]]:
    """Stdlib CA reader (chain-restricted, first altloc only)."""
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
            if resno not in out:
                out[resno] = np.array([x, y, z], dtype=float)
    sorted_items = sorted(out.items(), key=lambda t: t[0])
    if not sorted_items:
        return np.zeros((0, 3)), []
    coords = np.stack([c for _, c in sorted_items], axis=0)
    resnos = [r for r, _ in sorted_items]
    return coords, resnos


def compute_dfi(
    pdb_path: Path | str,
    *,
    chain: str = "A",
    cutoff: float = 10.0,
    classes: Optional[list[str]] = None,
    classes_resnos: Optional[list[int]] = None,
) -> DfiResult:
    """Compute per-residue DFI from a PDB.

    Args:
        pdb_path: PDB file with CA coordinates.
        chain: chain ID. Default "A".
        cutoff: Kirchhoff cutoff in Å. Default 10.0 (standard GNM).
        classes: optional list of class strings (length L). If provided,
                 returns per-class mean/std DFI in `per_class_mean/std`.
        classes_resnos: optional resnos that map to `classes` so we can
                 join them onto the CA-trace. If None, assumes
                 ``classes`` is already in CA-row order.
    """
    t0 = time.perf_counter()
    coords, resnos = _read_ca_coords(pdb_path, chain=chain)
    L = len(coords)
    if L < 3:
        raise ValueError(f"too few CA atoms for DFI ({L})")

    # Pairwise CA-CA distance matrix.
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = (diff * diff).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)              # exclude self

    # Kirchhoff matrix Γ: -1 for in-contact pairs, 0 else; diag = degree.
    contact = (d2 < cutoff * cutoff).astype(np.float64)
    gamma = -contact.copy()
    np.fill_diagonal(gamma, contact.sum(axis=1))

    # Pseudo-inverse via eigendecomposition (Γ is symmetric PSD with one
    # zero mode for translational invariance).
    eigvals, eigvecs = np.linalg.eigh(gamma)
    # Keep eigenvalues > tiny tolerance; invert; reproject.
    tol = max(1e-9, eigvals.max() * 1e-12)
    inv_eigvals = np.where(eigvals > tol, 1.0 / eigvals, 0.0)
    gamma_pinv = (eigvecs * inv_eigvals[None, :]) @ eigvecs.T

    # Per-residue mean-square fluctuation = diag(Γ⁺).
    msf = np.diag(gamma_pinv)
    # Negative due to numerics → clamp.
    msf = np.where(msf > 0, msf, 0.0)

    # Normalize so the mean across the protein is 1.0.
    mean_msf = msf.mean()
    dfi = msf / mean_msf if mean_msf > 0 else msf.copy()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Per-class summary.
    per_class_mean: dict[str, float] = {}
    per_class_std: dict[str, float] = {}
    if classes is not None:
        # Map classes to CA-row order.
        if classes_resnos is not None:
            cls_by_resno = dict(zip(classes_resnos, classes))
            class_per_row = [cls_by_resno.get(r, "?") for r in resnos]
        else:
            if len(classes) != L:
                LOGGER.warning(
                    "DFI: classes length (%d) != L (%d); skipping per-class",
                    len(classes), L,
                )
                class_per_row = []
            else:
                class_per_row = list(classes)
        for cls in set(c for c in class_per_row if c and c != "?"):
            mask = np.array([c == cls for c in class_per_row])
            if mask.sum() == 0:
                continue
            per_class_mean[cls] = float(dfi[mask].mean())
            per_class_std[cls] = float(dfi[mask].std())

    return DfiResult(
        dfi=dfi,
        resnos=resnos,
        cutoff_used=cutoff,
        elapsed_ms=elapsed_ms,
        per_class_mean=per_class_mean or None,    # type: ignore[arg-type]
        per_class_std=per_class_std or None,       # type: ignore[arg-type]
    )


__all__ = ["DfiResult", "compute_dfi"]
