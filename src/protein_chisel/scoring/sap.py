"""Shared hydrophobicity / SAP (spatial aggregation propensity) primitives.

Single source of truth for the Kyte-Doolittle scale, Tien max-SASA, and the
3-letter -> 1-letter map used by BOTH the driver's SAP proxy
(``scripts/iterative_design.py::_compute_sap_proxy``) and the adaptive
controller's surface-hydrophobicity axis (``sampling/adaptive_bias.py``).
Centralising them here removes the duplicated dicts (and the two diverging SAP
implementations) the 2026-06 audit flagged, and gives one place to evolve the
hydrophobicity model.

Two per-residue hydrophobicity weights:

- :func:`kd_weight_raw` -- signed Kyte-Doolittle. This is the *legacy* SAP-proxy
  weight; using it reproduces the historical ``sap_*`` columns byte-for-byte.

- :func:`kd_weight_corrected` -- centered and clamped at zero
  (``max(0, KD - KD_MEAN)``). The signed raw scale lets exposed *polar/charged*
  neighbours (negative KD) CANCEL hydrophobic contributions, and it under-counts
  alanine (raw KD only +1.8). Centering makes every below-average-hydrophobic
  residue contribute 0 (no cancellation) while alanine still scores positively
  (+2.29). This is the structure-aware solubility signal behind the corrected SAP
  (``sap_corr_*``) and the controller's planned SAP axis.

The spatial reduction :func:`sap_neighborhood_metrics` is pure and fully
host-testable (no freesasa), and is shared by both the raw and corrected proxies
so the only difference between them is the injected ``weight_fn``.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

# Kyte-Doolittle hydrophobicity (canonical; identical to the historical
# iterative_design.py KD_HYDROPHOBICITY).
KD_HYDROPHOBICITY: dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Tien et al. theoretical max residue SASA (Å²).
SASA_MAX_RESIDUE: dict[str, float] = {
    "A": 121, "C": 148, "D": 187, "E": 214, "F": 228, "G": 97, "H": 216,
    "I": 195, "K": 230, "L": 191, "M": 203, "N": 187, "P": 154, "Q": 214,
    "R": 265, "S": 143, "T": 163, "V": 165, "W": 264, "Y": 255,
}

# 3-letter -> 1-letter, including the His tautomer / KCX (carbamylated Lys)
# variants the pipeline emits.
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "HID": "H", "HIE": "H", "HIP": "H",
    "HIS_D": "H", "ILE": "I", "LEU": "L", "LYS": "K", "KCX": "K", "MET": "M",
    "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}

# Mean KD over the 20 canonical residues (~ -0.49); the centering offset.
KD_MEAN: float = sum(KD_HYDROPHOBICITY.values()) / len(KD_HYDROPHOBICITY)

# Default SAP neighbourhood radius (Å), matching the legacy proxy.
DEFAULT_SAP_RADIUS: float = 10.0


def kd_weight_raw(aa: str) -> float:
    """Signed Kyte-Doolittle weight (legacy SAP proxy; unknown AA -> 0.0)."""
    return KD_HYDROPHOBICITY.get(aa, 0.0)


def kd_weight_corrected(aa: str) -> float:
    """Centered, zero-clamped KD weight: ``max(0, KD - KD_MEAN)``.

    Below-average-hydrophobic residues (polar/charged) contribute 0 instead of
    cancelling hydrophobic neighbours; alanine still scores positively. Unknown
    AA -> 0.0 (an unknown must NOT inherit the +KD_MEAN offset and become a
    spurious positive contributor).
    """
    kd = KD_HYDROPHOBICITY.get(aa)
    if kd is None:
        return 0.0
    return max(0.0, kd - KD_MEAN)


def sap_neighborhood_metrics(
    aas: Sequence[Optional[str]],
    sasa_total: Sequence[float],
    cas: np.ndarray,
    *,
    weight_fn: Callable[[str], float],
    radius: float = DEFAULT_SAP_RADIUS,
) -> dict[str, float]:
    """Per-residue spatial SAP reduction -> ``{max, mean, p95}``.

    For each residue ``i``, sum over residues ``j`` whose CA is within ``radius``
    of CA(i)::

        SAP_i = Σ_j (sasa_total[j] / SASA_MAX_RESIDUE[aa_j]) * weight_fn(aa_j)

    ``aas`` are 1-letter codes (``None`` for unmapped residues, which are
    skipped). This reproduces the historical ``_compute_sap_proxy`` arithmetic
    exactly when ``weight_fn=kd_weight_raw`` (same neighbour order, same Python
    float accumulation), so the legacy ``sap_*`` columns stay byte-identical.

    Returns NaNs for an empty input (no residues).
    """
    n = len(aas)
    if n == 0:
        return {"max": float("nan"), "mean": float("nan"), "p95": float("nan")}
    cas_a = np.asarray(cas, dtype=float)
    per_res: list[float] = []
    for i in range(n):
        d = np.linalg.norm(cas_a - cas_a[i], axis=1)
        nbrs = np.where(d <= radius)[0]
        s = 0.0
        for j in nbrs:
            aa = aas[j]
            if aa is None:
                continue
            sasa_max = SASA_MAX_RESIDUE.get(aa, 200.0)
            s += (sasa_total[j] / sasa_max) * weight_fn(aa)
        per_res.append(s)
    arr = np.array(per_res)
    return {
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
    }
