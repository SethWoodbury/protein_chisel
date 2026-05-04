"""Swiss-Prot amino-acid composition reference data + checkers.

Two reference distributions are stored here as the canonical
"natural-protein composition" priors against which designs are
compared:

1. ``SWISSPROT_GLOBAL_2026_01`` — all reviewed Swiss-Prot proteins.
2. ``SWISSPROT_ENZYME_2026_01`` — only EC-annotated entries.

For each AA we store both the global percentage (residue-level
frequency over all proteins concatenated) and the per-sequence mean
± standard deviation, so callers can compute either a "library-level"
or a "per-design" z-score.

The expected use is:
  - For a given designed sequence, compute its AA composition.
  - For each AA, compute the z-score against the *enzyme* distribution.
  - Flag AAs whose abs(z) exceeds a per-rule threshold (default 2.0).

De-novo designs are EXPECTED to drift from these distributions a bit,
so the threshold is permissive by default (z=2.0); tighten as needed
per scaffold.

Cysteine is intentionally excluded from over-representation checks
when ``omit_AA`` includes "C" because the design protocol forbids it
explicitly.

Source: Swiss-Prot release 2026_01, n=574,627 proteins (global) /
n=279,501 proteins (EC-annotated). Computed by the user from UniProt
headers; the numbers below were checked against the official UniProt
Swiss-Prot composition table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AaStats:
    """Per-AA statistics from a reference distribution."""
    global_pct: float
    per_seq_mean_pct: float
    per_seq_std_pct: float


# All reviewed Swiss-Prot proteins, 2026_01
# n = 574,627 proteins; 208,473,776 standard-AA residues.
SWISSPROT_GLOBAL_2026_01: dict[str, AaStats] = {
    "A": AaStats(8.256, 8.443, 3.652),
    "C": AaStats(1.389, 1.484, 2.058),
    "D": AaStats(5.463, 5.294, 2.104),
    "E": AaStats(6.716, 6.593, 2.829),
    "F": AaStats(3.869, 3.879, 2.017),
    "G": AaStats(7.072, 7.177, 2.863),
    "H": AaStats(2.279, 2.232, 1.361),
    "I": AaStats(5.906, 6.063, 2.666),
    "K": AaStats(5.797, 6.160, 3.417),
    "L": AaStats(9.649, 9.574, 3.089),
    "M": AaStats(2.412, 2.551, 1.318),
    "N": AaStats(4.065, 3.936, 2.126),
    "P": AaStats(4.750, 4.536, 2.333),
    "Q": AaStats(3.933, 3.775, 2.012),
    "R": AaStats(5.529, 5.779, 3.062),
    "S": AaStats(6.664, 6.261, 2.637),
    "T": AaStats(5.366, 5.258, 1.946),
    "V": AaStats(6.855, 7.054, 2.427),
    "W": AaStats(1.106, 1.069, 1.059),
    "Y": AaStats(2.925, 2.883, 1.581),
}

# Swiss-Prot enzyme subset (EC-annotated), 2026_01
# n = 279,501 proteins; 116,050,827 standard-AA residues.
SWISSPROT_ENZYME_2026_01: dict[str, AaStats] = {
    "A": AaStats(8.699, 8.963, 3.302),
    "C": AaStats(1.275, 1.271, 1.054),
    "D": AaStats(5.735, 5.703, 1.585),
    "E": AaStats(6.708, 6.675, 2.155),
    "F": AaStats(3.887, 3.890, 1.579),
    "G": AaStats(7.471, 7.617, 2.148),
    "H": AaStats(2.367, 2.381, 1.098),
    "I": AaStats(6.118, 6.278, 2.310),
    "K": AaStats(5.477, 5.407, 2.558),
    "L": AaStats(9.641, 9.720, 2.391),
    "M": AaStats(2.445, 2.510, 1.099),
    "N": AaStats(3.899, 3.788, 1.781),
    "P": AaStats(4.618, 4.528, 1.583),
    "Q": AaStats(3.656, 3.569, 1.551),
    "R": AaStats(5.486, 5.488, 2.095),
    "S": AaStats(5.990, 5.704, 1.943),
    "T": AaStats(5.275, 5.194, 1.510),
    "V": AaStats(7.118, 7.254, 1.983),
    "W": AaStats(1.137, 1.101, 0.918),
    "Y": AaStats(3.000, 2.961, 1.336),
}


REFERENCE_DISTRIBUTIONS = {
    "swissprot_global_2026_01": SWISSPROT_GLOBAL_2026_01,
    "swissprot_enzyme_2026_01": SWISSPROT_ENZYME_2026_01,
}


def aa_composition_pct(sequence: str) -> dict[str, float]:
    """Per-AA percentage in a 1-letter sequence (canonical AAs only)."""
    seq = sequence.upper()
    L_canon = sum(1 for c in seq if c in SWISSPROT_GLOBAL_2026_01)
    if L_canon == 0:
        return {a: 0.0 for a in SWISSPROT_GLOBAL_2026_01}
    out: dict[str, float] = {}
    for aa in SWISSPROT_GLOBAL_2026_01:
        out[aa] = 100.0 * seq.count(aa) / L_canon
    return out


def aa_z_scores(
    sequence: str,
    reference: str = "swissprot_enzyme_2026_01",
    exclude_aas: str = "",
) -> dict[str, float]:
    """Per-AA z-score: (design_pct - ref_per_seq_mean) / ref_per_seq_std.

    Positive z = AA over-represented in this design vs reference. The
    reference is the per-sequence mean+SD, NOT the global frequency,
    because we're scoring one sequence at a time.

    AAs in ``exclude_aas`` (e.g. "C" when we omit cysteine) are
    returned as 0.0.
    """
    if reference not in REFERENCE_DISTRIBUTIONS:
        raise ValueError(
            f"unknown reference {reference!r}; "
            f"choose from {list(REFERENCE_DISTRIBUTIONS)}"
        )
    ref = REFERENCE_DISTRIBUTIONS[reference]
    pct = aa_composition_pct(sequence)
    excl = set(exclude_aas.upper())
    out: dict[str, float] = {}
    for aa, stats in ref.items():
        if aa in excl:
            out[aa] = 0.0
            continue
        if stats.per_seq_std_pct <= 0:
            out[aa] = 0.0
            continue
        out[aa] = (pct[aa] - stats.per_seq_mean_pct) / stats.per_seq_std_pct
    return out


def out_of_distribution_aas(
    sequence: str,
    *,
    reference: str = "swissprot_enzyme_2026_01",
    z_threshold: float = 2.0,
    direction: str = "high",     # "high" | "low" | "both"
    exclude_aas: str = "",
) -> dict[str, float]:
    """Return AAs whose z-score in ``sequence`` is over ``z_threshold``.

    ``direction``: "high" (over-represented), "low" (under-), "both".
    """
    z = aa_z_scores(sequence, reference=reference, exclude_aas=exclude_aas)
    out: dict[str, float] = {}
    for aa, z_val in z.items():
        if direction in ("high", "both") and z_val > z_threshold:
            out[aa] = z_val
        elif direction in ("low", "both") and z_val < -z_threshold:
            out[aa] = z_val
    return out


__all__ = [
    "AaStats",
    "REFERENCE_DISTRIBUTIONS",
    "SWISSPROT_ENZYME_2026_01",
    "SWISSPROT_GLOBAL_2026_01",
    "aa_composition_pct",
    "aa_z_scores",
    "out_of_distribution_aas",
]
