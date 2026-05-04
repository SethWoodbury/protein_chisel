"""Swiss-Prot amino-acid composition reference data + checkers.

Loads a richer baseline JSON (data/aa_composition_baselines_2026_01.json)
covering:

1. ``swissprot_reviewed_2026_01`` — all reviewed Swiss-Prot proteins
2. ``swissprot_reviewed_ec_annotated_2026_01`` — EC-annotated subset
3. ``ec_class_1_oxidoreductases_2026_01``
4. ``ec_class_2_transferases_2026_01``
5. ``ec_class_3_hydrolases_2026_01``    (PTE = EC 3.1.8.1)
6. ``ec_class_4_lyases_2026_01``

Each carries per-AA global %, per-sequence mean %, and per-sequence
SD %. Use the most-specific baseline available for the design's EC
class.

Default thresholds (from CODEX_IMPLEMENTATION_NOTES in the source JSON):
- WARN at |z| > 2.0
- FAIL at |z| > 3.0 AND |log2 enrichment| > 0.25

The expected use is:
  - For a given designed sequence, compute its AA composition.
  - For each AA, compute the z-score against the EC-matched
    distribution.
  - Flag AAs whose abs(z) exceeds a per-rule threshold.

De-novo designs are EXPECTED to drift from these distributions a bit,
so the threshold is permissive by default (z=2.0).

Cysteine is intentionally excluded from over-representation checks
when ``omit_AA`` includes "C" because the design protocol forbids it
explicitly.

Source: Swiss-Prot release 2026_01. n=574,627 proteins (reviewed) /
n=279,501 (EC-annotated) / n=64,973 (hydrolases) etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------


_DATA_FILE = Path(__file__).parent / "data" / "aa_composition_baselines_2026_01.json"
AA_ORDER_REF = "ACDEFGHIKLMNPQRSTVWY"


def _load_baselines_json() -> dict:
    """Read the bundled JSON of all reference distributions."""
    with open(_DATA_FILE) as fh:
        return json.load(fh)


_RAW = _load_baselines_json()
DEFAULT_THRESHOLDS = _RAW["CODEX_IMPLEMENTATION_NOTES"]["suggested_default_thresholds"]
# {"warn_if_abs_z_gt": 2.0, "fail_if_abs_z_gt": 3.0,
#  "also_require_abs_log2_enrichment_gt_for_fail": 0.25, ...}


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


def _parse_baselines_from_json() -> tuple[dict[str, dict[str, AaStats]], dict[str, dict]]:
    """Parse the JSON's AMINO_ACID_BASELINES into the AaStats structure.

    Returns (distributions, metadata) where:
      distributions[name] = {AA: AaStats}
      metadata[name] = {"baseline_type", "n_sequences", "ec_class", ...}
    """
    out: dict[str, dict[str, AaStats]] = {}
    meta: dict[str, dict] = {}
    raw = _RAW.get("AMINO_ACID_BASELINES", {})
    for name, data in raw.items():
        baseline_type = data.get("baseline_type")
        if baseline_type not in ("sequence_distribution", "global_only"):
            continue
        gpct = data.get("global_pct", {})
        mean = data.get("per_sequence_mean_pct", {})
        sd = data.get("per_sequence_sd_pct", {})
        d: dict[str, AaStats] = {}
        for aa in AA_ORDER_REF:
            d[aa] = AaStats(
                global_pct=float(gpct.get(aa, 0.0)),
                per_seq_mean_pct=float(mean.get(aa, 0.0)) if mean else 0.0,
                per_seq_std_pct=float(sd.get(aa, 0.0)) if sd else 0.0,
            )
        out[name] = d
        meta[name] = {
            "baseline_type": baseline_type,
            "n_sequences": data.get("n_sequences"),
            "ec_class": data.get("ec_class"),
            "ec_class_name": data.get("ec_class_name"),
            "statistical_qc_allowed": data.get("statistical_qc_allowed", False),
            "release": data.get("release"),
        }
    return out, meta


_LOADED_DISTS, BASELINE_METADATA = _parse_baselines_from_json()
# Alias the JSON's longer keys to short names that match the existing API
_ALIASES = {
    "swissprot_reviewed_2026_01": "uniprotkb_swissprot_reviewed_2026_01",
    "swissprot_enzyme_full_2026_01": "uniprotkb_swissprot_reviewed_ec_annotated_2026_01",
    "swissprot_ec1_oxidoreductases_2026_01": "uniprotkb_swissprot_reviewed_ec_class_1_oxidoreductases_2026_01",
    "swissprot_ec2_transferases_2026_01": "uniprotkb_swissprot_reviewed_ec_class_2_transferases_2026_01",
    "swissprot_ec3_hydrolases_2026_01": "uniprotkb_swissprot_reviewed_ec_class_3_hydrolases_2026_01",
    "swissprot_ec4_lyases_2026_01": "uniprotkb_swissprot_reviewed_ec_class_4_lyases_2026_01",
}
for short, long_name in _ALIASES.items():
    if long_name in _LOADED_DISTS:
        REFERENCE_DISTRIBUTIONS[short] = _LOADED_DISTS[long_name]
        BASELINE_METADATA[short] = BASELINE_METADATA[long_name]


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


def aa_log2_enrichment(
    sequence: str,
    reference: str = "swissprot_ec3_hydrolases_2026_01",
    exclude_aas: str = "",
) -> dict[str, float]:
    """Per-AA log2 enrichment vs the reference distribution's global %.

    Returns log2(design_pct / ref_global_pct) per AA. Used as a
    secondary check beyond z-score (per the JSON's hard rules:
    fail requires abs(z)>3 AND abs(log2_enrichment)>0.25).
    """
    import math
    if reference not in REFERENCE_DISTRIBUTIONS:
        raise ValueError(f"unknown reference {reference!r}")
    ref = REFERENCE_DISTRIBUTIONS[reference]
    pct = aa_composition_pct(sequence)
    excl = set(exclude_aas.upper())
    out: dict[str, float] = {}
    for aa, stats in ref.items():
        if aa in excl or stats.global_pct <= 0 or pct[aa] <= 0:
            out[aa] = 0.0
            continue
        out[aa] = math.log2(pct[aa] / stats.global_pct)
    return out


def aa_quality_check(
    sequence: str,
    reference: str = "swissprot_ec3_hydrolases_2026_01",
    exclude_aas: str = "C",
    warn_z: Optional[float] = None,
    fail_z: Optional[float] = None,
    fail_log2_enrichment: Optional[float] = None,
) -> dict:
    """Combined warn/fail check using both z-score AND log2 enrichment.

    Per the JSON's hard rules: only z-score is reliable when the
    baseline has per-sequence SD (i.e. ``baseline_type == 'sequence_
    distribution'``). For ``global_only`` baselines fall back to the
    log2 enrichment metric.

    Returns dict with:
        warn_aas:  AAs with abs(z) > warn_z
        fail_aas:  AAs with abs(z) > fail_z AND abs(log2) > fail_log2
        z_scores:  dict[AA -> z]
        log2_enrichment: dict[AA -> log2 enrichment]
    """
    warn_z = warn_z if warn_z is not None else float(DEFAULT_THRESHOLDS["warn_if_abs_z_gt"])
    fail_z = fail_z if fail_z is not None else float(DEFAULT_THRESHOLDS["fail_if_abs_z_gt"])
    fail_log2 = fail_log2_enrichment if fail_log2_enrichment is not None else float(
        DEFAULT_THRESHOLDS["also_require_abs_log2_enrichment_gt_for_fail"]
    )
    z = aa_z_scores(sequence, reference=reference, exclude_aas=exclude_aas)
    enr = aa_log2_enrichment(sequence, reference=reference, exclude_aas=exclude_aas)
    warn = {aa: zv for aa, zv in z.items() if abs(zv) > warn_z}
    fail = {
        aa: zv for aa, zv in z.items()
        if abs(zv) > fail_z and abs(enr.get(aa, 0)) > fail_log2
    }
    return {
        "warn_aas": warn,
        "fail_aas": fail,
        "z_scores": z,
        "log2_enrichment": enr,
        "reference": reference,
        "thresholds": {"warn_z": warn_z, "fail_z": fail_z, "fail_log2": fail_log2},
    }


__all__ = [
    "AA_ORDER_REF",
    "AaStats",
    "BASELINE_METADATA",
    "DEFAULT_THRESHOLDS",
    "REFERENCE_DISTRIBUTIONS",
    "SWISSPROT_ENZYME_2026_01",
    "SWISSPROT_GLOBAL_2026_01",
    "aa_composition_pct",
    "aa_log2_enrichment",
    "aa_quality_check",
    "aa_z_scores",
    "out_of_distribution_aas",
]
