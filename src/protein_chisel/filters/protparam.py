"""Biopython ProtParam-derived sequence filters.

Pure-sequence; no PyRosetta needed. Wraps Biopython's
``ProteinAnalysis`` and adds the ``charge_at_pH7_no_HIS`` variant which
matches the legacy Rosetta ``NetCharge`` filter (excludes histidine).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Biopython is in esmc.sif (pulled in by `esm` deps), pyrosetta.sif may not
# have it; tools that filter on host should use TSV/parquet outputs.
from Bio.SeqUtils.ProtParam import ProteinAnalysis  # type: ignore[import-not-found]


# Side-chain pKa values used for the no-HIS variant. The standard
# ProteinAnalysis.charge_at_pH includes histidine (pKa ~6.0), which makes
# net charge depend strongly on assumed protonation. The no-HIS variant
# matches Rosetta's NetCharge filter behavior used in our legacy scripts.
_PKA_POS = {"K": 10.5, "R": 12.5}    # excludes H
_PKA_NEG = {"D": 3.65, "E": 4.25}
_NTERM_PKA = 9.0
_CTERM_PKA = 2.0


@dataclass
class ProtParamResult:
    sequence: str
    length: int
    molecular_weight: float
    pi: float
    instability_index: float
    gravy: float
    aromaticity: float
    charge_at_pH7: float
    charge_at_pH7_no_HIS: float
    flexibility_mean: Optional[float] = None
    helix_frac_seq: float = 0.0  # ProtParam's secondary-structure-from-sequence
    turn_frac_seq: float = 0.0
    sheet_frac_seq: float = 0.0
    extinction_280nm_no_disulfide: float = 0.0
    extinction_280nm_disulfide: float = 0.0

    def to_dict(self, prefix: str = "protparam__") -> dict[str, float | int | str]:
        return {
            f"{prefix}length": self.length,
            f"{prefix}molecular_weight": self.molecular_weight,
            f"{prefix}pi": self.pi,
            f"{prefix}instability_index": self.instability_index,
            f"{prefix}gravy": self.gravy,
            f"{prefix}aromaticity": self.aromaticity,
            f"{prefix}charge_at_pH7": self.charge_at_pH7,
            f"{prefix}charge_at_pH7_no_HIS": self.charge_at_pH7_no_HIS,
            f"{prefix}flexibility_mean": self.flexibility_mean if self.flexibility_mean is not None else float("nan"),
            f"{prefix}helix_frac_seq": self.helix_frac_seq,
            f"{prefix}turn_frac_seq": self.turn_frac_seq,
            f"{prefix}sheet_frac_seq": self.sheet_frac_seq,
            f"{prefix}extinction_280nm_no_disulfide": self.extinction_280nm_no_disulfide,
            f"{prefix}extinction_280nm_disulfide": self.extinction_280nm_disulfide,
        }


def protparam_metrics(sequence: str, ph: float = 7.0) -> ProtParamResult:
    """Compute Biopython ProtParam metrics + charge_at_pH7_no_HIS variant."""
    seq = sequence.replace("*", "").upper()
    # ProtParam fails on non-canonical AAs. Strip them; warn implicitly via
    # length difference if the caller cares.
    canonical = "".join(c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY")
    if not canonical:
        raise ValueError("sequence is empty after stripping non-canonical AAs")

    pa = ProteinAnalysis(canonical)
    helix, turn, sheet = pa.secondary_structure_fraction()

    # Flexibility — list of float per residue; use mean. Some short sequences
    # raise; default to None on failure.
    try:
        flex = pa.flexibility()
        flex_mean = sum(flex) / len(flex) if flex else None
    except Exception:
        flex_mean = None

    # Extinction at 280 nm (per molar). Two values: oxidized (with all Cys
    # in disulfides) and reduced (no disulfides).
    try:
        ext_no, ext_with = pa.molar_extinction_coefficient()
    except Exception:
        ext_no, ext_with = 0.0, 0.0

    return ProtParamResult(
        sequence=canonical,
        length=len(canonical),
        molecular_weight=float(pa.molecular_weight()),
        pi=float(pa.isoelectric_point()),
        instability_index=float(pa.instability_index()),
        gravy=float(pa.gravy()),
        aromaticity=float(pa.aromaticity()),
        charge_at_pH7=float(pa.charge_at_pH(ph)),
        charge_at_pH7_no_HIS=float(_charge_at_ph_no_his(canonical, ph)),
        flexibility_mean=flex_mean,
        helix_frac_seq=float(helix),
        turn_frac_seq=float(turn),
        sheet_frac_seq=float(sheet),
        extinction_280nm_no_disulfide=float(ext_no),
        extinction_280nm_disulfide=float(ext_with),
    )


def _charge_at_ph_no_his(sequence: str, ph: float) -> float:
    """Henderson-Hasselbalch net charge excluding histidine.

    Matches the legacy Rosetta NetCharge filter convention used by the
    lab. HIS is excluded because its pKa (~6.0) lies near physiological
    pH and makes the value highly sensitive to local environment.
    """
    n_pos = sum(_hh_pos(ph, _PKA_POS[aa]) for aa in sequence if aa in _PKA_POS)
    n_neg = sum(_hh_neg(ph, _PKA_NEG[aa]) for aa in sequence if aa in _PKA_NEG)
    nterm = _hh_pos(ph, _NTERM_PKA)
    cterm = _hh_neg(ph, _CTERM_PKA)
    return n_pos + nterm - n_neg - cterm


def _hh_pos(ph: float, pka: float) -> float:
    """Fraction of a positive side chain that's protonated."""
    return 1.0 / (1.0 + 10 ** (ph - pka))


def _hh_neg(ph: float, pka: float) -> float:
    """Fraction of a negative side chain that's deprotonated (carries charge)."""
    return 1.0 / (1.0 + 10 ** (pka - ph))


__all__ = ["ProtParamResult", "protparam_metrics"]
