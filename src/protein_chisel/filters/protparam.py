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


# Side-chain pKa values from Pace 1999 / Bjellqvist 1994 textbook values.
# The "robust" variant (charge_at_pH_full_HH) uses ALL of these: K, R, H
# on the positive side; D, E, C, Y on the negative side; plus N/C termini.
# This is the canonical Henderson-Hasselbalch model.
#
# Three diagnostic variants are also recorded:
#   - charge_at_pH_DE_KR_only: minimalist (just D/E and K/R + termini),
#                              matches the legacy Rosetta NetCharge filter
#   - charge_at_pH_no_HIS:     same as full but EXCLUDES HIS (HIS pKa 6.0
#                              is highly context-dependent in proteins)
#   - charge_at_pH_HIS_half:   no_HIS + 0.5 × n_HIS (assumes HIS pKa ≈ 7.0)
_PKA_POS_FULL = {"K": 10.5, "R": 12.5, "H": 6.0}
_PKA_POS_NO_H = {"K": 10.5, "R": 12.5}
_PKA_POS_DE_KR = {"K": 10.5, "R": 12.5}        # minimalist
_PKA_NEG_FULL = {"D": 3.65, "E": 4.25, "C": 8.3, "Y": 10.5}
_PKA_NEG_DE_KR = {"D": 3.65, "E": 4.25}        # minimalist
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
    # ROBUST default: full Henderson-Hasselbalch on all 7 ionizable
    # residues (K, R, H, D, E, C, Y) + N/C termini. Used as the
    # filter charge in the v2 driver.
    charge_at_pH_full_HH: float = 0.0
    # DIAGNOSTIC: legacy Biopython value (similar to full_HH but
    # uses Bjellqvist 1994 pKa scale; tiny numeric differences).
    charge_at_pH7: float = 0.0
    # DIAGNOSTIC: HH excluding HIS (Cys/Tyr included). Useful when the
    # HIS protonation state is uncertain (metal-coordinating cat-HIS
    # are typically deprotonated at physiological pH regardless of
    # textbook pKa).
    charge_at_pH7_no_HIS: float = 0.0
    # DIAGNOSTIC: no_HIS + 0.5 × n_HIS (assumes HIS pKa ≈ 7.0,
    # half-protonated at pH 7).
    charge_at_pH7_HIS_half: float = 0.0
    # DIAGNOSTIC: minimalist legacy variant — only D/E/K/R + termini
    # (no H/C/Y). Matches the legacy Rosetta NetCharge filter.
    charge_at_pH_DE_KR_only: float = 0.0
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
            f"{prefix}charge_at_pH_full_HH": self.charge_at_pH_full_HH,
            f"{prefix}charge_at_pH7": self.charge_at_pH7,
            f"{prefix}charge_at_pH7_no_HIS": self.charge_at_pH7_no_HIS,
            f"{prefix}charge_at_pH7_HIS_half": self.charge_at_pH7_HIS_half,
            f"{prefix}charge_at_pH_DE_KR_only": self.charge_at_pH_DE_KR_only,
            f"{prefix}flexibility_mean": self.flexibility_mean if self.flexibility_mean is not None else float("nan"),
            f"{prefix}helix_frac_seq": self.helix_frac_seq,
            f"{prefix}turn_frac_seq": self.turn_frac_seq,
            f"{prefix}sheet_frac_seq": self.sheet_frac_seq,
            f"{prefix}extinction_280nm_no_disulfide": self.extinction_280nm_no_disulfide,
            f"{prefix}extinction_280nm_disulfide": self.extinction_280nm_disulfide,
        }


def protparam_metrics(sequence: str, ph: float = 7.8) -> ProtParamResult:
    """Compute Biopython ProtParam metrics + multiple charge variants.

    Default pH 7.8 (close to the user's typical PTE assay buffer at
    pH 8.0, slightly under for safety margin).

    Returns five charge variants — pick whichever fits your context:
      - ``charge_at_pH_full_HH``   ROBUST default: HH on K/R/H/D/E/C/Y + termini
      - ``charge_at_pH7``          Biopython HH (Bjellqvist 1994 pKas)
      - ``charge_at_pH7_no_HIS``   HH excluding HIS (cat-HIS coord metals)
      - ``charge_at_pH7_HIS_half`` no_HIS + 0.5 × n_HIS (HIS pKa ≈ 7.0)
      - ``charge_at_pH_DE_KR_only`` minimalist legacy (D/E + K/R + termini)

    The "pH7" suffix on three fields is historical (predates this rewrite);
    the actual pH used in all five is whatever ``ph`` is passed.
    """
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
        charge_at_pH_full_HH=float(_charge_at_ph_full_hh(canonical, ph)),
        charge_at_pH7=float(pa.charge_at_pH(ph)),
        charge_at_pH7_no_HIS=float(_charge_at_ph_no_his(canonical, ph)),
        charge_at_pH7_HIS_half=float(
            _charge_at_ph_no_his(canonical, ph)
            + 0.5 * canonical.count("H")
        ),
        charge_at_pH_DE_KR_only=float(_charge_at_ph_de_kr_only(canonical, ph)),
        flexibility_mean=flex_mean,
        helix_frac_seq=float(helix),
        turn_frac_seq=float(turn),
        sheet_frac_seq=float(sheet),
        extinction_280nm_no_disulfide=float(ext_no),
        extinction_280nm_disulfide=float(ext_with),
    )


def _charge_at_ph_full_hh(sequence: str, ph: float) -> float:
    """Henderson-Hasselbalch net charge — ROBUST: all 7 ionizables + termini.

    Includes K, R, H on the positive side; D, E, C, Y on the negative
    side; plus N/C termini. Standard textbook model; what most
    publications use as "net charge at pH X."
    """
    n_pos = sum(_hh_pos(ph, _PKA_POS_FULL[aa]) for aa in sequence if aa in _PKA_POS_FULL)
    n_neg = sum(_hh_neg(ph, _PKA_NEG_FULL[aa]) for aa in sequence if aa in _PKA_NEG_FULL)
    nterm = _hh_pos(ph, _NTERM_PKA)
    cterm = _hh_neg(ph, _CTERM_PKA)
    return n_pos + nterm - n_neg - cterm


def _charge_at_ph_no_his(sequence: str, ph: float) -> float:
    """Henderson-Hasselbalch net charge excluding histidine.

    Excludes H (uncertain pKa in proteins). Includes K, R on the
    positive side; D, E, C, Y on the negative side; plus N/C termini.
    """
    n_pos = sum(_hh_pos(ph, _PKA_POS_NO_H[aa]) for aa in sequence if aa in _PKA_POS_NO_H)
    n_neg = sum(_hh_neg(ph, _PKA_NEG_FULL[aa]) for aa in sequence if aa in _PKA_NEG_FULL)
    nterm = _hh_pos(ph, _NTERM_PKA)
    cterm = _hh_neg(ph, _CTERM_PKA)
    return n_pos + nterm - n_neg - cterm


def _charge_at_ph_de_kr_only(sequence: str, ph: float) -> float:
    """Minimalist legacy net charge: ONLY D, E, K, R + termini.

    Matches the legacy Rosetta NetCharge filter convention used in
    older lab scripts (no HIS, no Cys, no Tyr). Useful for back-compat
    comparison with prior runs.
    """
    n_pos = sum(_hh_pos(ph, _PKA_POS_DE_KR[aa]) for aa in sequence if aa in _PKA_POS_DE_KR)
    n_neg = sum(_hh_neg(ph, _PKA_NEG_DE_KR[aa]) for aa in sequence if aa in _PKA_NEG_DE_KR)
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
