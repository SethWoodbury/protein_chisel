"""Detect a pathologically hydrophobic / over-represented INPUT scaffold (seed triage).

The PLM fusion bias is conditioned on the seed, so on a pathological seed it AMPLIFIES
the bad composition — at the operating temperature (~0.15) the PLM mean (~1.36 nats) is
an ~8700x odds lock (see :mod:`protein_chisel.sampling.bias_scale`). Empirically, dropping
the PLM (``plm_strength=0``) on such a seed took GRAVY 1.34 -> -0.50 and Ala 27% -> 0.5%.

``assess_seed`` is the PURE detector. The opt-in policy that consumes it (e.g. forcing
``plm_strength=0`` for a run) lives in the driver, so this stays trivially unit-testable
and reference-free — it generalises to any scaffold without a per-class baseline.
"""
from __future__ import annotations

import dataclasses

# Hydrophobic residues for the hydrophobic-fraction trip (the classic aggregation-prone
# aliphatic + aromatic set; Cys included — hydrophobic and usually omitted anyway).
DEFAULT_HYDROPHOBIC_AAS = "AVLIMFWC"


@dataclasses.dataclass(frozen=True)
class SeedAssessment:
    """Result of :func:`assess_seed`.

    ``pathological`` is the actionable bit; ``reasons`` is a human-readable list (empty
    iff not pathological) for logging; the rest are the measured values.
    """
    pathological: bool
    reasons: list[str]
    gravy: float
    max_aa: str
    max_aa_frac: float
    hydrophobic_frac: float


def assess_seed(
    sequence: str,
    gravy: float,
    *,
    gravy_max: float = 0.4,
    max_aa_frac: float = 0.16,
    hydrophobic_frac_max: float = 0.50,
    hydrophobic_aas: str = DEFAULT_HYDROPHOBIC_AAS,
) -> SeedAssessment:
    """Flag a seed as pathological if its GRAVY, single-AA over-representation, or
    hydrophobic fraction exceed the given thresholds.

    Reference-free (raw observed values, no per-class baseline). ``gravy`` is supplied by
    the caller (already computed via protparam); composition is derived from ``sequence``.
    Any tripped threshold sets ``pathological=True`` and appends a reason. Defaults:
    GRAVY>0.4 (matches the existing input-hydrophobicity warning + the cycle band),
    any single AA >=16%, or >50% hydrophobic.
    """
    from protein_chisel.expression.aa_composition import aa_composition_pct
    pct = aa_composition_pct(sequence)                     # {AA: percent 0-100}
    if pct:
        max_aa, max_pct = max(pct.items(), key=lambda kv: kv[1])
    else:
        max_aa, max_pct = "", 0.0
    max_aa_frac_val = max_pct / 100.0
    hydro = {c for c in hydrophobic_aas.upper()}
    hydrophobic_frac = sum(p for a, p in pct.items() if a in hydro) / 100.0

    reasons: list[str] = []
    if gravy > gravy_max:
        reasons.append(f"GRAVY {gravy:+.2f} > {gravy_max:+.2f}")
    if max_aa_frac_val >= max_aa_frac:
        reasons.append(
            f"{max_aa!r} {max_aa_frac_val * 100:.1f}% >= {max_aa_frac * 100:.0f}% "
            f"(single-AA over-representation)")
    if hydrophobic_frac > hydrophobic_frac_max:
        reasons.append(
            f"hydrophobic fraction {hydrophobic_frac * 100:.1f}% > "
            f"{hydrophobic_frac_max * 100:.0f}%")

    return SeedAssessment(
        pathological=bool(reasons),
        reasons=reasons,
        gravy=float(gravy),
        max_aa=max_aa,
        max_aa_frac=max_aa_frac_val,
        hydrophobic_frac=hydrophobic_frac,
    )


def should_skip_plm(
    assessment: SeedAssessment | None,
    *,
    enabled: bool,
    current_plm_strength: float,
) -> bool:
    """Policy: skip the PLM bias (force ``plm_strength=0``) iff the triage is enabled,
    the seed is pathological, and the PLM is currently on (>0).

    Pure so the driver glue is a one-liner: ``if should_skip_plm(...): plm_strength = 0``.
    Returns ``False`` for a missing assessment (triage failed → leave the run unchanged).
    """
    return bool(
        enabled
        and assessment is not None
        and assessment.pathological
        and current_plm_strength > 0
    )
