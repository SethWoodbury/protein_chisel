"""AA-class-balanced compensatory bias for MPNN sampling.

When the AA composition of a design pool drifts within a class — e.g.
E (negative) is z=+5 over-represented while D (also negative) is z=-2
under-represented — naively suppressing E (the SOFT_BIAS rule's default
behavior) just reduces total negative charge. The user's better
approach: SWAP within the class. Push down E AND pull up D.

This module computes a global ``bias_AA`` string suitable for
LigandMPNN's ``--bias_AA`` argument. Bias values are nats:
   negative bias = discourage that AA
   positive bias = encourage that AA

Classes (AAs may belong to multiple):

  hydrophobic_aliphatic  : A, V, L, I, M, C
  aromatic               : F, W, Y, H
  negatively_charged     : D, E
  positively_charged     : K, R, H
  polar_uncharged        : S, T, N, Q, Y, C, H
  small                  : A, G, S, C, T
  proline_special        : P
  glycine_special        : G

For each class with >=2 members, we compute z-scores against the
reference distribution and apply compensatory bias *within the class*
when at least one member is over by more than ``balance_z_threshold``
AND another member is under by the same threshold.

Bias magnitude scales linearly with z, capped at ``max_bias_nats``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from protein_chisel.expression.aa_composition import (
    aa_z_scores, REFERENCE_DISTRIBUTIONS,
)


LOGGER = logging.getLogger("protein_chisel.expression.aa_class_balance")


AA_CLASSES: dict[str, list[str]] = {
    "hydrophobic_aliphatic": ["A", "V", "L", "I", "M", "C"],
    "aromatic":              ["F", "W", "Y", "H"],
    "negatively_charged":    ["D", "E"],
    "positively_charged":    ["K", "R", "H"],
    "polar_uncharged":       ["S", "T", "N", "Q", "Y", "C", "H"],
    "small":                 ["A", "G", "S", "C", "T"],
    # Single-member classes (no swap available)
    "proline_special":       ["P"],
    "glycine_special":       ["G"],
}


@dataclass
class AaBalanceTelemetry:
    bias_AA_string: str = ""
    per_aa_bias: dict[str, float] = field(default_factory=dict)
    swaps: list[dict] = field(default_factory=list)
    z_scores: dict[str, float] = field(default_factory=dict)
    reference: str = "swissprot_ec3_hydrolases_2026_01"

    def to_dict(self) -> dict:
        return {
            "bias_AA_string": self.bias_AA_string,
            "per_aa_bias": dict(self.per_aa_bias),
            "swaps": self.swaps,
            "z_scores": {k: round(v, 2) for k, v in self.z_scores.items()},
            "reference": self.reference,
        }


def compute_class_balanced_bias_AA(
    sequence: str,
    *,
    reference: str = "swissprot_ec3_hydrolases_2026_01",
    exclude_aas: str = "C",
    balance_z_threshold: float = 2.0,
    over_z_threshold: float = 3.0,
    max_bias_nats: float = 2.5,
    bias_per_z: float = 0.4,
) -> AaBalanceTelemetry:
    """Build a ``bias_AA`` string by class-balanced compensatory weights.

    Strategy per class:
      - Compute z-scores of every class member.
      - Identify the most over-represented (z_max) and most under-
        represented (z_min) members.
      - If z_max > balance_z_threshold AND z_min < -balance_z_threshold:
        downweight the over-rep AA and upweight the under-rep AA,
        each by ``bias_per_z * z`` (capped at +-max_bias_nats).
      - Singleton classes (P, G) are handled separately: only downweight
        if z > over_z_threshold (no swap partner available).

    Final per-AA bias is the SUM of contributions from all classes the
    AA belongs to (an AA in multiple classes can be biased multiple
    times, but the result is clamped at +-max_bias_nats).

    Returns:
      AaBalanceTelemetry with the assembled bias_AA string ready to
      pass to LigandMPNNConfig.bias_AA.
    """
    if reference not in REFERENCE_DISTRIBUTIONS:
        raise ValueError(f"unknown reference {reference!r}")
    excluded = set(exclude_aas.upper())

    z = aa_z_scores(sequence, reference=reference, exclude_aas=exclude_aas)
    per_aa_bias: dict[str, float] = {}
    swaps: list[dict] = []

    def _add(aa: str, delta: float):
        per_aa_bias[aa] = per_aa_bias.get(aa, 0.0) + delta

    for class_name, members in AA_CLASSES.items():
        eligible = [m for m in members if m not in excluded]
        if not eligible:
            continue
        if len(eligible) == 1:
            # Singleton class: just suppress if extreme.
            aa = eligible[0]
            zv = z.get(aa, 0.0)
            if zv > over_z_threshold:
                _add(aa, -min(max_bias_nats, bias_per_z * zv))
            continue
        # Multi-member class: find max-z and min-z within the class.
        zs = [(m, z.get(m, 0.0)) for m in eligible]
        zs.sort(key=lambda t: t[1])
        low_aa, low_z = zs[0]
        high_aa, high_z = zs[-1]
        if high_z > balance_z_threshold and low_z < -balance_z_threshold:
            down_mag = min(max_bias_nats, bias_per_z * high_z)
            up_mag = min(max_bias_nats, bias_per_z * abs(low_z))
            _add(high_aa, -down_mag)
            _add(low_aa, +up_mag)
            swaps.append({
                "class": class_name,
                "down_aa": high_aa, "down_z": round(high_z, 2),
                "up_aa": low_aa, "up_z": round(low_z, 2),
                "down_bias": round(-down_mag, 3),
                "up_bias": round(up_mag, 3),
            })

    # Clamp per-AA bias.
    for aa in list(per_aa_bias.keys()):
        per_aa_bias[aa] = max(-max_bias_nats, min(max_bias_nats, per_aa_bias[aa]))

    # Build "AA:val,AA:val" string. Drop near-zeros.
    parts = [
        f"{aa}:{val:.2f}"
        for aa, val in sorted(per_aa_bias.items())
        if abs(val) > 0.05
    ]
    bias_str = ",".join(parts)

    return AaBalanceTelemetry(
        bias_AA_string=bias_str,
        per_aa_bias=per_aa_bias,
        swaps=swaps,
        z_scores=dict(z),
        reference=reference,
    )


__all__ = [
    "AA_CLASSES",
    "AaBalanceTelemetry",
    "compute_class_balanced_bias_AA",
]
