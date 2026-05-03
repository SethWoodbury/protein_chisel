"""Severity levels + RuleHit dataclass for the expression rule engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Severity(IntEnum):
    """Severity of a rule firing.

    Order matters: higher value = stronger action. ``promote`` /
    ``demote`` shift by one notch. Used for the strict/standard/permissive
    profile presets.
    """
    WARN_ONLY = 0    # log, record in metrics, no MPNN/filter action
    SOFT_BIAS = 1    # downweight AAs in the MPNN per-residue bias
    HARD_OMIT = 2    # hard-omit AAs at sample time (--omit_AA_per_residue)
    HARD_FILTER = 3  # reject the entire sequence post-sampling

    def promote(self) -> "Severity":
        return Severity(min(int(self) + 1, int(Severity.HARD_FILTER)))

    def demote(self) -> "Severity":
        return Severity(max(int(self) - 1, int(Severity.WARN_ONLY)))


@dataclass(frozen=True)
class RuleHit:
    """One firing of a rule against a sequence.

    All position fields are 0-indexed against the **designed body**
    sequence (not full construct including tags).

    ``severity`` carries the *resolved* severity (after profile overrides
    + structure-aware modulation), not the rule's default.
    """
    rule_name: str
    severity: Severity
    start: int                 # 0-indexed inclusive
    end: int                   # 0-indexed exclusive
    matched: str               # actual matched substring (or "" for non-regex)
    reason: str                # human-readable
    suggested_omit_AAs: str = ""   # AAs to forbid/downweight at hit positions
    metadata: dict = field(default_factory=dict)

    def positions(self) -> range:
        """0-indexed positions covered by this hit."""
        return range(self.start, self.end)


__all__ = ["RuleHit", "Severity"]
