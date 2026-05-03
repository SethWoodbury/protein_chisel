"""Rule base class + registry + StructureContext.

A ``Rule`` is a callable evaluator for ONE expression-risk pattern.
Rules are *registered* at import time into a process-wide ``REGISTRY``
so that ``ExpressionRuleEngine`` can iterate without a hard-coded list.

Each rule is structurally pure: it reads a ``StructureContext`` (sequence
+ optional SS + SASA + position-class metadata) plus an
``ExpressionProfile`` (host + tags + per-rule overrides) and returns a
list of ``RuleHit``. The engine handles aggregation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from protein_chisel.expression.severity import RuleHit, Severity


if TYPE_CHECKING:
    from protein_chisel.expression.profiles import ExpressionProfile


LOGGER = logging.getLogger("protein_chisel.expression.rules")


@dataclass
class StructureContext:
    """Per-sequence context fed to every rule evaluation.

    ``sequence`` is the designed body — NOT the full expressed construct
    (i.e. tags excluded). All position indices in ``RuleHit.start/end``
    are against this body-relative coordinate system.

    ``protein_resnos`` (optional) maps body index -> PDB resseq for use
    by the engine when emitting ``--omit_AA_per_residue_multi`` JSONs.
    Without it, we assume body[i] -> chain+(i+1).
    """
    sequence: str
    ss_reduced: Optional[str] = None       # length L, "HEL"
    ss_full: Optional[str] = None
    ss_confidence: Optional[np.ndarray] = None
    sasa: Optional[np.ndarray] = None
    position_class: Optional[list[str]] = None
    catalytic_resnos_zero_idx: set[int] = field(default_factory=set)
    fixed_resnos_zero_idx: set[int] = field(default_factory=set)
    protein_resnos: Optional[list[int]] = None
    n_terminal_pair: Optional[tuple[str, str]] = None  # post-MetAP
    has_structure: bool = False

    @property
    def L(self) -> int:
        return len(self.sequence)


class Rule(ABC):
    """Base class for one expression-risk rule.

    Subclasses must define ``name``, ``default_severity``, optionally
    ``requires_structure``, and implement ``evaluate``. Severity returned
    in each ``RuleHit`` should already reflect any profile override or
    structure-aware modulation; the engine just aggregates.
    """
    name: str = ""
    default_severity: Severity = Severity.WARN_ONLY
    requires_structure: bool = False
    applies_to_hosts: tuple[str, ...] = ("*",)
    applies_to_compartments: tuple[str, ...] = ("*",)

    def is_active(self, profile: "ExpressionProfile") -> bool:
        """Whether this rule should run under the given profile."""
        if "*" not in self.applies_to_hosts and profile.host not in self.applies_to_hosts:
            return False
        if "*" not in self.applies_to_compartments and profile.compartment not in self.applies_to_compartments:
            return False
        return True

    def resolved_severity(self, profile: "ExpressionProfile") -> Severity:
        """Default severity after applying profile-level override + preset."""
        sev = self.default_severity
        if profile.preset == "strict":
            sev = sev.promote()
        elif profile.preset == "permissive":
            sev = sev.demote()
        if self.name in profile.severity_overrides:
            sev = profile.severity_overrides[self.name]
        return sev

    @abstractmethod
    def evaluate(
        self, ctx: StructureContext, profile: "ExpressionProfile",
    ) -> list[RuleHit]:
        """Return list of hits. Empty list = rule did not fire."""


class RuleRegistry:
    """Process-wide registry of Rule subclasses.

    Rules are typically registered at import time of
    ``builtin_rules.py``. Tests can clear / override.
    """
    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def register(self, rule: Rule) -> Rule:
        if not rule.name:
            raise ValueError(f"rule has empty name: {rule!r}")
        if rule.name in self._rules:
            LOGGER.warning("re-registering rule %r", rule.name)
        self._rules[rule.name] = rule
        return rule

    def all(self) -> list[Rule]:
        return list(self._rules.values())

    def by_name(self, name: str) -> Rule:
        return self._rules[name]

    def names(self) -> list[str]:
        return list(self._rules.keys())

    def clear(self) -> None:
        self._rules.clear()


REGISTRY = RuleRegistry()


__all__ = [
    "REGISTRY",
    "Rule",
    "RuleRegistry",
    "StructureContext",
]
