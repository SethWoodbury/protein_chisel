"""ExpressionRuleEngine — the orchestrator that runs all rules and
aggregates their hits into a single EngineResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from protein_chisel.expression.profiles import ExpressionProfile
from protein_chisel.expression.rules import (
    REGISTRY, Rule, RuleRegistry, StructureContext,
)
from protein_chisel.expression.severity import RuleHit, Severity


LOGGER = logging.getLogger("protein_chisel.expression.engine")


@dataclass
class EngineResult:
    sequence: str
    profile_name: str
    hits: list[RuleHit] = field(default_factory=list)

    @property
    def hard_filter_hits(self) -> list[RuleHit]:
        return [h for h in self.hits if h.severity == Severity.HARD_FILTER]

    @property
    def hard_omit_hits(self) -> list[RuleHit]:
        return [h for h in self.hits if h.severity == Severity.HARD_OMIT]

    @property
    def soft_bias_hits(self) -> list[RuleHit]:
        return [h for h in self.hits if h.severity == Severity.SOFT_BIAS]

    @property
    def warnings(self) -> list[RuleHit]:
        return [h for h in self.hits if h.severity == Severity.WARN_ONLY]

    def passes_hard_filter(self) -> bool:
        return not self.hard_filter_hits

    def hard_omit_per_residue(self) -> dict[int, str]:
        """0-indexed body position -> AAs to forbid."""
        out: dict[int, str] = {}
        for h in self.hard_omit_hits:
            for pos in h.positions():
                if h.suggested_omit_AAs:
                    cur = set(out.get(pos, ""))
                    cur.update(h.suggested_omit_AAs)
                    out[pos] = "".join(sorted(cur))
        return out

    def soft_bias_per_residue(self) -> dict[int, str]:
        """0-indexed body position -> AAs to downweight."""
        out: dict[int, str] = {}
        for h in self.soft_bias_hits:
            for pos in h.positions():
                if h.suggested_omit_AAs:
                    cur = set(out.get(pos, ""))
                    cur.update(h.suggested_omit_AAs)
                    out[pos] = "".join(sorted(cur))
        return out

    def to_omit_AA_json(
        self,
        chain: str,
        protein_resnos: Optional[list[int]] = None,
    ) -> dict[str, str]:
        """Render hard_omit_per_residue into the fused_mpnn JSON format
        ``{"<chain><resno>": "AAs"}``.

        ``protein_resnos`` maps body index -> PDB resseq. When omitted,
        body[i] -> ``<chain><i+1>``.
        """
        omit = self.hard_omit_per_residue()
        out: dict[str, str] = {}
        for body_pos, aas in omit.items():
            resno = protein_resnos[body_pos] if protein_resnos else body_pos + 1
            out[f"{chain}{resno}"] = aas
        return out

    def fail_reasons(self) -> list[str]:
        return [h.reason for h in self.hard_filter_hits]

    def summary(self) -> dict:
        return {
            "n_hits": len(self.hits),
            "n_hard_filter": len(self.hard_filter_hits),
            "n_hard_omit": len(self.hard_omit_hits),
            "n_soft_bias": len(self.soft_bias_hits),
            "n_warnings": len(self.warnings),
            "passes_hard_filter": self.passes_hard_filter(),
            "fail_reasons": self.fail_reasons(),
        }


class ExpressionRuleEngine:
    def __init__(
        self,
        profile: ExpressionProfile,
        registry: RuleRegistry = REGISTRY,
    ) -> None:
        self.profile = profile
        self.registry = registry

    def evaluate(
        self,
        sequence: str,
        *,
        ss_reduced: Optional[str] = None,
        ss_full: Optional[str] = None,
        ss_confidence: Optional[np.ndarray] = None,
        sasa: Optional[np.ndarray] = None,
        position_class: Optional[list[str]] = None,
        catalytic_resnos: Iterable[int] = (),
        fixed_resnos: Iterable[int] = (),
        protein_resnos: Optional[list[int]] = None,
    ) -> EngineResult:
        """Evaluate every active rule against the sequence.

        ``catalytic_resnos`` / ``fixed_resnos`` are PDB resseqs (1-indexed).
        We translate to 0-indexed body positions via ``protein_resnos``
        (if provided) or by assuming body[i] = resseq i+1.
        """
        seq = sequence.upper()
        L = len(seq)

        # Translate PDB resnos -> 0-indexed body positions
        if protein_resnos is not None:
            resno_to_idx = {r: i for i, r in enumerate(protein_resnos)}
        else:
            resno_to_idx = {i + 1: i for i in range(L)}

        cat_idx = {resno_to_idx[r] for r in catalytic_resnos if r in resno_to_idx}
        fix_idx = {resno_to_idx[r] for r in fixed_resnos if r in resno_to_idx}

        # Mature N-term: handle MetAP
        n_pair: Optional[tuple[str, str]] = None
        if (self.profile.metap_cleaves_n_terminal_M
                and len(seq) >= 2 and seq[0] == "M"
                and seq[1] in "ACGPSTV"):
            # Met cleaved -> position 1 of mature protein = seq[1]
            n_pair = (seq[1], seq[2] if len(seq) >= 3 else "")
        elif len(seq) >= 2:
            n_pair = (seq[0], seq[1])

        ctx = StructureContext(
            sequence=seq,
            ss_reduced=ss_reduced,
            ss_full=ss_full,
            ss_confidence=ss_confidence,
            sasa=sasa,
            position_class=position_class,
            catalytic_resnos_zero_idx=cat_idx,
            fixed_resnos_zero_idx=fix_idx,
            protein_resnos=protein_resnos,
            n_terminal_pair=n_pair,
            has_structure=any(
                x is not None for x in (ss_reduced, sasa, position_class)
            ),
        )

        hits: list[RuleHit] = []
        for rule in self.registry.all():
            if not rule.is_active(self.profile):
                continue
            if rule.requires_structure and not ctx.has_structure:
                LOGGER.debug(
                    "rule %s requires_structure but no SS/SASA available; skipping",
                    rule.name,
                )
                continue
            try:
                rule_hits = rule.evaluate(ctx, self.profile)
            except Exception as e:
                LOGGER.exception("rule %s raised; treating as no-hit", rule.name)
                continue
            hits.extend(rule_hits)

        return EngineResult(
            sequence=seq,
            profile_name=self.profile.name,
            hits=hits,
        )


__all__ = ["EngineResult", "ExpressionRuleEngine"]
