"""Expression-host risk modeling.

Modular registry of known proteolysis / aggregation / translation /
toxicity risks that can derail recombinant expression. Each rule has a
configurable severity (warn/soft-bias/hard-omit/hard-filter) and may
condition on secondary-structure / SASA / position class.

The engine takes a sequence (+ optional structural context) and a
profile (host + tags + induction + overrides) and returns an
``EngineResult`` carrying:
  - hard_filter_hits     -- reject the sequence
  - hard_omit_per_residue   -- forbid AAs at sample time via MPNN
  - soft_bias_per_residue   -- downweight AAs in MPNN bias
  - warnings             -- record metadata; no action

Usage::

    from protein_chisel.expression import (
        ExpressionRuleEngine, ExpressionProfile, Severity,
    )
    profile = ExpressionProfile.bl21_cytosolic_streptag()
    engine = ExpressionRuleEngine(profile=profile)
    res = engine.evaluate(seq, pdb_path=pdb, position_table=pt,
                          catalytic_resnos=[157], fixed_resnos=[157, ...])
    if not res.passes_hard_filter():
        ...  # reject
    omit = res.to_omit_AA_json(chain="A", protein_resnos=[10,11,12,...])
"""

from protein_chisel.expression.severity import RuleHit, Severity
from protein_chisel.expression.rules import (
    REGISTRY, Rule, RuleRegistry, StructureContext,
)
from protein_chisel.expression.profiles import ExpressionProfile
from protein_chisel.expression.engine import EngineResult, ExpressionRuleEngine


__all__ = [
    "EngineResult",
    "ExpressionProfile",
    "ExpressionRuleEngine",
    "REGISTRY",
    "Rule",
    "RuleHit",
    "RuleRegistry",
    "Severity",
    "StructureContext",
]
