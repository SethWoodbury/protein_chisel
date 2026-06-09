"""Driver-stage metric catalog + selection (see :mod:`.base` for the layering).

This package is the *declarative* metric layer for the ``iterative_design`` driver:
it catalogs which ``stage_*`` produces which TSV columns, each metric's role /
dependencies / cost, and resolves ``--metrics`` / ``--filters`` selections. It does
not compute anything — computation stays in the driver's stage functions, so the
default selection (``"all"``) is byte-identical to today.

Distinct from :mod:`protein_chisel.scoring.metrics` (per-candidate compute protocol)
and :mod:`protein_chisel.scoring.multi_objective` (ranking-objective specs, reused
here via ``objective_labels``).
"""
from protein_chisel.metrics.base import (
    COST_CHEAP, COST_EXPENSIVE, COST_MODERATE, COST_TRIVIAL, COSTS,
    DEP_ARPEGGIO, DEP_FPOCKET, DEP_FREESASA, DEP_LIGAND, DEP_PYROSETTA,
    DEP_ROSETTA, DEP_TUNNEL_SIF, KNOWN_DEPS,
    MetricDescriptor,
    ROLE_DIAGNOSTIC, ROLE_FILTER, ROLE_OBJECTIVE, ROLE_TRANSFORM, ROLES,
)
from protein_chisel.metrics.registry import (
    ResolvedSelection,
    all_metrics, available_metrics, get_metric, register_metric, resolve_metrics,
)

__all__ = [
    "MetricDescriptor", "ResolvedSelection",
    "all_metrics", "available_metrics", "get_metric", "register_metric",
    "resolve_metrics",
    "ROLE_FILTER", "ROLE_OBJECTIVE", "ROLE_DIAGNOSTIC", "ROLE_TRANSFORM", "ROLES",
    "DEP_LIGAND", "DEP_TUNNEL_SIF", "DEP_PYROSETTA", "DEP_FPOCKET",
    "DEP_ARPEGGIO", "DEP_ROSETTA", "DEP_FREESASA", "KNOWN_DEPS",
    "COST_TRIVIAL", "COST_CHEAP", "COST_MODERATE", "COST_EXPENSIVE", "COSTS",
]
