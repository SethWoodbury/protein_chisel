"""Driver-stage metric *catalog* — declarative descriptors for the metrics the
``scripts/iterative_design.py`` ``stage_*`` chain produces.

THREE related-but-distinct layers exist in this codebase; do NOT conflate them
(they have different granularities and different jobs):

  1. :mod:`protein_chisel.scoring.metrics` — a per-*candidate* COMPUTE protocol
     (``fn: Candidate -> MetricResult``) with caching + tier scheduling. Used by
     ``pipelines/``; **not** by the iterative-design driver.
  2. :mod:`protein_chisel.scoring.multi_objective` — the RANKING-objective spec
     (``MetricSpec(column, direction, weight, target)``) + ``DEFAULT_METRIC_SPECS``
     that feed ``mo_topsis``. This is the single source of truth for *ranking*; the
     catalog here **reuses** it (it does not re-encode weights/targets).
  3. this package (:mod:`protein_chisel.metrics`) — the driver-stage CATALOG +
     SELECTION layer. A declarative map of *which stage emits which TSV columns*,
     plus each metric's role, dependencies, and cost. It powers ``--metrics`` /
     ``--filters`` subset selection, verbose logging, and provenance. **It does not
     compute anything** — computation stays in the existing ``stage_*`` functions,
     so with the default selection (``all``) the pipeline runs the exact same code
     and the default TSV/filter/ranking output is byte-identical.

A :class:`MetricDescriptor` is therefore pure metadata: it answers "if the user
deselects metric X, which stage work can we skip and which TSV columns / ranking
objectives disappear?" without changing how anything is computed.

Mirrors the :mod:`protein_chisel.experts` registry pattern (catalog of named,
selectable units with dependency-aware resolution). Registry mechanics live in
:mod:`protein_chisel.metrics.registry`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple

# --- Roles ---------------------------------------------------------------
# What a metric DOES in the pipeline. A single metric can be both a filter and
# feed ranking (e.g. ``sap_max`` gates struct survivors AND is a min-objective);
# we classify by its *primary* gating role and list ranking via ``objective_labels``.
ROLE_FILTER = "filter"          # can drop rows (predicate/threshold)
ROLE_OBJECTIVE = "objective"    # feeds mo_topsis ranking (see objective_labels)
ROLE_DIAGNOSTIC = "diagnostic"  # computed + written to the TSV, never gates/ranks
ROLE_TRANSFORM = "transform"    # produces no metric columns (sample/restore/protonate)
ROLES: FrozenSet[str] = frozenset(
    {ROLE_FILTER, ROLE_OBJECTIVE, ROLE_DIAGNOSTIC, ROLE_TRANSFORM}
)

# --- Capability dependencies --------------------------------------------
# External capabilities a metric needs to actually run. If a capability is
# absent, the metric is skipped with a warning (never a crash) — portability.
DEP_LIGAND = "ligand"          # needs a HETATM ligand in the scaffold
DEP_TUNNEL_SIF = "tunnel_sif"  # needs pyKVFinder (rides in a container)
DEP_PYROSETTA = "pyrosetta"    # needs PyRosetta (pyrosetta.sif)
DEP_FPOCKET = "fpocket"        # needs the fpocket binary
DEP_ARPEGGIO = "arpeggio"      # needs pdbe-arpeggio (esmc.sif)
DEP_ROSETTA = "rosetta"        # needs PyRosetta + ligand .params (ddG panel)
DEP_FREESASA = "freesasa"      # needs the freesasa python package (SAP proxy)
DEP_CONTACT_MS = "contact_ms"  # needs py_contact_ms (esmc.sif) for the CMS panel
KNOWN_DEPS: FrozenSet[str] = frozenset(
    {DEP_LIGAND, DEP_TUNNEL_SIF, DEP_PYROSETTA, DEP_FPOCKET,
     DEP_ARPEGGIO, DEP_ROSETTA, DEP_FREESASA, DEP_CONTACT_MS}
)

# --- Cost classes (nominal per-design wall-clock; used for logging/ordering) ---
COST_TRIVIAL = "trivial"     # microseconds (pure arithmetic on existing columns)
COST_CHEAP = "cheap"         # milliseconds (pure-Python on one structure/sequence)
COST_MODERATE = "moderate"   # ~0.1-1 s (freesasa, geometry sweeps)
COST_EXPENSIVE = "expensive" # seconds+ (subprocess: fpocket, rosetta, arpeggio)
COSTS: FrozenSet[str] = frozenset(
    {COST_TRIVIAL, COST_CHEAP, COST_MODERATE, COST_EXPENSIVE}
)

# --- Producing stages (the stage_* function that emits a metric's columns) ---
# Keep these in sync with scripts/iterative_design.py stage_* names. Used to map
# a metric selection back onto which stage work can be skipped.
STAGE_SEQ_FILTER = "seq_filter"
STAGE_STRUCT_FILTER = "struct_filter"
STAGE_TUNNEL = "tunnel"
STAGE_FITNESS = "fitness"
STAGE_FPOCKET = "fpocket"
STAGE_ARPEGGIO = "arpeggio"
STAGE_CMS = "cms"
STAGE_ROSETTA = "rosetta"


@dataclass(frozen=True)
class MetricDescriptor:
    """Declarative description of one metric the driver's stage chain produces.

    Pure metadata — never computes. The catalog of these (see
    :mod:`protein_chisel.metrics.registry`) is what ``--metrics`` / ``--filters``
    select over, what verbose logging and provenance enumerate, and what tells the
    driver which stage work is skippable when a metric is deselected.

    Attributes:
        name: stable registry key (e.g. ``"sap"``, ``"fpocket"``, ``"instability"``).
            Lower-case, no spaces. This is what the user types in ``--metrics``.
        role: one of :data:`ROLES`.
        stage: the producing ``stage_*`` (one of the ``STAGE_*`` constants).
        description: human-readable one-liner (surfaced in verbose logs / docs).
        columns: exact verbatim TSV column names this metric contributes. Used by
            the column-order golden test + provenance. Empty for transforms.
        column_prefixes: prefixes for *dynamic* columns whose full set depends on
            run inputs (e.g. ``"dfi__mean__"`` for per-class DFI, ``"mpnn_"``).
        deps: capabilities required to run (subset of :data:`KNOWN_DEPS`). Absent
            capability ⇒ skipped with a warning, not a crash.
        cost: one of :data:`COSTS` (nominal per-design wall-clock).
        objective_labels: the ``multi_objective`` *labels* (see
            ``DEFAULT_METRIC_SPECS``) this metric feeds into ``mo_topsis``. Empty
            unless ``role`` involves ranking. The labels — not duplicated weights —
            are the link to the single ranking source of truth.
        filter_predicates: short ids of the predicate(s) a filter applies (for
            logging / docs; the actual thresholds live in the stage code + cycle
            config, not here).
        default_on: whether this metric is in the default ``all`` selection.
            Everything ships ``True`` so the default selection reproduces today.
        gates_survivors: True if deselecting this filter changes *which rows
            survive* (vs merely dropping a diagnostic column). Used to warn loudly
            when a selection would alter the survivor set.
    """

    name: str
    role: str
    stage: str
    description: str
    columns: Tuple[str, ...] = ()
    column_prefixes: Tuple[str, ...] = ()
    deps: FrozenSet[str] = frozenset()
    cost: str = COST_CHEAP
    objective_labels: Tuple[str, ...] = ()
    filter_predicates: Tuple[str, ...] = ()
    default_on: bool = True
    gates_survivors: bool = False

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                f"metric {self.name!r}: unknown role {self.role!r}; "
                f"expected one of {sorted(ROLES)}"
            )
        if self.stage and not isinstance(self.stage, str):
            raise ValueError(f"metric {self.name!r}: stage must be a string")
        if self.cost not in COSTS:
            raise ValueError(
                f"metric {self.name!r}: unknown cost {self.cost!r}; "
                f"expected one of {sorted(COSTS)}"
            )
        bad = set(self.deps) - KNOWN_DEPS
        if bad:
            raise ValueError(
                f"metric {self.name!r}: unknown deps {sorted(bad)}; "
                f"known: {sorted(KNOWN_DEPS)}"
            )
        if self.role == ROLE_FILTER and not self.gates_survivors:
            # A filter that doesn't gate survivors is a contradiction; catch it.
            raise ValueError(
                f"metric {self.name!r}: role=filter but gates_survivors=False"
            )

    def matches_column(self, col: str) -> bool:
        """True if ``col`` is one of this metric's exact or prefixed columns."""
        if col in self.columns:
            return True
        return any(col.startswith(p) for p in self.column_prefixes)

    def all_column_specs(self) -> Tuple[str, ...]:
        """The exact columns plus ``<prefix>*`` markers (for logging/docs)."""
        return tuple(self.columns) + tuple(f"{p}*" for p in self.column_prefixes)


__all__ = [
    "MetricDescriptor",
    "ROLE_FILTER", "ROLE_OBJECTIVE", "ROLE_DIAGNOSTIC", "ROLE_TRANSFORM", "ROLES",
    "DEP_LIGAND", "DEP_TUNNEL_SIF", "DEP_PYROSETTA", "DEP_FPOCKET",
    "DEP_ARPEGGIO", "DEP_ROSETTA", "DEP_FREESASA", "DEP_CONTACT_MS", "KNOWN_DEPS",
    "COST_TRIVIAL", "COST_CHEAP", "COST_MODERATE", "COST_EXPENSIVE", "COSTS",
    "STAGE_SEQ_FILTER", "STAGE_STRUCT_FILTER", "STAGE_TUNNEL", "STAGE_FITNESS",
    "STAGE_FPOCKET", "STAGE_ARPEGGIO", "STAGE_CMS", "STAGE_ROSETTA",
]
