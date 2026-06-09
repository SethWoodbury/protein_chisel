"""Driver-stage metric catalog + selection resolution.

The ``_CATALOG`` below enumerates every metric the ``scripts/iterative_design.py``
``stage_*`` chain produces, as :class:`~protein_chisel.metrics.base.MetricDescriptor`
metadata (NOT compute). Insertion order mirrors the stage emission order, so it
doubles as the canonical metric ordering for logging/provenance.

Selection (``--metrics`` / ``--filters``) resolves a name list/``"all"`` into the
ordered descriptors, dependency-checked against the run's available capabilities;
metrics whose capability is absent are *skipped with a warning*, never a crash. The
default selection (``"all"``) reproduces today's behavior exactly — this layer only
*gates*; computation stays in the existing stage functions.

Mirrors :mod:`protein_chisel.experts.registry`. Objective metrics link to the
single ranking source of truth via ``objective_labels`` (the labels in
``scoring/multi_objective.py::DEFAULT_METRIC_SPECS``); weights/targets are NOT
duplicated here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Sequence, Tuple, Union

from protein_chisel.metrics.base import (
    COST_CHEAP, COST_EXPENSIVE, COST_MODERATE, COST_TRIVIAL,
    DEP_ARPEGGIO, DEP_CONTACT_MS, DEP_FPOCKET, DEP_FREESASA, DEP_LIGAND,
    DEP_PYROSETTA, DEP_ROSETTA, DEP_TUNNEL_SIF,
    MetricDescriptor,
    ROLE_DIAGNOSTIC, ROLE_FILTER, ROLE_OBJECTIVE,
    STAGE_ARPEGGIO, STAGE_CMS, STAGE_FITNESS, STAGE_FPOCKET, STAGE_ROSETTA,
    STAGE_SEQ_FILTER, STAGE_STRUCT_FILTER, STAGE_TUNNEL,
)

LOGGER = logging.getLogger("protein_chisel.metrics.registry")


# ======================================================================
# The catalog — order mirrors the stage_* emission order in the driver.
# ======================================================================
# NB: ``columns`` strings are verbatim from the recon of stage_* (cross-checked
# against scripts/iterative_design.py). They feed provenance + the column-order
# golden test. Dynamic column families use ``column_prefixes``.
_CATALOG_LIST: List[MetricDescriptor] = [
    # ---- stage_seq_filter (cheap, pure sequence) ----------------------
    MetricDescriptor(
        name="length", role=ROLE_FILTER, stage=STAGE_SEQ_FILTER,
        description="Sequence length must equal the WT length.",
        columns=("length",), cost=COST_TRIVIAL,
        filter_predicates=("len != wt_length",), gates_survivors=True,
    ),
    MetricDescriptor(
        name="net_charge", role=ROLE_FILTER, stage=STAGE_SEQ_FILTER,
        description="Net charge at design pH (Henderson-Hasselbalch variants); "
                    "hard band + target objective.",
        columns=("net_charge_full_HH", "net_charge_no_HIS",
                 "net_charge_with_HIS_HH", "net_charge_HIS_half",
                 "net_charge_DE_KR_only", "design_ph"),
        cost=COST_TRIVIAL, objective_labels=("charge",),
        filter_predicates=("charge >= net_charge_max", "charge <= net_charge_min"),
        gates_survivors=True,
    ),
    MetricDescriptor(
        name="pi", role=ROLE_FILTER, stage=STAGE_SEQ_FILTER,
        description="Isoelectric point band + target objective.",
        columns=("pi",), cost=COST_TRIVIAL, objective_labels=("pi",),
        filter_predicates=("not (pi_min <= pi <= pi_max)",), gates_survivors=True,
    ),
    MetricDescriptor(
        name="instability", role=ROLE_FILTER, stage=STAGE_SEQ_FILTER,
        description="Guruprasad instability index; hard max + min objective.",
        columns=("instability_index",), cost=COST_TRIVIAL,
        objective_labels=("instability",),
        filter_predicates=("instability_index >= instability_max",),
        gates_survivors=True,
    ),
    MetricDescriptor(
        name="gravy", role=ROLE_FILTER, stage=STAGE_SEQ_FILTER,
        description="GRAVY hydropathy band + target objective.",
        columns=("gravy",), cost=COST_TRIVIAL, objective_labels=("gravy",),
        filter_predicates=("not (gravy_min <= gravy <= gravy_max)",),
        gates_survivors=True,
    ),
    MetricDescriptor(
        name="aliphatic", role=ROLE_FILTER, stage=STAGE_SEQ_FILTER,
        description="Aliphatic index min + target objective.",
        columns=("aliphatic_index",), cost=COST_TRIVIAL,
        objective_labels=("aliphatic",),
        filter_predicates=("aliphatic_index < aliphatic_min",),
        gates_survivors=True,
    ),
    MetricDescriptor(
        name="boman", role=ROLE_FILTER, stage=STAGE_SEQ_FILTER,
        description="Boman (protein-binding potential) index max + target objective.",
        columns=("boman_index",), cost=COST_TRIVIAL, objective_labels=("boman",),
        filter_predicates=("boman_index >= boman_max",), gates_survivors=True,
    ),
    MetricDescriptor(
        name="expression", role=ROLE_FILTER, stage=STAGE_SEQ_FILTER,
        description="Host-expression liability engine (hard-filter hits gate; "
                    "soft/omit hits are diagnostics).",
        columns=("n_expression_warnings", "n_expression_soft_bias_hits",
                 "n_expression_hard_omit_hits", "n_expression_hard_filter_hits",
                 "expression_rule_summary"),
        cost=COST_CHEAP, filter_predicates=("expression hard_filter_hits > 0",),
        gates_survivors=True,
    ),
    MetricDescriptor(
        name="protparam_aux", role=ROLE_DIAGNOSTIC, stage=STAGE_SEQ_FILTER,
        description="Auxiliary ProtParam diagnostics (aromaticity, flexibility, "
                    "SS-from-seq fractions, MW, extinction coeffs).",
        columns=("aromaticity", "flexibility_mean_seq", "helix_frac_seq",
                 "turn_frac_seq", "sheet_frac_seq", "molecular_weight",
                 "extinction_280nm_no_disulfide", "extinction_280nm_disulfide"),
        cost=COST_TRIVIAL,
    ),
    # NB: the ``selection__seq_filter_*`` "distance to passing" gap columns are
    # deliberately NOT catalogued as a metric — the original code prefixes them
    # ``selection__`` because they are backfill-*selection* bookkeeping (they rank
    # rescue candidates), not a user-selectable scientific metric. They are
    # classed as framework columns alongside the other ``selection__*`` columns.

    # ---- stage_struct_filter (structure; freesasa for SAP) ------------
    MetricDescriptor(
        name="cat_his_hbonds", role=ROLE_FILTER, stage=STAGE_STRUCT_FILTER,
        description="Count of side-chain H-bonds to catalytic HIS; >=1 required.",
        columns=("n_hbonds_to_cat_his",), cost=COST_CHEAP,
        objective_labels=("hbonds_to_cat",),
        filter_predicates=("n_hbonds_to_cat_his < 1",), gates_survivors=True,
    ),
    MetricDescriptor(
        name="ligand_int", role=ROLE_OBJECTIVE, stage=STAGE_STRUCT_FILTER,
        description="Geometric ligand-interaction panel "
                    "(hbond/salt-bridge/pi-pi/pi-cation/hydrophobic/vdw-clash).",
        columns=("ligand_int__n_total", "ligand_int__strength_total",
                 "ligand_int__n_hbond", "ligand_int__strength_hbond",
                 "ligand_int__n_salt_bridge", "ligand_int__strength_salt_bridge",
                 "ligand_int__n_pi_pi", "ligand_int__strength_pi_pi",
                 "ligand_int__n_pi_cation", "ligand_int__strength_pi_cation",
                 "ligand_int__n_hydrophobic", "ligand_int__strength_hydrophobic",
                 "ligand_int__n_vdw_clash", "ligand_int__strength_vdw_clash"),
        deps=frozenset({DEP_LIGAND}), cost=COST_CHEAP,
        objective_labels=("lig_int_strength",),
    ),
    MetricDescriptor(
        name="sap", role=ROLE_FILTER, stage=STAGE_STRUCT_FILTER,
        description="Spatial aggregation propensity (freesasa proxy); max gates, "
                    "max also a min-objective.",
        columns=("sap_max", "sap_mean", "sap_p95"),
        deps=frozenset({DEP_FREESASA}), cost=COST_MODERATE,
        objective_labels=("sap_max",),
        filter_predicates=("sap_max > sap_max_threshold",), gates_survivors=True,
    ),
    MetricDescriptor(
        name="clash", role=ROLE_FILTER, stage=STAGE_STRUCT_FILTER,
        description="Heavy-atom clash detection; severe clash gates (when "
                    "clash_filter on).",
        columns=("clash__n_total", "clash__n_to_catalytic", "clash__n_to_ligand",
                 "clash__has_severe", "clash__detail"),
        cost=COST_CHEAP, filter_predicates=("clash_filter and clash__has_severe",),
        gates_survivors=True,
    ),
    MetricDescriptor(
        name="preorg", role=ROLE_OBJECTIVE, stage=STAGE_STRUCT_FILTER,
        description="Active-site preorganization (H-bond/salt/pi network density "
                    "to catalytic + shells).",
        columns=("preorg__n_hbonds_to_cat", "preorg__n_salt_bridges_to_cat",
                 "preorg__n_pi_to_cat", "preorg__n_hbonds_within_shells",
                 "preorg__strength_total", "preorg__interactome_density",
                 "preorg__n_first_shell", "preorg__n_second_shell"),
        cost=COST_CHEAP, objective_labels=("preorg_strength",),
    ),
    MetricDescriptor(
        name="dfi", role=ROLE_DIAGNOSTIC, stage=STAGE_STRUCT_FILTER,
        description="Dynamic Flexibility Index (seed-invariant; broadcast to every "
                    "row), overall + per residue-class.",
        columns=("dfi__mean", "dfi__std", "dfi__max", "dfi__min", "dfi__elapsed_ms"),
        column_prefixes=("dfi__mean__", "dfi__std__"), cost=COST_CHEAP,
    ),

    # ---- stage_tunnel_metrics (ray-cast + optional pyKVFinder) --------
    MetricDescriptor(
        name="tunnel", role=ROLE_FILTER, stage=STAGE_TUNNEL,
        description="Ray-cast tunnel patency; verdict {buried, ligand_too_big} "
                    "hard-gates when hard_gate on; several min/max objectives.",
        columns=("tunnel__best_cone_mean_path", "tunnel__sidechain_blocked_fraction",
                 "tunnel__throat_bulky_designable_count", "tunnel__bottleneck_radius",
                 "tunnel__backbone_blocked_fraction", "tunnel__catalytic_blocked_fraction",
                 "tunnel__best_cone_axis_dot_ligand_pa", "tunnel__n_escape_cones",
                 "tunnel__verdict", "tunnel__elapsed_ms"),
        cost=COST_MODERATE,
        objective_labels=("tunnel_dsc_blocked", "tunnel_throat_blockers",
                          "tunnel_best_cone_path"),
        filter_predicates=("tunnel__verdict in {buried, ligand_too_big}",),
        gates_survivors=True,
    ),
    MetricDescriptor(
        name="pkvf", role=ROLE_OBJECTIVE, stage=STAGE_TUNNEL,
        description="pyKVFinder cavity geometry (volume/depth/openings); rank-down "
                    "objectives, never gates.",
        columns=("pkvf__cavity_volume", "pkvf__cavity_depth_max", "pkvf__n_cavities",
                 "pkvf__n_openings", "pkvf__has_opening", "pkvf__elapsed_ms"),
        deps=frozenset({DEP_TUNNEL_SIF}), cost=COST_MODERATE,
        objective_labels=("pkvf_cavity_depth", "pkvf_cavity_volume"),
    ),

    # ---- stage_fitness_score (PLM marginals; primary axis) ------------
    MetricDescriptor(
        name="fitness", role=ROLE_OBJECTIVE, stage=STAGE_FITNESS,
        description="PLM pseudo-log-likelihood fitness (ESM-C / SaProt / fused); "
                    "fused mean is the heaviest ranking axis.",
        columns=("fitness__logp_esmc_mean", "fitness__logp_saprot_mean",
                 "fitness__logp_fused_mean", "fitness__method",
                 "fitness__delta_vs_wt", "fitness__wt_logp_fused"),
        cost=COST_CHEAP, objective_labels=("fitness",),
    ),

    # ---- stage_fpocket_rank (fpocket binary) -------------------------
    MetricDescriptor(
        name="fpocket", role=ROLE_FILTER, stage=STAGE_FPOCKET,
        description="fpocket active-site pocket panel; druggability is a max-"
                    "objective AND the final-topk hard filter (>= druggability_min). "
                    "Gates survivors only at the final selection, not per-cycle.",
        columns=("fpocket__status", "fpocket__druggability", "fpocket__volume",
                 "fpocket__mean_alpha_sphere_radius", "fpocket__alpha_sphere_density",
                 "fpocket__n_alpha_spheres_near_catalytic",
                 "fpocket__mean_alpha_sphere_dist_to_catalytic",
                 "fpocket__n_pockets_found", "fpocket__score",
                 "fpocket__n_alpha_spheres", "fpocket__total_sasa",
                 "fpocket__polar_sasa", "fpocket__apolar_sasa",
                 "fpocket__hydrophobicity_score", "fpocket__polarity_score",
                 "fpocket__charge_score", "fpocket__volume_score",
                 "fpocket__polar_atoms_pct", "fpocket__apolar_atoms_pct",
                 "fpocket__apolar_alpha_sphere_proportion",
                 "fpocket__mean_local_hydrophobic_density",
                 "fpocket__mean_alpha_sphere_solvent_acc",
                 "fpocket__cent_of_mass_alpha_sphere_max_dist",
                 "fpocket__bottleneck_radius", "fpocket__min_alpha_sphere_radius",
                 "fpocket__alpha_sphere_radius_p10",
                 "fpocket__polar_alpha_sphere_proportion", "fpocket__n_rim_spheres"),
        deps=frozenset({DEP_FPOCKET}), cost=COST_EXPENSIVE,
        objective_labels=("druggability", "bottleneck", "pocket_hydrophobicity"),
        filter_predicates=("fpocket__druggability >= druggability_min (final)",),
        gates_survivors=True,
    ),

    # ---- final-only enrichments (opt-in stages) ----------------------
    MetricDescriptor(
        name="cms", role=ROLE_DIAGNOSTIC, stage=STAGE_CMS,
        description="Coventry contact-molecular-surface of the protein-ligand "
                    "interface (final top-K only; --cms_final).",
        columns=("cms__total",), deps=frozenset({DEP_CONTACT_MS}),
        cost=COST_EXPENSIVE, default_on=False,
    ),
    MetricDescriptor(
        name="rosetta", role=ROLE_DIAGNOSTIC, stage=STAGE_ROSETTA,
        description="PyRosetta ddG / interface-energy panel (final top-K only; "
                    "--rosetta_final).",
        columns=("rosetta__ddg", "rosetta__contact_molecular_surface",
                 "rosetta__ligand_interface_energy", "rosetta__total_energy",
                 "rosetta__n_hbonds_to_ligand"),
        deps=frozenset({DEP_ROSETTA}), cost=COST_EXPENSIVE, default_on=False,
    ),
    MetricDescriptor(
        name="arpeggio", role=ROLE_DIAGNOSTIC, stage=STAGE_ARPEGGIO,
        description="pdbe-arpeggio contact panel (optional; not in the default "
                    "finalize path).",
        column_prefixes=("arpeggio__",), deps=frozenset({DEP_ARPEGGIO}),
        cost=COST_EXPENSIVE, default_on=False,
    ),
]

_CATALOG: dict[str, MetricDescriptor] = {}
for _d in _CATALOG_LIST:
    if _d.name in _CATALOG:
        raise RuntimeError(f"duplicate metric name in catalog: {_d.name!r}")
    _CATALOG[_d.name] = _d
del _d


# ======================================================================
# Public API
# ======================================================================
def register_metric(desc: MetricDescriptor) -> None:
    """Register (or override) a metric descriptor by name."""
    _CATALOG[desc.name.strip().lower()] = desc


def available_metrics() -> List[str]:
    """Catalog metric names in canonical (stage-emission) order."""
    return list(_CATALOG.keys())


def get_metric(name: str) -> MetricDescriptor:
    """Look up one descriptor by name (raises ``KeyError`` if unknown)."""
    key = name.strip().lower()
    if key not in _CATALOG:
        raise KeyError(f"unknown metric {name!r}; available: {available_metrics()}")
    return _CATALOG[key]


def all_metrics() -> List[MetricDescriptor]:
    """All descriptors in canonical order."""
    return list(_CATALOG.values())


@dataclass
class ResolvedSelection:
    """Outcome of resolving a ``--metrics``/``--filters`` selection.

    Attributes:
        selected: descriptors that are active (in canonical order).
        skipped_unknown: requested names not found in the catalog.
        skipped_wrong_role: *explicitly* requested names that exist but whose role
            doesn't match the ``role`` filter (e.g. a ``--filters`` name that is an
            objective, not a filter). Empty for the ``"all"`` selection, where a
            role filter naturally narrows the full catalog (not a user error).
        skipped_missing_dep: (descriptor, missing-capabilities) for metrics that
            were requested/default-on but whose dependency is unavailable.
    """
    selected: List[MetricDescriptor] = field(default_factory=list)
    skipped_unknown: List[str] = field(default_factory=list)
    skipped_wrong_role: List[str] = field(default_factory=list)
    skipped_missing_dep: List[Tuple[MetricDescriptor, FrozenSet[str]]] = \
        field(default_factory=list)

    def names(self) -> List[str]:
        return [d.name for d in self.selected]

    def stages(self) -> List[str]:
        """Distinct producing stages among the selected metrics, in first-seen
        (canonical) order — i.e. the stage work that must run."""
        seen: List[str] = []
        for d in self.selected:
            if d.stage and d.stage not in seen:
                seen.append(d.stage)
        return seen

    def objective_labels(self) -> List[str]:
        """multi_objective labels contributed by the selected metrics (deduped,
        canonical order). Use to filter DEFAULT_METRIC_SPECS for ranking."""
        seen: List[str] = []
        for d in self.selected:
            for lbl in d.objective_labels:
                if lbl not in seen:
                    seen.append(lbl)
        return seen

    def gating_filters(self) -> List[MetricDescriptor]:
        """Selected metrics that gate the survivor set (role=filter)."""
        return [d for d in self.selected if d.role == ROLE_FILTER]


def _parse_selection(selection: Union[str, Sequence[str]]) -> Optional[List[str]]:
    """Normalize a selection. Returns ``None`` for the sentinel ``"all"``
    (meaning "every default-on metric"), else a lower-cased name list."""
    if isinstance(selection, str):
        s = selection.strip().lower()
        if s in ("", "all", "*"):
            return None
        parts = selection.split(",")
    else:
        parts = list(selection)
    return [p.strip().lower() for p in parts if str(p).strip()]


def resolve_metrics(
    selection: Union[str, Sequence[str]] = "all",
    *,
    capabilities: Optional[FrozenSet[str]] = None,
    role: Optional[str] = None,
) -> ResolvedSelection:
    """Resolve a ``--metrics`` selection into active descriptors.

    Args:
        selection: ``"all"`` (default → every ``default_on`` metric) or a CSV /
            sequence of metric names.
        capabilities: the run's available capabilities (subset of
            :data:`~protein_chisel.metrics.base.KNOWN_DEPS`). When provided, a
            metric whose ``deps`` are not all available is **skipped with a
            warning** (recorded in ``skipped_missing_dep``), never raising. When
            ``None``, dependency checks are bypassed (pure catalog introspection).
        role: optional role filter (e.g. ``ROLE_FILTER`` for ``--filters``).

    Returns:
        :class:`ResolvedSelection`. Order is always canonical (catalog order),
        independent of the order names were requested in, so output is stable.
    """
    wanted = _parse_selection(selection)
    explicit = wanted is not None
    res = ResolvedSelection()

    if wanted is None:
        chosen = [d for d in _CATALOG.values() if d.default_on]
    else:
        chosen = []
        for name in wanted:
            if name not in _CATALOG:
                res.skipped_unknown.append(name)
                LOGGER.warning("metrics: unknown metric %r — ignoring", name)
                continue
            chosen.append(_CATALOG[name])
        # de-dup while preserving canonical order
        chosen_names = {d.name for d in chosen}
        chosen = [d for d in _CATALOG.values() if d.name in chosen_names]

    for d in chosen:
        if role is not None and d.role != role:
            # Only a user error when the name was explicitly requested; for "all"
            # a role filter just narrows the catalog (expected, not reported).
            if explicit:
                res.skipped_wrong_role.append(d.name)
                LOGGER.warning(
                    "metrics: %r is not a %s metric (role=%s) — ignoring for this "
                    "selection", d.name, role, d.role)
            continue
        if capabilities is not None:
            missing = frozenset(d.deps) - frozenset(capabilities)
            if missing:
                res.skipped_missing_dep.append((d, missing))
                LOGGER.warning(
                    "metrics: skipping %r — missing capability(ies) %s",
                    d.name, sorted(missing),
                )
                continue
        res.selected.append(d)
    return res


__all__ = [
    "MetricDescriptor", "ResolvedSelection",
    "register_metric", "available_metrics", "get_metric", "all_metrics",
    "resolve_metrics",
]
