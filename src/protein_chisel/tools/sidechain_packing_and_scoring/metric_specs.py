"""MetricSpec adapters for the sidechain packing/scoring tools.

Each tool's existing wrapper (rotamer_score, rotalyze_score, etc.)
already returns a result dataclass with ``.to_dict(prefix=...)``. This
module wraps those wrappers into :class:`MetricSpec` instances that
:class:`scoring.tier.TierPlan` can compose.

The adapters are intentionally thin: they translate the
``Candidate``-flavoured input into the kwargs each tool expects, call
the tool, and feed the result through :func:`from_tool_result` so the
``<prefix>failed`` sentinel + provenance flow is consistent across
tools. The cost_seconds estimates are conservative wall-clock budgets
on a single a4000 -- the tier scheduler uses them for ordering, not
for hard timeouts.

Registration is opt-in via :func:`register_default_specs`. Don't
register on import (that would force every protein_chisel consumer
to pay PyRosetta / esmc.sif startup at import time even when they
never invoke these metrics). Pipelines call ``register_default_specs(registry)``
explicitly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from protein_chisel.scoring.metrics import (
    Candidate,
    MetricRegistry,
    MetricResult,
    MetricSpec,
    from_tool_result,
)


LOGGER = logging.getLogger("protein_chisel.metric_specs")


# ----------------------------------------------------------------------
# rotamer_score (Rosetta fa_dun, Dunbrack 2011)
# ----------------------------------------------------------------------


def _rotamer_score_metric(c: Candidate, params: dict) -> MetricResult:
    from protein_chisel.tools.sidechain_packing_and_scoring.rotamer_score import (
        RotamerScoreConfig, rotamer_score,
    )

    cfg_keys = set(RotamerScoreConfig.__dataclass_fields__)
    cfg_kwargs = {k: v for k, v in params.items() if k in cfg_keys}
    cfg = RotamerScoreConfig(**cfg_kwargs)
    rosetta_params = params.get("rosetta_params", [])

    cat = set(c.catalytic_resnos) if c.catalytic_resnos else None

    res = rotamer_score(
        c.structure_path,
        params=rosetta_params,
        catalytic_resnos=cat,
        config=cfg,
    )
    return from_tool_result("rotamer_score", res)


ROTAMER_SCORE_SPEC = MetricSpec(
    name="rotamer_score",
    fn=_rotamer_score_metric,
    kind="structure",
    cost_seconds=0.5,            # ms per residue, but pyrosetta startup adds 0.3s
    needs_gpu=False,
    description="Rosetta fa_dun (Dunbrack 2011) per-residue rotamer log-likelihood",
    cache_version=1,
)


# ----------------------------------------------------------------------
# rotalyze_score (MolProbity Top8000 KDE)
# ----------------------------------------------------------------------


def _rotalyze_metric(c: Candidate, params: dict) -> MetricResult:
    from protein_chisel.tools.sidechain_packing_and_scoring.rotalyze_score import (
        rotalyze_score,
    )
    cat = set(c.catalytic_resnos) if c.catalytic_resnos else None
    res = rotalyze_score(c.structure_path, catalytic_resnos=cat)
    return from_tool_result("rotalyze", res)


ROTALYZE_SPEC = MetricSpec(
    name="rotalyze",
    fn=_rotalyze_metric,
    kind="structure",
    cost_seconds=2.0,            # ~1s rotalyze + esmc.sif startup
    needs_gpu=False,
    description="MolProbity Top8000 KDE rotamer outlier classification",
    cache_version=1,
)


# ----------------------------------------------------------------------
# fpocket (geometry / druggability)
# ----------------------------------------------------------------------


def _fpocket_metric(c: Candidate, params: dict) -> MetricResult:
    from protein_chisel.tools.fpocket_run import fpocket_run
    res = fpocket_run(
        c.structure_path,
        fpocket_exe=params.get("fpocket_exe"),
        timeout=params.get("timeout", 300.0),
    )
    return from_tool_result("fpocket", res)


FPOCKET_SPEC = MetricSpec(
    name="fpocket",
    fn=_fpocket_metric,
    kind="structure",
    cost_seconds=5.0,            # 1-3s fpocket + binary launch overhead
    needs_gpu=False,
    description="fpocket pocket geometry + druggability",
    cache_version=1,
)


# ----------------------------------------------------------------------
# pippack_score (Kuhlman 2024, learned chi distributions)
# ----------------------------------------------------------------------


def _pippack_score_metric(c: Candidate, params: dict) -> MetricResult:
    from protein_chisel.tools.sidechain_packing_and_scoring.pippack_score import (
        pippack_score,
    )
    res = pippack_score(
        c.structure_path,
        model_name=params.get("model_name", "pippack_model_1"),
        timeout=params.get("timeout", 600.0),
    )
    return from_tool_result("pippack", res)


PIPPACK_SCORE_SPEC = MetricSpec(
    name="pippack",
    fn=_pippack_score_metric,
    kind="structure",
    cost_seconds=60.0,           # ~1s GPU inference + 30-50s sif startup
    needs_gpu=True,
    description="PIPPack repacked-vs-input chi MAE / rotamer recovery",
    cache_version=1,
)


# ----------------------------------------------------------------------
# flowpacker_score (Lee 2025, torsional flow likelihood)
# ----------------------------------------------------------------------


def _flowpacker_score_metric(c: Candidate, params: dict) -> MetricResult:
    from protein_chisel.tools.sidechain_packing_and_scoring.flowpacker_score import (
        flowpacker_score,
    )
    res = flowpacker_score(
        c.structure_path,
        checkpoint=params.get("checkpoint", "cluster"),
        seed=params.get("seed", 42),
        timeout=params.get("timeout", 600.0),
    )
    return from_tool_result("flowpacker", res)


FLOWPACKER_SCORE_SPEC = MetricSpec(
    name="flowpacker",
    fn=_flowpacker_score_metric,
    kind="structure",
    cost_seconds=90.0,           # ~30s ODE integration + sif startup
    needs_gpu=True,
    description="FlowPacker per-residue per-chi log-likelihood",
    cache_version=1,
)


# ----------------------------------------------------------------------
# attnpacker_score (McPartlon 2023, SE3 transformer; via assess_packing)
# ----------------------------------------------------------------------


def _attnpacker_score_metric(c: Candidate, params: dict) -> MetricResult:
    from protein_chisel.tools.sidechain_packing_and_scoring.attnpacker_pack import (
        attnpacker_score,
    )
    res = attnpacker_score(
        c.structure_path,
        timeout=params.get("timeout", 1800.0),
    )
    return from_tool_result("attnpacker", res)


ATTNPACKER_SCORE_SPEC = MetricSpec(
    name="attnpacker",
    fn=_attnpacker_score_metric,
    kind="structure",
    cost_seconds=120.0,          # ~30s inference + 60s sif startup
    needs_gpu=True,
    description="AttnPacker repacked-vs-input chi MAE / rotamer recovery",
    cache_version=1,
)


# ----------------------------------------------------------------------
# faspr_pack (classical CPU; emits a repacked PDB, not a score)
# ----------------------------------------------------------------------
# faspr is a *transform* (input -> repacked structure), not a metric.
# We don't register it as a MetricSpec by default; pipelines that want
# to use FASPR as an intermediate should call the wrapper directly and
# pipe the output into a downstream metric (e.g. rotalyze on the
# faspr-repacked structure as a "rotalyze after sane repack" signal).
# Exposing it here as a spec would be misleading -- a metric should
# return scalar values, not a new structure.


# ----------------------------------------------------------------------
# Bulk register
# ----------------------------------------------------------------------


def register_default_specs(registry: Optional[MetricRegistry] = None) -> MetricRegistry:
    """Register the canonical sidechain-tool MetricSpecs into ``registry``.

    Idempotent: rejects any already-registered name silently rather
    than raising. Returns the registry for chaining.

    Pipelines call this once at startup, then look up specs by name
    when building TierPlans.
    """
    from protein_chisel.scoring.metrics import get_registry

    reg = registry if registry is not None else get_registry()
    specs = [
        ROTAMER_SCORE_SPEC,
        ROTALYZE_SPEC,
        FPOCKET_SPEC,
        PIPPACK_SCORE_SPEC,
        FLOWPACKER_SCORE_SPEC,
        ATTNPACKER_SCORE_SPEC,
    ]
    for spec in specs:
        if spec.name in reg:
            continue
        reg.register(spec)
    return reg


__all__ = [
    "ATTNPACKER_SCORE_SPEC",
    "FLOWPACKER_SCORE_SPEC",
    "FPOCKET_SPEC",
    "PIPPACK_SCORE_SPEC",
    "ROTALYZE_SPEC",
    "ROTAMER_SCORE_SPEC",
    "register_default_specs",
]
