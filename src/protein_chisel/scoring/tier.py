"""Cheap-first / expensive-late tier scheduler.

Antipattern #2 from the architectural review: don't put GPU calls in
the inner sample loop unbatched, and don't run expensive metrics on
candidates that already failed cheap hard-constraints. The tier
scheduler implements both: each tier is a list of metric specs that
run on every survivor of the previous tier, and within a tier we group
by ``needs_gpu`` so all GPU calls happen back-to-back (one apptainer
launch overhead per tier).

Concept
=======
Given a list of candidates, a TierPlan looks like::

    tiers = [
        # tier 0: fast CPU sequence/structural primitives
        [registry["fa_dun"], registry["seq_filter"], registry["foldseek"]],
        # tier 1: cheap-GPU likelihoods
        [registry["esmc_naturalness"]],
        # tier 2: structure-conditioned likelihoods
        [registry["pippack_score"], registry["fpocket"], registry["rotalyze"]],
        # tier 3: heavy generative likelihoods (only the survivors)
        [registry["flowpacker_score"], registry["attnpacker_score"]],
    ]
    constraints_per_tier = [
        [HardConstraint("fa_dun__mean", "<=", 5.0)],   # gate after tier 0
        [],                                             # no gate after tier 1
        [HardConstraint("rotalyze__frac_outliers", "<=", 0.10)],
        [],
    ]
    survivor_topk_per_tier = [None, 50, 25, 10]   # downselect AFTER constraints

The scheduler returns a long-form metric DataFrame (one row per
candidate) with one column per metric output, and a per-tier
``tier_log`` describing how many candidates entered/left each tier
plus the wall-clock budget consumed.

Antipatterns 3 + 4 (separating evaluate from accept; logging which
constraint stopped which candidate) are addressed here -- evaluate_tiered
returns ALL the data; the caller decides what to do with it. Hard
constraints log per-row so a stuck optimizer can be diagnosed.

Antipattern 5 (no baked-in metric weights) is upstream: TierPlan is
serialisable so it can be loaded from yaml/json.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from protein_chisel.scoring.cache import (
    InMemoryCache, MetricCache, call_metric_cached,
)
from protein_chisel.scoring.metrics import (
    Candidate, MetricResult, MetricSpec,
)


LOGGER = logging.getLogger("protein_chisel.scoring.tier")


# ----------------------------------------------------------------------
# Hard constraint
# ----------------------------------------------------------------------
# A HardConstraint is reused from scoring.pareto if available so we don't
# have two slightly-different definitions; we mirror its API here for
# tier-local use and add a per-row "why-it-failed" log.


@dataclass
class TierConstraint:
    """Hard constraint applied between tiers.

    Attributes:
        column: name of the metric value column (e.g. ``fa_dun__mean``).
        op: one of ``"<", "<=", ">", ">=", "==", "!=", "in", "not in"``.
        value: comparator value.
        description: human-readable; used in the per-row failure log.
    """
    column: str
    op: str
    value: float | int | str | tuple
    description: str = ""

    def evaluate(self, series: pd.Series) -> pd.Series:
        col = series.get(self.column)
        if col is None or (isinstance(col, float) and np.isnan(col)):
            # NaN = "no info" = treat as failing the constraint, but
            # log it as missing rather than as a real outlier.
            return pd.Series([False])
        op = self.op
        v = self.value
        if op == "<":   return pd.Series([col < v])
        if op == "<=":  return pd.Series([col <= v])
        if op == ">":   return pd.Series([col > v])
        if op == ">=":  return pd.Series([col >= v])
        if op == "==":  return pd.Series([col == v])
        if op == "!=":  return pd.Series([col != v])
        if op == "in":     return pd.Series([col in v])
        if op == "not in": return pd.Series([col not in v])
        raise ValueError(f"unsupported op {op!r}")


# ----------------------------------------------------------------------
# TierPlan
# ----------------------------------------------------------------------


@dataclass
class TierPlan:
    """Tiered evaluation plan.

    The lists at index ``i`` describe what runs at tier ``i``. The
    constraints at tier ``i`` are applied AFTER the metrics at tier
    ``i`` finish; survivor_topk at tier ``i`` is applied AFTER the
    constraints (sorted by ``rank_score_col`` ascending if not None,
    else by the first metric column added in that tier).

    Attributes:
        tiers: per-tier list of MetricSpec.
        constraints_per_tier: per-tier list of TierConstraint;
            ``constraints_per_tier[i]`` is applied after ``tiers[i]``.
            Use ``[]`` for no gate.
        survivor_topk_per_tier: how many candidates to KEEP after the
            constraints at this tier. ``None`` keeps all that passed.
            Use small numbers (5-20) for the heaviest tiers.
        rank_score_col_per_tier: optional metric column used to rank
            for the survivor_topk pick at each tier. If None, the
            tier ranks by the first metric in ``tiers[i]``'s prefix.
        rank_ascending_per_tier: True = lower is better (default for
            energy-like metrics like fa_dun); False = higher is better
            (e.g. naturalness log-prob).
        tier_names: cosmetic names for logging.
    """
    tiers: list[list[MetricSpec]]
    constraints_per_tier: list[list[TierConstraint]] = field(default_factory=list)
    survivor_topk_per_tier: list[Optional[int]] = field(default_factory=list)
    rank_score_col_per_tier: list[Optional[str]] = field(default_factory=list)
    rank_ascending_per_tier: list[bool] = field(default_factory=list)
    tier_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.tiers)
        if not self.constraints_per_tier:
            self.constraints_per_tier = [[] for _ in range(n)]
        if not self.survivor_topk_per_tier:
            self.survivor_topk_per_tier = [None] * n
        if not self.rank_score_col_per_tier:
            self.rank_score_col_per_tier = [None] * n
        if not self.rank_ascending_per_tier:
            self.rank_ascending_per_tier = [True] * n
        if not self.tier_names:
            self.tier_names = [f"tier_{i}" for i in range(n)]
        # validate
        for lst, kind in [
            (self.constraints_per_tier, "constraints_per_tier"),
            (self.survivor_topk_per_tier, "survivor_topk_per_tier"),
            (self.rank_score_col_per_tier, "rank_score_col_per_tier"),
            (self.rank_ascending_per_tier, "rank_ascending_per_tier"),
            (self.tier_names, "tier_names"),
        ]:
            if len(lst) != n:
                raise ValueError(
                    f"{kind} has {len(lst)} elements, expected {n}"
                )


# ----------------------------------------------------------------------
# Result + log types
# ----------------------------------------------------------------------


@dataclass
class TierLogEntry:
    """One row of the tier_log returned by evaluate_tiered."""
    tier_idx: int
    tier_name: str
    n_candidates_in: int
    n_candidates_passed_constraints: int
    n_candidates_kept: int        # after survivor_topk
    metrics_run: list[str]
    wall_clock_seconds: float
    cache_hits: int
    cache_misses: int


@dataclass
class TierEvaluationResult:
    """What :func:`evaluate_tiered` returns.

    Attributes:
        metrics_df: long-form per-candidate DataFrame with
            ``candidate_id`` + every metric's prefixed columns. Rows
            for candidates that were dropped at an early tier still
            appear, with NaNs for the metrics they didn't reach (so
            consumers can do ``df[df['fa_dun__failed']==0]`` etc.
            without losing rows).
        survivors: list of candidate_ids that made it through the
            final tier.
        tier_log: per-tier diagnostics.
        constraint_failures: long-form table of (candidate_id, tier,
            constraint_column, constraint_op, value, observed_value)
            for every hard-constraint failure across the run -- useful
            for diagnosing stuck optimization (Antipattern #4).
    """
    metrics_df: pd.DataFrame
    survivors: list[str]
    tier_log: list[TierLogEntry]
    constraint_failures: pd.DataFrame


# ----------------------------------------------------------------------
# evaluate_tiered
# ----------------------------------------------------------------------


def evaluate_tiered(
    candidates: Iterable[Candidate],
    plan: TierPlan,
    cache: Optional[MetricCache] = None,
    *,
    extra_params_per_metric: Optional[dict[str, dict]] = None,
    verbose: bool = True,
) -> TierEvaluationResult:
    """Run cheap → expensive metrics with hard-constraint gating.

    Args:
        candidates: iterable of Candidate.
        plan: TierPlan describing tiers + gates + survivor_topk.
        cache: optional MetricCache backend; defaults to InMemoryCache.
            For production pipelines pass a JsonlCache pointing at the
            run's output dir so restarts pick up where they left off.
        extra_params_per_metric: overrides applied per metric name on
            top of ``MetricSpec.default_params``.
        verbose: when True, log per-tier survivor counts at INFO level.

    The function NEVER raises on metric failure (each MetricSpec has
    ``soft_fail=True`` by default; failed metrics get a ``<prefix>failed=1``
    column). Constraint violations DROP the candidate from the next
    tier but are logged in ``constraint_failures`` so the caller can
    inspect why a candidate got stuck.
    """
    # Be explicit: JsonlCache.__len__() returns 0 on a fresh instance,
    # which is falsy, so `cache or InMemoryCache()` would silently swap
    # the user's persistent cache for an in-memory one. Use `is None`.
    if cache is None:
        cache = InMemoryCache()
    extra = extra_params_per_metric or {}
    candidates = list(candidates)
    if not candidates:
        return TierEvaluationResult(
            metrics_df=pd.DataFrame(),
            survivors=[],
            tier_log=[],
            constraint_failures=pd.DataFrame(),
        )

    # Reject duplicate candidate_ids -- they'd silently merge metric
    # rows in the same dict and corrupt survivor selection.
    seen_ids: set[str] = set()
    for c in candidates:
        if c.candidate_id in seen_ids:
            raise ValueError(
                f"duplicate candidate_id {c.candidate_id!r} in input -- "
                "candidate_ids must be unique within a single evaluate_tiered call"
            )
        seen_ids.add(c.candidate_id)

    # one row per candidate, accumulating columns as tiers run.
    metrics_rows: dict[str, dict] = {
        c.candidate_id: {"candidate_id": c.candidate_id} for c in candidates
    }
    survivors: list[Candidate] = list(candidates)
    tier_log: list[TierLogEntry] = []
    constraint_failures: list[dict] = []

    for tier_idx, tier_metrics in enumerate(plan.tiers):
        n_in = len(survivors)
        if n_in == 0:
            tier_log.append(TierLogEntry(
                tier_idx=tier_idx,
                tier_name=plan.tier_names[tier_idx],
                n_candidates_in=0,
                n_candidates_passed_constraints=0,
                n_candidates_kept=0,
                metrics_run=[s.name for s in tier_metrics],
                wall_clock_seconds=0.0,
                cache_hits=0,
                cache_misses=0,
            ))
            continue

        if verbose:
            LOGGER.info(
                "tier %d (%s): %d candidates -> [%s]",
                tier_idx, plan.tier_names[tier_idx], n_in,
                ", ".join(s.name for s in tier_metrics),
            )

        t0 = time.perf_counter()
        cache_hits = 0
        cache_misses = 0

        # Group GPU specs together for batching opportunity (we don't
        # batch within a single MetricSpec call here -- that's the
        # individual wrapper's responsibility -- but we DO order GPU
        # calls back-to-back to avoid alternating sif startups).
        gpu_specs = [s for s in tier_metrics if s.needs_gpu]
        cpu_specs = [s for s in tier_metrics if not s.needs_gpu]
        ordered = cpu_specs + gpu_specs

        for spec in ordered:
            params = extra.get(spec.name, {})
            for cand in survivors:
                # Cache check first (counts hits/misses for telemetry).
                from protein_chisel.scoring.cache import make_cache_key
                key = make_cache_key(spec, cand, params)
                if cache.get(key) is not None:
                    cache_hits += 1
                else:
                    cache_misses += 1
                result = call_metric_cached(spec, cand, cache, params=params)
                metrics_rows[cand.candidate_id].update(result.values)

        elapsed = time.perf_counter() - t0

        # Apply constraints AFTER all metrics in this tier finish.
        passing_ids: list[str] = []
        for cand in survivors:
            row = pd.Series(metrics_rows[cand.candidate_id])
            ok = True
            for constraint in plan.constraints_per_tier[tier_idx]:
                if not bool(constraint.evaluate(row).iloc[0]):
                    ok = False
                    observed = row.get(constraint.column)
                    failure_reason = _classify_constraint_failure(
                        observed, constraint
                    )
                    constraint_failures.append({
                        "candidate_id": cand.candidate_id,
                        "tier_idx": tier_idx,
                        "tier_name": plan.tier_names[tier_idx],
                        "constraint_column": constraint.column,
                        "constraint_op": constraint.op,
                        "constraint_value": _serialize_value(constraint.value),
                        "observed_value": _serialize_value(observed),
                        "failure_reason": failure_reason,
                        "description": constraint.description,
                    })
                    # don't break -- log all violations so we know which
                    # constraints are co-violated.
            if ok:
                passing_ids.append(cand.candidate_id)

        n_passed = len(passing_ids)

        # Survivor top-K within those passing.
        topk = plan.survivor_topk_per_tier[tier_idx]
        ranking_col = (
            plan.rank_score_col_per_tier[tier_idx]
            or _first_metric_col_in(tier_metrics, metrics_rows)
        )
        ascending = plan.rank_ascending_per_tier[tier_idx]
        if topk is not None and topk < n_passed and ranking_col:
            ranked_rows = []
            for cid in passing_ids:
                row = metrics_rows[cid]
                v = row.get(ranking_col)
                # Coerce numeric and treat NaN / None / non-numeric / failed
                # rows as "worst possible" so they can never win top-K
                # regardless of sort direction.
                failed = bool(_row_has_failed_metric(row, tier_metrics))
                try:
                    fv = float(v) if v is not None else None
                    if fv is None or fv != fv:   # None or NaN
                        score = float("inf") if ascending else float("-inf")
                    elif failed:
                        score = float("inf") if ascending else float("-inf")
                    else:
                        score = fv
                except (TypeError, ValueError):
                    score = float("inf") if ascending else float("-inf")
                ranked_rows.append((cid, score))
            ranked_rows.sort(key=lambda x: x[1], reverse=not ascending)
            kept_ids = {cid for cid, _ in ranked_rows[:topk]}
            survivors = [c for c in survivors if c.candidate_id in kept_ids]
        else:
            survivors = [c for c in survivors if c.candidate_id in set(passing_ids)]

        n_kept = len(survivors)
        if verbose:
            LOGGER.info(
                "tier %d (%s) done in %.2fs: in=%d passed=%d kept=%d "
                "(cache_hits=%d misses=%d)",
                tier_idx, plan.tier_names[tier_idx], elapsed,
                n_in, n_passed, n_kept, cache_hits, cache_misses,
            )

        tier_log.append(TierLogEntry(
            tier_idx=tier_idx,
            tier_name=plan.tier_names[tier_idx],
            n_candidates_in=n_in,
            n_candidates_passed_constraints=n_passed,
            n_candidates_kept=n_kept,
            metrics_run=[s.name for s in tier_metrics],
            wall_clock_seconds=elapsed,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        ))

    metrics_df = pd.DataFrame(list(metrics_rows.values()))
    # Ensure constraint_failures has stable canonical columns even when
    # zero failures occurred -- so to_csv produces a parseable file
    # (with headers but no rows) instead of zero bytes.
    canonical_failure_cols = [
        "candidate_id", "tier_idx", "tier_name", "constraint_column",
        "constraint_op", "constraint_value", "observed_value",
        "failure_reason", "description",
    ]
    if constraint_failures:
        cf_df = pd.DataFrame(constraint_failures)
    else:
        cf_df = pd.DataFrame(columns=canonical_failure_cols)
    return TierEvaluationResult(
        metrics_df=metrics_df,
        survivors=[c.candidate_id for c in survivors],
        tier_log=tier_log,
        constraint_failures=cf_df,
    )


def _first_metric_col_in(
    specs: list[MetricSpec], rows: dict[str, dict],
) -> Optional[str]:
    """Pick a default ranking column.

    Rules:
      1. Skip the universal ``<prefix>failed`` flag -- it's a sentinel,
         not a quality score.
      2. Prefer numeric columns; fall back to anything else only as a
         last resort.
      3. Return the first match in spec-list order so deterministic
         ordering across runs.

    If nothing usable is found (e.g. all candidates had soft-failed
    metrics), return ``None`` -- caller will skip top-K and just keep
    everyone that passed constraints.
    """
    for spec in specs:
        for row in rows.values():
            for k, v in row.items():
                if not k.startswith(spec.prefix):
                    continue
                if k == f"{spec.prefix}failed":
                    continue
                if isinstance(v, (int, float, bool)) and not isinstance(v, str):
                    return k
    # Fall back: any prefixed key, even if non-numeric. The top-K
    # ranker has its own coerce-to-float guard, so a non-numeric
    # column will just push everyone to the "worst" tier.
    for spec in specs:
        for row in rows.values():
            for k in row:
                if k.startswith(spec.prefix) and k != f"{spec.prefix}failed":
                    return k
    return None


def _row_has_failed_metric(row: dict, specs: list[MetricSpec]) -> bool:
    """True if any metric in ``specs`` reported failed=1 for this row."""
    for spec in specs:
        if row.get(f"{spec.prefix}failed") == 1:
            return True
    return False


def _classify_constraint_failure(observed, constraint: TierConstraint) -> str:
    """Categorise the reason a constraint failed, for diagnostics.

    Categories:
        - "missing"    -- column absent from the row (metric never ran,
                          or upstream tier silently dropped it)
        - "nan"        -- column present but NaN (metric soft-failed)
        - "violation"  -- column present, numeric, just doesn't meet
                          the comparator
    """
    if observed is None:
        return "missing"
    try:
        if isinstance(observed, float) and observed != observed:  # NaN check
            return "nan"
    except TypeError:
        pass
    return "violation"


def _serialize_value(v):
    """Make a constraint observed/expected value safe for TSV/JSON.

    NaN -> "nan" string so it round-trips through CSV without dtype
    issues. None stays None.
    """
    if v is None:
        return None
    try:
        if isinstance(v, float) and v != v:
            return "nan"
    except TypeError:
        pass
    if isinstance(v, (tuple, list)):
        return list(v)
    return v


# ----------------------------------------------------------------------
# Convenience: serialise / deserialise a TierPlan from yaml-ish dict
# ----------------------------------------------------------------------


def write_tier_log(out_dir: Path, log: list[TierLogEntry]) -> Path:
    """Persist the tier log as a TSV + a JSON summary."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "tier_idx": e.tier_idx,
            "tier_name": e.tier_name,
            "n_candidates_in": e.n_candidates_in,
            "n_candidates_passed_constraints": e.n_candidates_passed_constraints,
            "n_candidates_kept": e.n_candidates_kept,
            "metrics_run": ",".join(e.metrics_run),
            "wall_clock_seconds": round(e.wall_clock_seconds, 2),
            "cache_hits": e.cache_hits,
            "cache_misses": e.cache_misses,
        }
        for e in log
    ]
    out_path = out_dir / "tier_log.tsv"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    return out_path


__all__ = [
    "TierConstraint",
    "TierEvaluationResult",
    "TierLogEntry",
    "TierPlan",
    "evaluate_tiered",
    "write_tier_log",
]
