"""tier_filter pipeline: rank candidates through cheap → expensive tiers.

Pipeline #1 from the architectural review. Single-pass: take a list of
candidates, run a TierPlan against them, persist outputs, return the
survivors.

What this pipeline owns
=======================
- The bridge between user-friendly inputs (a list of PDB paths +
  optional sequences + catalytic resnos) and the :class:`Candidate`
  objects that :func:`evaluate_tiered` consumes.
- A canonical TierPlan factory (:func:`default_enzyme_tier_plan`) that
  encodes the cheap-first / expensive-late ordering recommended by the
  architectural review.
- Persistence of all four output artifacts to the run directory:
    - ``metrics.tsv``           -- one row per candidate, all metrics
    - ``survivors.tsv``         -- the candidate_ids that passed every tier
    - ``tier_log.tsv``          -- per-tier wall-clock + survivor counts
    - ``constraint_failures.tsv`` -- diagnostics for why candidates were dropped
    - ``manifest.json``         -- run metadata + provenance

What this pipeline does NOT own
================================
- It does NOT generate candidates. The caller passes pre-existing PDBs.
  For "design + filter" workflows, layer this on top of
  :func:`sampling.biased_mpnn.biased_sample` (LigandMPNN with PLM-fused
  bias produces candidates -> tier_filter ranks them).
- It does NOT decide whether to run AF3 on the survivors. Returns
  the survivor candidate_ids; the caller chooses what to do next
  (e.g. submit an AF3 sbatch job for the top-K).

Output schema
=============
Both ``metrics.tsv`` and ``survivors.tsv`` are flat TSV. The
``manifest.json`` includes::

    {
      "pipeline": "tier_filter",
      "git_commit": "<hash>",
      "started_at": "<iso8601>",
      "finished_at": "<iso8601>",
      "n_candidates_in": int,
      "n_survivors": int,
      "tier_plan": [...],          # serialised TierPlan summary
      "cache_path": "<path>",
      "tier_log": [...]            # per-tier diagnostics
    }
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from protein_chisel.scoring.cache import JsonlCache, MetricCache
from protein_chisel.scoring.metrics import Candidate, MetricRegistry, get_registry
from protein_chisel.scoring.tier import (
    TierConstraint,
    TierEvaluationResult,
    TierPlan,
    evaluate_tiered,
    write_tier_log,
)


LOGGER = logging.getLogger("protein_chisel.tier_filter")


# ----------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------


@dataclass
class CandidateInput:
    """User-facing description of one input candidate.

    Pipelines accept a list of these and convert internally to
    :class:`Candidate`. The split exists because pipelines often want
    to sanity-check the inputs (e.g. ``structure_path.is_file()``)
    before kicking off the run.
    """
    candidate_id: str
    structure_path: str | Path
    sequence: Optional[str] = None
    ligand_params_path: Optional[str | Path] = None
    catalytic_resnos: Iterable[int] = ()
    parent_design_id: Optional[str] = None


@dataclass
class TierFilterConfig:
    """Knobs for :func:`tier_filter`.

    Attributes:
        out_dir: where to write metrics.tsv / tier_log.tsv / etc.
            Created if missing. ``cache_path`` defaults to
            ``out_dir / "metric_cache.jsonl"`` so distributed sbatch
            workers writing into the same out_dir share the cache.
        cache: explicit cache backend; if None, a JsonlCache rooted
            at ``out_dir / "metric_cache.jsonl"`` is created.
        verbose: log per-tier survivor counts + cache hit/miss totals.
        write_outputs: when False, skip persistence entirely (useful
            for unit tests).
    """
    out_dir: str | Path
    cache: Optional[MetricCache] = None
    verbose: bool = True
    write_outputs: bool = True


@dataclass
class TierFilterResult:
    """Return type for :func:`tier_filter`.

    Wraps :class:`TierEvaluationResult` plus persistence-side metadata
    (manifest path, where outputs landed) so callers can chain.
    """
    evaluation: TierEvaluationResult
    out_dir: Optional[Path]
    manifest_path: Optional[Path]
    survivors_df: pd.DataFrame  # candidate_id + ranking metric columns


# ----------------------------------------------------------------------
# Default tier plan (canonical cheap-first / expensive-late recipe)
# ----------------------------------------------------------------------


def default_enzyme_tier_plan(
    *,
    registry: Optional[MetricRegistry] = None,
    fa_dun_max_mean: float = 5.0,
    fa_dun_max_severe_outliers: int = 5,
    rotalyze_max_outlier_frac: float = 0.10,
    fpocket_min_druggability: float = 0.2,
    survivor_topk_after_cheap: Optional[int] = None,
    survivor_topk_after_pocket: Optional[int] = None,
    survivor_topk_after_likelihood: Optional[int] = None,
    enable_pippack: bool = True,
    enable_flowpacker: bool = True,
    enable_attnpacker: bool = False,
) -> TierPlan:
    """Build the canonical 4-tier plan recommended by the architectural review.

    Tiers:
      0 -- CPU + cheap structural primitives: rotamer_score (fa_dun)
      1 -- CPU + cheap KDE: rotalyze
      2 -- CPU + structural geometry: fpocket
      3 -- GPU + learned likelihoods: pippack, flowpacker (and optionally
           attnpacker)

    Each tier has soft hard-constraints that gate the next tier:
      after tier 0  ->  fa_dun mean <= ``fa_dun_max_mean`` AND
                        n_severe_outliers <= ``fa_dun_max_severe_outliers``
      after tier 1  ->  rotalyze frac_outliers <= ``rotalyze_max_outlier_frac``
      after tier 2  ->  fpocket most_druggable_score >= ``fpocket_min_druggability``
      after tier 3  ->  no hard gate (the survivor list is what AF3
                        receives; the caller can apply objective
                        weights post-hoc).

    Survivor top-K can downselect after each tier; defaults to None
    (keep all that passed). For 50-100 designs/round, sensible values
    are something like ``50, 25, 10`` after cheap/pocket/likelihood.
    """
    from protein_chisel.tools.sidechain_packing_and_scoring.metric_specs import (
        ATTNPACKER_SCORE_SPEC,
        FLOWPACKER_SCORE_SPEC,
        FPOCKET_SPEC,
        PIPPACK_SCORE_SPEC,
        ROTALYZE_SPEC,
        ROTAMER_SCORE_SPEC,
        register_default_specs,
    )

    reg = registry if registry is not None else get_registry()
    register_default_specs(reg)

    tier_0 = [ROTAMER_SCORE_SPEC]
    tier_1 = [ROTALYZE_SPEC]
    tier_2 = [FPOCKET_SPEC]
    tier_3: list = []
    if enable_pippack:
        tier_3.append(PIPPACK_SCORE_SPEC)
    if enable_flowpacker:
        tier_3.append(FLOWPACKER_SCORE_SPEC)
    if enable_attnpacker:
        tier_3.append(ATTNPACKER_SCORE_SPEC)

    constraints = [
        # tier 0 gate
        [
            TierConstraint(
                "rotamer_score__mean_fa_dun", "<=", fa_dun_max_mean,
                description="fa_dun mean reasonable",
            ),
            TierConstraint(
                "rotamer_score__n_severe_outliers", "<=",
                fa_dun_max_severe_outliers,
                description="few severe rotamer outliers",
            ),
        ],
        # tier 1 gate
        [
            TierConstraint(
                "rotalyze__frac_outliers", "<=", rotalyze_max_outlier_frac,
                description="MolProbity outlier frac reasonable",
            ),
        ],
        # tier 2 gate
        [
            TierConstraint(
                "fpocket__most_druggable_score", ">=", fpocket_min_druggability,
                description="binding pocket exists + somewhat druggable",
            ),
        ],
        # tier 3 -- no gate (let the caller rank/score after)
        [],
    ]

    plan = TierPlan(
        tiers=[tier_0, tier_1, tier_2, tier_3],
        constraints_per_tier=constraints,
        survivor_topk_per_tier=[
            survivor_topk_after_cheap,
            None,                                # tier 1 keeps all that passed
            survivor_topk_after_pocket,
            survivor_topk_after_likelihood,
        ],
        rank_score_col_per_tier=[
            "rotamer_score__mean_fa_dun",        # lower fa_dun = better
            None,
            "fpocket__most_druggable_score",     # higher druggability = better
            None,                                # tier 3: no built-in rank;
                                                  # caller chooses (logp_mean,
                                                  # rotamer_recovery, ...)
        ],
        rank_ascending_per_tier=[True, True, False, True],
        tier_names=["cheap_cpu", "cheap_kde", "pocket_geom", "learned_likelihood"],
    )
    return plan


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def tier_filter(
    candidates: Iterable[CandidateInput],
    plan: TierPlan,
    config: Optional[TierFilterConfig] = None,
) -> TierFilterResult:
    """Rank candidates through ``plan`` and persist outputs.

    Args:
        candidates: iterable of CandidateInput. Each must have a
            structure_path that exists.
        plan: TierPlan to evaluate.
        config: TierFilterConfig (out_dir, cache backend, verbose).

    Returns:
        TierFilterResult with the underlying TierEvaluationResult
        + the path to the manifest + a survivors-only DataFrame
        (sorted by the final tier's ranking column when available).

    Persistence:
        - ``out_dir/metrics.tsv`` -- per-candidate metric table (all rows)
        - ``out_dir/survivors.tsv`` -- survivor rows only
        - ``out_dir/tier_log.tsv`` -- per-tier diagnostics
        - ``out_dir/constraint_failures.tsv`` -- per-failure log
        - ``out_dir/manifest.json`` -- run metadata
    """
    if config is None:
        raise ValueError("tier_filter requires a TierFilterConfig (out_dir at minimum)")

    out_dir = Path(config.out_dir).resolve()
    if config.write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Set up the cache. Default to a JsonlCache in out_dir so multiple
    # workers writing to the same out_dir share warmth automatically.
    # When the caller passes an explicit cache, it wins -- but log if
    # we're shadowing an on-disk cache so the surprise is visible.
    cache = config.cache
    auto_cache_path = out_dir / "metric_cache.jsonl"
    if cache is None and config.write_outputs:
        cache = JsonlCache(auto_cache_path)
    elif cache is not None and auto_cache_path.is_file():
        LOGGER.info(
            "tier_filter: explicit cache %r shadows existing %s; "
            "no entries from the on-disk file will be consulted",
            cache, auto_cache_path,
        )

    candidate_inputs = list(candidates)
    if not candidate_inputs:
        raise ValueError("tier_filter received an empty candidate list")

    # Surface duplicate-id errors early (evaluate_tiered also catches
    # this, but we want to fail before any disk I/O so the user gets
    # a clear message).
    seen_ids: set[str] = set()
    for ci in candidate_inputs:
        if ci.candidate_id in seen_ids:
            raise ValueError(
                f"duplicate candidate_id {ci.candidate_id!r} in input -- "
                "ids must be unique within a single tier_filter run"
            )
        seen_ids.add(ci.candidate_id)

    # Validate inputs before starting (cheaper to fail loudly than to
    # crash mid-tier on a missing file).
    cands: list[Candidate] = []
    for ci in candidate_inputs:
        sp = Path(ci.structure_path)
        if not sp.is_file():
            raise FileNotFoundError(
                f"candidate {ci.candidate_id!r}: structure_path {sp} does not exist"
            )
        if ci.ligand_params_path is not None:
            lp = Path(ci.ligand_params_path)
            if not lp.is_file():
                raise FileNotFoundError(
                    f"candidate {ci.candidate_id!r}: ligand_params_path {lp} does not exist"
                )
        cands.append(Candidate(
            candidate_id=ci.candidate_id,
            sequence=ci.sequence,
            structure_path=sp.resolve(),
            ligand_params_path=Path(ci.ligand_params_path).resolve() if ci.ligand_params_path else None,
            catalytic_resnos=tuple(sorted(int(r) for r in ci.catalytic_resnos)),
            parent_design_id=ci.parent_design_id,
        ))

    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    LOGGER.info(
        "tier_filter starting: n_candidates=%d, out_dir=%s",
        len(cands), out_dir,
    )

    evaluation = evaluate_tiered(cands, plan, cache=cache, verbose=config.verbose)

    finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

    survivor_set = set(evaluation.survivors)
    survivors_df = evaluation.metrics_df[
        evaluation.metrics_df["candidate_id"].isin(survivor_set)
    ].copy()

    # Sort survivors by the LAST tier's ranking column when available,
    # so the top of the file is the design we'd send to AF3 first. Use
    # the matching tier's `rank_ascending` -- the previous version used
    # the last element unconditionally, which silently inverted sort
    # direction whenever the last tier had rank_col=None and an earlier
    # tier had a different ascending value.
    last_rank_col: Optional[str] = None
    last_rank_ascending = True
    for i in range(len(plan.rank_score_col_per_tier) - 1, -1, -1):
        col = plan.rank_score_col_per_tier[i]
        if col is not None:
            last_rank_col = col
            last_rank_ascending = plan.rank_ascending_per_tier[i]
            break
    if last_rank_col and last_rank_col in survivors_df.columns:
        survivors_df = survivors_df.sort_values(
            last_rank_col, ascending=last_rank_ascending, na_position="last",
        )

    manifest_path: Optional[Path] = None
    if config.write_outputs:
        manifest_path = _persist_outputs(
            out_dir=out_dir,
            evaluation=evaluation,
            survivors_df=survivors_df,
            plan=plan,
            started_at=started_at,
            finished_at=finished_at,
            n_in=len(cands),
        )

    LOGGER.info(
        "tier_filter done: n_survivors=%d (of %d), out_dir=%s",
        len(survivor_set), len(cands), out_dir,
    )

    return TierFilterResult(
        evaluation=evaluation,
        out_dir=out_dir if config.write_outputs else None,
        manifest_path=manifest_path,
        survivors_df=survivors_df,
    )


# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------


# Manifest schema version. Bump when the manifest fields change in
# breaking ways (column renamed, structure flattened, etc.).
MANIFEST_SCHEMA_VERSION = 1


def _persist_outputs(
    *,
    out_dir: Path,
    evaluation: TierEvaluationResult,
    survivors_df: pd.DataFrame,
    plan: TierPlan,
    started_at: str,
    finished_at: str,
    n_in: int,
) -> Path:
    """Write metrics.tsv / survivors.tsv / tier_log.tsv / constraint_failures.tsv
    + manifest.json.

    Files are TSV (not CSV) so commas in descriptions don't break parsers.
    NaN -> empty string (pandas default).

    Each TSV is written via ``write -> rename`` so concurrent sbatch
    workers (sharing the same out_dir) never see a torn / partially-
    written file. If a worker crashes mid-write, the ``.tmp`` file is
    left behind for cleanup but the final artifact is intact.

    The manifest carries a ``schema_version`` + ``artifacts`` block
    (path + sha256 + n_rows of each output) so downstream consumers
    can verify they're reading what this run produced, not a stale
    file left over from an earlier invocation.
    """
    artifacts: dict[str, dict] = {}

    artifacts["metrics_tsv"] = _atomic_write_tsv(
        evaluation.metrics_df, out_dir / "metrics.tsv",
    )
    artifacts["survivors_tsv"] = _atomic_write_tsv(
        survivors_df, out_dir / "survivors.tsv",
    )
    # Always write constraint_failures.tsv -- a re-run with zero
    # failures must NOT leave an old failure table behind, or downstream
    # consumers reading the file will think the new run had failures.
    artifacts["constraint_failures_tsv"] = _atomic_write_tsv(
        evaluation.constraint_failures, out_dir / "constraint_failures.tsv",
    )
    write_tier_log(out_dir, evaluation.tier_log)
    # tier_log.tsv is small + atomic-enough; record sha256 for completeness.
    tl_path = out_dir / "tier_log.tsv"
    if tl_path.is_file():
        artifacts["tier_log_tsv"] = _file_artifact_meta(tl_path)

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "pipeline": "tier_filter",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "n_candidates_in": int(n_in),
        "n_survivors": int(len(survivors_df)),
        "git_commit": _cached_git_commit(),
        "tier_plan": _serialise_tier_plan(plan),
        "tier_log": [
            {
                "tier_idx": e.tier_idx,
                "tier_name": e.tier_name,
                "n_in": e.n_candidates_in,
                "n_passed": e.n_candidates_passed_constraints,
                "n_kept": e.n_candidates_kept,
                "metrics_run": e.metrics_run,
                "wall_clock_seconds": round(e.wall_clock_seconds, 2),
                "cache_hits": e.cache_hits,
                "cache_misses": e.cache_misses,
            }
            for e in evaluation.tier_log
        ],
        "artifacts": artifacts,
    }
    # Manifest itself written atomically so an in-flight readers can't
    # see a half-written JSON doc.
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, default=str))
    tmp_manifest.replace(manifest_path)
    return manifest_path


def _atomic_write_tsv(df: pd.DataFrame, path: Path) -> dict:
    """Write a DataFrame to a TSV via tmp+rename so partial writes are
    invisible to concurrent readers.

    Returns ``{path, sha256, n_rows}`` for the manifest's ``artifacts``
    block. Caller is responsible for ensuring ``path.parent`` exists.
    """
    import hashlib
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)  # POSIX-atomic on the same filesystem
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return {
        "path": str(path.name),
        "sha256": h.hexdigest(),
        "n_rows": int(len(df)),
    }


def _file_artifact_meta(path: Path) -> dict:
    """Generate the ``{path, sha256}`` block for a non-DataFrame file."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return {"path": str(path.name), "sha256": h.hexdigest()}


def _serialise_tier_plan(plan: TierPlan) -> list[dict]:
    """JSON-friendly summary of the TierPlan (for the manifest)."""
    out = []
    for i, tier in enumerate(plan.tiers):
        out.append({
            "tier_idx": i,
            "tier_name": plan.tier_names[i],
            "metrics": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "needs_gpu": s.needs_gpu,
                    "cost_seconds": s.cost_seconds,
                    "cache_version": s.cache_version,
                }
                for s in tier
            ],
            "constraints": [
                {
                    "column": c.column,
                    "op": c.op,
                    "value": c.value,
                    "description": c.description,
                }
                for c in plan.constraints_per_tier[i]
            ],
            "survivor_topk": plan.survivor_topk_per_tier[i],
            "rank_score_col": plan.rank_score_col_per_tier[i],
            "rank_ascending": plan.rank_ascending_per_tier[i],
        })
    return out


_GIT_COMMIT_CACHE: Optional[str] = None
_GIT_COMMIT_RESOLVED = False


def _cached_git_commit() -> Optional[str]:
    """Return the git HEAD commit at module-load time, cached.

    Resolved exactly once per process. Avoids paying a subprocess
    fork on every pipeline invocation when the same Python interpreter
    is running many tier_filter calls (e.g. inside an MH loop).
    """
    global _GIT_COMMIT_CACHE, _GIT_COMMIT_RESOLVED
    if _GIT_COMMIT_RESOLVED:
        return _GIT_COMMIT_CACHE
    try:
        _GIT_COMMIT_CACHE = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],  # repo root
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        _GIT_COMMIT_CACHE = None
    _GIT_COMMIT_RESOLVED = True
    return _GIT_COMMIT_CACHE


__all__ = [
    "CandidateInput",
    "TierFilterConfig",
    "TierFilterResult",
    "default_enzyme_tier_plan",
    "tier_filter",
]
