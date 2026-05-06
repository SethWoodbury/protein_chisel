"""Multi-objective ranking via Pareto-front + TOPSIS tie-breaking.

Currently iterative_design_v2 sorts by ``(fitness desc, alpha_radius asc)``,
which silently collapses many objectives to two. This module exposes a
principled alternative:

    1. Drop rows failing any hard constraint.
    2. Compute Pareto-front membership across N objectives (each tagged
       MAX or MIN).
    3. Within the front, score every row by TOPSIS (closeness to ideal).
    4. Outside the front, fall back to TOPSIS so all rows get a global
       rank but front-members always rank above non-front members.

The function is pure (DataFrame in, DataFrame out, no side effects) and
deliberately doesn't depend on protein_chisel. Plug into the driver via:

    out = topsis_pareto_rank(
        ranked_df,
        [Objective("fitness__logp_fused_mean", "max", weight=2.0),
         Objective("fpocket__druggability", "max"),
         Objective("sap_max", "min"),
         Objective("net_charge_no_HIS", "min"),
         Objective("fpocket__n_alpha_spheres_near_catalytic", "max")],
    )
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass
class Objective:
    column: str
    direction: str = "max"   # "max" or "min"
    weight: float = 1.0


def _normalize(M: np.ndarray) -> np.ndarray:
    """Vector-normalize each column (TOPSIS step)."""
    norms = np.linalg.norm(M, axis=0)
    norms = np.where(norms == 0.0, 1.0, norms)
    return M / norms


def _pareto_mask(M: np.ndarray) -> np.ndarray:
    """Naive O(N^2) Pareto mask. M[i, j] is the j-th objective for design
    i, ALREADY oriented so larger is better (we flip MIN cols upstream)."""
    N = M.shape[0]
    on_front = np.ones(N, dtype=bool)
    for i in range(N):
        if not on_front[i]:
            continue
        # any other point strictly dominates i?
        dominates = np.all(M >= M[i], axis=1) & np.any(M > M[i], axis=1)
        if dominates.any():
            on_front[i] = False
    return on_front


def topsis_pareto_rank(
    df: pd.DataFrame, objectives: Sequence[Objective],
) -> pd.DataFrame:
    """Return df with added columns: ``mo_on_pareto``, ``mo_topsis``,
    ``mo_rank`` (1 = best). NaN-rows in any objective are sent to the
    bottom but kept in the table."""
    if len(objectives) == 0:
        raise ValueError("need >=1 objective")
    cols = [o.column for o in objectives]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing objective columns: {missing}")
    raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(raw).any(axis=1)
    M = raw[valid].copy()
    # Orient: every column larger-is-better.
    for j, o in enumerate(objectives):
        if o.direction == "min":
            M[:, j] = -M[:, j]
        elif o.direction != "max":
            raise ValueError(f"objective.direction must be 'max' or 'min', got {o.direction!r}")
    # TOPSIS on the oriented matrix
    Mn = _normalize(M)
    w = np.array([o.weight for o in objectives], dtype=float)
    Mw = Mn * w[None, :]
    ideal = Mw.max(axis=0)
    nadir = Mw.min(axis=0)
    d_pos = np.linalg.norm(Mw - ideal, axis=1)
    d_neg = np.linalg.norm(Mw - nadir, axis=1)
    denom = d_pos + d_neg
    topsis = np.where(denom > 0, d_neg / np.where(denom == 0, 1.0, denom), 0.5)
    on_front = _pareto_mask(M)
    out = df.copy()
    out["mo_on_pareto"] = False
    out["mo_topsis"] = np.nan
    out.loc[valid, "mo_on_pareto"] = on_front
    out.loc[valid, "mo_topsis"] = topsis
    # Final rank: front members first (by topsis desc), then non-front
    # by topsis desc, NaN rows last.
    sort_keys = (~out["mo_on_pareto"].fillna(True)).astype(int) * 1e9
    sort_keys = sort_keys - out["mo_topsis"].fillna(-1.0)
    out["mo_rank"] = sort_keys.rank(method="min").astype("Int64")
    return out


__all__ = ["Objective", "topsis_pareto_rank"]


# ---------------------------------------------------------------------------
# Extended API (2026-05-04): MetricSpec with "target" direction, default
# specs for PTE-style enzyme design, CLI parsing helpers, and active-site-
# aware diverse top-K selection. Backwards-compatible — Objective +
# topsis_pareto_rank above are unchanged.
# ---------------------------------------------------------------------------


from dataclasses import field as _field
from typing import Iterable, Optional
import logging as _logging

LOGGER = _logging.getLogger("protein_chisel.scoring.multi_objective")


@dataclass
class MetricSpec:
    """One axis of multi-objective ranking. Generalizes ``Objective`` with
    a third direction ``"target"``: deviation from a target value is
    treated as a min-objective."""

    column: str
    direction: str = "max"             # "max" | "min" | "target"
    weight: float = 1.0
    target: Optional[float] = None     # only for direction == "target"
    label: Optional[str] = None        # human-readable; defaults to `column`

    def __post_init__(self) -> None:
        if self.direction not in ("max", "min", "target"):
            raise ValueError(f"unknown direction {self.direction!r}")
        if self.direction == "target" and self.target is None:
            raise ValueError(f"target metric {self.column!r} needs a target value")
        if self.label is None:
            self.label = self.column


# Default metric set for PTE / hydrolase de-novo design — matches the
# columns emitted by iterative_design_v2 stage_seq_filter / struct_filter
# / fpocket_rank / fitness. Weights tuned 2026-05-04: fitness is 2× the
# nominal weight; primary structural / catalytic metrics 1×; physico-
# chemical "stay-in-band" target metrics 0.3× each (enough to break
# ties, not enough to outweigh real catalytic signals).
DEFAULT_METRIC_SPECS: list[MetricSpec] = [
    # Primary objectives
    MetricSpec("fitness__logp_fused_mean",  "max",    2.0, label="fitness"),
    MetricSpec("fpocket__druggability",     "max",    1.0, label="druggability"),
    MetricSpec("ligand_int__strength_total","max",    1.0, label="lig_int_strength"),
    MetricSpec("preorg__strength_total",    "max",    0.7, label="preorg_strength"),
    MetricSpec("n_hbonds_to_cat_his",       "max",    0.5, label="hbonds_to_cat"),
    # Stability / aggregation (moderate weight)
    MetricSpec("instability_index",         "min",    0.5, label="instability"),
    MetricSpec("sap_max",                   "min",    0.5, label="sap_max"),
    # Composition stay-in-band (target style)
    MetricSpec("boman_index",               "target", 0.3, target=2.5, label="boman"),
    MetricSpec("aliphatic_index",           "target", 0.3, target=95.0, label="aliphatic"),
    MetricSpec("gravy",                     "target", 0.3, target=-0.2, label="gravy"),
    # Charge / pI (target-style; hard bands enforced earlier)
    MetricSpec("net_charge_full_HH",        "target", 0.3, target=-10.0, label="charge"),
    MetricSpec("pi",                        "target", 0.3, target=5.5,   label="pi"),
    # Pocket geometry
    MetricSpec("fpocket__bottleneck_radius","target", 0.3, target=3.65,  label="bottleneck"),
    MetricSpec("fpocket__hydrophobicity_score", "target", 0.2, target=45.0, label="pocket_hydrophobicity"),
    # Tunnel patency / pocket accessibility (only present when --tunnel_metrics)
    # NaN-tolerant — if these columns are absent the spec is silently skipped.
    # Total weight halved from initial draft (0.6/0.5/0.4/0.4/0.3 = 2.2 was
    # squeezing out catalytic h-bonds and diversity in A/B testing).
    MetricSpec("tunnel__sidechain_blocked_fraction", "min", 0.3, label="tunnel_dsc_blocked"),
    MetricSpec("tunnel__throat_bulky_designable_count", "min", 0.25, label="tunnel_throat_blockers"),
    MetricSpec("tunnel__best_cone_mean_path", "max", 0.2, label="tunnel_best_cone_path"),
    # pyKVFinder-derived signals (NaN if pyKVFinder unavailable)
    MetricSpec("pkvf__cavity_depth_max", "max", 0.2, label="pkvf_cavity_depth"),
    MetricSpec("pkvf__cavity_volume", "max", 0.15, label="pkvf_cavity_volume"),
]


def _normalize_axis(values: np.ndarray, spec: MetricSpec) -> np.ndarray:
    """Normalize values to [0, 1] where 1 = ideal.

    NaN handling: replaced with the column's mean (so NaN rows score
    near-neutral after min-max normalization for direction == 'max'
    or 'min'). For direction == 'target', a NaN takes the mean's
    distance-to-target as its surrogate score — this is non-neutral
    but acceptable in practice (NaN should be rare; if it isn't, the
    spec's column shouldn't be in the ranking basket).
    """
    v = np.where(np.isnan(values), np.nanmean(values) if not np.isnan(values).all() else 0.5, values)
    if spec.direction == "max":
        lo, hi = float(v.min()), float(v.max())
        if hi == lo:
            return np.full_like(v, 0.5)
        return (v - lo) / (hi - lo)
    if spec.direction == "min":
        lo, hi = float(v.min()), float(v.max())
        if hi == lo:
            return np.full_like(v, 0.5)
        return (hi - v) / (hi - lo)
    # target
    t = spec.target
    abs_dev = np.abs(v - t)
    max_dev = float(abs_dev.max())
    if max_dev <= 0:
        return np.full_like(v, 1.0)
    return 1.0 - (abs_dev / max_dev)


def compute_topsis_scores_v2(
    df: pd.DataFrame,
    specs: Iterable[MetricSpec],
) -> tuple[np.ndarray, list[MetricSpec], pd.DataFrame]:
    """Compute TOPSIS scores using MetricSpec (with target direction).

    Returns (scores, used_specs, debug_df). Specs whose column is missing
    from ``df`` are silently dropped with a warning.
    """
    # Empty-pool guard: with zero rows there's nothing to rank, and
    # _normalize_axis() does v.min()/v.max() which raises on empty
    # arrays. Return empty results so callers can detect + skip cleanly.
    if len(df) == 0:
        return np.array([], dtype=float), [], pd.DataFrame()
    used: list[MetricSpec] = []
    cols_norm: list[np.ndarray] = []
    weights: list[float] = []
    for spec in specs:
        if spec.column not in df.columns:
            LOGGER.warning("multi_objective: column %r missing — skipping", spec.column)
            continue
        series = df[spec.column].to_numpy(dtype=float)
        norm = _normalize_axis(series, spec)
        cols_norm.append(norm)
        weights.append(float(spec.weight))
        used.append(spec)
    if not used:
        return np.full(len(df), 0.5), [], pd.DataFrame()
    M = np.stack(cols_norm, axis=1)
    W = np.array(weights) / sum(weights)
    Mw = M * W[None, :]
    ideal = Mw.max(axis=0)
    anti = Mw.min(axis=0)
    d_ideal = np.linalg.norm(Mw - ideal[None, :], axis=1)
    d_anti = np.linalg.norm(Mw - anti[None, :], axis=1)
    score = d_anti / (d_ideal + d_anti + 1e-12)
    debug = pd.DataFrame(M, columns=[s.label for s in used], index=df.index)
    debug["topsis_score"] = score
    return score, used, debug


# ---- Diversity-aware selection -----------------------------------------


def _hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def _hamming_at_positions(a: str, b: str, positions: list[int]) -> int:
    return sum(a[p] != b[p] for p in positions)


def select_diverse_topk_two_axis(
    df: pd.DataFrame,
    *,
    target_k: int,
    min_hamming_full: int = 3,
    primary_sphere_positions: Optional[list[int]] = None,
    min_hamming_active: int = 0,
    score_col: str = "mo_topsis",
    sequence_col: str = "sequence",
) -> pd.DataFrame:
    """Greedy max-min top-K with TWO Hamming constraints (global + active).

    Requires ``df`` already sorted by ``score_col`` descending. Adds an
    active-site Hamming gate alongside the existing global one — designs
    that differ globally but have IDENTICAL active sites are skipped.
    """
    selected_idx: list[int] = []
    selected_seqs: list[str] = []
    for i, row in df.iterrows():
        seq = row[sequence_col]
        if not all(_hamming(seq, s) >= min_hamming_full for s in selected_seqs):
            continue
        if primary_sphere_positions and min_hamming_active > 0:
            if not all(
                _hamming_at_positions(seq, s, primary_sphere_positions)
                >= min_hamming_active
                for s in selected_seqs
            ):
                continue
        selected_idx.append(i)
        selected_seqs.append(seq)
        if len(selected_idx) >= target_k:
            break
    return df.loc[selected_idx].copy()


# ---- CLI parsing ------------------------------------------------------


def parse_kv_string(s: str) -> dict[str, float]:
    """Parse 'key=val,key=val' → dict. Empty / whitespace → {}."""
    if not s or not s.strip():
        return {}
    out: dict[str, float] = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"bad token in kv-string: {tok!r}")
        k, v = tok.split("=", 1)
        out[k.strip()] = float(v)
    return out


def apply_cli_overrides(
    base: list[MetricSpec],
    weights: dict[str, float],
    targets: dict[str, float],
) -> list[MetricSpec]:
    """Apply CLI weight + target overrides. weight=0 drops the metric."""
    out: list[MetricSpec] = []
    for spec in base:
        w = weights.get(spec.label, weights.get(spec.column, spec.weight))
        if w == 0.0:
            LOGGER.info("multi_objective: dropping metric %r (weight=0)", spec.label)
            continue
        new_target = targets.get(spec.label, targets.get(spec.column, spec.target))
        out.append(MetricSpec(
            column=spec.column, direction=spec.direction,
            weight=w, target=new_target, label=spec.label,
        ))
    return out


__all__ = __all__ + [
    "DEFAULT_METRIC_SPECS",
    "MetricSpec",
    "apply_cli_overrides",
    "compute_topsis_scores_v2",
    "parse_kv_string",
    "select_diverse_topk_two_axis",
]
