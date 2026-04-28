"""Metric-specific aggregation across a PoseSet.

Codex's review called out that "average everything" is wrong: a clash in
*any* conformer should fail the whole sequence, while a descriptive
metric (Rg, helix_frac, ...) only needs mean ± std across conformers.
Apo vs holo metrics need explicit paired deltas. Cross-source agreement
(designed-model vs AF3) should be reported separately.

This module implements the policies:

- ``failure``    : worst-case (max), or quantile (95th).
- ``descriptive``: mean, std, min, max — keep all four.
- ``paired``     : explicit delta of two metrics (apo vs holo).
- ``vote``       : fraction of conformers passing a per-metric threshold.

Output: a per-(sequence_id) DataFrame; one row per design, with metric
columns suffixed by aggregation (``__mean``, ``__std``, ``__max``, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ----- aggregation policy declarations --------------------------------------


@dataclass
class AggregationPolicy:
    """Map metric prefix/regex to aggregation strategy.

    Each rule says: for column matching the prefix, apply this strategy.
    First-match wins. Define a default at the end if you want a catch-all.
    """

    rules: list[tuple[str, str]] = field(default_factory=list)
    default: str = "descriptive"  # default aggregation if no rule matches

    def strategy_for(self, col: str) -> str:
        for prefix, strategy in self.rules:
            if col.startswith(prefix):
                return strategy
        return self.default


def default_policy() -> AggregationPolicy:
    """Sensible defaults for our standard metric prefixes."""
    return AggregationPolicy(
        rules=[
            # Failure metrics — any bad conformer fails the design
            ("buns__n_buried_unsat", "failure"),
            ("backbone__chainbreak_above_4_5", "failure"),
            ("catres__n_broken_sidechains", "failure"),
            ("catres__bondlen_max_dev", "failure"),
            ("protease__n_total", "failure"),
            # Descriptive (mean + std + min + max)
            ("shape__", "descriptive"),
            ("ss__", "descriptive"),
            ("ligand__", "descriptive"),
            ("interact__", "descriptive"),
            ("buns__frac_unsat", "descriptive"),
            ("buns__n_buried_polar_total", "descriptive"),
            ("buns__n_whitelisted", "descriptive"),
            ("backbone__chainbreak_max", "descriptive"),
            ("backbone__rCA_nonadj_min", "descriptive"),
            ("catres__cart_bonded_avg", "descriptive"),
            ("catres__fa_dun_avg", "descriptive"),
            ("catres__cart_bonded_max", "descriptive"),
            ("catres__fa_dun_max", "descriptive"),
            # Sequence-only — same for every conformer; pick one
            ("protparam__", "first"),
            ("positions__", "first"),
        ],
        default="descriptive",
    )


# ----- core aggregation functions -------------------------------------------


_NUMERIC_KINDS = ("i", "u", "f", "b")


def aggregate_metric_table(
    metric_table_df: pd.DataFrame,
    policy: Optional[AggregationPolicy] = None,
    group_by: str = "sequence_id",
    keep_per_source: tuple[str, ...] = ("designed", "AF3_seed1", "AF3_refined"),
) -> pd.DataFrame:
    """Aggregate a per-pose MetricTable to per-(sequence_id) rows.

    Args:
        metric_table_df: long-form DataFrame with one row per (sequence_id,
            conformer_index). Must contain `sequence_id`. May contain
            `fold_source`.
        policy: AggregationPolicy mapping column-prefix → strategy.
        group_by: id column to group on.
        keep_per_source: also emit ``__source_<X>__<col>`` for each fold
            source you want to keep visible separately. Lets you preserve
            the designed-model-only number alongside the AF3 mean.
    """
    pol = policy or default_policy()
    out_rows: list[dict] = []

    for sid, group in metric_table_df.groupby(group_by, sort=False):
        row: dict = {group_by: sid}
        n = len(group)
        row["n_conformers"] = n
        if "fold_source" in group.columns:
            row["fold_sources"] = "|".join(sorted(group["fold_source"].astype(str).unique()))

        for col in metric_table_df.columns:
            if col in (group_by, "conformer_index", "fold_source", "is_apo", "pdb_path",
                       "ligand__name3"):
                continue
            series = group[col]
            if series.dtype.kind not in _NUMERIC_KINDS:
                # Non-numeric: just take the first non-null
                non_null = series.dropna()
                if len(non_null) > 0:
                    row[col] = non_null.iloc[0]
                continue

            strategy = pol.strategy_for(col)
            row.update(_apply(strategy, col, series))

        # Per-source metrics if fold_source present
        if "fold_source" in group.columns:
            for src in keep_per_source:
                sub = group[group["fold_source"] == src]
                if len(sub) == 0:
                    continue
                for col in metric_table_df.columns:
                    if col in (group_by, "conformer_index", "fold_source", "is_apo",
                               "pdb_path", "ligand__name3"):
                        continue
                    if sub[col].dtype.kind not in _NUMERIC_KINDS:
                        continue
                    val = float(sub[col].mean())
                    row[f"src__{src}__{col}"] = val

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def _apply(strategy: str, col: str, series: pd.Series) -> dict:
    """Apply one strategy and return prefixed column entries."""
    out: dict = {}
    arr = series.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        # Drop entirely if everything is NaN
        return out
    if strategy == "failure":
        out[f"{col}__max"] = float(arr.max())
        out[f"{col}__any_nonzero"] = bool((arr > 0).any())
    elif strategy == "descriptive":
        out[f"{col}__mean"] = float(arr.mean())
        out[f"{col}__std"] = float(arr.std(ddof=0))
        out[f"{col}__min"] = float(arr.min())
        out[f"{col}__max"] = float(arr.max())
    elif strategy == "first":
        out[col] = arr[0]
    elif strategy == "vote":
        out[f"{col}__pass_frac"] = float((arr > 0).mean())
    elif strategy == "quantile_95":
        out[f"{col}__q95"] = float(np.quantile(arr, 0.95))
    else:
        raise ValueError(f"unknown aggregation strategy {strategy!r}")
    return out


# ----- paired apo / holo deltas --------------------------------------------


def paired_apo_holo_delta(
    metric_table_df: pd.DataFrame,
    metric_columns: Iterable[str],
    pair_on: str = "sequence_id",
    apo_label: str = "apo",
    holo_label: str = "holo",
    fold_source_col: str = "fold_source",
) -> pd.DataFrame:
    """Compute per-design (holo - apo) deltas for selected metric columns.

    The PoseSet must have both an apo and a holo conformer of the same
    sequence_id (e.g. ``fold_source='AF3_apo'`` and ``fold_source='AF3_holo'``,
    or use the ``is_apo`` flag).
    """
    if "is_apo" in metric_table_df.columns:
        apo = metric_table_df[metric_table_df["is_apo"]]
        holo = metric_table_df[~metric_table_df["is_apo"]]
    else:
        apo = metric_table_df[metric_table_df[fold_source_col] == apo_label]
        holo = metric_table_df[metric_table_df[fold_source_col] == holo_label]

    rows: list[dict] = []
    for sid in metric_table_df[pair_on].unique():
        a = apo[apo[pair_on] == sid]
        h = holo[holo[pair_on] == sid]
        if len(a) == 0 or len(h) == 0:
            continue
        a_mean = a.mean(numeric_only=True)
        h_mean = h.mean(numeric_only=True)
        row = {pair_on: sid}
        for col in metric_columns:
            if col in a_mean.index and col in h_mean.index:
                row[f"delta__{col}"] = float(h_mean[col] - a_mean[col])
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "AggregationPolicy",
    "aggregate_metric_table",
    "default_policy",
    "paired_apo_holo_delta",
]
