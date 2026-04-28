"""Pareto-front extraction with ε-dominance.

Codex's review insisted on:
- Cap objectives at 3-5 (otherwise everything is non-dominated).
- ε-dominance binning so float-precision differences don't create
  spurious incomparable points.
- Hard-constraints first, Pareto on what survives.

This module provides:

- ``apply_hard_constraints(df, constraints)``: drop rows that fail
  any hard constraint.
- ``epsilon_pareto_front(df, objectives, eps_per_obj, ...)``:
  extract the Pareto front under ε-binning.
- ``crowding_distance(df, objectives)``: NSGA-II–style spacing metric
  for selecting diverse representatives within the front.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class HardConstraint:
    """A single hard-constraint rule.

    Examples:
        HardConstraint(column="protparam__pi", min_value=4.0, max_value=8.0)
        HardConstraint(column="buns__n_buried_unsat", max_value=2)
        HardConstraint(column="protease__n_total", max_value=0)
    """
    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""

    def applies_to(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean mask: True where the row passes this constraint."""
        s = df[self.column]
        mask = pd.Series(True, index=df.index)
        if self.min_value is not None:
            mask &= s >= self.min_value
        if self.max_value is not None:
            mask &= s <= self.max_value
        # Treat NaN as failing
        mask &= s.notna()
        return mask


def apply_hard_constraints(
    df: pd.DataFrame, constraints: Iterable[HardConstraint]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply hard constraints; return (filtered_df, drops_per_constraint)."""
    drops: dict[str, int] = {}
    surviving = pd.Series(True, index=df.index)
    for c in constraints:
        if c.column not in df.columns:
            continue  # silently skip unknown columns
        mask = c.applies_to(df)
        drops[c.description or c.column] = int((~mask & surviving).sum())
        surviving &= mask
    return df[surviving].copy(), drops


# ---------------------------------------------------------------------------
# ε-Pareto
# ---------------------------------------------------------------------------


@dataclass
class Objective:
    """One objective for Pareto comparison.

    `direction` is "min" if smaller-is-better, "max" otherwise. ε is the
    bin size used for ε-dominance: two points are equivalent on this
    objective if their values fall in the same bin.
    """

    column: str
    direction: str = "min"  # "min" or "max"
    epsilon: float = 0.0


def epsilon_pareto_front(
    df: pd.DataFrame, objectives: list[Objective]
) -> pd.DataFrame:
    """Return the ε-Pareto-non-dominated subset of `df`.

    A point P is ε-dominated by Q iff:
    - For every objective, Q is no worse than P (binned),
    - And Q is strictly better than P on at least one objective (binned).
    """
    if not objectives:
        return df.copy()

    # Pre-compute binned values; minimization-direction so smaller is always better.
    binned = np.zeros((len(df), len(objectives)), dtype=np.float64)
    for j, obj in enumerate(objectives):
        v = df[obj.column].to_numpy(dtype=float)
        if obj.direction == "max":
            v = -v
        if obj.epsilon > 0:
            v = np.floor(v / obj.epsilon) * obj.epsilon
        binned[:, j] = v

    n = len(df)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for k in range(n):
            if k == i or not keep[k]:
                continue
            # Does k dominate i?
            if _dominates(binned[k], binned[i]):
                keep[i] = False
                break
    return df[keep].copy()


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff a ≤ b component-wise AND a < b on at least one component."""
    return bool(np.all(a <= b) and np.any(a < b))


# ---------------------------------------------------------------------------
# Crowding distance
# ---------------------------------------------------------------------------


def crowding_distance(df: pd.DataFrame, objectives: list[Objective]) -> np.ndarray:
    """NSGA-II crowding distance per row, length len(df).

    The two boundary points (min and max along each objective) get
    +infinity. Use to pick well-spread representatives within the Pareto
    front.
    """
    n = len(df)
    if n == 0:
        return np.array([])
    if n <= 2:
        return np.array([np.inf] * n)

    crowd = np.zeros(n, dtype=np.float64)
    indices = np.arange(n)

    for obj in objectives:
        vals = df[obj.column].to_numpy(dtype=float)
        order = np.argsort(vals)
        sorted_vals = vals[order]
        # Boundaries get infinite distance
        crowd[order[0]] = np.inf
        crowd[order[-1]] = np.inf
        rng = sorted_vals[-1] - sorted_vals[0]
        if rng <= 0:
            continue
        for k in range(1, n - 1):
            crowd[order[k]] += (sorted_vals[k + 1] - sorted_vals[k - 1]) / rng

    return crowd


__all__ = [
    "HardConstraint",
    "Objective",
    "apply_hard_constraints",
    "crowding_distance",
    "epsilon_pareto_front",
]
