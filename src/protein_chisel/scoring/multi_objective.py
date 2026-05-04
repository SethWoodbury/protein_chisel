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
