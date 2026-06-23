"""Shared solubility-band predicate.

Single source of truth for "is this design inside the GRAVY + net-charge solubility
band?" — used by the driver's opt-in ``--ship_solubility_veto`` (WS-A) AND by the
PLM-refresh representative selection (WS-F). Centralising it here keeps the band
semantics (GRAVY inclusive, net-charge exclusive, fail-closed on missing data)
identical across both consumers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def within_solubility_band(
    df: pd.DataFrame,
    *,
    gravy_min: float,
    gravy_max: float,
    net_charge_min: float,
    net_charge_max: float,
) -> pd.Series:
    """Boolean mask: each row is inside the GRAVY + net-charge solubility band.

    Mirrors ``stage_seq_filter`` exactly: charge uses the full-HH column with
    EXCLUSIVE bounds (``net_charge_min < c < net_charge_max``); GRAVY uses INCLUSIVE
    bounds (``gravy_min <= g <= gravy_max``). Missing / non-numeric values fail
    closed (``False``) so a design can never qualify on absent data.
    """
    n = len(df)

    def _num(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series([np.nan] * n, index=df.index)

    g = _num("gravy")
    c = _num("net_charge_full_HH")
    ok = (
        (g >= gravy_min) & (g <= gravy_max)
        & (c > net_charge_min) & (c < net_charge_max)
    )
    return ok.fillna(False).astype(bool)


__all__ = ["within_solubility_band"]
