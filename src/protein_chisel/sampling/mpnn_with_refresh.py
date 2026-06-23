"""PLM-bias refresh loop (optional, COSTLY; off by default).

A static seed-derived PLM bias drifts as MPNN moves the sequence away from the
seed (documented in architecture.md / docs/future_plans.md). This refresh loop
re-grounds the bias mid-run: sample -> pick a representative survivor -> recompute
the expert marginals on that *drifted* sequence -> re-fuse -> resample, for K
rounds.

This module is a **pure orchestrator**: the expensive, container-bound steps
(MPNN sampling; recomputing ESM-C/SaProt marginals via esmc.sif) are injected as
callables, so the loop logic is fully unit-testable without models or containers.

``rounds == 0`` calls ``sample_fn`` exactly once and returns — identical to the
non-refresh path, so the default pipeline is unchanged.

Cost: each round adds ~one full masked-LM precompute (minutes on GPU, much more on
CPU). Enable only deliberately.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("protein_chisel.sampling.mpnn_with_refresh")


def choose_inband_representative(
    survivors: Optional[pd.DataFrame],
    *,
    gravy_min: float,
    gravy_max: float,
    net_charge_min: float,
    net_charge_max: float,
    fitness_col: str = "fitness__logp_fused_mean",
    seq_col: str = "sequence",
    require_passed_seq_filter: bool = True,
) -> Optional[str]:
    """Median-fitness sequence among IN-BAND survivors (the rep to re-ground on).

    'In-band' = the shared solubility band (``scoring.solubility.within_solubility_band``
    — GRAVY inclusive, net-charge exclusive, fail-closed). A representative must also
    have passed the seq filter (when that column is present and
    ``require_passed_seq_filter``). Among the qualified, pick the row NEAREST the
    median fitness (a *typical* soluble drift, not an outlier), deterministically
    tie-broken by higher fitness then original order; if no usable fitness, the first
    in-band row. Returns the sequence string the ESM-C recompute consumes, or ``None``
    when nothing qualifies (the caller then skips the refresh round).
    """
    if survivors is None or len(survivors) == 0 or seq_col not in survivors.columns:
        return None
    from protein_chisel.scoring.solubility import within_solubility_band
    inband = within_solubility_band(
        survivors, gravy_min=gravy_min, gravy_max=gravy_max,
        net_charge_min=net_charge_min, net_charge_max=net_charge_max)
    if require_passed_seq_filter and "passed_seq_filter" in survivors.columns:
        # Robust truthiness: a string "False"/"0" or NaN must read as FAILED (a naive
        # .astype(bool) reads any non-empty string — incl. "False" — as True).
        inband = inband & survivors["passed_seq_filter"].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes", "t"))
    cand = survivors[inband]
    # Drop null / empty sequences (a pd.NA would stringify to "<NA>").
    cand = cand[cand[seq_col].notna() & (cand[seq_col].astype(str).str.len() > 0)]
    if len(cand) == 0:
        return None
    if fitness_col in cand.columns:
        cfit = pd.to_numeric(cand[fitness_col], errors="coerce")
        finite = cand[np.isfinite(cfit)]
        if len(finite) > 0:
            ffit = pd.to_numeric(finite[fitness_col], errors="coerce")
            med = float(ffit.median())
            order = finite.assign(_d=(ffit - med).abs(), _f=ffit).sort_values(
                ["_d", "_f"], ascending=[True, False], kind="stable")
            return str(order.iloc[0][seq_col])
    return str(cand.iloc[0][seq_col])               # no usable fitness -> first in-band


def refuse_esmc_only(
    *,
    new_esmc_lp: np.ndarray,
    seed_saprot_lp: np.ndarray,
    position_classes: Sequence[str],
    fusion_cfg,
) -> np.ndarray:
    """Re-fuse the ESM-C-only refresh: fresh ESM-C marginals + the UNCHANGED seed
    SaProt marginals → ``(L, 20)`` bias.

    SaProt is structure-aware (it consumes the 3Di token); a masked-LM marginal on
    the *drifted sequence* would erase that structural signal, so SaProt must stay at
    its seed marginals (refresh ESM-C ONLY). The same ``fusion_cfg`` flows
    ``--plm_strength`` / ``--plm_class_strength`` through, and ``fuse_experts``
    recomputes entropy-match + disagreement-shrink on the new pair.
    """
    from protein_chisel.sampling.plm_fusion import fuse_experts
    return fuse_experts(
        [new_esmc_lp, seed_saprot_lp], list(position_classes),
        config=fusion_cfg, expert_names=["esmc", "saprot"]).bias


def run_with_refresh(
    initial_bias,
    *,
    rounds: int,
    sample_fn: Callable[[object], object],
    choose_representative_fn: Callable[[object], Optional[object]],
    recompute_bias_fn: Callable[[object, object], Optional[object]],
):
    """Run sampling with ``rounds`` bias-refresh iterations.

    Args:
        initial_bias: the starting (L, 20) bias (e.g. the cycle's fused bias).
        rounds: number of refresh iterations. 0 = sample once (no refresh).
        sample_fn(bias) -> sample_result: run MPNN with the given bias.
        choose_representative_fn(sample_result) -> drifted_sequence | None:
            pick the survivor to re-ground on (e.g. median-by-naturalness). Return
            None to stop refreshing (e.g. nothing survived).
        recompute_bias_fn(drifted_sequence, prev_bias) -> new_bias | None:
            recompute expert marginals on the drifted sequence and re-fuse into a
            new bias. Return None to stop (e.g. recompute failed / unavailable).

    Returns:
        The final ``sample_result`` (after the last successful round).
    """
    if not isinstance(rounds, int) or isinstance(rounds, bool):
        raise ValueError(f"rounds must be an int, got {type(rounds).__name__}")
    if rounds < 0:
        raise ValueError(f"rounds must be >= 0, got {rounds}")
    if rounds > 0:
        LOGGER.warning(
            "PLM refresh ENABLED (rounds=%d): each round adds ~one full masked-LM "
            "precompute (minutes on GPU, more on CPU). This is COSTLY.", rounds,
        )
    result = sample_fn(initial_bias)
    if rounds == 0:
        return result
    bias = initial_bias
    for r in range(1, rounds + 1):
        rep = choose_representative_fn(result)
        if rep is None:
            LOGGER.info("refresh round %d: no representative survivor; stopping.", r)
            break
        new_bias = recompute_bias_fn(rep, bias)
        if new_bias is None:
            LOGGER.info("refresh round %d: bias recompute unavailable; stopping.", r)
            break
        bias = new_bias
        result = sample_fn(bias)
        LOGGER.info("refresh round %d/%d: resampled on refreshed bias.", r, rounds)
    return result
