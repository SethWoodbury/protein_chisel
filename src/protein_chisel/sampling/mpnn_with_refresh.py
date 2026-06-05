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
from typing import Callable, Optional

LOGGER = logging.getLogger("protein_chisel.sampling.mpnn_with_refresh")


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
