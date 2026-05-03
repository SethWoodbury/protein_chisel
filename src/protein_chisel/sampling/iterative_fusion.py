"""Cross-cycle bias refinement: turn last cycle's survivors into the
next cycle's prior.

The cycle-0 bias comes from the calibrated PLM fusion (ESM-C + SaProt
log-odds, weighted by position class). For cycle k+1 we *augment*
that bias with consensus information learned from cycle k's survivors:
where ≥``consensus_threshold`` of survivors agree on an AA at a non-
fixed position, add ``+consensus_strength`` nats to that AA in the
bias.

This implements an evolving prior without ever overriding the
catalytic constraint or replacing the PLM signal at non-consensus
positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from protein_chisel.sampling.plm_fusion import AA_ORDER


LOGGER = logging.getLogger("protein_chisel.iterative_fusion")


_AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


@dataclass
class IterationBiasConfig:
    consensus_threshold: float = 0.85         # AA freq required to "agree"
    consensus_strength: float = 2.0           # nats added at agreed AA
    only_at_classes: tuple[str, ...] = (
        "buried", "surface", "first_shell", "pocket",
    )
    # Cap on how many positions can be augmented; protects diversity.
    max_augmented_fraction: float = 0.30


@dataclass
class IterationBiasTelemetry:
    n_survivors: int
    n_positions_total: int
    n_positions_eligible: int    # in only_at_classes
    n_positions_augmented: int
    augmented_resnos: list[int]  # 1-indexed protein resnos
    capped: bool                 # True if max_augmented_fraction was hit


def consensus_aa_frequencies(
    sequences: Sequence[str],
    L: int,
) -> np.ndarray:
    """Return (L, 20) per-position empirical AA frequency over sequences.

    Sequences must all have length L. Non-canonical letters at any
    position are skipped (so freq.sum(axis=1) <= 1, equal to the
    fraction of canonical AAs at that position).
    """
    if len(sequences) == 0:
        return np.zeros((L, 20), dtype=np.float64)
    counts = np.zeros((L, 20), dtype=np.float64)
    n_per_pos = np.zeros(L, dtype=np.float64)
    for s in sequences:
        if len(s) != L:
            raise ValueError(
                f"consensus: sequence length {len(s)} != L {L}"
            )
        for i, c in enumerate(s.upper()):
            j = _AA_TO_IDX.get(c)
            if j is None:
                continue
            counts[i, j] += 1.0
            n_per_pos[i] += 1.0
    safe_n = np.where(n_per_pos > 0, n_per_pos, 1.0)
    return counts / safe_n[:, None]


def build_iteration_bias(
    base_bias: np.ndarray,                          # (L, 20)
    survivor_sequences: Sequence[str],
    position_classes: Sequence[str],
    fixed_resnos_zero_indexed: Optional[Sequence[int]] = None,
    config: Optional[IterationBiasConfig] = None,
) -> tuple[np.ndarray, IterationBiasTelemetry]:
    """Augment ``base_bias`` with consensus from ``survivor_sequences``.

    Returns (bias_kp1, telemetry).

    - ``base_bias``: cycle-0 (L, 20) PLM-fusion bias.
    - ``survivor_sequences``: previous cycle's filter survivors.
    - ``position_classes``: length-L class labels (e.g. "active_site",
      "first_shell", "pocket", "buried", "surface", "ligand").
    - ``fixed_resnos_zero_indexed``: positions to NEVER augment, even
      if they're in ``only_at_classes``. Pass the catalytic indices.
    """
    cfg = config or IterationBiasConfig()
    if base_bias.ndim != 2 or base_bias.shape[1] != 20:
        raise ValueError(f"base_bias must be (L, 20), got {base_bias.shape}")
    L = base_bias.shape[0]
    if len(position_classes) != L:
        raise ValueError(
            f"position_classes length {len(position_classes)} != L {L}"
        )

    out_bias = base_bias.copy()
    telem = IterationBiasTelemetry(
        n_survivors=len(survivor_sequences),
        n_positions_total=L,
        n_positions_eligible=0,
        n_positions_augmented=0,
        augmented_resnos=[],
        capped=False,
    )

    if len(survivor_sequences) == 0:
        LOGGER.info("build_iteration_bias: no survivors -> using base_bias")
        return out_bias, telem

    fixed_set = set(int(i) for i in (fixed_resnos_zero_indexed or ()))
    eligible_classes = set(cfg.only_at_classes)
    eligible_mask = np.array([
        (cls in eligible_classes) and (i not in fixed_set)
        for i, cls in enumerate(position_classes)
    ], dtype=bool)
    telem.n_positions_eligible = int(eligible_mask.sum())

    freqs = consensus_aa_frequencies(list(survivor_sequences), L)
    top_freq = freqs.max(axis=1)
    top_aa = freqs.argmax(axis=1)

    # Candidate positions: eligible AND top_freq >= threshold
    candidates = np.where(
        eligible_mask & (top_freq >= cfg.consensus_threshold)
    )[0]

    # Cap: never augment more than max_augmented_fraction of L
    cap = int(np.floor(cfg.max_augmented_fraction * L))
    if len(candidates) > cap:
        # Pick the candidates with highest top_freq (most agreement first)
        order = np.argsort(-top_freq[candidates])
        candidates = candidates[order[:cap]]
        telem.capped = True
        LOGGER.warning(
            "build_iteration_bias: %d candidates exceeds cap %d (max_frac=%.2f); "
            "keeping top-agreement positions only",
            len(np.where(eligible_mask & (top_freq >= cfg.consensus_threshold))[0]),
            cap, cfg.max_augmented_fraction,
        )

    for i in candidates:
        out_bias[i, top_aa[i]] += cfg.consensus_strength
        telem.augmented_resnos.append(int(i + 1))  # 1-indexed for humans

    telem.n_positions_augmented = len(candidates)
    LOGGER.info(
        "build_iteration_bias: augmented %d/%d eligible positions "
        "(threshold=%.2f, strength=%.2f, capped=%s)",
        telem.n_positions_augmented, telem.n_positions_eligible,
        cfg.consensus_threshold, cfg.consensus_strength, telem.capped,
    )
    return out_bias, telem


__all__ = [
    "IterationBiasConfig",
    "IterationBiasTelemetry",
    "build_iteration_bias",
    "consensus_aa_frequencies",
]
