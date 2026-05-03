"""Sequence-level fitness scoring + strict deduplication.

Two paths:

1. **Cheap** — gather per-position log p(aa_i | seed-context) from the
   cached PLM marginals computed once on the seed PDB. Approximates the
   true sequence log-likelihood by treating positions as conditionally
   independent given the seed sequence. Fast (numpy gather + mean), no
   GPU. Used as the per-cycle ranking signal.

2. **Rigorous** — re-run ESM-C + SaProt masked-LM on each candidate
   sequence (one apptainer call per batch). Slow but proper. Used
   optionally on the final cycle.

Dedup is a hard requirement: collapse identical sequences before any
expensive scoring, and again before iteration cycles consume previous
survivors. ``deduplicate_by_sequence`` is the single point that enforces
this and emits ``seq_hash`` + ``n_dupes`` columns.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from protein_chisel.sampling.plm_fusion import AA_ORDER


LOGGER = logging.getLogger("protein_chisel.fitness_score")


# Index map AA letter -> column in the (L, 20) log-prob matrices.
AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}


@dataclass
class FitnessResult:
    sequence: str
    seq_hash: str
    logp_esmc_mean: float       # mean per-residue log p(aa_i | seed-context)
    logp_saprot_mean: float
    logp_fused_mean: float      # weighted by per-position β/γ from FusionConfig
    method: str                 # "seed_marginal" or "rescored"


def seq_hash(sequence: str) -> str:
    """Deterministic short hash for a 1-letter sequence.

    blake2b with 12-byte digest = 24 hex chars. Cryptographic-grade
    collision resistance for the population sizes we ever produce
    (~10^4 sequences/run).

    NOTE: input is upper-cased before hashing — same letter case
    invariance as ``deduplicate_by_sequence``.
    """
    if not isinstance(sequence, str):
        raise TypeError(f"seq_hash needs str, got {type(sequence).__name__}")
    return hashlib.blake2b(
        sequence.upper().encode("ascii"), digest_size=12,
    ).hexdigest()


def fitness_from_seed_marginals(
    sequence: str,
    log_probs_esmc: np.ndarray,         # (L, 20)
    log_probs_saprot: np.ndarray,       # (L, 20)
    weights_per_position: np.ndarray,   # (L, 2) — β, γ from FusionResult
) -> FitnessResult:
    """Per-position log p(seq[i] | seed-context) gathered from the cached
    masked-LM marginals on the SEED PDB, then averaged.

    This is an approximation to the true sequence log-likelihood — it
    assumes positions are conditionally independent given the seed
    sequence, which is wrong for compensatory mutations but is a fine
    *ranking* signal at zero compute cost beyond a numpy gather.
    """
    seq = sequence.upper()
    L = len(seq)
    if log_probs_esmc.shape != (L, 20):
        raise ValueError(
            f"esmc shape {log_probs_esmc.shape} != ({L}, 20) — "
            "candidate length doesn't match seed marginals"
        )
    if log_probs_saprot.shape != (L, 20):
        raise ValueError(
            f"saprot shape {log_probs_saprot.shape} != ({L}, 20)"
        )
    if weights_per_position.shape != (L, 2):
        raise ValueError(
            f"weights shape {weights_per_position.shape} != ({L}, 2)"
        )

    aa_idx = np.array(
        [AA_TO_IDX.get(c, -1) for c in seq], dtype=np.int64,
    )
    valid = aa_idx >= 0
    if not valid.all():
        # Non-canonical AA (X, *, etc.) — gather where valid; ignore else.
        bad = [c for c, ok in zip(seq, valid) if not ok]
        LOGGER.debug("fitness_from_seed_marginals: %d non-canonical AAs (%s) ignored",
                     (~valid).sum(), "".join(sorted(set(bad))))

    rows = np.arange(L)
    safe_idx = np.where(valid, aa_idx, 0)  # avoid OOB; we'll mask below
    lp_e = log_probs_esmc[rows, safe_idx]
    lp_s = log_probs_saprot[rows, safe_idx]

    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("sequence has zero canonical AAs")

    mean_e = float(np.where(valid, lp_e, 0.0).sum() / n_valid)
    mean_s = float(np.where(valid, lp_s, 0.0).sum() / n_valid)

    # Fused mean: weighted by per-position (β, γ) from the fusion. We
    # fall back to 1.0 weight where both are zero (active-site rows) so
    # the metric is still defined — those positions are usually fixed
    # anyway, so this is just defensive.
    beta = weights_per_position[:, 0]
    gamma = weights_per_position[:, 1]
    w_sum = beta + gamma
    safe_w = np.where(w_sum > 0, w_sum, 1.0)
    fused = (beta * lp_e + gamma * lp_s) / safe_w
    mean_f = float(np.where(valid, fused, 0.0).sum() / n_valid)

    return FitnessResult(
        sequence=seq,
        seq_hash=seq_hash(seq),
        logp_esmc_mean=mean_e,
        logp_saprot_mean=mean_s,
        logp_fused_mean=mean_f,
        method="seed_marginal",
    )


def deduplicate_by_sequence(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
) -> pd.DataFrame:
    """Drop rows with identical (case-insensitive) sequences.

    Adds ``seq_hash`` and ``n_dupes`` columns. Stable: keeps the first
    occurrence (preserving original sampler metadata).

    Empty input returns empty output without crashing.
    """
    if df is None or len(df) == 0:
        out = df.copy() if df is not None else pd.DataFrame()
        if "seq_hash" not in out.columns:
            out["seq_hash"] = pd.Series(dtype=object)
        if "n_dupes" not in out.columns:
            out["n_dupes"] = pd.Series(dtype="Int64")
        return out

    if sequence_col not in df.columns:
        raise KeyError(f"column {sequence_col!r} not in DataFrame")

    work = df.copy()
    # Case-fold defensively. fused_mpnn writes uppercase, but caller may
    # have introduced lowercase elsewhere.
    work[sequence_col] = work[sequence_col].astype(str).str.upper()
    work["seq_hash"] = work[sequence_col].map(seq_hash)
    counts = work["seq_hash"].value_counts()
    work["n_dupes"] = work["seq_hash"].map(counts).astype("Int64")
    deduped = work.drop_duplicates(subset="seq_hash", keep="first").reset_index(drop=True)
    n_in, n_out = len(work), len(deduped)
    if n_in != n_out:
        LOGGER.info("deduplicate_by_sequence: %d -> %d unique (%d dupes collapsed)",
                     n_in, n_out, n_in - n_out)
    return deduped


def score_dataframe_fitness(
    df: pd.DataFrame,
    log_probs_esmc: np.ndarray,
    log_probs_saprot: np.ndarray,
    weights_per_position: np.ndarray,
    sequence_col: str = "sequence",
    fitness_cache: Optional[dict[str, FitnessResult]] = None,
) -> pd.DataFrame:
    """Add fitness columns to df. Mutates a copy and returns it.

    Cache is an in-memory dict keyed by ``seq_hash``. Pass the same dict
    across cycles so survivors that re-emerge re-use scores.

    Adds columns: fitness__logp_esmc_mean, fitness__logp_saprot_mean,
    fitness__logp_fused_mean, fitness__method.
    """
    if fitness_cache is None:
        fitness_cache = {}
    out = df.copy()
    if "seq_hash" not in out.columns:
        out["seq_hash"] = out[sequence_col].astype(str).str.upper().map(seq_hash)

    rows = []
    for _, row in out.iterrows():
        h = row["seq_hash"]
        if h in fitness_cache:
            res = fitness_cache[h]
        else:
            res = fitness_from_seed_marginals(
                row[sequence_col],
                log_probs_esmc,
                log_probs_saprot,
                weights_per_position,
            )
            fitness_cache[h] = res
        rows.append({
            "fitness__logp_esmc_mean": res.logp_esmc_mean,
            "fitness__logp_saprot_mean": res.logp_saprot_mean,
            "fitness__logp_fused_mean": res.logp_fused_mean,
            "fitness__method": res.method,
        })
    fit_df = pd.DataFrame(rows, index=out.index)
    return pd.concat([out, fit_df], axis=1)


__all__ = [
    "AA_TO_IDX",
    "FitnessResult",
    "deduplicate_by_sequence",
    "fitness_from_seed_marginals",
    "score_dataframe_fitness",
    "seq_hash",
]
