"""Survivor-pool covariation / epistasis diagnostic (EXPERIMENTAL, off by default).

The per-cycle consensus uses single-site marginals only, which ignores
co-adaptation between positions. This module computes a regularized,
APC-corrected mutual-information (MIp) matrix over the surviving design sequences
to *surface* coupled position pairs — as a diagnostic and, optionally, a
"pair-aware consensus guard" (don't independently fix two anti-correlated
positions) or a reranker signal.

It is intentionally NOT injected into the per-position MPNN bias: that API is
strictly per-position and cannot carry pairwise terms (the principled epistasis
path is the decode-time PoE backend). Pairwise stats from a small survivor pool
are noisy, so MI is APC-corrected and pseudocount-regularized and treated as
advisory only.

Pure-Python/NumPy; no models or containers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 20 canonical AAs + gap/unknown bucket -> 21 symbols for frequency counting.
_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_IDX = {a: i for i, a in enumerate(_AA)}
_OTHER = 20  # any non-canonical char (X, gap, PTM 1-letter that slipped through)
_K = 21


@dataclass
class CovariationResult:
    mip: np.ndarray                 # (L, L) APC-corrected MI (MIp); diagonal 0
    top_pairs: list[tuple]          # [(i, j, mip_ij), ...] sorted desc, 0-indexed
    n_sequences: int
    length: int
    meta: dict = field(default_factory=dict)


def _encode(sequences):
    """Encode equal-length sequences to an (N, L) int array over 21 symbols."""
    seqs = [s for s in sequences if s]
    if not seqs:
        return np.zeros((0, 0), dtype=np.int64)
    L = len(seqs[0])
    seqs = [s for s in seqs if len(s) == L]   # drop ragged (defensive)
    arr = np.full((len(seqs), L), _OTHER, dtype=np.int64)
    for r, s in enumerate(seqs):
        for c, ch in enumerate(s):
            arr[r, c] = _AA_IDX.get(ch.upper(), _OTHER)
    return arr


def _site_freqs(col, pseudocount):
    f = np.bincount(col, minlength=_K).astype(np.float64) + pseudocount
    return f / f.sum()


def mutual_information_matrix(
    sequences,
    *,
    pseudocount: float = 0.5,
    apc: bool = True,
) -> np.ndarray:
    """Per-position-pair mutual information (nats), optionally APC-corrected (MIp).

    Pseudocount-regularized joint/marginal frequencies guard the small-N regime.
    APC (average product correction, Dunn et al. 2008) subtracts the background
    ``MI_i * MI_j / MI_mean`` that inflates MI from conservation/phylogeny.
    Returns an (L, L) symmetric matrix with zero diagonal.
    """
    arr = _encode(sequences)
    if arr.size == 0:
        return np.zeros((0, 0))
    N, L = arr.shape
    raw = np.zeros((L, L), dtype=np.float64)
    margs = [_site_freqs(arr[:, i], pseudocount) for i in range(L)]
    for i in range(L):
        fi = margs[i]
        for j in range(i + 1, L):
            # joint counts (K x K) with pseudocount
            joint = np.zeros((_K, _K), dtype=np.float64)
            np.add.at(joint, (arr[:, i], arr[:, j]), 1.0)
            joint += pseudocount
            joint /= joint.sum()
            fj = margs[j]
            outer = np.outer(fi, fj)
            with np.errstate(divide="ignore", invalid="ignore"):
                term = joint * np.log(joint / outer)
            mi = float(np.nansum(np.where(joint > 0, term, 0.0)))
            raw[i, j] = raw[j, i] = max(0.0, mi)
    if not apc or L < 2:
        return raw
    # APC: MIp(i,j) = MI(i,j) - (MIbar_i * MIbar_j) / MIbar
    col_mean = raw.sum(axis=1) / (L - 1)   # mean MI of each site (excl. diagonal)
    overall = raw.sum() / (L * (L - 1)) if L > 1 else 0.0
    if overall <= 0:
        return raw
    apc_term = np.outer(col_mean, col_mean) / overall
    mip = raw - apc_term
    np.fill_diagonal(mip, 0.0)
    return mip


def covariation_diagnostic(
    sequences,
    *,
    top_n: int = 25,
    pseudocount: float = 0.5,
    apc: bool = True,
) -> CovariationResult:
    """Compute the MIp matrix + the top-N most-coupled position pairs (0-indexed)."""
    seqs = [s for s in sequences if s]
    if seqs:
        L0 = len(seqs[0])
        seqs = [s for s in seqs if len(s) == L0]   # drop ragged (match MI)
    mip = mutual_information_matrix(seqs, pseudocount=pseudocount, apc=apc)
    L = mip.shape[0]
    pairs: list[tuple] = []
    if L >= 2:
        iu = np.triu_indices(L, k=1)
        vals = mip[iu]
        order = np.argsort(-vals)[:top_n]
        pairs = [(int(iu[0][k]), int(iu[1][k]), float(vals[k])) for k in order]
    return CovariationResult(
        mip=mip, top_pairs=pairs,
        n_sequences=len(seqs), length=L,
        meta={"pseudocount": pseudocount, "apc": apc},
    )


def anticorrelated_pairs(
    result: CovariationResult,
    *,
    threshold: float,
) -> list[tuple]:
    """Position pairs whose coupling exceeds ``threshold`` — candidates for the
    "pair-aware consensus guard" (avoid independently fixing both). Advisory."""
    return [(i, j, v) for (i, j, v) in result.top_pairs if v >= threshold]
