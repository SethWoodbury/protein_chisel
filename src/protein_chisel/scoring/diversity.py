"""Sequence-identity diversity over MUTABLE / POCKET positions only.

Codex's review explicitly called out: full-length Hamming is dominated by
surface noise that doesn't matter for function. This module computes
identity over a restricted subset of positions.

Two main entry points:

- ``hamming_distance(seq_a, seq_b, mask)`` for two sequences and a
  position mask.
- ``select_diverse(df, sequence_col, mask, k, min_distance)`` selects k
  representatives from a population such that pairwise Hamming distance
  on the masked positions is at least `min_distance`.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def hamming_distance(seq_a: str, seq_b: str, mask: Optional[Sequence[bool]] = None) -> int:
    """Hamming distance between two equal-length sequences.

    Args:
        mask: optional boolean mask of length len(seq_a). Only positions
            with mask[i] == True contribute. If None, all positions count.
    """
    if len(seq_a) != len(seq_b):
        raise ValueError(f"length mismatch: {len(seq_a)} vs {len(seq_b)}")
    if mask is None:
        return sum(a != b for a, b in zip(seq_a, seq_b))
    if len(mask) != len(seq_a):
        raise ValueError(f"mask length {len(mask)} != sequence length {len(seq_a)}")
    return sum(1 for a, b, m in zip(seq_a, seq_b, mask) if m and a != b)


def hamming_matrix(sequences: Sequence[str], mask: Optional[Sequence[bool]] = None) -> np.ndarray:
    """Pairwise Hamming distance matrix, shape (n, n)."""
    n = len(sequences)
    out = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(sequences[i], sequences[j], mask=mask)
            out[i, j] = out[j, i] = d
    return out


def select_diverse(
    df: pd.DataFrame,
    sequence_col: str,
    mask: Optional[Sequence[bool]] = None,
    k: int = 50,
    min_distance: int = 1,
    score_col: Optional[str] = None,
    score_direction: str = "max",
) -> pd.DataFrame:
    """Greedy diversity-aware selection.

    Algorithm:
        1. If ``score_col`` is given, sort the DataFrame by score
           (descending if "max", ascending if "min").
        2. Walk the list and accept each candidate iff its minimum
           Hamming distance (on masked positions) to all already-chosen
           is >= min_distance.
        3. Stop at k.

    Returns the selected subset preserving the score order.
    """
    if score_col is not None and score_col in df.columns:
        df_sorted = df.sort_values(
            score_col, ascending=(score_direction == "min")
        ).reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    selected_idxs: list[int] = []
    selected_seqs: list[str] = []
    for i, seq in enumerate(df_sorted[sequence_col].tolist()):
        if not selected_seqs:
            selected_idxs.append(i)
            selected_seqs.append(seq)
            continue
        min_d = min(
            hamming_distance(seq, s, mask=mask) for s in selected_seqs
        )
        if min_d >= min_distance:
            selected_idxs.append(i)
            selected_seqs.append(seq)
        if len(selected_idxs) >= k:
            break
    return df_sorted.iloc[selected_idxs].reset_index(drop=True)


def mask_from_position_table(
    pt_df: pd.DataFrame,
    mutable_classes: Iterable[str] = ("buried", "surface", "first_shell", "pocket"),
) -> list[bool]:
    """Build a per-position bool mask from a PositionTable DataFrame.

    Returns True for protein positions whose ``class`` is in
    ``mutable_classes`` (skips ligand and active_site by default — those
    are immutable in our pipelines, so they shouldn't influence
    diversity selection).
    """
    mask: list[bool] = []
    for _, row in pt_df.sort_values("resno").iterrows():
        if not bool(row["is_protein"]):
            continue
        mask.append(row["class"] in mutable_classes)
    return mask


__all__ = [
    "hamming_distance",
    "hamming_matrix",
    "mask_from_position_table",
    "select_diverse",
]
