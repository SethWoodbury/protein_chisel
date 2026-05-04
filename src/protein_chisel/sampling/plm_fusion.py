"""Calibrated fusion of ESM-C and SaProt log-probabilities.

The fusion turns per-position log-probs into a *bias* matrix suitable
for LigandMPNN's ``--bias_AA_per_residue``. Following codex's review:

1. Convert raw log-probs to **log-odds** by subtracting the AA background
   marginal (so "rare AA everywhere" doesn't dominate "wrong AA here").
2. **Entropy-match** the two models so neither dominates after summation
   — rescale each model's logits by τ such that the median per-position
   entropy matches across models.
3. Apply **position-class–dependent weights** (β for ESM-C, γ for SaProt)
   so e.g. surface positions get full PLM input, pocket-lining gets a
   fraction, active-site gets zero.
4. **Shrink at disagreement**: where the two models disagree (low cosine
   similarity of their per-position distributions), scale down both
   contributions toward zero.

Output: an additive bias matrix ``(L, 20)`` per protein. LigandMPNN
``--bias_AA_per_residue`` is added to its own per-position logits at
sample time, so the units are nats (log-space) and 0 means "no bias."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np


# UniProt amino-acid background frequencies (Swiss-Prot 2024). Used as
# the default `aa_background` for log-odds calibration.
UNIPROT_AA_BG: dict[str, float] = {
    "A": 0.0825, "R": 0.0553, "N": 0.0406, "D": 0.0545, "C": 0.0137,
    "Q": 0.0393, "E": 0.0675, "G": 0.0707, "H": 0.0227, "I": 0.0596,
    "L": 0.0966, "K": 0.0584, "M": 0.0242, "F": 0.0386, "P": 0.0470,
    "S": 0.0656, "T": 0.0534, "W": 0.0108, "Y": 0.0292, "V": 0.0686,
}

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_BG_VEC = np.array([UNIPROT_AA_BG[a] for a in AA_ORDER])
LOG_AA_BG = np.log(AA_BG_VEC)


@dataclass
class FusionConfig:
    aa_background: np.ndarray = field(default_factory=lambda: AA_BG_VEC.copy())
    entropy_match: bool = True
    # Per-position-class weights (β = γ = base_weights[class]). Keys map to
    # the `class` column from classify_positions.
    #
    # Defaults track the directional 6-class taxonomy from
    # tools/classify_positions.py (post 2026-05-04 rewrite). Legacy
    # 5-class keys (active_site / first_shell / pocket / buried /
    # surface) are auto-remapped via LEGACY_CLASS_REMAP at lookup
    # time, so old PositionTables continue to work but emit a
    # DeprecationWarning. Tunable per-class:
    #   primary_sphere   = catalytic + direct ligand contact
    #   secondary_sphere = coordinates a primary_sphere residue
    #                      (preorganization; literature 2nd shell)
    #   nearby_surface   = close to pocket but pointing out / exposed
    #   distal_buried    = far + buried (folding/stability — PLM useful)
    #   distal_surface   = far + surface (solubility — PLM most useful)
    class_weights: dict[str, float] = field(default_factory=lambda: {
        "primary_sphere":   0.05,
        "secondary_sphere": 0.20,
        "nearby_surface":   0.30,
        "distal_buried":    0.40,
        "distal_surface":   0.55,
        "ligand":           0.0,
        # Legacy keys kept so old configs still load — DEPRECATED:
        "active_site":      0.05,
        "first_shell":      0.15,
        "pocket":           0.20,
        "buried":           0.35,
        "surface":          0.55,
    })
    # Global multiplier on top of class_weights. Lets the driver expose
    # a single --plm_strength knob (default 1.0). Set < 1.0 to soften
    # PLM influence everywhere, > 1.0 to emphasize.
    global_strength: float = 1.0
    shrink_disagreement: bool = True
    # Cosine similarity threshold below which to shrink. 1.0 = perfect
    # agreement; 0 = orthogonal. Below `shrink_threshold`, weight is
    # scaled by the actual cosine value.
    shrink_threshold: float = 0.7


@dataclass
class FusionResult:
    bias: np.ndarray            # (L, 20) — additive bias for LigandMPNN
    log_odds_esmc: np.ndarray   # (L, 20) — calibrated ESM-C log-odds
    log_odds_saprot: np.ndarray # (L, 20) — calibrated SaProt log-odds
    weights_per_position: np.ndarray   # (L, 2) — final β, γ per position
    config: FusionConfig


def calibrate_log_odds(log_probs: np.ndarray, aa_bg: np.ndarray) -> np.ndarray:
    """Subtract log AA-background from log-probabilities.

    Args:
        log_probs: (L, 20) log-probabilities (rows sum to 1 in prob space).
        aa_bg: (20,) background AA frequencies (sum to 1).

    Returns:
        (L, 20) log-odds: ``log p(aa | ctx) - log p_bg(aa)``. Positive
        means the model prefers this AA above its baseline rate.
    """
    if log_probs.shape[1] != 20 or aa_bg.shape != (20,):
        raise ValueError(f"shape mismatch: log_probs={log_probs.shape}, aa_bg={aa_bg.shape}")
    return log_probs - np.log(aa_bg)[None, :]


def per_position_entropy(log_probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per row (in nats), shape (L,)."""
    p = np.exp(log_probs)
    # H = -Σ p log p, with stable convention 0 log 0 = 0
    return -(p * log_probs).sum(axis=-1)


def entropy_match_temperature(
    log_probs_a: np.ndarray, log_probs_b: np.ndarray
) -> tuple[float, float]:
    """Return multipliers (m_a, m_b) for each model's log-odds that
    equalize their median entropies.

    Standard temperature scaling: with logits x, applying temperature T
    gives ``softmax(x/T)``. T > 1 softens (raises entropy), T < 1 sharpens
    (lowers entropy). To pull a high-entropy model toward a target lower
    entropy, we want T < 1 i.e. multiply the logits by ``m = 1/T > 1``.

    With ``m_a = h_a / h_target``: if h_a > h_target (model A too soft),
    m_a > 1 and multiplying its logits by m_a sharpens it. Symmetric for
    model B. Apply with ``log_odds * m_a``.
    """
    h_a = float(np.median(per_position_entropy(log_probs_a)))
    h_b = float(np.median(per_position_entropy(log_probs_b)))
    # Geometric-mean target (between the two models).
    h_target = np.sqrt(h_a * h_b) if h_a > 0 and h_b > 0 else 1.0
    m_a = h_a / h_target if h_a > 0 else 1.0
    m_b = h_b / h_target if h_b > 0 else 1.0
    return float(m_a), float(m_b)


def cosine_similarity_per_position(
    log_probs_a: np.ndarray, log_probs_b: np.ndarray
) -> np.ndarray:
    """Cosine similarity between the two distributions, per position.

    Operates in probability space (so disagreement on rare AAs doesn't
    explode in log space). Returns (L,) values in [-1, 1] but for valid
    distributions in [0, 1].
    """
    p_a = np.exp(log_probs_a)
    p_b = np.exp(log_probs_b)
    num = (p_a * p_b).sum(axis=-1)
    denom = (
        np.linalg.norm(p_a, axis=-1) * np.linalg.norm(p_b, axis=-1) + 1e-12
    )
    return num / denom


def fuse_plm_logits(
    log_probs_esmc: np.ndarray,
    log_probs_saprot: np.ndarray,
    position_classes: Sequence[str],
    config: Optional[FusionConfig] = None,
) -> FusionResult:
    """Fuse ESM-C + SaProt per-position log-probs into a (L, 20) bias matrix.

    Args:
        log_probs_esmc:  (L, 20) ESM-C log-probabilities.
        log_probs_saprot: (L, 20) SaProt log-probabilities.
        position_classes: length-L list of class strings (active_site /
            first_shell / pocket / buried / surface). Drives β, γ
            position-class–dependent weights.
        config: FusionConfig.

    Returns:
        FusionResult with `bias` ready to feed to LigandMPNN.
    """
    if log_probs_esmc.shape != log_probs_saprot.shape:
        raise ValueError(
            f"shape mismatch: esmc={log_probs_esmc.shape}, saprot={log_probs_saprot.shape}"
        )
    cfg = config or FusionConfig()
    L = log_probs_esmc.shape[0]
    if len(position_classes) != L:
        raise ValueError(
            f"position_classes length {len(position_classes)} != L {L}"
        )

    # 1. Log-odds calibration
    lo_esmc = calibrate_log_odds(log_probs_esmc, cfg.aa_background)
    lo_saprot = calibrate_log_odds(log_probs_saprot, cfg.aa_background)

    # 2. Entropy-match (rescale to equalize median entropy)
    if cfg.entropy_match:
        m_e, m_s = entropy_match_temperature(log_probs_esmc, log_probs_saprot)
        # m > 1 means that model is softer than the target → multiply logits
        # by m to sharpen. m < 1 means already too sharp → multiply by m
        # (< 1) to soften.
        if m_e > 0:
            lo_esmc = lo_esmc * m_e
        if m_s > 0:
            lo_saprot = lo_saprot * m_s

    # 3. Per-position class weights. Auto-remap legacy class names to
    # the new directional taxonomy so old PositionTables still work.
    from protein_chisel.tools.classify_positions import (
        LEGACY_CLASS_REMAP, NEW_CLASSES,
    )
    def _lookup(cls: str) -> float:
        # Direct hit (new vocabulary).
        if cls in cfg.class_weights:
            return cfg.class_weights[cls]
        # Legacy → new remap.
        new_cls = LEGACY_CLASS_REMAP.get(cls)
        if new_cls is not None and new_cls in cfg.class_weights:
            return cfg.class_weights[new_cls]
        return 0.0
    base_weights = np.array(
        [_lookup(cls) for cls in position_classes], dtype=np.float64,
    ) * float(cfg.global_strength)  # (L,)
    # Same weight for both models initially; can be specialized later.
    beta = base_weights.copy()
    gamma = base_weights.copy()

    # 4. Shrinkage at disagreement
    if cfg.shrink_disagreement:
        cos = cosine_similarity_per_position(log_probs_esmc, log_probs_saprot)
        # Where cos >= shrink_threshold, no shrinkage. Below, scale by cos itself.
        shrink_factor = np.where(cos >= cfg.shrink_threshold, 1.0, np.maximum(cos, 0.0))
        beta = beta * shrink_factor
        gamma = gamma * shrink_factor

    # Final bias: weighted sum of calibrated log-odds
    bias = beta[:, None] * lo_esmc + gamma[:, None] * lo_saprot
    weights = np.stack([beta, gamma], axis=-1)
    return FusionResult(
        bias=bias,
        log_odds_esmc=lo_esmc,
        log_odds_saprot=lo_saprot,
        weights_per_position=weights,
        config=cfg,
    )


__all__ = [
    "AA_ORDER",
    "AA_BG_VEC",
    "FusionConfig",
    "FusionResult",
    "UNIPROT_AA_BG",
    "calibrate_log_odds",
    "cosine_similarity_per_position",
    "entropy_match_temperature",
    "fuse_plm_logits",
    "per_position_entropy",
]
