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
    # --- N-way expert extensions (default no-op; see fuse_experts) ---------
    # Per-expert global multiplier (missing key -> 1.0). Lets you up/down-weight
    # a specific expert (e.g. {"hermes": 0.25} to avoid double-counting structure
    # that MPNN already sees).
    expert_weights: dict[str, float] = field(default_factory=dict)
    # Per-(expert, position-class) multiplier (missing -> 1.0). e.g.
    # {"hermes": {"primary_sphere": 0.0}} to mute an expert at the active site.
    expert_class_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-expert extra temperature on the calibrated log-odds (missing -> 1.0),
    # applied multiplicatively after entropy-match.
    expert_temperatures: dict[str, float] = field(default_factory=dict)
    # Fusion-math version tag, recorded in run provenance.
    version: str = "fusion-v1"


@dataclass
class FusionResult:
    bias: np.ndarray            # (L, 20) — additive bias for LigandMPNN
    log_odds_esmc: np.ndarray   # (L, 20) — calibrated ESM-C log-odds (N=2 legacy)
    log_odds_saprot: np.ndarray # (L, 20) — calibrated SaProt log-odds (N=2 legacy)
    weights_per_position: np.ndarray   # (L, 2) — final β, γ per position (N=2 legacy)
    config: FusionConfig
    # --- N-way generic fields (populated by fuse_experts) -----------------
    # log_odds[i] = calibrated (L,20) log-odds for expert i; weights_per_expert
    # is (L, N); expert_names lists the experts in order. For the default 2-expert
    # case the legacy fields above are ALSO populated so existing callers are
    # untouched.
    log_odds: Optional[list] = None
    weights_per_expert: Optional[np.ndarray] = None
    expert_names: Optional[list] = None


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


def entropy_match_temperatures(
    logprobs_list: Sequence[np.ndarray],
) -> tuple[float, ...]:
    """N-way generalization of :func:`entropy_match_temperature`.

    Returns one multiplier per expert that rescales its log-odds toward the
    geometric-mean median entropy. **Reduces exactly to the pairwise function at
    N=2** (it delegates), so the default 2-expert path is byte-identical.
    """
    n = len(logprobs_list)
    if n == 2:
        return entropy_match_temperature(logprobs_list[0], logprobs_list[1])
    if n == 0:
        return ()
    hs = [float(np.median(per_position_entropy(lp))) for lp in logprobs_list]
    if all(h > 0 for h in hs):
        h_target = float(np.exp(np.mean(np.log(hs))))  # geometric mean
    else:
        h_target = 1.0
    return tuple(h / h_target if h > 0 else 1.0 for h in hs)


def mean_pairwise_cosine(logprobs_list: Sequence[np.ndarray]) -> np.ndarray:
    """Per-position mean pairwise cosine agreement across N experts.

    **Reduces exactly to :func:`cosine_similarity_per_position` at N=2** (it
    delegates). A single expert is treated as perfect agreement (all ones).
    """
    n = len(logprobs_list)
    if n == 2:
        return cosine_similarity_per_position(logprobs_list[0], logprobs_list[1])
    if n == 0:
        return np.zeros(0)
    L = logprobs_list[0].shape[0]
    if n < 2:
        return np.ones(L)
    acc = np.zeros(L)
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            acc = acc + cosine_similarity_per_position(logprobs_list[i], logprobs_list[j])
            pairs += 1
    return acc / pairs


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


def fuse_experts(
    expert_logprobs: Sequence[np.ndarray],
    position_classes: Sequence[str],
    config: Optional[FusionConfig] = None,
    expert_names: Optional[Sequence[str]] = None,
) -> FusionResult:
    """Fuse N per-position expert log-prob arrays into a (L, 20) MPNN bias.

    Generalizes :func:`fuse_plm_logits` from the two hard-coded PLMs (ESM-C +
    SaProt) to an arbitrary list of per-position experts (add HERMES, a third PLM,
    future models) while keeping our calibration (log-odds vs background →
    entropy-match → structural class weights → shrink-at-disagreement).

    The multi-expert / registry / product-of-experts design is borrowed from
    Sebastian (sebols) and Joe Mi's ``fused_mpnn_poe`` (decode-time PoE); here we
    keep the *static, calibrated* fusion and just make it N-way.

    **Default 2-expert case (no per-expert knobs set) delegates verbatim to
    ``fuse_plm_logits``**, so it is byte-identical to the legacy path. The generic
    fields (``log_odds``, ``weights_per_expert``, ``expert_names``) are always
    populated; the legacy ``log_odds_esmc/saprot`` + ``weights_per_position`` are
    also populated whenever N==2 so existing callers are untouched.
    """
    cfg = config or FusionConfig()
    n = len(expert_logprobs)
    if n == 0:
        raise ValueError("fuse_experts requires >= 1 expert")
    names = (list(expert_names) if expert_names is not None
             else [f"expert{i}" for i in range(n)])
    if len(names) != n:
        raise ValueError(f"expert_names length {len(names)} != n experts {n}")
    L = expert_logprobs[0].shape[0]
    if len(position_classes) != L:
        raise ValueError(f"position_classes length {len(position_classes)} != L {L}")
    for lp in expert_logprobs:
        if lp.shape != expert_logprobs[0].shape:
            raise ValueError("all experts must share the same (L, 20) shape")

    knobs_set = bool(cfg.expert_weights or cfg.expert_class_weights
                     or cfg.expert_temperatures)

    # ---- Fast path: default 2-expert -> delegate to the UNTOUCHED legacy fn ----
    # This is the byte-identity guarantee: the default pipeline runs literally the
    # same code as before; only opt-in (>=3 experts or per-expert knobs) diverges.
    if n == 2 and not knobs_set:
        res = fuse_plm_logits(expert_logprobs[0], expert_logprobs[1],
                              position_classes, cfg)
        # Generic fields alias the legacy arrays here (read-only consumers only —
        # do not mutate weights_per_expert in place or you corrupt
        # weights_per_position).
        res.log_odds = [res.log_odds_esmc, res.log_odds_saprot]
        res.weights_per_expert = res.weights_per_position
        res.expert_names = names
        return res

    # ---- General N-way path (>=3 experts, or per-expert knobs set) ----
    los = [calibrate_log_odds(lp, cfg.aa_background) for lp in expert_logprobs]
    if cfg.entropy_match:
        ms = entropy_match_temperatures(expert_logprobs)
        los = [lo * m if m > 0 else lo for lo, m in zip(los, ms)]
    los = [lo * float(cfg.expert_temperatures.get(nm, 1.0))
           for lo, nm in zip(los, names)]

    from protein_chisel.tools.classify_positions import LEGACY_CLASS_REMAP

    def _lookup(cls: str) -> float:
        if cls in cfg.class_weights:
            return cfg.class_weights[cls]
        new_cls = LEGACY_CLASS_REMAP.get(cls)
        if new_cls is not None and new_cls in cfg.class_weights:
            return cfg.class_weights[new_cls]
        return 0.0

    base_weights = np.array(
        [_lookup(c) for c in position_classes], dtype=np.float64,
    ) * float(cfg.global_strength)

    if cfg.shrink_disagreement and n >= 2:
        agree = mean_pairwise_cosine(expert_logprobs)
        shrink = np.where(agree >= cfg.shrink_threshold, 1.0, np.maximum(agree, 0.0))
    else:
        shrink = np.ones(L)

    weights = []
    for nm in names:
        w = base_weights * shrink * float(cfg.expert_weights.get(nm, 1.0))
        ecw = cfg.expert_class_weights.get(nm)
        if ecw:
            w = w * np.array([ecw.get(c, 1.0) for c in position_classes],
                             dtype=np.float64)
        weights.append(w)

    bias = np.zeros((L, 20), dtype=np.float64)
    for w, lo in zip(weights, los):
        bias = bias + w[:, None] * lo
    weights_per_expert = np.stack(weights, axis=-1)  # (L, N)

    return FusionResult(
        bias=bias,
        log_odds_esmc=los[0],
        log_odds_saprot=(los[1] if n >= 2 else None),
        weights_per_position=(weights_per_expert if n == 2 else None),
        config=cfg,
        log_odds=los,
        weights_per_expert=weights_per_expert,
        expert_names=names,
    )


def decoupled_fitness_weights(
    expert_logprobs: Sequence[np.ndarray],
    position_classes: Sequence[str],
    config: Optional[FusionConfig] = None,
    expert_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Per-position fitness weights DECOUPLED from ``--plm_strength``.

    The seed-marginal fitness ranking (``fitness_score.fitness_from_seed_marginals``)
    weights each position by the fusion ``(β, γ)``. Because ``global_strength``
    (= ``--plm_strength``) scales ``β = γ`` *uniformly*, the fused-mean ranking is
    INVARIANT to any ``strength > 0`` (the scale cancels in the ``(βx+γy)/(β+γ)``
    ratio) but COLLAPSES to a constant — every design ties at 0 — at ``strength = 0``,
    silently zeroing the weight-2.0 fitness objective. Computing the weights at
    ``global_strength = 1.0`` keeps the ranking meaningful regardless of the sampling
    strength: byte-identical for ``strength > 0``, a real rank (instead of ties) at 0.
    The sampling *bias* still honors the user's ``--plm_strength`` separately.

    Returns the ``(L, 2)`` ``weights_per_position`` (the 2-expert legacy fitness
    shape); for >2 experts it returns the generic ``(L, N)`` ``weights_per_expert``.
    """
    import dataclasses
    cfg = config or FusionConfig()
    cfg_unit = dataclasses.replace(cfg, global_strength=1.0)
    res = fuse_experts(expert_logprobs, position_classes, cfg_unit, expert_names)
    return (res.weights_per_position if res.weights_per_position is not None
            else res.weights_per_expert)


__all__ = [
    "AA_ORDER",
    "AA_BG_VEC",
    "FusionConfig",
    "decoupled_fitness_weights",
    "FusionResult",
    "UNIPROT_AA_BG",
    "calibrate_log_odds",
    "cosine_similarity_per_position",
    "entropy_match_temperature",
    "entropy_match_temperatures",
    "fuse_experts",
    "fuse_plm_logits",
    "mean_pairwise_cosine",
    "per_position_entropy",
]
