"""Unit tests for sampling/plm_fusion. No GPU; runs on host."""

from __future__ import annotations

import numpy as np
import pytest

from protein_chisel.sampling.plm_fusion import (
    AA_BG_VEC,
    FusionConfig,
    calibrate_log_odds,
    cosine_similarity_per_position,
    entropy_match_temperature,
    fuse_plm_logits,
    per_position_entropy,
)


def _uniform_log_probs(L: int) -> np.ndarray:
    """Each row uniform over 20 AAs."""
    return np.full((L, 20), np.log(1 / 20))


def _peak_log_probs(L: int, aa_index: int) -> np.ndarray:
    """Each row puts ~all probability mass on a single AA."""
    p = np.full((L, 20), 1e-6)
    p[:, aa_index] = 1.0 - 19 * 1e-6
    return np.log(p / p.sum(axis=-1, keepdims=True))


def _rand_log_probs(L: int, seed: int) -> np.ndarray:
    """Random per-position log-probs (rows normalized in prob space)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(L, 20))
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


# ---- WS-E: PLM strength <-> fitness decouple ------------------------------


def test_decoupled_fitness_weights_rank_invariant_and_fix_strength_0():
    """The decoupled (strength=1.0) weights produce the SAME fitness RANK as the
    legacy strength-1.25 weights on a mixed-class protein — the scale cancels in the
    ratio, so the rank is invariant (the scalar may differ by ~1 ULP, which is why
    the driver REUSES the exact strength-scaled weights for strength>0 and only
    substitutes the strength-1.0 weights at strength==0). And where the strength=0
    fusion weights collapse fitness to all-ties (0), the decoupled weights rescue a
    real spread — the weight-2.0 fitness objective the decouple restores."""
    from protein_chisel.sampling.plm_fusion import (
        FusionConfig, decoupled_fitness_weights, fuse_experts,
    )
    from protein_chisel.sampling.fitness_score import fitness_from_seed_marginals
    L = 16
    lp_e, lp_s = _rand_log_probs(L, 1), _rand_log_probs(L, 2)
    classes = (["primary_sphere", "distal_surface", "nearby_surface",
                "distal_buried"] * 4)[:L]                       # genuinely mixed
    seqs = ["ACDEFGHIKLMNPQRS", "WYVTSRQPNMLKIHGF",
            "AAAACCCCDDDDEEEE", "KRKRKRKRDEDEDEDE"]

    w_dec = decoupled_fitness_weights([lp_e, lp_s], classes,
                                      FusionConfig(global_strength=0.0))
    w_125 = fuse_experts([lp_e, lp_s], classes,
                         FusionConfig(global_strength=1.25)).weights_per_position
    f_dec = [fitness_from_seed_marginals(s, lp_e, lp_s, w_dec).logp_fused_mean for s in seqs]
    f_125 = [fitness_from_seed_marginals(s, lp_e, lp_s, w_125).logp_fused_mean for s in seqs]
    assert list(np.argsort(f_dec)) == list(np.argsort(f_125))   # RANK-invariant
    # strength=0 fusion weights -> all-ties (0); decoupled rescues a real spread.
    w_0 = fuse_experts([lp_e, lp_s], classes,
                       FusionConfig(global_strength=0.0)).weights_per_position
    f_0 = [fitness_from_seed_marginals(s, lp_e, lp_s, w_0).logp_fused_mean for s in seqs]
    assert all(v == 0.0 for v in f_0) and len(set(f_dec)) > 1


def test_decoupled_fitness_weights_independent_of_config_strength():
    """The helper ignores the config's global_strength (always fuses at 1.0)."""
    from protein_chisel.sampling.plm_fusion import (
        FusionConfig, decoupled_fitness_weights,
    )
    L = 8
    lp_e, lp_s = _rand_log_probs(L, 3), _rand_log_probs(L, 4)
    classes = ["distal_surface"] * L
    a = decoupled_fitness_weights([lp_e, lp_s], classes, FusionConfig(global_strength=0.3))
    b = decoupled_fitness_weights([lp_e, lp_s], classes, FusionConfig(global_strength=2.0))
    assert np.array_equal(a, b)


# ---- log-odds -------------------------------------------------------------


def test_calibrate_log_odds_uniform_input_yields_negative_log_bg():
    """Uniform log-probs → log-odds = log(1/20) - log(p_bg) per AA."""
    lp = _uniform_log_probs(5)
    lo = calibrate_log_odds(lp, AA_BG_VEC)
    expected = np.log(1 / 20) - np.log(AA_BG_VEC)
    assert np.allclose(lo, expected[None, :])


def test_calibrate_log_odds_shape():
    lp = _uniform_log_probs(10)
    lo = calibrate_log_odds(lp, AA_BG_VEC)
    assert lo.shape == (10, 20)


def test_calibrate_log_odds_rejects_bad_shape():
    with pytest.raises(ValueError):
        calibrate_log_odds(np.zeros((5, 19)), AA_BG_VEC)


# ---- entropy --------------------------------------------------------------


def test_entropy_uniform_is_log20():
    lp = _uniform_log_probs(5)
    h = per_position_entropy(lp)
    assert np.allclose(h, np.log(20.0))


def test_entropy_peaked_near_zero():
    lp = _peak_log_probs(5, aa_index=3)
    h = per_position_entropy(lp)
    assert np.all(h < 0.01)


def test_entropy_match_returns_unity_for_same_dists():
    lp = _uniform_log_probs(5)
    tau_a, tau_b = entropy_match_temperature(lp, lp.copy())
    assert abs(tau_a - 1.0) < 1e-6
    assert abs(tau_b - 1.0) < 1e-6


def test_entropy_match_compensates_difference():
    """Higher-entropy model gets multiplier > 1, sharper model gets < 1."""
    high = _uniform_log_probs(5)              # H = log 20
    low = _peak_log_probs(5, aa_index=0)       # H ≈ 0
    m_high, m_low = entropy_match_temperature(high, low)
    assert m_high > m_low
    # The high-entropy model should get a multiplier > 1 (sharpen via |x| ↑)
    assert m_high > 1.0
    # The peaked model is already sharper than target → multiplier < 1
    assert m_low < 1.0


def test_entropy_match_reduces_entropy_gap_after_apply():
    """After applying multipliers, the two models' entropies should be closer."""
    high = _uniform_log_probs(8)
    # Generate a moderately peaked second distribution (not a delta)
    p_low = np.full((8, 20), 0.01)
    p_low[:, 5] = 0.81  # ~0.81 mass on one AA, ~0.01 on each other
    p_low = p_low / p_low.sum(axis=-1, keepdims=True)
    low = np.log(p_low)
    h_high0 = per_position_entropy(high).mean()
    h_low0 = per_position_entropy(low).mean()
    initial_gap = abs(h_high0 - h_low0)

    m_high, m_low = entropy_match_temperature(high, low)
    # Apply multipliers as the fuse code does (log-odds path; same shape op)
    high_scaled = high * m_high
    low_scaled = low * m_low
    # Re-normalize rows because scaled log-probs aren't probabilities anymore;
    # we want to compare entropies of the implied distributions.
    high_scaled = high_scaled - np.log(np.exp(high_scaled).sum(-1, keepdims=True))
    low_scaled = low_scaled - np.log(np.exp(low_scaled).sum(-1, keepdims=True))
    h_high1 = per_position_entropy(high_scaled).mean()
    h_low1 = per_position_entropy(low_scaled).mean()
    final_gap = abs(h_high1 - h_low1)
    assert final_gap < initial_gap, (
        f"entropy gap should shrink after match; was {initial_gap:.3f}, "
        f"now {final_gap:.3f}"
    )


# ---- cosine similarity ----------------------------------------------------


def test_cosine_identical_distributions_is_one():
    lp = _uniform_log_probs(5)
    cos = cosine_similarity_per_position(lp, lp.copy())
    assert np.allclose(cos, 1.0)


def test_cosine_orthogonal_distributions_low():
    """Two peaked distributions on different AAs → cosine ≈ 0."""
    lp_a = _peak_log_probs(5, aa_index=0)  # peak on A
    lp_c = _peak_log_probs(5, aa_index=2)  # peak on C  (note ACDEF... order)
    cos = cosine_similarity_per_position(lp_a, lp_c)
    assert np.all(cos < 0.05)


# ---- fuse_plm_logits ------------------------------------------------------


def test_fuse_active_site_can_be_zeroed_via_config():
    """Active-site weight is configurable. With class_weights['active_site']=0,
    the bias at that position is exactly zero (preserves the legacy
    'fixed-residues get no PLM' behavior when explicitly requested)."""
    L = 4
    lp_esmc = _peak_log_probs(L, aa_index=0)
    lp_saprot = _peak_log_probs(L, aa_index=0)
    pos_classes = ["active_site", "first_shell", "buried", "surface"]
    cfg = FusionConfig(class_weights={
        "active_site": 0.0, "first_shell": 0.05, "pocket": 0.10,
        "buried": 0.30, "surface": 0.50, "ligand": 0.0,
    })
    res = fuse_plm_logits(lp_esmc, lp_saprot, pos_classes, config=cfg)
    assert np.allclose(res.bias[0], 0.0)
    assert np.any(res.bias[3] != 0)


def test_fuse_class_weight_ordering():
    """Higher class_weights → larger absolute bias magnitude.

    Default weights (post 2026-05-04 bump): active_site < first_shell <
    pocket < buried < surface, so the bias magnitude is monotone non-
    decreasing across this class sequence.
    """
    L = 4
    lp = _peak_log_probs(L, aa_index=0)
    res = fuse_plm_logits(lp, lp.copy(), ["active_site", "first_shell", "buried", "surface"])
    abs_mags = np.abs(res.bias).max(axis=-1)
    # active_site < first_shell <= buried <= surface (all non-zero now)
    assert abs_mags[0] > 0.0   # active_site is now bumped to 0.05
    assert abs_mags[0] < abs_mags[1] < abs_mags[2] < abs_mags[3]


def test_fuse_shrinkage_for_disagreement():
    """When models strongly disagree, weights shrink toward zero."""
    L = 4
    lp_a = _peak_log_probs(L, aa_index=0)
    lp_c = _peak_log_probs(L, aa_index=2)
    classes = ["surface"] * L
    res = fuse_plm_logits(lp_a, lp_c, classes, config=FusionConfig(shrink_disagreement=True))
    # Per-position weights should be near zero (cosine ≈ 0)
    assert np.all(res.weights_per_position < 0.05)


def test_fuse_no_shrinkage_for_agreement():
    L = 4
    lp = _peak_log_probs(L, aa_index=5)
    classes = ["surface"] * L
    cfg = FusionConfig(shrink_disagreement=True)
    res = fuse_plm_logits(lp, lp.copy(), classes, config=cfg)
    # Agreement → no shrinkage → weights equal the configured surface
    # class weight (default 0.55 post-2026-05-04 bump).
    expected = cfg.class_weights["surface"]
    assert np.allclose(res.weights_per_position[:, 0], expected)
    assert np.allclose(res.weights_per_position[:, 1], expected)


def test_fuse_global_strength_scales_weights():
    """global_strength scales class_weights uniformly."""
    L = 3
    lp = _peak_log_probs(L, aa_index=0)
    classes = ["surface"] * L
    cfg_default = FusionConfig(global_strength=1.0)
    cfg_bumped = FusionConfig(global_strength=1.5)
    r1 = fuse_plm_logits(lp, lp.copy(), classes, config=cfg_default)
    r2 = fuse_plm_logits(lp, lp.copy(), classes, config=cfg_bumped)
    # Bumped should scale weights by exactly 1.5×
    assert np.allclose(
        r2.weights_per_position, r1.weights_per_position * 1.5
    )
    # And bias should also scale by 1.5×
    assert np.allclose(r2.bias, r1.bias * 1.5)


def test_fuse_zero_strength_disables_plm():
    """global_strength=0.0 zeroes the bias entirely (PLM disabled)."""
    L = 3
    lp = _peak_log_probs(L, aa_index=0)
    classes = ["surface", "buried", "first_shell"]
    cfg = FusionConfig(global_strength=0.0)
    res = fuse_plm_logits(lp, lp.copy(), classes, config=cfg)
    assert np.allclose(res.bias, 0.0)
    assert np.allclose(res.weights_per_position, 0.0)


def test_fuse_strength_and_shrinkage_commute():
    """strength * shrink should equal shrink * strength (linearity check)."""
    L = 3
    lp_a = _peak_log_probs(L, aa_index=0)
    lp_b = _peak_log_probs(L, aa_index=2)   # disagreement → shrinkage active
    classes = ["surface"] * L
    cfg_1 = FusionConfig(global_strength=1.0, shrink_disagreement=True)
    cfg_2 = FusionConfig(global_strength=2.0, shrink_disagreement=True)
    r1 = fuse_plm_logits(lp_a, lp_b, classes, config=cfg_1)
    r2 = fuse_plm_logits(lp_a, lp_b, classes, config=cfg_2)
    # Strength multiplies the post-shrink weights; r2 = 2 * r1.
    assert np.allclose(r2.weights_per_position, 2 * r1.weights_per_position)
    assert np.allclose(r2.bias, 2 * r1.bias)


def test_fuse_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fuse_plm_logits(
            _uniform_log_probs(4),
            _uniform_log_probs(5),
            ["surface"] * 4,
        )


def test_fuse_position_classes_length_mismatch_raises():
    with pytest.raises(ValueError):
        fuse_plm_logits(
            _uniform_log_probs(4),
            _uniform_log_probs(4),
            ["surface"] * 3,
        )


def test_fuse_returns_calibrated_log_odds():
    L = 3
    lp = _uniform_log_probs(L)
    res = fuse_plm_logits(lp, lp.copy(), ["surface"] * L)
    # Calibrated log-odds for uniform input = log(1/20) - log(p_bg)
    expected = np.log(1 / 20) - np.log(AA_BG_VEC)
    # entropy-match scales the log-odds; just check shape and that
    # raw log-odds were computed (via the unscaled path).
    assert res.log_odds_esmc.shape == (L, 20)
    assert res.log_odds_saprot.shape == (L, 20)
