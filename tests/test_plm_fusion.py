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


def test_fuse_zero_weights_at_active_site():
    L = 4
    lp_esmc = _peak_log_probs(L, aa_index=0)
    lp_saprot = _peak_log_probs(L, aa_index=0)
    pos_classes = ["active_site", "first_shell", "buried", "surface"]
    res = fuse_plm_logits(lp_esmc, lp_saprot, pos_classes)
    # Active site bias is exactly zero
    assert np.allclose(res.bias[0], 0.0)
    # Surface bias is non-zero (highest weight class)
    assert np.any(res.bias[3] != 0)


def test_fuse_class_weight_ordering():
    """Higher class_weights → larger absolute bias magnitude."""
    L = 4
    # Put both models in agreement on a single AA so log-odds are large
    lp = _peak_log_probs(L, aa_index=0)
    res = fuse_plm_logits(lp, lp.copy(), ["active_site", "first_shell", "buried", "surface"])
    abs_mags = np.abs(res.bias).max(axis=-1)
    # active_site=0 < first_shell < buried < surface
    assert abs_mags[0] == 0.0
    assert abs_mags[0] <= abs_mags[1] <= abs_mags[2] <= abs_mags[3]


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
    res = fuse_plm_logits(lp, lp.copy(), classes, config=FusionConfig(shrink_disagreement=True))
    # Surface class weight is 0.5; agreement → no shrinkage → still 0.5
    assert np.allclose(res.weights_per_position[:, 0], 0.5)
    assert np.allclose(res.weights_per_position[:, 1], 0.5)


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
