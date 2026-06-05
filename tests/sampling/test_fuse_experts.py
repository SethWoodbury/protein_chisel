"""Tests for the N-way expert fusion (fuse_experts) — the Phase-1 foundation.

The headline contract: the default 2-expert path is BYTE-IDENTICAL to the legacy
``fuse_plm_logits`` (it delegates to it), and the N-way helpers reduce exactly to
the pairwise ones at N=2.
"""
from __future__ import annotations

import numpy as np
import pytest

from protein_chisel.sampling.plm_fusion import (
    AA_ORDER,
    FusionConfig,
    cosine_similarity_per_position,
    entropy_match_temperature,
    entropy_match_temperatures,
    fuse_experts,
    fuse_plm_logits,
    mean_pairwise_cosine,
)

_CLASSES = ["primary_sphere", "secondary_sphere", "nearby_surface",
            "distal_buried", "distal_surface", "ligand"]


def _rand_logprobs(L, seed):
    """Random valid (L,20) log-probabilities (rows normalize in prob space)."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((L, 20))
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return logp


def _classes(L, seed=0):
    rng = np.random.default_rng(seed)
    return [_CLASSES[i] for i in rng.integers(0, len(_CLASSES), size=L)]


# ----------------------------------------------------------------------
# Headline: default 2-expert == legacy, byte-identical
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [
    FusionConfig(),
    FusionConfig(global_strength=1.25),   # the runtime default
    FusionConfig(global_strength=0.0),
    FusionConfig(entropy_match=False),
    FusionConfig(shrink_disagreement=False),
    FusionConfig(shrink_threshold=0.5),
])
def test_default_two_expert_byte_identical(cfg):
    L = 37
    e = _rand_logprobs(L, 1)
    s = _rand_logprobs(L, 2)
    pc = _classes(L, 3)
    legacy = fuse_plm_logits(e, s, pc, cfg)
    new = fuse_experts([e, s], pc, config=cfg, expert_names=["esmc", "saprot"])
    assert np.array_equal(new.bias, legacy.bias)                       # exact
    assert np.array_equal(new.log_odds_esmc, legacy.log_odds_esmc)
    assert np.array_equal(new.log_odds_saprot, legacy.log_odds_saprot)
    assert np.array_equal(new.weights_per_position, legacy.weights_per_position)
    # generic fields populated too
    assert new.expert_names == ["esmc", "saprot"]
    assert np.array_equal(new.weights_per_expert, legacy.weights_per_position)


def test_legacy_class_names_still_work_identically():
    L = 20
    e, s = _rand_logprobs(L, 4), _rand_logprobs(L, 5)
    pc = (["active_site", "first_shell", "pocket", "buried", "surface"] * 4)[:L]
    legacy = fuse_plm_logits(e, s, pc)
    new = fuse_experts([e, s], pc, expert_names=["esmc", "saprot"])
    assert np.array_equal(new.bias, legacy.bias)


# ----------------------------------------------------------------------
# Helper reductions at N=2 (exact)
# ----------------------------------------------------------------------
def test_entropy_match_reduces_at_n2():
    a, b = _rand_logprobs(15, 6), _rand_logprobs(15, 7)
    assert entropy_match_temperatures([a, b]) == entropy_match_temperature(a, b)


def test_mean_pairwise_cosine_reduces_at_n2():
    a, b = _rand_logprobs(15, 8), _rand_logprobs(15, 9)
    assert np.array_equal(mean_pairwise_cosine([a, b]),
                          cosine_similarity_per_position(a, b))


def test_mean_pairwise_cosine_single_expert_is_ones():
    a = _rand_logprobs(11, 10)
    assert np.array_equal(mean_pairwise_cosine([a]), np.ones(11))


# ----------------------------------------------------------------------
# N-way behavior
# ----------------------------------------------------------------------
def test_three_identical_experts_triples_single():
    L = 12
    e = _rand_logprobs(L, 11)
    pc = _classes(L, 12)
    # identical experts → perfect agreement → no shrink; bias == 3× one-expert term
    three = fuse_experts([e, e, e], pc, expert_names=["a", "b", "c"])
    one_weight = three.weights_per_expert[:, 0]
    expected = sum(one_weight[:, None] * three.log_odds[i] for i in range(3))
    assert np.allclose(three.bias, expected)
    assert three.weights_per_expert.shape == (L, 3)


def test_zero_weight_expert_contributes_nothing_to_sum():
    # A zero-WEIGHT expert drops out of the weighted sum, but (by design) still
    # participates in entropy-match + shrink calibration. To fully exclude an
    # expert, omit it from the list; expert_weights is for DOWN-weighting.
    L = 18
    e, s, h = _rand_logprobs(L, 13), _rand_logprobs(L, 14), _rand_logprobs(L, 15)
    pc = _classes(L, 16)
    cfg = FusionConfig(expert_weights={"hermes": 0.0})
    out3 = fuse_experts([e, s, h], pc, config=cfg,
                        expert_names=["esmc", "saprot", "hermes"])
    assert np.allclose(out3.weights_per_expert[:, 2], 0.0)       # hermes muted
    expected = (out3.weights_per_expert[:, 0][:, None] * out3.log_odds[0]
                + out3.weights_per_expert[:, 1][:, None] * out3.log_odds[1])
    assert np.allclose(out3.bias, expected)                      # only e+s contribute


def test_per_expert_class_weight_zeroes_active_site():
    L = 6
    e, s, h = _rand_logprobs(L, 1), _rand_logprobs(L, 2), _rand_logprobs(L, 3)
    pc = list(_CLASSES)  # one of each class, incl. primary_sphere at index 0
    cfg = FusionConfig(expert_class_weights={"hermes": {"primary_sphere": 0.0}})
    out = fuse_experts([e, s, h], pc, config=cfg,
                       expert_names=["esmc", "saprot", "hermes"])
    # hermes weight is the 3rd column; must be 0 at the primary_sphere row
    hermes_w = out.weights_per_expert[:, 2]
    assert hermes_w[pc.index("primary_sphere")] == 0.0


def test_empty_expert_list_raises():
    with pytest.raises(ValueError):
        fuse_experts([], [])


def test_noop_knobs_at_n2_still_equals_legacy():
    # Forcing the slow path with no-op (1.0) knobs must still match legacy bias.
    L = 22
    e, s = _rand_logprobs(L, 30), _rand_logprobs(L, 31)
    pc = _classes(L, 32)
    cfg = FusionConfig(expert_weights={"esmc": 1.0, "saprot": 1.0},
                       expert_temperatures={"esmc": 1.0, "saprot": 1.0})
    legacy = fuse_plm_logits(e, s, pc)
    forced_slow = fuse_experts([e, s], pc, config=cfg,
                               expert_names=["esmc", "saprot"])
    assert np.allclose(forced_slow.bias, legacy.bias)


def test_per_expert_temperature_sharpens():
    L = 10
    e, s = _rand_logprobs(L, 33), _rand_logprobs(L, 34)
    pc = ["distal_surface"] * L  # high PLM weight so the effect is visible
    base = fuse_experts([e, s], pc, expert_names=["esmc", "saprot"],
                        config=FusionConfig(expert_weights={"esmc": 1.0}))  # force slow, t=1
    hot = fuse_experts([e, s], pc, expert_names=["esmc", "saprot"],
                       config=FusionConfig(expert_temperatures={"esmc": 2.0}))
    # esmc's log-odds contribution is doubled in `hot` -> bias differs.
    assert not np.allclose(base.bias, hot.bias)


def test_n3_disagreeing_experts_apply_shrink():
    L = 14
    a, b, c = _rand_logprobs(L, 40), _rand_logprobs(L, 41), _rand_logprobs(L, 42)
    pc = ["distal_surface"] * L
    agree = mean_pairwise_cosine([a, b, c])
    # random independent experts disagree -> some positions below the 0.7 default
    assert (agree < 0.7).any()
    out = fuse_experts([a, b, c], pc, expert_names=["a", "b", "c"])
    # where agreement is below threshold, the per-expert weight is shrunk < base
    base_w = (np.array([0.55] * L) * 1.0)  # distal_surface default * global_strength
    shrunk = out.weights_per_expert[:, 0] < base_w - 1e-9
    assert shrunk.any()


def test_shape_and_length_validation():
    L = 8
    e, s = _rand_logprobs(L, 50), _rand_logprobs(L, 51)
    with pytest.raises(ValueError):       # position_classes wrong length
        fuse_experts([e, s], _classes(L - 1), expert_names=["esmc", "saprot"])
    with pytest.raises(ValueError):       # mismatched expert shapes
        fuse_experts([e, _rand_logprobs(L + 1, 52)], _classes(L),
                     expert_names=["esmc", "saprot"])
    with pytest.raises(ValueError):       # expert_names length mismatch
        fuse_experts([e, s], _classes(L), expert_names=["only_one"])
