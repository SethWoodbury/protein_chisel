"""Tests for sampling/struct_aware_bias.py.

V1 (trust modulation) is fully exercised here. V2 (Dunbrack
marginalisation) ships as a skeleton so its tests assert the
NotImplementedError contract for now.

Edge cases covered (most flagged by codex/agent reviewers):

- residue with all-NaN chi log-probs (Gly/Ala) -> NO_CHI_TRUST passthrough
- catalytic residues are passthrough regardless of trust score
- length mismatch between PLM logits and packer output -> ValueError
- one-residue protein (degenerate -- still works)
- trust_floor / trust_ceiling clipping
- trust_temperature controls sharpness predictably
- aggregate="mean" vs "sum" semantics
- all-NaN row in nansum: must NOT silently become 0
- bias of all -inf at a position (hard mask)
- non-monotonic trust input (random per-residue)
- concrete numerics: known chi NLL -> known trust
- multiplying by trust=0 zeroes bias correctly (full flatten)
- multiplying by trust=1 preserves bias exactly
"""

from __future__ import annotations

import numpy as np
import pytest

from protein_chisel.sampling.plm_fusion import AA_BG_VEC, FusionConfig
from protein_chisel.sampling.struct_aware_bias import (
    NO_CHI_AAS,
    NO_CHI_TRUST,
    StructAwareBiasConfig,
    StructAwareBiasResult,
    aa_logprior_from_chi,
    apply_chi_trust_to_bias,
    fuse_plm_struct_logits,
    per_position_chi_nll,
    trust_from_chi_logp,
)


# ----------------------------------------------------------------------
# per_position_chi_nll: aggregation semantics
# ----------------------------------------------------------------------


def test_per_position_chi_nll_mean_ignores_nan():
    """All-NaN row in a Gly/Ala position produces NaN (documented
    contract). The implementation uses manual sum/count to avoid
    numpy's 'Mean of empty slice' warning -- see
    test_per_position_chi_nll_no_runtime_warning below.
    """
    arr = np.array([
        [-1.0, -2.0, np.nan, np.nan],   # Val: 2 chis
        [-3.0, np.nan, np.nan, np.nan], # Ser: 1 chi
        [np.nan, np.nan, np.nan, np.nan], # Gly: no chis
    ])
    out = per_position_chi_nll(arr, aggregate="mean")
    np.testing.assert_allclose(out[:2], [-1.5, -3.0])
    assert np.isnan(out[2])


def test_per_position_chi_nll_sum_preserves_nan_row():
    """A row of all NaN must not become 0 under sum."""
    arr = np.array([
        [-1.0, -2.0, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ])
    out = per_position_chi_nll(arr, aggregate="sum")
    assert out[0] == -3.0
    assert np.isnan(out[1])


def test_per_position_chi_nll_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(L, K\)"):
        per_position_chi_nll(np.ones((10,)))   # 1-D
    with pytest.raises(ValueError, match=r"\(L, K\)"):
        per_position_chi_nll(np.ones((3, 2, 2)))  # 3-D


def test_per_position_chi_nll_unknown_aggregate():
    with pytest.raises(ValueError, match="unsupported aggregate"):
        per_position_chi_nll(np.ones((1, 4)), aggregate="median")


# ----------------------------------------------------------------------
# trust_from_chi_logp: monotonicity, clipping, NaN handling
# ----------------------------------------------------------------------


def test_trust_increases_with_chi_logp():
    """More plausible rotamer (higher logp) -> higher trust."""
    cfg = StructAwareBiasConfig(trust_temperature=5.0, trust_floor=0.0, trust_ceiling=1.0)
    arr = np.array([-10.0, -5.0, -2.0, 0.0])
    trust = trust_from_chi_logp(arr, config=cfg)
    assert np.all(np.diff(trust) > 0)


def test_trust_at_reference_is_half():
    """At chi_logp = reference_logp, trust = sigmoid(0) = 0.5."""
    cfg = StructAwareBiasConfig(
        reference_logp=-3.0, trust_temperature=2.0,
        trust_floor=0.0, trust_ceiling=1.0,
    )
    out = trust_from_chi_logp(np.array([-3.0]), config=cfg)
    np.testing.assert_allclose(out, [0.5])


def test_trust_max_confidence_approaches_one():
    """Max-confidence packer (chi_logp = 0) should give trust > 0.5
    after centering, NOT exactly 0.5 (codex's critique of raw sigmoid).
    """
    cfg = StructAwareBiasConfig(
        reference_logp=-3.0, trust_temperature=2.0,
        trust_floor=0.0, trust_ceiling=1.0,
    )
    out = trust_from_chi_logp(np.array([0.0]), config=cfg)
    # sigmoid((0 - (-3)) / 2) = sigmoid(1.5) ~ 0.818
    assert out[0] > 0.7
    assert out[0] < 0.9


def test_trust_uniform_packer_does_not_erase_lm_signal():
    """At chi_logp = -log(20) (a packer with no opinion across 20 chi bins),
    trust should be ~0.5 -- not zero. The LM bias must NOT be erased
    by an uncertain packer (codex's critique of raw sigmoid: NLL=0
    used to map to 0.5 of MAX, conflating 'no info' with 'half-as-good').
    """
    cfg = StructAwareBiasConfig(
        reference_logp=-3.0,    # ~ -log(20)
        trust_temperature=2.0,
        trust_floor=0.0, trust_ceiling=1.0,
    )
    out = trust_from_chi_logp(np.array([-3.0]), config=cfg)
    np.testing.assert_allclose(out, [0.5])

    # And with the default (production) config, trust_floor=0.5 means
    # even a fully-uniform packer keeps trust>=0.5.
    out_default = trust_from_chi_logp(np.array([-3.0]))
    assert out_default[0] >= 0.5


def test_trust_temperature_controls_sharpness():
    """Smaller T = sharper transition around the reference."""
    arr = np.array([-10.0])
    t_sharp = trust_from_chi_logp(
        arr, config=StructAwareBiasConfig(
            reference_logp=-3.0, trust_temperature=1.0,
            trust_floor=0.0, trust_ceiling=1.0,
        ),
    )
    t_soft = trust_from_chi_logp(
        arr, config=StructAwareBiasConfig(
            reference_logp=-3.0, trust_temperature=10.0,
            trust_floor=0.0, trust_ceiling=1.0,
        ),
    )
    # Both are below 0.5 (chi_logp << reference); sharper T pushes lower.
    assert t_sharp[0] < t_soft[0]
    assert t_sharp[0] < 0.01      # very sharp transition
    assert t_soft[0] > 0.2        # very soft


def test_trust_floor_and_ceiling_clip():
    cfg = StructAwareBiasConfig(trust_temperature=5.0, trust_floor=0.2, trust_ceiling=0.9)
    out = trust_from_chi_logp(np.array([-100.0, 0.0, +100.0]), config=cfg)
    assert out[0] == 0.2     # floor
    assert out[2] == 0.9     # ceiling


def test_trust_nan_input_returns_no_chi_trust():
    arr = np.array([-1.0, np.nan, -5.0])
    out = trust_from_chi_logp(arr)
    assert out[1] == NO_CHI_TRUST   # = 1.0 by default
    assert not np.isnan(out[0])
    assert not np.isnan(out[2])


def test_trust_invalid_temperature():
    with pytest.raises(ValueError, match="trust_temperature"):
        trust_from_chi_logp(
            np.array([0.0]),
            config=StructAwareBiasConfig(trust_temperature=0.0),
        )
    with pytest.raises(ValueError, match="trust_temperature"):
        trust_from_chi_logp(
            np.array([0.0]),
            config=StructAwareBiasConfig(trust_temperature=-1.0),
        )


def test_trust_invalid_floor_ceiling_ordering():
    """trust_floor must be <= trust_ceiling and both in [0, 1]."""
    with pytest.raises(ValueError, match="trust_floor"):
        trust_from_chi_logp(
            np.array([0.0]),
            config=StructAwareBiasConfig(trust_floor=0.6, trust_ceiling=0.4),
        )
    with pytest.raises(ValueError, match="trust_floor"):
        trust_from_chi_logp(
            np.array([0.0]),
            config=StructAwareBiasConfig(trust_floor=-0.1),
        )
    with pytest.raises(ValueError, match="trust_floor"):
        trust_from_chi_logp(
            np.array([0.0]),
            config=StructAwareBiasConfig(trust_ceiling=1.5),
        )


def test_trust_extreme_logp_does_not_overflow():
    """Numerically stable sigmoid: very large negative or positive
    chi_logp should produce finite, in-range trust values."""
    arr = np.array([-1e6, -1e3, -1.0, 0.0, 1e3, 1e6])
    out = trust_from_chi_logp(
        arr, config=StructAwareBiasConfig(
            reference_logp=0.0, trust_temperature=1.0,
            trust_floor=0.0, trust_ceiling=1.0,
        ),
    )
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_trust_monotonic_with_chi_logp():
    """Random per-residue chi_logp -> trust must be monotonically
    non-decreasing in chi_logp. This is the contract: more confident
    packer -> at least as much trust."""
    rng = np.random.RandomState(42)
    arr = np.sort(rng.uniform(-10, 0, size=50))   # increasing
    cfg = StructAwareBiasConfig(trust_floor=0.0, trust_ceiling=1.0)
    out = trust_from_chi_logp(arr, config=cfg)
    # Allow tiny numerical noise.
    assert np.all(np.diff(out) >= -1e-12)


# ----------------------------------------------------------------------
# apply_chi_trust_to_bias: math + catalytic passthrough
# ----------------------------------------------------------------------


def test_trust_one_preserves_bias_exactly():
    bias = np.random.RandomState(0).randn(5, 20)
    trust = np.ones(5)
    out, cat = apply_chi_trust_to_bias(bias, trust)
    np.testing.assert_array_equal(out, bias)
    assert cat == ()


def test_trust_zero_zeroes_bias_completely():
    bias = np.random.RandomState(0).randn(5, 20)
    trust = np.zeros(5)
    out, _ = apply_chi_trust_to_bias(bias, trust)
    np.testing.assert_array_equal(out, np.zeros((5, 20)))


def test_apply_chi_trust_partial_modulation():
    bias = np.ones((3, 20))
    trust = np.array([1.0, 0.5, 0.0])
    out, _ = apply_chi_trust_to_bias(bias, trust)
    np.testing.assert_array_equal(out[0], np.ones(20))
    np.testing.assert_array_equal(out[1], 0.5 * np.ones(20))
    np.testing.assert_array_equal(out[2], np.zeros(20))


def test_catalytic_passthrough_default_resno_to_index():
    """Default mapping is resno -> resno-1 (1-indexed PDB to 0-indexed)."""
    bias = np.ones((10, 20))
    trust = np.zeros(10)   # would zero everything
    cat_resnos = [3, 7]    # 1-indexed in PDB
    out, cat_idx = apply_chi_trust_to_bias(bias, trust, catalytic_resnos=cat_resnos)
    # Positions 2 and 6 (0-indexed) keep the original bias.
    np.testing.assert_array_equal(out[2], np.ones(20))
    np.testing.assert_array_equal(out[6], np.ones(20))
    # Other positions are zeroed.
    np.testing.assert_array_equal(out[0], np.zeros(20))
    np.testing.assert_array_equal(out[5], np.zeros(20))
    assert cat_idx == (2, 6)


def test_catalytic_passthrough_explicit_mapping():
    """When PDB resseq doesn't match 1-indexed positions (gaps, multi-chain
    etc.), the caller can supply an explicit resno -> 0-index map."""
    bias = np.ones((4, 20))
    trust = np.zeros(4)
    # PDB resseq 100, 101, 102, 103 -> 0-indexed positions 0, 1, 2, 3
    rmap = {100: 0, 101: 1, 102: 2, 103: 3}
    out, cat_idx = apply_chi_trust_to_bias(
        bias, trust, catalytic_resnos=[101, 103], resno_to_index=rmap,
    )
    assert cat_idx == (1, 3)
    # 0 and 2 are zeroed; 1 and 3 are preserved.
    np.testing.assert_array_equal(out[0], np.zeros(20))
    np.testing.assert_array_equal(out[1], np.ones(20))
    np.testing.assert_array_equal(out[2], np.zeros(20))
    np.testing.assert_array_equal(out[3], np.ones(20))


def test_catalytic_resnos_outside_range_silently_dropped():
    bias = np.ones((3, 20))
    trust = np.ones(3)
    out, cat_idx = apply_chi_trust_to_bias(bias, trust, catalytic_resnos=[100])
    # resno 100 -> 99 -> outside [0, 3) -> dropped
    assert cat_idx == ()


def test_apply_chi_trust_rejects_invalid_shapes():
    with pytest.raises(ValueError, match=r"\(L, 20\)"):
        apply_chi_trust_to_bias(np.zeros((5, 19)), np.ones(5))
    with pytest.raises(ValueError, match="trust shape"):
        apply_chi_trust_to_bias(np.zeros((5, 20)), np.ones(4))


def test_apply_chi_trust_rejects_out_of_range_trust():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_chi_trust_to_bias(np.zeros((3, 20)), np.array([1.5, 0.5, 0.0]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_chi_trust_to_bias(np.zeros((3, 20)), np.array([-0.1, 0.5, 1.0]))


def test_apply_chi_trust_preserves_neg_inf_hard_mask():
    """A position with bias=-inf for some AA (a hard 'never propose this
    AA' mask) must be PRESERVED through trust modulation, including at
    trust=0. Codex critique: ``-inf * 0 = NaN`` would silently corrupt
    a hard mask. We mask finite entries only.
    """
    bias = np.zeros((1, 20))
    bias[0, 5] = -np.inf
    # trust=0.5: finite entries shrink, -inf stays -inf.
    out, _ = apply_chi_trust_to_bias(bias, np.array([0.5]))
    assert np.isneginf(out[0, 5])
    assert not np.any(np.isnan(out))
    # trust=0: finite entries become 0, -inf stays -inf.
    out0, _ = apply_chi_trust_to_bias(bias, np.array([0.0]))
    assert np.isneginf(out0[0, 5])
    assert not np.any(np.isnan(out0))


# ----------------------------------------------------------------------
# fuse_plm_struct_logits: end-to-end orchestration
# ----------------------------------------------------------------------


def _flat_logp(L: int, seed: int = 0) -> np.ndarray:
    """Return (L, 20) log-probabilities (uniform = log(1/20))."""
    return np.full((L, 20), np.log(1.0 / 20))


def test_fuse_plm_struct_returns_correct_shape():
    L = 5
    res = fuse_plm_struct_logits(
        log_probs_esmc=_flat_logp(L),
        log_probs_saprot=_flat_logp(L),
        chi_logp_per_position=np.zeros(L),
        position_classes=["surface"] * L,
    )
    assert res.bias.shape == (L, 20)
    assert res.trust_per_position.shape == (L,)
    assert res.modulation_path == "v1_trust"


def test_fuse_plm_struct_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="PLM shape mismatch"):
        fuse_plm_struct_logits(
            log_probs_esmc=np.zeros((5, 20)),
            log_probs_saprot=np.zeros((6, 20)),
            chi_logp_per_position=np.zeros(5),
            position_classes=["surface"] * 5,
        )
    with pytest.raises(ValueError, match="chi_logp shape"):
        fuse_plm_struct_logits(
            log_probs_esmc=np.zeros((5, 20)),
            log_probs_saprot=np.zeros((5, 20)),
            chi_logp_per_position=np.zeros(4),   # wrong length
            position_classes=["surface"] * 5,
        )


def test_fuse_plm_struct_one_residue_protein():
    """Degenerate but should not crash."""
    res = fuse_plm_struct_logits(
        log_probs_esmc=_flat_logp(1),
        log_probs_saprot=_flat_logp(1),
        chi_logp_per_position=np.array([-1.0]),
        position_classes=["surface"],
    )
    assert res.bias.shape == (1, 20)
    assert 0.0 <= res.trust_per_position[0] <= 1.0


def test_fuse_plm_struct_catalytic_preserved():
    """Catalytic residues keep their PLM bias even when chi info is bad."""
    L = 5
    res = fuse_plm_struct_logits(
        log_probs_esmc=_flat_logp(L),
        log_probs_saprot=_flat_logp(L),
        chi_logp_per_position=np.full(L, -100.0),  # severe outliers everywhere
        position_classes=["surface"] * L,
        catalytic_resnos=[3],   # PDB resseq 3 -> 0-indexed 2
        struct_config=StructAwareBiasConfig(
            trust_floor=0.0,    # let the rest flatten
        ),
    )
    # All non-catalytic positions: trust ~ 0 -> bias ~ 0.
    # Catalytic position 2: passthrough -> bias = plm_fusion.bias[2].
    assert 2 in res.catalytic_indices
    np.testing.assert_array_equal(
        res.bias[2], res.plm_fusion.bias[2],
    )


def test_fuse_plm_struct_modulation_path_label():
    res = fuse_plm_struct_logits(
        log_probs_esmc=_flat_logp(3),
        log_probs_saprot=_flat_logp(3),
        chi_logp_per_position=np.zeros(3),
        position_classes=["surface"] * 3,
    )
    assert res.modulation_path == "v1_trust"


def test_fuse_plm_struct_nan_chi_passthrough():
    """A row with NaN chi_logp gets NO_CHI_TRUST -> bias preserved."""
    L = 3
    res = fuse_plm_struct_logits(
        log_probs_esmc=_flat_logp(L),
        log_probs_saprot=_flat_logp(L),
        chi_logp_per_position=np.array([-3.0, np.nan, -3.0]),
        position_classes=["surface"] * L,
    )
    # Position 1 has trust = NO_CHI_TRUST = 1.0 -> bias matches plm_fusion.
    np.testing.assert_array_equal(res.bias[1], res.plm_fusion.bias[1])
    # Other positions are modulated.
    assert not np.array_equal(res.bias[0], res.plm_fusion.bias[0])


def test_fuse_plm_struct_extreme_logp_does_not_explode():
    """Very large negative chi_logp shouldn't produce NaN bias rows."""
    L = 3
    res = fuse_plm_struct_logits(
        log_probs_esmc=_flat_logp(L),
        log_probs_saprot=_flat_logp(L),
        chi_logp_per_position=np.array([-1e6, -1e6, -1e6]),
        position_classes=["surface"] * L,
    )
    assert not np.any(np.isnan(res.bias))


# ----------------------------------------------------------------------
# V2 contract: skeleton raises NotImplementedError
# ----------------------------------------------------------------------


def test_v2_aa_logprior_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="V2"):
        aa_logprior_from_chi(
            chi_log_probs_per_residue_per_chi=np.zeros((5, 4, 36)),
            phi_psi_per_residue=np.zeros((5, 2)),
        )


def test_v2_dunbrack_table_placeholder_raises():
    from protein_chisel.sampling.struct_aware_bias import DunbrackTable
    with pytest.raises(NotImplementedError, match="Dunbrack"):
        DunbrackTable()


# ----------------------------------------------------------------------
# Concrete numeric sanity check
# ----------------------------------------------------------------------


def test_known_logp_to_known_trust_to_known_bias():
    """Anchor: chi_logp = -5, reference = -3, T = 2 -> z = -1 ->
    sigmoid(-1) ~= 0.2689.  bias = trust * plm_bias.
    """
    plm_bias = np.array([[1.0] * 20])
    trust = trust_from_chi_logp(
        np.array([-5.0]),
        config=StructAwareBiasConfig(
            reference_logp=-3.0, trust_temperature=2.0,
            trust_floor=0.0, trust_ceiling=1.0,
        ),
    )
    np.testing.assert_allclose(trust, [1.0 / (1.0 + np.exp(1.0))], rtol=1e-5)
    out, _ = apply_chi_trust_to_bias(plm_bias, trust)
    np.testing.assert_allclose(out, plm_bias * trust[:, None], rtol=1e-5)


# ----------------------------------------------------------------------
# Constants exposed
# ----------------------------------------------------------------------


def test_no_chi_aas_canonical():
    """Documented contract: only G and A are zero-chi in our world."""
    assert NO_CHI_AAS == frozenset({"G", "A"})


# ----------------------------------------------------------------------
# Regression tests for second-round implementation review
# ----------------------------------------------------------------------


def test_reference_logp_auto_derives_from_chi_bin_count():
    """Default reference_logp = -log(chi_bin_count) (codex/agent fix:
    hardcoded -3.0 was wrong for PIPPack's 36-bin chi grid)."""
    cfg36 = StructAwareBiasConfig(chi_bin_count=36)
    cfg20 = StructAwareBiasConfig(chi_bin_count=20)
    assert cfg36.resolved_reference_logp() == pytest.approx(-np.log(36), rel=1e-6)
    assert cfg20.resolved_reference_logp() == pytest.approx(-np.log(20), rel=1e-6)


def test_reference_logp_explicit_override():
    """Explicit reference_logp supersedes the chi_bin_count auto-derive."""
    cfg = StructAwareBiasConfig(chi_bin_count=36, reference_logp=-2.5)
    assert cfg.resolved_reference_logp() == -2.5


def test_invalid_chi_bin_count_raises():
    cfg = StructAwareBiasConfig(chi_bin_count=0)
    with pytest.raises(ValueError, match="chi_bin_count"):
        cfg.resolved_reference_logp()


def test_trust_temperature_nan_rejected():
    """Codex caught: NaN passes <= 0 silently; need np.isfinite."""
    with pytest.raises(ValueError, match="trust_temperature"):
        trust_from_chi_logp(
            np.array([0.0]),
            config=StructAwareBiasConfig(trust_temperature=float("nan")),
        )


def test_trust_temperature_inf_rejected():
    with pytest.raises(ValueError, match="trust_temperature"):
        trust_from_chi_logp(
            np.array([0.0]),
            config=StructAwareBiasConfig(trust_temperature=float("inf")),
        )


def test_apply_chi_trust_rejects_nan_trust():
    """Trust array containing NaN must raise (was silently passing the
    `(>=0) & (<=1)` check because NaN comparisons return False)."""
    with pytest.raises(ValueError, match="NaN"):
        apply_chi_trust_to_bias(
            np.zeros((3, 20)),
            np.array([0.5, np.nan, 0.5]),
        )


def test_fuse_plm_struct_empty_protein_raises():
    """L=0 must raise rather than silently return shape (0, 20)."""
    with pytest.raises(ValueError, match="empty protein"):
        fuse_plm_struct_logits(
            log_probs_esmc=np.zeros((0, 20)),
            log_probs_saprot=np.zeros((0, 20)),
            chi_logp_per_position=np.zeros(0),
            position_classes=[],
        )


def test_fuse_plm_struct_position_classes_length_mismatch_raises():
    """Wrong-length position_classes propagates from fuse_plm_logits."""
    with pytest.raises(ValueError):
        fuse_plm_struct_logits(
            log_probs_esmc=np.zeros((5, 20)),
            log_probs_saprot=np.zeros((5, 20)),
            chi_logp_per_position=np.zeros(5),
            position_classes=["surface"] * 4,   # off by one
        )


def test_fuse_plm_struct_all_nan_chi_full_passthrough():
    """Every position is Gly/Ala (all-NaN chi_logp) -> trust = 1.0
    everywhere -> bias matches plm_fusion exactly."""
    L = 5
    res = fuse_plm_struct_logits(
        log_probs_esmc=_flat_logp(L),
        log_probs_saprot=_flat_logp(L),
        chi_logp_per_position=np.full(L, np.nan),
        position_classes=["surface"] * L,
    )
    np.testing.assert_array_equal(res.bias, res.plm_fusion.bias)
    assert np.all(res.trust_per_position == NO_CHI_TRUST)


def test_catalytic_passthrough_false_actually_disables_passthrough():
    """If config.catalytic_passthrough=False, catalytic resnos are
    modulated like everyone else. Codex caught the flag was documented
    but ignored."""
    L = 3
    res = fuse_plm_struct_logits(
        log_probs_esmc=_flat_logp(L),
        log_probs_saprot=_flat_logp(L),
        chi_logp_per_position=np.full(L, -100.0),  # severe -> trust ~ 0
        position_classes=["surface"] * L,
        catalytic_resnos=[1, 2, 3],
        struct_config=StructAwareBiasConfig(
            catalytic_passthrough=False, trust_floor=0.0,
        ),
    )
    # Catalytic indices were NOT passthrough'd.
    assert res.catalytic_indices == ()
    # Bias is fully shrunk everywhere, including catalytic positions.
    assert np.allclose(res.bias, np.zeros_like(res.bias), atol=1e-6)


def test_resno_to_index_default_warns(caplog):
    """resno_to_index=None with non-empty catalytic_resnos must emit a
    warning (the identity fallback is a multi-chain footgun)."""
    import logging
    bias = np.zeros((10, 20))
    trust = np.ones(10)
    with caplog.at_level(logging.WARNING, logger="protein_chisel.struct_aware_bias"):
        apply_chi_trust_to_bias(bias, trust, catalytic_resnos=[3])
    assert any("identity mapping" in r.message for r in caplog.records)


def test_resno_to_index_explicit_no_warning(caplog):
    """When resno_to_index is given, the warning must NOT fire."""
    import logging
    bias = np.zeros((10, 20))
    trust = np.ones(10)
    with caplog.at_level(logging.WARNING, logger="protein_chisel.struct_aware_bias"):
        apply_chi_trust_to_bias(
            bias, trust,
            catalytic_resnos=[3],
            resno_to_index={3: 2},
        )
    assert not any("identity mapping" in r.message for r in caplog.records)


def test_per_position_chi_nll_no_runtime_warning():
    """The all-NaN-row reduction must not emit a RuntimeWarning
    (codex caught: np.nanmean warns; we replaced with manual sum/count)."""
    arr = np.array([
        [-1.0, -2.0, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = per_position_chi_nll(arr, aggregate="mean")
    assert out[0] == -1.5
    assert np.isnan(out[1])
    out2 = per_position_chi_nll(arr, aggregate="sum")
    assert out2[0] == -3.0
    assert np.isnan(out2[1])
