"""Temperature-aware bias scaling helpers (sampling.bias_scale).

LigandMPNN samples ``softmax((logits + bias)/T)``, so a bias of ``b`` nats shifts an
amino acid's odds by ``exp(b/T)``. These helpers let callers express intent in
temperature-invariant odds space and convert to the right nats at the actual sampling
temperature.
"""
import math

import pytest

from protein_chisel.sampling.bias_scale import (
    nats_for_odds, odds_for_nats, effective_clamp_nats,
    ODDS_NUDGE, ODDS_STRONG, ODDS_LOCK,
)


def test_effective_clamp_nats_legacy_when_no_max_odds():
    # max_odds=None -> the raw max_nats (byte-identical legacy path), regardless of T.
    assert effective_clamp_nats(0.6, None, 0.15) == 0.6
    assert effective_clamp_nats(0.6, None, None) == 0.6


def test_effective_clamp_nats_is_odds_space_when_set():
    # max_odds set + T given -> temperature-invariant nats for that odds multiplier.
    assert effective_clamp_nats(0.6, ODDS_STRONG, 0.15) == pytest.approx(0.15 * math.log(8.0))
    assert effective_clamp_nats(0.6, ODDS_STRONG, 0.30) == pytest.approx(0.30 * math.log(8.0))
    # falls back to legacy max_nats if T is missing/invalid (can't compute odds)
    assert effective_clamp_nats(0.6, ODDS_STRONG, None) == 0.6
    assert effective_clamp_nats(0.6, ODDS_STRONG, 0.0) == 0.6


def test_odds_for_nats_matches_exp_bias_over_T():
    # The MPNN sampling identity: effective odds = exp(bias / T).
    assert odds_for_nats(0.6, 0.15) == pytest.approx(math.exp(0.6 / 0.15))   # ~55x
    assert odds_for_nats(-0.21, 0.15) == pytest.approx(math.exp(-0.21 / 0.15))  # ~0.25x
    assert odds_for_nats(0.0, 0.2) == pytest.approx(1.0)


def test_nats_for_odds_is_inverse_of_odds_for_nats():
    for T in (0.15, 0.18, 0.20, 0.30):
        for M in (1.5, 2.0, 8.0, 100.0, 0.1):
            b = nats_for_odds(M, T)
            assert odds_for_nats(b, T) == pytest.approx(M)
            # closed form: b = T * ln(M)
            assert b == pytest.approx(T * math.log(M))


def test_nats_for_odds_is_temperature_proportional():
    # The SAME odds intent needs LESS nats at lower T (the whole point).
    assert nats_for_odds(ODDS_STRONG, 0.15) < nats_for_odds(ODDS_STRONG, 0.30)
    # 8x at T=0.15 is ~0.31 nats, not the raw "0.6" the pipeline used for ~strong.
    assert nats_for_odds(ODDS_STRONG, 0.15) == pytest.approx(0.15 * math.log(8.0))


def test_odds_vocabulary_ordered_and_sane():
    assert 1.0 < ODDS_NUDGE < ODDS_STRONG < ODDS_LOCK
    # A "lock" is strong but NOT a true ban — finite nats, never -inf.
    assert math.isfinite(nats_for_odds(ODDS_LOCK, 0.15))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        nats_for_odds(bad, 0.15)
    with pytest.raises(ValueError):
        nats_for_odds(2.0, bad)
    with pytest.raises(ValueError):
        odds_for_nats(0.5, bad)
