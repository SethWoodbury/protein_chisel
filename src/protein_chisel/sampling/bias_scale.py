"""Temperature-aware bias scaling for LigandMPNN per-position bias.

LigandMPNN samples ``softmax((logits + bias) / T)`` (verified in the fused_mpnn
``model_utils.py`` sampler), so a bias of ``b`` nats shifts an amino acid's odds by
``exp(b / T)`` — **not** ``exp(b)``. Lowering the sampling temperature therefore
amplifies *every* bias by ``1/T``. Calibrating a bias in raw nats (as the pipeline
historically did) silently turns a "gentle nudge" at T=1.0 into a near-deterministic
lock at T=0.15 — e.g. the PLM fusion mean (~1.36 nats) is ~8,700× odds at T=0.15, while
the charge controller (~0.6 nats) is only ~55× and is dwarfed by the locks it stacks
under. (Confirmed by an independent codex + subagent bias-scaling audit.)

These helpers let callers reason in **odds space**, which is temperature-invariant:
pick the odds multiplier you want ("8× more likely") and convert to the right nats at
the cycle's actual sampling temperature via :func:`nats_for_odds`. The intent then means
the same thing whether the cycle samples at T=0.20 or anneals to T=0.15.
"""
from __future__ import annotations

import math

# Odds-multiplier vocabulary (temperature-invariant intent). Convert to nats at the
# actual sampling temperature with nats_for_odds(); these are deliberately modest
# relative to the historical raw-nat magnitudes (which were locks at low T).
ODDS_NUDGE = 2.0      # gentle: ~2× more/less likely — lets other signals still win
ODDS_STRONG = 8.0     # firm: clearly preferred/disfavored
ODDS_LOCK = 100.0     # near-deterministic — but NOT a true ban (use an omit mask for that)


def nats_for_odds(odds_multiplier: float, temperature: float) -> float:
    """Nats of bias that yield ``odds_multiplier`` at sampling ``temperature``.

    Because MPNN samples ``softmax((logits + bias)/T)``, the odds shift is
    ``exp(bias/T)``; inverting gives ``bias = T * ln(odds_multiplier)``. Expressing
    intent as an odds multiplier makes it invariant to the temperature schedule.
    Inverse of :func:`odds_for_nats`.
    """
    if not odds_multiplier > 0:
        raise ValueError(f"odds_multiplier must be > 0, got {odds_multiplier}")
    if not temperature > 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return temperature * math.log(odds_multiplier)


def odds_for_nats(nats: float, temperature: float) -> float:
    """Effective odds multiplier of a ``nats`` bias at sampling ``temperature``.

    ``exp(nats / temperature)`` — the quantity a reader should think about when judging
    whether a bias is a nudge, a strong push, or an effective lock.
    """
    if not temperature > 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return math.exp(nats / temperature)


def effective_clamp_nats(max_nats, max_odds, temperature):
    """The per-axis bias clamp in nats — temperature-invariant when opted in.

    When ``max_odds`` is given AND ``temperature`` is usable (>0), return the nats that
    yield that odds multiplier at this cycle's temperature (so a controller's authority
    is invariant to the T-schedule). Otherwise return the legacy raw ``max_nats`` — so a
    caller that does not set ``max_odds`` is byte-identical to the prior behaviour.
    """
    if max_odds is not None and temperature is not None and temperature > 0:
        return nats_for_odds(max_odds, temperature)
    return max_nats
