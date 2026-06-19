"""Shared 'bulky blocker' AA classification (tunnel_metrics.bulky_blocker_aas).

The throat-feedback controller's ``_BLOCKER_WEIGHT`` table is the single source of
truth for which residues are "bulky" (channel-constricting). ``bulky_blocker_aas``
exposes that classification as a deterministic 1-letter string, so other opt-in
mechanisms (e.g. the WS-G tunnel-lining omit) stay consistent with the throat *by
construction* instead of hand-maintaining a parallel list.
"""
from protein_chisel.tools.tunnel_metrics import bulky_blocker_aas


def test_bulky_default_includes_lysine_and_arginine():
    # Lys (Cb->NZ ~5.5 A) and Arg (~6 A) are long enough to line + constrict the
    # throat: both sit at _BLOCKER_WEIGHT == 0.70 == the default bulky_threshold,
    # so both must count as bulky.
    bulky = bulky_blocker_aas()                      # default threshold 0.70
    assert "K" in bulky
    assert "R" in bulky
    # Full >=0.70 set: aromatics W/F/Y/H + long charged R/K.
    assert bulky == "FHKRWY"


def test_bulky_threshold_one_is_tryptophan_only():
    assert bulky_blocker_aas(1.0) == "W"


def test_bulky_at_aromatic_threshold():
    # >=0.85 = Trp(1.0) + Phe/Tyr/His(0.85).
    assert bulky_blocker_aas(0.85) == "FHWY"


def test_bulky_lower_threshold_adds_medium_tier():
    # >=0.55 brings in the medium hydrophobics M/L/I/V.
    assert bulky_blocker_aas(0.55) == "FHIKLMRVWY"


def test_bulky_is_sorted_and_deterministic():
    b = bulky_blocker_aas(0.45)
    assert list(b) == sorted(b)
    # Same call twice is stable.
    assert bulky_blocker_aas(0.45) == b
