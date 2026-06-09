"""Lock the His tautomer -> Rosetta residue-type translation.

Rosetta only knows `HIS` (epsilon, proton on NE2 — its default) and `HIS_D`
(delta, proton on ND1). The Amber-style codes HIE/HID/HIP from REMARK 668 /
pdb_restoration must translate to those, or the catalytic His tautomer silently
reverts to default at protonation. This guards against an accidental HIE<->HID
swap (which would flip catalytic protonation) and against regressions to the
prior "unknown HIE -> default" behavior.
"""
from __future__ import annotations

from protein_chisel.tools.protonate_final import (
    _ROSETTA_TO_STATE,
    _TAUTOMER_TO_ROSETTA_RESTYPE,
)


def test_epsilon_maps_to_default_his():
    # HIE / HIS_E (epsilon, NE2) -> Rosetta default "HIS"
    assert _TAUTOMER_TO_ROSETTA_RESTYPE["HIE"] == "HIS"
    assert _TAUTOMER_TO_ROSETTA_RESTYPE["HIS_E"] == "HIS"


def test_delta_maps_to_his_d():
    # HID (delta, ND1) -> Rosetta "HIS_D"
    assert _TAUTOMER_TO_ROSETTA_RESTYPE["HID"] == "HIS_D"


def test_cation_maps_to_his_p():
    assert _TAUTOMER_TO_ROSETTA_RESTYPE["HIP"] == "HIS_P"


def test_no_accidental_swap():
    # The dangerous bug would be HIE->HIS_D or HID->HIS (flipped tautomer).
    assert _TAUTOMER_TO_ROSETTA_RESTYPE["HIE"] != "HIS_D"
    assert _TAUTOMER_TO_ROSETTA_RESTYPE["HID"] != "HIS"


def test_roundtrip_consistent_with_state_map():
    # The forward map (tautomer->restype) must be consistent with the reverse
    # state map (restype->STATE) used to rebuild REMARK 668.
    assert _ROSETTA_TO_STATE["HIS"] == "HIE"     # default HIS reports epsilon
    assert _ROSETTA_TO_STATE["HIS_D"] == "HID"   # HIS_D reports delta
    # HIE -> HIS -> HIE  and  HID -> HIS_D -> HID
    assert _ROSETTA_TO_STATE[_TAUTOMER_TO_ROSETTA_RESTYPE["HIE"]] == "HIE"
    assert _ROSETTA_TO_STATE[_TAUTOMER_TO_ROSETTA_RESTYPE["HID"]] == "HID"
