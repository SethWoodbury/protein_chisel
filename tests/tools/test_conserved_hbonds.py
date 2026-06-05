"""Unit tests for tools.conserved_hbonds (designable sidechain H-bond detection)."""
from __future__ import annotations

import random

import pytest

from protein_chisel.tools import conserved_hbonds as ch
from protein_chisel.tools.conserved_hbonds import (
    ConservableHbond,
    build_interaction_network,
    find_conservable_sidechain_hbonds,
    normalize_probability,
    roll_conserved,
    select_conservable_resnos,
    strength_bin,
)


def _atom(serial, name, resname, chain, resno, x, y, z, record="ATOM", element="C"):
    """One PDB ATOM/HETATM line with the columns the parser reads."""
    return (
        f"{record.ljust(6)}{serial:>5} {name.ljust(4)[:4]}{resname.ljust(5)[:5]}"
        f"{chain}{resno:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{1.0:>6.2f}{0.0:>6.2f}          {element:>2}\n"
    )


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("".join(lines) + "END\n")
    return str(p)


# ---- a designable Tyr54 whose OH H-bonds a catalytic His16 ND1 at ~2.7 A ----
def _tyr_his(tmp_path):
    lines = [
        # designable Tyr54: antecedent CZ at z=0, OH at z=1.4 (sidechain donor/acceptor)
        _atom(1, "CZ", "TYR", "A", 54, 0, 0, 0.0, element="C"),
        _atom(2, "OH", "TYR", "A", 54, 0, 0, 1.4, element="O"),
        _atom(3, "N", "TYR", "A", 54, 0, 5, 0.0, element="N"),   # far backbone
        _atom(4, "CA", "TYR", "A", 54, 1, 5, 0.0, element="C"),
        # catalytic His16: ND1 acceptor at z=4.3 (OH...ND1 = 2.9), CG antecedent far
        _atom(5, "ND1", "HIS", "A", 16, 0, 0, 4.3, element="N"),
        _atom(6, "CG", "HIS", "A", 16, 0, 0, 5.7, element="C"),
    ]
    return _write(tmp_path, "tyr_his.pdb", lines)


def test_tyr_sidechain_to_catalytic_his(tmp_path):
    recs = find_conservable_sidechain_hbonds(
        _tyr_his(tmp_path), designable_resnos=[54], catalytic_resnos=[16], chain="A")
    assert len(recs) == 1
    r = recs[0]
    assert r.resno == 54 and r.sidechain_atom == "OH"
    assert r.partner_kind == "catalytic" and r.partner_resno == 16
    assert 2.8 < r.distance < 3.0 and r.strength_bin == "super_strong"
    assert not r.clashes


def test_backbone_donor_excluded(tmp_path):
    lines = [
        # designable Lys30: backbone N donor near the ligand; sidechain NZ far away
        _atom(1, "N", "LYS", "A", 30, 0, 0, 0.0, element="N"),
        _atom(2, "CA", "LYS", "A", 30, 0, 0, -1.5, element="C"),
        _atom(3, "NZ", "LYS", "A", 30, 9, 9, 9.0, element="N"),
        # ligand acceptor O at z=2.8 (backbone N...O = 2.8)
        _atom(4, "O1", "LIG", "B", 200, 0, 0, 2.8, record="HETATM", element="O"),
    ]
    pdb = _write(tmp_path, "bb.pdb", lines)
    recs = find_conservable_sidechain_hbonds(
        pdb, designable_resnos=[30], catalytic_resnos=[], chain="A")
    assert recs == []  # backbone N H-bond doesn't count


def test_sidechain_to_ligand_included(tmp_path):
    lines = [
        _atom(1, "CB", "SER", "A", 40, 0, 0, 0.0, element="C"),
        _atom(2, "OG", "SER", "A", 40, 0, 0, 1.4, element="O"),
        _atom(3, "O3", "LIG", "B", 200, 0, 0, 4.1, record="HETATM", element="O"),
    ]
    pdb = _write(tmp_path, "lig.pdb", lines)
    recs = find_conservable_sidechain_hbonds(
        pdb, designable_resnos=[40], catalytic_resnos=[], include_ligand=True, chain="A")
    assert len(recs) == 1 and recs[0].partner_kind == "ligand"
    assert recs[0].partner_resno == -1 and recs[0].sidechain_atom == "OG"


def test_his_tautomer_both_nitrogens(tmp_path):
    # designable His50 (labeled HIE) must still donate from ND1 and accept on NE2
    lines = [
        _atom(1, "CG", "HIE", "A", 50, 0, 0, 0.0, element="C"),
        _atom(2, "ND1", "HIE", "A", 50, 0, 0, 1.4, element="N"),       # treat as donor
        _atom(3, "CD2", "HIE", "A", 50, 2, 0, 0.0, element="C"),
        _atom(4, "NE2", "HIE", "A", 50, 3.4, 0, 0.0, element="N"),     # treat as acceptor
        # catalytic Asp60 OD1 acceptor near ND1 (ND1...OD1 = 2.7)
        _atom(5, "OD1", "ASP", "A", 60, 0, 0, 4.1, element="O"),
        _atom(6, "CG", "ASP", "A", 60, 0, 0, 5.5, element="C"),
        # catalytic Ser70 OG donor near NE2 (NE2...OG = 2.7), antecedent CB
        _atom(7, "OG", "SER", "A", 70, 6.1, 0, 0.0, element="O"),
        _atom(8, "CB", "SER", "A", 70, 7.5, 0, 0.0, element="C"),
    ]
    pdb = _write(tmp_path, "his.pdb", lines)
    recs = find_conservable_sidechain_hbonds(
        pdb, designable_resnos=[50], catalytic_resnos=[60, 70], chain="A")
    atoms_used = {r.sidechain_atom for r in recs}
    assert "ND1" in atoms_used and "NE2" in atoms_used  # both tautomer N participate


def test_clash_excluded_then_kept(tmp_path):
    # Ser40 OG H-bonds catalytic His16 ND1; Ser CB also clashes a fixed backbone atom.
    lines = [
        _atom(1, "CB", "SER", "A", 40, 0, 0, 0.0, element="C"),
        _atom(2, "OG", "SER", "A", 40, 0, 0, 1.4, element="O"),
        _atom(3, "ND1", "HIS", "A", 16, 0, 0, 4.1, element="N"),
        _atom(4, "CG", "HIS", "A", 16, 0, 0, 5.5, element="C"),
        # a fixed residue's backbone O 1.2 A from Ser CB -> clash (won't be redesigned)
        _atom(5, "O", "GLY", "A", 90, 1.2, 0, 0.0, element="O"),
        _atom(6, "CA", "GLY", "A", 90, 2.5, 0, 0.0, element="C"),
    ]
    pdb = _write(tmp_path, "clash.pdb", lines)
    recs = find_conservable_sidechain_hbonds(
        pdb, designable_resnos=[40], catalytic_resnos=[16], chain="A")
    assert len(recs) == 1 and recs[0].clashes and recs[0].clash_with.startswith("GLY90")


def test_strength_bins():
    assert strength_bin(0.95) == "super_strong"
    assert strength_bin(0.70) == "strong"
    assert strength_bin(0.50) == "moderate"
    assert strength_bin(0.30) == "weak"
    assert strength_bin(0.05) == "super_weak"


# ----------------------------------------------------------------------
# shared selection / rolling helpers (used by chisel_ligandMPNN.py AND
# iterative_design.py — one implementation for both)
# ----------------------------------------------------------------------
def test_normalize_probability_decimal_and_percentage():
    assert normalize_probability(0.8) == 0.8
    assert normalize_probability(0.0) == 0.0
    assert normalize_probability(1.0) == 1.0
    assert normalize_probability(80) == pytest.approx(0.8)   # percentage -> /100
    assert normalize_probability(100) == pytest.approx(1.0)


def test_normalize_probability_errors():
    with pytest.raises(ValueError):
        normalize_probability(-0.1)
    with pytest.raises(ValueError):
        normalize_probability(150)


def test_roll_conserved_bounds_and_reproducible():
    items = [10, 20, 30, 40, 50]                 # resno ints
    assert roll_conserved(items, 1.0, random.Random(0)) == set(items)
    assert roll_conserved(items, 0.0, random.Random(0)) == set()
    a = roll_conserved(items, 0.5, random.Random("7:0"))
    b = roll_conserved(items, 0.5, random.Random("7:0"))
    assert a == b and a.issubset(set(items))     # same seed -> identical
    # type-agnostic: also works on "A157"-style labels
    labels = ["A10", "A20", "A30"]
    assert roll_conserved(labels, 1.0, random.Random(0)) == set(labels)


def _rec(resno, *, clashes=False, clash_with=""):
    return ConservableHbond(
        resno=resno, resname="SER", sidechain_atom="OG",
        partner_kind="ligand", partner_resno=-1, partner_resname="LIG",
        partner_atom="O1", distance=2.8, strength=0.9, strength_bin="super_strong",
        hypothesized_donor=f"SER{resno}/OG", hypothesized_acceptor="LIG/O1",
        clashes=clashes, clash_with=clash_with,
    )


def test_select_conservable_resnos_excludes_clashing_by_default():
    recs = [_rec(40), _rec(30, clashes=True, clash_with="GLY90/O"), _rec(40)]
    candidates, excluded = select_conservable_resnos(recs)
    assert candidates == [40]                    # deduped + sorted, clashing dropped
    assert excluded == [(30, "GLY90/O")]


def test_select_conservable_resnos_keep_clashing():
    recs = [_rec(40), _rec(30, clashes=True, clash_with="GLY90/O")]
    candidates, excluded = select_conservable_resnos(recs, keep_clashing=True)
    assert candidates == [30, 40] and excluded == []


# ----------------------------------------------------------------------
# build_interaction_network (recursive shells from the active site)
# ----------------------------------------------------------------------
def test_network_depth1_reduces_to_single_shell(tmp_path):
    # Real geometry: Tyr54 OH -> catalytic His16 ND1. depth=1 must match the
    # legacy find_conservable + select + roll path exactly.
    pdb = _tyr_his(tmp_path)
    recs = find_conservable_sidechain_hbonds(
        pdb, designable_resnos=[54], catalytic_resnos=[16], chain="A")
    cands, _ = select_conservable_resnos(recs)
    import random
    legacy_rolled = roll_conserved(cands, 1.0, random.Random("7:0"))
    net = build_interaction_network(
        pdb, designable_resnos=[54], catalytic_resnos=[16], include_ligand=False,
        chain="A", depth=1, prob=1.0, seed="7:0")
    assert net.rolled == legacy_rolled == {54}
    assert len(net.shells) == 1 and net.shells[0].depth == 1


def test_network_unsupported_type_raises(tmp_path):
    pdb = _tyr_his(tmp_path)
    with pytest.raises(NotImplementedError):
        build_interaction_network(
            pdb, designable_resnos=[54], catalytic_resnos=[16],
            interaction_types=("salt_bridge",), depth=1)


def test_network_grows_shells_bfs(tmp_path, monkeypatch):
    # Mock the detector to make a 2-shell chain: ligand<-res40 (shell1),
    # res40<-res50 (shell2). Verifies BFS frontier advance + per-shell rolls.
    def fake_detect(pdb_path, *, designable_resnos, catalytic_resnos,
                    user_fixed_resnos=(), include_ligand=True, chain="A", **kw):
        anchors = set(catalytic_resnos) | set(user_fixed_resnos)
        if include_ligand:                       # shell 1: bonds the ligand
            return [_rec(40)] if 40 in set(designable_resnos) else []
        if 40 in anchors:                        # shell 2: bonds res 40
            return [_rec(50)] if 50 in set(designable_resnos) else []
        return []
    monkeypatch.setattr(ch, "find_conservable_sidechain_hbonds", fake_detect)

    net1 = build_interaction_network(
        "x.pdb", designable_resnos=[40, 50], catalytic_resnos=[16],
        depth=1, prob=1.0, seed="s")
    assert net1.rolled == {40} and len(net1.shells) == 1

    net2 = build_interaction_network(
        "x.pdb", designable_resnos=[40, 50], catalytic_resnos=[16],
        depth=2, prob=1.0, seed="s")
    assert net2.rolled == {40, 50}
    assert [s.depth for s in net2.shells] == [1, 2]
    assert net2.shells[1].anchors == [40]        # shell-2 bonded shell-1's residue


def test_network_ligand_only_anchors_still_conserves(tmp_path):
    # Regression (C1): with NO catalytic/user-fixed residues but ligand active,
    # shell 1 must still detect+conserve ligand H-bonders (legacy behavior).
    lines = [
        _atom(1, "CB", "SER", "A", 40, 0, 0, 0.0, element="C"),
        _atom(2, "OG", "SER", "A", 40, 0, 0, 1.4, element="O"),
        _atom(3, "O3", "LIG", "B", 200, 0, 0, 4.1, record="HETATM", element="O"),
    ]
    pdb = _write(tmp_path, "ligonly.pdb", lines)
    net = build_interaction_network(
        pdb, designable_resnos=[40], catalytic_resnos=[], include_ligand=True,
        chain="A", depth=1, prob=1.0, seed="s")
    assert net.rolled == {40} and len(net.shells) == 1


def test_network_duplicate_hbond_type_ok(tmp_path):
    pdb = _tyr_his(tmp_path)
    net = build_interaction_network(
        pdb, designable_resnos=[54], catalytic_resnos=[16], include_ligand=False,
        interaction_types=("hbond", "hbond"), depth=1, prob=1.0, seed="s")
    assert net.rolled == {54}


def test_network_shell_decay_zero_stops_after_shell1(monkeypatch):
    # decay=0 -> shell-2 prob 0 -> nothing rolled -> network stops at shell 1.
    def fake_detect(pdb_path, *, designable_resnos, catalytic_resnos,
                    user_fixed_resnos=(), include_ligand=True, chain="A", **kw):
        if include_ligand:
            return [_rec(40)] if 40 in set(designable_resnos) else []
        if 40 in (set(catalytic_resnos) | set(user_fixed_resnos)):
            return [_rec(50)] if 50 in set(designable_resnos) else []
        return []
    monkeypatch.setattr(ch, "find_conservable_sidechain_hbonds", fake_detect)
    net = build_interaction_network(
        "x.pdb", designable_resnos=[40, 50], catalytic_resnos=[16],
        depth=3, prob=1.0, shell_decay=0.0, seed="s")
    assert net.rolled == {40}            # shell 1 rolled; shell 2 prob 0 -> none


def test_network_stops_when_shell_empty(tmp_path, monkeypatch):
    # Only shell 1 produces a residue; depth=3 should stop after shell 2 is empty.
    def fake_detect(pdb_path, *, designable_resnos, catalytic_resnos,
                    user_fixed_resnos=(), include_ligand=True, chain="A", **kw):
        return [_rec(40)] if include_ligand and 40 in set(designable_resnos) else []
    monkeypatch.setattr(ch, "find_conservable_sidechain_hbonds", fake_detect)
    net = build_interaction_network(
        "x.pdb", designable_resnos=[40, 50], catalytic_resnos=[16],
        depth=3, prob=1.0, seed="s")
    assert net.rolled == {40}
    assert len(net.shells) == 1                  # stopped once the frontier emptied
