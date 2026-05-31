"""Unit tests for tools.conserved_hbonds (designable sidechain H-bond detection)."""
from __future__ import annotations

from protein_chisel.tools.conserved_hbonds import (
    find_conservable_sidechain_hbonds,
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
