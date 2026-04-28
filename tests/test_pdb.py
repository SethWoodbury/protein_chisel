"""Tests for io/pdb.py — pure-text PDB helpers.

Uses the real test PDBs under /home/woodbuse/testing_space/align_seth_test/.
The KCX edge case (line 3: LYS at A64 with carbamate cap) is exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from protein_chisel.io.pdb import (
    AtomRecord,
    CatalyticResidue,
    extract_sequence,
    find_ligand,
    is_apo,
    parse_atom_record,
    parse_catres_spec,
    parse_remark_666,
    summarize_pdb,
    write_remark_666,
    ResidueRef,
)


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"
AF3_APO_PDB = TEST_DIR / "af3_pred.pdb"
REFINED_PDB = TEST_DIR / "refined.pdb"


# ---- atom record parsing ---------------------------------------------------


def test_parse_atom_record_atom_line():
    line = "ATOM      1  N   MET A   1       9.490 -14.537 -15.054  1.00  0.00           N  \n"
    rec = parse_atom_record(line)
    assert rec is not None
    assert rec.record == "ATOM"
    assert rec.serial == 1
    assert rec.name == "N"
    assert rec.res_name == "MET"
    assert rec.chain == "A"
    assert rec.res_seq == 1
    assert rec.element == "N"
    assert rec.x == pytest.approx(9.490)


def test_parse_atom_record_hetatm_zinc():
    line = "HETATM 3252 ZN2  YYE B 209       1.388   2.963   2.314  1.00  0.00          ZN  \n"
    rec = parse_atom_record(line)
    assert rec is not None
    assert rec.record == "HETATM"
    assert rec.name == "ZN2"
    assert rec.res_name == "YYE"
    assert rec.chain == "B"
    assert rec.element == "ZN"


def test_parse_atom_record_returns_none_for_non_atom():
    assert parse_atom_record("REMARK 666 MATCH TEMPLATE B YYE  209\n") is None
    assert parse_atom_record("HEADER\n") is None
    assert parse_atom_record("") is None


# ---- REMARK 666 parsing ----------------------------------------------------


def test_parse_remark_666_design_pdb():
    """The user's test PDB has 6 catalytic residues; the 3rd is the KCX edge case."""
    catres = parse_remark_666(DESIGN_PDB)
    assert len(catres) == 6
    # The keys are MOTIF residue numbers (catalytic residues on the protein)
    assert set(catres) == {188, 184, 64, 41, 148, 187}

    # Lys64 = the carbamylated lysine (KCX), referenced by REMARK 666 line 3
    lys64 = catres[64]
    assert lys64.chain == "A"
    assert lys64.name3 == "LYS"
    assert lys64.resno == 64
    assert lys64.target_chain == "B"
    assert lys64.target_name3 == "YYE"
    assert lys64.target_resno == 209
    assert lys64.cst_no == 3  # 3rd constraint
    assert lys64.cst_no_var == 1


def test_parse_remark_666_apo_pdb_empty():
    """The AF3 apo prediction has no REMARK 666 lines."""
    catres = parse_remark_666(AF3_APO_PDB)
    assert catres == {}


def test_parse_remark_666_refined_pdb():
    """The refined PDB (AF3 + ligand re-aligned) carries the same matcher residues."""
    catres = parse_remark_666(REFINED_PDB)
    assert len(catres) == 6


def test_remark_666_round_trip(tmp_path: Path):
    catres = parse_remark_666(DESIGN_PDB)
    out = tmp_path / "rewritten.pdb"
    write_remark_666(DESIGN_PDB, out, catres)
    catres2 = parse_remark_666(out)
    assert catres == catres2


def test_remark_666_serialization_format():
    cr = CatalyticResidue(
        chain="A", name3="HIS", resno=188,
        target_chain="B", target_name3="YYE", target_resno=209,
        cst_no=1, cst_no_var=1,
    )
    line = cr.to_remark_line()
    assert line.startswith("REMARK 666 MATCH TEMPLATE B YYE")
    assert "MATCH MOTIF A HIS" in line
    assert "188" in line
    assert line.endswith("\n")


def test_remark_666_inserts_when_no_existing(tmp_path: Path):
    """Writing remarks into a PDB that had none should put them just before ATOM."""
    src = tmp_path / "noremarks.pdb"
    src.write_text("HEADER\nATOM      1  N   MET A   1       0.0   0.0   0.0  1.00  0.00           N\nEND\n")
    catres = {
        188: CatalyticResidue(
            chain="A", name3="HIS", resno=188,
            target_chain="B", target_name3="YYE", target_resno=209,
            cst_no=1, cst_no_var=1,
        )
    }
    dst = tmp_path / "out.pdb"
    write_remark_666(src, dst, catres)
    body = dst.read_text()
    assert "REMARK 666" in body
    # Order: HEADER, REMARK 666, ATOM
    header_idx = body.index("HEADER")
    remark_idx = body.index("REMARK 666")
    atom_idx = body.index("ATOM")
    assert header_idx < remark_idx < atom_idx


# ---- catres spec parser ---------------------------------------------------


def test_parse_catres_spec_single():
    refs = parse_catres_spec(["A94"])
    assert refs == [ResidueRef(chain="A", resno=94)]


def test_parse_catres_spec_range():
    refs = parse_catres_spec(["A94-96"])
    assert refs == [
        ResidueRef("A", 94),
        ResidueRef("A", 95),
        ResidueRef("A", 96),
    ]


def test_parse_catres_spec_mixed():
    refs = parse_catres_spec(["A94-96", "B101"])
    assert len(refs) == 4
    assert refs[3] == ResidueRef("B", 101)


def test_parse_catres_spec_invalid():
    with pytest.raises(ValueError):
        parse_catres_spec(["junk"])


def test_parse_catres_spec_empty_skipped():
    assert parse_catres_spec(["", "A1", "  "]) == [ResidueRef("A", 1)]


# ---- summary / ligand detection -------------------------------------------


def test_summarize_design_pdb():
    s = summarize_pdb(DESIGN_PDB)
    assert s.n_atom == 3249
    assert s.n_hetatm == 39  # 39 YYE atoms (substrate + 2 Zn + carbamate cap)
    assert "A" in s.protein_chains
    # YYE on chain B
    assert any(name3 == "YYE" and chain == "B" for chain, name3, _ in s.ligand_residues)
    assert "ZN" in s.elements


def test_summarize_apo_pdb():
    s = summarize_pdb(AF3_APO_PDB)
    assert s.n_atom > 0
    assert s.n_hetatm == 0
    assert s.ligand_residues == []


def test_find_ligand_design():
    lig = find_ligand(DESIGN_PDB)
    assert lig is not None
    assert lig == ("B", "YYE", 209)


def test_find_ligand_apo():
    assert find_ligand(AF3_APO_PDB) is None


def test_is_apo_classification():
    assert is_apo(AF3_APO_PDB) is True
    assert is_apo(DESIGN_PDB) is False
    assert is_apo(REFINED_PDB) is False


# ---- sequence extraction --------------------------------------------------


def test_extract_sequence_design():
    seq = extract_sequence(DESIGN_PDB)
    # design.pdb chain A has 208 residues (protein), 3249 ATOM records total
    # (~15.6 atoms/residue with hydrogens, matches expectation).
    assert len(seq) == 208
    # Catalytic residues from REMARK 666 should be present at their resnos.
    catres = parse_remark_666(DESIGN_PDB)
    aa3to1 = {"HIS": "H", "LYS": "K", "GLU": "E"}
    for resno, cr in catres.items():
        # resnos are 1-indexed in PDBs and match list index = resno - 1
        assert seq[resno - 1] == aa3to1[cr.name3], (
            f"resno {resno} expected {cr.name3} got {seq[resno - 1]}"
        )


def test_extract_sequence_af3():
    """AF3 prediction of the same design should have the same length."""
    seq_design = extract_sequence(DESIGN_PDB)
    seq_af3 = extract_sequence(AF3_APO_PDB)
    assert len(seq_design) == len(seq_af3)
    # And identical sequence (same designed sequence, different fold)
    assert seq_design == seq_af3
