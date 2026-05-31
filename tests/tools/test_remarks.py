"""Unit tests for protein_chisel.tools.remarks (REMARK transfer + DESIGN_PATH)."""
from __future__ import annotations

from pathlib import Path

from protein_chisel.tools import remarks

# A minimal "design" output PDB (no REMARKs — like a fresh LigandMPNN dump).
_OUT_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "TER\nEND\n"
)

# A seed/input PDB carrying several REMARK types to rescue.
_IN_PDB = (
    "REMARK 665 column-key header for REMARK 666\n"
    "REMARK 666 MATCH TEMPLATE B LIG  200 MATCH MOTIF A HIS   60  1  1\n"
    "REMARK QCB TOTAL_CHARGE +1\n"
    "REMARK DESIGN_PATH rfd3 output /net/scratch/seed.pdb\n"
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "TER\nEND\n"
)


def _write(p: Path, text: str) -> str:
    p.write_text(text)
    return str(p)


def test_transfer_rescues_input_remarks_in_canonical_order(tmp_path):
    out = _write(tmp_path / "design.pdb", _OUT_PDB)
    inp = _write(tmp_path / "seed.pdb", _IN_PDB)
    remarks.reorganize_pdb_remarks(out, inp)
    text = Path(out).read_text()
    assert "REMARK 665" in text
    assert "REMARK 666 MATCH TEMPLATE B LIG  200" in text
    assert "REMARK QCB TOTAL_CHARGE +1" in text
    assert "REMARK DESIGN_PATH rfd3 output /net/scratch/seed.pdb" in text
    # canonical order: numbered (665/666) -> QCB -> DESIGN_PATH -> body
    lines = text.splitlines()
    i666 = next(i for i, l in enumerate(lines) if l.startswith("REMARK 666"))
    iqcb = next(i for i, l in enumerate(lines) if l.startswith("REMARK QCB"))
    idp = next(i for i, l in enumerate(lines) if l.startswith("REMARK DESIGN_PATH"))
    iatom = next(i for i, l in enumerate(lines) if l.startswith("ATOM"))
    assert i666 < iqcb < idp < iatom


def test_design_path_stage_appends_exactly_one(tmp_path):
    out = _write(tmp_path / "design.pdb", _OUT_PDB)
    inp = _write(tmp_path / "seed.pdb", _IN_PDB)
    remarks.reorganize_pdb_remarks(out, inp, design_path_stage="chisel_ligandmpnn")
    lines = Path(out).read_text().splitlines()
    chisel = [l for l in lines if l.startswith("REMARK DESIGN_PATH chisel_ligandmpnn output")]
    assert len(chisel) == 1
    assert chisel[0].endswith(str(Path(out)))           # normalized output path
    # the upstream rfd3 DESIGN_PATH is preserved alongside the new one
    assert any(l.startswith("REMARK DESIGN_PATH rfd3 output") for l in lines)


def test_design_path_stage_none_adds_nothing(tmp_path):
    out = _write(tmp_path / "design.pdb", _OUT_PDB)
    inp = _write(tmp_path / "seed.pdb", _IN_PDB)
    remarks.reorganize_pdb_remarks(out, inp, design_path_stage=None)
    lines = Path(out).read_text().splitlines()
    assert not any("chisel_ligandmpnn" in l for l in lines)


def test_truncation_and_exact_dups_deduped(tmp_path):
    full = "REMARK 666 MATCH TEMPLATE B LIG  200 MATCH MOTIF A HIS   60  1  1  trailing detail"
    out = _write(tmp_path / "design.pdb",
                 f"{full[:80]}\n"                       # col-80 truncation of input line
                 "REMARK QCB TOTAL_CHARGE +1\n"          # exact dup of an input line
                 "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
                 "TER\nEND\n")
    inp = _write(tmp_path / "seed.pdb",
                 f"{full}\nREMARK QCB TOTAL_CHARGE +1\n"
                 "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
                 "TER\nEND\n")
    remarks.reorganize_pdb_remarks(out, inp)
    lines = Path(out).read_text().splitlines()
    assert sum(l.startswith("REMARK 666") for l in lines) == 1   # truncated dropped
    assert [l for l in lines if l.startswith("REMARK 666")][0] == full
    assert sum(l.startswith("REMARK QCB") for l in lines) == 1   # exact dup deduped


def test_reorganize_without_input_just_records_design_path(tmp_path):
    out = _write(tmp_path / "design.pdb", _OUT_PDB)
    remarks.reorganize_pdb_remarks(out, None, design_path_stage="iterative_design")
    lines = Path(out).read_text().splitlines()
    dp = [l for l in lines if l.startswith("REMARK DESIGN_PATH iterative_design output")]
    assert len(dp) == 1


def test_transfer_remarks_to_dir_skips_intermediates(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write(d / "a.pdb", _OUT_PDB)
    _write(d / "b.pdb", _OUT_PDB)
    _write(d / "a.protonated.pdb", _OUT_PDB)           # intermediate -> skipped
    inp = _write(tmp_path / "seed.pdb", _IN_PDB)        # seed lives outside the design dir
    n = remarks.transfer_remarks_to_dir(d, inp, design_path_stage="chisel_ligandmpnn")
    assert n == 2
    assert "REMARK 666" in (d / "a.pdb").read_text()
    assert "REMARK 666" in (d / "b.pdb").read_text()
    assert "REMARK 666" not in (d / "a.protonated.pdb").read_text()


def test_replace_remark_block_overrides_after_666(tmp_path):
    pdb = _write(tmp_path / "x.pdb",
                 "REMARK 666 MATCH TEMPLATE B LIG  200 MATCH MOTIF A LYS   76  4  1\n"
                 "REMARK 668   4   A LYS     76   LYS -   -                       LYS\n"  # stale
                 "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
                 "END\n")
    new = ["REMARK 667 header\n",
           "REMARK 668   4   A LYS     76   LYS KCX -                       LYS\n"]
    assert remarks.replace_remark_block(pdb, new) is True
    lines = Path(pdb).read_text().splitlines()
    six68 = [l for l in lines if l.startswith("REMARK 668")]
    assert len(six68) == 1 and "KCX" in six68[0]          # stale replaced
    i666 = next(i for i, l in enumerate(lines) if l.startswith("REMARK 666"))
    i667 = next(i for i, l in enumerate(lines) if l.startswith("REMARK 667"))
    iatom = next(i for i, l in enumerate(lines) if l.startswith("ATOM"))
    assert i666 < i667 < iatom                            # inserted after 666, before body


def test_replace_remark_block_empty_is_noop(tmp_path):
    pdb = _write(tmp_path / "x.pdb", _OUT_PDB)
    before = Path(pdb).read_text()
    assert remarks.replace_remark_block(pdb, []) is False
    assert Path(pdb).read_text() == before


def test_normalize_design_path_line():
    line = "REMARK DESIGN_PATH stage output /a//b/./c.pdb\n"
    assert remarks.normalize_design_path_line(line) == \
        "REMARK DESIGN_PATH stage output /a/b/c.pdb\n"
    other = "REMARK 666 MATCH TEMPLATE B LIG 200\n"
    assert remarks.normalize_design_path_line(other) == other


def test_is_intermediate():
    assert remarks.is_intermediate("d.protonated.pdb")
    assert remarks.is_intermediate("d.rosetta.pdb")
    assert not remarks.is_intermediate("d.pdb")
