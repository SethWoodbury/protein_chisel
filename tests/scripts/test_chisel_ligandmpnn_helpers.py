"""Tests for pure-python helpers in scripts/chisel_ligandMPNN.py."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import chisel_ligandMPNN as w  # noqa: E402


def test_parse_probability_decimal():
    assert w.parse_probability(0.8) == 0.8
    assert w.parse_probability(0.0) == 0.0
    assert w.parse_probability(1.0) == 1.0


def test_parse_probability_percentage(caplog):
    assert w.parse_probability(80) == pytest.approx(0.8)
    assert w.parse_probability(100) == pytest.approx(1.0)


def test_parse_probability_errors():
    with pytest.raises(SystemExit):
        w.parse_probability(150)
    with pytest.raises(SystemExit):
        w.parse_probability(-0.1)


def test_roll_conserved_bounds():
    labels = ["A10", "A20", "A30"]
    assert w._roll_conserved(labels, 1.0, random.Random(0)) == set(labels)
    assert w._roll_conserved(labels, 0.0, random.Random(0)) == set()


def test_roll_conserved_seed_reproducible():
    labels = ["A10", "A20", "A30", "A40", "A50"]
    a = w._roll_conserved(labels, 0.5, random.Random("seed:1"))
    b = w._roll_conserved(labels, 0.5, random.Random("seed:1"))
    assert a == b                      # same seed -> identical
    assert a.issubset(set(labels))


def test_fixed_residues_json(tmp_path):
    out = tmp_path / "fx.json"
    w._fixed_residues_json("/abs/x.pdb", ["A54", "A11", "A54"], out)
    data = json.loads(out.read_text())
    assert list(data.keys()) == [str(Path("/abs/x.pdb").resolve())]
    # de-duped and sorted by (chain, resno)
    assert data[str(Path("/abs/x.pdb").resolve())] == ["A11", "A54"]


def test_label_sort_key():
    assert sorted(["A100", "A9", "A11"], key=w._label_sort_key) == ["A9", "A11", "A100"]


# --------------------------------------------------------------------------
# sequence de-dup + diversity
# --------------------------------------------------------------------------
def _atom(serial, name, resname, chain, resno, x, y, z, record="ATOM", element="C"):
    """One ATOM/HETATM line in the exact PDB columns io.pdb.parse_atom_record reads."""
    rec = "ATOM  " if record == "ATOM" else "HETATM"
    return (f"{rec:<6}{serial:>5} {name:<4}"[:16].ljust(16)
            + f" {resname:>3} {chain:1}{resno:>4}    "
            + f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}"
            + "          " + f"{element:>2}\n")


def _write_pdb(path, residues, ligand=True):
    """residues = list of (resno, resname, x); each gets a CA at (x,0,0). Ligand
    'LIG' (chain B) at the origin when ``ligand``."""
    lines, serial = [], 1
    for resno, resname, x in residues:
        lines.append(_atom(serial, "CA", resname, "A", resno, x, 0.0, 0.0, element="C"))
        serial += 1
    if ligand:
        lines.append(_atom(serial, "C1", "LIG", "B", 900, 0.0, 0.0, 0.0,
                           record="HETATM", element="C"))
    path.write_text("".join(lines) + "END\n")
    return str(path)


def test_read_protein_chain_canonicalizes_variants(tmp_path):
    # protonation / PTM variants collapse to the canonical 1-letter identity
    p = _write_pdb(tmp_path / "v.pdb",
                   [(1, "HIE", 0.0), (2, "KCX", 3.0), (3, "CYX", 6.0), (4, "ALA", 9.0)],
                   ligand=False)
    _ch, seq, keys, _heavy = w._read_protein_chain(p)
    assert seq == "HKCA" and len(keys) == 4


def test_pocket_mask_within_cutoff(tmp_path):
    # residues at x = 0, 3 are within 8 A of the ligand at origin; 20, 23 are not
    p = _write_pdb(tmp_path / "pk.pdb",
                   [(1, "ALA", 0.0), (2, "GLY", 3.0), (3, "SER", 20.0), (4, "THR", 23.0)])
    mask, n = w._pocket_mask(p, None, 8.0)
    assert mask == [True, True, False, False] and n == 2


def test_dedup_keeps_input_and_removes_newer_mpnn(tmp_path):
    import os
    coords = lambda names: [(i + 1, nm, x) for i, (nm, x) in
                            enumerate(zip(names, (0.0, 3.0, 20.0, 23.0)))]
    inp = tmp_path / "input.pdb"
    _write_pdb(inp, coords(["ALA", "GLY", "SER", "THR"]))            # AGST
    _write_pdb(tmp_path / "d1.pdb", coords(["ALA", "GLY", "SER", "THR"]))  # == input
    d2 = _write_pdb(tmp_path / "d2.pdb", coords(["ALA", "VAL", "SER", "LEU"]))  # AVSL
    d3 = _write_pdb(tmp_path / "d3.pdb", coords(["ALA", "VAL", "SER", "LEU"]))  # == d2, newer
    _write_pdb(tmp_path / "d4.pdb", coords(["ASP", "GLY", "THR", "TRP"]))  # DGTW
    os.utime(d2, (1000, 1000)); os.utime(d3, (2000, 2000))  # d3 is the newer dup

    info = w.dedup_and_diversity(str(tmp_path), "input.pdb", str(inp), cutoff=8.0, do_dedup=True)

    assert inp.exists()                                   # input copy never removed
    assert not (tmp_path / "d1.pdb").exists()             # design == input -> removed
    assert not (tmp_path / "d3.pdb").exists()             # newer MPNN dup -> removed
    assert (tmp_path / "d2.pdb").exists()                 # older of the pair kept
    assert (tmp_path / "d4.pdb").exists()
    assert set(info["removed"]) == {"d1.pdb", "d3.pdb"}
    assert info["n_unique"] == 2                          # AVSL, DGTW
    assert info["len_full"] == 4 and info["mean_full"] == 4.0   # all 4 positions differ
    assert info["n_pocket"] == 2 and info["mean_pocket"] == 2.0  # both pocket positions differ


def test_dedup_off_keeps_files_but_still_reports(tmp_path):
    coords = lambda names: [(i + 1, nm, x) for i, (nm, x) in
                            enumerate(zip(names, (0.0, 3.0, 20.0, 23.0)))]
    _write_pdb(tmp_path / "a.pdb", coords(["ALA", "VAL", "SER", "LEU"]))
    _write_pdb(tmp_path / "b.pdb", coords(["ALA", "VAL", "SER", "LEU"]))   # dup sequence
    _write_pdb(tmp_path / "c.pdb", coords(["ASP", "GLY", "THR", "TRP"]))
    info = w.dedup_and_diversity(str(tmp_path), None, str(tmp_path / "a.pdb"),
                                 cutoff=8.0, do_dedup=False)
    assert info["removed"] == []                          # nothing deleted
    assert (tmp_path / "b.pdb").exists()
    assert info["n_unique"] == 2                          # diversity over UNIQUE seqs
