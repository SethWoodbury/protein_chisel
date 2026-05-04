"""Tests for tools.pdb_restoration — heavy-atom -> Rosetta-protonated PDB.

Covers:
  * REMARK 666 / HETNAM / LINK passthrough
  * HIS tautomer detection from HD1 / HE2 atoms
  * HIS tautomer detection from explicit 5-char labels (HIS_D)
  * KCX detection from carbamate atoms
  * end-to-end restore_pdb_features round trip
  * smoke test against the real PTE_i1 seed (FS269 reference)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from protein_chisel.tools.pdb_restoration import (
    build_his_tautomer_map,
    detect_kcx_residues,
    extract_remark_lines,
    restore_pdb_features,
)


# ---------------------------------------------------------------------- #
# Synthetic-PDB fixtures
# ---------------------------------------------------------------------- #


_HEADER = (
    "REMARK 666 MATCH TEMPLATE B YYE  203 MATCH MOTIF A HIS  132  1  1\n"
    "REMARK 666 MATCH TEMPLATE B YYE  203 MATCH MOTIF A LYS  157  3  1\n"
    "REMARK PDBinfo-LABEL:  132 motif\n"
    "REMARK PDBinfo-LABEL:  157 motif\n"
    "HETNAM     YYE B 203  YYE\n"
)

# Reference: HIS at 60 with HD1 (delta tautomer, becomes HIS_D),
#            HIS at 64 with HE2 (epsilon tautomer, becomes HIE),
#            HIS at 128 with both (HIP),
#            KCX at 157 (LYS + CX, OQ1, OQ2).
_REF_PDB = _HEADER + (
    "ATOM      1  N   HIS A  60       0.000   0.000   0.000  1.00  0.00           N  \n"
    "ATOM      2  CA  HIS A  60       1.000   0.000   0.000  1.00  0.00           C  \n"
    "ATOM      3  ND1 HIS A  60       2.000   0.000   0.000  1.00  0.00           N  \n"
    "ATOM      4  NE2 HIS A  60       3.000   0.000   0.000  1.00  0.00           N  \n"
    "ATOM      5  HD1 HIS A  60       2.000   1.000   0.000  1.00  0.00           H  \n"
    "ATOM      6  N   HIS A  64       0.000   0.000   5.000  1.00  0.00           N  \n"
    "ATOM      7  CA  HIS A  64       1.000   0.000   5.000  1.00  0.00           C  \n"
    "ATOM      8  ND1 HIS A  64       2.000   0.000   5.000  1.00  0.00           N  \n"
    "ATOM      9  NE2 HIS A  64       3.000   0.000   5.000  1.00  0.00           N  \n"
    "ATOM     10  HE2 HIS A  64       3.000   1.000   5.000  1.00  0.00           H  \n"
    "ATOM     11  N   HIS A 128       0.000   0.000  10.000  1.00  0.00           N  \n"
    "ATOM     12  CA  HIS A 128       1.000   0.000  10.000  1.00  0.00           C  \n"
    "ATOM     13  ND1 HIS A 128       2.000   0.000  10.000  1.00  0.00           N  \n"
    "ATOM     14  NE2 HIS A 128       3.000   0.000  10.000  1.00  0.00           N  \n"
    "ATOM     15  HD1 HIS A 128       2.000   1.000  10.000  1.00  0.00           H  \n"
    "ATOM     16  HE2 HIS A 128       3.000   1.000  10.000  1.00  0.00           H  \n"
    "ATOM     17  N   LYS A 157       0.000   0.000  15.000  1.00  0.00           N  \n"
    "ATOM     18  CA  LYS A 157       1.000   0.000  15.000  1.00  0.00           C  \n"
    "ATOM     19  NZ  LYS A 157       2.000   0.000  15.000  1.00  0.00           N  \n"
    "ATOM     20  CX  LYS A 157       3.000   0.000  15.000  1.00  0.00           C  \n"
    "ATOM     21  OQ1 LYS A 157       4.000   0.000  15.000  1.00  0.00           O  \n"
    "ATOM     22  OQ2 LYS A 157       3.000   1.000  15.000  1.00  0.00           O  \n"
    "END\n"
)

# MPNN output: same residues but heavy-atom-only, all "HIS", no REMARKs.
_MPNN_PDB = (
    "ATOM      1  N   HIS A  60       0.000   0.000   0.000  1.00  0.00           N  \n"
    "ATOM      2  CA  HIS A  60       1.000   0.000   0.000  1.00  0.00           C  \n"
    "ATOM      3  ND1 HIS A  60       2.000   0.000   0.000  1.00  0.00           N  \n"
    "ATOM      4  NE2 HIS A  60       3.000   0.000   0.000  1.00  0.00           N  \n"
    "ATOM      6  N   HIS A  64       0.000   0.000   5.000  1.00  0.00           N  \n"
    "ATOM      7  CA  HIS A  64       1.000   0.000   5.000  1.00  0.00           C  \n"
    "ATOM      8  ND1 HIS A  64       2.000   0.000   5.000  1.00  0.00           N  \n"
    "ATOM      9  NE2 HIS A  64       3.000   0.000   5.000  1.00  0.00           N  \n"
    "ATOM     11  N   HIS A 128       0.000   0.000  10.000  1.00  0.00           N  \n"
    "ATOM     12  CA  HIS A 128       1.000   0.000  10.000  1.00  0.00           C  \n"
    "ATOM     13  ND1 HIS A 128       2.000   0.000  10.000  1.00  0.00           N  \n"
    "ATOM     14  NE2 HIS A 128       3.000   0.000  10.000  1.00  0.00           N  \n"
    "ATOM     17  N   LYS A 157       0.000   0.000  15.000  1.00  0.00           N  \n"
    "ATOM     18  CA  LYS A 157       1.000   0.000  15.000  1.00  0.00           C  \n"
    "ATOM     19  NZ  LYS A 157       2.000   0.000  15.000  1.00  0.00           N  \n"
    "END\n"
)


@pytest.fixture
def synth_ref_and_mpnn(tmp_path: Path) -> tuple[Path, Path]:
    ref = tmp_path / "ref.pdb"
    mpnn = tmp_path / "mpnn.pdb"
    ref.write_text(_REF_PDB)
    mpnn.write_text(_MPNN_PDB)
    return ref, mpnn


# ---------------------------------------------------------------------- #
# extract_remark_lines
# ---------------------------------------------------------------------- #


def test_extract_remark_lines_keeps_only_known_prefixes(
    synth_ref_and_mpnn: tuple[Path, Path],
) -> None:
    ref, _ = synth_ref_and_mpnn
    lines = extract_remark_lines(ref)
    assert any(l.startswith("REMARK 666") for l in lines)
    assert any(l.startswith("REMARK PDBinfo-LABEL") for l in lines)
    assert any(l.startswith("HETNAM") for l in lines)
    # No ATOM lines should leak through.
    assert not any(l.startswith("ATOM") for l in lines)
    # Two REMARK 666 + two PDBinfo-LABEL + one HETNAM
    assert len(lines) == 5


def test_extract_remark_lines_stops_at_atom(tmp_path: Path) -> None:
    p = tmp_path / "p.pdb"
    p.write_text(
        "REMARK 666 BEFORE\n"
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "REMARK 666 AFTER\n"
    )
    lines = extract_remark_lines(p)
    assert len(lines) == 1
    assert "BEFORE" in lines[0]


# ---------------------------------------------------------------------- #
# build_his_tautomer_map
# ---------------------------------------------------------------------- #


def test_his_tautomer_map_from_hydrogens(
    synth_ref_and_mpnn: tuple[Path, Path],
) -> None:
    ref, _ = synth_ref_and_mpnn
    m = build_his_tautomer_map(ref)
    assert m[("A", 60)] == "HIS_D"
    assert m[("A", 64)] == "HIE"
    assert m[("A", 128)] == "HIP"


def test_his_tautomer_map_from_explicit_label(tmp_path: Path) -> None:
    # Heavy-atom only with explicit 5-char Rosetta label HIS_D.
    # PDB column layout for 5-char resnames: cols 17-21 resname, col 22 chain.
    # Build via concat to avoid hidden whitespace drift in the test source.
    pdb = tmp_path / "rosetta.pdb"
    def _line(serial: int, atom: str, x: float) -> str:
        return (
            "ATOM  "                       # 1-6
            f"{serial:>5d}"                 # 7-11
            " "                              # 12
            f"{atom:^4s}"                   # 13-16 atom name (centered)
            "HIS_D"                          # 17-21 resname (5-char)
            "A"                              # 22 chain
            "  60"                           # 23-26 resno
            " "                              # 27 icode
            "   "                            # 28-30 blank
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}"  # 31-54 xyz
            "  1.00"                         # 55-60 occ
            "  0.00"                         # 61-66 bfactor
            "          "                    # 67-76 blank
            " N"                             # 77-78 element
            "  \n"                           # 79-80 charge + newline
        )
    line_a = _line(1, "N", 0.0)
    line_b = _line(2, "CA", 1.0)
    assert line_a[16:21] == "HIS_D"
    assert line_a[21:22] == "A"
    assert line_a[22:26] == "  60"
    pdb.write_text(line_a + line_b)
    m = build_his_tautomer_map(pdb)
    assert m[("A", 60)] == "HIS_D"


# ---------------------------------------------------------------------- #
# detect_kcx_residues
# ---------------------------------------------------------------------- #


def test_detect_kcx_from_cap_atoms(synth_ref_and_mpnn: tuple[Path, Path]) -> None:
    ref, _ = synth_ref_and_mpnn
    kcx = detect_kcx_residues(ref)
    assert kcx == {("A", 157): True}


def test_detect_kcx_from_explicit_label(tmp_path: Path) -> None:
    pdb = tmp_path / "p.pdb"
    pdb.write_text(
        "ATOM      1  N   KCX A 157       0.000   0.000   0.000  1.00  0.00           N  \n"
    )
    kcx = detect_kcx_residues(pdb)
    assert kcx[("A", 157)] is True


# ---------------------------------------------------------------------- #
# restore_pdb_features (synthetic round-trip)
# ---------------------------------------------------------------------- #


def test_restore_pdb_features_synth(
    synth_ref_and_mpnn: tuple[Path, Path], tmp_path: Path,
) -> None:
    ref, mpnn = synth_ref_and_mpnn
    out = tmp_path / "restored.pdb"
    stats = restore_pdb_features(
        mpnn_pdb=mpnn, ref_pdb=ref, out_pdb=out,
        catalytic_resnos=[60, 64, 128, 132, 157],
    )

    text = out.read_text()
    # Header passed through.
    assert "REMARK 666 MATCH TEMPLATE" in text
    assert "REMARK PDBinfo-LABEL:  132" in text
    assert "HETNAM     YYE" in text

    # HIS relabels: 60 -> HIS_D, 64 -> HIE, 128 -> HIP.
    his60 = [l for l in text.splitlines() if "A  60" in l and l.startswith("ATOM")]
    his64 = [l for l in text.splitlines() if "A  64" in l and l.startswith("ATOM")]
    his128 = [l for l in text.splitlines() if "A 128" in l and l.startswith("ATOM")]
    assert his60 and all("HIS_D" in l for l in his60)
    assert his64 and all(" HIE " in l for l in his64)
    assert his128 and all(" HIP " in l for l in his128)

    # KCX: residue 157 relabeled and OQ1/OQ2/CX present.
    res157 = [l for l in text.splitlines() if "A 157" in l and l.startswith("ATOM")]
    assert res157 and all(" KCX " in l or "KCX A" in l for l in res157)
    atom_names_157 = {l[12:16].strip() for l in res157}
    assert {"CX", "OQ1", "OQ2"}.issubset(atom_names_157)

    # Hydrogens: HD1 on 60, HE2 on 64, HD1+HE2 on 128.
    h60 = {l[12:16].strip() for l in his60 if l[76:78].strip() == "H"}
    h64 = {l[12:16].strip() for l in his64 if l[76:78].strip() == "H"}
    h128 = {l[12:16].strip() for l in his128 if l[76:78].strip() == "H"}
    assert "HD1" in h60
    assert "HE2" in h64
    assert {"HD1", "HE2"}.issubset(h128)

    assert stats["header_lines_restored"] == 5
    assert stats["his_relabeled"] >= 8  # 4 atoms each on 60/64/128 minus default
    assert stats["kcx_relabeled"] >= 3  # N, CA, NZ at minimum
    assert stats["kcx_atoms_inserted"] == 3  # CX, OQ1, OQ2
    assert stats["hydrogens_copied"] >= 4


def test_restore_skips_kcx_when_residue_mutated(tmp_path: Path) -> None:
    """If MPNN mutated res 157 from LYS to ALA we must NOT relabel it KCX."""
    ref = tmp_path / "ref.pdb"
    ref.write_text(_REF_PDB)
    mpnn = tmp_path / "mpnn.pdb"
    mpnn.write_text(
        "ATOM      1  N   ALA A 157       0.000   0.000  15.000  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA A 157       1.000   0.000  15.000  1.00  0.00           C  \n"
        "ATOM      3  CB  ALA A 157       2.000   0.000  15.000  1.00  0.00           C  \n"
    )
    out = tmp_path / "restored.pdb"
    stats = restore_pdb_features(
        mpnn_pdb=mpnn, ref_pdb=ref, out_pdb=out,
        catalytic_resnos=[157],
    )
    text = out.read_text()
    assert " ALA " in text
    assert " KCX " not in text
    assert stats["kcx_relabeled"] == 0
    assert stats["kcx_atoms_inserted"] == 0


# ---------------------------------------------------------------------- #
# Real seed: PTE_i1 FS269 reference
# ---------------------------------------------------------------------- #


_PTE_I1_REF = Path(
    "/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/"
    "ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb"
)


@pytest.mark.skipif(
    not _PTE_I1_REF.is_file(),
    reason="PTE_i1 FS269 reference PDB not available on this host",
)
def test_pte_i1_seed_tautomers_and_header() -> None:
    header = extract_remark_lines(_PTE_I1_REF)
    # Must carry six REMARK 666 motif lines + HETNAM YYE.
    assert sum(l.startswith("REMARK 666") for l in header) == 6
    assert any(l.startswith("HETNAM     YYE") for l in header)

    his_map = build_his_tautomer_map(_PTE_I1_REF)
    # All four catalytic HIS in PTE_i1 carry HD1 (delta tautomer).
    for resno in (60, 64, 128, 132):
        assert his_map[("A", resno)] == "HIS_D", (
            f"expected HIS_D at A{resno}, got {his_map.get(('A', resno))}"
        )
    # Residue 157 in this seed is plain LYS (not KCX).
    kcx = detect_kcx_residues(_PTE_I1_REF)
    assert ("A", 157) not in kcx


@pytest.mark.skipif(
    not _PTE_I1_REF.is_file(),
    reason="PTE_i1 FS269 reference PDB not available on this host",
)
def test_pte_i1_round_trip_self(tmp_path: Path) -> None:
    """Running the seed through restore (with itself as MPNN input) should
    produce a PDB that still contains the six REMARK 666 lines and re-labels
    every catalytic HIS to HIS_D."""
    out = tmp_path / "restored.pdb"
    # Strip hydrogens + REMARKs from the seed to simulate MPNN heavy-atom output.
    mpnn = tmp_path / "mpnn.pdb"
    with open(_PTE_I1_REF) as fin, open(mpnn, "w") as fout:
        for line in fin:
            if line.startswith("REMARK") or line.startswith("HETNAM") or line.startswith("LINK"):
                continue
            if line.startswith("ATOM") and line[76:78].strip() == "H":
                continue
            # MPNN always writes HIS, never the 5-char label.
            if line.startswith("ATOM") and line[16:21].strip() in {"HIS_D", "HIS_E"}:
                line = line[:16] + " HIS " + line[21:]
            fout.write(line)

    stats = restore_pdb_features(
        mpnn_pdb=mpnn, ref_pdb=_PTE_I1_REF, out_pdb=out,
        catalytic_resnos=[60, 64, 128, 131, 132, 157],
    )
    text = out.read_text()
    assert sum(l.startswith("REMARK 666") for l in text.splitlines()) == 6
    for resno in (60, 64, 128, 132):
        # Every atom of this residue should now carry HIS_D.
        for line in text.splitlines():
            if line.startswith("ATOM") and f"A{resno:>4d}" in line[20:26]:
                assert "HIS_D" in line, (
                    f"resno {resno} not relabeled to HIS_D: {line!r}"
                )
    assert stats["header_lines_restored"] >= 6
    assert stats["his_relabeled"] >= 4
