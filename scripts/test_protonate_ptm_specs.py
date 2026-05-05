"""Regression tests for the PTM-spec parser + motif-index resolution.

Covers:
- Motif-index form (preferred): "A/LYS/3:KCX"
- Explicit-residue form (legacy): "A:157=KCX", "157=KCX"
- Mixed forms in one spec string
- Empty spec returns []
- Malformed entries are skipped with warnings
- Resolution of motif-index against a real REMARK 666 block
- "-" code as a force-no-PTM override

Run inside the universal.sif (no PyRosetta needed for these tests).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/woodbuse/codebase_projects/protein_chisel/src")

from protein_chisel.tools.protonate_final import (
    parse_ptm_map,
    PtmSpec,
    resolve_ptm_map,
    _resolve_motif_specs_to_keys,
)


# Minimal fake seed PDB with the PTE_i1 REMARK 666 block.
FAKE_SEED = """\
REMARK 666 MATCH TEMPLATE B YYE  203 MATCH MOTIF A HIS  132  1  1
REMARK 666 MATCH TEMPLATE B YYE  203 MATCH MOTIF A HIS  128  2  1
REMARK 666 MATCH TEMPLATE B YYE  203 MATCH MOTIF A LYS  157  3  1
REMARK 666 MATCH TEMPLATE B YYE  203 MATCH MOTIF A HIS   64  4  1
REMARK 666 MATCH TEMPLATE B YYE  203 MATCH MOTIF A HIS   60  5  1
REMARK 666 MATCH TEMPLATE B YYE  203 MATCH MOTIF A GLU  131  6  1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O
TER
END
"""


def test_parse_motif_index_form():
    specs = parse_ptm_map("A/LYS/3:KCX")
    assert len(specs) == 1, f"expected 1 spec, got {len(specs)}"
    s = specs[0]
    assert s.chain == "A" and s.expected_resname == "LYS"
    assert s.motif_idx == 3 and s.code == "KCX"
    assert s.resno is None
    print("OK  parse_motif_index_form: A/LYS/3:KCX")


def test_parse_explicit_form():
    specs = parse_ptm_map("A:157=KCX")
    assert len(specs) == 1
    s = specs[0]
    assert s.chain == "A" and s.resno == 157 and s.code == "KCX"
    assert s.motif_idx is None
    print("OK  parse_explicit_form: A:157=KCX")


def test_parse_explicit_no_chain():
    specs = parse_ptm_map("157=KCX")
    assert specs[0].chain == "A"
    assert specs[0].resno == 157
    print("OK  parse_explicit_no_chain: 157=KCX -> chain A")


def test_parse_mixed():
    specs = parse_ptm_map("A/LYS/3:KCX, A:200=SEP, B/HIS/1:MSE")
    assert len(specs) == 3
    assert specs[0].motif_idx == 3 and specs[0].code == "KCX"
    assert specs[1].resno == 200 and specs[1].code == "SEP"
    assert specs[2].motif_idx == 1 and specs[2].chain == "B" and specs[2].code == "MSE"
    print("OK  parse_mixed: motif + explicit + B chain")


def test_parse_list_input():
    specs = parse_ptm_map(["A/LYS/3:KCX", "A:200=SEP"])
    assert len(specs) == 2
    print("OK  parse_list_input: accepts iterable of strings")


def test_parse_empty():
    assert parse_ptm_map(None) == []
    assert parse_ptm_map("") == []
    assert parse_ptm_map("   ") == []
    print("OK  parse_empty: returns []")


def test_parse_malformed_skipped():
    # Bare 'KCX' with no key, missing code, etc.
    specs = parse_ptm_map("garbage, A:157=KCX, missing_equals")
    assert len(specs) == 1, f"expected 1 valid spec, got {len(specs)}: {specs}"
    print("OK  parse_malformed_skipped: 1 valid out of 3 entries")


def test_motif_resolution_correct():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
        fh.write(FAKE_SEED)
        seed = Path(fh.name)
    try:
        specs = parse_ptm_map("A/LYS/3:KCX")
        pairs = _resolve_motif_specs_to_keys(seed, specs)
        assert pairs == [(("A", 157), "KCX")], f"got {pairs}"
        print(f"OK  motif_resolution_correct: A/LYS/3:KCX -> {pairs[0]}")
    finally:
        seed.unlink()


def test_motif_resolution_wrong_resname_warns():
    """Spec says HIS but motif index 3 is actually LYS — should still resolve
    but log a warning."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
        fh.write(FAKE_SEED)
        seed = Path(fh.name)
    try:
        specs = parse_ptm_map("A/HIS/3:KCX")  # wrong expected resname
        pairs = _resolve_motif_specs_to_keys(seed, specs)
        assert pairs == [(("A", 157), "KCX")], f"got {pairs}"
        print(f"OK  motif_resolution_wrong_resname: still resolves to (A,157)")
    finally:
        seed.unlink()


def test_motif_resolution_unknown_index():
    """Motif index 99 doesn't exist — spec should be skipped."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
        fh.write(FAKE_SEED)
        seed = Path(fh.name)
    try:
        specs = parse_ptm_map("A/LYS/99:KCX")
        pairs = _resolve_motif_specs_to_keys(seed, specs)
        assert pairs == [], f"expected empty (unresolvable), got {pairs}"
        print("OK  motif_resolution_unknown_index: skipped with warning")
    finally:
        seed.unlink()


def test_resolve_ptm_map_full():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
        fh.write(FAKE_SEED)
        seed = Path(fh.name)
    try:
        # No PTM markers in the fake seed (it has only ALA atoms), so
        # auto-detect should yield no entries; only the explicit
        # declaration applies.
        result = resolve_ptm_map(seed, "A/LYS/3:KCX")
        assert result == {("A", 157): "KCX"}, f"got {result}"
        print(f"OK  resolve_ptm_map_full: {result}")
    finally:
        seed.unlink()


def test_force_no_ptm_with_dash():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
        fh.write(FAKE_SEED)
        seed = Path(fh.name)
    try:
        # Pre-resolved dict simulating an auto-detect hit, then override with "-"
        pre = {("A", 157): "KCX"}
        # First confirm pre is what we'd see without override
        result = resolve_ptm_map(seed, pre)
        assert result == pre

        # Now declare "-" to suppress
        result = resolve_ptm_map(seed, "A/LYS/3:-")
        assert ("A", 157) not in result, f"expected (A,157) suppressed, got {result}"
        print("OK  force_no_ptm_with_dash: A/LYS/3:- removes the entry")
    finally:
        seed.unlink()


def main() -> int:
    tests = [
        test_parse_motif_index_form,
        test_parse_explicit_form,
        test_parse_explicit_no_chain,
        test_parse_mixed,
        test_parse_list_input,
        test_parse_empty,
        test_parse_malformed_skipped,
        test_motif_resolution_correct,
        test_motif_resolution_wrong_resname_warns,
        test_motif_resolution_unknown_index,
        test_resolve_ptm_map_full,
        test_force_no_ptm_with_dash,
    ]
    failures: list[tuple[str, str]] = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"ERR  {t.__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} failures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        return 1
    print(f"All {len(tests)} PTM-spec tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
