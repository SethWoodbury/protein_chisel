"""Tests for the final rank-order rename + DESIGN_PATH collapse (pure-Python)."""
from __future__ import annotations

from pathlib import Path

from protein_chisel.tools.finalize_names import finalize_design_names

_ATOM = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
_UPSTREAM = [
    ("rfd3", "input", "/home/x/theozyme.pdb"),
    ("rfd3", "output", "/scratch/rfd3/out.cif.gz"),
    ("predesign_cart_relax", "output", "/scratch/predesign/idealized.pdb"),
    ("chisel_ligandmpnn", "output", "/scratch/mpnn/design.pdb"),
]
_INTERMEDIATE = [
    ("iterative_design", "output", "/scratch/run/final_topk/topk_pdbs/x_lmpnn_9.pdb"),
    ("protonate_topk", "output", "/scratch/run/final_topk/topk_pdbs_protonated/x_lmpnn_9.protonated.pdb"),
]


def _pdb_text(extra_design_paths=()):
    dps = list(_UPSTREAM) + list(_INTERMEDIATE) + list(extra_design_paths)
    hdr = (
        "REMARK 665 legend\n"
        "REMARK 666 MATCH TEMPLATE B LIG  200 MATCH MOTIF A HIS   16  1  1\n"
        "REMARK 667 legend\n"
        "REMARK 668   1   A HIS     16   HID -   HD1                     HIS\n"
        "REMARK QCB TOTAL_CHARGE +1\n"
    )
    hdr += "".join(f"REMARK DESIGN_PATH {s} {k} {p}\n" for (s, k, p) in dps)
    return hdr + _ATOM + "END\n"


def _make_run(tmp_path, design_names, ranked_ids, *, with_input=True, run_meta=True):
    """Create a published run dir: design PDBs + a metrics TSV (RANK order)."""
    root = tmp_path / "run"
    root.mkdir()
    for nm in design_names:
        (root / f"{nm}.pdb").write_text(_pdb_text())
    rows = ["id\tpdb_path\tis_input\tscore"]
    for rid in ranked_ids:  # design rows, in rank order (best first)
        rows.append(f"{rid}\t{root / (rid + '.pdb')}\tFalse\t0.9")
    if with_input:
        (root / "input.pdb").write_text(_pdb_text())
        rows.append(f"input\t{root / 'input.pdb'}\tTrue\t")
    body = "\n".join(rows) + "\n"
    meta = '# RUN_META: {"tool":"chisel","cycles":3}\n' if run_meta else ""
    (root / "chiseled_design_metrics.tsv").write_text(meta + body)
    return root


def _ids_in_tsv(root):
    import csv
    lines = (root / "chiseled_design_metrics.tsv").read_text().splitlines()
    body = [l for l in lines if not l.startswith("# RUN_META:")]
    r = list(csv.DictReader(body, delimiter="\t"))
    return r


def test_rank_order_and_keep_stem(tmp_path):
    # 3 designs with varied lmpnn indices; TSV rank: 28 best, then 154, then 7.
    root = _make_run(tmp_path,
                     ["pte_FS014_chisel_154", "pte_FS014_chisel_28", "pte_FS014_chisel_7"],
                     ["pte_FS014_chisel_28", "pte_FS014_chisel_154", "pte_FS014_chisel_7"])
    out = finalize_design_names(root)
    # 3 designs -> indices 0..2 -> single digit (max index 2, never chisel_10).
    assert out["status"] == "ok" and out["renamed"] == 3 and out["width"] == 1
    names = sorted(p.name for p in root.glob("*.pdb"))
    assert names == ["input.pdb", "pte_FS014_chisel_0.pdb",
                     "pte_FS014_chisel_1.pdb", "pte_FS014_chisel_2.pdb"]
    # rank 0 (best) == the design that was ranked first in the TSV (lmpnn 28)
    rows = _ids_in_tsv(root)
    design = [r for r in rows if r["is_input"] != "True"]
    assert design[0]["id"] == "pte_FS014_chisel_0"
    assert Path(design[0]["pdb_path"]).name == "pte_FS014_chisel_0.pdb"


def test_input_reference_untouched(tmp_path):
    root = _make_run(tmp_path, ["x_chisel_3", "x_chisel_9"],
                     ["x_chisel_9", "x_chisel_3"])
    finalize_design_names(root)
    assert (root / "input.pdb").exists()          # input copy not renamed
    rows = _ids_in_tsv(root)
    inp = [r for r in rows if r["is_input"] == "True"]
    assert len(inp) == 1 and inp[0]["id"] == "input"


def test_collision_safe_swap(tmp_path):
    # 2 designs -> width 1, targets a_chisel_0 / a_chisel_1. Source names collide
    # with the ranks: 'a_chisel_1' is ranked best (->0), 'a_chisel_0' is ranked
    # worst (->1). In-place rename would clash; fresh-dir swap must not lose/clobber.
    root = _make_run(tmp_path, ["a_chisel_0", "a_chisel_1"],
                     ["a_chisel_1", "a_chisel_0"])
    # tag bodies so we can verify content didn't get clobbered/swapped wrongly
    (root / "a_chisel_1.pdb").write_text(_pdb_text() + "REMARK TAG best\n")
    (root / "a_chisel_0.pdb").write_text(_pdb_text() + "REMARK TAG worst\n")
    finalize_design_names(root)
    names = sorted(p.name for p in root.glob("*.pdb"))
    assert names == ["a_chisel_0.pdb", "a_chisel_1.pdb", "input.pdb"]
    # rank0 file (a_chisel_0) must carry the 'best' tag (was source a_chisel_1)
    assert "REMARK TAG best" in (root / "a_chisel_0.pdb").read_text()
    assert "REMARK TAG worst" in (root / "a_chisel_1.pdb").read_text()


def test_design_path_collapsed(tmp_path):
    root = _make_run(tmp_path, ["x_chisel_5"], ["x_chisel_5"])
    finalize_design_names(root)
    txt = (root / "x_chisel_0.pdb").read_text()   # 1 design -> width 1
    assert "DESIGN_PATH iterative_design" not in txt
    assert "DESIGN_PATH protonate_topk" not in txt
    import os
    dp = [l for l in txt.splitlines() if l.startswith("REMARK DESIGN_PATH chisel_iterative_design output")]
    assert len(dp) == 1
    assert dp[0].endswith(os.path.abspath(str(root / "x_chisel_0.pdb")))
    # upstream chain + other REMARKs preserved
    for keep in ("DESIGN_PATH rfd3 input", "DESIGN_PATH predesign_cart_relax",
                 "DESIGN_PATH chisel_ligandmpnn", "REMARK 665", "REMARK 666",
                 "REMARK 667", "REMARK 668", "REMARK QCB"):
        assert keep in txt


def test_keep_intermediate_flag(tmp_path):
    root = _make_run(tmp_path, ["x_chisel_5"], ["x_chisel_5"])
    finalize_design_names(root, keep_intermediate=True)
    txt = (root / "x_chisel_0.pdb").read_text()   # 1 design -> width 1
    assert "DESIGN_PATH iterative_design" in txt
    assert "DESIGN_PATH protonate_topk" in txt
    assert "DESIGN_PATH chisel_iterative_design output" in txt


def test_run_meta_preserved(tmp_path):
    root = _make_run(tmp_path, ["x_chisel_5"], ["x_chisel_5"])
    finalize_design_names(root)
    first = (root / "chiseled_design_metrics.tsv").read_text().splitlines()[0]
    assert first == '# RUN_META: {"tool":"chisel","cycles":3}'


def test_idempotent(tmp_path):
    root = _make_run(tmp_path, ["x_chisel_5", "x_chisel_9"], ["x_chisel_9", "x_chisel_5"])
    finalize_design_names(root)
    snap = {p.name: p.read_text() for p in sorted(root.glob("*.pdb"))}
    tsv1 = (root / "chiseled_design_metrics.tsv").read_text()
    finalize_design_names(root)                    # second run
    snap2 = {p.name: p.read_text() for p in sorted(root.glob("*.pdb"))}
    assert snap == snap2
    assert (root / "chiseled_design_metrics.tsv").read_text() == tsv1


def test_width_two_digits(tmp_path):
    names = [f"y_chisel_{i}" for i in range(12)]   # 12 designs -> width 2
    root = _make_run(tmp_path, names, names)
    out = finalize_design_names(root)
    assert out["width"] == 2
    assert (root / "y_chisel_00.pdb").exists() and (root / "y_chisel_11.pdb").exists()


def test_width_one_digit_at_ten(tmp_path):
    # 10 designs -> indices 0..9; max index is 9 so a single digit suffices. We
    # never emit chisel_10, so the count crossing 10 must NOT bump the padding.
    names = [f"z_chisel_{i}" for i in range(10)]
    root = _make_run(tmp_path, names, names)
    out = finalize_design_names(root)
    assert out["width"] == 1
    assert (root / "z_chisel_0.pdb").exists() and (root / "z_chisel_9.pdb").exists()
    assert not (root / "z_chisel_00.pdb").exists()


def test_width_bumps_to_two_at_eleven(tmp_path):
    # 11 designs -> indices 0..10; chisel_10 needs 2 digits, so all become 2.
    names = [f"z_chisel_{i}" for i in range(11)]
    root = _make_run(tmp_path, names, names)
    out = finalize_design_names(root)
    assert out["width"] == 2
    assert (root / "z_chisel_00.pdb").exists() and (root / "z_chisel_10.pdb").exists()


def test_width_no_bump_at_100(tmp_path):
    # 100 designs -> indices 0..99, all 2-digit. Count is 3-digit but the largest
    # index is only 99, so padding must stay at 2 (no over-pad to chisel_099).
    names = [f"y_chisel_{i}" for i in range(100)]
    root = _make_run(tmp_path, names, names)
    out = finalize_design_names(root)
    assert out["width"] == 2
    assert (root / "y_chisel_00.pdb").exists() and (root / "y_chisel_99.pdb").exists()
    assert not (root / "y_chisel_000.pdb").exists()


def test_width_bumps_to_three_at_101(tmp_path):
    # 101 designs -> indices 0..100; chisel_100 needs 3 digits, so all become 3.
    names = [f"y_chisel_{i}" for i in range(101)]
    root = _make_run(tmp_path, names, names)
    out = finalize_design_names(root)
    assert out["width"] == 3
    assert (root / "y_chisel_000.pdb").exists() and (root / "y_chisel_100.pdb").exists()


def test_partial_swap_failure_no_data_loss(tmp_path, monkeypatch):
    # If os.replace fails mid-swap, finalize must raise but LOSE NO design content
    # (staged copies retained in the temp dir; originals not pre-deleted).
    import pytest
    import protein_chisel.tools.finalize_names as fn
    root = _make_run(tmp_path, ["x_chisel_3", "x_chisel_9", "x_chisel_5"],
                     ["x_chisel_9", "x_chisel_3", "x_chisel_5"])
    for nm, tag in [("x_chisel_9", "T9"), ("x_chisel_3", "T3"), ("x_chisel_5", "T5")]:
        (root / f"{nm}.pdb").write_text(_pdb_text() + f"REMARK TAG {tag}\n")

    import os as _os
    real_replace = _os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:           # first os.replace = swap-1; second = swap-2
            raise OSError("injected mid-swap failure")
        return real_replace(src, dst)

    monkeypatch.setattr(fn.os, "replace", flaky)
    with pytest.raises(OSError):
        fn.finalize_design_names(root)
    # Every design's content must still exist somewhere under root (designs dir or
    # the retained .finalize_tmp_* recovery dir).
    blob = "".join(p.read_text() for p in root.rglob("*.pdb"))
    for tag in ("T9", "T3", "T5"):
        assert f"REMARK TAG {tag}" in blob, f"design content {tag} was LOST"


def test_no_designs_noop(tmp_path):
    root = tmp_path / "run"; root.mkdir()
    (root / "input.pdb").write_text(_pdb_text())
    (root / "chiseled_design_metrics.tsv").write_text(
        "id\tpdb_path\tis_input\tinput\t" + str(root / "input.pdb") + "\tTrue\n")
    # malformed minimal: just ensure no crash / no-op when no design rows
    out = finalize_design_names(root)
    assert out["renamed"] == 0


def test_missing_tsv_noop(tmp_path):
    root = tmp_path / "empty"; root.mkdir()
    out = finalize_design_names(root)
    assert out["status"] == "no_tsv" and out["renamed"] == 0


def test_custom_design_token(tmp_path):
    # WS-H: --design_token replaces the 'chisel' token in the shipped name.
    root = _make_run(tmp_path, ["seed_x_chisel_0", "seed_x_chisel_1"],
                     ["seed_x_chisel_1", "seed_x_chisel_0"])  # rank: _1 best -> 0
    out = finalize_design_names(root, design_token="chiseli2")
    assert out["status"] == "ok" and out["renamed"] == 2
    names = sorted(p.name for p in root.glob("*.pdb"))
    assert names == ["input.pdb", "seed_x_chiseli2_0.pdb", "seed_x_chiseli2_1.pdb"]
    rows = _ids_in_tsv(root)
    design = [r for r in rows if r["is_input"] != "True"]
    assert design[0]["id"] == "seed_x_chiseli2_0"
    # Idempotent re-run with the SAME token must not double-append.
    finalize_design_names(root, design_token="chiseli2")
    assert sorted(p.name for p in root.glob("*.pdb")) == names


def test_custom_token_preserves_input_chisel_in_stem(tmp_path):
    # Real i2 case: the input stem itself contains a prior '_chisel_62'. Only the
    # TRAILING design index is replaced; the input's own _chisel_62 is preserved.
    root = _make_run(tmp_path, ["pte_chisel_62_af3i2_chisel_5"],
                     ["pte_chisel_62_af3i2_chisel_5"])
    finalize_design_names(root, design_token="chiseli2")
    names = sorted(p.name for p in root.glob("*.pdb"))
    assert names == ["input.pdb", "pte_chisel_62_af3i2_chiseli2_0.pdb"]


def test_default_token_is_chisel(tmp_path):
    # Byte-identity: omitting design_token == the legacy 'chisel' behavior.
    root = _make_run(tmp_path, ["q_chisel_3"], ["q_chisel_3"])
    finalize_design_names(root)  # no design_token
    assert (root / "q_chisel_0.pdb").exists()


def test_invalid_design_token_rejected(tmp_path):
    import pytest
    root = _make_run(tmp_path, ["w_chisel_1"], ["w_chisel_1"])
    for bad in ("chisel_i2", "chisel.i2", "chisel i2", ""):
        with pytest.raises(ValueError):
            finalize_design_names(root, design_token=bad)
