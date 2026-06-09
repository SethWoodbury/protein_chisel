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
    assert out["status"] == "ok" and out["renamed"] == 3 and out["width"] == 2
    names = sorted(p.name for p in root.glob("*.pdb"))
    assert names == ["input.pdb", "pte_FS014_chisel_00.pdb",
                     "pte_FS014_chisel_01.pdb", "pte_FS014_chisel_02.pdb"]
    # rank 0 (best) == the design that was ranked first in the TSV (lmpnn 28)
    rows = _ids_in_tsv(root)
    design = [r for r in rows if r["is_input"] != "True"]
    assert design[0]["id"] == "pte_FS014_chisel_00"
    assert Path(design[0]["pdb_path"]).name == "pte_FS014_chisel_00.pdb"


def test_input_reference_untouched(tmp_path):
    root = _make_run(tmp_path, ["x_chisel_3", "x_chisel_9"],
                     ["x_chisel_9", "x_chisel_3"])
    finalize_design_names(root)
    assert (root / "input.pdb").exists()          # input copy not renamed
    rows = _ids_in_tsv(root)
    inp = [r for r in rows if r["is_input"] == "True"]
    assert len(inp) == 1 and inp[0]["id"] == "input"


def test_collision_safe_swap(tmp_path):
    # Source names collide with target ranks: 'a_chisel_01' is ranked best (->00),
    # 'a_chisel_00' is ranked worst (->01). In-place rename would clash; fresh-dir
    # swap must not lose/clobber either.
    root = _make_run(tmp_path, ["a_chisel_00", "a_chisel_01"],
                     ["a_chisel_01", "a_chisel_00"])
    # tag bodies so we can verify content didn't get clobbered/swapped wrongly
    (root / "a_chisel_01.pdb").write_text(_pdb_text() + "REMARK TAG best\n")
    (root / "a_chisel_00.pdb").write_text(_pdb_text() + "REMARK TAG worst\n")
    finalize_design_names(root)
    names = sorted(p.name for p in root.glob("*.pdb"))
    assert names == ["a_chisel_00.pdb", "a_chisel_01.pdb", "input.pdb"]
    # rank0 file (a_chisel_00) must carry the 'best' tag (was source a_chisel_01)
    assert "REMARK TAG best" in (root / "a_chisel_00.pdb").read_text()
    assert "REMARK TAG worst" in (root / "a_chisel_01.pdb").read_text()


def test_design_path_collapsed(tmp_path):
    root = _make_run(tmp_path, ["x_chisel_5"], ["x_chisel_5"])
    finalize_design_names(root)
    txt = (root / "x_chisel_00.pdb").read_text()
    assert "DESIGN_PATH iterative_design" not in txt
    assert "DESIGN_PATH protonate_topk" not in txt
    import os
    dp = [l for l in txt.splitlines() if l.startswith("REMARK DESIGN_PATH chisel_iterative_design output")]
    assert len(dp) == 1
    assert dp[0].endswith(os.path.abspath(str(root / "x_chisel_00.pdb")))
    # upstream chain + other REMARKs preserved
    for keep in ("DESIGN_PATH rfd3 input", "DESIGN_PATH predesign_cart_relax",
                 "DESIGN_PATH chisel_ligandmpnn", "REMARK 665", "REMARK 666",
                 "REMARK 667", "REMARK 668", "REMARK QCB"):
        assert keep in txt


def test_keep_intermediate_flag(tmp_path):
    root = _make_run(tmp_path, ["x_chisel_5"], ["x_chisel_5"])
    finalize_design_names(root, keep_intermediate=True)
    txt = (root / "x_chisel_00.pdb").read_text()
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


def test_width_three_digits_at_100(tmp_path):
    names = [f"y_chisel_{i}" for i in range(100)]  # 100 designs -> count-width 3
    root = _make_run(tmp_path, names, names)
    out = finalize_design_names(root)
    assert out["width"] == 3
    assert (root / "y_chisel_000.pdb").exists() and (root / "y_chisel_099.pdb").exists()


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
