"""Tests for pure-python helpers in scripts/iterative_design.py.

We import the script as a module (its sibling-script imports are
defensive sys.path inserts so this works on host pytest as long as
PYTHONPATH includes the repo's src/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import iterative_design as v2   # noqa: E402


# ----------------------------------------------------------------------
# compute_catalytic_neighbor_omit_dict
# ----------------------------------------------------------------------


def _pt(rows):
    """Build a tiny PositionTable.df from rows of (resno, chain, name1,
    is_protein)."""
    df = pd.DataFrame(rows, columns=["resno", "chain", "name1", "is_protein"])
    return df


def test_omit_dict_catalytic_K_forbids_KR_at_neighbors():
    """Catalytic K at resno 50 -> forbid K/R at 49 and 51 (assuming
    they're protein and not also catalytic)."""
    df = _pt([
        (48, "A", "L", True),
        (49, "A", "A", True),
        (50, "A", "K", True),    # catalytic K
        (51, "A", "A", True),
        (52, "A", "L", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[50],
        chain="A",
    )
    assert out == {"A49": "KR", "A51": "KR"}


def test_omit_dict_catalytic_R_forbids_KR_at_neighbors():
    df = _pt([
        (10, "A", "G", True),
        (11, "A", "R", True),    # catalytic R
        (12, "A", "S", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[11],
        chain="A",
    )
    assert out == {"A10": "KR", "A12": "KR"}


def test_omit_dict_skips_non_K_R_catalytic_residues():
    """HIS/GLU catalytic residues don't trigger neighbor omission —
    they don't form OmpT motifs."""
    df = _pt([
        (60, "A", "L", True),
        (61, "A", "H", True),    # catalytic HIS
        (62, "A", "L", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[61],
        chain="A",
    )
    assert out == {}


def test_omit_dict_skips_neighbor_that_is_also_catalytic():
    """If catalytic residues are adjacent (e.g. K-K active site), don't
    over-constrain the second one."""
    df = _pt([
        (50, "A", "K", True),    # catalytic K
        (51, "A", "K", True),    # also catalytic K
        (52, "A", "L", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[50, 51],
        chain="A",
    )
    # 50's right neighbor (51) is in fixed_set -> skip
    # 51's left neighbor (50) is in fixed_set -> skip
    # 51's right neighbor (52) is NOT fixed -> forbid
    assert out == {"A52": "KR"}


def test_omit_dict_skips_off_chain_neighbors():
    """N-terminal catalytic K (resno 1) has no resno=0 neighbor on the
    chain; only one neighbor gets the omit."""
    df = _pt([
        (1, "A", "K", True),     # catalytic K, N-terminal
        (2, "A", "A", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[1],
        chain="A",
    )
    assert out == {"A2": "KR"}


def test_omit_dict_pte_i1_real_classify_table_yields_156_158_only():
    """Snapshot test on the actual PTE_i1 PositionTable from a previous
    run: catalytic set = (60,64,128,131,132,157), only 157 is K, so
    expected output is {"A156": "KR", "A158": "KR"}."""
    classify_path = Path(
        "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260503-145751/"
        "classify/positions.tsv"
    )
    if not classify_path.is_file():
        pytest.skip("PTE_i1 classify_positions output not available")
    df = pd.read_csv(classify_path, sep="\t")
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=v2.DEFAULT_CATRES,
        chain="A",
    )
    assert out == {"A156": "KR", "A158": "KR"}


def test_omit_dict_custom_forbid_aas():
    """Caller can change the forbidden AA set."""
    df = _pt([
        (10, "A", "K", True),
        (11, "A", "K", True),    # catalytic K
        (12, "A", "L", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[11],
        chain="A",
        forbid_aas="K",  # only forbid K, allow R
    )
    assert out == {"A10": "K", "A12": "K"}


def test_omit_dict_chain_filter():
    """Only neighbors on the same chain as the catalytic residue are
    forbidden."""
    df = _pt([
        (50, "A", "K", True),    # catalytic K on chain A
        (51, "A", "L", True),
        (50, "B", "L", True),    # different chain, same resno (ligand)
        (51, "B", "K", True),
    ])
    out = v2.compute_catalytic_neighbor_omit_dict(
        position_table_df=df,
        fixed_resnos=[50],
        chain="A",
    )
    # Only A51 should appear (not B51, not the off-chain partner)
    assert out == {"A51": "KR"}


# ----------------------------------------------------------------------
# Feature 1: per-cycle probabilistic conserved-H-bond fixing (stage_sample)
# Feature 2: canonical REMARK transfer (stage_restore_pdbs)
# ----------------------------------------------------------------------
def _conserve_rec(resno, *, kind="ligand", clashes=False, clash_with=""):
    from protein_chisel.tools.conserved_hbonds import ConservableHbond
    return ConservableHbond(
        resno=resno, resname="SER", sidechain_atom="OG",
        partner_kind=kind, partner_resno=(-1 if kind == "ligand" else 16),
        partner_resname=("LIG" if kind == "ligand" else "HIS"),
        partner_atom="O1", distance=2.8, strength=0.9, strength_bin="super_strong",
        hypothesized_donor=f"SER{resno}/OG", hypothesized_acceptor="LIG/O1",
        clashes=clashes, clash_with=clash_with,
    )


def _run_stage_sample_capture(tmp_path, monkeypatch, *, keep_clashing):
    """Drive stage_sample with detection + MPNN stubbed; return the fixed_resnos
    actually handed to LigandMPNN."""
    import numpy as np
    from types import SimpleNamespace

    from protein_chisel.tools import conserved_hbonds as ch
    from protein_chisel.tools import ligand_mpnn as lm

    # Detection returns a clean ligand-H-bonder (40) + a clashing one (30).
    monkeypatch.setattr(
        ch, "find_conservable_sidechain_hbonds",
        lambda *a, **k: [_conserve_rec(40),
                         _conserve_rec(30, clashes=True, clash_with="GLY90/O")],
    )
    captured: dict = {}

    def fake_sample(**kw):
        captured["fixed"] = sorted(int(r) for r in kw["fixed_resnos"])
        cs = SimpleNamespace(to_disk=lambda *a, **k: None, df=[])
        return SimpleNamespace(candidate_set=cs)

    monkeypatch.setattr(lm, "sample_with_ligand_mpnn", fake_sample)

    monkeypatch.setattr(v2, "CONSERVE_HBONDS", True)
    monkeypatch.setattr(v2, "CONSERVE_HBOND_PROB", 1.0)          # roll-in everything
    monkeypatch.setattr(v2, "CONSERVE_KEEP_CLASHING", keep_clashing)
    monkeypatch.setattr(v2, "CONSERVE_SEED_BASE", 1)
    monkeypatch.setattr(v2, "CONSERVE_ANCHORS", ("ligand", "catalytic"))

    seed = tmp_path / "seed.pdb"
    seed.write_text("END\n")
    cc = SimpleNamespace(cycle_idx=0, n_samples=2, sampling_temperature=0.2,
                         use_side_chain_context=0, enhance="", omit_AA="X")
    v2.stage_sample(
        cycle_cfg=cc, seed_pdb=seed, bias=np.zeros((4, 20)),
        protein_resnos=[16, 30, 40, 41], fixed_resnos=[16],
        out_dir=tmp_path / "samp", chain="A",
    )
    return captured["fixed"]


def test_stage_sample_merges_conserved_into_fixed_excludes_clashing(tmp_path, monkeypatch):
    # catalytic 16 + rolled non-clashing 40; clashing 30 is dropped.
    assert _run_stage_sample_capture(tmp_path, monkeypatch, keep_clashing=False) == [16, 40]


def test_stage_sample_keep_clashing_includes_30(tmp_path, monkeypatch):
    assert _run_stage_sample_capture(tmp_path, monkeypatch, keep_clashing=True) == [16, 30, 40]


def test_stage_restore_transfers_canonical_remarks(tmp_path, monkeypatch):
    from protein_chisel.tools import pdb_restoration as pr

    seed = tmp_path / "seed.pdb"
    seed.write_text(
        "REMARK 665 legend\n"
        "REMARK 666 MATCH TEMPLATE B LIG  200 MATCH MOTIF A HIS   16  1  1\n"
        "REMARK 667 legend\n"
        "REMARK 668   1   A HIS     16   HID -   HD1                     HIS\n"
        "REMARK QCB TOTAL_CHARGE +1\n"
        "REMARK DESIGN_PATH rfd3 output /scratch/x.pdb\n"
        "ATOM      1  CA  HIS A  16       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    out_dir = tmp_path / "restored"
    out_dir.mkdir()

    def fake_restore(**kw):
        p = kw["out_pdb_dir"] / "c0.pdb"
        p.write_text(
            "ATOM      1  CA  HIS A  16       0.000   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
        return {"c0": p}

    monkeypatch.setattr(pr, "restore_sample_dir", fake_restore)
    monkeypatch.setattr(v2, "TRANSFER_REMARKS", True)

    v2.stage_restore_pdbs(
        sample_dir=tmp_path, ref_pdb=seed, out_pdb_dir=out_dir,
        pdb_basename="x", candidate_ids=["c0"], chain="A",
        catalytic_resnos=[16], catalytic_hydrogens=False,
    )
    txt = (out_dir / "c0.pdb").read_text()
    # restore_sample_dir only carries 666; the shared remarks module rescues the rest.
    assert "REMARK 665" in txt
    assert "REMARK 666" in txt
    assert "REMARK 668" in txt
    assert "REMARK QCB TOTAL_CHARGE +1" in txt
    assert "REMARK DESIGN_PATH rfd3 output /scratch/x.pdb" in txt  # chain carried
    # Intermediate restored PDBs must NOT get a stage stamp — that's stamped once
    # at final adoption, pointing at the adopted top-K (not an intermediate path).
    assert "DESIGN_PATH iterative_design" not in txt


def test_write_final_topk_stamps_remarks(tmp_path, monkeypatch):
    # Guards the seed_pdb threading at both _write_final_topk_artifacts call sites.
    monkeypatch.setattr(v2, "TRANSFER_REMARKS", True)
    seed = tmp_path / "seed.pdb"
    seed.write_text(
        "REMARK 666 MATCH TEMPLATE B LIG  200 MATCH MOTIF A HIS   16  1  1\n"
        "REMARK QCB TOTAL_CHARGE +1\n"
        "ATOM      1  CA  HIS A  16       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    src = tmp_path / "src.pdb"
    src.write_text(
        "ATOM      1  CA  HIS A  16       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    top = pd.DataFrame([{"id": "d0", "sequence": "ACDE"}])
    v2._write_final_topk_artifacts(
        top=top, final_dir=final_dir, pdb_map={"d0": src}, seed_pdb=seed)
    out = (final_dir / "topk_pdbs" / "d0.pdb").read_text()
    assert "REMARK 666" in out and "REMARK QCB TOTAL_CHARGE +1" in out
    assert "REMARK DESIGN_PATH iterative_design" in out
