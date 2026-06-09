"""Host tests for the driver's two PoE modes in stage_sample (score-only +
emit-inputs), loaded via the same module-import pattern as test_gating_primitives.
The default 'bias' backend leaves both globals None (covered by test_gating_primitives'
import); here we exercise the opt-in branches without any container/sampler."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "iterative_design.py"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("itd_poe", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"driver not importable: {exc}")
    return mod


def _reset(driver):
    driver._POE_SAMPLE_DIR = None
    driver._POE_EMIT_INPUTS_DIR = None
    driver._POE_EXPERTS = ()
    driver._POE_LAMBDAS = ()


def _make_poe_output(root: Path, stem: str):
    (root / "seqs").mkdir(parents=True)
    (root / "seqs" / f"{stem}.fa").write_text(
        f">{stem}, T=0.1, seed=0, num_res=5\nMKLVA\n"
        f">{stem}, id=1, T=0.1, seed=0, overall_confidence=0.5, seq_rec=0.7\nMKLVC\n"
        f">{stem}, id=2, T=0.1, seed=0, overall_confidence=0.4, seq_rec=0.6\nMKLVD\n"
    )
    packed = root / "packed"
    packed.mkdir()
    for i in (1, 2):
        (packed / f"{stem}_packed_{i}_1.pdb").write_text("ATOM\nEND\n")
    return root


def test_score_only_builds_candidates_and_symlinks_packed(driver, tmp_path):
    stem = "myseed"
    poe_dir = _make_poe_output(tmp_path / "poe", stem)
    out_dir = tmp_path / "01_sample"
    try:
        driver._POE_SAMPLE_DIR = str(poe_dir)
        driver._POE_EXPERTS = ("esm",)
        driver._POE_LAMBDAS = (0.2,)
        cand_tsv = driver.stage_sample(
            cycle_cfg=driver.CycleConfig(cycle_idx=0),
            seed_pdb=tmp_path / f"{stem}.pdb",          # only .stem is used here
            bias=np.zeros((5, 20)), protein_resnos=[1, 2, 3, 4, 5],
            fixed_resnos=set(), out_dir=out_dir,
        )
        assert cand_tsv == out_dir / "candidates.tsv" and cand_tsv.is_file()
        import pandas as pd
        df = pd.read_csv(cand_tsv, sep="\t")
        assert len(df) == 3 and list(df["sampler"]) == ["fused_mpnn_poe"] * 3
        assert df.iloc[1]["id"] == f"{stem}_lmpnn_001"
        # packed/ symlinked so stage_restore_pdbs finds <stem>_packed_<idx>_1.pdb
        link = out_dir / "packed"
        assert link.is_symlink() or link.is_dir()
        assert (link / f"{stem}_packed_1_1.pdb").exists()
    finally:
        _reset(driver)


def test_score_only_missing_poe_fasta_raises(driver, tmp_path):
    out_dir = tmp_path / "01_sample"
    try:
        driver._POE_SAMPLE_DIR = str(tmp_path / "empty_poe")
        (tmp_path / "empty_poe" / "seqs").mkdir(parents=True)
        driver._POE_EXPERTS = ("esm",); driver._POE_LAMBDAS = (0.2,)
        with pytest.raises(RuntimeError, match="no sequences"):
            driver.stage_sample(
                cycle_cfg=driver.CycleConfig(cycle_idx=0),
                seed_pdb=tmp_path / "nope.pdb",
                bias=np.zeros((5, 20)), protein_resnos=[1, 2, 3, 4, 5],
                fixed_resnos=set(), out_dir=out_dir)
    finally:
        _reset(driver)


def test_emit_inputs_writes_jsons_and_exits(driver, tmp_path):
    emit = tmp_path / "poe_inputs"
    out_dir = tmp_path / "01_sample"
    L = 6
    try:
        driver._POE_EMIT_INPUTS_DIR = str(emit)
        with pytest.raises(SystemExit) as ei:
            driver.stage_sample(
                cycle_cfg=driver.CycleConfig(cycle_idx=0),
                seed_pdb=tmp_path / "seed.pdb",        # path only resolved, not read
                bias=np.zeros((L, 20)),
                protein_resnos=list(range(1, L + 1)),
                fixed_resnos={1, 3},
                out_dir=out_dir,
                omit_AA_per_residue={"A2": "C"},
            )
        assert ei.value.code == 0
        import json
        bias = json.loads((emit / "bias.json").read_text())
        fixed = json.loads((emit / "fixed.json").read_text())
        omit = json.loads((emit / "omit.json").read_text())
        # multi-format keyed by the resolved seed path
        key = str((tmp_path / "seed.pdb").resolve())
        assert key in bias and key in fixed and key in omit
        # fixed residues -> ['A1','A3'] (sorted, chain-labeled)
        assert sorted(fixed[key]) == ["A1", "A3"]
        assert omit[key] == {"A2": "C"}
    finally:
        _reset(driver)


def test_emit_inputs_default_none_does_not_trigger(driver):
    # sanity: with both globals reset, the PoE branches are inert
    _reset(driver)
    assert driver._POE_EMIT_INPUTS_DIR is None and driver._POE_SAMPLE_DIR is None
