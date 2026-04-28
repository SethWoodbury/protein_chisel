"""Host unit tests for ligand_mpnn helpers (no LigandMPNN execution)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from protein_chisel.tools.ligand_mpnn import (
    LigandMPNNConfig,
    MPNN_AA_ORDER,
    _build_bias_per_residue_multi,
    _build_fixed_residues_multi,
    _config_hash,
    _parse_header,
    _parse_output_fasta,
    _residue_label,
)
from protein_chisel.sampling.plm_fusion import AA_ORDER as PLM_AA_ORDER


def test_mpnn_aa_order_string():
    assert MPNN_AA_ORDER == "ACDEFGHIKLMNPQRSTVWYX"
    assert PLM_AA_ORDER == "ACDEFGHIKLMNPQRSTVWY"


def test_residue_label_format():
    assert _residue_label("A", 92) == "A92"
    assert _residue_label("C", 1) == "C1"


def test_build_fixed_residues_multi():
    out = _build_fixed_residues_multi("/x/design.pdb", [3, 1, 2, 1], "A")
    paths = list(out.keys())
    assert len(paths) == 1
    # dedup + sort + chain prefix
    assert out[paths[0]] == ["A1", "A2", "A3"]


def test_build_bias_per_residue_multi_layout():
    bias = np.zeros((3, 20))
    bias[0, 0] = 1.5      # boost A at residue 0
    bias[2, 8] = -2.0     # penalize K at residue 2 (PLM_AA_ORDER index 8 is K)
    out = _build_bias_per_residue_multi(
        "/x/d.pdb", bias, chain="A", protein_resnos=[10, 11, 12],
    )
    paths = list(out.keys())
    assert len(paths) == 1
    inner = out[paths[0]]
    # Row 1 was all zeros; should be skipped
    assert "A11" not in inner
    assert "A10" in inner and "A12" in inner
    assert inner["A10"]["A"] == 1.5
    assert inner["A12"]["K"] == -2.0


def test_build_bias_per_residue_multi_rejects_shape_mismatch():
    bias = np.zeros((3, 20))
    with pytest.raises(ValueError):
        _build_bias_per_residue_multi(
            "/x.pdb", bias, chain="A", protein_resnos=[1, 2],  # too short
        )
    with pytest.raises(ValueError):
        _build_bias_per_residue_multi(
            "/x.pdb", np.zeros((3, 19)), chain="A", protein_resnos=[1, 2, 3],
        )


def test_parse_header_extracts_numeric_fields():
    h = "design, T=0.1, seed=42, overall_confidence=0.85, ligand_confidence=0.78, seq_rec=0.45"
    meta = _parse_header(h)
    assert meta["t"] == pytest.approx(0.1)
    assert meta["seed"] == 42.0
    assert meta["overall_confidence"] == pytest.approx(0.85)
    assert meta["seq_rec"] == pytest.approx(0.45)


def test_parse_output_fasta(tmp_path: Path):
    fasta = tmp_path / "x.fa"
    fasta.write_text(
        ">design, T=0.1, seed=0, num_res=200, model_path=path\n"
        "MEEVEEYARLVIE\n"
        ">design, id=0, T=0.1, seed=0, overall_confidence=0.81, seq_rec=0.50\n"
        "MGGVEEYARLVIE\n"
    )
    out = _parse_output_fasta(fasta)
    assert len(out) == 2
    assert out[0][1] == "MEEVEEYARLVIE"
    assert out[1][2]["seq_rec"] == 0.5


def test_parse_output_fasta_missing_file_returns_empty(tmp_path: Path):
    assert _parse_output_fasta(tmp_path / "absent.fa") == []


def test_config_hash_changes_with_params():
    cfg = LigandMPNNConfig(temperature=0.1)
    h1 = _config_hash(cfg, n=10, fixed=[1, 2], biased=True)
    cfg2 = LigandMPNNConfig(temperature=0.2)
    h2 = _config_hash(cfg2, n=10, fixed=[1, 2], biased=True)
    h3 = _config_hash(cfg, n=20, fixed=[1, 2], biased=True)
    h4 = _config_hash(cfg, n=10, fixed=[1, 2, 3], biased=True)
    assert len({h1, h2, h3, h4}) == 4


def test_config_repack_everything_default_zero():
    """Critical for theozyme protection — fixed residues must NOT be repacked."""
    cfg = LigandMPNNConfig()
    assert cfg.repack_everything == 0
    assert cfg.ligand_mpnn_use_side_chain_context == 1
    assert cfg.pack_side_chains == 1


def test_config_default_omit_AA_includes_C():
    """C and X are commonly-omitted; we want this default for surface design."""
    cfg = LigandMPNNConfig()
    assert "C" in cfg.omit_AA
    assert "X" in cfg.omit_AA
