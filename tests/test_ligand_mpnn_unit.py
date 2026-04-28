"""Host unit tests for ligand_mpnn helpers (no LigandMPNN execution)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from protein_chisel.tools.ligand_mpnn import (
    MPNN_AA_ORDER, _bias_to_mpnn_format, _build_bias_by_res_jsonl,
    _build_fixed_positions_jsonl, _config_hash, _parse_header,
    _parse_output_fasta, LigandMPNNConfig,
)


def test_mpnn_aa_order_string():
    assert MPNN_AA_ORDER == "ACDEFGHIKLMNPQRSTVWYX"
    assert len(MPNN_AA_ORDER) == 21


def test_bias_to_mpnn_format_pads_x_column():
    bias = np.random.normal(0, 1, (10, 20))
    out = _bias_to_mpnn_format(bias)
    assert out.shape == (10, 21)
    assert np.allclose(out[:, :20], bias)
    assert np.all(out[:, 20] == 0.0)


def test_bias_to_mpnn_format_rejects_wrong_shape():
    with pytest.raises(ValueError):
        _bias_to_mpnn_format(np.zeros((10, 19)))


def test_build_bias_by_res_jsonl_structure():
    bias = np.random.normal(0, 1, (5, 20))
    payload = json.loads(_build_bias_by_res_jsonl("design", "A", bias))
    assert "design" in payload
    assert "A" in payload["design"]
    arr = np.array(payload["design"]["A"])
    assert arr.shape == (5, 21)


def test_build_fixed_positions_jsonl_dedupes_and_sorts():
    payload = json.loads(_build_fixed_positions_jsonl("d", "A", [3, 1, 2, 1]))
    assert payload["d"]["A"] == [1, 2, 3]


def test_parse_header_extracts_numeric_fields():
    h = "T=0.1, sample=3, score=1.234, seq_recovery=0.85"
    meta = _parse_header(h)
    assert meta["t"] == pytest.approx(0.1)
    assert meta["sample"] == 3.0
    assert meta["score"] == pytest.approx(1.234)
    assert meta["seq_recovery"] == pytest.approx(0.85)


def test_parse_output_fasta(tmp_path: Path):
    fasta = tmp_path / "x.fa"
    fasta.write_text(
        ">T=0.1, sample=0, score=0.5\n"
        "MEEVEEYARLVIE\n"
        ">T=0.1, sample=1, score=1.2\n"
        "MGGVEEYARLVIE\n"
    )
    out = _parse_output_fasta(fasta)
    assert len(out) == 2
    assert out[0][1] == "MEEVEEYARLVIE"
    assert out[1][2]["score"] == pytest.approx(1.2)


def test_parse_output_fasta_missing_file_returns_empty(tmp_path: Path):
    assert _parse_output_fasta(tmp_path / "absent.fa") == []


def test_config_hash_changes_with_params():
    cfg = LigandMPNNConfig(sampling_temp=0.1)
    h1 = _config_hash(cfg, n=10, fixed=[1, 2], biased=True)
    cfg2 = LigandMPNNConfig(sampling_temp=0.2)
    h2 = _config_hash(cfg2, n=10, fixed=[1, 2], biased=True)
    h3 = _config_hash(cfg, n=20, fixed=[1, 2], biased=True)
    h4 = _config_hash(cfg, n=10, fixed=[1, 2, 3], biased=True)
    assert len({h1, h2, h3, h4}) == 4
