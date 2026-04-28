"""Tests for io/schemas.py — the artifact contracts everything else depends on."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from protein_chisel.io.schemas import (
    CandidateSet,
    Manifest,
    MetricTable,
    POSITION_TABLE_REQUIRED,
    PositionTable,
    PoseEntry,
    PoseSet,
    manifest_matches,
    sha256_file,
    sha256_obj,
)


# ---- hashing --------------------------------------------------------------


def test_sha256_file_stable(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2 == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_obj_order_independent():
    a = {"x": 1, "y": [1, 2], "z": {"a": True, "b": None}}
    b = {"z": {"b": None, "a": True}, "y": [1, 2], "x": 1}
    assert sha256_obj(a) == sha256_obj(b)


# ---- manifest -------------------------------------------------------------


def test_manifest_hash_excludes_timestamp(tmp_path: Path):
    inp = tmp_path / "in.pdb"
    inp.write_text("ATOM\n")
    m1 = Manifest.for_stage("classify_positions", [inp], {"k": 1}, {"pyrosetta": "2024.39"})
    m2 = Manifest.for_stage("classify_positions", [inp], {"k": 1}, {"pyrosetta": "2024.39"})
    # created_at and host should not affect the hash
    assert m1.hash() == m2.hash()


def test_manifest_hash_changes_on_input_change(tmp_path: Path):
    inp = tmp_path / "in.pdb"
    inp.write_text("ATOM\n")
    m1 = Manifest.for_stage("x", [inp], {"k": 1}, {})
    inp.write_text("ATOM 2\n")
    m2 = Manifest.for_stage("x", [inp], {"k": 1}, {})
    assert m1.hash() != m2.hash()


def test_manifest_hash_changes_on_config_change(tmp_path: Path):
    inp = tmp_path / "in.pdb"
    inp.write_text("ATOM\n")
    m1 = Manifest.for_stage("x", [inp], {"k": 1}, {})
    m2 = Manifest.for_stage("x", [inp], {"k": 2}, {})
    assert m1.hash() != m2.hash()


def test_manifest_round_trip(tmp_path: Path):
    inp = tmp_path / "in.pdb"
    inp.write_text("ATOM\n")
    m = Manifest.for_stage("x", [inp], {"k": 1}, {"pyrosetta": "v"})
    out = tmp_path / "manifest.json"
    m.to_json(out)
    loaded = Manifest.from_json(out)
    assert loaded.hash() == m.hash()
    assert loaded.stage == "x"
    assert loaded.inputs == m.inputs


def test_manifest_matches_only_when_identical(tmp_path: Path):
    inp = tmp_path / "in.pdb"
    inp.write_text("ATOM\n")
    m = Manifest.for_stage("x", [inp], {"k": 1}, {})
    out = tmp_path / "manifest.json"
    m.to_json(out)
    assert manifest_matches(m, out) is True
    inp.write_text("ATOM x\n")
    m2 = Manifest.for_stage("x", [inp], {"k": 1}, {})
    assert manifest_matches(m2, out) is False  # different input hash
    assert manifest_matches(m, out) is True   # original still matches the saved file


def test_manifest_matches_returns_false_for_missing_file(tmp_path: Path):
    m = Manifest(stage="x", inputs={}, config={}, tool_versions={})
    assert manifest_matches(m, tmp_path / "nope.json") is False


# ---- PoseSet --------------------------------------------------------------


def test_pose_set_from_single_pdb(tmp_path: Path):
    p = tmp_path / "design.pdb"
    p.write_text("ATOM\n")
    ps = PoseSet.from_single_pdb(p)
    assert len(ps) == 1
    assert ps.entries[0].fold_source == "designed"
    assert ps.entries[0].sequence_id == "design"


def test_pose_set_filter():
    ps = PoseSet([
        PoseEntry(path="/x/a.pdb", sequence_id="s1", fold_source="designed"),
        PoseEntry(path="/x/b.pdb", sequence_id="s1", fold_source="AF3_seed1"),
        PoseEntry(path="/x/c.pdb", sequence_id="s2", fold_source="AF3_seed1"),
    ])
    assert len(ps.filter(fold_source="designed")) == 1
    assert len(ps.filter(sequence_id="s1")) == 2


def test_pose_set_by_sequence():
    ps = PoseSet([
        PoseEntry(path="/x/a.pdb", sequence_id="s1", fold_source="designed"),
        PoseEntry(path="/x/b.pdb", sequence_id="s1", fold_source="AF3_seed1", conformer_index=1),
        PoseEntry(path="/x/c.pdb", sequence_id="s2", fold_source="designed"),
    ])
    grouped = ps.by_sequence()
    assert set(grouped) == {"s1", "s2"}
    assert len(grouped["s1"]) == 2


def test_pose_set_round_trip(tmp_path: Path):
    ps = PoseSet([
        PoseEntry(path="/x/a.pdb", sequence_id="s1", fold_source="designed", meta={"note": "test"}),
    ])
    out = tmp_path / "ps.json"
    ps.to_json(out)
    loaded = PoseSet.from_json(out)
    assert len(loaded) == 1
    assert loaded.entries[0].meta == {"note": "test"}


# ---- PositionTable --------------------------------------------------------


def _minimal_position_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "pose_id": ["d1"] * n,
        "resno": list(range(1, n + 1)),
        "chain": ["A"] * n,
        "name3": ["MET", "GLY", "HIS"][:n],
        "name1": ["M", "G", "H"][:n],
        "is_protein": [True] * n,
        "is_catalytic": [False, False, True][:n],
        "class": ["surface", "surface", "active_site"][:n],
        "sasa": [50.0, 30.0, 5.0][:n],
        "dist_ligand": [10.0, 8.0, 3.0][:n],
        "dist_catalytic": [3.0, 5.0, 0.0][:n],
        "ss": ["H", "L", "H"][:n],
        "ss_reduced": ["H", "L", "H"][:n],
        "in_pocket": [False, False, True][:n],
        "phi": [-60.0, -90.0, -65.0][:n],
        "psi": [-45.0, 0.0, -45.0][:n],
    })


def test_position_table_required_columns(tmp_path: Path):
    df = _minimal_position_df()
    pt = PositionTable(df=df)
    out = tmp_path / "pos.parquet"
    pt.to_parquet(out)
    loaded = PositionTable.from_parquet(out)
    assert len(loaded.df) == 3
    for col in POSITION_TABLE_REQUIRED:
        assert col in loaded.df.columns


def test_position_table_rejects_missing_columns():
    df = _minimal_position_df().drop(columns=["sasa"])
    with pytest.raises(ValueError, match="sasa"):
        PositionTable(df=df)


# ---- CandidateSet ---------------------------------------------------------


def test_candidate_set_round_trip(tmp_path: Path):
    df = pd.DataFrame({
        "id": ["c1", "c2"],
        "sequence": ["MGHHHHHH", "MGAATTGGG"],
        "parent_design_id": ["d1", "d1"],
        "sampler": ["ligand_mpnn", "ligand_mpnn"],
        "sampler_params_hash": ["abc", "abc"],
    })
    cs = CandidateSet(df=df)
    fasta = tmp_path / "c.fasta"
    meta = tmp_path / "c.parquet"
    cs.to_disk(fasta, meta)
    assert fasta.read_text() == ">c1\nMGHHHHHH\n>c2\nMGAATTGGG\n"
    loaded = CandidateSet.from_disk(meta)
    assert list(loaded.df["sequence"]) == ["MGHHHHHH", "MGAATTGGG"]


# ---- MetricTable ----------------------------------------------------------


def test_metric_table_merge():
    a = MetricTable(df=pd.DataFrame({
        "sequence_id": ["s1", "s2"],
        "conformer_index": [0, 0],
        "rosetta__total_score": [-100.0, -90.0],
    }))
    b = MetricTable(df=pd.DataFrame({
        "sequence_id": ["s1", "s2"],
        "conformer_index": [0, 0],
        "esmc__pseudo_perplexity": [3.5, 4.0],
    }))
    merged = a.merge(b)
    assert "rosetta__total_score" in merged.df.columns
    assert "esmc__pseudo_perplexity" in merged.df.columns
    assert len(merged.df) == 2


def test_metric_table_rejects_missing_id_columns():
    with pytest.raises(ValueError):
        MetricTable(df=pd.DataFrame({"x": [1, 2]}))
