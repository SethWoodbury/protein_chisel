"""Tests for the PoE backend foundation (validation + command construction +
FASTA loading). Pure/host-side — no container execution."""
from __future__ import annotations

import pytest

from protein_chisel.sampling.mpnn_backends import (
    BACKEND_BIAS, BACKEND_POE, DEFAULT_POE_LIGAND_CHECKPOINT, SUPPORTED_POE_EXPERTS,
    build_poe_command, load_poe_candidates, poe_output_fasta, validate_expert_lambdas,
)


# ---- validate_expert_lambdas --------------------------------------------
def test_validate_ok_single_and_multi():
    exp, lam = validate_expert_lambdas("e1", "0.2")
    assert exp == ["e1"] and lam == [0.2]
    exp, lam = validate_expert_lambdas("hermes,e1", "0.2,0.3")
    assert exp == ["hermes", "e1"] and lam == [0.2, 0.3]
    # accepts sequences too
    exp, lam = validate_expert_lambdas(["esm"], [0.5])
    assert exp == ["esm"] and lam == [0.5]


def test_validate_rejects_unknown_expert():
    with pytest.raises(ValueError, match="unknown expert"):
        validate_expert_lambdas("warpdrive", "0.2")


def test_validate_rejects_count_mismatch():
    with pytest.raises(ValueError, match="counts must match"):
        validate_expert_lambdas("hermes,e1", "0.2")


def test_validate_rejects_lambda_sum_ge_one():
    with pytest.raises(ValueError, match="must be < 1.0"):
        validate_expert_lambdas("hermes,e1", "0.6,0.5")
    # exactly 1.0 is invalid (lambda_mpnn would be 0)
    with pytest.raises(ValueError, match="must be < 1.0"):
        validate_expert_lambdas("hermes,e1", "0.5,0.5")


def test_validate_rejects_out_of_range_lambda():
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        validate_expert_lambdas("e1", "0")
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        validate_expert_lambdas("e1", "1.5")


def test_validate_rejects_empty_experts():
    with pytest.raises(ValueError, match="empty"):
        validate_expert_lambdas("", "")


def test_supported_experts_match_readme():
    assert "hermes" in SUPPORTED_POE_EXPERTS and "e1" in SUPPORTED_POE_EXPERTS
    assert "esm" in SUPPORTED_POE_EXPERTS and "vesm" in SUPPORTED_POE_EXPERTS


# ---- build_poe_command --------------------------------------------------
def test_build_command_core_flags():
    cmd = build_poe_command(
        pdb_path="/in/seed.pdb", out_folder="/out",
        experts=["e1"], lambdas=[0.2],
        bias_json="/tmp/bias.json", omit_json="/tmp/omit.json",
        fixed_json="/tmp/fixed.json",
    )
    s = " ".join(cmd)
    assert cmd[:4] == ["apptainer", "exec", "--nv", str(__import__(
        "protein_chisel.paths", fromlist=["POE_MPNN_SIF"]).POE_MPNN_SIF)]
    assert "run.py" in s
    assert "--model_type ligand_mpnn" in s
    assert "--checkpoint_ligand_mpnn " + DEFAULT_POE_LIGAND_CHECKPOINT in s
    assert "--pdb_path /in/seed.pdb" in s
    assert "--out_folder /out" in s
    assert "--additional_experts e1" in s
    assert "--additional_expert_lambdas 0.2" in s
    assert "--bias_AA_per_residue_multi /tmp/bias.json" in s
    assert "--omit_AA_per_residue_multi /tmp/omit.json" in s
    assert "--fixed_residues_multi /tmp/fixed.json" in s
    # packing defaults match the in-driver sampler (PDBs out, catalytic rotamers kept)
    assert "--pack_side_chains 1" in s
    assert "--repack_everything 0" in s
    assert "--packed_suffix _packed" in s


def test_build_command_multi_expert_joins_with_commas():
    cmd = build_poe_command(
        pdb_path="x.pdb", out_folder="o",
        experts=["hermes", "e1"], lambdas=[0.2, 0.3],
    )
    s = " ".join(cmd)
    assert "--additional_experts hermes,e1" in s
    assert "--additional_expert_lambdas 0.2,0.3" in s
    # no JSONs passed -> those flags absent
    assert "--bias_AA_per_residue_multi" not in s
    assert "--fixed_residues_multi" not in s


def test_build_command_hermes_probs_and_overrides():
    cmd = build_poe_command(
        pdb_path="x.pdb", out_folder="o", experts=["hermes"], lambdas=[0.25],
        hermes_probs="/h.csv", checkpoint="/my/ckpt.pt",
        batch_size=2, number_of_batches=5, temperature=0.2, seed=42,
    )
    s = " ".join(cmd)
    assert "--hermes_probs /h.csv" in s
    assert "--checkpoint_ligand_mpnn /my/ckpt.pt" in s
    assert "--batch_size 2" in s and "--number_of_batches 5" in s
    assert "--temperature 0.2" in s and "--seed 42" in s


def test_build_command_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_poe_command(pdb_path="x", out_folder="o",
                          experts=["a", "b"], lambdas=[0.2])


# ---- output path + FASTA loading ---------------------------------------
def test_poe_output_fasta_path():
    p = poe_output_fasta("/out", "myseed", file_ending="")
    assert str(p).endswith("/out/seqs/myseed.fa")
    p2 = poe_output_fasta("/out", "myseed", file_ending="_v2")
    assert str(p2).endswith("/out/seqs/myseed.fa_v2")


def test_load_poe_candidates_parses_fasta(tmp_path):
    # fused_mpnn_poe writes the same format as LigandMPNN: input header first,
    # then one record per design with id=/T=/seq_rec= fields.
    fa = tmp_path / "seqs" / "seed.fa"
    fa.parent.mkdir(parents=True)
    fa.write_text(
        ">seed, T=0.1, seed=0, num_res=10\nMKLVAAAAAA\n"
        ">seed, id=0, T=0.1, seed=0, overall_confidence=0.55, seq_rec=0.6\nMKLVCCCCCC\n"
        ">seed, id=1, T=0.1, seed=0, overall_confidence=0.51, seq_rec=0.4\nMKLVDDDDDD\n"
    )
    recs = load_poe_candidates(fa)
    assert len(recs) == 3                      # input header + 2 designs
    # design records carry parsed numeric fields
    h1, seq1, fields1 = recs[1]
    assert seq1 == "MKLVCCCCCC"
    assert fields1.get("id") == 0.0 and fields1.get("seq_rec") == 0.6


def test_load_poe_candidates_missing_file(tmp_path):
    assert load_poe_candidates(tmp_path / "nope.fa") == []


def test_backend_constants():
    assert BACKEND_BIAS == "bias" and BACKEND_POE == "poe"
