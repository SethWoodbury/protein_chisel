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


def test_build_command_forwards_global_omit_and_sc_context():
    cmd = build_poe_command(pdb_path="x.pdb", out_folder="o", experts=["esm"],
                            lambdas=[0.2], omit_AA="CX", use_side_chain_context=1)
    s = " ".join(cmd)
    assert "--omit_AA CX" in s                       # global composition constraint
    assert "--ligand_mpnn_use_side_chain_context 1" in s
    # default: no --omit_AA emitted (empty)
    s0 = " ".join(build_poe_command(pdb_path="x.pdb", out_folder="o",
                                    experts=["esm"], lambdas=[0.2]))
    assert "--omit_AA" not in s0


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


# ---- candidate_set_from_poe_dir + sampler hash --------------------------
def _make_poe_dir(tmp_path, stem):
    seqs = tmp_path / "seqs"
    seqs.mkdir(parents=True)
    (seqs / f"{stem}.fa").write_text(
        f">{stem}, T=0.1, seed=42, num_res=5, num_ligand_res=3\nMKLVA\n"
        f">{stem}, id=1, T=0.1, seed=42, overall_confidence=0.5, seq_rec=0.7\nMKLVC\n"
        f">{stem}, id=2, T=0.1, seed=42, overall_confidence=0.4, seq_rec=0.6\nMKLVD\n"
    )
    return tmp_path


def test_candidate_set_from_poe_dir_shape(tmp_path):
    from protein_chisel.sampling.mpnn_backends import candidate_set_from_poe_dir
    stem = "pte_seed"
    _make_poe_dir(tmp_path, stem)
    cset = candidate_set_from_poe_dir(
        tmp_path, stem, parent_design_id="run0", experts=["esm"], lambdas=[0.2])
    df = cset.df
    assert len(df) == 3                                    # WT input + 2 designs
    # ids keep the _lmpnn_ form (so restore_sample_dir + rename work)
    assert list(df["id"]) == [f"{stem}_lmpnn_000", f"{stem}_lmpnn_001",
                              f"{stem}_lmpnn_002"]
    assert bool(df.iloc[0]["is_input"]) is True
    assert bool(df.iloc[1]["is_input"]) is False
    assert list(df["sampler"]) == ["fused_mpnn_poe"] * 3   # provenance tag
    assert df.iloc[1]["sequence"] == "MKLVC"
    # parsed header fields surfaced as mpnn_* columns
    assert "mpnn_seq_rec" in df.columns and df.iloc[1]["mpnn_seq_rec"] == 0.7
    # restore mapping: _lmpnn_001 -> packed _packed_1_1.pdb (idx parity), checked by
    # construction (idx 1 design). All rows share one sampler_params_hash.
    assert df["sampler_params_hash"].nunique() == 1


def test_candidate_set_from_poe_dir_missing_fasta_raises(tmp_path):
    from protein_chisel.sampling.mpnn_backends import candidate_set_from_poe_dir
    (tmp_path / "seqs").mkdir()
    with pytest.raises(RuntimeError, match="no sequences"):
        candidate_set_from_poe_dir(tmp_path, "nope", experts=["esm"], lambdas=[0.2])


def test_poe_sampler_params_hash_stable_and_sensitive():
    from protein_chisel.sampling.mpnn_backends import poe_sampler_params_hash
    h1 = poe_sampler_params_hash(["esm"], [0.2])
    h2 = poe_sampler_params_hash(["esm"], [0.2])
    h3 = poe_sampler_params_hash(["esm"], [0.3])         # different lambda
    assert h1 == h2 and h1 != h3 and len(h1) == 12
