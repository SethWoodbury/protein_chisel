"""Tests for utils/pose — runs inside pyrosetta.sif.

Marked `cluster` so default `pytest` runs skip them. Run via:

    apptainer exec --bind /home/woodbuse/codebase_projects/protein_chisel:/code \
        --env PYTHONPATH=/code/src \
        /net/software/containers/pyrosetta.sif \
        python -m pytest -m cluster /code/tests/test_pose.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest


# These tests need PyRosetta. They're marked `cluster` and skipped by default.
pytestmark = pytest.mark.cluster


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"
PARAMS_DIR = Path(
    "/home/woodbuse/testing_space/scaffold_optimization/"
    "ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params"
)


@pytest.fixture(scope="module")
def initialized_pose():
    from protein_chisel.utils.pose import init_pyrosetta, pose_from_file

    init_pyrosetta(params=[PARAMS_DIR])
    return pose_from_file(DESIGN_PDB)


def test_pose_loads(initialized_pose):
    pose = initialized_pose
    assert pose.size() == 209  # 208 protein + 1 YYE ligand
    assert pose.num_chains() == 2


def test_find_ligand(initialized_pose):
    from protein_chisel.utils.pose import find_ligand_seqpos, get_ligand_seqposes

    assert find_ligand_seqpos(initialized_pose) == 209
    assert get_ligand_seqposes(initialized_pose) == [209]


def test_default_scorefxn_runs(initialized_pose):
    from protein_chisel.utils.pose import get_default_scorefxn

    sfxn = get_default_scorefxn()
    score = sfxn(initialized_pose)
    # any finite score is a sanity pass; designed proteins are usually < 0
    assert isinstance(score, float)


def test_hbond_set_extraction(initialized_pose):
    from protein_chisel.utils.pose import get_hbond_set, hbonds_as_dicts

    hbset = get_hbond_set(initialized_pose)
    assert hbset.nhbonds() > 0  # designed proteins have many hbonds
    rows = hbonds_as_dicts(initialized_pose, hbset)
    assert len(rows) == hbset.nhbonds()
    sample = rows[0]
    assert {"donor_res", "acceptor_res", "energy"} <= set(sample)
    # All energies negative (favorable hbonds)
    assert all(r["energy"] < 0 for r in rows)


def test_per_residue_sasa_includes_all_residues(initialized_pose):
    from protein_chisel.utils.pose import get_per_residue_sasa

    sasa = get_per_residue_sasa(initialized_pose)
    assert len(sasa) == initialized_pose.size()
    # SASA is in Å²; whole-pose total should be hundreds to thousands.
    total = sum(sasa.values())
    assert 1000 < total < 50000, total


def test_get_sasa_for_specific_residue(initialized_pose):
    """Pick the catalytic Lys64 (KCX edge case) and confirm SASA is returned."""
    from protein_chisel.utils.pose import getSASA

    sasa = getSASA(initialized_pose, resno=64)
    assert sasa >= 0.0


def test_thread_sequence_round_trip(initialized_pose):
    """Threading the original sequence onto the pose should give back ~the same pose."""
    from protein_chisel.utils.pose import thread_sequence

    pose = initialized_pose
    seq_before = pose.sequence()
    # Drop the ligand char at the end
    protein_seq = seq_before[: pose.size() - 1] if pose.residue(pose.size()).is_ligand() else seq_before
    threaded = thread_sequence(pose, protein_seq)
    seq_after = threaded.sequence()
    assert seq_after[: len(protein_seq)] == protein_seq


def test_mutate_residue_changes_aa(initialized_pose):
    """Mutate residue 1 to ALA and verify."""
    from protein_chisel.utils.pose import mutate_residue

    pose = initialized_pose.clone()
    mutate_residue(pose, 1, "A")
    assert pose.residue(1).name3() == "ALA"


def test_per_atom_sasa_for_ligand_atoms(initialized_pose):
    """Probe the carbamate cap atoms (C1/O1/O2) which are part of the YYE ligand."""
    from protein_chisel.utils.pose import get_per_atom_sasa

    surf_vol = get_per_atom_sasa(initialized_pose, probe_radius=2.8)
    pose = initialized_pose
    yye = pose.residue(209)  # ligand
    target_names = {"C1", "O1", "O2"}
    found = {}
    for i in range(1, yye.natoms() + 1):
        name = yye.atom_name(i).strip()
        if name in target_names:
            found[name] = surf_vol.surf(209, i)
    assert set(found) == target_names
    # The carbamate cap atoms point toward the catalytic lysine; SASA should
    # be moderate (not buried, not fully exposed). We don't pin a number — a
    # finite, ~non-negative SASA is the contract (allowing tiny FP noise).
    for name, sasa in found.items():
        assert sasa > -1e-6, f"{name} SASA is meaningfully negative: {sasa}"
