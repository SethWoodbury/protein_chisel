"""Cluster test for tools/rosetta_metrics.

Spawns pyrosetta.sif, runs the metrics XML against a known PDB+params,
and verifies the result dict has the expected keys + reasonable values.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest


pytestmark = pytest.mark.cluster


PDB_PATH = Path(
    "/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/"
    "ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb"
)
PARAMS_PATH = Path(
    "/home/woodbuse/testing_space/scaffold_optimization/"
    "ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params/YYE.params"
)


def _skip_if_missing():
    sif = Path("/net/software/containers/pyrosetta.sif")
    if not sif.is_file():
        pytest.skip(f"pyrosetta.sif not available at {sif}")
    if not PDB_PATH.is_file():
        pytest.skip(f"Test PDB not found: {PDB_PATH}")
    if not PARAMS_PATH.is_file():
        pytest.skip(f"Test params not found: {PARAMS_PATH}")


def test_compute_rosetta_metrics_smoke():
    _skip_if_missing()
    from protein_chisel.tools.rosetta_metrics import (
        RosettaMetricsConfig,
        compute_rosetta_metrics,
    )

    # YYE has phosphate (P1, O5, O2) and amide N (N1).
    key_atoms = ["P1", "O5", "O2", "N1"]
    res = compute_rosetta_metrics(
        PDB_PATH,
        ligand_params=PARAMS_PATH,
        key_atoms=key_atoms,
        ligand_exposed_atoms=key_atoms,
        config=RosettaMetricsConfig(ligand_chain="B"),
    )

    # Ligand should have been found (non-zero seqpos)
    assert res.ligand_seqpos > 0

    # Per-atom hbond counts: one entry per key atom, all >= 0
    assert set(res.per_atom_hbonds) == set(key_atoms)
    for atom, n in res.per_atom_hbonds.items():
        assert n >= 0, f"{atom}_hbond is negative: {n}"

    # Filter scalars: a few canary values that should always be set + finite.
    assert math.isfinite(res.contact_molecular_surface)
    assert math.isfinite(res.ddg)
    assert math.isfinite(res.ligand_interface_energy)
    assert math.isfinite(res.total_pose_sasa)
    # Designed proteins typically have hundreds of residues.
    assert res.total_residues_in_design_plus_ligand > 50

    # Counts >= 0
    assert res.number_DSSP_helices_in_design >= 0
    assert res.number_DSSP_sheets_in_design >= 0
    assert res.number_DSSP_loops_in_design >= 0

    # Simple metrics
    assert isinstance(res.secondary_structure, str)
    assert len(res.secondary_structure) > 0
    assert isinstance(res.secondary_structure_DSSP_reduced_alphabet, str)
    # SAP and EC are enabled by default
    assert math.isfinite(res.SAP_score)
    assert math.isfinite(res.electrostatic_complementarity)

    # H-bonds to ligand: integer count, each row is fully populated
    assert res.n_hbonds_to_ligand == len(res.ligand_hbonds_table)
    for row in res.ligand_hbonds_table:
        assert {"donor_res", "donor_atom", "acceptor_res",
                "acceptor_atom", "energy"} <= set(row)
        # Hbond energies are favorable (negative)
        assert row["energy"] < 0

    # ligand_exposed_atoms_sasa was requested; should be finite + non-negative
    assert math.isfinite(res.ligand_exposed_atoms_sasa)
    assert res.ligand_exposed_atoms_sasa >= 0


def test_to_dict_has_expected_keys():
    _skip_if_missing()
    from protein_chisel.tools.rosetta_metrics import (
        RosettaMetricsConfig,
        compute_rosetta_metrics,
    )

    # YYE has phosphate (P1, O5, O2) and amide N (N1).
    key_atoms = ["P1", "O5", "O2", "N1"]
    res = compute_rosetta_metrics(
        PDB_PATH,
        ligand_params=PARAMS_PATH,
        key_atoms=key_atoms,
        config=RosettaMetricsConfig(ligand_chain="B"),
    )
    d = res.to_dict()

    expected = {
        "rosetta__contact_molecular_surface",
        "rosetta__ddg",
        "rosetta__ligand_interface_energy",
        "rosetta__total_residues_in_design_plus_ligand",
        "rosetta__hydrophobic_residues_in_design",
        "rosetta__aliphatic_residues_in_design",
        "rosetta__net_charge_in_design_NOT_w_HIS",
        "rosetta__dSasa_fraction",
        "rosetta__number_DSSP_helices_in_design",
        "rosetta__number_DSSP_sheets_in_design",
        "rosetta__number_DSSP_loops_in_design",
        "rosetta__holes_in_design_lower_is_better",
        "rosetta__interface_holes_at_ligand",
        "rosetta__num_residues_at_ligand_interface",
        "rosetta__shape_complementarity_interface_area",
        "rosetta__shape_complementarity_median_distance_at_interface",
        "rosetta__hydrophobic_exposure_sasa_in_design",
        "rosetta__sasa_ligand_interface",
        "rosetta__total_pose_sasa",
        "rosetta__bad_torsion_preproline",
        "rosetta__longest_cont_polar_seg",
        "rosetta__longest_cont_apolar_seg",
        "rosetta__total_rosetta_energy_metric",
        "rosetta__secondary_structure",
        "rosetta__secondary_structure_DSSP_reduced_alphabet",
        "rosetta__SAP_score",
        "rosetta__electrostatic_complementarity",
        "rosetta__ligand_seqpos",
        "rosetta__ligand_exposed_atoms_sasa",
        "rosetta__n_hbonds_to_ligand",
    }
    for ka in key_atoms:
        expected.add(f"rosetta__{ka}_hbond")
    assert expected <= set(d), f"missing keys: {expected - set(d)}"


def test_metric_spec_present():
    """The MetricSpec adapter should be importable + correctly configured."""
    from protein_chisel.tools.rosetta_metrics import ROSETTA_METRICS_SPEC

    assert ROSETTA_METRICS_SPEC is not None
    assert ROSETTA_METRICS_SPEC.name == "rosetta_metrics"
    assert ROSETTA_METRICS_SPEC.prefix == "rosetta__"
    assert ROSETTA_METRICS_SPEC.cache_version == 1
    assert ROSETTA_METRICS_SPEC.kind == "structure+ligand"
