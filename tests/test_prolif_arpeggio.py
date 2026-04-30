"""Cluster tests for tools/prolif_fingerprint and tools/arpeggio_interactions.

ProLIF runs in esmc.sif. Arpeggio needs mmCIF input — we generate one
on the fly from design.pdb using biotite if available; otherwise the
test skips itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.cluster


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"


# ---- ProLIF ----------------------------------------------------------------


def test_prolif_fingerprint_design_returns_or_fails_softly():
    """ProLIF either runs or fails softly with rdkit_failure=True.

    On the YYE ligand (carbamylated cap + 2 Zn), RDKit's bond perception
    from PDB heavy atoms fails with AtomValenceException — that is an
    expected limitation documented on the wrapper. The wrapper should
    return a populated result with rdkit_failure=True rather than raise.
    """
    from protein_chisel.tools.prolif_fingerprint import prolif_fingerprint

    res = prolif_fingerprint(DESIGN_PDB)
    if res.rdkit_failure:
        # The contract: empty result with the failure flag set.
        assert res.n_interactions == 0
        assert res.n_residues_with_interactions == 0
        assert res.rdkit_failure_reason  # non-empty
    else:
        # If RDKit happened to perceive the ligand, we expect interactions.
        assert res.n_interactions > 0
        assert res.boolean_df.shape[0] == 1


def test_prolif_to_dict_has_required_keys():
    from protein_chisel.tools.prolif_fingerprint import prolif_fingerprint

    res = prolif_fingerprint(DESIGN_PDB)
    d = res.to_dict()
    assert "prolif__n_interactions" in d
    assert "prolif__rdkit_failure" in d


def test_prolif_wrong_chain_raises():
    from protein_chisel.tools.prolif_fingerprint import prolif_fingerprint

    with pytest.raises(RuntimeError, match="no atoms match"):
        # Chain Z doesn't exist — should raise, not widen
        prolif_fingerprint(
            DESIGN_PDB, ligand_chain="Z", ligand_resname="YYE", ligand_resno=209,
        )


# ---- Arpeggio --------------------------------------------------------------


def test_arpeggio_rejects_pdb_input():
    """Pdbe-arpeggio requires mmCIF; passing a .pdb must fail fast."""
    from protein_chisel.tools.arpeggio_interactions import arpeggio_interactions

    with pytest.raises(ValueError, match="mmCIF"):
        arpeggio_interactions(DESIGN_PDB)


def test_arpeggio_runs_on_mmcif(tmp_path: Path):
    """Convert design.pdb to mmCIF and run arpeggio on it.

    Skipped when openbabel isn't installed (pdbe-arpeggio's hard dep).
    """
    pytest.importorskip("biotite.structure.io")
    pytest.importorskip("openbabel", reason="pdbe-arpeggio needs openbabel")
    from biotite.structure.io import pdb as bpdb
    from biotite.structure.io import pdbx as bpdbx
    from protein_chisel.tools.arpeggio_interactions import arpeggio_interactions

    cif_path = tmp_path / "design.cif"
    struct = bpdb.PDBFile.read(str(DESIGN_PDB)).get_structure(model=1)
    cif_file = bpdbx.CIFFile()
    bpdbx.set_structure(cif_file, struct)
    cif_file.write(str(cif_path))

    res = arpeggio_interactions(cif_path, keep_outputs=False)
    # Either contacts are detected or arpeggio bailed gracefully.
    if res.n_total_contacts > 0:
        assert sum(res.n_contacts_by_type.values()) > 0
