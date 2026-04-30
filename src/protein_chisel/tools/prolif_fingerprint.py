"""ProLIF wrapper — interaction fingerprint suitable for ML/data science.

ProLIF (https://github.com/chemosim-lab/ProLIF) detects:
hbond (donor / acceptor), salt bridge (cation / anion), π-stacking
(face-to-face, edge-to-face), π-cation, hydrophobic, vdW contact, metal
acceptor / donor.

For a single PDB it returns a per-(residue × interaction-type) boolean
table; we expose this directly + per-type and per-residue rollups.

Run inside esmc.sif (where prolif is installed). RDKit + MDAnalysis are
ProLIF deps — also in esmc.sif.

Reference:
    Bouysset, C., & Fiorucci, S. (2021). ProLIF: a library to encode
    molecular interactions as fingerprints. JCheminform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ProLIFResult:
    """A ProLIF interaction fingerprint.

    The ``boolean_df`` columns are MultiIndex of (ligand, residue,
    interaction_type) — same as ProLIF's native output. ``per_residue_counts``
    rolls up across interaction types per residue; useful for ranking
    "interaction-hub" residues. ``per_type_counts`` is the global tally.
    """

    boolean_df: pd.DataFrame                # raw ProLIF output
    per_residue_counts: pd.Series           # by residue label
    per_type_counts: pd.Series              # by interaction type
    n_interactions: int = 0
    n_residues_with_interactions: int = 0

    def to_dict(self, prefix: str = "prolif__") -> dict[str, float | int]:
        out: dict[str, float | int] = {
            f"{prefix}n_interactions": self.n_interactions,
            f"{prefix}n_residues_with_interactions": self.n_residues_with_interactions,
        }
        for typ, n in self.per_type_counts.items():
            out[f"{prefix}{typ}__count"] = int(n)
        return out


def prolif_fingerprint(
    pdb_path: str | Path,
    ligand_resname: Optional[str] = None,
    ligand_chain: Optional[str] = None,
    ligand_resno: Optional[int] = None,
    interactions: Optional[list[str]] = None,
) -> ProLIFResult:
    """Compute a protein-ligand interaction fingerprint with ProLIF.

    Args:
        pdb_path: input PDB.
        ligand_resname: HETATM residue name for the ligand. Defaults to
            the first non-water HETATM.
        ligand_chain: HETATM chain id (PDB column). Defaults to the first
            non-water HETATM's chain.
        ligand_resno: HETATM residue number — required for unambiguous
            multi-copy ligands. Defaults to the first non-water HETATM's
            resno.
        interactions: list of ProLIF interaction class names to enable
            (defaults to ProLIF's default set).

    Note:
        Strength-weighted output is intentionally NOT exposed yet —
        ProLIF's metadata interface for distances is non-stable across
        versions. Use ``boolean_df`` for ML feature input today.
    """
    import prolif as plf
    import MDAnalysis as mda

    pdb_path = str(Path(pdb_path).resolve())

    if ligand_resname is None or ligand_chain is None or ligand_resno is None:
        from protein_chisel.io.pdb import find_ligand
        info = find_ligand(pdb_path)
        if info is None:
            raise RuntimeError(f"no ligand HETATMs found in {pdb_path}")
        ligand_chain = ligand_chain or info[0]
        ligand_resname = ligand_resname or info[1]
        ligand_resno = ligand_resno if ligand_resno is not None else info[2]

    u = mda.Universe(pdb_path)
    protein_sel = u.select_atoms("protein")
    # MDAnalysis: chainID is the PDB column; resid matches PDB resnum.
    sel_str = (
        f"resname {ligand_resname} and chainID {ligand_chain} and resid {int(ligand_resno)}"
    )
    ligand_sel = u.select_atoms(sel_str)
    if len(ligand_sel) == 0:
        raise RuntimeError(
            f"no atoms match `{sel_str}` in {pdb_path} — verify chain/resno"
        )

    protein_mol = plf.Molecule.from_mda(protein_sel)
    ligand_mol = plf.Molecule.from_mda(ligand_sel)

    fp = plf.Fingerprint(interactions=interactions) if interactions else plf.Fingerprint()
    fp.run_from_iterable([ligand_mol], protein_mol, n_jobs=1, progress=False)
    df = fp.to_dataframe()  # MultiIndex (ligand, protein, interaction)

    bool_df = df.astype(bool)
    per_type = bool_df.sum(axis=0).groupby(level="interaction").sum()
    per_residue = bool_df.sum(axis=0).groupby(level="protein").sum()

    return ProLIFResult(
        boolean_df=df,
        per_residue_counts=per_residue,
        per_type_counts=per_type,
        n_interactions=int(bool_df.values.sum()),
        n_residues_with_interactions=int((per_residue > 0).sum()),
    )


__all__ = ["ProLIFResult", "prolif_fingerprint"]
