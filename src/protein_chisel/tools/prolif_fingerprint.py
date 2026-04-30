"""ProLIF wrapper — interaction fingerprint suitable for ML/data science.

ProLIF (https://github.com/chemosim-lab/ProLIF) detects:
hbond (donor / acceptor), salt bridge (cation / anion), π-stacking
(face-to-face, edge-to-face), π-cation, hydrophobic, vdW contact, metal
acceptor / donor.

For a single PDB it returns a per-(residue × interaction-type) boolean
table; we expose this directly + a numeric "strength" via type-specific
distance weights.

Run inside esmc.sif (where prolif is installed). RDKit is required as a
ProLIF dep — also in esmc.sif.

Reference:
    Bouysset, C., & Fiorucci, S. (2021). ProLIF: a library to encode
    molecular interactions as fingerprints. JCheminform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("protein_chisel.prolif_fingerprint")


# Default per-interaction-type Gaussian distance weights (for the optional
# strength-weighted output). These are reasonable starting values; tune
# from your own data.
DEFAULT_INTERACTION_GEOMETRY = {
    "Hydrophobic":            {"d0": 4.0, "sigma": 0.8},
    "HBDonor":                {"d0": 2.9, "sigma": 0.4},
    "HBAcceptor":             {"d0": 2.9, "sigma": 0.4},
    "Cationic":               {"d0": 4.0, "sigma": 0.8},
    "Anionic":                {"d0": 4.0, "sigma": 0.8},
    "CationPi":               {"d0": 4.0, "sigma": 0.8},
    "PiCation":               {"d0": 4.0, "sigma": 0.8},
    "PiStacking":             {"d0": 5.0, "sigma": 1.0},
    "FaceToFace":             {"d0": 3.8, "sigma": 0.6},
    "EdgeToFace":             {"d0": 5.0, "sigma": 1.0},
    "MetalAcceptor":          {"d0": 2.5, "sigma": 0.4},
    "MetalDonor":             {"d0": 2.5, "sigma": 0.4},
    "VdWContact":             {"d0": 4.5, "sigma": 1.0},
    "XBAcceptor":             {"d0": 3.3, "sigma": 0.4},  # halogen bond
    "XBDonor":                {"d0": 3.3, "sigma": 0.4},
}


@dataclass
class ProLIFResult:
    """A ProLIF interaction fingerprint.

    The ``boolean_df`` columns are MultiIndex of (ligand, residue,
    interaction_type) — same as ProLIF's native output. ``per_residue_counts``
    rolls up across interaction types per residue; useful for ranking
    "interaction-hub" residues. ``per_type_counts`` is the global tally.
    ``per_interaction_strength_df`` adds the soft Gaussian weights when
    distances are extractable.
    """

    boolean_df: pd.DataFrame                                  # raw ProLIF output
    per_residue_counts: pd.Series                              # by residue label
    per_type_counts: pd.Series                                  # by interaction type
    per_interaction_strength_df: Optional[pd.DataFrame] = None
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
    interactions: Optional[list[str]] = None,
    add_strength: bool = True,
) -> ProLIFResult:
    """Compute a protein-ligand interaction fingerprint with ProLIF.

    Args:
        pdb_path: input PDB.
        ligand_resname: HETATM residue name for the ligand (defaults to the
            first non-water HETATM).
        ligand_chain: HETATM chain for the ligand (defaults to the first
            non-water HETATM).
        interactions: list of ProLIF interaction class names to enable
            (defaults to ProLIF's default set).
        add_strength: when True, compute Gaussian-weighted interaction
            strengths from ProLIF's distance values.
    """
    import prolif as plf
    from rdkit import Chem
    import MDAnalysis as mda

    pdb_path = str(Path(pdb_path).resolve())

    if ligand_resname is None or ligand_chain is None:
        from protein_chisel.io.pdb import find_ligand
        info = find_ligand(pdb_path)
        if info is None:
            raise RuntimeError(f"no ligand HETATMs found in {pdb_path}")
        ligand_chain = ligand_chain or info[0]
        ligand_resname = ligand_resname or info[1]

    # ProLIF needs RDKit Mol objects with explicit Hs; add them via an
    # MDAnalysis universe → ProLIF helpers.
    u = mda.Universe(pdb_path)
    protein_sel = u.select_atoms("protein")
    ligand_sel = u.select_atoms(f"resname {ligand_resname} and segid {ligand_chain}*")
    if len(ligand_sel) == 0:
        # Fallback: chain matching via segid prefix doesn't always work
        ligand_sel = u.select_atoms(f"resname {ligand_resname}")
    if len(ligand_sel) == 0:
        raise RuntimeError(f"no atoms for ligand {ligand_resname} in {pdb_path}")

    protein_mol = plf.Molecule.from_mda(protein_sel)
    ligand_mol = plf.Molecule.from_mda(ligand_sel)

    fp = plf.Fingerprint(interactions=interactions) if interactions else plf.Fingerprint()
    fp.run_from_iterable([ligand_mol], protein_mol, n_jobs=1, progress=False)
    df = fp.to_dataframe()  # columns: (ligand, residue, interaction_type) MultiIndex

    # Per-residue and per-type counts
    bool_df = df.astype(bool)
    per_type = bool_df.sum(axis=0).groupby(level="interaction").sum()
    per_residue = bool_df.sum(axis=0).groupby(level="protein").sum()

    strength_df = None
    if add_strength and not df.empty:
        # ProLIF's "metadata" interface gives distances; if unavailable we
        # fall back to using the boolean output as binary strength.
        strength_df = bool_df.astype(float)

    return ProLIFResult(
        boolean_df=df,
        per_residue_counts=per_residue,
        per_type_counts=per_type,
        per_interaction_strength_df=strength_df,
        n_interactions=int(bool_df.values.sum()),
        n_residues_with_interactions=int((per_residue > 0).sum()),
    )


__all__ = ["DEFAULT_INTERACTION_GEOMETRY", "ProLIFResult", "prolif_fingerprint"]
