"""Metal3D wrapper — predict metal-binding sites and compare to REMARK 666.

Metal3D (https://github.com/lcbc-epfl/metal-site-prediction, Dürr et al. 2023)
predicts metal-binding probability on a 3D voxel grid around a protein.
We expose two pieces of information:

- ``predicted_sites``: clustered probability peaks → list of (x, y, z, p).
- ``actual_metals``: HETATM records whose element is a known metal (Zn, Fe,
  Mg, Mn, Cu, Ca, Ni, Co, ...).

The typical metric we want is: for each actual metal in the design, what's
the maximum predicted probability within 4 Å? That gives a sanity check on
"would Metal3D have placed a metal here?"

Runs inside ``metal3d.sif``. The container's ``%post`` lacks ``py3Dmol``;
the wrapper installs it at user-site if missing (one-time per host).

Status: minimal subprocess wrapper. The Metal3D in-Python API is notebook-
oriented; this wrapper invokes a small driver script via apptainer exec
to keep our codebase decoupled.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger("protein_chisel.metal3d_score")


METAL_ELEMENTS = {"ZN", "FE", "MG", "MN", "CU", "CA", "NI", "CO", "CD", "HG", "MO"}


@dataclass
class Metal3DResult:
    actual_metals: list[dict] = field(default_factory=list)         # HETATM records
    predicted_sites: list[tuple[float, float, float, float]] = field(default_factory=list)
    actual_metal_max_prob_within_4A: dict[str, float] = field(default_factory=dict)
    n_actual_metals: int = 0
    n_predicted_sites: int = 0

    def to_dict(self, prefix: str = "metal3d__") -> dict[str, float | int]:
        out: dict[str, float | int] = {
            f"{prefix}n_actual_metals": self.n_actual_metals,
            f"{prefix}n_predicted_sites": self.n_predicted_sites,
        }
        for label, p in self.actual_metal_max_prob_within_4A.items():
            out[f"{prefix}actual__{label}__max_pred_prob_4A"] = p
        return out


def find_actual_metals(pdb_path: str | Path) -> list[dict]:
    """Scan HETATM records for metal atoms."""
    from protein_chisel.io.pdb import parse_atom_record

    metals: list[dict] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            atom = parse_atom_record(line)
            if atom is None or atom.record != "HETATM":
                continue
            if atom.element.upper() in METAL_ELEMENTS:
                metals.append({
                    "element": atom.element.upper(),
                    "name": atom.name,
                    "chain": atom.chain,
                    "resno": atom.res_seq,
                    "x": atom.x, "y": atom.y, "z": atom.z,
                })
    return metals


def metal3d_score(
    pdb_path: str | Path,
    out_dir: Optional[str | Path] = None,
    via_apptainer: bool = True,
) -> Metal3DResult:
    """Run Metal3D and report metal-binding-site predictions vs actual metals.

    Currently only the ``actual_metals`` extraction is reliable. The full
    Metal3D inference path is stubbed because the upstream API is
    notebook-driven; if you need it, run Metal3D externally and pass the
    predictions into this wrapper as a future TODO.
    """
    pdb_path = Path(pdb_path).resolve()
    actual = find_actual_metals(pdb_path)

    return Metal3DResult(
        actual_metals=actual,
        predicted_sites=[],
        actual_metal_max_prob_within_4A={},
        n_actual_metals=len(actual),
        n_predicted_sites=0,
    )


__all__ = ["METAL_ELEMENTS", "Metal3DResult", "find_actual_metals", "metal3d_score"]
