"""Shape metrics: Rg (proper sqrt-mean-square), length-normalized Rg,
asphericity, acylindricity, relative shape anisotropy, principal lengths.

Replaces process_diffusion3's non-standard `get_ROG` (which returned the
max distance from centroid, not the standard Rg).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from protein_chisel.utils.geometry import ca_coords, shape_descriptors


@dataclass
class ShapeMetricsResult:
    rg: float
    rg_norm: float
    asphericity: float
    acylindricity: float
    rel_shape_anisotropy: float
    principal_length_1: float  # smallest sqrt-eigenvalue
    principal_length_2: float
    principal_length_3: float  # largest
    n_residues: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "shape__rg": self.rg,
            "shape__rg_norm": self.rg_norm,
            "shape__asphericity": self.asphericity,
            "shape__acylindricity": self.acylindricity,
            "shape__rel_shape_anisotropy": self.rel_shape_anisotropy,
            "shape__principal_length_1": self.principal_length_1,
            "shape__principal_length_2": self.principal_length_2,
            "shape__principal_length_3": self.principal_length_3,
            "shape__n_residues": self.n_residues,
        }


def shape_metrics(
    pdb_path: str | Path,
    params: list[str | Path] = (),
    chain_id: str | None = None,
) -> ShapeMetricsResult:
    """Compute shape descriptors on protein CA atoms (excludes ligand)."""
    from protein_chisel.utils.pose import init_pyrosetta, pose_from_file

    init_pyrosetta(params=list(params))
    pose = pose_from_file(pdb_path)

    coords = ca_coords(pose, chain_id=chain_id)
    sd = shape_descriptors(coords)
    return ShapeMetricsResult(
        rg=sd["rg"],
        rg_norm=sd["rg_norm"],
        asphericity=sd["asphericity"],
        acylindricity=sd["acylindricity"],
        rel_shape_anisotropy=sd["rel_shape_anisotropy"],
        principal_length_1=sd["principal_length_1"],
        principal_length_2=sd["principal_length_2"],
        principal_length_3=sd["principal_length_3"],
        n_residues=len(coords),
    )


__all__ = ["ShapeMetricsResult", "shape_metrics"]
