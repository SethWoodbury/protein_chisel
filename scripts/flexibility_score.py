"""Per-residue flexibility score via Anisotropic Network Model (ANM).

Cheap, single-purpose, pure-numpy. ~50 lines of substance.
Use: `python flexibility_score.py <design.pdb>` -> per-residue RMSF.

Theory: Build a Cα Hessian with spring constants γ for every Cα-Cα
pair within a cutoff (12 A standard for ANM). Diagonalize, drop the
6 zero modes, weight remaining mode amplitudes by 1/eigenvalue, sum
back to per-residue mean-square fluctuation. Output RMSF in
arbitrary units (relative within design).

Usage in v2:
    from flexibility_score import per_residue_rmsf
    rmsf = per_residue_rmsf(pdb_path, chain="A")
    # rmsf is (L,) ndarray; high values = flexible
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _read_ca_coords(pdb_path: Path, chain: str = "A") -> tuple[list[int], np.ndarray]:
    resnos, coords = [], []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21:22] != chain:
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            resno = int(line[22:26])
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        resnos.append(resno)
        coords.append((x, y, z))
    return resnos, np.array(coords, dtype=np.float64)


def per_residue_rmsf(
    pdb_path: Path | str,
    chain: str = "A",
    cutoff: float = 12.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Anisotropic Network Model per-residue RMSF (arb. units).

    Returns a length-L array; higher = more flexible.
    """
    resnos, coords = _read_ca_coords(Path(pdb_path), chain)
    n = len(coords)
    if n < 3:
        return np.zeros(n)
    # Pairwise distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    contacts = (dist < cutoff) & (dist > 1e-6)
    # Build 3n × 3n Hessian
    H = np.zeros((3 * n, 3 * n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if not contacts[i, j]:
                continue
            r_ij = diff[j, i]                    # vector i->j
            d2 = float(dist[i, j] ** 2)
            outer = gamma * np.outer(r_ij, r_ij) / d2
            H[3*i:3*i+3, 3*j:3*j+3] -= outer
            H[3*j:3*j+3, 3*i:3*i+3] -= outer
            H[3*i:3*i+3, 3*i:3*i+3] += outer
            H[3*j:3*j+3, 3*j:3*j+3] += outer
    # Eigendecompose; drop 6 trivial modes (rigid body)
    eigvals, eigvecs = np.linalg.eigh(H)
    mask = eigvals > 1e-6
    inv_lambdas = np.where(mask, 1.0 / np.maximum(eigvals, 1e-12), 0.0)
    inv_lambdas[:6] = 0.0   # extra safety (sorted ascending)
    # MSF_i = sum_k (1/λ_k) * |eigvec_{k,i}|^2 over the 3 components of i
    weighted = (eigvecs ** 2) * inv_lambdas[None, :]
    msf_xyz = weighted.sum(axis=1)                # (3n,)
    msf = msf_xyz.reshape(n, 3).sum(axis=1)
    return np.sqrt(np.maximum(msf, 0.0))


def rigidity_score(
    pdb_path: Path | str,
    chain: str = "A",
    catalytic_resnos: tuple[int, ...] = (60, 64, 128, 131, 132, 157),
) -> dict:
    """One-line per-design rigidity report.

    Rigidity = -log10(mean RMSF_active_site / mean RMSF_overall).
    Negative = active site MORE flexible than rest (bad for catalysis);
    positive = active site rigid (good).
    """
    resnos, _ = _read_ca_coords(Path(pdb_path), chain)
    rmsf = per_residue_rmsf(pdb_path, chain=chain)
    if len(rmsf) == 0:
        return {"rigidity": 0.0, "rmsf_overall": 0.0, "rmsf_catalytic": 0.0}
    cat_idx = [i for i, r in enumerate(resnos) if r in catalytic_resnos]
    overall = float(rmsf.mean())
    cat = float(rmsf[cat_idx].mean()) if cat_idx else overall
    rigidity = float(-np.log10(max(cat / max(overall, 1e-9), 1e-9)))
    return {
        "rigidity_score": rigidity,
        "rmsf_overall": overall,
        "rmsf_catalytic_mean": cat,
        "rmsf_p95": float(np.percentile(rmsf, 95)),
        "rmsf_max": float(rmsf.max()),
    }


if __name__ == "__main__":
    pdb = Path(sys.argv[1])
    out = rigidity_score(pdb)
    for k, v in out.items():
        print(f"{k:25s} = {v:.4f}")
