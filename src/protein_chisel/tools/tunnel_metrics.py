"""Pocket-accessibility (tunnel patency) scoring for enzyme designs.

Inline-fast (<0.5 s per ~200-residue PDB) tunnel-quality metrics designed
to flag designs whose active-site pocket has GREAT interior geometry but
a CONSTRICTED entrance from bulk solvent — the failure mode that
fpocket's bottleneck_radius and friends miss because they only describe
the pocket interior.

The crucial design choice (per the design-review with codex):

    Atoms are partitioned into three classes based on their fixability
    by the LigandMPNN sequence designer:

        BACKBONE_FIXED  N/CA/C/O of any residue. Backbone is part of the
                        scaffold and CANNOT be moved by sequence design.
        CATALYTIC_SC    side-chain atoms of catalytic residues (those
                        listed in REMARK 666; sidechain identity is
                        pinned by the matcher and CANNOT be redesigned).
        DESIGNABLE_SC   side-chain atoms of all OTHER protein residues.
                        These are the only atoms LigandMPNN can change.

    The 8-column output is split by purpose:

      * HARD GATES (drop the design — no resequence can fix it):
            tunnel__verdict ∈ {buried, ligand_too_big}
            tunnel__backbone_blocked_fraction > 0.5
            tunnel__bottleneck_radius < ligand_min_radius

      * TOPSIS criteria (rank survivors — these ARE actionable):
            maximize tunnel__best_cone_mean_path
            minimize tunnel__sidechain_blocked_fraction
            minimize tunnel__throat_bulky_designable_count

      * Diagnostic (do NOT weight, but useful in reports):
            tunnel__catalytic_blocked_fraction
            tunnel__best_cone_axis_dot_ligand_pa
            tunnel__n_escape_cones

    The split prevents the ranker from being confused by unfixable
    constraints — MPNN only sees what MPNN can act on.

Public API:

    ``score_tunnels(pdb_path, position_table, catres, ligand_radii) -> TunnelScores``

    A pure-Python implementation using NumPy + scipy.spatial only. No
    PyRosetta, no fpocket, no apptainer. Cost ~0.2-0.3 s/PDB.

    For literature-defensible cavity/opening detection, see also
    ``pyKVFinder_score()`` which wraps pyKVFinder in box-mode anchored on
    the catalytic centroid (~50-300 ms/PDB; requires the
    ``universal_with_tunnel_tools.sif`` image at
    ``/net/software/containers/users/woodbuse/``). The two scorers
    measure related but non-identical quantities; running BOTH and
    flagging disagreement is itself signal.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Backbone atoms (fixed by scaffold; cannot be moved by sequence design).
_BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O", "OXT"})


# Heavy-atom van-der-Waals radius used as a uniform clearance for ray-cast
# blockage. 1.7 Å is the standard "carbon" radius and a reasonable mean
# across protein heavy atoms (1.5 N / 1.55 O / 1.7 C / 1.8 S).
_VDW_RADIUS = 1.7


# Mass-weighted "blocker bulkiness" by AA — used in
# ``tunnel__throat_bulky_designable_count``. Codex's recommended weights:
# Trp is the worst offender, aromatic and large hydrophobic next, then
# charged/long. Small residues get a small but nonzero weight so they
# still register if many cluster at the throat.
_BLOCKER_WEIGHT: dict[str, float] = {
    "TRP": 1.00,
    "PHE": 0.85, "TYR": 0.85, "HIS": 0.85,
    "ARG": 0.70, "LYS": 0.70,
    "MET": 0.55, "LEU": 0.55, "ILE": 0.55, "VAL": 0.55,
    "GLN": 0.45, "GLU": 0.45, "ASN": 0.45, "ASP": 0.45,
    "THR": 0.30, "SER": 0.30, "CYS": 0.30, "PRO": 0.30,
    "ALA": 0.20, "GLY": 0.05,
}


@dataclasses.dataclass
class TunnelConfig:
    """Tunable knobs for tunnel scoring. Defaults validated on PTE_i1."""
    n_rays: int = 240
    """Fibonacci-distributed rays cast from ligand centroid."""
    n_cones: int = 12
    """Candidate cone centers used for directional aggregation."""
    cone_half_angle_deg: float = 25.0
    """Half-angle of the angular cone used for directional aggregation."""
    max_ray_length: float = 30.0
    """Cap on per-ray search distance (Å). Rays escaping past this are
    treated as 'open' (free path = max_ray_length)."""
    escape_distance: float = 12.0
    """Minimum free-path length (Å) for a ray to be considered 'escaping
    to bulk solvent'."""
    throat_band_min: float = 3.5
    """Inner edge of the throat-residue shell (sc-min-dist to ligand, Å)."""
    throat_band_max: float = 9.0
    """Outer edge of the throat-residue shell."""
    backbone_blocked_gate: float = 0.5
    """Fraction of best-cone rays first-hitting backbone above which the
    design is hard-gated (scaffold-dead, unfixable)."""
    cone_separation_deg: float = 60.0
    """Minimum angular separation between independent escape cones."""
    cone_mean_path_threshold: float = 8.0
    """Mean ray free-path (Å) inside a cone above which the cone counts
    as a viable escape route for n_escape_cones."""
    burial_mean_path_threshold: float = 4.0
    """If best_cone_mean_path < this AND n_escape_cones == 0 => buried."""


@dataclasses.dataclass
class TunnelScores:
    """Eight metrics + verdict from ``score_tunnels``.

    Attribute names match the column names in topk.tsv (with an added
    ``tunnel__`` prefix at write time).
    """
    # TOPSIS criteria
    best_cone_mean_path: float
    sidechain_blocked_fraction: float
    throat_bulky_designable_count: float

    # HARD GATES
    bottleneck_radius: float
    backbone_blocked_fraction: float

    # Diagnostics
    catalytic_blocked_fraction: float
    best_cone_axis_dot_ligand_pa: float
    n_escape_cones: int

    # Categorical verdict
    verdict: str  # "buried" | "ligand_too_big" | "fixable" | "OK"

    # Auxiliary (not promoted to topk.tsv but useful for plotting)
    best_cone_axis: tuple[float, float, float]
    n_rays_used: int
    elapsed_ms: float

    def to_dict(self, prefix: str = "tunnel__") -> dict:
        """Flat dict ready to merge into a per-design row."""
        out: dict = {
            f"{prefix}best_cone_mean_path": self.best_cone_mean_path,
            f"{prefix}sidechain_blocked_fraction": self.sidechain_blocked_fraction,
            f"{prefix}throat_bulky_designable_count": self.throat_bulky_designable_count,
            f"{prefix}bottleneck_radius": self.bottleneck_radius,
            f"{prefix}backbone_blocked_fraction": self.backbone_blocked_fraction,
            f"{prefix}catalytic_blocked_fraction": self.catalytic_blocked_fraction,
            f"{prefix}best_cone_axis_dot_ligand_pa": self.best_cone_axis_dot_ligand_pa,
            f"{prefix}n_escape_cones": self.n_escape_cones,
            f"{prefix}verdict": self.verdict,
            f"{prefix}elapsed_ms": self.elapsed_ms,
        }
        return out


# ---------------------------------------------------------------------------
# PDB loading + atom-class attribution
# ---------------------------------------------------------------------------


# Atom-class enum (kept as small ints for vectorized ops)
_CLS_LIGAND = 0
_CLS_BACKBONE_FIXED = 1
_CLS_CATALYTIC_SC = 2
_CLS_DESIGNABLE_SC = 3


def _load_atoms(
    pdb_path: str | Path,
    catalytic_resnos: Iterable[int],
    chain: str = "A",
    ligand_chain: Optional[str] = None,
    ligand_resname: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int, str, str]]]:
    """Read a PDB and partition atoms into LIGAND / BB / CAT_SC / DESIGN_SC.

    Args:
        pdb_path: PDB to read.
        catalytic_resnos: int iterable of catalytic motif residue numbers
            on ``chain``. Sidechains of these residues -> CATALYTIC_SC.
        chain: Protein chain ID (default A; PTE_i1 monomer).
        ligand_chain: Optional. If None, auto-detect the largest non-water
            HETATM group as the ligand.
        ligand_resname: Optional. If None, paired auto-detect with chain.

    Returns:
        (coords [N,3] float32,
         classes [N] int8 with values _CLS_LIGAND/.../_CLS_DESIGNABLE_SC,
         per_atom_meta — list of (chain, resno, resname, atom_name) for
         downstream throat-residue selection)
    """
    cat_set = {int(r) for r in catalytic_resnos}

    coords: list[tuple[float, float, float]] = []
    classes: list[int] = []
    meta: list[tuple[str, int, str, str]] = []

    # Auto-detect ligand if needed: largest non-water HETATM group
    het_groups: dict[tuple[str, int, str], int] = {}
    if ligand_chain is None or ligand_resname is None:
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith("HETATM"):
                    continue
                rn = line[17:20].strip()
                if rn in {"HOH", "WAT", "DOD", "TIP", "TIP3"}:
                    continue
                ch = line[21:22]
                try:
                    rno = int(line[22:26])
                except ValueError:
                    continue
                key = (ch, rno, rn)
                het_groups[key] = het_groups.get(key, 0) + 1
        if het_groups:
            (ligand_chain, _, ligand_resname) = max(
                het_groups.keys(), key=lambda k: het_groups[k]
            )

    with open(pdb_path) as fh:
        for line in fh:
            is_atom = line.startswith("ATOM")
            is_het = line.startswith("HETATM")
            if not (is_atom or is_het):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            ch = line[21:22]
            try:
                rno = int(line[22:26])
            except ValueError:
                continue
            element = line[76:78].strip() if len(line) >= 78 else ""

            # Classify
            if is_het and ch == ligand_chain and resname == ligand_resname:
                cls = _CLS_LIGAND
            elif is_het:
                # Other HETATM (e.g. metals, cofactors not in REMARK 666)
                # — treat as backbone-fixed (immovable scaffold furniture)
                cls = _CLS_BACKBONE_FIXED
            elif is_atom and ch == chain:
                if atom_name in _BACKBONE_ATOMS:
                    cls = _CLS_BACKBONE_FIXED
                elif rno in cat_set:
                    cls = _CLS_CATALYTIC_SC
                else:
                    cls = _CLS_DESIGNABLE_SC
            else:
                # Other chain: treat as backbone-fixed (it's not designable
                # by the LigandMPNN call which only redesigns `chain`)
                cls = _CLS_BACKBONE_FIXED

            # Skip hydrogens — VdW geometry is dominated by heavy atoms
            if element == "H" or (not element and atom_name.startswith("H")):
                continue

            coords.append((x, y, z))
            classes.append(cls)
            meta.append((ch, rno, resname, atom_name))

    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(classes, dtype=np.int8),
        meta,
    )


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return n unit vectors approximately uniformly distributed on the
    sphere (Fibonacci/Saff-Kuijlaars spiral). Output shape: (n, 3)."""
    indices = np.arange(0, n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * indices  # golden-angle increment
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _ligand_principal_axis(
    ligand_coords: np.ndarray,
    protein_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroid, principal_axis) for the ligand.

    The principal axis is the largest-eigenvalue PCA direction; we choose
    its sign so the axis points TOWARD the protein-distal end (smaller
    mean distance to all protein atoms = "outside" pointer).
    """
    centroid = ligand_coords.mean(axis=0)
    if len(ligand_coords) < 2:
        # degenerate ligand (single atom) — fall back to centroid - protein_com
        return centroid, _outward_unit_axis(centroid, protein_coords)
    centered = ligand_coords - centroid
    cov = centered.T @ centered / max(1, len(centered) - 1)
    evals, evecs = np.linalg.eigh(cov)
    pa = evecs[:, -1]  # largest eigenvalue
    # Decide sign: project ligand atoms ± along pa, pick the sign whose
    # tip is FARTHEST from the protein (i.e. the outward end).
    plus_tip = centroid + pa * np.linalg.norm(centered, axis=1).max()
    minus_tip = centroid - pa * np.linalg.norm(centered, axis=1).max()
    if (np.linalg.norm(plus_tip - protein_coords.mean(axis=0))
            < np.linalg.norm(minus_tip - protein_coords.mean(axis=0))):
        pa = -pa
    return centroid.astype(np.float32), pa.astype(np.float32)


def _outward_unit_axis(centroid: np.ndarray, protein_coords: np.ndarray) -> np.ndarray:
    """Fallback outward axis: ligand_centroid - protein_com."""
    pcom = protein_coords.mean(axis=0)
    v = centroid - pcom
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (v / n).astype(np.float32)


# ---------------------------------------------------------------------------
# Ray casting (vectorized perpendicular-distance method)
# ---------------------------------------------------------------------------


def _cast_rays(
    origin: np.ndarray,
    directions: np.ndarray,           # (n_rays, 3) unit vectors
    target_coords: np.ndarray,         # (n_atoms, 3) — protein atoms only
    target_classes: np.ndarray,        # (n_atoms,) int8 — class of each
    max_distance: float,
    vdw_radius: float = _VDW_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute first-hit distance + first-hit class for each ray.

    Vectorized: for each (ray, atom) pair, compute (a) atom's projection
    distance onto the ray (positive = ahead of origin), (b) perpendicular
    distance to the ray. Atoms whose perp-distance < vdw_radius and whose
    projection is in (0, max_distance] count as hits; the smallest
    projection is the first hit. Cost: O(n_rays * n_atoms) memory; for
    240 rays × 1500 atoms = 360k floats; ~50 ms.

    Returns:
        hit_distances (n_rays,) — first-hit projection distance, or
            max_distance for rays that escape.
        hit_classes (n_rays,) int8 — class of the first-hit atom, or -1
            for rays that escape without hitting anything.
    """
    n_rays = directions.shape[0]
    n_atoms = target_coords.shape[0]
    if n_atoms == 0:
        return (
            np.full(n_rays, max_distance, dtype=np.float32),
            np.full(n_rays, -1, dtype=np.int8),
        )
    # delta[i,j] = atom_j - origin = (n_atoms, 3); broadcast against rays
    delta = target_coords - origin  # (n_atoms, 3)
    # projection[r, j] = delta[j] . dir[r]   shape (n_rays, n_atoms)
    proj = delta @ directions.T   # (n_atoms, n_rays)
    proj = proj.T  # (n_rays, n_atoms)
    # squared distance from atom j to origin (independent of ray)
    sq_dist = (delta * delta).sum(axis=1)  # (n_atoms,)
    # perpendicular squared distance: |delta|^2 - proj^2
    perp_sq = sq_dist[None, :] - proj * proj
    # Hit condition: perp_sq <= vdw^2 AND proj in (0, max_distance]
    hit_mask = (
        (perp_sq <= vdw_radius * vdw_radius)
        & (proj > 0.0)
        & (proj <= max_distance)
    )
    # For each ray, find the smallest proj where hit_mask is True
    proj_for_min = np.where(hit_mask, proj, np.inf)
    first_idx = np.argmin(proj_for_min, axis=1)            # (n_rays,)
    first_proj = proj_for_min[np.arange(n_rays), first_idx]  # (n_rays,)
    no_hit = ~np.isfinite(first_proj)

    hit_distances = np.where(no_hit, max_distance,
                              first_proj).astype(np.float32)
    hit_classes = np.where(no_hit, -1,
                            target_classes[first_idx]).astype(np.int8)
    return hit_distances, hit_classes


# ---------------------------------------------------------------------------
# Cone aggregation
# ---------------------------------------------------------------------------


def _angular_cone_stats(
    ray_dirs: np.ndarray,
    hit_distances: np.ndarray,
    hit_classes: np.ndarray,
    cone_centers: np.ndarray,         # (n_cones, 3) unit vectors
    cone_half_angle_rad: float,
) -> dict:
    """For each candidate cone center, compute mean ray free-path of rays
    inside the cone + class breakdown of those rays' first hits.

    Returns a dict with:
        per_cone_mean_path: (n_cones,) mean free-path of rays in the cone
        per_cone_n_rays:    (n_cones,) ray count in the cone
        per_cone_bb_frac:   fraction of cone-rays first-hitting BACKBONE
        per_cone_cat_frac:  fraction first-hitting CATALYTIC_SC
        per_cone_dsc_frac:  fraction first-hitting DESIGNABLE_SC
        best_idx:           index of the cone with the largest mean_path
                            (ties broken by closest-to-ligand-PA elsewhere)
    """
    cos_thresh = math.cos(cone_half_angle_rad)
    # cos(angle) between each ray and each cone center
    cos_angles = ray_dirs @ cone_centers.T   # (n_rays, n_cones)
    in_cone = cos_angles >= cos_thresh        # (n_rays, n_cones)

    n_cones = cone_centers.shape[0]
    per_cone_n_rays = in_cone.sum(axis=0)
    # mean_path per cone (NaN if empty)
    masked_paths = np.where(in_cone, hit_distances[:, None], np.nan)
    with np.errstate(invalid="ignore"):
        per_cone_mean_path = np.nanmean(masked_paths, axis=0)
    per_cone_mean_path = np.nan_to_num(per_cone_mean_path, nan=0.0)

    # Class breakdown
    bb_mask = (hit_classes[:, None] == _CLS_BACKBONE_FIXED) & in_cone
    cat_mask = (hit_classes[:, None] == _CLS_CATALYTIC_SC) & in_cone
    dsc_mask = (hit_classes[:, None] == _CLS_DESIGNABLE_SC) & in_cone

    n_safe = np.maximum(per_cone_n_rays, 1)
    bb_frac = bb_mask.sum(axis=0) / n_safe
    cat_frac = cat_mask.sum(axis=0) / n_safe
    dsc_frac = dsc_mask.sum(axis=0) / n_safe

    return {
        "per_cone_mean_path": per_cone_mean_path,
        "per_cone_n_rays": per_cone_n_rays,
        "per_cone_bb_frac": bb_frac,
        "per_cone_cat_frac": cat_frac,
        "per_cone_dsc_frac": dsc_frac,
    }


# ---------------------------------------------------------------------------
# Bottleneck radius along a path
# ---------------------------------------------------------------------------


def _bottleneck_along_axis(
    origin: np.ndarray,
    axis: np.ndarray,
    immovable_coords: np.ndarray,
    t_start: float = 3.0,
    t_end: float = 14.0,
    step: float = 0.5,
    vdw_radius: float = _VDW_RADIUS,
) -> float:
    """Bottleneck (free clearance) at the THROAT only.

    Walks `origin + t*axis` for t in [t_start, t_end]. At each step,
    measures the min perpendicular distance to any immovable atom
    (BACKBONE_FIXED ∪ CATALYTIC_SC). Returns the minimum clearance
    minus vdw_radius (so the result is the free space available for a
    point probe).

    Why a bounded interval: the relevant constriction is at the
    **entrance** (past the ligand surface, before bulk solvent). Going
    out to 30 Å picks up irrelevant backbone collisions deep in the
    bulk along the axis projection. Default [3, 14] Å covers ligand
    surface → throat → first solvent shell.
    """
    if immovable_coords.size == 0:
        return float(t_end)
    n_steps = max(2, int(math.ceil((t_end - t_start) / step)) + 1)
    ts = np.linspace(t_start, t_end, n_steps, dtype=np.float32)
    points = origin + axis[None, :] * ts[:, None]   # (n_steps, 3)
    dists = np.linalg.norm(
        points[:, None, :] - immovable_coords[None, :, :], axis=-1
    )                                                  # (n_steps, n_atoms)
    min_dist_per_step = dists.min(axis=1)              # (n_steps,)
    bottleneck_clearance = float(min_dist_per_step.min()) - vdw_radius
    return max(0.0, bottleneck_clearance)


# ---------------------------------------------------------------------------
# Throat residue analysis
# ---------------------------------------------------------------------------


def _throat_designable_blockers(
    ligand_coords: np.ndarray,
    protein_coords: np.ndarray,
    protein_classes: np.ndarray,
    protein_meta: list[tuple[str, int, str, str]],
    cone_axis: np.ndarray,
    cone_half_angle_rad: float,
    band_min: float,
    band_max: float,
) -> tuple[float, list[tuple[int, str, float]]]:
    """Find designable side-chain residues whose any heavy atom is in
    the throat band (sc-min-dist to ligand in [band_min, band_max]) AND
    inside the best escape cone. Return (mass-weighted count, per-residue
    breakdown).

    Per-residue breakdown is a list of (resno, resname, weight) entries
    that the caller can use for diagnostics or to feed back into the
    bias mechanism (these are the EXACT positions where MPNN should
    swap to a smaller residue to open the throat).
    """
    if len(protein_coords) == 0:
        return 0.0, []
    ligand_centroid = ligand_coords.mean(axis=0)
    cos_thresh = math.cos(cone_half_angle_rad)

    # Filter to designable sc atoms only
    is_dsc = protein_classes == _CLS_DESIGNABLE_SC
    if not is_dsc.any():
        return 0.0, []
    dsc_coords = protein_coords[is_dsc]
    dsc_meta_idx = np.flatnonzero(is_dsc)

    # Atom -> ligand distance; pick atoms in the throat band
    dl = np.linalg.norm(dsc_coords - ligand_centroid, axis=1)
    in_band = (dl >= band_min) & (dl <= band_max)
    if not in_band.any():
        return 0.0, []

    # Atom in cone: project (atom - ligand_centroid) onto cone_axis,
    # check angle to cone_axis
    delta = dsc_coords - ligand_centroid
    norm = np.linalg.norm(delta, axis=1) + 1e-9
    cos_a = (delta @ cone_axis) / norm
    in_cone = cos_a >= cos_thresh

    keep = in_band & in_cone
    if not keep.any():
        return 0.0, []

    # Group by (chain, resno); accumulate one weight per residue
    weights: dict[tuple[str, int], tuple[str, float]] = {}
    for local_idx in np.flatnonzero(keep):
        global_idx = int(dsc_meta_idx[local_idx])
        ch, rno, resn, _atom = protein_meta[global_idx]
        if (ch, rno) in weights:
            continue
        w = _BLOCKER_WEIGHT.get(resn, 0.20)
        weights[(ch, rno)] = (resn, w)
    total_weight = sum(w for _, w in weights.values())
    breakdown = sorted(
        ((rno, resn, w) for ((ch, rno), (resn, w)) in weights.items()),
        key=lambda t: -t[2],
    )
    return float(total_weight), breakdown


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def score_tunnels(
    pdb_path: str | Path,
    catalytic_resnos: Iterable[int],
    *,
    chain: str = "A",
    ligand_chain: Optional[str] = None,
    ligand_resname: Optional[str] = None,
    ligand_min_radius: Optional[float] = None,
    config: Optional[TunnelConfig] = None,
    return_breakdown: bool = False,
) -> TunnelScores | tuple[TunnelScores, list[tuple[int, str, float]]]:
    """Score one design's pocket-accessibility / tunnel patency.

    Args:
        pdb_path: Path to the design PDB.
        catalytic_resnos: int iterable of catalytic motif resnos
            (typically derived from REMARK 666 in the seed PDB and
            propagated through ``DEFAULT_CATRES`` in iterative_design_v2).
        chain: Protein chain ID. Default "A" (PTE_i1 monomer).
        ligand_chain: Ligand chain ID. Auto-detect if None.
        ligand_resname: Ligand 3-letter code. Auto-detect if None.
        ligand_min_radius: Smallest projected radius of the ligand (Å)
            across rotational orientations. Used as the hard-gate
            threshold for ``bottleneck_radius < ligand_min_radius``. If
            None, the ligand-fit gate is skipped.
        config: Tunable knobs.
        return_breakdown: If True, also return the per-residue blocker
            breakdown — list of (resno, resname, mass_weight) tuples for
            designable side chains in the throat band, useful for
            biasing future MPNN cycles to swap those positions.

    Returns:
        TunnelScores, OR (TunnelScores, breakdown) if return_breakdown.
    """
    import time
    t0 = time.perf_counter()
    if config is None:
        config = TunnelConfig()

    # ---- Load + classify atoms -------------------------------------------
    coords_all, classes_all, meta_all = _load_atoms(
        pdb_path,
        catalytic_resnos=catalytic_resnos,
        chain=chain,
        ligand_chain=ligand_chain,
        ligand_resname=ligand_resname,
    )
    if len(coords_all) == 0:
        raise RuntimeError(f"No atoms parsed from {pdb_path}")

    is_lig = classes_all == _CLS_LIGAND
    is_protein = ~is_lig
    if not is_lig.any():
        raise RuntimeError(f"No ligand atoms found in {pdb_path}")
    if not is_protein.any():
        raise RuntimeError(f"No protein atoms found in {pdb_path}")

    ligand_coords = coords_all[is_lig]
    protein_coords = coords_all[is_protein]
    protein_classes = classes_all[is_protein]
    protein_meta = [m for m, p in zip(meta_all, is_protein) if p]

    # ---- Ligand frame -----------------------------------------------------
    centroid, ligand_pa = _ligand_principal_axis(ligand_coords, protein_coords)

    # ---- Ray-cast from centroid (target = protein only; never ligand) ----
    directions = _fibonacci_sphere(config.n_rays)
    hit_dist, hit_class = _cast_rays(
        origin=centroid,
        directions=directions,
        target_coords=protein_coords,
        target_classes=protein_classes,
        max_distance=config.max_ray_length,
    )

    # ---- Cone aggregation -------------------------------------------------
    # Use a Fibonacci sphere of n_cones cone centers; this gives uniform
    # angular coverage. Pick the cone with the largest mean_path and
    # check tie-break against ligand_pa.
    cone_centers = _fibonacci_sphere(config.n_cones)
    cone_stats = _angular_cone_stats(
        ray_dirs=directions,
        hit_distances=hit_dist,
        hit_classes=hit_class,
        cone_centers=cone_centers,
        cone_half_angle_rad=math.radians(config.cone_half_angle_deg),
    )
    per_cone_mean_path = cone_stats["per_cone_mean_path"]
    per_cone_n_rays = cone_stats["per_cone_n_rays"]
    # Pick best cone: prefer cones containing the ligand PA within 60°
    # (tie-break in favor of substrate's natural approach), else the
    # cone with the largest mean path overall.
    pa_dot = cone_centers @ ligand_pa
    pa_aligned = pa_dot >= math.cos(math.radians(60.0))
    if pa_aligned.any():
        scores = np.where(pa_aligned, per_cone_mean_path, -np.inf)
    else:
        scores = per_cone_mean_path
    best_idx = int(np.argmax(scores))

    best_cone_axis = cone_centers[best_idx]
    best_cone_mean_path = float(per_cone_mean_path[best_idx])
    bb_frac = float(cone_stats["per_cone_bb_frac"][best_idx])
    cat_frac = float(cone_stats["per_cone_cat_frac"][best_idx])
    dsc_frac = float(cone_stats["per_cone_dsc_frac"][best_idx])

    # ---- n_escape_cones: how many cones (separated by >cone_separation)
    # have mean_path > cone_mean_path_threshold ---------------------------
    sep_cos = math.cos(math.radians(config.cone_separation_deg))
    viable = per_cone_mean_path >= config.cone_mean_path_threshold
    # Greedy non-overlapping selection: highest mean_path first
    order = np.argsort(-per_cone_mean_path)
    selected: list[int] = []
    for i in order:
        if not viable[i]:
            continue
        if all(np.dot(cone_centers[i], cone_centers[j]) < sep_cos
               for j in selected):
            selected.append(int(i))
    n_escape_cones = len(selected)

    # ---- Bottleneck radius along best cone (immovable atoms only) -------
    immovable = (
        (protein_classes == _CLS_BACKBONE_FIXED)
        | (protein_classes == _CLS_CATALYTIC_SC)
    )
    immovable_coords = protein_coords[immovable]
    bottleneck = _bottleneck_along_axis(
        origin=centroid,
        axis=best_cone_axis,
        immovable_coords=immovable_coords,
    )

    # ---- Throat designable blockers (mass-weighted) ---------------------
    throat_count, breakdown = _throat_designable_blockers(
        ligand_coords=ligand_coords,
        protein_coords=protein_coords,
        protein_classes=protein_classes,
        protein_meta=protein_meta,
        cone_axis=best_cone_axis,
        cone_half_angle_rad=math.radians(config.cone_half_angle_deg),
        band_min=config.throat_band_min,
        band_max=config.throat_band_max,
    )

    # ---- Verdict --------------------------------------------------------
    # Note: the straight-line bottleneck along the best cone axis is
    # NOT reliable for ligand-fit gating because real tunnels CURVE
    # through proteins; a cone axis hits residues every few Å even on a
    # patent tunnel. We therefore do NOT use it as a hard gate here.
    # The ligand-fit gate is enforced downstream via pyKVFinder's
    # cavity_volume / has_opening when that scorer is available.
    if (n_escape_cones == 0 and
            best_cone_mean_path < config.burial_mean_path_threshold):
        verdict = "buried"
    elif bb_frac > config.backbone_blocked_gate:
        verdict = "buried"  # backbone-dominated → unfixable
    else:
        # Distinguish "OK" (well above thresholds) from "fixable" (open
        # but throat needs sequence work)
        if dsc_frac > 0.35 or throat_count >= 3.0:
            verdict = "fixable"
        else:
            verdict = "OK"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    scores = TunnelScores(
        best_cone_mean_path=best_cone_mean_path,
        sidechain_blocked_fraction=dsc_frac,
        throat_bulky_designable_count=float(throat_count),
        bottleneck_radius=float(bottleneck),
        backbone_blocked_fraction=bb_frac,
        catalytic_blocked_fraction=cat_frac,
        best_cone_axis_dot_ligand_pa=float(np.dot(best_cone_axis, ligand_pa)),
        n_escape_cones=n_escape_cones,
        verdict=verdict,
        best_cone_axis=tuple(best_cone_axis.tolist()),
        n_rays_used=config.n_rays,
        elapsed_ms=elapsed_ms,
    )
    if return_breakdown:
        return scores, breakdown
    return scores


# ---------------------------------------------------------------------------
# pyKVFinder wrapper (optional, more accurate cavity detection)
# ---------------------------------------------------------------------------


def pyKVFinder_score(
    pdb_path: str | Path,
    catalytic_resnos: Iterable[int],
    *,
    chain: str = "A",
    box_padding: float = 12.0,
    probe_in: float = 1.4,
    probe_out: float = 4.0,
) -> dict:
    """Run pyKVFinder in box-mode anchored on the catalytic Cα centroid.

    Returns a dict with:
        pkvf__cavity_volume:    largest cavity volume (Å³)
        pkvf__cavity_depth_max: max grid-point depth in the cavity (Å)
        pkvf__n_cavities:       number of detected cavities
        pkvf__has_opening:      bool — does the largest cavity reach
                                bulk solvent? (bool but stored as int)
        pkvf__elapsed_ms:       wall time

    Requires pyKVFinder; raises ImportError if unavailable. Designed to
    run inside ``universal_with_tunnel_tools.sif`` (which has it pip-
    installed). Cost ~50-300 ms/PDB per the survey.
    """
    import time
    t0 = time.perf_counter()
    try:
        import pyKVFinder
    except ImportError:
        raise ImportError(
            "pyKVFinder not installed. Run from "
            "/net/software/containers/users/woodbuse/universal_with_tunnel_tools.sif "
            "or pip install pyKVFinder."
        )

    # Find catalytic Cα centroid (anchor for the box)
    cat_set = {int(r) for r in catalytic_resnos}
    cas: list[tuple[float, float, float]] = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21:22] != chain:
                continue
            try:
                rno = int(line[22:26])
            except ValueError:
                continue
            if rno not in cat_set:
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            cas.append((x, y, z))
    if not cas:
        raise RuntimeError(f"No catalytic CA atoms found for resnos={cat_set}")
    cat_centroid = np.array(cas).mean(axis=0)

    # Box: ±box_padding around the catalytic centroid
    box = (
        (float(cat_centroid[0] - box_padding),
         float(cat_centroid[1] - box_padding),
         float(cat_centroid[2] - box_padding)),
        (float(cat_centroid[0] + box_padding),
         float(cat_centroid[1] + box_padding),
         float(cat_centroid[2] + box_padding)),
    )

    # pyKVFinder 0.9 requires `vertices` (an explicit grid box) for detect()
    atomic = pyKVFinder.read_pdb(str(pdb_path))
    vertices = pyKVFinder.get_vertices(atomic, probe_out=probe_out, step=0.6)
    ncav, cavities = pyKVFinder.detect(
        atomic, vertices,
        probe_in=probe_in, probe_out=probe_out,
    )
    if ncav <= 0:
        return {
            "pkvf__cavity_volume": 0.0,
            "pkvf__cavity_depth_max": 0.0,
            "pkvf__n_cavities": 0,
            "pkvf__has_opening": 0,
            "pkvf__elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }
    # pyKVFinder 0.9 returns:
    #   spatial()  -> (cavity_grid, surface_dict, volume_dict)
    #   depth()    -> (depth_grid, max_depth_dict, avg_depth_dict)
    #   openings() -> (n_openings, openings_grid, areas_dict)
    # We use openings() as the authoritative "is the cavity surface-
    # connected?" signal — depth alone doesn't tell us this since both
    # buried and open cavities have non-zero interior depth.
    spatial_out = pyKVFinder.spatial(cavities)
    volumes = spatial_out[2] if isinstance(spatial_out, tuple) and len(spatial_out) >= 3 else spatial_out
    depth_out = pyKVFinder.depth(cavities)
    max_depths = depth_out[1] if isinstance(depth_out, tuple) and len(depth_out) >= 2 else None
    try:
        openings_out = pyKVFinder.openings(cavities)
        n_openings = int(openings_out[0]) if isinstance(openings_out, tuple) else int(openings_out)
    except Exception as exc:
        LOGGER.debug("pyKVFinder.openings failed (%s); falling back to depth heuristic", exc)
        n_openings = -1  # unknown

    # Largest cavity by volume
    if hasattr(volumes, "items") and len(volumes) > 0:
        items = list(volumes.items())
        largest_key, largest_vol = max(items, key=lambda kv: kv[1])
        if isinstance(max_depths, dict) and largest_key in max_depths:
            depth_max = float(max_depths[largest_key])
        elif isinstance(max_depths, dict) and len(max_depths) > 0:
            depth_max = float(max(max_depths.values()))
        else:
            depth_max = 0.0
    else:
        largest_vol = 0.0
        depth_max = 0.0
    largest_vol = float(largest_vol)
    # has_opening: prefer pyKVFinder.openings() count (authoritative).
    # Fall back to depth-based heuristic if that call failed.
    if n_openings >= 0:
        has_opening = int(n_openings > 0)
    else:
        has_opening = int(depth_max > 0.5)
    return {
        "pkvf__cavity_volume": largest_vol,
        "pkvf__cavity_depth_max": depth_max,
        "pkvf__n_cavities": int(ncav),
        "pkvf__n_openings": int(n_openings) if n_openings >= 0 else -1,
        "pkvf__has_opening": has_opening,
        "pkvf__elapsed_ms": (time.perf_counter() - t0) * 1000.0,
    }


# ----------------------------------------------------------------------------
# Throat-blocker feedback (cycle-to-cycle MPNN bias reinforcement)
# ----------------------------------------------------------------------------


def aggregate_blocker_stats(
    pdb_paths: Iterable[str | Path],
    catalytic_resnos: Iterable[int],
    *,
    chain: str = "A",
    ligand_resname: Optional[str] = None,
    config: Optional[TunnelConfig] = None,
) -> dict[int, dict]:
    """Aggregate per-design throat-blocker breakdowns across many designs.

    For each PDB in ``pdb_paths``, runs ``score_tunnels(return_breakdown=True)``
    and accumulates per-(resno) statistics:

        {
            resno: {
                "n_observed":   int,            # designs with this position blocked
                "total_weight": float,          # sum of mass_weights at this pos
                "avg_weight":   float,          # total_weight / n_designs
                "aa_counts":    {AA: count},    # how often each AA appears here
                "top_aa":       str,            # most-frequent blocker AA
            }
        }

    Args:
        pdb_paths: iterable of design PDBs (typically a cycle's survivors).
        catalytic_resnos: catres to pass to ``score_tunnels``.
        chain: protein chain ID.
        ligand_resname: optional ligand auto-detect.
        config: TunnelConfig (defaults shared with score_tunnels).

    Returns:
        ``{resno: stats_dict}``. Empty dict if no designs scored.
    """
    from collections import Counter, defaultdict

    pdb_paths = [Path(p) for p in pdb_paths]
    if not pdb_paths:
        return {}
    if config is None:
        config = TunnelConfig()

    pos_total: dict[int, float] = defaultdict(float)
    pos_aa_count: dict[int, Counter] = defaultdict(Counter)
    n_designs_scored = 0

    for pdb in pdb_paths:
        try:
            _scores, breakdown = score_tunnels(
                pdb_path=pdb,
                catalytic_resnos=list(catalytic_resnos),
                chain=chain,
                ligand_resname=ligand_resname,
                config=config,
                return_breakdown=True,
            )
        except Exception as exc:
            LOGGER.debug("aggregate_blocker_stats: skipping %s (%s)",
                          pdb.name, exc)
            continue
        n_designs_scored += 1
        for resno, resname, weight in breakdown:
            pos_total[resno] += float(weight)
            pos_aa_count[resno][resname] += 1

    if n_designs_scored == 0:
        return {}

    out: dict[int, dict] = {}
    for resno, total_w in pos_total.items():
        counts = pos_aa_count[resno]
        if not counts:
            continue
        top_aa, _ = counts.most_common(1)[0]
        out[resno] = {
            "n_observed": sum(counts.values()),
            "total_weight": float(total_w),
            "avg_weight": float(total_w / n_designs_scored),
            "aa_counts": dict(counts),
            "top_aa": top_aa,
            "n_designs_scored": n_designs_scored,
        }
    return out


# Mass-weight threshold above which an AA is considered "bulky enough to
# pre-emptively downweight at known throat positions."
_BULKY_THRESHOLD: float = 0.55


def build_throat_bias_delta(
    blocker_stats: dict[int, dict],
    *,
    L: int,
    protein_resnos: Iterable[int],
    fixed_resnos: Iterable[int],
    avg_weight_threshold: float = 0.30,
    base_strength: float = 1.0,
    observed_extra: float = 0.5,
    bulky_threshold: float = _BULKY_THRESHOLD,
    max_total_per_aa: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Convert aggregated blocker stats into a (L, 20) bias delta.

    Algorithm:
      For each position with avg_weight >= threshold:
        For each AA whose blocker mass-weight >= bulky_threshold (i.e. all
        bulky AAs that COULD become blockers there):
          delta[res_idx, aa_idx] -= base_strength * mass_weight(AA)
        For the OBSERVED top blocker AA at that position, additionally:
          delta[res_idx, top_aa_idx] -= observed_extra
        Cap the magnitude per (position, AA) to max_total_per_aa.

      Catalytic / fixed residues are SKIPPED entirely (their identity is
      pinned by REMARK 666).

    Args:
        blocker_stats: output of ``aggregate_blocker_stats``.
        L: protein length (for bias shape).
        protein_resnos: 1-indexed PDB resnos in protein-array order.
        fixed_resnos: catalytic / pinned positions to SKIP.
        avg_weight_threshold: only positions with avg_weight >= this trigger.
        base_strength: nats applied to each bulky AA at the position.
        observed_extra: extra nats applied to the OBSERVED top blocker AA.
        bulky_threshold: minimum mass-weight for an AA to count as bulky.
        max_total_per_aa: cap on the total magnitude at any (pos, AA).

    Returns:
        (delta [L, 20] float32, telemetry dict with positions + AAs touched)
    """
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {a: i for i, a in enumerate(AA_ORDER)}

    delta = np.zeros((L, 20), dtype=np.float32)
    resno_to_idx = {int(r): i for i, r in enumerate(protein_resnos)}
    fixed_set = {int(r) for r in fixed_resnos}

    telem: dict = {
        "n_positions_targeted": 0,
        "positions": [],  # list of {resno, top_aa, avg_weight, n_aas_biased}
    }

    for resno, stats in blocker_stats.items():
        if resno in fixed_set:
            continue
        if stats["avg_weight"] < avg_weight_threshold:
            continue
        idx = resno_to_idx.get(int(resno))
        if idx is None:
            continue

        n_aas_biased = 0
        # Pre-emptive bulky-AA downweight at this position
        for aa_3 in _BLOCKER_WEIGHT:
            mw = _BLOCKER_WEIGHT[aa_3]
            if mw < bulky_threshold:
                continue
            # Convert 3-letter to 1-letter via an inline map
            aa_1 = {
                "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
                "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
                "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
                "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
            }.get(aa_3)
            if aa_1 is None or aa_1 not in aa_to_idx:
                continue
            ai = aa_to_idx[aa_1]
            delta[idx, ai] -= base_strength * mw

        # Extra penalty on the OBSERVED top blocker AA
        top_aa_3 = stats["top_aa"]
        top_aa_1 = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        }.get(top_aa_3)
        if top_aa_1 and top_aa_1 in aa_to_idx:
            delta[idx, aa_to_idx[top_aa_1]] -= observed_extra
            n_aas_biased += 1

        # Cap magnitudes per (pos, AA) — never apply more than max_total_per_aa
        delta[idx] = np.maximum(delta[idx], -max_total_per_aa)

        n_aas_biased = int((delta[idx] < -1e-6).sum())
        telem["positions"].append({
            "resno": int(resno),
            "top_aa": top_aa_3,
            "avg_weight": float(stats["avg_weight"]),
            "n_observed": int(stats["n_observed"]),
            "n_aas_biased": n_aas_biased,
            "max_penalty_nats": float(-delta[idx].min()),
        })

    telem["n_positions_targeted"] = len(telem["positions"])
    return delta, telem
