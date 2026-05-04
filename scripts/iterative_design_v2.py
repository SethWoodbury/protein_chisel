"""Iterative PLM-fusion-driven design loop for the PTE_i1 scaffold.

Runs INSIDE universal.sif (which has fused_mpnn + Bio + pandas + the
host-mounted fpocket binary). Reads precomputed PLM artifacts:

    plm_artifacts/
        esmc_log_probs.npy            (L, 20)
        saprot_log_probs.npy          (L, 20)
        fusion_bias.npy               (L, 20) -- cycle-0 bias
        fusion_log_odds_esmc.npy      (L, 20) -- calibrated log-odds
        fusion_log_odds_saprot.npy    (L, 20)
        fusion_weights.npy            (L, 2)  -- per-pos β, γ
        manifest.json

And a PositionTable (from pyrosetta.sif's classify_positions).

Per-cycle flow:
    sample (LigandMPNN with cycle-k bias)
    -> restore PDBs (REMARK 666 + HETNAM + LINK)
    -> dedup by sequence
    -> cheap seq filter (charge, OmpT, length)
    -> struct filter (h-bond + SAP-proxy)
    -> per-sequence fitness from cached PLM marginals
    -> fpocket scoring on survivors
    -> rank survivors

Across cycles:
    cycle 0:  bias = base PLM-fusion bias
    cycle k+1: bias = base + consensus augmentation from cycle k survivors
                 (only at non-fixed positions of class buried/surface/
                  first_shell/pocket; capped at 30% of L)

Final stage:
    union all cycles' survivors, dedup, pick top_k via greedy
    Hamming-distance diversity, write top_k.fasta + top_k_pdbs/.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("iterative_design_v2")


# ----------------------------------------------------------------------
# Hard-coded constants for the PTE_i1 scaffold
# ----------------------------------------------------------------------

DEFAULT_INPUT_PDB = Path(
    "/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/"
    "ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb"
)
DEFAULT_LIG_PARAMS = Path(
    "/home/woodbuse/testing_space/scaffold_optimization/"
    "ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params/YYE.params"
)

# REMARK 666 catalytic resnos (1-indexed PDB resseq) on chain A.
DEFAULT_CATRES = (60, 64, 128, 131, 132, 157)
CATALYTIC_HIS_RESNOS = (60, 64, 128, 132)
CHAIN = "A"

# Apptainer / cluster paths
UNIVERSAL_SIF = Path("/net/software/containers/universal.sif")
FPOCKET_BIN = Path("/net/software/lab/fpocket/bin/fpocket")

DEFAULT_OUT_ROOT = Path("/net/scratch/woodbuse")

# fused_mpnn checkpoints (same as v1 driver; see notes there)
LMPNN_CKPT = "/net/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt"
SC_CKPT = "/net/databases/mpnn/packer_weights/s_300756.pt"


# ----------------------------------------------------------------------
# Cycle config
# ----------------------------------------------------------------------


AVAILABLE_ENHANCE_CHECKPOINTS: tuple[str, ...] = (
    "plddt_residpo_alpha_20250116-aec4d0c4",
    "plddt_residpo_combine_from_timo_100k_20250905-36329ea5",
    "plddt_preetham_20241018-5cb969e8",
    "plddt_3_20240930-f9c9ea0f",
    "plddt_4_20241003-a358098e",
    "plddt_16_20240910-b65a33eb",
)


@dataclass
class CycleConfig:
    cycle_idx: int
    n_samples: int = 500
    sampling_temperature: float = 0.20
    net_charge_max: float = -10.0     # net_charge_no_HIS strictly < this
    sap_max_threshold: float = 15.0
    consensus_threshold: float = 0.85
    consensus_strength: float = 2.0
    # pI window. With net_charge_no_HIS < -10 the cycle-0 design pI
    # distribution is 4.55-5.35 (WT is 4.58). pi_min=5.0 selects the
    # least-acidic tail (~1% pass cycle 0). Low pass rate is FINE in
    # cycle 0: the consensus mechanism uses survivors to tighten the
    # bias for cycle 1+, pulling subsequent cycles toward less-acidic
    # designs. Aspirational target is 6.5-7.2 from the user's notes
    # (mature B. diminuta PTE-like cores assayed at pH 7.5-8.5); not
    # simultaneously achievable with charge<-10 but iteration tightens.
    pi_min: float = 5.0
    pi_max: float = 7.5
    # fpocket-druggability filter on the active-site pocket. Designs
    # with druggability < this are dropped (no detectable active-site
    # cavity = bad design). Set to 0 to disable.
    fpocket_druggability_min: float = 0.30
    # Heavy-atom clash check. Designs with severe clashes (any heavy-
    # atom pair < 1.5 A between designed sidechain and catalytic /
    # ligand) are dropped. With sc_context=0 MPNN can't see catalytic
    # rotamers and sometimes proposes residues the packer can't fit.
    clash_filter: bool = True
    clash_severe_distance: float = 1.5
    # AAs MPNN may never sample. "X" (UNK) is always omitted by fused_mpnn
    # convention; everything else is opt-in. Default empty -> no AA-omission.
    # The PTE_i1 sbatch passes "CX" to also exclude cysteine (no Cys is
    # catalytic in this scaffold). If your scaffold has a catalytic Cys,
    # pass "" or list only the ones you actually want to forbid.
    omit_AA: str = "X"
    # MPNN flags. Default use_side_chain_context=0 (diverse first-shell
    # sampling). The clash failure mode (Y/F/W at residues with Cb < 5A
    # of catalytic sidechain) is prevented at sample time by
    # compute_clash_prone_first_shell_omits, which auto-forbids bulky
    # AAs at those positions. So sc=0 is now safe.
    use_side_chain_context: int = 0
    enhance: Optional[str] = None


@dataclass
class IterativeRunConfig:
    cycles: list[CycleConfig] = field(default_factory=list)
    target_n_topk: int = 50
    diversity_min_hamming: int = 3
    chain: str = CHAIN
    fix_remark_666: bool = True


def default_cycles(
    omit_AA: str = "X",
    use_side_chain_context: int = 0,
    enhance: Optional[str] = None,
    pi_min: float = 5.0,
    pi_max: float = 7.5,
    fpocket_druggability_min: float = 0.30,
    clash_filter: bool = True,
) -> list[CycleConfig]:
    """Three-cycle exploration → exploitation schedule.

    ``omit_AA`` is the AAs MPNN may never sample.
    ``use_side_chain_context``: 0 (default) for diverse first-shell
    sampling (per ablation); pass 1 to give MPNN the catalytic
    sidechain rotamers (more WT-conservative).
    ``enhance``: optional fused_mpnn enhancer checkpoint name e.g.
    ``"plddt_residpo_alpha_20250116-aec4d0c4"``. Boosts mean fitness
    by ~0.02 nats/residue at a diversity cost.
    """
    # Per-monomer charge schedule. The user's target range is -4 to
    # -15 per monomer, bracketing both the supercharged-PTE paper's
    # -7/monomer (Sukhwal 2024, dimer = -14) and pushing toward more
    # acidic for cycle-2 selection pressure. Schedule: c0 lenient
    # (<-4) -> c1 mid (<-10) -> c2 most restrictive (<-15).
    common = dict(
        omit_AA=omit_AA,
        use_side_chain_context=use_side_chain_context,
        enhance=enhance,
        pi_min=pi_min, pi_max=pi_max,
        fpocket_druggability_min=fpocket_druggability_min,
        clash_filter=clash_filter,
    )
    return [
        CycleConfig(
            cycle_idx=0, n_samples=500, sampling_temperature=0.20,
            net_charge_max=-4.0, sap_max_threshold=15.0, **common,
        ),
        CycleConfig(
            cycle_idx=1, n_samples=400, sampling_temperature=0.18,
            net_charge_max=-10.0, sap_max_threshold=13.0, **common,
        ),
        CycleConfig(
            cycle_idx=2, n_samples=300, sampling_temperature=0.15,
            net_charge_max=-15.0, sap_max_threshold=12.0, **common,
        ),
    ]


# ----------------------------------------------------------------------
# Stage helpers (lifted / adapted from iterative_design_PTE_i1.py)
# ----------------------------------------------------------------------


def compute_catalytic_neighbor_omit_dict(
    *,
    position_table_df,                              # PositionTable.df
    fixed_resnos: Iterable[int],
    chain: str = CHAIN,
    forbid_at_neighbors_of: tuple[str, ...] = ("K", "R"),
    forbid_aas: str = "KR",
) -> dict[str, str]:
    """Build the per-residue omit_AA dict for fused_mpnn.

    For each *fixed* residue whose 1-letter AA is in ``forbid_at_neighbors_of``
    (catalytic K or R by default), forbid ``forbid_aas`` (default "KR") at
    the immediately adjacent protein resnos (resno-1 and resno+1) on the
    same chain. Returns ``{"<chain><resno>": "KR", ...}``.

    Skips neighbors that:
      - aren't on the same chain
      - aren't protein residues
      - are themselves in the fixed/catalytic set (don't constrain catalytic AAs)

    For PTE_i1 with catalytic K157: returns ``{"A156": "KR", "A158": "KR"}``,
    which forbids K and R at PDB resnos 156 and 158 -> no design can put
    a K or R adjacent to the catalytic K157, breaking the unsolvable
    KK-at-157-158 OmpT motif at sample time. Surface residues, no
    catalytic geometry impact.
    """
    fixed_set = set(int(r) for r in fixed_resnos)
    df = position_table_df
    prot = df[(df["is_protein"]) & (df["chain"] == chain)].sort_values("resno")
    resno_to_aa = dict(zip(prot["resno"].astype(int), prot["name1"]))
    resnos_in_chain = set(resno_to_aa.keys())

    out: dict[str, str] = {}
    for r in sorted(fixed_set):
        aa = resno_to_aa.get(r)
        if aa not in forbid_at_neighbors_of:
            continue
        for nb in (r - 1, r + 1):
            if nb in fixed_set:
                continue            # don't constrain another catalytic residue
            if nb not in resnos_in_chain:
                continue            # off-chain or non-protein
            out[f"{chain}{nb}"] = forbid_aas
    return out


def compute_clash_prone_first_shell_omits(
    *,
    seed_pdb: Path,
    position_table_df,
    fixed_resnos: Iterable[int],
    chain: str = CHAIN,
    cb_clearance_threshold: float = 5.0,
    forbid_aas: str = "YFWHM",
    eligible_classes: tuple[str, ...] = ("first_shell", "buried"),
) -> dict[str, str]:
    """Auto-detect first-shell positions where MPNN cannot place bulky
    side-chains without clashing with fixed catalytic atoms.

    Per the side-chain-packer ablation (commit log), every learned and
    rotamer-library packer converges to the same Y/F/W rotamer at
    PDB-resno 35 in PTE_i1 (chi1 ~ g-) which clashes with catalytic E131
    + nearby F135 -- the only non-clashing rotamer is in a low-prior
    chi1 ~ 90-150 deg window that no packer picks. The fix is sample-
    time: forbid Y/F/W at positions whose CB is within
    ``cb_clearance_threshold`` of any fixed-residue sidechain heavy
    atom, so MPNN never proposes them in the first place.

    Returns ``{<chain><resno>: "YFWHM", ...}`` for the
    --omit_AA_per_residue_multi JSON.
    """
    from protein_chisel.structure.clash_check import (
        SIDECHAIN_ATOM_NAMES, _read_atoms,
    )
    fixed_set = set(int(r) for r in fixed_resnos)
    eligible_classes_set = set(eligible_classes)
    df = position_table_df
    prot = df[(df["is_protein"]) & (df["chain"] == chain)].sort_values("resno")
    eligible_resnos = prot[
        (prot["class"].isin(eligible_classes_set))
        & (~prot["resno"].astype(int).isin(fixed_set))
    ]["resno"].astype(int).tolist()

    atoms = _read_atoms(Path(seed_pdb))
    cb_by_resno: dict[int, np.ndarray] = {}
    fixed_sc_atoms: list[np.ndarray] = []
    for a in atoms:
        if a["chain_id"] != chain or a["record"] != "ATOM":
            continue
        if a["res_seq"] in fixed_set:
            sc_names = SIDECHAIN_ATOM_NAMES.get(a["res_name"], set())
            if a["atom_name"] in sc_names:
                fixed_sc_atoms.append(np.array([a["x"], a["y"], a["z"]]))
        elif a["res_seq"] in eligible_resnos and a["atom_name"] == "CB":
            cb_by_resno[a["res_seq"]] = np.array([a["x"], a["y"], a["z"]])

    if not fixed_sc_atoms:
        return {}
    fixed_arr = np.array(fixed_sc_atoms)
    out: dict[str, str] = {}
    for resno, cb in cb_by_resno.items():
        d = np.linalg.norm(fixed_arr - cb, axis=1).min()
        if d < cb_clearance_threshold:
            out[f"{chain}{resno}"] = forbid_aas
    return out


def compute_graded_clash_bias(
    *,
    seed_pdb: Path,
    position_table_df,
    fixed_resnos: Iterable[int],
    chain: str = CHAIN,
    cb_clearance_threshold: float = 5.0,
    eligible_classes: tuple[str, ...] = ("first_shell", "buried"),
    bulky_aas: str = "YFWHMRK",   # K is as long as R (Cb->NZ ~6 A)
    # Per-AA bias = -bias_strength_per_pct_clash * clash_pct.
    # Crude 9-stub rotamer grid produces small clash percentages
    # (typically 0.1-0.3), so we need a high strength to give a
    # meaningful nudge. At strength=20: 20% clash -> -4 nats (firm
    # discouragement, still sample-able when context strongly favors).
    bias_strength_per_pct_clash: float = 20.0,
    rotamer_grid_chi1: tuple[float, ...] = (-60, 60, 180),
    rotamer_grid_chi2: tuple[float, ...] = (-60, 60, 180),
    clash_atom_distance: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Per-residue × per-AA clash-aware bias matrix (L, 20).

    For each (clash-prone position, bulky AA) pair, sample a 9-rotamer
    chi1×chi2 grid, place a tip atom at canonical reach distance, and
    count what fraction of rotamers come within ``clash_atom_distance``
    of any fixed-residue sidechain heavy atom. The bias added to the
    base PLM-fusion bias at that (position, AA) is::

        bias[i, j] -= bias_strength_per_pct_clash * clash_fraction

    Result: positions where Y/F/W literally have no fitting rotamer
    get -3 nats; positions where they fit fine get 0; in-between get
    proportional. Replaces the previous all-or-nothing hard-omit which
    forbade Y/F/W/H/M at every clash-prone position even when they fit.

    Returns (bias_matrix, telemetry_dict). bias_matrix is shape (L, 20)
    in PLM_AA_ORDER ('ACDEFGHIKLMNPQRSTVWY').
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.sampling.plm_fusion import AA_ORDER
    from protein_chisel.structure.clash_check import (
        SIDECHAIN_ATOM_NAMES, _read_atoms,
    )

    AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}
    fixed_set = set(int(r) for r in fixed_resnos)
    eligible_classes_set = set(eligible_classes)

    # Approximate sidechain reach distance (Cb -> tip atom) per AA in Å.
    # Used for the rotamer stub. Crude but consistent across AAs.
    SIDECHAIN_REACH = {
        "Y": 5.5, "F": 5.0, "W": 5.7, "H": 4.5, "M": 4.6, "R": 6.0,
        "K": 5.5, "L": 3.8, "I": 3.6, "Q": 4.5, "N": 3.5, "E": 4.5,
        "D": 3.5, "S": 2.8, "T": 3.0, "V": 2.8, "A": 1.5, "P": 2.5,
        "G": 0.0, "C": 2.8,
    }

    df = position_table_df
    prot = df[(df["is_protein"]) & (df["chain"] == chain)].sort_values("resno")
    eligible_resnos = prot[
        (prot["class"].isin(eligible_classes_set))
        & (~prot["resno"].astype(int).isin(fixed_set))
    ]["resno"].astype(int).tolist()

    atoms = _read_atoms(Path(seed_pdb))
    cb_by_resno: dict[int, np.ndarray] = {}
    ca_by_resno: dict[int, np.ndarray] = {}
    fixed_sc_atoms: list[np.ndarray] = []
    for a in atoms:
        if a["chain_id"] != chain or a["record"] != "ATOM":
            continue
        if a["res_seq"] in fixed_set:
            sc_names = SIDECHAIN_ATOM_NAMES.get(a["res_name"], set())
            if a["atom_name"] in sc_names:
                fixed_sc_atoms.append(np.array([a["x"], a["y"], a["z"]]))
        else:
            if a["atom_name"] == "CB":
                cb_by_resno[a["res_seq"]] = np.array([a["x"], a["y"], a["z"]])
            elif a["atom_name"] == "CA":
                ca_by_resno[a["res_seq"]] = np.array([a["x"], a["y"], a["z"]])

    L = len(prot)
    resno_to_idx = {int(r): i for i, r in enumerate(prot["resno"].astype(int))}
    bias = np.zeros((L, 20), dtype=np.float64)
    telemetry = {"per_position": {}, "n_positions_biased": 0}

    if not fixed_sc_atoms:
        return bias, telemetry
    fixed_arr = np.array(fixed_sc_atoms)

    # Generate stub rotamer tip directions: rotate Cb-Cα vector at chi1,
    # chi2 deltas (just a coarse grid; we only need the direction).
    # The grid is chi1 × chi2 (nine combinations).
    rng = np.random.default_rng(0)
    rot_directions: list[np.ndarray] = []
    for c1 in rotamer_grid_chi1:
        for c2 in rotamer_grid_chi2:
            # Normalized random direction biased by chi1/chi2 angles
            # (this is intentionally coarse — we just want diversity of
            # tip directions; full Dunbrack would be too heavy).
            phi = np.deg2rad(c1)
            psi = np.deg2rad(c2)
            d = np.array([
                np.cos(phi) * np.cos(psi),
                np.sin(phi) * np.cos(psi),
                np.sin(psi),
            ])
            d /= np.linalg.norm(d)
            rot_directions.append(d)

    for resno in eligible_resnos:
        if resno not in cb_by_resno or resno not in ca_by_resno:
            continue
        cb = cb_by_resno[resno]
        ca = ca_by_resno[resno]
        # AA-aware quick reject. The shortest reach in bulky_aas
        # determines the gating distance: if Cb is more than
        # (longest_reach + cb_clearance_threshold) from ANY fixed
        # sidechain, NO bulky AA can clash even with a fully extended
        # rotamer; skip entirely. (Was a flat 5.0 A cutoff regardless
        # of which AA we cared about, which over-flagged short-reach
        # AAs like H at long-reach gating distances.)
        max_reach = max(SIDECHAIN_REACH.get(aa, 0.0) for aa in bulky_aas)
        cb_min = float(np.linalg.norm(fixed_arr - cb, axis=1).min())
        if cb_min >= max_reach + 1.5:    # +1.5 A vdW slop
            continue
        # Build a coordinate frame at Cb
        cb_axis = (cb - ca) / max(np.linalg.norm(cb - ca), 1e-9)
        # An arbitrary perp vector
        perp = np.cross(cb_axis, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-3:
            perp = np.cross(cb_axis, np.array([0.0, 1.0, 0.0]))
        perp /= max(np.linalg.norm(perp), 1e-9)
        perp2 = np.cross(cb_axis, perp)
        perp2 /= max(np.linalg.norm(perp2), 1e-9)

        # Cb->nearest-fixed-atom direction. If Cb-Cα-fixed_atom angle
        # exceeds 110 deg the sidechain points AWAY from the fixed atom;
        # short-reach AAs (H, M) cannot reach over the backbone to
        # clash. We softer-bias those.
        nearest_fixed_idx = int(np.argmin(np.linalg.norm(fixed_arr - cb, axis=1)))
        nearest_fixed = fixed_arr[nearest_fixed_idx]
        cb_to_fixed = nearest_fixed - cb
        cb_to_fixed_norm = cb_to_fixed / max(np.linalg.norm(cb_to_fixed), 1e-9)
        cos_axis_to_fixed = float(np.dot(cb_axis, cb_to_fixed_norm))
        # cos_axis_to_fixed > 0  -> Cb points toward fixed atom (high-clash risk)
        # cos_axis_to_fixed < 0  -> Cb points away (low risk for short reach)
        per_aa_clash_pct = {}
        for aa in bulky_aas:
            reach = SIDECHAIN_REACH.get(aa, 4.5)
            # If Cb points away from fixed and AA reach is short,
            # impossible to reach -> skip entirely.
            if cos_axis_to_fixed < -0.3 and reach < 4.5:
                per_aa_clash_pct[aa] = 0.0
                continue
            n_clash = 0
            n_total = 0
            for d in rot_directions:
                tip = cb + reach * (
                    d[0] * cb_axis + d[1] * perp + d[2] * perp2
                )
                d_min = float(np.linalg.norm(fixed_arr - tip, axis=1).min())
                n_total += 1
                if d_min < clash_atom_distance:
                    n_clash += 1
            pct = n_clash / max(n_total, 1)
            per_aa_clash_pct[aa] = pct
            if pct > 0:
                bias[resno_to_idx[resno], AA_TO_IDX[aa]] -= bias_strength_per_pct_clash * pct
        telemetry["per_position"][resno] = {
            "cb_min_to_fixed": cb_min,
            "per_aa_clash_pct": per_aa_clash_pct,
        }
        if any(p > 0 for p in per_aa_clash_pct.values()):
            telemetry["n_positions_biased"] += 1

    return bias, telemetry


def compute_first_shell_diversity_omits(
    *,
    position_table_df,
    fixed_resnos: Iterable[int],
    chain: str = CHAIN,
    fraction_to_diversify: float = 0.30,
    eligible_classes: tuple[str, ...] = ("first_shell",),
    seed: Optional[int] = None,
) -> dict[str, str]:
    """Build a per-residue omit dict that forbids the WT AA at a random
    subset of first-shell (or other-class) positions.

    Forces MPNN to break out of the WT identity at structurally
    constrained positions where it otherwise just recovers the seed AA.
    Catalytic / fixed positions are skipped.

    For PTE_i1 with 14 first_shell positions and fraction=0.30:
        ~4 random positions/cycle have their WT identity forbidden,
        so each cycle explores a different non-WT axis at a different
        subset of first-shell positions.
    """
    rng = np.random.default_rng(seed)
    fixed_set = set(int(r) for r in fixed_resnos)
    df = position_table_df
    prot = df[(df["is_protein"]) & (df["chain"] == chain)].sort_values("resno")
    eligible = prot[
        (prot["class"].isin(eligible_classes))
        & (~prot["resno"].astype(int).isin(fixed_set))
    ]
    n = max(1, int(len(eligible) * fraction_to_diversify))
    picks = rng.choice(len(eligible), size=min(n, len(eligible)), replace=False)
    out: dict[str, str] = {}
    for i in picks:
        row = eligible.iloc[int(i)]
        out[f"{chain}{int(row['resno'])}"] = str(row["name1"])
    return out


def merge_omit_dicts(*dicts: dict[str, str]) -> dict[str, str]:
    """Union AAs to forbid across multiple per-residue omit dicts."""
    out: dict[str, str] = {}
    for d in dicts:
        for k, aas in d.items():
            cur = set(out.get(k, ""))
            cur.update(aas)
            out[k] = "".join(sorted(cur))
    return out


def _parse_remark_block(ref_pdb: Path) -> list[str]:
    """Pull REMARK 666 / HETNAM / LINK / REMARK PDBinfo-LABEL from ref.

    Thin wrapper over ``protein_chisel.tools.pdb_restoration.extract_remark_lines``
    kept for back-compat; new code should call the tool directly.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.pdb_restoration import extract_remark_lines
    return extract_remark_lines(ref_pdb)


def stage_sample(
    *,
    cycle_cfg: CycleConfig,
    seed_pdb: Path,
    bias: np.ndarray,
    protein_resnos: list[int],
    fixed_resnos: Iterable[int],
    out_dir: Path,
    chain: str = CHAIN,
    omit_AA_per_residue: Optional[dict[str, str]] = None,
) -> Path:
    """Sample N candidates via LigandMPNN with the given (L,20) bias.

    ``omit_AA_per_residue`` is the fused_mpnn ``--omit_AA_per_residue_multi``
    payload: ``{"<chain><resno>": "AAs_to_forbid"}``. For PTE_i1 we set
    {"A156": "KR", "A158": "KR"} so MPNN can't pick K/R adjacent to the
    catalytic K157 (otherwise it forces the KK-OmpT motif at sample time).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.ligand_mpnn import (
        LigandMPNNConfig, sample_with_ligand_mpnn,
    )

    extra_flags = [
        "--checkpoint_ligand_mpnn", LMPNN_CKPT,
        "--checkpoint_path_sc", SC_CKPT,
    ]
    if omit_AA_per_residue:
        # Build the multi-format JSON: {"<pdb_path>": {"<chain><resno>": "AAs"}}
        omit_path = out_dir / "omit_AA_per_residue.json"
        omit_path.write_text(json.dumps({
            str(Path(seed_pdb).resolve()): omit_AA_per_residue
        }, indent=2))
        extra_flags += ["--omit_AA_per_residue_multi", str(omit_path)]
        LOGGER.info("stage_sample[cycle=%d]: omit_AA_per_residue=%s",
                     cycle_cfg.cycle_idx, omit_AA_per_residue)

    cfg = LigandMPNNConfig(
        temperature=cycle_cfg.sampling_temperature,
        batch_size=10,
        repack_everything=0,        # CRITICAL: keep catalytic rotamers intact
        pack_side_chains=1,         # write packed PDBs
        ligand_mpnn_use_side_chain_context=cycle_cfg.use_side_chain_context,
        enhance=cycle_cfg.enhance,
        omit_AA=cycle_cfg.omit_AA,
        # No global bias_AA -- per-residue bias is the whole point.
        extra_flags=tuple(extra_flags),
    )

    LOGGER.info(
        "stage_sample[cycle=%d]: n=%d, T=%.3f, fixed=%s, mean_abs_bias=%.4f",
        cycle_cfg.cycle_idx, cycle_cfg.n_samples,
        cycle_cfg.sampling_temperature, sorted(set(fixed_resnos)),
        float(np.abs(bias).mean()),
    )

    res = sample_with_ligand_mpnn(
        pdb_path=seed_pdb,
        chain=chain,
        fixed_resnos=sorted(set(fixed_resnos)),
        bias_per_residue=bias,
        protein_resnos=protein_resnos,
        n_samples=cycle_cfg.n_samples,
        config=cfg,
        out_dir=out_dir,
        parent_design_id=f"PTE_i1_c{cycle_cfg.cycle_idx}",
        via_apptainer=False,         # already inside universal.sif
    )
    cand_fasta = out_dir / "candidates.fasta"
    cand_tsv = out_dir / "candidates.tsv"
    res.candidate_set.to_disk(cand_fasta, cand_tsv)
    LOGGER.info("stage_sample[cycle=%d]: produced %d rows (incl. WT input header)",
                 cycle_cfg.cycle_idx, len(res.candidate_set.df))
    return cand_tsv


def stage_restore_pdbs(
    *,
    sample_dir: Path,
    ref_pdb: Path,
    out_pdb_dir: Path,
    pdb_basename: str,
    candidate_ids: list[str],
    chain: str = CHAIN,
    catalytic_resnos: Iterable[int] = DEFAULT_CATRES,
    catalytic_hydrogens: bool = True,
) -> dict[str, Path]:
    """Restore REMARK 666 / HETNAM / LINK + HIS tautomer (HIS_D / HIE / HIP)
    + KCX cap atoms + catalytic hydrogens onto the packed MPNN PDBs.

    Delegates to ``protein_chisel.tools.pdb_restoration.restore_sample_dir``;
    see that module for the full restoration semantics. The signature is
    kept stable so call sites in this driver are unchanged.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.pdb_restoration import restore_sample_dir
    LOGGER.info(
        "stage_restore_pdbs: restoring header+tautomers for %d candidates",
        len(candidate_ids),
    )
    return restore_sample_dir(
        sample_dir=sample_dir,
        ref_pdb=ref_pdb,
        out_pdb_dir=out_pdb_dir,
        pdb_basename=pdb_basename,
        candidate_ids=candidate_ids,
        chain=chain,
        catalytic_resnos=catalytic_resnos,
        catalytic_hydrogens=catalytic_hydrogens,
    )


OMPT_ONLY_PATTERNS = [
    ("ompT_KK", r"KK"), ("ompT_KR", r"KR"),
    ("ompT_RK", r"RK"), ("ompT_RR", r"RR"),
]


def _count_ompt_motifs(sequence: str) -> int:
    """Count OmpT-class dibasic motifs (KK/KR/RK/RR) anywhere in seq."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.filters.protease_sites import find_protease_sites
    return len(find_protease_sites(
        sequence, extra_patterns=OMPT_ONLY_PATTERNS, skip_default=True,
    ).hits)


def stage_seq_filter(
    *,
    candidates_tsv: Path,
    out_dir: Path,
    net_charge_max: float,
    wt_length: int,
    expression_engine,                # ExpressionRuleEngine
    seed_ss_reduced: Optional[str] = None,
    seed_sasa: Optional[np.ndarray] = None,
    seed_position_class: Optional[list[str]] = None,
    seed_protein_resnos: Optional[list[int]] = None,
    catalytic_resnos: Iterable[int] = (),
    fixed_resnos: Iterable[int] = (),
    pi_min: float = 0.0,
    pi_max: float = 14.0,
) -> Path:
    """Cheap sequence-only filter: charge, length, expression-rule HARD_FILTERs.

    The expression engine encodes all known E. coli expression risks
    (ssrA, signal peptides, AMP-like, hydrophobic C-tails, etc.). Per-
    sequence: any HARD_FILTER hit rejects the sequence; SOFT_BIAS and
    HARD_OMIT hits are recorded in metadata and applied at MPNN sample
    time in the next cycle (see ``stage_sample``).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.filters.protparam import protparam_metrics
    from protein_chisel.sampling.fitness_score import deduplicate_by_sequence

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(candidates_tsv, sep="\t")
    if "is_input" in df.columns:
        df = df[~df["is_input"].astype(bool)].copy()
    df = deduplicate_by_sequence(df)
    LOGGER.info("stage_seq_filter: input n=%d (post-dedup, post-WT-drop)", len(df))

    rows: list[dict] = []
    for _, row in df.iterrows():
        seq = row["sequence"]
        reasons: list[str] = []
        if len(seq) != wt_length:
            reasons.append(f"length {len(seq)} != WT {wt_length}")
        pp = protparam_metrics(seq)
        if pp.charge_at_pH7_no_HIS >= net_charge_max:
            reasons.append(
                f"net_charge_no_HIS={pp.charge_at_pH7_no_HIS:.2f} >= {net_charge_max}"
            )
        if not (pi_min <= pp.pi <= pi_max):
            reasons.append(f"pI={pp.pi:.2f} outside [{pi_min}, {pi_max}]")

        # Run the full expression-rule engine (covers OmpT, ssrA, Tat, signal
        # peptides, AMP-like, polyproline, SecM, cytosolic disulfides, and
        # tag-protease internal sites). Structure-aware rules use the SEED
        # PDB's SS / SASA / class -- designs share the seed backbone.
        eng_res = expression_engine.evaluate(
            seq,
            ss_reduced=seed_ss_reduced,
            sasa=seed_sasa,
            position_class=seed_position_class,
            catalytic_resnos=catalytic_resnos,
            fixed_resnos=fixed_resnos,
            protein_resnos=seed_protein_resnos,
        )
        for h in eng_res.hard_filter_hits:
            reasons.append(f"{h.rule_name}: {h.reason}")
        n_warnings = len(eng_res.warnings)
        n_soft_bias = len(eng_res.soft_bias_hits)
        n_hard_omit = len(eng_res.hard_omit_hits)

        rows.append({
            **row.to_dict(),
            "length": len(seq),
            "net_charge_no_HIS": pp.charge_at_pH7_no_HIS,
            "instability_index": pp.instability_index,
            "gravy": pp.gravy,
            "pi": pp.pi,
            "n_expression_warnings": n_warnings,
            "n_expression_soft_bias_hits": n_soft_bias,
            "n_expression_hard_omit_hits": n_hard_omit,
            "n_expression_hard_filter_hits": len(eng_res.hard_filter_hits),
            "expression_rule_summary": ";".join(
                f"{h.rule_name}={h.severity.name}" for h in eng_res.hits
            ),
            "passed_seq_filter": not reasons,
            "fail_reasons": "; ".join(reasons),
        })

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0:
        survivors = out_df[out_df["passed_seq_filter"]].copy()
        rejects = out_df[~out_df["passed_seq_filter"]].copy()
    else:
        survivors = out_df.copy()
        rejects = out_df.copy()
    survivors.to_csv(out_dir / "survivors_seq.tsv", sep="\t", index=False)
    rejects.to_csv(out_dir / "rejects_seq.tsv", sep="\t", index=False)
    LOGGER.info(
        "stage_seq_filter: %d / %d passed (charge<%.0f + expression rules)",
        len(survivors), len(out_df), net_charge_max,
    )
    return out_dir / "survivors_seq.tsv"


def _read_atoms(pdb_path: Path) -> list[dict]:
    """Minimal stdlib PDB ATOM/HETATM parser, format-aware.

    Reads res_name from cols 16-20 (overlaps altloc col 16). In standard
    PDB col 16 is a space + cols 17-19 are the 3-char res_name, so
    line[16:21].strip() returns the canonical 3-char name. In Rosetta-
    extended PDB the 5-char form (e.g. "HIS_D") fills cols 16-20 and
    line[16:21].strip() returns "HIS_D". Chain is at col 21 in both.
    """
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                atoms.append({
                    "record": line[:6].strip(),
                    "atom_name": line[12:16].strip(),
                    "res_name": line[16:21].strip(),
                    "chain_id": line[21].strip(),
                    "res_seq": int(line[22:26].strip() or 0),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "element": line[76:78].strip(),
                })
            except (ValueError, IndexError):
                continue
    return atoms


def _detect_hbond_to_ligand(
    pdb_path: Path,
    chain: str = CHAIN,
    distance_cutoff: float = 3.5,
) -> list[dict]:
    """Heavy-atom geometric H-bond detection: any HETATM atom (N/O/S)
    within ``distance_cutoff`` of any protein donor/acceptor heavy atom.

    Returned hits exclude trivial covalent contacts (within same residue).
    """
    atoms = _read_atoms(pdb_path)
    DA_NAMES = {
        "N", "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
        "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "ND1", "SG",
    }
    protein = [
        a for a in atoms
        if (a["chain_id"] == chain and a["atom_name"] in DA_NAMES
            and a["element"] in ("N", "O", "S"))
    ]
    ligand = [
        a for a in atoms
        if (a["record"] == "HETATM" and a["element"] in ("N", "O", "S"))
    ]
    hits = []
    for L in ligand:
        for p in protein:
            d = ((L["x"]-p["x"])**2 + (L["y"]-p["y"])**2 + (L["z"]-p["z"])**2)**0.5
            if d <= distance_cutoff:
                hits.append({
                    "ligand_resname": L["res_name"], "ligand_atom": L["atom_name"],
                    "protein_resno": p["res_seq"], "protein_atom": p["atom_name"],
                    "protein_resname": p["res_name"], "distance": round(d, 3),
                })
    return hits


def _detect_ligand_contacts(
    pdb_path: Path,
    chain: str = CHAIN,
    hbond_cutoff: float = 3.5,
    salt_bridge_cutoff: float = 4.5,
    aromatic_cutoff: float = 5.5,
    hydrophobic_cutoff: float = 5.0,
) -> dict:
    """Cheap geometric detection of common protein↔ligand interactions.

    All checks are heavy-atom distance only (no hydrogens, no angle
    geometry). Sub-millisecond per design. Categories:

      n_hbonds:        protein N/O/S ↔ ligand N/O/S (<= hbond_cutoff)
      n_salt_bridges:  protein K/R sidechain N ↔ ligand O (<= salt_bridge_cutoff)
                    + protein D/E sidechain O ↔ ligand N (<= salt_bridge_cutoff)
      n_aromatic:      protein F/W/Y/H aromatic atom ↔ ligand C-aromatic
                    (<= aromatic_cutoff, no plane-angle check)
      n_hydrophobic:   protein A/V/L/I/M/F/W/Y/C ↔ ligand C
                    (<= hydrophobic_cutoff)
      n_total:         sum

    Reported as metrics; not used as filters by default.
    """
    atoms = _read_atoms(pdb_path)
    DA_NAMES = {"N","O","OD1","OD2","OE1","OE2","OG","OG1","OH",
                "ND2","NE","NE1","NE2","NH1","NH2","NZ","ND1","SG"}
    AROMATIC_NAMES_BY_RES = {
        "PHE": {"CG","CD1","CD2","CE1","CE2","CZ"},
        "TYR": {"CG","CD1","CD2","CE1","CE2","CZ"},
        "TRP": {"CG","CD1","CD2","CE2","CE3","NE1","CZ2","CZ3","CH2"},
        "HIS": {"CG","ND1","CE1","NE2","CD2"},
        "HID": {"CG","ND1","CE1","NE2","CD2"},
        "HIE": {"CG","ND1","CE1","NE2","CD2"},
        "HIP": {"CG","ND1","CE1","NE2","CD2"},
        "HIS_D": {"CG","ND1","CE1","NE2","CD2"},
    }
    HYDROPHOBIC_RESIDUES = {"ALA","VAL","LEU","ILE","MET","PHE","TRP","TYR","CYS"}
    BASIC_NS = {"NZ", "NH1", "NH2"}        # K NZ; R NH1/NH2
    ACIDIC_OS = {"OD1", "OD2", "OE1", "OE2"}

    protein = [a for a in atoms if (a["chain_id"] == chain and a["record"] == "ATOM")]
    ligand = [a for a in atoms if a["record"] == "HETATM"]
    if not ligand or not protein:
        return {"n_hbonds": 0, "n_salt_bridges": 0, "n_aromatic": 0,
                "n_hydrophobic": 0, "n_total": 0}

    n_hb = n_sb = n_aro = n_hyd = 0
    hb2 = hbond_cutoff ** 2
    sb2 = salt_bridge_cutoff ** 2
    ar2 = aromatic_cutoff ** 2
    hy2 = hydrophobic_cutoff ** 2
    for L in ligand:
        Lx, Ly, Lz = L["x"], L["y"], L["z"]
        Lel = L["element"]
        for p in protein:
            dx = p["x"] - Lx; dy = p["y"] - Ly; dz = p["z"] - Lz
            r2 = dx*dx + dy*dy + dz*dz
            if r2 > ar2 and r2 > hy2:
                continue
            # H-bond
            if (p["atom_name"] in DA_NAMES and p["element"] in ("N","O","S")
                    and Lel in ("N","O","S") and r2 <= hb2):
                n_hb += 1
            # Salt bridge K/R-N ↔ ligand O   OR   D/E-O ↔ ligand N
            if r2 <= sb2:
                if p["atom_name"] in BASIC_NS and Lel == "O":
                    n_sb += 1
                elif p["atom_name"] in ACIDIC_OS and Lel == "N":
                    n_sb += 1
            # Aromatic
            if (p["res_name"] in AROMATIC_NAMES_BY_RES
                    and p["atom_name"] in AROMATIC_NAMES_BY_RES[p["res_name"]]
                    and Lel == "C" and r2 <= ar2):
                n_aro += 1
            # Hydrophobic C-C
            if (p["res_name"] in HYDROPHOBIC_RESIDUES
                    and p["element"] == "C" and Lel == "C" and r2 <= hy2):
                n_hyd += 1
    return {
        "n_hbonds": n_hb, "n_salt_bridges": n_sb,
        "n_aromatic": n_aro, "n_hydrophobic": n_hyd,
        "n_total": n_hb + n_sb + n_aro + n_hyd,
    }


def _detect_hbond_to_his_sidechain(
    pdb_path: Path,
    catalytic_his_resnos: Iterable[int],
    chain: str = CHAIN,
    distance_cutoff: float = 3.5,
) -> list[dict]:
    """Heavy-atom geometric H-bond detection: catalytic HIS NE2/ND1 ↔ any
    non-self protein donor/acceptor atom (N, O, S) within 3.5 Å."""
    atoms = _read_atoms(pdb_path)
    cat_set = set(catalytic_his_resnos)
    his_targets = [
        a for a in atoms
        if (a["chain_id"] == chain and a["res_seq"] in cat_set
            and a["res_name"] in ("HIS", "HIS_D", "HIP")
            and a["atom_name"] in ("ND1", "NE2"))
    ]
    DA_NAMES = {
        "N", "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
        "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "ND1", "SG",
    }
    cands = [
        a for a in atoms
        if (a["chain_id"] == chain and a["atom_name"] in DA_NAMES
            and a["element"] in ("N", "O", "S")
            and not (a["res_seq"] in cat_set
                     and a["atom_name"] in ("ND1", "NE2")))
    ]
    hits = []
    for h in his_targets:
        for c in cands:
            if abs(c["res_seq"] - h["res_seq"]) <= 1:
                continue
            d = ((h["x"]-c["x"])**2 + (h["y"]-c["y"])**2 + (h["z"]-c["z"])**2)**0.5
            if d <= distance_cutoff:
                hits.append({
                    "his_resno": h["res_seq"], "his_atom": h["atom_name"],
                    "partner_resno": c["res_seq"], "partner_atom": c["atom_name"],
                    "partner_resname": c["res_name"], "distance": round(d, 3),
                })
    return hits


# Kyte-Doolittle hydrophobicity + Tien max-SASA — same as v1 driver.
KD_HYDROPHOBICITY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
SASA_MAX_RESIDUE = {
    "A": 121, "C": 148, "D": 187, "E": 214, "F": 228, "G": 97,  "H": 216,
    "I": 195, "K": 230, "L": 191, "M": 203, "N": 187, "P": 154, "Q": 214,
    "R": 265, "S": 143, "T": 163, "V": 165, "W": 264, "Y": 255,
}
THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","HID":"H","HIE":"H","HIP":"H","HIS_D":"H","ILE":"I",
    "LEU":"L","LYS":"K","KCX":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}


def _compute_sap_proxy(pdb_path: Path) -> Optional[dict]:
    """Lauer-style SAP via freesasa SASA + Kyte-Doolittle hydrophobicity.

    Per-residue SAP_i = sum over atoms within 10 Å of CA(i):
        (SASA(j) / SASA_max(j)) * KD(restype(j))
    """
    try:
        import freesasa
    except ImportError:
        return None
    try:
        freesasa.setVerbosity(freesasa.silent)
        struct = freesasa.Structure(str(pdb_path))
        result = freesasa.calc(struct)
    except Exception as e:
        LOGGER.warning("freesasa load failed for %s: %s", pdb_path.name, e)
        return None

    n_atoms = struct.nAtoms()
    res_data: dict[tuple[str, int], dict] = {}
    for i in range(n_atoms):
        chain_id = struct.chainLabel(i)
        if chain_id != CHAIN:
            continue
        rn = struct.residueNumber(i).strip()
        try:
            res_seq = int(rn)
        except ValueError:
            continue
        rname = struct.residueName(i).strip()
        atom = struct.atomName(i).strip()
        x, y, z = struct.coord(i)
        sasa_i = result.atomArea(i)
        key = (chain_id, res_seq)
        if key not in res_data:
            res_data[key] = {"resname": rname, "ca": None, "sasa": 0.0, "atoms_xyz": []}
        res_data[key]["sasa"] += sasa_i
        res_data[key]["atoms_xyz"].append((x, y, z))
        if atom == "CA":
            res_data[key]["ca"] = (x, y, z)

    if not res_data:
        return None
    keys_sorted = sorted(res_data.keys(), key=lambda k: k[1])
    cas = []
    for k in keys_sorted:
        ca = res_data[k]["ca"]
        if ca is None:
            ca = res_data[k]["atoms_xyz"][0] if res_data[k]["atoms_xyz"] else (0, 0, 0)
        cas.append(ca)
    cas_a = np.array(cas, dtype=float)

    sap_per_res = []
    for i, k_i in enumerate(keys_sorted):
        d = np.linalg.norm(cas_a - cas_a[i], axis=1)
        nbrs = np.where(d <= 10.0)[0]
        s = 0.0
        for j in nbrs:
            rj = res_data[keys_sorted[j]]["resname"]
            aa = THREE_TO_ONE.get(rj)
            if aa is None:
                continue
            sasa_j = res_data[keys_sorted[j]]["sasa"]
            sasa_max = SASA_MAX_RESIDUE.get(aa, 200.0)
            s += (sasa_j / sasa_max) * KD_HYDROPHOBICITY.get(aa, 0.0)
        sap_per_res.append(s)

    arr = np.array(sap_per_res)
    return {
        "sap_max": float(np.max(arr)),
        "sap_mean": float(np.mean(arr)),
        "sap_p95": float(np.percentile(arr, 95)),
    }


def stage_struct_filter(
    *,
    survivors_seq_tsv: Path,
    pdb_map: dict[str, Path],
    out_dir: Path,
    sap_max_threshold: float,
    catalytic_his_resnos: Iterable[int] = CATALYTIC_HIS_RESNOS,
    fixed_resnos: Iterable[int] = (),
    clash_filter: bool = True,
    clash_severe_distance: float = 1.5,
) -> Path:
    """Apply h-bond + SAP-proxy structural filter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(survivors_seq_tsv, sep="\t")
    LOGGER.info("stage_struct_filter: input n=%d", len(df))

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.structure import detect_clashes

    rows: list[dict] = []
    hbond_rows: list[dict] = []
    for _, row in df.iterrows():
        cid = row["id"]
        pdb = pdb_map.get(cid)
        if pdb is None or not pdb.is_file():
            rows.append({**row.to_dict(),
                         "n_hbonds_to_cat_his": 0,
                         "sap_max": float("nan"),
                         "sap_mean": float("nan"),
                         "sap_p95": float("nan"),
                         "clash__n_total": 0,
                         "clash__n_to_catalytic": 0,
                         "clash__n_to_ligand": 0,
                         "clash__has_severe": 0,
                         "passed_struct_filter": False,
                         "struct_fail": f"pdb_missing: {pdb}"})
            continue
        hbonds = _detect_hbond_to_his_sidechain(pdb, catalytic_his_resnos)
        for h in hbonds:
            hbond_rows.append({"id": cid, **h})
        # Cheap geometric protein<->ligand interaction panel via the
        # generalizable tools.geometric_interactions module: H-bonds
        # with antecedent-angle check, donor/acceptor type-aware salt
        # bridges, π-π with plane angle, π-cation, hydrophobic, and
        # vdW-clash detection -- each with a Gaussian strength score.
        # ~50 ms/design. Replaces the previous embedded helper.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from protein_chisel.tools.geometric_interactions import detect_interactions
        panel = detect_interactions(pdb, chain=CHAIN, selection="protein_vs_ligand")
        gi_metrics = panel.to_dict("ligand_int__")
        sap = _compute_sap_proxy(pdb) or {}
        sap_max = sap.get("sap_max", float("nan"))
        # Clash detection (catalytic + ligand vs designed sidechains)
        clash = detect_clashes(
            pdb, catalytic_resnos=fixed_resnos, chain=CHAIN,
            severe_distance=clash_severe_distance,
        )
        clash_dict = clash.to_dict()
        reasons: list[str] = []
        if len(hbonds) < 1:
            reasons.append("no h-bonds to catalytic HIS")
        if sap_max == sap_max and sap_max > sap_max_threshold:
            reasons.append(f"sap_max={sap_max:.2f} > {sap_max_threshold}")
        if clash_filter and clash.has_severe_clash:
            reasons.append(
                f"severe clash (n_cat={clash.clashes_to_catalytic}, "
                f"n_lig={clash.clashes_to_ligand}, "
                f"detail={clash_dict['clash__detail']})"
            )
        rows.append({**row.to_dict(),
                     "n_hbonds_to_cat_his": len(hbonds),
                     **gi_metrics,
                     "sap_max": sap_max,
                     "sap_mean": sap.get("sap_mean", float("nan")),
                     "sap_p95": sap.get("sap_p95", float("nan")),
                     **clash_dict,
                     "passed_struct_filter": not reasons,
                     "struct_fail": "; ".join(reasons)})

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0 and "passed_struct_filter" in out_df.columns:
        survivors = out_df[out_df["passed_struct_filter"]].copy()
        rejects = out_df[~out_df["passed_struct_filter"]].copy()
    else:
        # Empty input -> empty output with the canonical columns. Don't
        # crash later filters that read this file.
        empty_cols = list(df.columns) + [
            "n_hbonds_to_cat_his", "sap_max", "sap_mean", "sap_p95",
            "passed_struct_filter", "struct_fail",
        ]
        survivors = pd.DataFrame(columns=empty_cols)
        rejects = pd.DataFrame(columns=empty_cols)
    survivors.to_csv(out_dir / "survivors_struct.tsv", sep="\t", index=False)
    rejects.to_csv(out_dir / "rejects_struct.tsv", sep="\t", index=False)
    pd.DataFrame(hbond_rows).to_csv(out_dir / "hbond_details.tsv", sep="\t", index=False)
    LOGGER.info(
        "stage_struct_filter: %d / %d passed (h-bond to cat-HIS + sap_max<=%.0f)",
        len(survivors), len(out_df), sap_max_threshold,
    )
    return out_dir / "survivors_struct.tsv"


def stage_fitness_score(
    *,
    survivors_struct_tsv: Path,
    out_dir: Path,
    log_probs_esmc: np.ndarray,
    log_probs_saprot: np.ndarray,
    weights_per_position: np.ndarray,
    fitness_cache: dict,
) -> Path:
    """Score each survivor's fitness from cached PLM marginals."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.sampling.fitness_score import score_dataframe_fitness

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(survivors_struct_tsv, sep="\t")
    LOGGER.info("stage_fitness_score: input n=%d (cache size=%d)",
                 len(df), len(fitness_cache))
    scored = score_dataframe_fitness(
        df, log_probs_esmc, log_probs_saprot, weights_per_position,
        fitness_cache=fitness_cache,
    )
    out_path = out_dir / "scored.tsv"
    scored.to_csv(out_path, sep="\t", index=False)
    LOGGER.info(
        "stage_fitness_score: cache size now %d; logp_fused mean=%.3f",
        len(fitness_cache), float(scored["fitness__logp_fused_mean"].mean()),
    )
    return out_path


def _run_fpocket(
    pdb_path: Path,
    work_dir: Path,
    catalytic_resnos: Optional[Iterable[int]] = None,
    chain: str = CHAIN,
    pocket_distance_cutoff: float = 6.0,
) -> Optional[dict]:
    """Run fpocket and pick the pocket containing the active site.

    fpocket has a buffer overflow on long filenames so we copy to
    ``design.pdb`` inside work_dir before invoking.

    When ``catalytic_resnos`` is provided, we DON'T pick the most
    druggable pocket — we pick the pocket whose bounding alpha-spheres
    are within ``pocket_distance_cutoff`` of the most catalytic residues.
    This is what we actually want: the pocket where our ligand sits.
    Without this, fpocket will sometimes report a tighter peripheral
    pocket on a different face of the protein.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    local = work_dir / "design.pdb"
    local.write_bytes(pdb_path.read_bytes())
    try:
        proc = subprocess.run(
            [str(FPOCKET_BIN), "-f", str(local)],
            cwd=str(work_dir),
            capture_output=True, text=True, timeout=300, check=False,
        )
        info_txt = work_dir / "design_out" / "design_info.txt"
        pdb_out = work_dir / "design_out" / "design_out.pdb"
        if proc.returncode != 0 or not info_txt.is_file():
            LOGGER.warning("fpocket failed for %s: rc=%d", pdb_path.name, proc.returncode)
            return None
        if catalytic_resnos is None:
            return _parse_fpocket_largest(info_txt)
        # Active-site-aware: parse all pockets, score each by its
        # alpha-sphere proximity to catalytic residues, return best.
        return _parse_fpocket_active_site(
            info_txt, pdb_out, local,
            catalytic_resnos=catalytic_resnos,
            chain=chain,
            distance_cutoff=pocket_distance_cutoff,
        )
    except subprocess.TimeoutExpired:
        LOGGER.warning("fpocket timeout (>300s) for %s", pdb_path.name)
        return None


def _parse_fpocket_active_site(
    info_txt: Path,
    pdb_out: Path,
    design_pdb: Path,
    *,
    catalytic_resnos: Iterable[int],
    chain: str,
    distance_cutoff: float = 6.0,
) -> Optional[dict]:
    """Pick the fpocket pocket containing the active site.

    fpocket's ``<stem>_out.pdb`` lists every alpha sphere as a HETATM
    record with res_name = "STP" and a unique res_seq per pocket
    (1, 2, 3...). For each pocket, count alpha spheres within
    ``distance_cutoff`` of any catalytic CA atom. Pick the pocket
    with the highest count and return its info.txt entries plus a
    ``mean_alpha_sphere_distance_to_catalytic`` proxy for "how
    centered on the active site".
    """
    pockets = _parse_all_fpocket_pockets(info_txt)
    if not pockets:
        return None

    # Catalytic CA coords
    cat_set = set(int(r) for r in catalytic_resnos)
    cat_coords: list[np.ndarray] = []
    with open(design_pdb) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21].strip() != chain:
                continue
            try:
                resno = int(line[22:26].strip())
            except ValueError:
                continue
            if resno not in cat_set:
                continue
            cat_coords.append(np.array([
                float(line[30:38]), float(line[38:46]), float(line[46:54]),
            ]))
    if not cat_coords:
        return _parse_fpocket_largest(info_txt)
    cat_arr = np.array(cat_coords)

    # Per-pocket alpha sphere coords (from <stem>_out.pdb HETATM STP)
    pocket_spheres: dict[int, list[np.ndarray]] = {}
    if pdb_out.is_file():
        with open(pdb_out) as fh:
            for line in fh:
                if not line.startswith("HETATM"):
                    continue
                if line[17:20].strip() != "STP":
                    continue
                try:
                    pidx = int(line[22:26].strip())
                except ValueError:
                    continue
                pocket_spheres.setdefault(pidx, []).append(np.array([
                    float(line[30:38]), float(line[38:46]), float(line[46:54]),
                ]))

    # Score each pocket: count of alpha spheres within distance_cutoff of
    # ANY catalytic CA. Tie-break by proximity (mean min-distance).
    scored = []
    for p in pockets:
        idx = p["pocket_idx"]
        spheres = pocket_spheres.get(idx, [])
        if not spheres:
            continue
        sph_arr = np.array(spheres)
        # Min distance from each sphere to the nearest catalytic CA
        d = np.linalg.norm(
            sph_arr[:, None, :] - cat_arr[None, :, :], axis=-1,
        ).min(axis=1)
        n_close = int((d <= distance_cutoff).sum())
        mean_d = float(d.mean())
        scored.append((n_close, -mean_d, p, mean_d))   # higher n_close, lower mean_d wins
    if not scored:
        return _parse_fpocket_largest(info_txt)
    scored.sort(reverse=True)
    n_close, _, best, mean_d = scored[0]
    out = dict(best)
    out["n_alpha_spheres_near_catalytic"] = n_close
    out["mean_alpha_sphere_dist_to_catalytic"] = mean_d
    return out


def _parse_all_fpocket_pockets(info_txt: Path) -> list[dict]:
    """Parse ALL pockets from info.txt, not just the most druggable."""
    pockets: list[dict] = []
    cur: dict = {}
    with open(info_txt) as fh:
        for line in fh:
            m = re.match(r"^Pocket\s+(\d+)\s*:", line)
            if m:
                if cur:
                    pockets.append(cur)
                cur = {"pocket_idx": int(m.group(1))}
                continue
            m = re.match(r"^\s*([A-Za-z0-9_/. ]+):\s*([+-]?[\d.eE+-]+)", line)
            if m and cur:
                key = m.group(1).strip().lower().replace(" ", "_").replace(".", "")
                try:
                    cur[key] = float(m.group(2))
                except ValueError:
                    pass
    if cur:
        pockets.append(cur)
    return pockets


def _parse_fpocket_largest(info_txt: Path) -> Optional[dict]:
    """Parse ``info.txt``; return the most druggable pocket dict."""
    pockets: list[dict] = []
    cur: dict = {}
    with open(info_txt) as fh:
        for line in fh:
            m = re.match(r"^Pocket\s+(\d+)\s*:", line)
            if m:
                if cur:
                    pockets.append(cur)
                cur = {"pocket_idx": int(m.group(1))}
                continue
            m = re.match(r"^\s*([A-Za-z0-9_/. ]+):\s*([+-]?[\d.eE+-]+)", line)
            if m and cur:
                key = m.group(1).strip().lower().replace(" ", "_").replace(".", "")
                try:
                    cur[key] = float(m.group(2))
                except ValueError:
                    pass
    if cur:
        pockets.append(cur)
    if not pockets:
        return None
    pockets.sort(key=lambda p: p.get("druggability_score", 0), reverse=True)
    return pockets[0]


def stage_fpocket_rank(
    *,
    scored_tsv: Path,
    pdb_map: dict[str, Path],
    out_dir: Path,
    catalytic_resnos: Optional[Iterable[int]] = None,
    chain: str = CHAIN,
) -> Path:
    """Run fpocket on every fitness-scored survivor and emit ranked.tsv.

    When ``catalytic_resnos`` is given, fpocket is constrained to the
    pocket whose alpha-spheres cluster around the catalytic residues
    (the active-site pocket where our ligand binds), not the most
    druggable pocket of any kind. Critical for enzyme designs: without
    the constraint fpocket sometimes reports a peripheral pocket on
    a different face of the protein.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(scored_tsv, sep="\t")
    LOGGER.info("stage_fpocket_rank: input n=%d (active-site constraint=%s)",
                 len(df), "yes" if catalytic_resnos else "no")
    fpocket_dir = out_dir / "per_design_fpocket"
    fpocket_dir.mkdir(exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        cid = row["id"]
        pdb = pdb_map.get(cid)
        info = _run_fpocket(
            pdb, fpocket_dir / cid,
            catalytic_resnos=catalytic_resnos, chain=chain,
        ) if pdb else None
        rows.append({
            **row.to_dict(),
            "fpocket__druggability": info.get("druggability_score", 0.0) if info else 0.0,
            "fpocket__volume": info.get("volume", 0.0) if info else 0.0,
            "fpocket__mean_alpha_sphere_radius":
                info.get("mean_alpha_sphere_radius", 0.0) if info else 0.0,
            "fpocket__alpha_sphere_density":
                info.get("alpha_sphere_density", 0.0) if info else 0.0,
            "fpocket__n_alpha_spheres_near_catalytic":
                info.get("n_alpha_spheres_near_catalytic", 0) if info else 0,
            "fpocket__mean_alpha_sphere_dist_to_catalytic":
                info.get("mean_alpha_sphere_dist_to_catalytic", float("nan"))
                if info else float("nan"),
            "fpocket__n_pockets_found": 1 if info else 0,
        })
    ranked = pd.DataFrame(rows).sort_values(
        # primary: fitness desc; secondary: tighter pocket
        ["fitness__logp_fused_mean", "fpocket__mean_alpha_sphere_radius"],
        ascending=[False, True], na_position="last",
    )
    out_path = out_dir / "ranked.tsv"
    ranked.to_csv(out_path, sep="\t", index=False)
    LOGGER.info("stage_fpocket_rank: top fitness=%.3f, top radius=%.3f",
                 float(ranked["fitness__logp_fused_mean"].iloc[0]) if len(ranked) else float('nan'),
                 float(ranked["fpocket__mean_alpha_sphere_radius"].iloc[0]) if len(ranked) else float('nan'))
    return out_path


def stage_arpeggio_final(
    *,
    topk_pdb_dir: Path,
    topk_tsv: Path,
    out_dir: Path,
) -> None:
    """Run pdbe-arpeggio on the final top-K PDBs and merge per-design
    contact counts into topk.tsv.

    Arpeggio gives the proper physics panel: hbond / weak_hbond /
    halogen / ionic / metal_complex / aromatic / hydrophobic /
    carbonyl / polar / weak_polar / vdw / vdw_clash. Slow per-design
    (~5-15 s for a small enzyme + ligand), so reserved for the final
    characterization, not the inner sample/filter loop.

    Spawns a separate apptainer call into esmc.sif (where pdbe-arpeggio
    is installed via pip). If arpeggio fails or isn't available,
    silently degrades -- topk.tsv just won't gain the arpeggio
    columns. The simpler in-loop H-bond detection is independent.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(topk_tsv, sep="\t")
    if "id" not in df.columns:
        LOGGER.warning("stage_arpeggio_final: topk.tsv has no 'id' column; skipping")
        return

    # Build a one-shot script the inner apptainer call will run.
    inner_script = out_dir / "_run_arpeggio_inner.py"
    inner_script.write_text(
        '"""Inner script: convert PDBs to mmCIF, run pdbe-arpeggio, dump per-id JSON."""\n'
        'import sys, json\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, "/code/src")\n'
        'from protein_chisel.tools.arpeggio_interactions import arpeggio_interactions\n'
        '\n'
        'pdb_dir = Path(sys.argv[1])\n'
        'out_path = Path(sys.argv[2])\n'
        '\n'
        'try:\n'
        '    import biotite.structure.io.pdb as bpdb\n'
        '    import biotite.structure.io.pdbx as pdbx_io\n'
        '    have_biotite = True\n'
        'except ImportError:\n'
        '    have_biotite = False\n'
        '\n'
        'results = {}\n'
        'errors = []\n'
        'for pdb in sorted(pdb_dir.glob("*.pdb")):\n'
        '    try:\n'
        '        if have_biotite:\n'
        '            cif = pdb.with_suffix(".cif")\n'
        '            struct = bpdb.PDBFile.read(str(pdb)).get_structure(model=1)\n'
        '            f = pdbx_io.CIFFile(); pdbx_io.set_structure(f, struct)\n'
        '            f.write(str(cif))\n'
        '            res = arpeggio_interactions(cif_path=cif, timeout=180)\n'
        '            cif.unlink(missing_ok=True)\n'
        '        else:\n'
        '            res = arpeggio_interactions(cif_path=pdb, timeout=180)\n'
        '        results[pdb.stem] = res.to_dict("arpeggio__")\n'
        '    except Exception as e:\n'
        '        errors.append((pdb.stem, str(e)[:200]))\n'
        '        results[pdb.stem] = {}\n'
        '\n'
        'out_path.write_text(json.dumps({"results": results, "errors": errors}, indent=2))\n'
    )

    json_out = out_dir / "arpeggio_per_design.json"
    cmd = [
        "apptainer", "exec",
        "--bind", "/home/woodbuse/codebase_projects/protein_chisel:/code",
        "--bind", "/net/scratch",
        "--env", "PYTHONPATH=/code/src",
        "/net/software/containers/users/woodbuse/esmc.sif",
        "python", str(inner_script),
        str(topk_pdb_dir),
        str(json_out),
    ]
    LOGGER.info("stage_arpeggio_final: running on %d top-K PDBs (this is slow, ~10s/PDB)",
                 len(df))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, check=False,
        )
        if proc.returncode != 0:
            LOGGER.warning("stage_arpeggio_final: arpeggio inner call rc=%d; "
                            "stderr tail: %s", proc.returncode, proc.stderr[-500:])
            return
    except Exception as e:
        LOGGER.warning("stage_arpeggio_final: exception spawning arpeggio: %s", e)
        return

    if not json_out.is_file():
        LOGGER.warning("stage_arpeggio_final: arpeggio JSON not produced at %s", json_out)
        return
    data = json.loads(json_out.read_text())
    LOGGER.info("stage_arpeggio_final: %d designs scored, %d errors",
                 len(data["results"]), len(data["errors"]))

    # Merge per-id columns into topk.tsv
    rows: list[dict] = []
    for _, row in df.iterrows():
        cid = row["id"]
        ad = data["results"].get(cid, {})
        rows.append({**row.to_dict(), **ad})
    enriched = pd.DataFrame(rows)
    enriched_path = out_dir / "topk_with_arpeggio.tsv"
    enriched.to_csv(enriched_path, sep="\t", index=False)
    LOGGER.info("stage_arpeggio_final: wrote %s", enriched_path)


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def stage_diverse_topk(
    *,
    pool_df: pd.DataFrame,
    pdb_map: dict[str, Path],
    out_dir: Path,
    target_k: int,
    min_hamming: int,
) -> Path:
    """Greedy top-K with Hamming-distance diversity over all sequences in
    ``pool_df`` (already deduped, sorted descending by fitness)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(pool_df) == 0:
        LOGGER.warning("stage_diverse_topk: empty pool")
        return out_dir / "topk.tsv"

    selected_idx: list[int] = []
    selected_seqs: list[str] = []
    for i, row in pool_df.iterrows():
        seq = row["sequence"]
        if all(_hamming(seq, s) >= min_hamming for s in selected_seqs):
            selected_idx.append(i)
            selected_seqs.append(seq)
        if len(selected_idx) >= target_k:
            break
    top = pool_df.loc[selected_idx].copy()
    top.to_csv(out_dir / "topk.tsv", sep="\t", index=False)
    fasta = out_dir / "topk.fasta"
    with open(fasta, "w") as fh:
        for _, row in top.iterrows():
            fh.write(f">{row['id']}\n{row['sequence']}\n")
    pdb_out = out_dir / "topk_pdbs"
    pdb_out.mkdir(exist_ok=True)
    for _, row in top.iterrows():
        src = pdb_map.get(row["id"])
        if src and src.is_file():
            shutil.copy2(src, pdb_out / src.name)
    LOGGER.info("stage_diverse_topk: selected %d / %d (target=%d, min_hamming=%d)",
                 len(top), len(pool_df), target_k, min_hamming)
    return out_dir / "topk.tsv"


# ----------------------------------------------------------------------
# One iteration cycle
# ----------------------------------------------------------------------


def run_cycle(
    *,
    cycle_cfg: CycleConfig,
    seed_pdb: Path,
    base_bias: np.ndarray,
    log_probs_esmc: np.ndarray,
    log_probs_saprot: np.ndarray,
    weights_per_position: np.ndarray,
    position_classes: list[str],
    protein_resnos: list[int],
    fixed_resnos: list[int],
    survivors_prev: Optional[pd.DataFrame],
    cycle_dir: Path,
    fitness_cache: dict,
    wt_length: int,
    expression_engine,                # ExpressionRuleEngine
    seed_ss_reduced: Optional[str] = None,
    seed_sasa: Optional[np.ndarray] = None,
    seed_position_class: Optional[list[str]] = None,
    seed_protein_resnos: Optional[list[int]] = None,
    position_table_df=None,           # for first-shell diversity injection
    omit_AA_per_residue: Optional[dict[str, str]] = None,
    catalytic_his_resnos: Iterable[int] = CATALYTIC_HIS_RESNOS,
) -> tuple[Optional[pd.DataFrame], dict[str, Path]]:
    """Run ONE iteration cycle. Returns (ranked DataFrame, pdb_map)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.sampling.iterative_fusion import (
        IterationBiasConfig, build_iteration_bias,
    )

    cycle_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0. Build cycle-k bias --------------------------------------
    bias_dir = cycle_dir / "00_bias"
    bias_dir.mkdir(exist_ok=True)
    if survivors_prev is None or len(survivors_prev) == 0:
        bias_k = base_bias.copy()
        telem = {"used_consensus": False, "n_survivors": 0}
    else:
        # Convert catalytic resnos to zero-indexed protein-array indices
        resno_to_idx = {r: i for i, r in enumerate(protein_resnos)}
        fixed_idx = [resno_to_idx[r] for r in fixed_resnos if r in resno_to_idx]
        cfg = IterationBiasConfig(
            consensus_threshold=cycle_cfg.consensus_threshold,
            consensus_strength=cycle_cfg.consensus_strength,
        )
        bias_k, t = build_iteration_bias(
            base_bias=base_bias,
            survivor_sequences=survivors_prev["sequence"].tolist(),
            position_classes=position_classes,
            fixed_resnos_zero_indexed=fixed_idx,
            protein_resnos=protein_resnos,
            config=cfg,
        )
        telem = {
            "used_consensus": True,
            "n_survivors": t.n_survivors,
            "n_positions_eligible": t.n_positions_eligible,
            "n_positions_augmented": t.n_positions_augmented,
            "augmented_resnos_1idx": t.augmented_resnos,
            "capped": t.capped,
        }
    np.save(bias_dir / "bias.npy", bias_k)
    with open(bias_dir / "telemetry.json", "w") as fh:
        json.dump(telem, fh, indent=2)

    # ---- 1. Sample --------------------------------------------------
    sample_dir = cycle_dir / "01_sample"
    # Diversity injection: per-cycle, randomly forbid the WT identity at
    # ~30% of first-shell positions so MPNN's structure-conditioned bias
    # toward WT identities at structurally constrained sites is broken.
    # Different positions per cycle -> across all cycles the union covers
    # most first-shell positions.
    diversity_omit: dict[str, str] = {}
    if position_table_df is not None:
        # Per agent-review bug: cycle_idx*7919 with cycle_idx=0 gives
        # seed=0, so cycle 0 always picks the same 4 first-shell
        # positions across every production run. Mix in a process-wide
        # constant + run-time minute so different runs explore different
        # subsets, while still being deterministic within a single run.
        cycle_seed = (cycle_cfg.cycle_idx + 1) * 7919 + (
            int(_dt.datetime.now().minute) * 31
        )
        diversity_omit = compute_first_shell_diversity_omits(
            position_table_df=position_table_df,
            fixed_resnos=fixed_resnos,
            chain=CHAIN,
            fraction_to_diversify=0.30,
            seed=cycle_seed,
        )
    LOGGER.info(
        "cycle %d: diversity-omit at first-shell = %s",
        cycle_cfg.cycle_idx, diversity_omit,
    )
    merged_omit = merge_omit_dicts(omit_AA_per_residue or {}, diversity_omit)
    cand_tsv = stage_sample(
        cycle_cfg=cycle_cfg, seed_pdb=seed_pdb, bias=bias_k,
        protein_resnos=protein_resnos, fixed_resnos=fixed_resnos,
        out_dir=sample_dir,
        omit_AA_per_residue=merged_omit,
    )

    # ---- 2. Restore PDBs --------------------------------------------
    cand_df = pd.read_csv(cand_tsv, sep="\t")
    pdb_basename = seed_pdb.stem
    pdb_map = stage_restore_pdbs(
        sample_dir=sample_dir, ref_pdb=seed_pdb,
        out_pdb_dir=sample_dir / "pdbs_restored",
        pdb_basename=pdb_basename,
        candidate_ids=cand_df["id"].tolist(),
    )

    # ---- 3. Cheap seq filter (does its own dedup) -------------------
    seq_filter_dir = cycle_dir / "02_seq_filter"
    survivors_seq = stage_seq_filter(
        candidates_tsv=cand_tsv, out_dir=seq_filter_dir,
        net_charge_max=cycle_cfg.net_charge_max,
        wt_length=wt_length,
        expression_engine=expression_engine,
        seed_ss_reduced=seed_ss_reduced,
        seed_sasa=seed_sasa,
        seed_position_class=seed_position_class,
        seed_protein_resnos=seed_protein_resnos,
        catalytic_resnos=fixed_resnos,
        fixed_resnos=fixed_resnos,
        pi_min=cycle_cfg.pi_min,
        pi_max=cycle_cfg.pi_max,
    )

    # ---- 4. Struct filter -------------------------------------------
    struct_filter_dir = cycle_dir / "03_struct_filter"
    survivors_struct = stage_struct_filter(
        survivors_seq_tsv=survivors_seq, pdb_map=pdb_map,
        out_dir=struct_filter_dir,
        sap_max_threshold=cycle_cfg.sap_max_threshold,
        catalytic_his_resnos=catalytic_his_resnos,
        fixed_resnos=fixed_resnos,
        clash_filter=cycle_cfg.clash_filter,
        clash_severe_distance=cycle_cfg.clash_severe_distance,
    )

    n_struct = len(pd.read_csv(survivors_struct, sep="\t"))
    if n_struct == 0:
        LOGGER.warning("cycle %d: zero struct survivors -- nothing to score/rank",
                        cycle_cfg.cycle_idx)
        return None, pdb_map

    # ---- 5. Fitness scoring -----------------------------------------
    fitness_dir = cycle_dir / "04_fitness"
    scored = stage_fitness_score(
        survivors_struct_tsv=survivors_struct, out_dir=fitness_dir,
        log_probs_esmc=log_probs_esmc,
        log_probs_saprot=log_probs_saprot,
        weights_per_position=weights_per_position,
        fitness_cache=fitness_cache,
    )

    # ---- 6. Fpocket rank (constrained to active-site pocket) -------
    fpocket_dir = cycle_dir / "05_fpocket"
    ranked = stage_fpocket_rank(
        scored_tsv=scored, pdb_map=pdb_map, out_dir=fpocket_dir,
        catalytic_resnos=fixed_resnos,
    )
    ranked_df = pd.read_csv(ranked, sep="\t")
    return ranked_df, pdb_map


# ----------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed_pdb", type=Path, default=DEFAULT_INPUT_PDB)
    p.add_argument("--ligand_params", type=Path, default=DEFAULT_LIG_PARAMS)
    p.add_argument("--plm_artifacts_dir", type=Path, required=True)
    p.add_argument("--position_table", type=Path, required=True)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--target_k", type=int, default=50)
    p.add_argument("--min_hamming", type=int, default=3)
    p.add_argument("--cycles", type=int, default=3,
                   help="Use the default 3-cycle schedule when 3 (default), "
                        "or override with a single-cycle short test (1).")
    p.add_argument("--omit_AA", type=str, default="X",
                   help="AAs MPNN may never sample. Default 'X' (UNK only) -- "
                        "no canonical AAs are silently forbidden. Pass 'CX' "
                        "to also exclude cysteine (recommended for scaffolds "
                        "with no catalytic Cys); add others as needed.")
    p.add_argument("--expression_profile", type=str,
                   default="bl21_cytosolic_streptag",
                   choices=["bl21_cytosolic_streptag", "k12_cytosolic",
                            "bl21_periplasmic"],
                   help="Host-expression profile that drives the rule engine.")
    p.add_argument("--expression_overrides", type=str, default="",
                   help="Comma-sep rule_name=SEVERITY overrides, e.g. "
                        "'kr_neighbor_dibasic=HARD_OMIT,polyproline_stall=WARN_ONLY'. "
                        "SEVERITY in {WARN_ONLY,SOFT_BIAS,HARD_OMIT,HARD_FILTER}.")
    p.add_argument("--use_side_chain_context", type=int, default=0,
                   choices=[0, 1],
                   help="LigandMPNN flag. 0 (default) = MPNN sees only "
                        "backbone + ligand atoms (better first-shell "
                        "diversity). The clash failure mode that previously "
                        "made sc=0 unsafe (Y/F/W at first-shell positions "
                        "vs catalytic sidechains) is now prevented at "
                        "sample time by auto-detected per-residue omits "
                        "for clash-prone positions. 1 = MPNN sees catalytic "
                        "sidechain rotamers (more WT-conservative).")
    p.add_argument("--enhance", type=str, default=None,
                   choices=[None, *AVAILABLE_ENHANCE_CHECKPOINTS],
                   help="Optional pLDDT-enhanced fused_mpnn checkpoint name. "
                        "Default None (use base ligand_mpnn). Available choices "
                        f"({len(AVAILABLE_ENHANCE_CHECKPOINTS)}): "
                        + ", ".join(AVAILABLE_ENHANCE_CHECKPOINTS))
    p.add_argument("--pi_min", type=float, default=5.0,
                   help="Minimum theoretical pI. Default 5.0 selects the "
                        "least-acidic ~1%% of cycle-0 designs at "
                        "--net_charge_max<-10. Low cycle-0 pass rate is "
                        "fine: consensus-bias iteration in cycle 1+ pulls "
                        "subsequent cycles toward less-acidic sequences. "
                        "Relax to 4.7 for higher cycle-0 pass at the cost "
                        "of weaker selection pressure.")
    p.add_argument("--pi_max", type=float, default=7.5)
    p.add_argument("--fpocket_druggability_min", type=float, default=0.30,
                   help="Drop designs with fpocket-druggability below this "
                        "(no detectable active-site cavity = bad design). "
                        "Set to 0 to disable.")
    p.add_argument("--no_clash_filter", action="store_true",
                   help="Disable the heavy-atom clash check between catalytic+"
                        "ligand and designed sidechains. Off by default; only "
                        "use if you're debugging clash false positives.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / f"iterative_design_v2_PTE_i1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("=== run dir: %s ===", run_dir)

    # ---- Load PLM artifacts -----------------------------------------
    art = args.plm_artifacts_dir
    log_probs_esmc = np.load(art / "esmc_log_probs.npy")
    log_probs_saprot = np.load(art / "saprot_log_probs.npy")
    base_bias = np.load(art / "fusion_bias.npy")
    weights_per_position = np.load(art / "fusion_weights.npy")
    LOGGER.info("loaded PLM artifacts: L=%d, mean_abs_bias=%.4f",
                 base_bias.shape[0], float(np.abs(base_bias).mean()))

    # ---- Load PositionTable -----------------------------------------
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.io.schemas import PositionTable
    from protein_chisel.io.pdb import extract_sequence

    pt = PositionTable.from_parquet(args.position_table)
    protein_rows = pt.df[pt.df["is_protein"]].sort_values("resno").reset_index(drop=True)
    position_classes = protein_rows["class"].tolist()
    protein_resnos = protein_rows["resno"].astype(int).tolist()
    L = len(protein_resnos)
    LOGGER.info("position table: L=%d, class counts=%s",
                 L, protein_rows["class"].value_counts().to_dict())

    wt_seq = extract_sequence(args.seed_pdb, chain=CHAIN)
    if len(wt_seq) != L:
        raise RuntimeError(f"WT seq length {len(wt_seq)} != PositionTable {L}")
    # Threshold OmpT motifs at WT count: don't reject sequences that
    # are no worse than WT for E. coli expression. WT has dibasic motifs
    # that are forced by the catalytic K157 (KK at 157-158); a strict
    # 0-motif rule rejects WT and every faithful design.
    # ---- Build expression-rule engine -------------------------------
    from protein_chisel.expression import (
        ExpressionRuleEngine, ExpressionProfile,
    )
    import protein_chisel.expression.builtin_rules  # noqa: F401 — registers
    from protein_chisel.structure import SSProvider

    base_profile = {
        "bl21_cytosolic_streptag": ExpressionProfile.bl21_cytosolic_streptag,
        "k12_cytosolic": ExpressionProfile.k12_cytosolic,
        "bl21_periplasmic": ExpressionProfile.bl21_periplasmic,
    }[args.expression_profile]()
    profile = ExpressionProfile.from_overrides_string(
        base_profile, args.expression_overrides,
    )
    LOGGER.info("expression profile: %s (preset=%s)", profile.name, profile.preset)
    if profile.severity_overrides:
        LOGGER.info("severity overrides: %s",
                     {k: v.name for k, v in profile.severity_overrides.items()})
    expression_engine = ExpressionRuleEngine(profile=profile)

    # ---- Compute SS consensus + per-residue features once on seed PDB
    LOGGER.info("computing SS consensus on seed PDB (3 algorithms)")
    ss = SSProvider().from_pdb(args.seed_pdb, chain=CHAIN)
    LOGGER.info("SS: H=%d E=%d L=%d (mean confidence %.2f, used %s, failed %s)",
                 ss.ss_reduced.count("H"), ss.ss_reduced.count("E"),
                 ss.ss_reduced.count("L"), float(ss.confidence.mean()),
                 ss.used_algos, ss.failed_algos or "none")
    if len(ss.ss_reduced) != L:
        raise RuntimeError(f"SS length {len(ss.ss_reduced)} != L {L}")
    seed_sasa = protein_rows["sasa"].astype(float).values

    # ---- Pre-flight: evaluate WT against the engine ---------------
    wt_eng = expression_engine.evaluate(
        wt_seq,
        ss_reduced=ss.ss_reduced,
        sasa=seed_sasa,
        position_class=position_classes,
        catalytic_resnos=DEFAULT_CATRES,
        fixed_resnos=DEFAULT_CATRES,
        protein_resnos=protein_resnos,
    )
    LOGGER.info("WT engine eval: %s", wt_eng.summary())
    expression_omit = wt_eng.to_omit_AA_json("A", protein_resnos=protein_resnos)
    LOGGER.info("expression-engine HARD_OMIT JSON: %s", expression_omit)

    # Graded clash-aware bias replacing the previous hard-omit. Per the
    # rotamer-feasibility audit (commit logs + scripts/audit_clash_omits.py),
    # no (clash-prone-pos, bulky-AA) pair has >50% clashing rotamers in
    # a 9-rotamer grid stub, so the previous hard-omit was unjustified.
    # Now: add a per-position per-AA bias proportional to the clash %
    # to the base PLM-fusion bias (max -3 nats at 100% clash, 0 at 0%).
    # MPNN can still pick a "clash-prone" AA when other context strongly
    # favors it; the filter-time severe-clash check (1.5 A) catches the
    # remaining hard failures.
    clash_bias, clash_telem = compute_graded_clash_bias(
        seed_pdb=args.seed_pdb,
        position_table_df=pt.df,
        fixed_resnos=DEFAULT_CATRES,
        chain=CHAIN,
        cb_clearance_threshold=5.0,
        bulky_aas="YFWHMR",
    )
    LOGGER.info(
        "graded clash bias: %d positions biased, mean magnitude=%.3f nats",
        clash_telem["n_positions_biased"], float(np.abs(clash_bias).mean()),
    )
    base_bias = base_bias + clash_bias   # added to the cycle-0 fusion bias
    omit_AA_per_residue = expression_omit
    LOGGER.info("structural omit_AA (from rule engine only): %s", omit_AA_per_residue)

    # ---- Cycle schedule ---------------------------------------------
    cycles = default_cycles(
        omit_AA=args.omit_AA,
        use_side_chain_context=args.use_side_chain_context,
        enhance=args.enhance,
        pi_min=args.pi_min, pi_max=args.pi_max,
        fpocket_druggability_min=args.fpocket_druggability_min,
        clash_filter=not args.no_clash_filter,
    )
    if args.cycles == 1:
        cycles = cycles[:1]   # short-test mode
    elif args.cycles != 3:
        # Honor any positive int by truncating / extending the default schedule.
        cycles = cycles[: max(1, args.cycles)]
    LOGGER.info("cycle schedule: %d cycles, omit_AA=%r", len(cycles), args.omit_AA)

    # ---- Loop cycles ------------------------------------------------
    all_ranked: list[pd.DataFrame] = []
    all_pdb_maps: dict[str, Path] = {}
    fitness_cache: dict = {}
    survivors_prev: Optional[pd.DataFrame] = None
    fixed_resnos = list(DEFAULT_CATRES)

    for cyc in cycles:
        cycle_dir = run_dir / f"cycle_{cyc.cycle_idx:02d}"
        ranked_df, pdb_map = run_cycle(
            cycle_cfg=cyc, seed_pdb=args.seed_pdb,
            base_bias=base_bias,
            log_probs_esmc=log_probs_esmc,
            log_probs_saprot=log_probs_saprot,
            weights_per_position=weights_per_position,
            position_classes=position_classes,
            protein_resnos=protein_resnos,
            fixed_resnos=fixed_resnos,
            survivors_prev=survivors_prev,
            cycle_dir=cycle_dir,
            fitness_cache=fitness_cache,
            wt_length=L,
            expression_engine=expression_engine,
            seed_ss_reduced=ss.ss_reduced,
            seed_sasa=seed_sasa,
            seed_position_class=position_classes,
            seed_protein_resnos=protein_resnos,
            position_table_df=pt.df,
            omit_AA_per_residue=omit_AA_per_residue,
        )
        if ranked_df is not None and len(ranked_df) > 0:
            ranked_df = ranked_df.copy()
            ranked_df["cycle"] = cyc.cycle_idx
            all_ranked.append(ranked_df)
            survivors_prev = ranked_df
        all_pdb_maps.update(pdb_map)

    # ---- Final pool: concat + dedup --------------------------------
    final_dir = run_dir / "final_topk"
    final_dir.mkdir(parents=True, exist_ok=True)
    if all_ranked:
        from protein_chisel.sampling.fitness_score import deduplicate_by_sequence

        pool = pd.concat(all_ranked, ignore_index=True)
        pool = deduplicate_by_sequence(pool)
        # Drop designs with no detectable active-site pocket. fpocket
        # druggability < threshold means the constrained search at the
        # catalytic site couldn't find a pocket -- bad design.
        druggability_min = cycles[0].fpocket_druggability_min
        if "fpocket__druggability" in pool.columns and druggability_min > 0:
            n_before = len(pool)
            pool = pool[pool["fpocket__druggability"] >= druggability_min].copy()
            LOGGER.info(
                "fpocket-druggability filter: %d -> %d (cutoff=%.2f)",
                n_before, len(pool), druggability_min,
            )
        pool = pool.sort_values(
            ["fitness__logp_fused_mean", "fpocket__mean_alpha_sphere_radius"],
            ascending=[False, True], na_position="last",
        ).reset_index(drop=True)
        pool.to_csv(final_dir / "all_survivors.tsv", sep="\t", index=False)
        LOGGER.info("final pool: %d unique survivors across %d cycles",
                     len(pool), len(all_ranked))

        stage_diverse_topk(
            pool_df=pool, pdb_map=all_pdb_maps,
            out_dir=final_dir,
            target_k=args.target_k, min_hamming=args.min_hamming,
        )
    else:
        LOGGER.warning("final: zero survivors across all cycles!")

    # ---- Manifest ---------------------------------------------------
    manifest = {
        "pipeline": "iterative_design_v2",
        "seed_pdb": str(args.seed_pdb),
        "ligand_params": str(args.ligand_params),
        "plm_artifacts_dir": str(args.plm_artifacts_dir),
        "position_table": str(args.position_table),
        "fixed_resnos": list(DEFAULT_CATRES),
        "catalytic_his_resnos": list(CATALYTIC_HIS_RESNOS),
        "wt_length": L,
        "target_k": args.target_k,
        "diversity_min_hamming": args.min_hamming,
        "n_cycles_run": len(cycles),
        "cycle_configs": [asdict(c) for c in cycles],
        "outputs": {
            "run_dir": str(run_dir),
            "final_topk_fasta": str(final_dir / "topk.fasta"),
            "final_topk_pdbs": str(final_dir / "topk_pdbs"),
            "all_survivors": str(final_dir / "all_survivors.tsv"),
        },
        "started_at": timestamp,
    }
    with open(run_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("=== DONE -- top-K at %s ===", final_dir)


if __name__ == "__main__":
    main()
