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
# These default values match the PTE_i1 SEED1 (FS269) scaffold. They are
# AUTO-OVERRIDDEN at the start of main() by parsing REMARK 666 in the
# user-supplied --seed_pdb so the same code/sbatch works on every
# scaffold in a design campaign (catalytic resnos shift between
# scaffolds even though motif structure is preserved).
DEFAULT_CATRES = (60, 64, 128, 131, 132, 157)
CATALYTIC_HIS_RESNOS = (60, 64, 128, 132)
CHAIN = "A"


def _derive_catres_from_remark_666(seed_pdb: Path | str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Read REMARK 666 from ``seed_pdb`` and return:

        (all_catalytic_resnos, his_only_catalytic_resnos)

    sorted ascending. Falls back to the module defaults
    (PTE_i1 SEED1) if no REMARK 666 entries are found.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.protonate_final import parse_remark_666 as _parse666
    entries = _parse666(seed_pdb)
    if not entries:
        return DEFAULT_CATRES, CATALYTIC_HIS_RESNOS
    all_resnos = tuple(sorted({e.motif_resno for e in entries}))
    his_resnos = tuple(sorted({
        e.motif_resno for e in entries if e.motif_resname.upper() == "HIS"
    }))
    return all_resnos, his_resnos

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
    net_charge_max: float = -4.0      # accept if net_charge_no_HIS < net_charge_max
    net_charge_min: float = -18.0     # accept if net_charge_no_HIS > net_charge_min
    # SAP_max threshold on the FREESASA-PROXY scale (Lauer-style 10A
    # neighborhood SASA-weighted hydrophobicity). For PTE_i1 the proxy
    # ranges 0.3-3.2 across designs; this filter is essentially OFF
    # at threshold 100 (vestigial -- kept for the metric).
    # The user's target "SAP < 15" is on the ROSETTA SAP scale; that
    # is computed in the OPTIONAL rosetta-final stage (top-K only)
    # because PyRosetta SapScoreMetric is too slow for inner-loop use.
    # WT PTE_i1 Rosetta SAP = 28.4; "<15" is not achievable for this
    # scaffold, so we apply it as soft ranking only at final stage.
    sap_max_threshold: float = 100.0
    consensus_threshold: float = 0.85
    consensus_strength: float = 2.0
    consensus_max_fraction: float = 0.30
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
    # Per-cycle light-filter thresholds (annealing strategy). Default
    # values match the global defaults so "constant" strategy is a no-op.
    instability_max: float = 60.0
    gravy_min: float = -0.8
    gravy_max: float = 0.3
    aliphatic_min: float = 40.0
    boman_max: float = 4.5
    # Per-cycle TOPSIS weight overrides (annealing strategy). Maps
    # metric label -> weight; merged with the global default specs at
    # ranking time. Empty dict = use global defaults.
    topsis_weight_overrides: dict[str, float] = field(default_factory=dict)
    # If True, this cycle's survivors_prev for the next cycle is chosen
    # by TOPSIS (full multi-objective). If False (legacy), by fitness alone.
    use_topsis_for_survivors: bool = False


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
    strategy: str = "constant",
    consensus_threshold: float = 0.85,
    consensus_strength: float = 2.0,
    consensus_max_fraction: float = 0.30,
) -> list[CycleConfig]:
    """Three-cycle exploration → exploitation schedule.

    ``strategy``:
      - ``"constant"`` (default, legacy): all cycles share the same
        filter thresholds and TOPSIS weights. Survivors fed forward
        by fitness alone.
      - ``"annealing"``: cycle 0 has gentle-loose light filters AND
        fitness-heavy TOPSIS weights (explore); cycle 1 is balanced;
        cycle 2 uses the strict default filters AND default TOPSIS
        weights with TOPSIS-based survivor selection (exploit). Hard
        filters (charge band, pi band) stay constant throughout per
        the user's preference; only the LIGHT filters (instability,
        GRAVY, aliphatic, boman) and TOPSIS weights anneal.
    """
    if strategy not in ("constant", "annealing"):
        raise ValueError(f"strategy must be 'constant' or 'annealing', got {strategy!r}")
    common = dict(
        omit_AA=omit_AA,
        use_side_chain_context=use_side_chain_context,
        enhance=enhance,
        pi_min=pi_min, pi_max=pi_max,
        fpocket_druggability_min=fpocket_druggability_min,
        clash_filter=clash_filter,
        consensus_threshold=consensus_threshold,
        consensus_strength=consensus_strength,
        consensus_max_fraction=consensus_max_fraction,
    )
    # Charge band and pi band stay constant (user pref: "the current
    # range should be the final one"). Annealing only relaxes the
    # *light* filters (instability/GRAVY/aliphatic/boman) early and
    # uses TOPSIS for survivor selection late.
    if strategy == "constant":
        return [
            CycleConfig(
                cycle_idx=0, n_samples=500, sampling_temperature=0.20,
                net_charge_max=-4.0, net_charge_min=-18.0,
                sap_max_threshold=100.0, **common,
            ),
            CycleConfig(
                cycle_idx=1, n_samples=400, sampling_temperature=0.18,
                net_charge_max=-4.0, net_charge_min=-18.0,
                sap_max_threshold=100.0, **common,
            ),
            CycleConfig(
                cycle_idx=2, n_samples=300, sampling_temperature=0.15,
                net_charge_max=-4.0, net_charge_min=-18.0,
                sap_max_threshold=100.0, **common,
            ),
        ]
    # Annealing — gentle relaxation, never tighten beyond defaults.
    # Cycle 0 (explore): light filters loose; TOPSIS heavy on fitness;
    #                    survivors picked by fitness (legacy).
    # Cycle 1 (transition): light filters slightly loose; balanced
    #                    TOPSIS weights (defaults).
    # Cycle 2 (exploit): light filters at default; default TOPSIS
    #                    weights; survivors picked by TOPSIS so the
    #                    final pool reinforces multi-objective good.
    return [
        CycleConfig(
            cycle_idx=0, n_samples=500, sampling_temperature=0.20,
            net_charge_max=-4.0, net_charge_min=-18.0,
            sap_max_threshold=100.0,
            instability_max=80.0, gravy_min=-1.0, gravy_max=0.4,
            aliphatic_min=30.0, boman_max=5.5,
            topsis_weight_overrides={
                "fitness": 3.0,            # explore aggressively on fitness
                "instability": 0.1, "gravy": 0.1, "aliphatic": 0.1,
                "boman": 0.1, "pocket_hydrophobicity": 0.1,
            },
            use_topsis_for_survivors=False,
            **common,
        ),
        CycleConfig(
            cycle_idx=1, n_samples=400, sampling_temperature=0.18,
            net_charge_max=-4.0, net_charge_min=-18.0,
            sap_max_threshold=100.0,
            instability_max=70.0, gravy_min=-0.9, gravy_max=0.35,
            aliphatic_min=35.0, boman_max=5.0,
            topsis_weight_overrides={},        # balanced (defaults)
            use_topsis_for_survivors=True,
            **common,
        ),
        CycleConfig(
            cycle_idx=2, n_samples=300, sampling_temperature=0.15,
            net_charge_max=-4.0, net_charge_min=-18.0,
            sap_max_threshold=100.0,
            instability_max=60.0, gravy_min=-0.8, gravy_max=0.3,
            aliphatic_min=40.0, boman_max=4.5,
            topsis_weight_overrides={},        # balanced (defaults)
            use_topsis_for_survivors=True,
            **common,
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
    eligible_classes: tuple[str, ...] = (
        "first_shell", "buried",                                  # legacy
        "primary_sphere", "secondary_sphere", "distal_buried",    # new
    ),
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
    eligible_classes: tuple[str, ...] = (
        "first_shell", "buried",                                  # legacy
        "primary_sphere", "secondary_sphere", "distal_buried",    # new
    ),
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
    eligible_classes: tuple[str, ...] = (
        "first_shell",                            # legacy
        "primary_sphere", "secondary_sphere",     # new
    ),
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
    bias_AA: str = "",
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
        # Global bias_AA: class-balanced compensatory bias built from
        # the previous cycle's survivor pool composition (see
        # protein_chisel.expression.aa_class_balance). Cycle 0 leaves this
        # empty; cycles 1+ get e.g. "E:-1.40,D:1.45,..." to swap within
        # an over/under-rep class. Applies uniformly to every position
        # on top of the per-residue PLM-fusion bias.
        bias_AA=bias_AA,
        extra_flags=tuple(extra_flags),
    )

    LOGGER.info(
        "stage_sample[cycle=%d]: n=%d, T=%.3f, fixed=%s, "
        "mean_abs_bias=%.4f, bias_AA=%r",
        cycle_cfg.cycle_idx, cycle_cfg.n_samples,
        cycle_cfg.sampling_temperature, sorted(set(fixed_resnos)),
        float(np.abs(bias).mean()), bias_AA or "(none)",
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
    catalytic_resnos: Optional[Iterable[int]] = None,
    catalytic_hydrogens: bool = True,
) -> dict[str, Path]:
    """Restore REMARK 666 / HETNAM / LINK + HIS tautomer (HIS_D / HIE / HIP)
    + KCX cap atoms + catalytic hydrogens onto the packed MPNN PDBs.

    Delegates to ``protein_chisel.tools.pdb_restoration.restore_sample_dir``;
    see that module for the full restoration semantics. The signature is
    kept stable so call sites in this driver are unchanged.
    """
    if catalytic_resnos is None:
        catalytic_resnos = DEFAULT_CATRES
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
    net_charge_min: float = -50.0,
    wt_length: int = 200,
    expression_engine,                # ExpressionRuleEngine
    seed_ss_reduced: Optional[str] = None,
    seed_sasa: Optional[np.ndarray] = None,
    seed_position_class: Optional[list[str]] = None,
    seed_protein_resnos: Optional[list[int]] = None,
    catalytic_resnos: Iterable[int] = (),
    design_ph: float = 7.5,
    fixed_resnos: Iterable[int] = (),
    # N/C-term sequence pads added to the design body BEFORE computing
    # sequence-only metrics. Default empty so legacy behavior is
    # preserved; pass e.g. n_term_pad="MSG" / c_term_pad="GSA" so
    # protparam reflects the full expressed protein after vector
    # tags. Affects charge, pI, GRAVY, instability, aliphatic, boman.
    # N/C-term sequence pads added to the design body BEFORE computing
    # ProtParam metrics (charge, pI, GRAVY, instability, aliphatic,
    # boman). The expression rule engine deliberately sees the UNPADDED
    # design body so its structure-aware rules (kr_neighbor_dibasic
    # etc.) align with the seed PDB's per-residue SS / SASA / class.
    # Per the user's spec: "use these N- and C-term adds for sequence-
    # specific, structure-agnostic calculations." Junction-induced
    # liabilities (e.g. dibasic spanning the tag/design boundary) are
    # NOT caught here — would require a separate expression-engine
    # invocation on the padded sequence with no structure context.
    n_term_pad: str = "",
    c_term_pad: str = "",
    # Light de-novo filters on cheap sequence-only metrics. Generous
    # thresholds — only catch truly broken designs, not most of the pool.
    instability_max: float = 60.0,        # Guruprasad 1990; lit 40 is for natives
    gravy_min: float = -0.8,              # typical soluble: -0.4 to 0
    gravy_max: float = +0.3,
    aliphatic_min: float = 40.0,          # thermostable: ~85-100
    boman_max: float = 4.5,               # PPI-prone above ~2.5
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
        pp = protparam_metrics(
            seq, ph=design_ph,
            n_term_pad=n_term_pad, c_term_pad=c_term_pad,
        )
        # Filter on the ROBUST full-HH charge (all 7 ionizables + termini).
        # The minimalist 'no_HIS' (still computed below as a diagnostic
        # column) is too lenient because it misses Cys/Tyr at high pH.
        if pp.charge_at_pH_full_HH >= net_charge_max:
            reasons.append(
                f"net_charge_full_HH={pp.charge_at_pH_full_HH:.2f} >= {net_charge_max}"
            )
        if pp.charge_at_pH_full_HH <= net_charge_min:
            reasons.append(
                f"net_charge_full_HH={pp.charge_at_pH_full_HH:.2f} <= {net_charge_min}"
            )
        if not (pi_min <= pp.pi <= pi_max):
            reasons.append(f"pI={pp.pi:.2f} outside [{pi_min}, {pi_max}]")

        # Light de-novo filters on cheap sequence-only metrics. Each
        # threshold is set so they catch *truly broken* designs only —
        # the production pool typically passes all of these comfortably.
        if pp.instability_index >= instability_max:
            reasons.append(
                f"instability_index={pp.instability_index:.1f} >= {instability_max}"
            )
        if not (gravy_min <= pp.gravy <= gravy_max):
            reasons.append(
                f"GRAVY={pp.gravy:+.3f} outside [{gravy_min}, {gravy_max}]"
            )
        if pp.aliphatic_index < aliphatic_min:
            reasons.append(
                f"aliphatic_index={pp.aliphatic_index:.1f} < {aliphatic_min}"
            )
        if pp.boman_index >= boman_max:
            reasons.append(
                f"boman_index={pp.boman_index:.2f} >= {boman_max}"
            )

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
            # Robust filter charge (all 7 ionizable groups).
            "net_charge_full_HH": pp.charge_at_pH_full_HH,
            # Diagnostic charge variants for back-compat / sensitivity analysis.
            "net_charge_no_HIS": pp.charge_at_pH7_no_HIS,
            "net_charge_with_HIS_HH": pp.charge_at_pH7,        # Biopython
            "net_charge_HIS_half": pp.charge_at_pH7_HIS_half,  # HIS = +0.5
            "net_charge_DE_KR_only": pp.charge_at_pH_DE_KR_only,  # legacy
            "design_ph": design_ph,
            "instability_index": pp.instability_index,
            "gravy": pp.gravy,
            "pi": pp.pi,
            # Cheap sequence-only metrics added 2026-05-04 for diagnostic +
            # light filtering. All sub-ms.
            "aliphatic_index": pp.aliphatic_index,
            "boman_index": pp.boman_index,
            "aromaticity": pp.aromaticity,
            "flexibility_mean_seq": pp.flexibility_mean if pp.flexibility_mean is not None else float("nan"),
            "helix_frac_seq": pp.helix_frac_seq,
            "turn_frac_seq": pp.turn_frac_seq,
            "sheet_frac_seq": pp.sheet_frac_seq,
            "molecular_weight": pp.molecular_weight,
            "extinction_280nm_no_disulfide": pp.extinction_280nm_no_disulfide,
            "extinction_280nm_disulfide": pp.extinction_280nm_disulfide,
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
    catalytic_his_resnos: Optional[Iterable[int]] = None,
    fixed_resnos: Iterable[int] = (),
    clash_filter: bool = True,
    clash_severe_distance: float = 1.5,
    # For DFI per-class summary — pre-computed once on the seed
    # and broadcast to all designs (DFI is design-invariant for
    # fixed-backbone runs).
    seed_dfi_metrics: Optional[dict] = None,
) -> Path:
    """Apply h-bond + SAP-proxy structural filter."""
    if catalytic_his_resnos is None:
        catalytic_his_resnos = CATALYTIC_HIS_RESNOS
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(survivors_seq_tsv, sep="\t")
    LOGGER.info("stage_struct_filter: input n=%d", len(df))

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    # Build worker arg tuples. Per-design work (sap + clash + preorg +
    # ligand_int + h-bond detection) is independent → Pool-able.
    cat_his_list = list(catalytic_his_resnos)
    fixed_list = list(fixed_resnos)
    work_args: list[tuple] = []
    for _, row in df.iterrows():
        cid = row["id"]
        pdb = pdb_map.get(cid)
        work_args.append((
            cid, pdb, cat_his_list, fixed_list,
            clash_severe_distance, sap_max_threshold,
            seed_dfi_metrics,
        ))

    from protein_chisel.utils.resources import pool_workers
    n_workers = pool_workers(len(work_args), cap=8)
    LOGGER.info(
        "stage_struct_filter: input n=%d, n_workers=%d (parallel SAP+clash+"
        "preorg+ligand_int)", len(df), n_workers,
    )

    if n_workers == 1:
        results = [_struct_filter_worker(a) for a in work_args]
    else:
        from multiprocessing import Pool
        with Pool(n_workers) as pool:
            results = pool.map(_struct_filter_worker, work_args, chunksize=1)
    by_cid = {cid: (row, hbonds, reasons) for cid, row, hbonds, reasons in results}

    rows: list[dict] = []
    hbond_rows: list[dict] = []
    for _, row in df.iterrows():
        cid = row["id"]
        wrow, hbonds_, reasons = by_cid.get(cid, ({}, [], ["worker_missing"]))
        hbond_rows.extend(hbonds_)
        # Apply severe-clash filter (worker doesn't have clash_filter flag)
        if clash_filter and wrow.get("clash__has_severe"):
            reasons = list(reasons) + [
                f"severe clash (n_cat={wrow.get('clash__n_to_catalytic',0)}, "
                f"n_lig={wrow.get('clash__n_to_ligand',0)}, "
                f"detail={wrow.get('clash__detail','')})"
            ]
        rows.append({
            **row.to_dict(),
            **wrow,
            "passed_struct_filter": not reasons,
            "struct_fail": "; ".join(reasons),
        })

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
    wt_fitness: Optional[float] = None,
) -> Path:
    """Score each survivor's fitness from cached PLM marginals.

    If ``wt_fitness`` is provided, also computes
    ``fitness__delta_vs_wt = design - wt_fitness`` per row.
    Positive = design is more PLM-natural per residue than WT;
    negative = design is less natural than WT.
    """
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
    if wt_fitness is not None and "fitness__logp_fused_mean" in scored.columns:
        scored["fitness__delta_vs_wt"] = (
            scored["fitness__logp_fused_mean"] - wt_fitness
        )
        scored["fitness__wt_logp_fused"] = float(wt_fitness)
    out_path = out_dir / "scored.tsv"
    scored.to_csv(out_path, sep="\t", index=False)
    log_msg = (
        f"stage_fitness_score: cache size now {len(fitness_cache)}; "
        f"logp_fused mean={float(scored['fitness__logp_fused_mean'].mean()):.3f}"
    )
    if wt_fitness is not None:
        log_msg += (
            f", wt={wt_fitness:.3f}, "
            f"delta_vs_wt mean={float(scored['fitness__delta_vs_wt'].mean()):+.3f}"
        )
    LOGGER.info(log_msg)
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
        pockets_subdir = work_dir / "design_out" / "pockets"
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
            pockets_dir=pockets_subdir if pockets_subdir.is_dir() else None,
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
    pockets_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Pick the fpocket pocket containing the active site.

    fpocket's ``<stem>_out.pdb`` lists every alpha sphere as a HETATM
    record with res_name = "STP" and a unique res_seq per pocket
    (1, 2, 3...). For each pocket, count alpha spheres within
    ``distance_cutoff`` of any catalytic CA atom. Pick the pocket
    with the highest count and return its info.txt entries plus a
    ``mean_alpha_sphere_distance_to_catalytic`` proxy for "how
    centered on the active site".

    If ``pockets_dir`` is given (the fpocket ``pockets/`` subdir which
    holds ``pocketN_vert.pqr``), we additionally compute per-sphere
    bottleneck-radius statistics from the chosen pocket's PQR file.
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

    # Bottleneck-radius proxy: read the chosen pocket's PQR (per-sphere
    # radius) and compute narrowest-passage statistics. Spheres in the
    # upper distance-from-catalytic quartile are the lip/exit; their
    # smallest radius is what a substrate must squeeze past on the way
    # in. Cheap (~ms per design, single PQR read).
    if pockets_dir is not None:
        idx = best["pocket_idx"]
        pqr = pockets_dir / f"pocket{idx}_vert.pqr"
        if pqr.is_file():
            extras = _compute_pocket_radius_stats(pqr, cat_arr)
            out.update(extras)
    return out


def _compute_pocket_radius_stats(
    pqr_path: Path, cat_arr: np.ndarray,
) -> dict[str, float]:
    """Read fpocket ``pocketN_vert.pqr`` and compute radius-based
    bottleneck stats.

    PQR format (fpocket): atom name in cols 12-16, x,y,z in
    cols 30:38, 38:46, 46:54, then charge then radius (last float).
    Atom name "O" = polar sphere; "C" = apolar sphere.

    Returns:
        min_alpha_sphere_radius:
            absolute smallest sphere in the pocket. fpocket clamps to
            its [3.0, 6.0] default range so this is rarely below 3.4
            but flags egregious narrow-points.
        alpha_sphere_radius_p10:
            10th-percentile radius — robust narrow-point measure.
        bottleneck_radius:
            min radius among spheres in the upper distance-from-
            catalytic quartile (>= 75th percentile distance). These
            spheres line the channel exit / mouth of the pocket; their
            min radius approximates the constriction a substrate must
            pass through on the way to the catalytic center. This is
            the metric to consult when bulky R/K residues block the
            active site.
        polar_alpha_sphere_proportion:
            fraction of spheres tagged "O" (polar). Complements
            ``apolar_alpha_sphere_proportion`` from info.txt.
        n_rim_spheres / rim_distance_threshold:
            diagnostics for the bottleneck calculation.
    """
    radii: list[float] = []
    coords: list[np.ndarray] = []
    polar: list[bool] = []
    with open(pqr_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            try:
                xyz = np.array([
                    float(line[30:38]), float(line[38:46]), float(line[46:54]),
                ])
                tail = line[54:].split()
                radius = float(tail[-1])
            except (ValueError, IndexError):
                continue
            atom_name = line[12:16].strip()
            radii.append(radius)
            coords.append(xyz)
            polar.append(atom_name.startswith("O"))
    if not radii:
        return {}
    r = np.array(radii)
    xyz = np.array(coords)
    pol = np.array(polar)
    # Min distance per sphere to nearest catalytic CA
    d = np.linalg.norm(
        xyz[:, None, :] - cat_arr[None, :, :], axis=-1,
    ).min(axis=1)
    p75 = float(np.percentile(d, 75)) if len(d) >= 4 else float(d.max())
    rim_mask = d >= p75
    if rim_mask.sum() == 0:
        rim_mask = np.ones_like(d, dtype=bool)
    return {
        "min_alpha_sphere_radius": float(r.min()),
        "alpha_sphere_radius_p10": float(np.percentile(r, 10)),
        "bottleneck_radius": float(r[rim_mask].min()),
        "polar_alpha_sphere_proportion": float(pol.mean()),
        "n_rim_spheres": int(rim_mask.sum()),
        "rim_distance_threshold": p75,
    }


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
            m = re.match(r"^\s*([A-Za-z0-9_/.\- ]+?):\s*([+-]?[\d.eE+-]+)", line)
            if m and cur:
                key = (
                    m.group(1).strip().lower()
                    .replace(" ", "_").replace(".", "").replace("-", "_")
                )
                # Collapse runs of underscores from "Cent. of mass - Alpha"
                while "__" in key:
                    key = key.replace("__", "_")
                key = key.strip("_")
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
            m = re.match(r"^\s*([A-Za-z0-9_/.\- ]+?):\s*([+-]?[\d.eE+-]+)", line)
            if m and cur:
                key = (
                    m.group(1).strip().lower()
                    .replace(" ", "_").replace(".", "").replace("-", "_")
                )
                # Collapse runs of underscores from "Cent. of mass - Alpha"
                while "__" in key:
                    key = key.replace("__", "_")
                key = key.strip("_")
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


def annotate_seed_tunnel_residues(
    *,
    seed_pdb: Path,
    out_path: Path,
    catalytic_resnos: Iterable[int],
    chain: str = CHAIN,
    proximity_cutoff: float = 6.0,
) -> Path:
    """Run fpocket once on the seed PDB and write a per-residue
    annotation TSV: ``resno, is_tunnel_lining, min_dist_to_alpha_sphere``.

    A residue is "tunnel-lining" if any of its sidechain (or CA) atoms
    sits within ``proximity_cutoff`` Å of any active-site alpha-sphere.
    These are the positions where bulky/charged residues most directly
    affect channel width and pocket accessibility.

    Cheap: one fpocket run on the seed (~0.6 s) plus per-atom distance
    matrix (microseconds). Output is a 4-column TSV consumed by
    ``scripts/audit_pocket_metrics.py`` and any downstream PositionTable
    extension.
    """
    work_dir = out_path.parent / "_seed_fpocket_workspace"
    work_dir.mkdir(parents=True, exist_ok=True)
    # Holo seed PDBs typically include the substrate/cofactor as HETATM,
    # which fpocket excludes from pocket detection — that collapses the
    # active-site cavity to a tiny vestige. Strip HETATM to an apo PDB
    # so the active-site cavity is detected the same way as the
    # in-cycle designed PDBs (which have no ligand).
    apo_pdb = work_dir / "seed_apo.pdb"
    with open(seed_pdb) as src, open(apo_pdb, "w") as dst:
        for line in src:
            if line.startswith("HETATM"):
                continue
            dst.write(line)
    info = _run_fpocket(
        apo_pdb, work_dir, catalytic_resnos=catalytic_resnos, chain=chain,
    )
    if info is None:
        LOGGER.warning("seed fpocket failed; tunnel annotation will be empty")
        pd.DataFrame(columns=[
            "resno", "is_tunnel_lining", "min_dist_to_alpha_sphere",
            "is_catalytic",
        ]).to_csv(out_path, sep="\t", index=False)
        return out_path

    pidx = info["pocket_idx"]
    pqr_path = work_dir / "design_out" / "pockets" / f"pocket{pidx}_vert.pqr"
    sphere_xyz: list[np.ndarray] = []
    if pqr_path.is_file():
        with open(pqr_path) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                try:
                    sphere_xyz.append(np.array([
                        float(line[30:38]), float(line[38:46]), float(line[46:54]),
                    ]))
                except ValueError:
                    continue
    if not sphere_xyz:
        LOGGER.warning("seed fpocket: no alpha spheres for pocket %d", pidx)
        pd.DataFrame(columns=[
            "resno", "is_tunnel_lining", "min_dist_to_alpha_sphere",
            "is_catalytic",
        ]).to_csv(out_path, sep="\t", index=False)
        return out_path
    sph_arr = np.array(sphere_xyz)

    # Build per-residue atom xyz lists from the seed PDB.
    cat_set = set(int(r) for r in catalytic_resnos)
    res_atoms: dict[int, list[np.ndarray]] = {}
    with open(seed_pdb) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[21].strip() != chain:
                continue
            try:
                resno = int(line[22:26].strip())
                xyz = np.array([
                    float(line[30:38]), float(line[38:46]), float(line[46:54]),
                ])
            except ValueError:
                continue
            res_atoms.setdefault(resno, []).append(xyz)

    rows = []
    for resno in sorted(res_atoms):
        atoms = np.array(res_atoms[resno])
        # Min distance from any atom of this residue to any alpha sphere
        d = np.linalg.norm(
            atoms[:, None, :] - sph_arr[None, :, :], axis=-1,
        ).min()
        rows.append({
            "resno": int(resno),
            "is_tunnel_lining": bool(d <= proximity_cutoff),
            "min_dist_to_alpha_sphere": float(d),
            "is_catalytic": bool(resno in cat_set),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    LOGGER.info(
        "seed tunnel annotation: %d residues, %d tunnel-lining (cutoff=%.1f Å)",
        len(df), int(df["is_tunnel_lining"].sum()), proximity_cutoff,
    )
    # Workspace is a few hundred KB; keep it for debugging.
    return out_path


def _struct_filter_worker(args: tuple) -> tuple:
    """Module-level per-design struct-filter worker (Pool-friendly).

    Args tuple:
        (cid, pdb_path, catalytic_his_resnos, fixed_resnos,
         clash_severe_distance, sap_max_threshold)
    Returns: (cid, row_dict, hbond_list, struct_fail_reasons)
    """
    (cid, pdb, cat_his, fixed_, sev_dist, sap_max_thr,
     seed_dfi_metrics_) = args
    if pdb is None or not Path(pdb).is_file():
        # Schema-consistent empty row — every key the parent loop
        # writes must be present so missing-PDB rows don't NaN-leak
        # into the rest of the TSV. Codex r2 caught two missing keys
        # (clash__detail string + ligand_int__* numeric panel).
        empty_row = {
            "n_hbonds_to_cat_his": 0,
            "sap_max": float("nan"), "sap_mean": float("nan"),
            "sap_p95": float("nan"),
            "clash__n_total": 0, "clash__n_to_catalytic": 0,
            "clash__n_to_ligand": 0, "clash__has_severe": 0,
            "clash__detail": "",
            # ligand_int__* — full panel default to 0 / 0.0
            "ligand_int__n_total": 0, "ligand_int__strength_total": 0.0,
            "ligand_int__n_hbond": 0, "ligand_int__strength_hbond": 0.0,
            "ligand_int__n_salt_bridge": 0,
            "ligand_int__strength_salt_bridge": 0.0,
            "ligand_int__n_pi_pi": 0, "ligand_int__strength_pi_pi": 0.0,
            "ligand_int__n_pi_cation": 0,
            "ligand_int__strength_pi_cation": 0.0,
            "ligand_int__n_hydrophobic": 0,
            "ligand_int__strength_hydrophobic": 0.0,
            "ligand_int__n_vdw_clash": 0,
            "ligand_int__strength_vdw_clash": 0.0,
            "preorg__n_hbonds_to_cat": 0,
            "preorg__n_salt_bridges_to_cat": 0,
            "preorg__n_pi_to_cat": 0,
            "preorg__n_hbonds_within_shells": 0,
            "preorg__strength_total": 0.0,
            "preorg__interactome_density": 0.0,
            "preorg__n_first_shell": 0, "preorg__n_second_shell": 0,
            "struct_fail_reason": f"pdb_missing: {pdb}",
            "_passed": False,
        }
        return cid, empty_row, [], [f"pdb_missing: {pdb}"]
    # Lazy imports inside worker so each Pool process re-imports cleanly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.geometric_interactions import detect_interactions
    from protein_chisel.scoring.preorganization import preorganization_score
    from protein_chisel.structure import detect_clashes

    # H-bond to catalytic HIS
    hbonds = _detect_hbond_to_his_sidechain(pdb, cat_his)
    hbond_dicts = [{"id": cid, **h} for h in hbonds]

    # Geometric interaction panel (~50 ms)
    panel = detect_interactions(pdb, chain=CHAIN, selection="protein_vs_ligand")
    gi_metrics = panel.to_dict("ligand_int__")

    # SAP proxy via freesasa
    sap = _compute_sap_proxy(pdb) or {}
    sap_max = sap.get("sap_max", float("nan"))

    # Clash detection (catalytic + ligand vs designed sidechains)
    clash = detect_clashes(
        pdb, catalytic_resnos=fixed_, chain=CHAIN,
        severe_distance=sev_dist,
    )
    clash_dict = clash.to_dict()

    # Preorganization (~10-20 ms)
    try:
        preorg_metrics = preorganization_score(
            pdb, catalytic_resnos=list(fixed_), chain=CHAIN,
        )
    except Exception as e:    # pragma: no cover
        LOGGER.warning("preorganization failed for %s: %s", cid, e)
        preorg_metrics = {
            "preorg__n_hbonds_to_cat": 0,
            "preorg__n_salt_bridges_to_cat": 0,
            "preorg__n_pi_to_cat": 0,
            "preorg__n_hbonds_within_shells": 0,
            "preorg__strength_total": 0.0,
            "preorg__interactome_density": 0.0,
            "preorg__n_first_shell": 0, "preorg__n_second_shell": 0,
        }

    # Filter reasons
    reasons: list[str] = []
    if len(hbonds) < 1:
        reasons.append("no h-bonds to catalytic HIS")
    if sap_max == sap_max and sap_max > sap_max_thr:
        reasons.append(f"sap_max={sap_max:.2f} > {sap_max_thr}")
    # Severe-clash filter applied by caller (it has the boolean flag).

    row = {
        "n_hbonds_to_cat_his": len(hbonds),
        **gi_metrics,
        "sap_max": sap_max,
        "sap_mean": sap.get("sap_mean", float("nan")),
        "sap_p95": sap.get("sap_p95", float("nan")),
        **clash_dict,
        **preorg_metrics,
        **(seed_dfi_metrics_ or {}),
    }
    return cid, row, hbond_dicts, reasons


def _fpocket_worker(args: tuple) -> tuple:
    """Module-level fpocket worker — must be picklable for Pool().

    Args tuple: (cid, pdb_path, out_dir, catalytic_resnos_list, chain).
    Returns (cid, info_dict_or_None).
    """
    cid_, pdb_, out_, cat_, chain_ = args
    if pdb_ is None:
        return cid_, None
    try:
        return cid_, _run_fpocket(
            pdb_, out_, catalytic_resnos=cat_, chain=chain_,
        )
    except Exception as e:    # pragma: no cover -- fpocket can flake
        LOGGER.warning("fpocket failed for %s: %s", cid_, e)
        return cid_, None


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
    fpocket_dir = out_dir / "per_design_fpocket"
    fpocket_dir.mkdir(exist_ok=True)

    # Parallelize fpocket invocations across CPUs (uses centralized
    # resource detection). Each call is a standalone fpocket subprocess
    # writing to its own temp dir.
    from protein_chisel.utils.resources import pool_workers, detect_n_cpus
    cpus_available, _src = detect_n_cpus()
    n_workers = pool_workers(len(df), cpu_budget=cpus_available, cap=8)
    LOGGER.info(
        "stage_fpocket_rank: input n=%d (active-site constraint=%s, "
        "n_workers=%d/%d cpus)",
        len(df), "yes" if catalytic_resnos else "no",
        n_workers, cpus_available,
    )

    # Build worker arg tuples (only picklable types)
    cat_list = list(catalytic_resnos) if catalytic_resnos else []
    work_args: list[tuple] = []
    for _, row in df.iterrows():
        cid = row["id"]
        pdb = pdb_map.get(cid)
        work_args.append((cid, pdb, fpocket_dir / cid, cat_list, chain))

    if n_workers == 1 or len(work_args) <= 2:
        # Small pool / serial path — avoids Pool startup overhead.
        infos = dict(_fpocket_worker(a) for a in work_args)
    else:
        from multiprocessing import Pool
        with Pool(n_workers) as pool:
            infos = dict(pool.map(_fpocket_worker, work_args, chunksize=1))

    rows = []
    for _, row in df.iterrows():
        cid = row["id"]
        info = infos.get(cid)
        # Helper to read keys safely with NaN fallback
        def _g(key, default=float("nan")):
            return info.get(key, default) if info else default
        # Polar / apolar atom percentages. info.txt gives only
        # ``proportion_of_polar_atoms`` (a percent already, not a
        # fraction). Apolar pct is the complement.
        polar_pct = _g("proportion_of_polar_atoms", float("nan"))
        apolar_pct = (
            (100.0 - polar_pct)
            if isinstance(polar_pct, (int, float)) and polar_pct == polar_pct
            else float("nan")
        )
        rows.append({
            **row.to_dict(),
            # ---- existing core metrics (unchanged) -----------------
            "fpocket__druggability": _g("druggability_score", 0.0),
            "fpocket__volume": _g("volume", 0.0),
            "fpocket__mean_alpha_sphere_radius":
                _g("mean_alpha_sphere_radius", 0.0),
            "fpocket__alpha_sphere_density": _g("alpha_sphere_density", 0.0),
            "fpocket__n_alpha_spheres_near_catalytic":
                _g("n_alpha_spheres_near_catalytic", 0),
            "fpocket__mean_alpha_sphere_dist_to_catalytic":
                _g("mean_alpha_sphere_dist_to_catalytic"),
            "fpocket__n_pockets_found": 1 if info else 0,
            # ---- pocket character (info.txt) -----------------------
            "fpocket__score": _g("score"),
            "fpocket__n_alpha_spheres": _g("number_of_alpha_spheres"),
            "fpocket__total_sasa": _g("total_sasa"),
            "fpocket__polar_sasa": _g("polar_sasa"),
            "fpocket__apolar_sasa": _g("apolar_sasa"),
            "fpocket__hydrophobicity_score": _g("hydrophobicity_score"),
            "fpocket__polarity_score": _g("polarity_score"),
            "fpocket__charge_score": _g("charge_score"),
            "fpocket__volume_score": _g("volume_score"),
            "fpocket__polar_atoms_pct": polar_pct,
            "fpocket__apolar_atoms_pct": apolar_pct,
            "fpocket__apolar_alpha_sphere_proportion":
                _g("apolar_alpha_sphere_proportion"),
            "fpocket__mean_local_hydrophobic_density":
                _g("mean_local_hydrophobic_density"),
            "fpocket__mean_alpha_sphere_solvent_acc":
                _g("mean_alp_sph_solvent_access"),
            "fpocket__cent_of_mass_alpha_sphere_max_dist":
                _g("cent_of_mass_alpha_sphere_max_dist"),
            # ---- bottleneck / channel stats (from PQR) -------------
            "fpocket__bottleneck_radius": _g("bottleneck_radius"),
            "fpocket__min_alpha_sphere_radius": _g("min_alpha_sphere_radius"),
            "fpocket__alpha_sphere_radius_p10":
                _g("alpha_sphere_radius_p10"),
            "fpocket__polar_alpha_sphere_proportion":
                _g("polar_alpha_sphere_proportion"),
            "fpocket__n_rim_spheres": _g("n_rim_spheres", 0),
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
# Optional final-stage enrichment: CMS (cross-sif) + Rosetta DDG
# ----------------------------------------------------------------------


def stage_cms_final(
    *,
    topk_tsv: Path,
    pdb_map: dict[str, Path],
    out_dir: Path,
) -> Path:
    """Add Coventry-distance-weighted Contact Molecular Surface to top-K.

    Uses ``protein_chisel.tools.contact_ms.contact_ms_protein_ligand``,
    which depends on ``py_contact_ms`` (only present in esmc.sif). When
    we're already running inside esmc.sif this is just an in-process
    call; otherwise we shell out via :func:`esmc_call`.

    ~3-4 s per design — only run on the final top-K, never per-cycle.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(topk_tsv, sep="\t")
    if len(df) == 0:
        LOGGER.warning("stage_cms_final: empty top-K, nothing to do")
        df.to_csv(out_dir / "topk_with_cms.tsv", sep="\t", index=False)
        return out_dir / "topk_with_cms.tsv"

    # Try in-process first (we're inside esmc.sif); fall back to a
    # cross-sif batch call if py_contact_ms isn't importable here.
    try:
        from protein_chisel.tools.contact_ms import contact_ms_protein_ligand
        cms_vals: list[float] = []
        for _, row in df.iterrows():
            pdb = pdb_map.get(row["id"])
            if pdb is None or not pdb.is_file():
                cms_vals.append(float("nan"))
                continue
            res = contact_ms_protein_ligand(pdb)
            cms_vals.append(float(res.total_cms))
        df["cms__total"] = cms_vals
        LOGGER.info("stage_cms_final: in-process; mean CMS=%.2f",
                     float(np.nanmean(cms_vals)) if cms_vals else 0.0)
    except ImportError:
        LOGGER.info("stage_cms_final: py_contact_ms not local -> esmc.sif")
        from protein_chisel.utils.apptainer import esmc_call
        # Build a tiny in-container worker that consumes a JSON list of
        # PDB paths and emits {id: cms} as JSON to stdout.
        pdb_paths = {row["id"]: str(pdb_map.get(row["id"])) for _, row in df.iterrows()}
        worker = (
            "import json, sys\n"
            "sys.path.insert(0, '/home/woodbuse/codebase_projects/protein_chisel/src')\n"
            "from protein_chisel.tools.contact_ms import contact_ms_protein_ligand\n"
            "items = json.loads(sys.argv[1])\n"
            "out = {}\n"
            "for k, p in items.items():\n"
            "    if not p:\n"
            "        out[k] = float('nan')\n"
            "        continue\n"
            "    try:\n"
            "        out[k] = float(contact_ms_protein_ligand(p).total_cms)\n"
            "    except Exception:\n"
            "        out[k] = float('nan')\n"
            "print('<<<CMS_JSON_BEGIN>>>'); print(json.dumps(out)); print('<<<CMS_JSON_END>>>')\n"
        )
        result = (
            esmc_call(nv=False)
            .with_bind("/home/woodbuse/codebase_projects/protein_chisel")
            .run(["python", "-c", worker, json.dumps(pdb_paths)],
                 capture_output=True, check=True)
        )
        blob = (result.stdout
                .split("<<<CMS_JSON_BEGIN>>>", 1)[1]
                .split("<<<CMS_JSON_END>>>", 1)[0]
                .strip())
        cms_map = json.loads(blob)
        df["cms__total"] = [float(cms_map.get(rid, float("nan"))) for rid in df["id"]]
        LOGGER.info("stage_cms_final: cross-sif; mean CMS=%.2f",
                     float(df["cms__total"].mean()))

    out_path = out_dir / "topk_with_cms.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def _sanitize_pdb_for_rosetta(src: Path, dst: Path) -> Path:
    """Normalize Rosetta-extended residue names (HIS_D, HIS_E) -> HIS.

    The pipeline writes 5-char residue names so downstream packers can
    distinguish HIS tautomers, but ``pose_from_file`` rejects those
    names. Rewrite cols 17-21 to a standard ' HIS ' label.
    """
    out_lines: list[str] = []
    for line in src.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 21:
            rn5 = line[16:21]
            if rn5 in ("HIS_D", "HIS_E"):
                line = line[:16] + " HIS " + line[21:]
        out_lines.append(line)
    dst.write_text("\n".join(out_lines) + "\n")
    return dst


def stage_rosetta_final(
    *,
    topk_tsv: Path,
    pdb_map: dict[str, Path],
    out_dir: Path,
    ligand_params: Path,
    key_atoms: Iterable[str] = ("P1", "O5", "O1", "O4"),
) -> Path:
    """Add Rosetta no-repack DDG (and other comprehensive metrics) to top-K.

    ~10 s/design: only viable as a final-stage enrichment over the
    diversity-selected top-K. Gated behind ``--rosetta_final``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(topk_tsv, sep="\t")
    if len(df) == 0:
        LOGGER.warning("stage_rosetta_final: empty top-K, nothing to do")
        df.to_csv(out_dir / "topk_with_rosetta.tsv", sep="\t", index=False)
        return out_dir / "topk_with_rosetta.tsv"

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.rosetta_metrics import (
        compute_rosetta_metrics, RosettaMetricsConfig,
    )

    sanitize_dir = out_dir / "_sanitized_pdbs"
    sanitize_dir.mkdir(exist_ok=True)
    rows: list[dict] = []
    cfg = RosettaMetricsConfig(include_ec=False, include_sap=False)
    for _, row in df.iterrows():
        pdb = pdb_map.get(row["id"])
        if pdb is None or not pdb.is_file():
            rows.append({"id": row["id"], "rosetta__ddg": float("nan")})
            continue
        sanitized = _sanitize_pdb_for_rosetta(pdb, sanitize_dir / pdb.name)
        try:
            res = compute_rosetta_metrics(
                pdb_path=sanitized,
                ligand_params=ligand_params,
                key_atoms=list(key_atoms),
                config=cfg,
            )
            rows.append({
                "id": row["id"],
                "rosetta__ddg": float(res.ddg),
                "rosetta__contact_molecular_surface": float(res.contact_molecular_surface),
                "rosetta__ligand_interface_energy": float(res.ligand_interface_energy),
                "rosetta__total_energy": float(res.total_rosetta_energy_metric),
                "rosetta__n_hbonds_to_ligand": int(res.n_hbonds_to_ligand),
            })
        except Exception as e:  # pragma: no cover -- log + skip per-design
            LOGGER.warning("rosetta_final failed for %s: %s", row["id"], e)
            rows.append({"id": row["id"], "rosetta__ddg": float("nan")})

    rosetta_df = pd.DataFrame(rows)
    merged = df.merge(rosetta_df, on="id", how="left")
    out_path = out_dir / "topk_with_rosetta.tsv"
    merged.to_csv(out_path, sep="\t", index=False)
    LOGGER.info("stage_rosetta_final: enriched %d top-K with rosetta__ddg "
                 "(mean=%.2f)", len(merged),
                 float(merged["rosetta__ddg"].mean(skipna=True)))
    return out_path


def stage_protonate_final_topk(
    *,
    topk_pdb_dir: Path,
    seed_pdb: Path,
    ligand_params: Path,
    pyrosetta_sif: Path,
    out_dir: Path,
    ptm: str = "",
) -> Path:
    """Hydrate every top-K PDB via PyRosetta and write a downstream-clean copy.

    For each ``*.pdb`` under ``topk_pdb_dir``:
      1. Loads the PDB into a PyRosetta pose (places ideal hydrogens on
         every residue based on the residue type — catalytic tautomers
         like HIS_D are preserved by the input residue label).
      2. Dumps the pose to a ``.rosetta.pdb`` intermediate.
      3. Combines that with the seed's REMARK 666 + ligand HETATM block
         (incl. seed hydrogens), normalizes 5-char Rosetta tautomer
         labels back to standard 3-char names, emits a REMARK 668
         protonation-state table paired by index to REMARK 666, and
         writes ``<stem>.protonated.pdb`` into ``out_dir``.

    Runs INSIDE pyrosetta.sif via subprocess. Off the hot path: only
    runs once per pipeline, on the final top-K (~50 PDBs).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not topk_pdb_dir.is_dir():
        LOGGER.warning("stage_protonate_final_topk: %s not a directory; skipping",
                        topk_pdb_dir)
        return out_dir
    if not pyrosetta_sif.is_file():
        LOGGER.warning("stage_protonate_final_topk: pyrosetta_sif %s not found; "
                        "skipping (designs will keep HIS_D-style labels)",
                        pyrosetta_sif)
        return out_dir

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.utils.apptainer import pyrosetta_call, in_apptainer

    # Nested-apptainer doesn't work on this cluster: when stage 3 runs
    # inside universal.sif, the apptainer binary isn't on PATH inside
    # the container so spawning pyrosetta.sif from here fails. The
    # production sbatch (run_iterative_design_v2.sbatch) handles this
    # by running protonate_final as a separate post-stage outside any
    # container. Detect the in-container case and skip with a clear
    # log line so the user knows where the missing protonation is.
    if in_apptainer():
        LOGGER.info(
            "stage_protonate_final_topk: detected we are running INSIDE "
            "a container; skipping in-driver invocation. "
            "run_iterative_design_v2.sbatch invokes the protonation as "
            "a separate stage 4 after this driver returns. "
            "Look for *.protonated.pdb in %s after the sbatch finishes.",
            out_dir,
        )
        return out_dir

    n_pdbs = sum(1 for p in topk_pdb_dir.iterdir() if p.suffix == ".pdb")
    LOGGER.info("stage_protonate_final_topk: hydrating %d top-K PDBs via PyRosetta -> %s",
                 n_pdbs, out_dir)

    summary_json = out_dir / "_protonation_summary.json"
    driver_script = (
        Path(__file__).resolve().parents[0] / "protonate_final_topk.py"
    )
    call = (
        pyrosetta_call()
        .with_bind(str(topk_pdb_dir.parent.resolve()))
        .with_bind(str(out_dir.resolve()))
        .with_bind(str(Path(seed_pdb).resolve().parent))
        .with_bind(str(Path(ligand_params).resolve().parent))
    )
    args = [
        "--topk_dir", str(topk_pdb_dir.resolve()),
        "--seed_pdb", str(Path(seed_pdb).resolve()),
        "--ligand_params", str(Path(ligand_params).resolve()),
        "--out_dir", str(out_dir.resolve()),
        "--summary_json", str(summary_json.resolve()),
    ]
    if ptm:
        args += ["--ptm", ptm]
    try:
        result = call.run_python(driver_script, args, check=True)
        LOGGER.info("stage_protonate_final_topk: done (%d clean PDBs at %s)",
                     n_pdbs, out_dir)
        if result.stdout:
            LOGGER.debug("stage_protonate_final_topk stdout: %s",
                          result.stdout[:2000])
    except Exception as e:
        LOGGER.error("stage_protonate_final_topk failed: %s", e)
    return out_dir


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
    seed_dfi_metrics: Optional[dict] = None,
    wt_fitness: Optional[float] = None,
    position_table_df=None,           # for first-shell diversity injection
    omit_AA_per_residue: Optional[dict[str, str]] = None,
    catalytic_his_resnos: Optional[Iterable[int]] = None,
    balance_z_threshold: float = 2.0,
    design_ph: float = 7.5,
    instability_max: float = 60.0,
    gravy_min: float = -0.8,
    gravy_max: float = 0.3,
    aliphatic_min: float = 40.0,
    boman_max: float = 4.5,
    n_term_pad: str = "",
    c_term_pad: str = "",
    omit_M_at_pos1: bool = True,
) -> tuple[Optional[pd.DataFrame], dict[str, Path]]:
    """Run ONE iteration cycle. Returns (ranked DataFrame, pdb_map)."""
    if catalytic_his_resnos is None:
        catalytic_his_resnos = CATALYTIC_HIS_RESNOS
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
            max_augmented_fraction=cycle_cfg.consensus_max_fraction,
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

    # ---- 0b. Class-balanced compensatory bias_AA -------------------
    # Built from the previous cycle's survivor pool. For each AA class
    # (negatively_charged, hydrophobic_aliphatic, ...), if one member is
    # over-rep (z > 2) and another is under-rep (z < -2), swap: down-
    # weight the over-rep AA AND up-weight the under-rep AA. Address
    # cases like "E z=+5, D z=-2": instead of just suppressing E (which
    # only reduces total negative charge), encourage D to take its place.
    bias_AA_str = ""
    if survivors_prev is not None and len(survivors_prev) > 0:
        from protein_chisel.expression.aa_class_balance import (
            compute_class_balanced_bias_AA,
        )
        # Pool survivors into one mega-sequence: this gives a count-
        # weighted average composition (each survivor contributes equally
        # since they're the same length L).
        pool_seq = "".join(survivors_prev["sequence"].astype(str).tolist())
        # exclude_aas matches cycle_cfg.omit_AA (default "X" or "CX") so
        # we don't try to up-weight an AA the sampler can't pick anyway.
        excl = "".join(c for c in cycle_cfg.omit_AA.upper() if c != "X")
        balance_telem = compute_class_balanced_bias_AA(
            pool_seq,
            reference="swissprot_ec3_hydrolases_2026_01",
            exclude_aas=excl,
            # Threshold 2.0: only fire swaps when BOTH ends of the
            # class imbalance are clearly extreme (over-rep > +2σ AND
            # under-rep < −2σ). Keeps the bias_AA quiet under moderate
            # imbalance so PLM + structure can drive composition; only
            # corrects truly pathological pools. (User preference;
            # earlier 1.5 was too aggressive for moderate cases like
            # E z=+5 paired with D z=−1.7.)
            balance_z_threshold=balance_z_threshold,
            over_z_threshold=3.0,
            max_bias_nats=2.5,
            bias_per_z=0.4,
        )
        bias_AA_str = balance_telem.bias_AA_string
        with open(bias_dir / "class_balance_telemetry.json", "w") as fh:
            json.dump(balance_telem.to_dict(), fh, indent=2)
        if bias_AA_str:
            LOGGER.info(
                "cycle %d class-balanced bias_AA: %s (swaps=%d)",
                cycle_cfg.cycle_idx, bias_AA_str, len(balance_telem.swaps),
            )
            for sw in balance_telem.swaps:
                if sw.get("up_aa") is not None:
                    LOGGER.info(
                        "  swap[%s]: %s(z=%+.2f)->%+.2f  %s(z=%+.2f)->%+.2f",
                        sw["class"], sw["down_aa"], sw["down_z"], sw["down_bias"],
                        sw["up_aa"], sw["up_z"], sw["up_bias"],
                    )
                else:
                    # downweight_only (extreme over with no swap partner)
                    LOGGER.info(
                        "  downweight[%s]: %s(z=%+.2f)->%+.2f (no partner)",
                        sw["class"], sw["down_aa"], sw["down_z"], sw["down_bias"],
                    )
        else:
            LOGGER.info("cycle %d class-balanced bias_AA: (no swaps triggered)",
                         cycle_cfg.cycle_idx)

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
    # Position-1 M omit: 100% of designs across rounds 1–7 had M at
    # position 1 (the start codon Met from the seed PDB), giving zero
    # diversity at that position. With an MSG vector tag (--n_term_pad
    # MSG), the Met that the ribosome inserts is at position 1 of the
    # tag, not position 1 of the design body — so the design body's
    # position 1 is just an internal residue. Hard-omit M there to
    # break the artifact.
    if omit_M_at_pos1:
        # MPNN expects per-residue label '<chain><resno>' with PDB resno;
        # protein_resnos[0] is the first protein residue's pose-resno
        # (1-indexed). Use that.
        first_resno = sorted(protein_resnos)[0]
        m_omit = {f"{CHAIN}{first_resno}": "M"}
        merged_omit = merge_omit_dicts(merged_omit, m_omit)
        LOGGER.info("cycle %d: pos-1 M omit added (%s)",
                     cycle_cfg.cycle_idx, m_omit)
    cand_tsv = stage_sample(
        cycle_cfg=cycle_cfg, seed_pdb=seed_pdb, bias=bias_k,
        protein_resnos=protein_resnos, fixed_resnos=fixed_resnos,
        out_dir=sample_dir,
        omit_AA_per_residue=merged_omit,
        bias_AA=bias_AA_str,
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
        net_charge_min=cycle_cfg.net_charge_min,
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
        design_ph=design_ph,
        instability_max=instability_max,
        gravy_min=gravy_min,
        gravy_max=gravy_max,
        aliphatic_min=aliphatic_min,
        boman_max=boman_max,
        n_term_pad=n_term_pad,
        c_term_pad=c_term_pad,
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
        seed_dfi_metrics=seed_dfi_metrics,
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
        wt_fitness=wt_fitness,
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
    p.add_argument("--run_dir_marker", type=Path, default=None,
                   help="If set, write the absolute path of this run's "
                        "run_dir to this file as soon as it's created. "
                        "Useful for shell wrappers (sbatch) that need to "
                        "locate run_dir for downstream stages.")
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
    p.add_argument("--cms_final", action="store_true",
                   help="After stage_diverse_topk, run Coventry Contact "
                        "Molecular Surface on the top-K only (~3-4 s/design, "
                        "needs esmc.sif). Adds 'cms__total' column.")
    p.add_argument("--rosetta_final", action="store_true",
                   help="After stage_diverse_topk, run the comprehensive "
                        "Rosetta no-repack metrics panel (DDG + interface "
                        "energy + ...) on the top-K only. ~30-60 s/design, "
                        "needs pyrosetta.sif. Default OFF.")
    p.add_argument("--protonate_final", action="store_true", default=True,
                   help="After stage_diverse_topk, hydrate every top-K PDB "
                        "via PyRosetta and write a downstream-clean "
                        "<stem>.protonated.pdb with full hydrogens, "
                        "standard 3-char residue names (no HIS_D), the "
                        "seed's REMARK 666 + ligand block, and a new "
                        "REMARK 668 protonation table. Default ON. "
                        "Needs pyrosetta.sif.")
    p.add_argument("--no_protonate_final", dest="protonate_final",
                   action="store_false",
                   help="Disable the post-design PyRosetta protonation step.")
    p.add_argument("--pyrosetta_sif", type=Path,
                   default=Path("/net/software/containers/pyrosetta.sif"),
                   help="Path to pyrosetta.sif used by --protonate_final.")
    p.add_argument("--ptm", type=str,
                   default="",
                   help="Comma-separated PTM declarations recorded in the "
                        "output PDB's REMARK 668 block. ANNOTATION ONLY — "
                        "residues are NOT modified for Rosetta/sequence/"
                        "protonation handling. Two spec formats: "
                        "(a) motif-index form 'A/LYS/3:KCX' = chain A "
                        "REMARK 666 motif index 3 (expected LYS) -> KCX "
                        "(preferred for catalytic residues; stable across "
                        "a design campaign even when sequence position "
                        "varies); (b) explicit-residue form 'A:157=KCX'. "
                        "Empty default — caller must opt in per scaffold. "
                        "PTE_i1: 'A/LYS/3:KCX' (catalytic lysine motif). "
                        "Use '-' as code to force no-PTM annotation.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Set log level to DEBUG. The per-cycle metrics "
                        "snapshot (cycle_metrics.tsv + cycle_metrics.json) "
                        "is ALWAYS written to the run dir regardless of "
                        "this flag — it captures input n, post-seq-filter "
                        "n, post-struct-filter n, post-fpocket-rank n, and "
                        "the cycle's fitness mean/min/max, sap_max mean, "
                        "druggability mean, charge mean, hamming mean. "
                        "Useful for diagnosing how filters and bias shape "
                        "the population during iterative optimization.")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Set log level to WARNING (suppresses per-stage "
                        "INFO chatter). Mutually exclusive with --verbose.")
    p.add_argument("--save_intermediates", action="store_true",
                   help="Write a heavy diagnostic dump alongside the cycle "
                        "metrics: all_designs_per_cycle.tsv contains every "
                        "design seen across every cycle (incl. those culled "
                        "by struct or fpocket filters) with its full metric "
                        "panel. Use for deep diagnostics — adds ~5-50 MB to "
                        "the run dir depending on cycle counts and survivor "
                        "ratios. Off by default; only top-K survivors are "
                        "retained in topk.tsv.")
    p.add_argument("--consensus_threshold", type=float, default=0.90,
                   help="Cycle k+1 consensus reinforcement: AA frequency "
                        "across cycle-k survivors required to 'agree' "
                        "before that AA's bias is reinforced. Default 0.85. "
                        "Raise to 0.90+ to require stronger agreement and "
                        "preserve diversity. Round-6/7 with 0.85 lost ~50%% "
                        "of pairwise hamming vs rounds 1-5 (when consensus "
                        "was silently dead due to a class-name bug).")
    p.add_argument("--consensus_strength", type=float, default=1.0,
                   help="Bias magnitude (nats) added at consensus-agreed "
                        "(position, AA) pairs. Default 2.0; lower (e.g. "
                        "1.0) reduces over-collapse to consensus.")
    p.add_argument("--consensus_max_fraction", type=float, default=0.15,
                   help="Max fraction of eligible positions that consensus "
                        "can augment per cycle. Default 0.30; lower (e.g. "
                        "0.15) preserves more positional diversity by "
                        "reinforcing only the strongest-agreement positions.")
    p.add_argument("--strategy", type=str, default="annealing",
                   choices=["constant", "annealing"],
                   help="Cycle schedule: 'constant' = same filter "
                        "thresholds + TOPSIS weights every cycle (legacy "
                        "default); 'annealing' = light filters loose in "
                        "cycle 0, tightening to defaults by cycle 2; "
                        "TOPSIS weights fitness-heavy cycle 0, balanced "
                        "cycle 1+; cycles 1+ pick survivors by TOPSIS "
                        "(multi-objective) instead of by fitness alone. "
                        "Hard filters (charge band, pi band, severe "
                        "clash) stay constant across cycles in BOTH "
                        "strategies — only the light/soft components "
                        "anneal.")
    p.add_argument("--rank_weights", type=str, default="",
                   help="Multi-objective weight overrides as 'k=v,k=v'. "
                        "Keys can be metric labels: fitness, druggability, "
                        "lig_int_strength, preorg_strength, hbonds_to_cat, "
                        "instability, sap_max, boman, aliphatic, gravy, "
                        "charge, pi, bottleneck, pocket_hydrophobicity. "
                        "Default weights — fitness=2.0, druggability=1.0, "
                        "lig_int_strength=1.0, preorg_strength=0.7, "
                        "hbonds_to_cat=0.5, instability=0.5, sap_max=0.5, "
                        "all target metrics 0.3 (boman, aliphatic, gravy, "
                        "charge, pi, bottleneck) + 0.2 (pocket_hydrophobicity). "
                        "Set weight=0 to drop a metric from ranking.")
    p.add_argument("--rank_targets", type=str, default="",
                   help="Multi-objective target value overrides. Same key "
                        "names as --rank_weights. Example: "
                        "'aliphatic=100,boman=2.0,charge=-12'. Only "
                        "applies to target-direction metrics.")
    p.add_argument("--min_hamming_active", type=int, default=0,
                   help="Minimum active-site (primary_sphere) Hamming "
                        "between top-K designs, alongside the full-"
                        "sequence Hamming. Default 0 (disabled). Set "
                        "≥ 2 to enforce active-site diversity even "
                        "between designs that differ globally.")
    p.add_argument("--instability_max", type=float, default=60.0,
                   help="Light filter on Guruprasad 1990 instability index. "
                        "Lit threshold for native E. coli expression is 40, "
                        "but de novo designs run higher; default 60 catches "
                        "truly broken sequences only. Set 9999 to disable.")
    p.add_argument("--gravy_min", type=float, default=-0.8,
                   help="Light filter on Kyte-Doolittle GRAVY. Typical "
                        "soluble proteins fall in [-0.4, 0]; default [-0.8, "
                        "0.3] is generous.")
    p.add_argument("--gravy_max", type=float, default=0.3)
    p.add_argument("--aliphatic_min", type=float, default=40.0,
                   help="Light filter on Ikai 1980 aliphatic index. "
                        "Thermostable native: ~85-100. Default lower bound "
                        "40 catches only extremely low-aliphatic outliers.")
    p.add_argument("--boman_max", type=float, default=4.5,
                   help="Light filter on Boman index (PPI/sticky propensity). "
                        "Boman 2003 threshold ~2.5; default 4.5 catches only "
                        "extreme cases.")
    p.add_argument("--n_term_pad", type=str, default="MSG",
                   help="N-terminal sequence pad added to the design body "
                        "BEFORE computing sequence-only metrics (charge, "
                        "pI, GRAVY, instability, aliphatic, boman). "
                        "Default 'MSG' matches a typical E. coli vector "
                        "tag — the actual expressed protein is "
                        "M-S-G-[design]-G-S-A. Pass '' to disable.")
    p.add_argument("--c_term_pad", type=str, default="GSA",
                   help="C-terminal sequence pad — see --n_term_pad. "
                        "Default 'GSA'. Pass '' to disable.")
    p.add_argument("--no_omit_M_at_pos1", action="store_true",
                   help="By default, position 1 of the design body is "
                        "hard-omitted from M (start codon Met is in the "
                        "vector tag, not the design). Pass this flag to "
                        "disable the omit and let MPNN sample M there.")
    p.add_argument("--design_ph", type=float, default=7.8,
                   help="pH at which net charge / pI / etc. are computed. "
                        "Default 7.8 (close to PTE assay buffer pH 8.0, "
                        "with a small safety margin). The robust filter "
                        "charge uses Henderson-Hasselbalch on K/R/H + D/E/"
                        "C/Y + N/C termini (Pace 1999 / Bjellqvist 1994 "
                        "pKas). Four diagnostic variants are also recorded "
                        "(no_HIS, HIS_half, DE_KR_only, Biopython) for "
                        "comparison/sensitivity analysis.")
    p.add_argument("--balance_z_threshold", type=float, default=2.0,
                   help="Class-balanced bias_AA z-cutoff. A swap fires "
                        "only when one class member is over-rep > +z AND "
                        "another is under-rep < -z (default 2.0; the user "
                        "noted 2-3 is reasonable, ≤1.5 is too aggressive).")
    p.add_argument("--plm_strength", type=float, default=1.25,
                   help="Global multiplier on PLM fusion class weights "
                        "(applied uniformly to ESM-C and SaProt at every "
                        "position). Default 1.25 — empirical sweep across "
                        "rounds 1–5 (2026-05-04) on PTE_i1 found 1.2–1.3 "
                        "the sweet spot: best fitness recovery, tightest "
                        "druggability distribution, and best primary-shell "
                        "diversity, without saturating PLM signal. Pass "
                        "0.7 to soften (more MPNN structural fidelity), "
                        "1.5+ for maximum PLM influence (diminishing "
                        "returns; charge SD inflates). Must be ≥ 0; 0.0 "
                        "disables PLM bias entirely.")
    args = p.parse_args()
    if args.plm_strength < 0:
        p.error("--plm_strength must be >= 0 "
                "(negative would invert the PLM signal)")
    if args.plm_strength > 5.0:
        LOGGER.warning(
            "--plm_strength=%.2f is very large; PLM bias may dominate "
            "MPNN's structure-conditioned logits (collapse to PLM "
            "consensus). Typical range 0.5-2.0.", args.plm_strength,
        )

    if args.verbose and args.quiet:
        raise SystemExit("--verbose and --quiet are mutually exclusive")
    log_level = logging.DEBUG if args.verbose else (
        logging.WARNING if args.quiet else logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Auto-derive catalytic resnos from the seed's REMARK 666 block. This
    # makes the same driver/sbatch work on any scaffold in the design
    # campaign — even though the catalytic His/Lys/Glu sequence positions
    # vary between scaffolds (e.g. SEED1 LYS 157 vs SEED2 LYS 19), the
    # REMARK 666 block records them and we adopt those positions for the
    # filter / fixed-residue / catres-aware code paths.
    global DEFAULT_CATRES, CATALYTIC_HIS_RESNOS
    derived_catres, derived_his = _derive_catres_from_remark_666(args.seed_pdb)
    if derived_catres != DEFAULT_CATRES:
        LOGGER.info(
            "auto-derived catalytic resnos from seed REMARK 666: "
            "all_catres=%s (was default %s); his_only=%s (was default %s)",
            derived_catres, DEFAULT_CATRES, derived_his, CATALYTIC_HIS_RESNOS,
        )
    DEFAULT_CATRES = derived_catres
    CATALYTIC_HIS_RESNOS = derived_his

    # Include microseconds + PID to prevent concurrent-job collisions on
    # second-precision timestamps (real bug observed during a 4-job
    # parallel sweep — two jobs that started in the same second wrote
    # to the same run_dir and overwrote each other's outputs).
    import os as _os_pid
    ts_micro = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]  # ms precision
    timestamp = f"{ts_micro}-pid{_os_pid.getpid()}"
    run_dir = args.out_root / f"iterative_design_v2_PTE_i1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("=== run dir: %s ===", run_dir)
    if args.run_dir_marker:
        try:
            args.run_dir_marker.parent.mkdir(parents=True, exist_ok=True)
            args.run_dir_marker.write_text(str(run_dir.resolve()) + "\n")
        except Exception as exc:
            LOGGER.warning("could not write run_dir_marker %s: %s",
                            args.run_dir_marker, exc)

    # ---- Detect available compute resources -----------------------------
    # Single source of truth for CPU/GPU counts; consumed by all parallel
    # stages (fpocket, struct_filter, restore_sample_dir). Also configures
    # PyTorch thread count to match the slurm allocation (avoids
    # oversubscription when running on a CPU node with cpus_per_task < total cores).
    from protein_chisel.utils.resources import (
        detect_resources, configure_torch_threads,
    )
    resources = detect_resources()
    # Pin torch threads for the parent process only on CPU runs.
    # On GPU jobs, the empirical test (round 2) showed setting
    # threads-in-parent caused multiprocessing.Pool workers (which
    # fork from parent) to inherit threads=N each, leading to N×N
    # thread oversubscription (4 workers × 4 threads = 16 on 4 CPUs).
    # Better: let GPU runs use defaults; only constrain CPU.
    if resources.n_gpus == 0:
        configure_torch_threads(resources.n_cpus)

    # ---- Load PLM artifacts -----------------------------------------
    art = args.plm_artifacts_dir
    log_probs_esmc = np.load(art / "esmc_log_probs.npy")
    log_probs_saprot = np.load(art / "saprot_log_probs.npy")
    cached_base_bias = np.load(art / "fusion_bias.npy")
    cached_weights = np.load(art / "fusion_weights.npy")
    LOGGER.info("loaded raw PLM log-probs: L=%d", log_probs_esmc.shape[0])

    # ---- Load PositionTable -----------------------------------------
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.io.schemas import PositionTable
    from protein_chisel.io.pdb import extract_sequence
    from protein_chisel.sampling.plm_fusion import FusionConfig, fuse_plm_logits

    pt = PositionTable.from_parquet(args.position_table)
    # Detect legacy (5-class) PositionTable and re-classify with the new
    # directional 6-class taxonomy if so. Cheap (~50 ms) and gives the
    # PLM-fusion + diagnostic columns access to the new metrics.
    legacy_classes_set = {"active_site", "first_shell", "pocket", "buried", "surface"}
    pt_classes = set(pt.df.loc[pt.df["is_protein"], "class"].astype(str).unique())
    if pt_classes & legacy_classes_set and "primary_sphere" not in pt_classes:
        from protein_chisel.tools.classify_positions import (
            classify_positions, ClassifyConfig,
        )
        LOGGER.info(
            "position_table is legacy 5-class; re-classifying with directional "
            "6-class taxonomy. Original classes: %s",
            sorted(pt_classes),
        )
        pt = classify_positions(
            pdb_path=args.seed_pdb,
            params=[args.ligand_params],
            config=ClassifyConfig(),
        )
        # Also save a sidecar so subsequent runs can skip the re-classify.
        sidecar = run_dir / "position_table_v2.parquet"
        pt.to_parquet(sidecar)
        LOGGER.info("re-classified PositionTable saved to %s", sidecar)

    protein_rows = pt.df[pt.df["is_protein"]].sort_values("resno").reset_index(drop=True)
    position_classes = protein_rows["class"].tolist()
    protein_resnos = protein_rows["resno"].astype(int).tolist()
    L = len(protein_resnos)
    LOGGER.info("position table: L=%d, class counts=%s",
                 L, protein_rows["class"].value_counts().to_dict())

    # ---- Recompute PLM fusion at runtime ----------------------------
    # The cached fusion_bias.npy was built with whatever class_weights
    # the precompute step used. We re-fuse here so the *current*
    # FusionConfig.class_weights take effect (e.g. after bumping
    # active_site / first_shell / pocket weights). Cheap: a few
    # numpy ops on (L, 20) matrices. We then snapshot the runtime
    # result to the run dir so offline analysis/replays use the
    # *actual* bias the cycles saw, not the stale cached one.
    fusion_cfg = FusionConfig(global_strength=args.plm_strength)
    fusion_res = fuse_plm_logits(
        log_probs_esmc=log_probs_esmc,
        log_probs_saprot=log_probs_saprot,
        position_classes=position_classes,
        config=fusion_cfg,
    )
    base_bias = fusion_res.bias
    weights_per_position = fusion_res.weights_per_position
    fusion_dir = run_dir / "fusion_runtime"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    np.save(fusion_dir / "base_bias.npy", base_bias)
    np.save(fusion_dir / "weights_per_position.npy", weights_per_position)
    np.save(fusion_dir / "log_odds_esmc.npy", fusion_res.log_odds_esmc)
    np.save(fusion_dir / "log_odds_saprot.npy", fusion_res.log_odds_saprot)
    with open(fusion_dir / "fusion_config.json", "w") as fh:
        json.dump({
            "class_weights": fusion_cfg.class_weights,
            "global_strength": fusion_cfg.global_strength,
            "entropy_match": fusion_cfg.entropy_match,
            "shrink_disagreement": fusion_cfg.shrink_disagreement,
            "shrink_threshold": fusion_cfg.shrink_threshold,
            "cached_artifact_dir": str(art),
            "cached_mean_abs_bias": float(np.abs(cached_base_bias).mean()),
            "runtime_mean_abs_bias": float(np.abs(base_bias).mean()),
        }, fh, indent=2)
    LOGGER.info(
        "PLM fusion (runtime): mean_abs_bias=%.4f (cached was %.4f); "
        "global_strength=%.2f, class_weights=%s",
        float(np.abs(base_bias).mean()),
        float(np.abs(cached_base_bias).mean()),
        args.plm_strength, fusion_cfg.class_weights,
    )
    # Diagnostic: per-class total bias mass with the new weights.
    import collections as _coll
    cls_mass: dict[str, float] = _coll.defaultdict(float)
    cls_count: dict[str, int] = _coll.defaultdict(int)
    abs_bias_per_pos = np.abs(base_bias).sum(axis=-1)
    for cls, m in zip(position_classes, abs_bias_per_pos):
        cls_mass[cls] += float(m)
        cls_count[cls] += 1
    for cls in sorted(cls_mass):
        LOGGER.info(
            "  class %s: n=%d, total_|bias|=%.2f, mean_|bias|/pos=%.3f nats",
            cls, cls_count[cls], cls_mass[cls],
            cls_mass[cls] / max(1, cls_count[cls]),
        )

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

    # ---- One-shot: tunnel-residue annotation on seed PDB -----------
    # Identifies residues whose atoms sit within 6 Å of any
    # active-site alpha-sphere — these are the channel-lining
    # positions where bulky/charged residues most directly impact
    # pocket accessibility. Written as a sidecar TSV next to manifest
    # for downstream tools (incl. scripts/audit_pocket_metrics.py).
    seed_tunnel_path = run_dir / "seed_tunnel_residues.tsv"
    try:
        annotate_seed_tunnel_residues(
            seed_pdb=args.seed_pdb,
            out_path=seed_tunnel_path,
            catalytic_resnos=DEFAULT_CATRES,
            chain=CHAIN,
            proximity_cutoff=6.0,
        )
    except Exception as exc:
        LOGGER.warning("seed tunnel annotation failed: %s", exc)

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
        strategy=args.strategy,
        consensus_threshold=args.consensus_threshold,
        consensus_strength=args.consensus_strength,
        consensus_max_fraction=args.consensus_max_fraction,
    )
    LOGGER.info("strategy: %s", args.strategy)
    if args.strategy == "annealing":
        for c in cycles:
            LOGGER.info(
                "  cycle %d: instability_max=%.0f, gravy=[%.2f, %.2f], "
                "aliphatic_min=%.0f, boman_max=%.1f, "
                "topsis_overrides=%s, use_topsis=%s",
                c.cycle_idx, c.instability_max, c.gravy_min, c.gravy_max,
                c.aliphatic_min, c.boman_max,
                c.topsis_weight_overrides or "(defaults)",
                c.use_topsis_for_survivors,
            )
    if args.cycles == 1:
        cycles = cycles[:1]   # short-test mode
    elif args.cycles != 3:
        # Honor any positive int by truncating / extending the default schedule.
        cycles = cycles[: max(1, args.cycles)]
    LOGGER.info("cycle schedule: %d cycles, omit_AA=%r", len(cycles), args.omit_AA)

    # ---- Pre-compute seed DFI once (design-invariant for fixed-backbone) --
    # DFI was previously computed per-design (~80 ms × 200 designs/cycle =
    # ~16 s/cycle wasted). Now computed once on the seed and broadcast.
    seed_dfi_metrics: Optional[dict] = None
    try:
        from protein_chisel.scoring.dfi import compute_dfi
        seed_dfi = compute_dfi(
            args.seed_pdb, chain=CHAIN,
            classes=position_classes, classes_resnos=protein_resnos,
        )
        seed_dfi_metrics = seed_dfi.to_dict()
        LOGGER.info(
            "seed DFI (computed once): mean=%.3f, primary=%.3f, distal_buried=%.3f",
            seed_dfi_metrics.get("dfi__mean", float("nan")),
            seed_dfi_metrics.get("dfi__mean__primary_sphere", float("nan")),
            seed_dfi_metrics.get("dfi__mean__distal_buried", float("nan")),
        )
    except Exception as exc:
        LOGGER.warning("seed DFI compute failed (%s); designs get NaN", exc)

    # ---- Pre-compute WT fitness once (PLM gather on the seed sequence) ---
    # Used to populate `fitness__delta_vs_wt` per design row. Positive
    # delta = more PLM-natural per residue than WT.
    wt_fitness: Optional[float] = None
    try:
        from protein_chisel.sampling.fitness_score import (
            fitness_from_seed_marginals,
        )
        wt_res = fitness_from_seed_marginals(
            wt_seq, log_probs_esmc, log_probs_saprot, weights_per_position,
        )
        wt_fitness = float(wt_res.logp_fused_mean)
        LOGGER.info(
            "WT fitness (gather on seed sequence): logp_fused=%.4f "
            "(esmc=%.4f, saprot=%.4f). delta_vs_wt > 0 means design is "
            "more PLM-natural per residue than WT.",
            wt_res.logp_fused_mean, wt_res.logp_esmc_mean,
            wt_res.logp_saprot_mean,
        )
    except Exception as exc:
        LOGGER.warning("WT fitness compute failed (%s); delta_vs_wt = NaN", exc)

    # ---- Loop cycles ------------------------------------------------
    all_ranked: list[pd.DataFrame] = []
    all_pdb_maps: dict[str, Path] = {}
    fitness_cache: dict = {}
    survivors_prev: Optional[pd.DataFrame] = None
    fixed_resnos = list(DEFAULT_CATRES)

    # Per-cycle metrics snapshot — written to run_dir/cycle_metrics.tsv at the
    # end so the user can grep / plot how filter populations and quality
    # change across cycles. Always written; --verbose adds more granular
    # debug output to the log itself but the TSV is always there.
    cycle_metric_rows: list[dict] = []

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
            seed_dfi_metrics=seed_dfi_metrics,
            wt_fitness=wt_fitness,
            position_table_df=pt.df,
            omit_AA_per_residue=omit_AA_per_residue,
            balance_z_threshold=args.balance_z_threshold,
            design_ph=args.design_ph,
            # Per-cycle filter thresholds: in 'annealing' strategy these
            # come from CycleConfig (which override global defaults);
            # in 'constant' strategy CycleConfig fields == global defaults
            # so this is a no-op.
            instability_max=cyc.instability_max if args.strategy == "annealing"
                            else args.instability_max,
            gravy_min=cyc.gravy_min if args.strategy == "annealing"
                      else args.gravy_min,
            gravy_max=cyc.gravy_max if args.strategy == "annealing"
                      else args.gravy_max,
            aliphatic_min=cyc.aliphatic_min if args.strategy == "annealing"
                          else args.aliphatic_min,
            boman_max=cyc.boman_max if args.strategy == "annealing"
                      else args.boman_max,
            n_term_pad=args.n_term_pad,
            c_term_pad=args.c_term_pad,
            omit_M_at_pos1=not args.no_omit_M_at_pos1,
        )
        if ranked_df is not None and len(ranked_df) > 0:
            ranked_df = ranked_df.copy()
            ranked_df["cycle"] = cyc.cycle_idx
            all_ranked.append(ranked_df)
            # Per-cycle survivor selection: by fitness (legacy) OR by
            # TOPSIS (annealing). When use_topsis_for_survivors is True,
            # we compute the multi-objective score over the cycle's
            # ranked_df and feed the top survivors (sorted by mo_topsis)
            # into next cycle's consensus reinforcement, so the
            # iteration improves on ALL objectives, not just fitness.
            if cyc.use_topsis_for_survivors:
                from protein_chisel.scoring.multi_objective import (
                    DEFAULT_METRIC_SPECS, apply_cli_overrides,
                    compute_topsis_scores_v2, parse_kv_string,
                )
                cycle_specs = apply_cli_overrides(
                    DEFAULT_METRIC_SPECS,
                    {**parse_kv_string(args.rank_weights),
                     **cyc.topsis_weight_overrides},
                    parse_kv_string(args.rank_targets),
                )
                cyc_scores, _used, _dbg = compute_topsis_scores_v2(
                    ranked_df, cycle_specs,
                )
                ranked_df["mo_topsis_cycle"] = cyc_scores
                survivors_prev = ranked_df.sort_values(
                    "mo_topsis_cycle", ascending=False,
                ).reset_index(drop=True)
                LOGGER.info(
                    "cycle %d survivors fed forward by TOPSIS: top score=%.3f",
                    cyc.cycle_idx, float(cyc_scores.max()) if len(cyc_scores) else 0.0,
                )
            else:
                # Legacy: by fitness alone.
                survivors_prev = ranked_df
        all_pdb_maps.update(pdb_map)

        # Snapshot metrics for this cycle (best-effort; never blocks).
        cycle_row: dict = {
            "cycle": cyc.cycle_idx,
            "strategy": getattr(cyc, "strategy", "n/a"),
            "ranked_n": int(len(ranked_df)) if ranked_df is not None else 0,
        }
        try:
            for prefix, glob in (
                ("seq", "02_seq_filter/survivors_seq.tsv"),
                ("struct", "03_struct_filter/survivors_struct.tsv"),
                ("scored", "04_fitness/scored.tsv"),
                ("pocket", "05_fpocket/ranked.tsv"),
            ):
                p = cycle_dir / glob
                if p.is_file():
                    try:
                        n = sum(1 for _ in open(p)) - 1  # minus header
                    except Exception:
                        n = -1
                    cycle_row[f"n_{prefix}"] = n
        except Exception:
            pass
        if ranked_df is not None and len(ranked_df) > 0:
            for col, prefix in (
                ("fitness__logp_fused_mean", "fitness"),
                ("sap_max", "sap_max"),
                ("fpocket__druggability", "druggability"),
                ("net_charge_no_HIS", "charge"),
                ("pi", "pi"),
                ("pairwise_hamming_full", "hamming"),
            ):
                if col in ranked_df.columns:
                    vals = pd.to_numeric(ranked_df[col], errors="coerce").dropna()
                    if len(vals) > 0:
                        cycle_row[f"{prefix}_mean"] = float(vals.mean())
                        cycle_row[f"{prefix}_min"] = float(vals.min())
                        cycle_row[f"{prefix}_max"] = float(vals.max())
        cycle_metric_rows.append(cycle_row)
        if args.verbose:
            LOGGER.info("cycle %d metrics snapshot: %s",
                        cyc.cycle_idx,
                        ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in cycle_row.items()))

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
        # ---- Multi-objective ranking over the full pool ---------------
        # TOPSIS over a configurable basket of metrics with CLI-tunable
        # weights / targets. Replaces the legacy 2-key (fitness,
        # alpha_radius) sort. The legacy sort is still computed and
        # written as `legacy_rank_score` for back-compat / debugging.
        from protein_chisel.scoring.multi_objective import (
            DEFAULT_METRIC_SPECS, apply_cli_overrides, compute_topsis_scores_v2,
            parse_kv_string, select_diverse_topk_two_axis,
        )
        rank_weights = parse_kv_string(args.rank_weights)
        rank_targets = parse_kv_string(args.rank_targets)
        active_specs = apply_cli_overrides(
            DEFAULT_METRIC_SPECS, rank_weights, rank_targets,
        )
        scores, used_specs, _debug = compute_topsis_scores_v2(pool, active_specs)
        pool["mo_topsis"] = scores
        # Legacy diagnostic — keep next to mo_topsis so we can compare.
        pool["legacy_rank_score"] = (
            pool["fitness__logp_fused_mean"].rank(ascending=False)
            + pool["fpocket__mean_alpha_sphere_radius"].rank(ascending=True)
        )
        pool = pool.sort_values(
            ["mo_topsis", "fitness__logp_fused_mean"],
            ascending=[False, False], na_position="last",
        ).reset_index(drop=True)
        pool.to_csv(final_dir / "all_survivors.tsv", sep="\t", index=False)
        LOGGER.info("final pool: %d unique survivors across %d cycles",
                     len(pool), len(all_ranked))
        LOGGER.info("multi-objective ranking applied with %d active specs:",
                     len(used_specs))
        for s in used_specs:
            LOGGER.info("  %-25s direction=%-7s weight=%.2f target=%s",
                         s.label, s.direction, s.weight, s.target)
        LOGGER.info("top-5 mo_topsis scores: %s",
                     pool["mo_topsis"].head(5).round(3).tolist())

        # ---- Diverse top-K (full + active-site Hamming) ---------------
        # Build position-index list for primary_sphere from the
        # re-classified PositionTable so the active-site Hamming gate
        # has the right indices.
        primary_positions: Optional[list[int]] = None
        if args.min_hamming_active > 0:
            try:
                pt_protein = pt.df[pt.df["is_protein"]].sort_values("resno").reset_index(drop=True)
                primary_positions = [
                    i for i, cls in enumerate(pt_protein["class"].astype(str).tolist())
                    if cls == "primary_sphere"
                ]
                LOGGER.info(
                    "active-site Hamming gate: %d primary_sphere positions, "
                    "min_hamming_active=%d",
                    len(primary_positions), args.min_hamming_active,
                )
            except Exception as exc:
                LOGGER.warning("could not extract primary positions: %s", exc)

        top = select_diverse_topk_two_axis(
            pool, target_k=args.target_k,
            min_hamming_full=args.min_hamming,
            primary_sphere_positions=primary_positions,
            min_hamming_active=args.min_hamming_active,
            score_col="mo_topsis",
        )
        # Write the top-K artifact files (PDB copy, FASTA, TSV).
        top.to_csv(final_dir / "topk.tsv", sep="\t", index=False)
        with open(final_dir / "topk.fasta", "w") as fh:
            for _, row in top.iterrows():
                fh.write(f">{row['id']}\n{row['sequence']}\n")
        pdb_out = final_dir / "topk_pdbs"
        pdb_out.mkdir(exist_ok=True)
        for _, row in top.iterrows():
            src = all_pdb_maps.get(row["id"])
            if src and src.is_file():
                shutil.copy2(src, pdb_out / src.name)
        topk_tsv = final_dir / "topk.tsv"
        LOGGER.info(
            "stage_diverse_topk: selected %d / %d "
            "(target=%d, min_hamming=%d, min_hamming_active=%d)",
            len(top), len(pool), args.target_k,
            args.min_hamming, args.min_hamming_active,
        )
        # ---- Optional final-stage enrichments on top-K only --------
        if args.cms_final:
            topk_tsv = stage_cms_final(
                topk_tsv=topk_tsv, pdb_map=all_pdb_maps,
                out_dir=final_dir / "cms_final",
            )
        if args.rosetta_final:
            stage_rosetta_final(
                topk_tsv=topk_tsv, pdb_map=all_pdb_maps,
                out_dir=final_dir / "rosetta_final",
                ligand_params=args.ligand_params,
            )
        if args.protonate_final:
            stage_protonate_final_topk(
                topk_pdb_dir=final_dir / "topk_pdbs",
                seed_pdb=args.seed_pdb,
                ligand_params=args.ligand_params,
                pyrosetta_sif=args.pyrosetta_sif,
                out_dir=final_dir / "topk_pdbs_protonated",
                ptm=args.ptm,
            )
    else:
        LOGGER.warning("final: zero survivors across all cycles!")

    # ---- Per-cycle metrics snapshot ---------------------------------
    if cycle_metric_rows:
        cycle_metrics_df = pd.DataFrame(cycle_metric_rows)
        cycle_metrics_df.to_csv(run_dir / "cycle_metrics.tsv", sep="\t", index=False)
        with open(run_dir / "cycle_metrics.json", "w") as fh:
            json.dump(cycle_metric_rows, fh, indent=2, default=str)
        LOGGER.info("wrote per-cycle metrics: %s", run_dir / "cycle_metrics.tsv")

    # ---- Optional: dump every design seen across every cycle --------
    if args.save_intermediates and all_ranked:
        all_designs_path = run_dir / "all_designs_per_cycle.tsv"
        all_concat = pd.concat(all_ranked, ignore_index=True)
        all_concat.to_csv(all_designs_path, sep="\t", index=False)
        LOGGER.info(
            "save_intermediates: wrote %d designs across %d cycles to %s "
            "(%.1f MB)",
            len(all_concat), len(cycles), all_designs_path,
            all_designs_path.stat().st_size / 1024 / 1024,
        )
        # Pretty-print a compact one-liner for each cycle to the log.
        for row in cycle_metric_rows:
            LOGGER.info(
                "  cycle %d: ranked=%d  n_seq=%s  n_struct=%s  n_pocket=%s  "
                "fitness=%.3f±%s  sap_max=%.2f  drugg=%.2f",
                row.get("cycle", -1), row.get("ranked_n", 0),
                row.get("n_seq", "?"), row.get("n_struct", "?"),
                row.get("n_pocket", "?"),
                row.get("fitness_mean", float("nan")),
                f"{row.get('fitness_max', float('nan')):.3f}",
                row.get("sap_max_mean", float("nan")),
                row.get("druggability_mean", float("nan")),
            )

    # ---- End-of-run summary block (always printed) -------------------
    final_topk_count = 0
    final_unique_seqs = 0
    final_pdb_count = 0
    try:
        topk_tsv_path = final_dir / "topk.tsv"
        if topk_tsv_path.is_file():
            final_df = pd.read_csv(topk_tsv_path, sep="\t")
            final_topk_count = len(final_df)
            seq_col = "sequence" if "sequence" in final_df.columns else (
                "seq" if "seq" in final_df.columns else None
            )
            if seq_col is not None:
                final_unique_seqs = final_df[seq_col].nunique()
        pdb_out_dir = final_dir / "topk_pdbs"
        if pdb_out_dir.is_dir():
            final_pdb_count = sum(1 for p in pdb_out_dir.iterdir() if p.suffix == ".pdb")
    except Exception as exc:
        LOGGER.warning("end-of-run summary computation failed: %s", exc)

    LOGGER.info(
        "=== FINAL SUMMARY ===  top-K rows=%d  unique_seqs=%d  PDBs=%d  "
        "(pool->dedup->topk pruned %d -> %d)",
        final_topk_count, final_unique_seqs, final_pdb_count,
        sum(len(df) for df in all_ranked) if all_ranked else 0,
        final_topk_count,
    )

    # ---- Manifest ---------------------------------------------------
    manifest = {
        "pipeline": "iterative_design_v2",
        "seed_pdb": str(args.seed_pdb),
        "ligand_params": str(args.ligand_params),
        "plm_artifacts_dir": str(args.plm_artifacts_dir),
        "position_table": str(args.position_table),
        "fixed_resnos": list(DEFAULT_CATRES),  # auto-derived from REMARK 666 in main()
        "catalytic_his_resnos": list(CATALYTIC_HIS_RESNOS),  # auto-derived from REMARK 666 in main()
        "wt_length": L,
        "target_k": args.target_k,
        "diversity_min_hamming": args.min_hamming,
        "n_cycles_run": len(cycles),
        "cycle_configs": [asdict(c) for c in cycles],
        "ptm_spec": getattr(args, "ptm", ""),
        "final_topk_count": final_topk_count,
        "final_unique_sequences": final_unique_seqs,
        "final_pdb_count": final_pdb_count,
        "outputs": {
            "run_dir": str(run_dir),
            "final_topk_fasta": str(final_dir / "topk.fasta"),
            "final_topk_pdbs": str(final_dir / "topk_pdbs"),
            "final_topk_pdbs_protonated": str(final_dir / "topk_pdbs_protonated"),
            "all_survivors": str(final_dir / "all_survivors.tsv"),
            "cycle_metrics_tsv": str(run_dir / "cycle_metrics.tsv"),
        },
        "started_at": timestamp,
    }
    with open(run_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("=== DONE -- top-K at %s ===", final_dir)


if __name__ == "__main__":
    main()
