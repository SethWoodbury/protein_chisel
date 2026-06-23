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
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("iterative_design")


# ----------------------------------------------------------------------
# Hard-coded constants for the PTE_i1 scaffold
# ----------------------------------------------------------------------

# CLI defaults are intentionally None so the user must pass --seed_pdb /
# --ligand_params explicitly (or set them via env vars in the sbatch
# wrapper). The previous values pointed at woodbuse-private paths and
# weren't portable. The sbatch wrapper enforces these as REQUIRED env
# vars, so anyone using run_chisel_design.sh already provides
# them; this just removes the silent footgun for direct CLI users.
DEFAULT_INPUT_PDB: Optional[Path] = None
DEFAULT_LIG_PARAMS: Optional[Path] = None

# REMARK 666 catalytic resnos (1-indexed PDB resseq) on chain A.
# These default values match the PTE_i1 SEED1 (FS269) scaffold. They are
# AUTO-OVERRIDDEN at the start of main() by parsing REMARK 666 in the
# user-supplied --seed_pdb so the same code/sbatch works on every
# scaffold in a design campaign (catalytic resnos shift between
# scaffolds even though motif structure is preserved).
DEFAULT_CATRES = (60, 64, 128, 131, 132, 157)
CATALYTIC_HIS_RESNOS = (60, 64, 128, 132)
CHAIN = "A"

# Bulky AAs for the always-on graded-clash bias (compute_graded_clash_bias).
# Long/aromatic side chains that can collide with a fixed catalytic atom: aromatics
# Y/F/W/H, long aliphatic M, and the long charged R AND K. K and R are the same
# length tier (Cb->NZ ~5.5 A / Cb->CZ ~6 A), so they must be treated symmetrically —
# shared here so the function default and its call site cannot drift apart.
_CLASH_BULKY_AAS = "YFWHMRK"

# --metrics objective-gating filter (add-on #7). Set once in main() from the
# resolved metric selection: a frozenset of multi_objective labels to keep in the
# TOPSIS basket, or None for "no gating" (the default --metrics all => byte-identical
# ranking). Read by the ranking sites + the backfill/rescue helpers via
# select_specs_by_label(); a module global avoids threading it through every
# helper call site (the helpers run once per process).
_RANKING_LABEL_FILTER = None

# --filters survivor-gating set (add-on #7). Set once in main(): a frozenset of
# filter metric names allowed to DROP designs, or None for "all filters on" (the
# default --filters all => byte-identical survivor sets). Read via
# _filter_active() at every filter predicate (seq/struct/clash/tunnel/fpocket).
# A module global avoids threading it through the struct worker's positional-tuple
# argument and the many backfill/rescue call sites (run once per process).
_ACTIVE_FILTERS = None


def _filter_active(name: str) -> bool:
    """True if filter ``name`` may drop designs (i.e. it is selected). With the
    default --filters all (``_ACTIVE_FILTERS is None``) every filter is active, so
    gating is a no-op and survivor sets are byte-identical."""
    return _ACTIVE_FILTERS is None or name in _ACTIVE_FILTERS


# Catalytic-HIS H-bond structural requirement (any-enzyme generalization). The
# default struct filter rejects a design with 0 side-chain H-bonds to a catalytic
# HIS. That is correct for a His-containing active site but rejects EVERY design
# for an enzyme whose mechanism has no catalytic His. Set in main() from
# ``not args.no_require_cat_his_hbond`` (kept as a module global like
# DEFAULT_CATRES so the threaded struct-filter worker reads it without threading
# through its positional-tuple). Default True => byte-identical (criterion ON).
REQUIRE_CAT_HIS_HBOND = True


# ---- Decode-time PoE backend (add-on, opt-in --mpnn_backend poe) ----------
# Nested apptainer is blocked inside the stage-3 container, so the PoE sampler runs
# as a SEPARATE HOST stage and its candidates feed the driver's score/rank one-shot.
# stage_sample handles two PoE modes via these module globals (both None by default
# => the default `bias` backend is the in-process sampler, byte-identical):
#   * _POE_EMIT_INPUTS_DIR: write the cycle-0 bias/fixed/omit JSONs (the exact ones
#     this run computed, incl. conserved-hbond rolls) for the host PoE stage, then
#     exit. Reuses the driver's prep (no duplication / drift).
#   * _POE_SAMPLE_DIR: build the candidate pool from a finished PoE output dir
#     (candidate_set_from_poe_dir) instead of sampling; PoE forces a single cycle.
_POE_EMIT_INPUTS_DIR = None
_POE_SAMPLE_DIR = None
_POE_EXPERTS: tuple = ()
_POE_LAMBDAS: tuple = ()
_POE_TEMPERATURE = 0.1   # temperature the host PoE stage sampled at (provenance only)


# Probabilistic conserved-sidechain-H-bond fixing (Feature 1) + canonical REMARK
# transfer (Feature 2). Set in main() from argparse; kept as module globals (like
# DEFAULT_CATRES above) so stage_sample/stage_restore_pdbs read them without
# threading through run_cycle's large signature. The detection/rolling/transfer
# logic lives in the SHARED protein_chisel.tools.{conserved_hbonds,remarks}
# modules — the same code scripts/chisel_ligandMPNN.py uses.
CONSERVE_HBONDS = False
CONSERVE_HBOND_PROB = 0.8
CONSERVE_HBOND_MAX_DIST = 3.9
CONSERVE_HBOND_MAX_ANGLE = 90.0
CONSERVE_ANCHORS: tuple[str, ...] = ("ligand", "catalytic", "user_fixed")
CONSERVE_KEEP_CLASHING = False
CONSERVE_SEED_BASE = 0
# Active-site interaction-network growth (Phase 2 / add-on #6). depth=1 +
# hbond + no-grow = the legacy single-shell conserved-hbond fixing (byte-identical).
CONSERVE_HBOND_DEPTH = 1
CONSERVE_INTERACTION_TYPES: tuple[str, ...] = ("hbond",)
CONSERVE_SHELL_DECAY = 1.0
CONSERVE_GROW_NETWORK = False
# Cross-cycle accumulator: when CONSERVE_GROW_NETWORK, residues pinned in a cycle
# become extra anchors next cycle (the network deepens as design proceeds).
_CONSERVE_GROWN_ANCHORS: set[int] = set()
TRANSFER_REMARKS = True


def _parse_bool_arg(value: str | bool) -> bool:
    """Parse a CLI boolean from common true/false spellings."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected boolean true/false for this flag, got {value!r}",
    )


def _fraction_arg(value: str) -> float:
    """Parse a CLI fraction, requiring a finite value in (0, 1].

    Used for ``--aa_fraction_cap``: a cap of 0/negative/NaN or an absurdly low
    value would omit every (or nearly every) amino acid, which fused MPNN encodes
    as equal ``-1e8`` logits → it then samples *uniformly from the "forbidden"
    set* rather than respecting the omit. Reject those at the CLI boundary.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}")
    if not math.isfinite(v) or not (0.0 < v <= 1.0):
        raise argparse.ArgumentTypeError(
            f"expected a finite fraction in (0, 1], got {value!r}",
        )
    return v


def _nonneg_finite_arg(value: str, *, max_value: Optional[float] = None) -> float:
    """Parse a CLI magnitude, requiring a finite value in ``[0, max_value]``.

    Used for nats magnitudes (e.g. ``--composition_soft_bias_nats``): a NaN/inf
    bias would propagate into the sampling softmax and poison every probability,
    and an absurd-but-finite value (e.g. ``1e100``) casts to ``-inf`` in the
    float32 sampler bias — ``max_value`` rejects those at the CLI boundary.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}")
    if not math.isfinite(v) or v < 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a finite value >= 0, got {value!r}",
        )
    if max_value is not None and v > max_value:
        raise argparse.ArgumentTypeError(
            f"expected a value <= {max_value}, got {value!r}",
        )
    return v


def _scc_schedule_arg(value: str) -> list[int]:
    """Parse ``--use_side_chain_context_schedule '1,1,0'`` -> ``[1, 1, 0]``.

    A comma-separated per-cycle 0/1 schedule for LigandMPNN's side-chain-context
    flag (ON early for clash avoidance, OFF late for first-shell diversity).
    Every entry must be EXACTLY 0 or 1, and at least one entry is required — a
    typo (``2``, ``0.5``, an empty field) would otherwise silently mis-set the
    sampler's context flag for a whole cycle. The list is broadcast/truncated to
    the run's cycle count by :func:`_apply_scc_schedule`.

    The parse is pure (no ``protein_chisel`` import), so ``--help`` builds even
    without the package on ``PYTHONPATH``.
    """
    parts = [p.strip() for p in str(value).split(",")]
    if not parts or any(p == "" for p in parts):
        raise argparse.ArgumentTypeError(
            f"--use_side_chain_context_schedule must be a non-empty comma-"
            f"separated list of 0/1 (e.g. '1,1,0'), got {value!r}",
        )
    out: list[int] = []
    for p in parts:
        if p not in ("0", "1"):
            raise argparse.ArgumentTypeError(
                f"--use_side_chain_context_schedule entries must each be 0 or 1, "
                f"got {p!r} in {value!r}",
            )
        out.append(int(p))
    return out


def _apply_scc_schedule(cycles: list, schedule: list[int]) -> None:
    """Override each cycle's ``use_side_chain_context`` from ``schedule`` IN PLACE.

    Broadcast/truncate rule (clear + documented): the schedule is matched to the
    cycle count position-by-position; if it is SHORTER than the cycle count, its
    LAST entry is repeated to fill the remaining (later) cycles (so ``'1,0'`` over
    3 cycles becomes ``[1, 0, 0]`` — ON early, OFF late carries through); if it is
    LONGER, the extra trailing entries are ignored (truncate to the cycle count).
    A length-1 schedule broadcasts that single value to every cycle.

    Called ONLY when ``--use_side_chain_context_schedule`` is given, so when the
    flag is absent each cycle keeps the uniform ``--use_side_chain_context``
    value => byte-identical.
    """
    if not cycles or not schedule:
        return
    last = schedule[-1]
    for i, cyc in enumerate(cycles):
        cyc.use_side_chain_context = schedule[i] if i < len(schedule) else last


def _resolve_composition_pool(
    *,
    survivors_prev,
    fallback_pool,
    composition_pool_fallback: bool,
):
    """Choose the pool the WS-C cap / class-balance / soft-bias should derive from.

    The WS-C levers normally read the previous cycle's *survivor* pool. On a
    hydrophobic seed where ~100% of samples fail the GRAVY band, that survivor
    pool is empty, so the levers never fire and the run ships a runaway single-AA
    composition (the ~26%-Ala backfill mode). With ``--composition_pool_fallback``
    set, when survivors are empty we instead derive them from the PREVIOUS cycle's
    full SAMPLED (pre-band-filter) pool — the SAME source the adaptive controller
    reads (``_load_cycle_seq_stage_pool``) — so the cap/class-balance/soft-bias
    still engage.

    Returns the fallback DataFrame ONLY when: the flag is set AND survivors are
    empty/None AND a non-empty fallback pool exists. Otherwise returns ``None``
    (the gated block then runs unchanged on survivors => byte-identical when the
    flag is off, survivors exist, or there is no previous pool, e.g. cycle 0).
    """
    if not composition_pool_fallback:
        return None
    if survivors_prev is not None and len(survivors_prev) > 0:
        return None
    if fallback_pool is None or len(fallback_pool) == 0:
        return None
    return fallback_pool


def _aa_reference_arg(value: str) -> str:
    """Validate ``--aa_reference NAME`` against the bundled baseline keys.

    NAME selects which Swiss-Prot AA-composition distribution the over-
    representation baseline is scored against (z-scores / class-balance / the
    hydrophobic over-rep mask). The default is the EC-3 hydrolase baseline; a
    non-hydrolase enzyme should select its own EC class (e.g.
    ``swissprot_ec2_transferases_2026_01``) so designs are compared to the
    right distribution rather than the wrong one.

    Fail-fast at parse time on an unknown key, listing the valid keys, so a
    typo can never silently fall through to the wrong reference. The import is
    deliberately LAZY (inside the function body): ``type=`` callables are only
    invoked when a value is actually supplied, never for ``--help``, so this
    keeps ``--help`` working even without ``protein_chisel`` on the path.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.expression.aa_composition import REFERENCE_DISTRIBUTIONS
    if value not in REFERENCE_DISTRIBUTIONS:
        raise argparse.ArgumentTypeError(
            f"unknown --aa_reference {value!r}; choose from "
            f"{sorted(REFERENCE_DISTRIBUTIONS)}",
        )
    return value


def _parse_charge_band_arg(s: str) -> tuple:
    """Parse ``--adaptive_charge_band 'LO,HI'`` -> ``(lo, hi)`` floats.

    Requires EXACTLY two finite values (a 3-field value was previously truncated
    silently). ``default_axes`` additionally validates ``lo < hi`` at controller
    setup, so a bad band fails fast at startup.
    """
    parts = str(s).split(",")
    if len(parts) != 2:
        raise ValueError(
            f"--adaptive_charge_band must be 'LO,HI' (exactly two values), got {s!r}")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(
            f"--adaptive_charge_band values must be numbers, got {s!r}")
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"--adaptive_charge_band values must be finite, got {s!r}")
    return (lo, hi)


def _parse_catalytic_resnos_arg(s: str) -> tuple[int, ...]:
    """Parse ``--catalytic_resnos '10,20,30'`` -> ``(10, 20, 30)``, sorted
    ascending and de-duplicated.

    These are 1-indexed PDB resseq numbers on the catalytic chain (CHAIN). An
    explicit override exists for scaffolds whose seed PDB lacks a REMARK 666
    block: without it the driver falls back to the PTE_i1 builtin positions,
    which are wrong for any other enzyme. Requires at least one value and
    every value a positive int (resseq is 1-indexed) — a typo'd/empty value
    would otherwise silently pin the wrong residues.
    """
    parts = [tok.strip() for tok in str(s).split(",") if tok.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            f"--catalytic_resnos must list >=1 residue number, got {s!r}")
    out: set[int] = set()
    for tok in parts:
        try:
            v = int(tok)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"--catalytic_resnos values must be integers, got {tok!r}")
        if v < 1:
            raise argparse.ArgumentTypeError(
                f"--catalytic_resnos values must be positive (1-indexed "
                f"resseq), got {v}")
        out.add(v)
    return tuple(sorted(out))


def _parse_plm_class_strength(s: str) -> dict:
    """Parse ``--plm_class_strength 'class=val,...'`` -> ``{class: weight}``.

    ABSOLUTE per-class overrides of ``FusionConfig.class_weights`` (not multipliers):
    e.g. ``distal_surface=0.3`` sets that class's PLM weight to 0.3 (then scaled by
    the global ``--plm_strength``). Validates each key is a known position class and
    each value is finite and ``>= 0`` — a typo'd class would otherwise be a silent
    no-op (``_lookup`` just wouldn't read it). Empty string -> ``{}`` (byte-identical
    default).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.scoring.multi_objective import parse_kv_string
    from protein_chisel.tools.classify_positions import NEW_CLASSES
    overrides = parse_kv_string(s)            # raises on a missing '=' / non-float
    # Only the CURRENT (directional 6-class) taxonomy: legacy tables are re-classified
    # to these names before fusion, so a legacy key (surface/buried/…) would be a
    # silent no-op — reject it (the "no silent no-op" guarantee, per review).
    known = set(NEW_CLASSES)
    for cls, val in overrides.items():
        if cls not in known:
            raise ValueError(
                f"--plm_class_strength: unknown position class {cls!r}; "
                f"choose from {sorted(known)}")
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(
                f"--plm_class_strength: {cls}={val} must be a finite weight >= 0")
    return overrides


def _resolve_catalytic_resnos(
    override: Optional[Iterable[int]],
    seed_pdb: Path | str,
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """Resolve the catalytic residue set for ANY scaffold, by priority:

        1. ``override`` (``--catalytic_resnos``) — explicit user intent.
        2. The seed PDB's ``REMARK 666`` motif block — auto-derived.
        3. The hard-coded PTE_i1 builtin (``DEFAULT_CATRES`` /
           ``CATALYTIC_HIS_RESNOS``) — correct ONLY for that scaffold.

    Returns ``(all_catalytic_resnos, his_only_catalytic_resnos, source)`` with
    ``source`` in ``{"override", "remark_666", "builtin"}``, all resno tuples
    sorted ascending.

    When (and only when) it falls through to case 3 — no override AND no
    REMARK 666 — it emits a LOUD ``LOGGER.warning`` naming the PTE-specific
    residues, flagging them as almost certainly wrong for a non-PTE scaffold,
    and pointing at ``--catalytic_resnos``. The VALUES are unchanged from the
    historic silent fallback (byte-identical: same residues, just warned).

    For the override case the HIS subset is taken to be the SAME set as the
    overridden resnos: every downstream consumer of the HIS subset (e.g.
    ``_detect_hbond_to_his_sidechain``) is resname-guarded, so a non-HIS resno
    in that set simply never matches a HIS atom — a safe superset.
    """
    if override is not None:
        ov = tuple(sorted({int(r) for r in override}))
        if ov:
            return ov, ov, "override"

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.protonate_final import parse_remark_666 as _parse666
    entries = _parse666(seed_pdb)
    if entries:
        all_resnos = tuple(sorted({e.motif_resno for e in entries}))
        his_resnos = tuple(sorted({
            e.motif_resno for e in entries if e.motif_resname.upper() == "HIS"
        }))
        return all_resnos, his_resnos, "remark_666"

    # No override and no REMARK 666 -> the PTE_i1 builtin. Correct ONLY for the
    # PTE scaffold; loudly warn so a non-PTE run can't silently pin the wrong
    # residues (the historic failure mode this resolver exists to fix).
    LOGGER.warning(
        "No --catalytic_resnos override and NO REMARK 666 block in seed_pdb "
        "(%s): falling back to the hard-coded PTE_i1 catalytic residues "
        "all=%s his=%s. These are almost certainly WRONG for a non-PTE "
        "scaffold and will pin/fix the wrong positions. Pass "
        "--catalytic_resnos 'r1,r2,...' (1-indexed resseq on chain %s) or add "
        "a REMARK 666 motif block to the seed PDB.",
        seed_pdb, DEFAULT_CATRES, CATALYTIC_HIS_RESNOS, CHAIN,
    )
    return DEFAULT_CATRES, CATALYTIC_HIS_RESNOS, "builtin"


# Apptainer / cluster paths
UNIVERSAL_SIF = Path("/net/software/containers/universal.sif")
FPOCKET_BIN = Path("/net/software/lab/fpocket/bin/fpocket")

DEFAULT_OUT_ROOT = Path(os.environ.get("WORK_ROOT") or
                          f"/net/scratch/{os.environ.get('USER', 'unknown')}")

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
    # Acceptance bands. Each default is the CURRENT hardcoded FINAL
    # (cycle-2, strictest) value, so an unparameterised call reproduces
    # the legacy per-cycle schedule byte-for-byte. CLI flags / env vars
    # override these (see main()'s --net_charge_* / --gravy_* / etc.).
    net_charge_min: float = -18.0,
    net_charge_max: float = -4.0,
    sap_max_threshold: float = 100.0,
    instability_max: float = 60.0,
    gravy_min: float = -0.8,
    gravy_max: float = 0.3,
    aliphatic_min: float = 40.0,
    boman_max: float = 4.5,
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

    Acceptance bands (``net_charge_min/max``, ``sap_max_threshold``,
    ``instability_max``, ``gravy_min/max``, ``aliphatic_min``,
    ``boman_max``, ``pi_min/max``) are the FINAL (strictest) band:
      - ``constant`` strategy applies each one to EVERY cycle directly.
      - ``annealing`` strategy applies the value to cycle 2 (final) and
        relaxes cycles 1 and 0 from it by the fixed legacy offsets; the
        charge / pi / sap bands stay CONSTANT across cycles ("only the
        light filters anneal"). With all bands at their defaults the
        annealing schedule is byte-identical to the legacy hardcoded one.
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
    # Charge / sap / pi bands stay CONSTANT across cycles (user pref:
    # "the current range should be the final one") in BOTH strategies.
    charge_sap = dict(
        net_charge_max=net_charge_max, net_charge_min=net_charge_min,
        sap_max_threshold=sap_max_threshold,
    )
    if strategy == "constant":
        return [
            CycleConfig(
                cycle_idx=0, n_samples=500, sampling_temperature=0.20,
                instability_max=instability_max,
                gravy_min=gravy_min, gravy_max=gravy_max,
                aliphatic_min=aliphatic_min, boman_max=boman_max,
                **charge_sap, **common,
            ),
            CycleConfig(
                cycle_idx=1, n_samples=400, sampling_temperature=0.18,
                instability_max=instability_max,
                gravy_min=gravy_min, gravy_max=gravy_max,
                aliphatic_min=aliphatic_min, boman_max=boman_max,
                **charge_sap, **common,
            ),
            CycleConfig(
                cycle_idx=2, n_samples=300, sampling_temperature=0.15,
                instability_max=instability_max,
                gravy_min=gravy_min, gravy_max=gravy_max,
                aliphatic_min=aliphatic_min, boman_max=boman_max,
                **charge_sap, **common,
            ),
        ]
    # Annealing — gentle relaxation, never tighten beyond the passed FINAL
    # value. The passed value is the cycle-2 (strictest) band; cycles 1 and
    # 0 relax from it by the FIXED legacy offsets (with all bands at their
    # defaults this reproduces the legacy hardcoded schedule byte-for-byte).
    # Cycle 0 (explore): light filters loose; TOPSIS heavy on fitness;
    #                    survivors picked by fitness (legacy).
    # Cycle 1 (transition): light filters slightly loose; balanced
    #                    TOPSIS weights (defaults).
    # Cycle 2 (exploit): light filters at the final band; default TOPSIS
    #                    weights; survivors picked by TOPSIS so the final
    #                    pool reinforces multi-objective good.
    return [
        CycleConfig(
            cycle_idx=0, n_samples=500, sampling_temperature=0.20,
            instability_max=instability_max + 20.0,
            gravy_min=gravy_min - 0.20, gravy_max=gravy_max + 0.10,
            aliphatic_min=aliphatic_min - 10.0, boman_max=boman_max + 1.0,
            topsis_weight_overrides={
                "fitness": 3.0,            # explore aggressively on fitness
                "instability": 0.1, "gravy": 0.1, "aliphatic": 0.1,
                "boman": 0.1, "pocket_hydrophobicity": 0.1,
            },
            use_topsis_for_survivors=False,
            **charge_sap, **common,
        ),
        CycleConfig(
            cycle_idx=1, n_samples=400, sampling_temperature=0.18,
            instability_max=instability_max + 10.0,
            gravy_min=gravy_min - 0.10, gravy_max=gravy_max + 0.05,
            aliphatic_min=aliphatic_min - 5.0, boman_max=boman_max + 0.5,
            topsis_weight_overrides={},        # balanced (defaults)
            use_topsis_for_survivors=True,
            **charge_sap, **common,
        ),
        CycleConfig(
            cycle_idx=2, n_samples=300, sampling_temperature=0.15,
            instability_max=instability_max,
            gravy_min=gravy_min, gravy_max=gravy_max,
            aliphatic_min=aliphatic_min, boman_max=boman_max,
            topsis_weight_overrides={},        # balanced (defaults)
            use_topsis_for_survivors=True,
            **charge_sap, **common,
        ),
    ]


def debug_short_test_cycles(
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
    net_charge_min: float = -18.0,
    net_charge_max: float = -4.0,
    sap_max_threshold: float = 100.0,
    instability_max: float = 60.0,
    gravy_min: float = -0.8,
    gravy_max: float = 0.3,
    aliphatic_min: float = 40.0,
    boman_max: float = 4.5,
) -> list[CycleConfig]:
    """Hardcoded fast smoke-test preset: 20/10/10 samples across 3 cycles.

    Activated by ``--debug-short-test``. **Unconditionally** clobbers
    ``args.target_k`` to 5 and ``args.cycles`` to 3, even if the caller
    set them to other values on the same command line; a WARNING is
    logged when that happens so it isn't silent. Intended for end-to-
    end pipeline validation only — never use for production runs. Band
    overrides are forwarded so a debug run honors the same flags.
    """
    cycles = default_cycles(
        omit_AA=omit_AA,
        use_side_chain_context=use_side_chain_context,
        enhance=enhance,
        pi_min=pi_min,
        pi_max=pi_max,
        fpocket_druggability_min=fpocket_druggability_min,
        clash_filter=clash_filter,
        strategy=strategy,
        consensus_threshold=consensus_threshold,
        consensus_strength=consensus_strength,
        consensus_max_fraction=consensus_max_fraction,
        net_charge_min=net_charge_min,
        net_charge_max=net_charge_max,
        sap_max_threshold=sap_max_threshold,
        instability_max=instability_max,
        gravy_min=gravy_min,
        gravy_max=gravy_max,
        aliphatic_min=aliphatic_min,
        boman_max=boman_max,
    )
    for cyc, n_samples in zip(cycles, (20, 10, 10)):
        cyc.n_samples = n_samples
    return cycles[:3]


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
    bulky_aas: str = _CLASH_BULKY_AAS,   # K is as long as R (Cb->NZ ~6 A)
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

    Result: positions where a bulky AA has no fitting rotamer get the full
    ``-bias_strength_per_pct_clash`` (−20 nats at the default strength=20);
    positions where they fit fine get 0; in-between get proportional. Replaces
    the previous all-or-nothing hard-omit which forbade the bulky set at every
    clash-prone position even when they fit.

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


def _read_seed_tunnel_lining(tsv_path) -> set:
    """Resnos with ``is_tunnel_lining==True`` from ``seed_tunnel_residues.tsv``.

    SINGLE source of truth for "tunnel lining" — the seed fpocket annotation
    (``annotate_seed_tunnel_residues``), shared by WS-G's omit and WS-D's surface-
    scope exclusion so the two can never drift. Returns an empty set on ANY failure
    (missing file, empty/failed annotation, schema change) so callers degrade to a
    no-op rather than crash.
    """
    try:
        df = pd.read_csv(tsv_path, sep="\t")
        return {int(r) for r in df.loc[df["is_tunnel_lining"].astype(bool), "resno"]}
    except Exception:
        return set()


def _canonical_omit_aas(omit_AA: str) -> str:
    """The canonical AAs the sampler is forbidden to pick, derived from an omit string.

    Strips the structural ``X`` placeholder and upper-cases; returns the remaining hard-
    omitted canonical AAs (e.g. ``"CX"`` -> ``"C"``, ``"WX"`` -> ``"W"``). Shared by the
    WS-C composition exclusion and the seed-triage z-gate exclusion so that an AA the
    sampler CANNOT pick is never up-weighted, capped, nor used as an over-representation
    signal (codex: a single-cysteine special-case missed ``--omit_AA WX`` etc.).
    """
    return "".join(c for c in str(omit_AA).upper() if c != "X")


def _build_tunnel_lining_omit(lining_resnos, chain: str, omit_aas: str, *,
                              fixed_resnos=()) -> dict[str, str]:
    """WS-G ``--omit_tunnel_lining``: forbid bulky/hydrophobic AAs at tunnel-lining
    positions to keep the substrate channel open.

    Returns ``{"<chain><resno>": AAs}`` for each NON-fixed lining resno (fixed /
    catalytic residues keep their pinned identity). ``{}`` when the lining set or the
    canonical AA set is empty → byte-identical no-op. Pure; mirrors
    :func:`_build_fraction_cap_omit`. The default set (``_OMIT_TUNNEL_LINING_DEFAULT``
    = ``FHKRWY`` = the throat's bulky-blocker set) is the aromatics W/F/Y/H + the long
    charged R/K that line and constrict a tunnel; **Alanine is intentionally excluded**
    — it is small and cannot constrict (controlling Ala over-representation is WS-C's
    job, not WS-G's), and the medium hydrophobics I/L/M/V are left to the soft throat-
    feedback bias rather than a permanent hard ban.
    """
    aas = "".join(sorted({a for a in str(omit_aas).upper() if a in _CANONICAL_AAS}))
    if not aas:
        return {}
    # Never omit so many AAs that a lining position is left with fewer than
    # _MIN_SAMPLEABLE_AAS_AFTER_CAP choices (fused MPNN would otherwise sample
    # uniformly from the "forbidden" set) — fail fast on an absurd user set (codex).
    if len(_CANONICAL_AAS) - len(aas) < _MIN_SAMPLEABLE_AAS_AFTER_CAP:
        raise ValueError(
            f"--omit_tunnel_lining_aas {omit_aas!r} omits {len(aas)} canonical AAs, "
            f"leaving fewer than {_MIN_SAMPLEABLE_AAS_AFTER_CAP} sampleable at lining "
            f"positions; use a smaller set (default {_OMIT_TUNNEL_LINING_DEFAULT}).")
    fixed = {int(r) for r in fixed_resnos}
    return {f"{chain}{int(r)}": aas for r in lining_resnos if int(r) not in fixed}


# The 20 canonical amino acids (set; order-independent membership tests).
_CANONICAL_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _clamp_bias_total(
    bias_k: np.ndarray,
    clamp: Optional[float],
    bias_AA_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """WS-E ``--bias_total_clamp``: bound the EFFECTIVE per-(pos, AA) sampling bias.

    LigandMPNN adds the per-residue bias (``bias_k``) AND the global ``bias_AA``
    separately, so the effective bias at ``(pos, AA)`` is their sum. When
    ``bias_AA_vec`` (a ``(20,)`` per-AA vector parsed from the FINAL serialized
    ``bias_AA`` string) is given, the clamp bounds ``bias_k + bias_AA`` to
    ``±clamp`` and returns the adjusted ``bias_k`` (so the sum stays in band, while
    the separately-passed global term is preserved); a cell already in band is
    unchanged. Without ``bias_AA_vec`` it clamps ``bias_k`` alone.

    ``clamp is None`` returns ``bias_k`` UNCHANGED (the same object) — the
    byte-identical default. Bounds the otherwise-uncapped consensus(+2.0) + PLM-peak
    stack that, at T≈0.15, locks a cell near-deterministically (~10¹³× odds).
    """
    if clamp is None:
        return bias_k
    n = abs(float(clamp))
    if bias_AA_vec is None:
        return np.clip(bias_k, -n, n)
    g = np.asarray(bias_AA_vec, dtype=bias_k.dtype)[None, :]
    total = bias_k + g
    clipped = np.clip(total, -n, n)
    # Only adjust cells that actually exceeded the band; leave in-band bias_k EXACT
    # (the (bias_k+g)-g round-trip would otherwise perturb in-band cells by ~1e-7 in
    # float32 and inflate the "n cells adjusted" telemetry).
    return np.where(clipped != total, clipped - g, bias_k).astype(bias_k.dtype)


# F3 (v1.4.0): the bias-sum safety cap is ON BY DEFAULT at this many nats. 3 nats
# preserves a legit ~3-nat single-source PLM peak (so a clean PLM-on run is essentially
# unaffected) while capping the pathological 6-20-nat double-count lock (codex: a fixed-
# nats cap is the right semantic — it allows <=3 nats at EVERY temperature, unlike the
# odds form or the coordinator's 1e3x, both of which would clip a legit peak). This is a
# deliberate default-path change (like the 1.2.0 damping flip); --no_bias_total_clamp
# (or NO_BIAS_TOTAL_CLAMP=1) restores the exact pre-1.4.0 (unclamped) path.
_DEFAULT_BIAS_TOTAL_CLAMP_NATS = 3.0


def _resolve_bias_total_clamp_default(
    *,
    bias_total_clamp: Optional[float],
    bias_total_clamp_odds: Optional[float],
    no_clamp: bool,
) -> Optional[float]:
    """Decide the effective ``--bias_total_clamp`` (nats) after applying the F3 default.

    Pure precedence resolver (keeps the driver glue a one-liner + makes the rule unit-
    testable without argparse):

    * ``no_clamp`` (``--no_bias_total_clamp``) -> ``None`` — the EXACT legacy path
      (``_clamp_bias_total`` is then the identity no-op => byte-identical to pre-1.4.0).
    * an explicit ``--bias_total_clamp`` (not ``None``, INCLUDING ``0.0``) -> kept verbatim
      (the user's value wins; never silently bumped to the default).
    * an explicit ``--bias_total_clamp_odds`` -> leave the nats clamp ``None`` so the odds
      path owns the clamp that cycle AND the two never collide / falsely trip the existing
      nats-vs-odds mutual-exclusion guard.
    * otherwise (a bare run) -> inject :data:`_DEFAULT_BIAS_TOTAL_CLAMP_NATS` (3.0).

    NOTE: callers must run this AFTER the existing nats-vs-odds mutual-exclusion check so
    the injected default never participates in that check.
    """
    if no_clamp:
        return None
    if bias_total_clamp is not None:
        return bias_total_clamp
    if bias_total_clamp_odds is not None:
        return None
    return _DEFAULT_BIAS_TOTAL_CLAMP_NATS


# WS-C fraction cap never leaves a designable position with fewer than this many
# sampleable AAs (guards the all-AAs-omitted → uniform-from-forbidden MPNN failure).
_MIN_SAMPLEABLE_AAS_AFTER_CAP = 3

# WS-G --omit_tunnel_lining_aas default = the throat's bulky-blocker set
# (tunnel_metrics.bulky_blocker_aas(0.70) = "_BLOCKER_WEIGHT >= 0.70"): aromatics
# W/F/Y/H + the long charged R/K. A guard-tested *literal* (not an import-time call)
# so arg-parsing never has to import protein_chisel — see
# test_ws_g_default_constant_matches_throat_bulky_set, which fails loudly if the
# blocker weights drift out of sync with this string.
_OMIT_TUNNEL_LINING_DEFAULT = "FHKRWY"

# Upper bound (nats) on --composition_soft_bias_nats: well past a hard ban (the
# effective odds penalty is exp(nats / T); at T≈0.15 even 0.5 is ~28x), and small
# enough that the bias never overflows the float32 sampler matrix to -inf.
_SOFT_BIAS_NATS_MAX = 20.0

# WS-C composition soft-bias (pool-derived per cycle): a per-residue SOFT_BIAS
# liability is applied only if it appears in >= _SOFT_BIAS_MIN_SUPPORT of the
# survivor pool (so a one-off survivor can't pollute the bias), and only LOCAL
# hits (span <= _SOFT_BIAS_MAX_SPAN_FRAC * L) count — whole-protein composition
# hits are left to the suppress-all / fraction-cap levers.
_SOFT_BIAS_MIN_SUPPORT = 0.5
_SOFT_BIAS_MAX_SPAN_FRAC = 0.5


def _build_fraction_cap_omit(
    pool_seq: str,
    cap: Optional[float],
    *,
    protein_resnos: Iterable[int],
    fixed_resnos: Iterable[int],
    chain: str = CHAIN,
    exclude_aas: str = "",
) -> dict[str, str]:
    """WS-C per-AA fraction cap → per-residue omit dict (``--aa_fraction_cap``).

    Returns ``{"<chain><resno>": "AAs"}`` forbidding every amino acid whose
    fraction in ``pool_seq`` is at/over ``cap`` (e.g. 0.15), at every NON-FIXED
    designable position (``protein_resnos`` minus ``fixed_resnos``). Catalytic /
    fixed positions keep their identity (they are never redesigned). Members of
    ``exclude_aas`` (already hard-omitted, e.g. cysteine) are never re-capped.

    ``cap is None`` (the default) or no AA over the cap → ``{}`` (a no-op, so the
    merged omit — and the whole run — is byte-identical). Pure + reference-free.
    """
    if cap is None:
        return {}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.expression.aa_composition import AA_ORDER_REF, over_cap_aas
    capped = over_cap_aas(pool_seq, cap, exclude_aas=exclude_aas)
    if not capped:
        return {}
    # SAFETY GUARD: never omit so many AAs that the sampler is left with fewer
    # than _MIN_SAMPLEABLE_AAS_AFTER_CAP choices. Omitting ALL canonical AAs makes
    # fused MPNN's per-AA omit logits equal (-1e8), so it samples uniformly from
    # the supposedly-forbidden set — a silent failure worse than a crash. A cap
    # this aggressive is a misconfiguration (cap too low for a low-diversity
    # pool); skip it this cycle and surface it loudly.
    n_sampleable = len(set(AA_ORDER_REF) - {c for c in exclude_aas.upper()})
    if len(capped) > max(0, n_sampleable - _MIN_SAMPLEABLE_AAS_AFTER_CAP):
        LOGGER.error(
            "aa_fraction_cap=%.4f would omit %d of %d sampleable AAs (%s), leaving "
            "fewer than %d — cap too low for this pool; SKIPPING the cap this cycle.",
            cap, len(capped), n_sampleable, "".join(capped),
            _MIN_SAMPLEABLE_AAS_AFTER_CAP,
        )
        return {}
    cap_str = "".join(capped)
    fixed_set = {int(r) for r in fixed_resnos}
    return {
        f"{chain}{int(r)}": cap_str
        for r in protein_resnos if int(r) not in fixed_set
    }


def _enforce_min_sampleable_after_cap(
    merged_omit: dict[str, str],
    base_omit: dict[str, str],
    global_omit_AA: str,
    min_keep: int = _MIN_SAMPLEABLE_AAS_AFTER_CAP,
) -> dict[str, str]:
    """Post-merge safety net for ``--aa_fraction_cap``.

    The per-position fraction-cap omit is unioned with the structural omits
    (expression hard-omit, first-shell diversity, position-1 M) *and* the global
    ``omit_AA``. Even when the cap-set alone left enough AAs, that union can drive
    an individual position to **zero** sampleable canonical AAs — which fused MPNN
    encodes as all-equal ``-1e8`` logits and then samples *uniformly from the
    "forbidden" set*. For any position the merge leaves with fewer than
    ``min_keep`` sampleable canonical AAs, REVERT that position to its pre-cap
    (``base_omit``) value — the structural omits never over-omit, so the revert is
    always safe. Returns a (possibly modified) copy; ``merged_omit`` is unchanged.
    """
    g = _CANONICAL_AAS & set(str(global_omit_AA).upper())
    n_canon = len(_CANONICAL_AAS)
    out = dict(merged_omit)
    reverted: list[str] = []
    structural: list[str] = []                # still degenerate even without the cap
    for key, aas in list(out.items()):
        omitted = g | (_CANONICAL_AAS & set(str(aas).upper()))
        if n_canon - len(omitted) < min_keep:
            b = base_omit.get(key, "")
            if b:
                out[key] = b
            else:
                del out[key]
            reverted.append(key)
            # Reverting removes the cap; verify the structural base itself (with
            # the global omit_AA) is safe. If not, the over-omit is NOT cap-induced
            # — the guard cannot fix it (the cap is already gone), so surface it.
            base_omitted = g | (_CANONICAL_AAS & set(str(b).upper()))
            if n_canon - len(base_omitted) < min_keep:
                structural.append(key)
    if reverted:
        LOGGER.error(
            "aa_fraction_cap: reverted the cap at %d position(s) %s — the merged "
            "omit (cap + structural + global omit_AA) would have left fewer than "
            "%d sampleable AAs there.",
            len(reverted), sorted(reverted)[:10], min_keep,
        )
    if structural:
        LOGGER.error(
            "aa_fraction_cap guard: position(s) %s remain below %d sampleable AAs "
            "EVEN WITHOUT THE CAP (structural omit + global omit_AA) — a pre-existing "
            "over-omit the cap guard cannot repair.",
            sorted(structural)[:10], min_keep,
        )
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

    # ---- PoE score-only mode: candidates come from a finished host PoE stage,
    # NOT from in-process sampling. Build the CandidateSet from the PoE output dir
    # and link its packed/ PDBs into out_dir so stage_restore_pdbs finds them
    # unchanged (the PoE output is structurally identical to our sampler's). ----
    if _POE_SAMPLE_DIR is not None:
        from protein_chisel.sampling.mpnn_backends import candidate_set_from_poe_dir
        cset = candidate_set_from_poe_dir(
            _POE_SAMPLE_DIR, seed_pdb.stem,
            parent_design_id=f"PTE_i1_poe_c{cycle_cfg.cycle_idx}",
            experts=_POE_EXPERTS, lambdas=_POE_LAMBDAS,
            temperature=_POE_TEMPERATURE,   # the temp PoE actually sampled at
        )
        cand_fasta = out_dir / "candidates.fasta"
        cand_tsv = out_dir / "candidates.tsv"
        cset.to_disk(cand_fasta, cand_tsv)
        packed_link = out_dir / "packed"
        if not packed_link.exists():
            os.symlink(Path(_POE_SAMPLE_DIR).resolve() / "packed", packed_link)
        LOGGER.info("stage_sample[cycle=%d]: PoE score-only — %d candidates from %s "
                    "(experts=%s lambdas=%s)", cycle_cfg.cycle_idx,
                    len(cset.df), _POE_SAMPLE_DIR, list(_POE_EXPERTS),
                    list(_POE_LAMBDAS))
        return cand_tsv

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

    # ---- Feature 1: per-cycle probabilistic conserved-sidechain-H-bond fixing.
    # Detect designable sidechains H-bonding to the ligand / fixed residues and
    # probabilistically pin them into THIS cycle's fixed set (independent roll per
    # cycle, reproducible from CONSERVE_SEED_BASE). Uses the SHARED
    # protein_chisel.tools.conserved_hbonds helpers (same as chisel_ligandMPNN.py).
    fixed_resnos = set(int(r) for r in fixed_resnos)
    if CONSERVE_HBONDS:
        from protein_chisel.tools.conserved_hbonds import build_interaction_network
        designable = sorted(set(int(r) for r in protein_resnos) - fixed_resnos)
        # When growing the network across cycles, residues pinned in earlier
        # cycles join the active-site anchors so the network deepens over time.
        extra_anchors = (_CONSERVE_GROWN_ANCHORS & set(protein_resnos)
                         if CONSERVE_GROW_NETWORK else set())
        seed_str = f"{CONSERVE_SEED_BASE}:{cycle_cfg.cycle_idx}"
        # depth=1 + ("hbond",) + no extra anchors == legacy single-shell fixing.
        net = build_interaction_network(
            seed_pdb,
            designable_resnos=designable,
            catalytic_resnos=(fixed_resnos if "catalytic" in CONSERVE_ANCHORS else ()),
            user_fixed_resnos=extra_anchors,
            include_ligand=("ligand" in CONSERVE_ANCHORS),
            interaction_types=CONSERVE_INTERACTION_TYPES,
            chain=chain,
            depth=CONSERVE_HBOND_DEPTH,
            prob=CONSERVE_HBOND_PROB,
            shell_decay=CONSERVE_SHELL_DECAY,
            keep_clashing=CONSERVE_KEEP_CLASHING,
            seed=seed_str,
            max_dist=CONSERVE_HBOND_MAX_DIST,
            max_angle_deg=CONSERVE_HBOND_MAX_ANGLE,
        )
        for shell in net.shells:
            for resno, clash_with in shell.excluded:
                LOGGER.info(
                    "stage_sample[cycle=%d]: excluding conserved A%d "
                    "(sidechain clashes %s; --conserve_keep_clashing to keep)",
                    cycle_cfg.cycle_idx, resno, clash_with,
                )
            LOGGER.info(
                "stage_sample[cycle=%d]: conserve shell %d (p=%.2f, seed=%s): "
                "fixing %s of candidates %s",
                cycle_cfg.cycle_idx, shell.depth,
                CONSERVE_HBOND_PROB * (CONSERVE_SHELL_DECAY ** (shell.depth - 1)),
                seed_str, shell.rolled, shell.candidates,
            )
        rolled = net.rolled
        fixed_resnos |= rolled
        if CONSERVE_GROW_NETWORK:
            _CONSERVE_GROWN_ANCHORS.update(rolled)

    LOGGER.info(
        "stage_sample[cycle=%d]: n=%d, T=%.3f, fixed=%s, "
        "mean_abs_bias=%.4f, bias_AA=%r",
        cycle_cfg.cycle_idx, cycle_cfg.n_samples,
        cycle_cfg.sampling_temperature, sorted(set(fixed_resnos)),
        float(np.abs(bias).mean()), bias_AA or "(none)",
    )

    # ---- PoE emit-inputs mode: write the exact cycle-0 bias/fixed/omit JSONs this
    # run computed (incl. conserved-hbond rolls just applied above) for the host PoE
    # stage to consume, then exit. Reuses the shared _build_* helpers — the JSON
    # layout is identical to what sample_with_ligand_mpnn passes to the sampler. ----
    if _POE_EMIT_INPUTS_DIR is not None:
        from protein_chisel.tools.ligand_mpnn import (
            _build_bias_per_residue_multi, _build_fixed_residues_multi,
        )
        emit = Path(_POE_EMIT_INPUTS_DIR)
        emit.mkdir(parents=True, exist_ok=True)
        # CRITICAL: key the JSONs by the LITERAL seed path that the host PoE run.py
        # receives as --pdb_path (str(seed_pdb), NOT .resolve()). The shell passes the
        # same raw $SEED_PDB to both the driver (--seed_pdb) and run.py (--pdb_path),
        # and run.py looks up bias/fixed/omit by that verbatim string. On symlinked
        # /net/scratch -> /mnt/net/scratch compute nodes, .resolve() rewrites the
        # prefix => fixed_residues_multi[pdb] KeyErrors (crash) and bias/omit silently
        # fall back to empty (PoE samples WITHOUT our calibrated bias). Same defect +
        # fix as chisel_ligandMPNN @1b4606e. The _build_* helpers key by .resolve(), so
        # re-key their single-entry dicts to the literal path here.
        literal_key = str(seed_pdb)
        _bias = _build_bias_per_residue_multi(seed_pdb, bias, chain, protein_resnos)
        _fixed = _build_fixed_residues_multi(seed_pdb, sorted(set(fixed_resnos)), chain)
        (emit / "bias.json").write_text(json.dumps(
            {literal_key: next(iter(_bias.values()), {})}, indent=2))
        (emit / "fixed.json").write_text(json.dumps(
            {literal_key: next(iter(_fixed.values()), [])}, indent=2))
        (emit / "omit.json").write_text(json.dumps(
            {literal_key: (omit_AA_per_residue or {})}, indent=2))
        LOGGER.info("PoE emit-inputs: wrote bias/fixed/omit JSONs -> %s "
                    "(key=%s, fixed=%d residues, omit=%d) — exiting (host PoE next)",
                    emit, literal_key, len(set(fixed_resnos)),
                    len(omit_AA_per_residue or {}))
        sys.exit(0)

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
    out_map = restore_sample_dir(
        sample_dir=sample_dir,
        ref_pdb=ref_pdb,
        out_pdb_dir=out_pdb_dir,
        pdb_basename=pdb_basename,
        candidate_ids=candidate_ids,
        chain=chain,
        catalytic_resnos=catalytic_resnos,
        catalytic_hydrogens=catalytic_hydrogens,
    )
    # ---- Feature 2: canonical REMARK transfer (no DESIGN_PATH stamp here).
    # restore_sample_dir only carries REMARK 666 / HETNAM / LINK / PDBinfo-LABEL;
    # rescue the rest of the seed's records (665/667/668/QCB/DESIGN_PATH chain)
    # onto the per-cycle restored PDBs via the SHARED protein_chisel.tools.remarks
    # module. These are INTERMEDIATES, so we pass design_path_stage=None — the
    # single iterative_design DESIGN_PATH line is stamped once at final adoption
    # (_write_final_topk_artifacts), pointing at the adopted top-K PDB rather than
    # an intermediate cycle path.
    if TRANSFER_REMARKS:
        from protein_chisel.tools.remarks import transfer_remarks_to_dir
        n = transfer_remarks_to_dir(
            out_pdb_dir, ref_pdb,
            transfer_input=True,
            design_path_stage=None,
        )
        LOGGER.info(
            "stage_restore_pdbs: carried canonical REMARKs (665/666/667/668/QCB/"
            "DESIGN_PATH chain) onto %d intermediate PDB(s)", n,
        )
    return out_map


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

    # --filters gating: a filter that isn't selected must not append a reject
    # reason (so it stops dropping designs). Reads the module-global selection via
    # _filter_active(); at --filters all every filter is on => byte-identical.
    # Predicates are short-circuited by `_on(name) and <cond>` so the (cheap) check
    # isn't even evaluated when the filter is off.
    _on = _filter_active

    rows: list[dict] = []
    for _, row in df.iterrows():
        seq = row["sequence"]
        reasons: list[str] = []
        if _on("length") and len(seq) != wt_length:
            reasons.append(f"length {len(seq)} != WT {wt_length}")
        pp = protparam_metrics(
            seq, ph=design_ph,
            n_term_pad=n_term_pad, c_term_pad=c_term_pad,
        )
        # Filter on the ROBUST full-HH charge (all 7 ionizables + termini).
        # The minimalist 'no_HIS' (still computed below as a diagnostic
        # column) is too lenient because it misses Cys/Tyr at high pH.
        if _on("net_charge") and pp.charge_at_pH_full_HH >= net_charge_max:
            reasons.append(
                f"net_charge_full_HH={pp.charge_at_pH_full_HH:.2f} >= {net_charge_max}"
            )
        if _on("net_charge") and pp.charge_at_pH_full_HH <= net_charge_min:
            reasons.append(
                f"net_charge_full_HH={pp.charge_at_pH_full_HH:.2f} <= {net_charge_min}"
            )
        if _on("pi") and not (pi_min <= pp.pi <= pi_max):
            reasons.append(f"pI={pp.pi:.2f} outside [{pi_min}, {pi_max}]")

        # Light de-novo filters on cheap sequence-only metrics. Each
        # threshold is set so they catch *truly broken* designs only —
        # the production pool typically passes all of these comfortably.
        if _on("instability") and pp.instability_index >= instability_max:
            reasons.append(
                f"instability_index={pp.instability_index:.1f} >= {instability_max}"
            )
        if _on("gravy") and not (gravy_min <= pp.gravy <= gravy_max):
            reasons.append(
                f"GRAVY={pp.gravy:+.3f} outside [{gravy_min}, {gravy_max}]"
            )
        if _on("aliphatic") and pp.aliphatic_index < aliphatic_min:
            reasons.append(
                f"aliphatic_index={pp.aliphatic_index:.1f} < {aliphatic_min}"
            )
        if _on("boman") and pp.boman_index >= boman_max:
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
        if _on("expression"):
            for h in eng_res.hard_filter_hits:
                reasons.append(f"{h.rule_name}: {h.reason}")
        n_warnings = len(eng_res.warnings)
        n_soft_bias = len(eng_res.soft_bias_hits)
        n_hard_omit = len(eng_res.hard_omit_hits)
        # Preserve the discrete fail count, but also record how far the
        # sequence sits beyond each numeric threshold. This lets final
        # seq-stage backfill prefer near-misses over sequences that fail
        # by a wide margin, without changing the production hard filters.
        # Backfill "distance to passing" gaps. A deselected filter (--filters)
        # contributes 0 so it can't influence backfill ordering either. At
        # --filters all every _on() is True => identical gaps.
        length_gap = (abs(len(seq) - wt_length) / max(1, wt_length)
                      if _on("length") else 0.0)
        charge_high_gap = (_normalized_upper_bound_gap(
            pp.charge_at_pH_full_HH, net_charge_max) if _on("net_charge") else 0.0)
        charge_low_gap = (_normalized_lower_bound_gap(
            pp.charge_at_pH_full_HH, net_charge_min) if _on("net_charge") else 0.0)
        pi_gap = (_normalized_interval_gap(pp.pi, pi_min, pi_max)
                  if _on("pi") else 0.0)
        instability_gap = (_normalized_upper_bound_gap(
            pp.instability_index, instability_max) if _on("instability") else 0.0)
        gravy_gap = (_normalized_interval_gap(pp.gravy, gravy_min, gravy_max)
                     if _on("gravy") else 0.0)
        aliphatic_gap = (_normalized_lower_bound_gap(
            pp.aliphatic_index, aliphatic_min) if _on("aliphatic") else 0.0)
        boman_gap = (_normalized_upper_bound_gap(pp.boman_index, boman_max)
                     if _on("boman") else 0.0)
        seq_filter_numeric_gap = float(
            length_gap
            + charge_high_gap
            + charge_low_gap
            + pi_gap
            + instability_gap
            + gravy_gap
            + aliphatic_gap
            + boman_gap
        )

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
            "selection__seq_filter_numeric_gap": seq_filter_numeric_gap,
            "selection__seq_filter_gap_length": length_gap,
            "selection__seq_filter_gap_charge_high": charge_high_gap,
            "selection__seq_filter_gap_charge_low": charge_low_gap,
            "selection__seq_filter_gap_pi": pi_gap,
            "selection__seq_filter_gap_instability": instability_gap,
            "selection__seq_filter_gap_gravy": gravy_gap,
            "selection__seq_filter_gap_aliphatic": aliphatic_gap,
            "selection__seq_filter_gap_boman": boman_gap,
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


def _compute_sap_proxy(pdb_path: Path, corrected: bool = False) -> Optional[dict]:
    """SAP proxy via freesasa SASA + Kyte-Doolittle hydrophobicity.

    Per-residue SAP_i = sum over residues j within 10 Å of CA(i):
        (SASA(j) / SASA_max(j)) * weight(restype(j))

    Scales + the spatial reduction live in the shared ``protein_chisel.scoring.sap``
    module (one source of truth, also used by the adaptive controller). The legacy
    ``sap_*`` columns use the signed Kyte-Doolittle weight and are byte-identical to
    before. When ``corrected`` is set it ALSO emits ``sap_corr_*`` using the
    centered, zero-clamped weight, so exposed polar residues can no longer cancel
    hydrophobic neighbours and alanine surfaces register (the legacy proxy's blind
    spots, per the 2026-06 audit).
    """
    try:
        import freesasa
    except ImportError:
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.scoring.sap import (
        THREE_TO_ONE, kd_weight_raw, kd_weight_corrected, sap_neighborhood_metrics,
    )
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
    aas = [THREE_TO_ONE.get(res_data[k]["resname"]) for k in keys_sorted]
    sasa_total = [res_data[k]["sasa"] for k in keys_sorted]

    legacy = sap_neighborhood_metrics(
        aas, sasa_total, cas_a, weight_fn=kd_weight_raw)
    out = {
        "sap_max": legacy["max"],
        "sap_mean": legacy["mean"],
        "sap_p95": legacy["p95"],
    }
    if corrected:
        corr = sap_neighborhood_metrics(
            aas, sasa_total, cas_a, weight_fn=kd_weight_corrected)
        out["sap_corr_max"] = corr["max"]
        out["sap_corr_mean"] = corr["mean"]
        out["sap_corr_p95"] = corr["p95"]
    return out


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
    sap_corrected: bool = False,
) -> Path:
    """Apply h-bond + SAP-proxy structural filter.

    ``sap_corrected`` (opt-in; default False → byte-identical) additionally emits
    ``sap_corr_*`` columns (centered, polar-cancellation-free SAP from the shared
    ``scoring.sap`` module) alongside the legacy ``sap_*``.
    """
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
            seed_dfi_metrics, sap_corrected,
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
        # ThreadPoolExecutor (NOT multiprocessing.Pool) — same reasoning
        # as stage_fpocket_rank: stage_struct_filter runs *after*
        # stage_sample, which initializes torch.cuda inside LigandMPNN.
        # multiprocessing.Pool with the default fork start method then
        # forks a CUDA-initialized parent, which is fork-unsafe and is
        # the leading hypothesis for the catastrophic failure mode where
        # downstream subprocess work (e.g. fpocket) collapses across the
        # whole batch. The struct-filter worker is dominated by numpy
        # geometry + freesasa C calls + biopython parsing, all of which
        # release the GIL, so threads scale well enough in practice and
        # are fork-safe.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_struct_filter_worker, work_args))
    by_cid = {cid: (row, hbonds, reasons) for cid, row, hbonds, reasons in results}

    rows: list[dict] = []
    hbond_rows: list[dict] = []
    for _, row in df.iterrows():
        cid = row["id"]
        wrow, hbonds_, reasons = by_cid.get(cid, ({}, [], ["worker_missing"]))
        hbond_rows.extend(hbonds_)
        # Apply severe-clash filter (worker doesn't have clash_filter flag).
        # Gated by --filters via _filter_active("clash"); all on => identical.
        if clash_filter and _filter_active("clash") and wrow.get("clash__has_severe"):
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
            *(["sap_corr_max", "sap_corr_mean", "sap_corr_p95"]
              if sap_corrected else []),
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


def stage_tunnel_metrics(
    *,
    survivors_struct_tsv: Path,
    pdb_map: dict[str, Path],
    out_dir: Path,
    catalytic_resnos: Iterable[int],
    ligand_min_radius: Optional[float] = None,
    ligand_resname: Optional[str] = None,
    chain: str = "A",
    hard_gate: bool = True,
) -> Path:
    """Pocket-accessibility / tunnel-patency scoring on struct survivors.

    Annotates each survivor with two scorers in parallel:

      * Homegrown ray-cast (tunnel__*): attribution-aware geometric
        score that distinguishes BACKBONE / CATALYTIC_SC / DESIGNABLE_SC
        blockers — only designable-sidechain blockage flows into the
        ranker since that's all MPNN can fix.

      * pyKVFinder (pkvf__*): cavity volume / depth / surface
        connectivity. ``pkvf__has_opening == 0`` is the strongest signal
        that the pocket is buried and inaccessible.

    Optionally HARD GATES designs whose verdict ∈ {buried, ligand_too_big}
    or whose pyKVFinder reports no opening to bulk solvent — those
    cannot be saved by sequence redesign and shouldn't be wasted on by
    downstream stages.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(survivors_struct_tsv, sep="\t")
    if len(df) == 0:
        LOGGER.warning("stage_tunnel_metrics: empty input; nothing to do")
        df.to_csv(out_dir / "survivors_tunnel.tsv", sep="\t", index=False)
        return out_dir / "survivors_tunnel.tsv"

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.tunnel_metrics import (
        score_tunnels, pyKVFinder_score, TunnelConfig,
    )

    cat_list = list(catalytic_resnos)
    cfg = TunnelConfig()
    LOGGER.info(
        "stage_tunnel_metrics: scoring %d designs (catres=%s, "
        "ligand_min_radius=%s, hard_gate=%s)",
        len(df), cat_list, ligand_min_radius, hard_gate,
    )

    # pyKVFinder is optional (depends on universal_with_tunnel_tools.sif)
    try:
        import pyKVFinder  # noqa: F401
        have_pkvf = True
    except ImportError:
        LOGGER.warning(
            "stage_tunnel_metrics: pyKVFinder unavailable in this "
            "environment; pkvf__* columns will be NaN. Run inside "
            "/net/software/containers/users/woodbuse/"
            "universal_with_tunnel_tools.sif for full coverage."
        )
        have_pkvf = False

    rows: list[dict] = []
    n_buried = 0
    n_ligand_too_big = 0
    n_pkvf_no_opening = 0
    for _, row in df.iterrows():
        rec: dict = {"id": row["id"]}
        pdb = pdb_map.get(row["id"])
        if pdb is None or not pdb.is_file():
            rec["tunnel__verdict"] = "missing_pdb"
            rows.append(rec)
            continue
        try:
            scores = score_tunnels(
                pdb_path=pdb,
                catalytic_resnos=cat_list,
                chain=chain,
                ligand_resname=ligand_resname,
                ligand_min_radius=ligand_min_radius,
                config=cfg,
            )
            rec.update(scores.to_dict())
            if scores.verdict == "buried":
                n_buried += 1
            elif scores.verdict == "ligand_too_big":
                n_ligand_too_big += 1
        except Exception as exc:
            LOGGER.warning("score_tunnels failed for %s: %s", row["id"], exc)
            rec["tunnel__verdict"] = "error"

        if have_pkvf:
            try:
                rec.update(pyKVFinder_score(
                    pdb_path=pdb,
                    catalytic_resnos=cat_list,
                    chain=chain,
                ))
                if rec.get("pkvf__has_opening", 1) == 0:
                    n_pkvf_no_opening += 1
            except Exception as exc:
                LOGGER.warning("pyKVFinder_score failed for %s: %s",
                                row["id"], exc)
        rows.append(rec)

    metrics_df = pd.DataFrame(rows)
    merged = df.merge(metrics_df, on="id", how="left")

    # Hard-gate logic: drop only designs the homegrown ray-cast says are
    # truly unfixable (verdict='buried' = backbone-dominated or no
    # escape cones). pyKVFinder columns are surfaced as TOPSIS-rankable
    # metrics but NOT as hard gates — n_openings > 0 is too permissive
    # and depth_max=0 too aggressive for blanket rejection. Designs
    # with bad pkvf signals get RANKED DOWN, not killed.
    if hard_gate and _filter_active("tunnel"):
        before = len(merged)
        bad_verdict = merged["tunnel__verdict"].astype(str).isin(
            ["buried", "ligand_too_big"]
        )
        keep_mask = ~bad_verdict
        merged = merged[keep_mask].reset_index(drop=True)
        n_dropped = before - len(merged)
        LOGGER.info(
            "stage_tunnel_metrics hard_gate: dropped %d / %d "
            "(buried=%d, ligand_too_big=%d) -> %d kept "
            "(pkvf_no_opening=%d kept as TOPSIS rank-down signal)",
            n_dropped, before, n_buried, n_ligand_too_big,
            len(merged), n_pkvf_no_opening,
        )

    out_path = out_dir / "survivors_tunnel.tsv"
    merged.to_csv(out_path, sep="\t", index=False)
    LOGGER.info(
        "stage_tunnel_metrics: %d designs kept; mean ray-cast time=%.1f ms; "
        "tunnel__sidechain_blocked_fraction mean=%.2f",
        len(merged),
        float(merged.get("tunnel__elapsed_ms", pd.Series([float('nan')])).mean()) if len(merged) else 0.0,
        float(merged.get("tunnel__sidechain_blocked_fraction", pd.Series([float('nan')])).mean()) if len(merged) else 0.0,
    )
    return out_path


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

    fpocket has a buffer overflow on long input-path strings, not just
    long basenames. So we must do BOTH:
      1. copy to a short local filename (``design.pdb``), and
      2. invoke fpocket with that short RELATIVE name from ``cwd=work_dir``.

    Passing an absolute path like ``/very/long/.../design.pdb`` can still
    SIGABRT with ``*** buffer overflow detected ***`` even though the
    basename itself is short. This was reproduced directly on the shared
    cluster fpocket binary on 2026-05-06.

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
    cmd = [str(FPOCKET_BIN), "-f", local.name]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True, text=True, timeout=300, check=False,
        )
        info_txt = work_dir / "design_out" / "design_info.txt"
        pdb_out = work_dir / "design_out" / "design_out.pdb"
        pockets_subdir = work_dir / "design_out" / "pockets"
        if proc.returncode != 0 or not info_txt.is_file():
            stderr_tail = (proc.stderr or "")[-500:].strip()
            stdout_tail = (proc.stdout or "")[-200:].strip()
            LOGGER.warning(
                "fpocket failed for %s: rc=%d info_exists=%s cmd=%s "
                "cwd_len=%d input_path_len=%d stderr_tail=%r stdout_tail=%r",
                pdb_path.name, proc.returncode, info_txt.is_file(),
                " ".join(cmd), len(str(work_dir)), len(str(local)),
                stderr_tail, stdout_tail,
            )
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
         clash_severe_distance, sap_max_threshold, seed_dfi_metrics,
         sap_corrected)
    Returns: (cid, row_dict, hbond_list, struct_fail_reasons)
    """
    (cid, pdb, cat_his, fixed_, sev_dist, sap_max_thr,
     seed_dfi_metrics_, sap_corrected_) = args
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
        if sap_corrected_:
            empty_row.update({
                "sap_corr_max": float("nan"), "sap_corr_mean": float("nan"),
                "sap_corr_p95": float("nan"),
            })
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

    # SAP proxy via freesasa (+ optional corrected sap_corr_* when requested)
    sap = _compute_sap_proxy(pdb, corrected=sap_corrected_) or {}
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

    # Filter reasons (gated by --filters via _filter_active; all on => identical).
    # The cat-HIS H-bond requirement is ADDITIONALLY gated by REQUIRE_CAT_HIS_HBOND
    # (--no_require_cat_his_hbond): an enzyme with no catalytic His can never satisfy
    # it, so opt out of just this criterion. Default True => byte-identical.
    reasons: list[str] = []
    if REQUIRE_CAT_HIS_HBOND and _filter_active("cat_his_hbonds") and len(hbonds) < 1:
        reasons.append("no h-bonds to catalytic HIS")
    if _filter_active("sap") and sap_max == sap_max and sap_max > sap_max_thr:
        reasons.append(f"sap_max={sap_max:.2f} > {sap_max_thr}")
    # Severe-clash filter applied by caller (it has the boolean flag).

    row = {
        "n_hbonds_to_cat_his": len(hbonds),
        **gi_metrics,
        "sap_max": sap_max,
        "sap_mean": sap.get("sap_mean", float("nan")),
        "sap_p95": sap.get("sap_p95", float("nan")),
        **({
            "sap_corr_max": sap.get("sap_corr_max", float("nan")),
            "sap_corr_mean": sap.get("sap_corr_mean", float("nan")),
            "sap_corr_p95": sap.get("sap_corr_p95", float("nan")),
        } if sap_corrected_ else {}),
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
    # Default high enough to use the full slurm CPU allocation; the
    # actual worker count is still bounded by detect_n_cpus().
    fpocket_cap = 20
    fpocket_cap_env = os.environ.get("PROTEIN_CHISEL_FPOCKET_MAX_WORKERS", "").strip()
    if fpocket_cap_env:
        try:
            fpocket_cap = max(1, int(fpocket_cap_env))
        except ValueError:
            LOGGER.warning(
                "ignoring invalid PROTEIN_CHISEL_FPOCKET_MAX_WORKERS=%r",
                fpocket_cap_env,
            )
    n_workers = pool_workers(len(df), cpu_budget=cpus_available, cap=fpocket_cap)
    LOGGER.info(
        "stage_fpocket_rank: input n=%d (active-site constraint=%s, "
        "n_workers=%d/%d cpus, cap=%d)",
        len(df), "yes" if catalytic_resnos else "no",
        n_workers, cpus_available, fpocket_cap,
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
        # ThreadPoolExecutor (NOT multiprocessing.Pool): fpocket is an
        # external subprocess so workers are I/O-bound launchers; threads
        # are sufficient. Critically, threads avoid fork-after-CUDA-init
        # — the leading hypothesis for the catastrophic fpocket-collapse
        # mode where fpocket binaries exit -6 (SIGABRT) en masse on GPU
        # nodes after stage 2/3 has touched torch.cuda. (Codex deep-dive
        # 2026-05-06 verified that isolated stage_fpocket_rank() succeeds
        # on the same inputs that failed during the long-lived run.)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            infos = dict(pool.map(_fpocket_worker, work_args))

    # Retry any failed designs once serially. Transient subprocess flakes
    # are common (rc=-6 often goes away on a clean retry); a single
    # serial pass before declaring failure prevents a flake-induced
    # whole-batch zero-druggability collapse.
    failed_cids = [cid for cid, inf in infos.items() if inf is None]
    if failed_cids:
        LOGGER.warning(
            "stage_fpocket_rank: %d/%d designs failed in parallel pass; "
            "retrying serially", len(failed_cids), len(infos),
        )
        retry_recovered = 0
        for arg in work_args:
            cid_, pdb_, out_, cat_, chain_ = arg
            if cid_ not in failed_cids:
                continue
            _, info2 = _fpocket_worker(arg)
            if info2 is not None:
                infos[cid_] = info2
                retry_recovered += 1
        n_still_failed = len(failed_cids) - retry_recovered
        LOGGER.info(
            "stage_fpocket_rank: retry recovered %d/%d (still failed: %d)",
            retry_recovered, len(failed_cids), n_still_failed,
        )

    rows = []
    for _, row in df.iterrows():
        cid = row["id"]
        info = infos.get(cid)
        rows.append({
            **row.to_dict(),
            **_fpocket_metrics_from_info(info),
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
    _repo_root = str(Path(__file__).resolve().parents[1])
    cmd = [
        "apptainer", "exec",
        "--bind", f"{_repo_root}:/code",
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


def _hamming_at_positions(a: str, b: str, positions: list[int]) -> int:
    return sum(
        a[p] != b[p]
        for p in positions
        if p < len(a) and p < len(b)
    )


def _normalized_upper_bound_gap(value: float, upper: float) -> float:
    """Return a scale-free nonnegative gap for an upper-bound violation."""
    return max(0.0, float(value) - float(upper)) / max(1.0, abs(float(upper)))


def _normalized_lower_bound_gap(value: float, lower: float) -> float:
    """Return a scale-free nonnegative gap for a lower-bound violation."""
    return max(0.0, float(lower) - float(value)) / max(1.0, abs(float(lower)))


def _normalized_interval_gap(value: float, lower: float, upper: float) -> float:
    """Return distance outside a closed interval, normalized by interval width."""
    width = max(1e-6, float(upper) - float(lower))
    if value < lower:
        return (float(lower) - float(value)) / width
    if value > upper:
        return (float(value) - float(upper)) / width
    return 0.0


def _select_diverse_topk_progressive(
    df: pd.DataFrame,
    *,
    target_k: int,
    min_hamming_full: int,
    primary_sphere_positions: Optional[list[int]] = None,
    min_hamming_active: int = 0,
    sequence_col: str = "sequence",
    seed_sequences: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Greedy diverse top-K with progressive relaxation of Hamming gates.

    Selection respects the incoming row order, so callers should pre-sort
    ``df`` by the desired priority (e.g. pass-first, then near-miss, then
    hard tool failures). We first enforce the requested full-sequence and
    active-site Hamming thresholds; if that yields fewer than ``target_k``,
    we relax both thresholds stepwise down to zero. As a final resort, we
    fill any remaining slots in pure rank order so callers who request 50
    designs get 50 whenever at least 50 unique candidates exist.
    """
    if len(df) == 0 or target_k <= 0:
        return df.head(0).copy(), {"relaxation_steps": [], "filled_without_hamming": 0}

    selected_idx: list[int] = []
    selected_set: set[int] = set()
    selected_seqs: list[str] = list(seed_sequences or [])
    relaxation_steps: list[dict] = []
    seen_thresholds: set[tuple[int, int]] = set()

    max_delta = max(int(min_hamming_full), int(min_hamming_active))
    for delta in range(max_delta + 1):
        h_full = max(0, int(min_hamming_full) - delta)
        h_active = max(0, int(min_hamming_active) - delta)
        threshold_pair = (h_full, h_active)
        if threshold_pair in seen_thresholds:
            continue
        seen_thresholds.add(threshold_pair)

        added = 0
        for i, row in df.iterrows():
            if i in selected_set:
                continue
            seq = str(row[sequence_col])
            if not all(_hamming(seq, s) >= h_full for s in selected_seqs):
                continue
            if primary_sphere_positions and h_active > 0:
                if not all(
                    _hamming_at_positions(seq, s, primary_sphere_positions) >= h_active
                    for s in selected_seqs
                ):
                    continue
            selected_idx.append(i)
            selected_set.add(i)
            selected_seqs.append(seq)
            added += 1
            if len(selected_idx) >= target_k:
                break

        relaxation_steps.append({
            "min_hamming_full": h_full,
            "min_hamming_active": h_active,
            "added": added,
            "selected_total": len(selected_idx),
        })
        if len(selected_idx) >= target_k:
            break

    filled_without_hamming = 0
    if len(selected_idx) < target_k:
        for i, row in df.iterrows():
            if i in selected_set:
                continue
            selected_idx.append(i)
            selected_set.add(i)
            selected_seqs.append(str(row[sequence_col]))
            filled_without_hamming += 1
            if len(selected_idx) >= target_k:
                break

    telemetry = {
        "relaxation_steps": relaxation_steps,
        "filled_without_hamming": filled_without_hamming,
        "n_selected": len(selected_idx),
        "n_seed_sequences": len(seed_sequences or []),
    }
    return df.loc[selected_idx].copy(), telemetry


def _assign_unique_topk_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return a copy with collision-safe final output ids.

    LigandMPNN numbering restarts every cycle (e.g. ``..._lmpnn_004`` can
    appear in cycle 0 and again in cycle 2), so final top-K rows selected
    across multiple cycles can share the same ``id`` even when their
    sequences differ. If we export those rows verbatim, later PDB copies
    collide on filename and one design silently overwrites another.

    We preserve the original sampler id in ``source_id`` and only rewrite
    the published ``id`` when a collision exists, appending the source
    cycle plus a repeat index. Example:

        FOO_lmpnn_004  ->  FOO_lmpnn_004_c01_r1
        FOO_lmpnn_004  ->  FOO_lmpnn_004_c02_r2
    """
    if len(df) == 0 or "id" not in df.columns:
        return df.copy(), 0

    out = df.copy()
    out["source_id"] = out["id"].astype(str)
    counts = out["source_id"].value_counts()
    if int((counts > 1).sum()) == 0:
        return out, 0

    seen: dict[str, int] = {}
    new_ids: list[str] = []
    renamed = 0
    for _, row in out.iterrows():
        source_id = str(row["source_id"])
        if counts[source_id] == 1:
            new_ids.append(source_id)
            continue
        seen[source_id] = seen.get(source_id, 0) + 1
        cycle_val = row.get("cycle", None)
        suffix_parts: list[str] = []
        if cycle_val == cycle_val:
            try:
                suffix_parts.append(f"c{int(cycle_val):02d}")
            except Exception:
                suffix_parts.append(f"c{cycle_val}")
        suffix_parts.append(f"r{seen[source_id]}")
        new_ids.append(f"{source_id}_{'_'.join(suffix_parts)}")
        renamed += 1
    out["id"] = new_ids
    return out, renamed


def _fpocket_metrics_from_info(info: Optional[dict]) -> dict[str, object]:
    """Map a parsed fpocket info dict to the published fpocket__* schema."""
    def _g(key: str, default: object = float("nan")) -> object:
        return info.get(key, default) if info else default

    polar_pct = _g("proportion_of_polar_atoms", float("nan"))
    apolar_pct = (
        (100.0 - polar_pct)
        if isinstance(polar_pct, (int, float)) and polar_pct == polar_pct
        else float("nan")
    )
    return {
        "fpocket__status": "ok" if info else "failed",
        "fpocket__druggability": _g("druggability_score"),
        "fpocket__volume": _g("volume"),
        "fpocket__mean_alpha_sphere_radius": _g("mean_alpha_sphere_radius"),
        "fpocket__alpha_sphere_density": _g("alpha_sphere_density"),
        "fpocket__n_alpha_spheres_near_catalytic":
            _g("n_alpha_spheres_near_catalytic"),
        "fpocket__mean_alpha_sphere_dist_to_catalytic":
            _g("mean_alpha_sphere_dist_to_catalytic"),
        "fpocket__n_pockets_found": 1 if info else float("nan"),
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
        "fpocket__bottleneck_radius": _g("bottleneck_radius"),
        "fpocket__min_alpha_sphere_radius": _g("min_alpha_sphere_radius"),
        "fpocket__alpha_sphere_radius_p10":
            _g("alpha_sphere_radius_p10"),
        "fpocket__polar_alpha_sphere_proportion":
            _g("polar_alpha_sphere_proportion"),
        "fpocket__n_rim_spheres": _g("n_rim_spheres", 0),
    }


def _row_source_id(row: pd.Series) -> str:
    """Return the pre-export source id for a selected row."""
    source_id = row.get("source_id", row.get("id", ""))
    return str(row.get("id", "")) if pd.isna(source_id) else str(source_id)


def _candidate_source_id_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized source-id view used by final export / refill logic."""
    if len(df) == 0:
        return pd.Series(dtype=object)
    if "source_id" in df.columns:
        return df["source_id"].astype(str)
    return df["id"].astype(str)


def _select_materializable_topk(
    *,
    selected_df: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    pdb_map: dict[str, Path],
    target_k: int,
    min_hamming_full: int,
    primary_sphere_positions: Optional[list[int]],
    min_hamming_active: int,
    allow_backfill: bool,
) -> tuple[pd.DataFrame, dict]:
    """Ensure the final top-K can be materialized to real unique PDB files.

    The in-memory selector can pick rows whose source PDB later turns out to
    be missing (e.g. a partially-failed restore / publish step). Before we
    write final artifacts, drop any non-materializable rows and, when
    backfill is enabled, refill from the remaining ranked pool until we hit
    ``target_k`` or truly run out of usable candidates.
    """
    selected = selected_df.copy().reset_index(drop=True)
    source_ids_all = _candidate_source_id_series(candidate_pool)
    source_ids_selected = _candidate_source_id_series(selected)
    available_mask = source_ids_selected.map(
        lambda sid: bool((src := pdb_map.get(sid)) and src.is_file()),
    )
    dropped_missing = selected.loc[~available_mask].copy()
    if len(dropped_missing):
        LOGGER.warning(
            "stage_diverse_topk: dropped %d selected rows whose source PDBs "
            "were missing before final export",
            len(dropped_missing),
        )
    selected = selected.loc[available_mask].reset_index(drop=True)

    refill_rounds: list[dict[str, int]] = []
    while allow_backfill and len(selected) < target_k:
        used_source_ids = set(_candidate_source_id_series(selected).tolist())
        remainder = candidate_pool.loc[
            ~source_ids_all.isin(used_source_ids)
        ].copy().reset_index(drop=True)
        if len(remainder) == 0:
            break
        remainder_source_ids = _candidate_source_id_series(remainder)
        remainder = remainder.loc[
            remainder_source_ids.map(
                lambda sid: bool((src := pdb_map.get(sid)) and src.is_file()),
            )
        ].reset_index(drop=True)
        if len(remainder) == 0:
            break
        need = target_k - len(selected)
        extra, extra_telem = _select_diverse_topk_progressive(
            remainder,
            target_k=need,
            min_hamming_full=min_hamming_full,
            primary_sphere_positions=primary_sphere_positions,
            min_hamming_active=min_hamming_active,
            seed_sequences=selected["sequence"].astype(str).tolist(),
        )
        if len(extra) == 0:
            break
        refill_rounds.append({
            "requested": int(need),
            "selected": int(len(extra)),
            "filled_without_hamming": int(
                extra_telem.get("filled_without_hamming", 0),
            ),
        })
        selected = pd.concat([selected, extra], ignore_index=True)

    if len(selected) > target_k:
        selected = selected.head(target_k).copy()

    selected, n_ids_renamed = _assign_unique_topk_ids(selected)
    telemetry = {
        "dropped_missing_source_pdbs": int(len(dropped_missing)),
        "refill_rounds": refill_rounds,
        "n_ids_renamed": int(n_ids_renamed),
        "n_selected_materializable": int(len(selected)),
    }
    return selected, telemetry


def _write_final_topk_artifacts(
    *,
    top: pd.DataFrame,
    final_dir: Path,
    pdb_map: dict[str, Path],
    seed_pdb: Optional[Path] = None,
) -> tuple[Path, pd.DataFrame]:
    """Write a self-consistent top-K artifact set and return the realized rows.

    Pure writer (single responsibility): it copies PDBs + writes the TSV/FASTA for
    whatever rows it is given. The opt-in solubility veto is applied UPSTREAM by
    ``_apply_solubility_veto`` so this stays a faithful "write what you're handed"
    step (and so the caller's row-count bookkeeping reflects the post-veto set).
    """
    pdb_out = final_dir / "topk_pdbs"
    if pdb_out.exists():
        shutil.rmtree(pdb_out)
    pdb_out.mkdir(exist_ok=True)

    copied_rows: list[dict[str, object]] = []
    for _, row in top.iterrows():
        source_id = _row_source_id(row)
        src = pdb_map.get(source_id)
        if not src or not src.is_file():
            LOGGER.error(
                "stage_diverse_topk: selected row %s had no materializable "
                "source PDB at export time (source_id=%s)",
                row["id"], source_id,
            )
            continue
        try:
            shutil.copy2(src, pdb_out / f"{row['id']}.pdb")
            copied_rows.append(row.to_dict())
        except Exception as exc:
            LOGGER.error(
                "stage_diverse_topk: failed copying %s -> %s (%s)",
                src, pdb_out / f"{row['id']}.pdb", exc,
            )
    # Canonical REMARK transfer + DESIGN_PATH provenance on the shipped top-K
    # (Feature 2), so adopted designs always carry 665/666/667/668/QCB. Shared
    # protein_chisel.tools.remarks module; see stage_restore_pdbs.
    if TRANSFER_REMARKS and copied_rows and seed_pdb is not None:
        from protein_chisel.tools.remarks import transfer_remarks_to_dir
        transfer_remarks_to_dir(
            pdb_out, seed_pdb,
            transfer_input=True,
            design_path_stage="iterative_design",
        )
    materialized_top = (
        pd.DataFrame(copied_rows)
        if copied_rows else top.head(0).copy()
    )
    topk_tsv = final_dir / "topk.tsv"
    materialized_top.to_csv(topk_tsv, sep="\t", index=False)
    with open(final_dir / "topk.fasta", "w") as fh:
        for _, row in materialized_top.iterrows():
            fh.write(f">{row['id']}\n{row['sequence']}\n")
    return topk_tsv, materialized_top


def _build_input_reference_row(
    *,
    template_df: pd.DataFrame,
    seed_pdb: Path,
    wt_seq: str,
    wt_fitness: Optional[float],
    expression_result,
    seed_ss_reduced: Optional[str],
    seed_sasa: Optional[np.ndarray],
    position_classes: list[str],
    protein_resnos: list[int],
    catalytic_resnos: Iterable[int],
    fixed_resnos: Iterable[int],
    design_ph: float,
    n_term_pad: str,
    c_term_pad: str,
    net_charge_max: float,
    net_charge_min: float,
    instability_max: float,
    gravy_min: float,
    gravy_max: float,
    aliphatic_min: float,
    boman_max: float,
    pi_min: float,
    pi_max: float,
    clash_filter: bool,
    clash_severe_distance: float,
    sap_max_threshold: float,
    sap_corrected: bool = False,
    seed_dfi_metrics: Optional[dict],
    tunnel_metrics_enabled: bool,
    ligand_min_radius: Optional[float],
    ligand_resname: Optional[str],
    log_probs_esmc: np.ndarray,
    log_probs_saprot: np.ndarray,
    weights_per_position: np.ndarray,
    final_dir: Path,
    fpocket_druggability_min: float,
) -> pd.DataFrame:
    """Build a one-row metrics DataFrame for the original input reference PDB."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.filters.protparam import protparam_metrics
    from protein_chisel.sampling.fitness_score import (
        fitness_from_seed_marginals, seq_hash,
    )

    row: dict[str, object] = {col: pd.NA for col in template_df.columns}
    seed_id = seed_pdb.stem
    pp = protparam_metrics(
        wt_seq, ph=design_ph, n_term_pad=n_term_pad, c_term_pad=c_term_pad,
    )
    # --filters gating mirrors stage_seq_filter so the input-reference row's
    # pass/fail flags reflect the same selection (all on => byte-identical).
    seq_reasons: list[str] = []
    if _filter_active("net_charge") and pp.charge_at_pH_full_HH >= net_charge_max:
        seq_reasons.append(
            f"net_charge_full_HH={pp.charge_at_pH_full_HH:.2f} >= {net_charge_max}",
        )
    if _filter_active("net_charge") and pp.charge_at_pH_full_HH <= net_charge_min:
        seq_reasons.append(
            f"net_charge_full_HH={pp.charge_at_pH_full_HH:.2f} <= {net_charge_min}",
        )
    if _filter_active("pi") and not (pi_min <= pp.pi <= pi_max):
        seq_reasons.append(f"pI={pp.pi:.2f} outside [{pi_min}, {pi_max}]")
    if _filter_active("instability") and pp.instability_index >= instability_max:
        seq_reasons.append(
            f"instability_index={pp.instability_index:.1f} >= {instability_max}",
        )
    if _filter_active("gravy") and not (gravy_min <= pp.gravy <= gravy_max):
        seq_reasons.append(
            f"GRAVY={pp.gravy:+.3f} outside [{gravy_min}, {gravy_max}]",
        )
    if _filter_active("aliphatic") and pp.aliphatic_index < aliphatic_min:
        seq_reasons.append(
            f"aliphatic_index={pp.aliphatic_index:.1f} < {aliphatic_min}",
        )
    if _filter_active("boman") and pp.boman_index >= boman_max:
        seq_reasons.append(
            f"boman_index={pp.boman_index:.2f} >= {boman_max}",
        )
    if _filter_active("expression"):
        for hit in getattr(expression_result, "hard_filter_hits", []):
            seq_reasons.append(f"{hit.rule_name}: {hit.reason}")

    row.update({
        "id": seed_id,
        "source_id": seed_id,
        "sequence": wt_seq,
        "parent_design_id": "input_reference",
        "sampler": "input_reference",
        "sampler_params_hash": "",
        "is_input": True,
        "header": seed_id,
        "seq_hash": seq_hash(wt_seq),
        "n_dupes": 1,
        "length": len(wt_seq),
        "net_charge_full_HH": pp.charge_at_pH_full_HH,
        "net_charge_no_HIS": pp.charge_at_pH7_no_HIS,
        "net_charge_with_HIS_HH": pp.charge_at_pH7,
        "net_charge_HIS_half": pp.charge_at_pH7_HIS_half,
        "net_charge_DE_KR_only": pp.charge_at_pH_DE_KR_only,
        "design_ph": design_ph,
        "instability_index": pp.instability_index,
        "gravy": pp.gravy,
        "pi": pp.pi,
        "aliphatic_index": pp.aliphatic_index,
        "boman_index": pp.boman_index,
        "aromaticity": pp.aromaticity,
        "flexibility_mean_seq": (
            pp.flexibility_mean if pp.flexibility_mean is not None else float("nan")
        ),
        "helix_frac_seq": pp.helix_frac_seq,
        "turn_frac_seq": pp.turn_frac_seq,
        "sheet_frac_seq": pp.sheet_frac_seq,
        "molecular_weight": pp.molecular_weight,
        "extinction_280nm_no_disulfide": pp.extinction_280nm_no_disulfide,
        "extinction_280nm_disulfide": pp.extinction_280nm_disulfide,
        "n_expression_warnings": len(getattr(expression_result, "warnings", [])),
        "n_expression_soft_bias_hits": len(getattr(expression_result, "soft_bias_hits", [])),
        "n_expression_hard_omit_hits": len(getattr(expression_result, "hard_omit_hits", [])),
        "n_expression_hard_filter_hits": len(getattr(expression_result, "hard_filter_hits", [])),
        "expression_rule_summary": expression_result.summary(),
        "passed_seq_filter": not seq_reasons,
        "fail_reasons": "; ".join(seq_reasons),
        "cycle": -1,
        "selection__bucket": "input_reference",
        "selection__bucket_priority": -1,
        "seed_pdb": str(seed_pdb),
        "seed_basename": seed_id,
        "pdb_path": "",
    })

    _, struct_row, _, struct_reasons = _struct_filter_worker((
        seed_id, seed_pdb, list(catalytic_resnos), list(fixed_resnos),
        clash_severe_distance, sap_max_threshold, seed_dfi_metrics,
        sap_corrected,
    ))
    struct_reasons = list(struct_reasons)
    if clash_filter and _filter_active("clash") and struct_row.get("clash__has_severe"):
        struct_reasons.append(
            "severe clash "
            f"(n_cat={struct_row.get('clash__n_to_catalytic', 0)}, "
            f"n_lig={struct_row.get('clash__n_to_ligand', 0)}, "
            f"detail={struct_row.get('clash__detail', '')})",
        )
    row.update(struct_row)
    row["passed_struct_filter"] = not struct_reasons
    row["struct_fail"] = "; ".join(struct_reasons)

    if tunnel_metrics_enabled:
        try:
            from protein_chisel.tools.tunnel_metrics import (
                TunnelConfig, pyKVFinder_score, score_tunnels,
            )
            t_scores = score_tunnels(
                pdb_path=seed_pdb,
                catalytic_resnos=list(catalytic_resnos),
                chain=CHAIN,
                ligand_resname=ligand_resname,
                ligand_min_radius=ligand_min_radius,
                config=TunnelConfig(),
            )
            row.update(t_scores.to_dict())
        except Exception as exc:
            LOGGER.warning("input reference tunnel scoring failed: %s", exc)
        try:
            import pyKVFinder  # noqa: F401
            row.update(pyKVFinder_score(
                pdb_path=seed_pdb,
                catalytic_resnos=list(catalytic_resnos),
                chain=CHAIN,
            ))
        except ImportError:
            pass
        except Exception as exc:
            LOGGER.warning("input reference pyKVFinder failed: %s", exc)

    try:
        wt_res = fitness_from_seed_marginals(
            wt_seq, log_probs_esmc, log_probs_saprot, weights_per_position,
        )
        row.update({
            "fitness__logp_esmc_mean": float(wt_res.logp_esmc_mean),
            "fitness__logp_saprot_mean": float(wt_res.logp_saprot_mean),
            "fitness__logp_fused_mean": float(wt_res.logp_fused_mean),
            "fitness__method": "seed_marginals",
            "fitness__delta_vs_wt": 0.0 if wt_fitness is not None else float("nan"),
            "fitness__wt_logp_fused": (
                float(wt_fitness) if wt_fitness is not None else float("nan")
            ),
        })
    except Exception as exc:
        LOGGER.warning("input reference fitness scoring failed: %s", exc)

    fpocket_info = _run_fpocket(
        seed_pdb,
        final_dir / "_input_reference_fpocket",
        catalytic_resnos=list(catalytic_resnos),
        chain=CHAIN,
    )
    row.update(_fpocket_metrics_from_info(fpocket_info))
    if fpocket_druggability_min > 0 and _filter_active("fpocket"):
        druggability = row.get("fpocket__druggability", float("nan"))
        if isinstance(druggability, (int, float)) and druggability == druggability:
            row["selection__hard_final_filter_passed"] = (
                float(druggability) >= float(fpocket_druggability_min)
            )
            row["selection__fpocket_gap"] = max(
                0.0, float(fpocket_druggability_min) - float(druggability),
            )
        else:
            row["selection__hard_final_filter_passed"] = pd.NA
            row["selection__fpocket_gap"] = pd.NA

    out = pd.DataFrame([row])
    for col in template_df.columns:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[template_df.columns]
    return out


def _load_cycle_seq_stage_pool(cycle_dir: Path, cycle_idx: int) -> pd.DataFrame:
    """Load the union of per-cycle seq-filter survivors and rejects.

    These rows represent designs that were successfully sampled and restored to
    real PDB files, even if later structural / pocket filters eliminated them.
    They are the last safe backfill tier when the ranked structural pool is too
    small or completely empty.
    """
    frames: list[pd.DataFrame] = []
    for fname in ("survivors_seq.tsv", "rejects_seq.tsv"):
        p = cycle_dir / "02_seq_filter" / fname
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p, sep="\t")
        except Exception as exc:
            LOGGER.warning("could not read %s for seq-stage backfill: %s", p, exc)
            continue
        if len(df) == 0:
            continue
        df = df.copy()
        df["cycle"] = cycle_idx
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "id" in merged.columns and "source_id" not in merged.columns:
        merged["source_id"] = merged["id"].astype(str)
    return merged


def _build_seq_stage_backfill_pool(
    *,
    seq_stage_rows: list[pd.DataFrame],
    existing_pool: Optional[pd.DataFrame],
    log_probs_esmc: np.ndarray,
    log_probs_saprot: np.ndarray,
    weights_per_position: np.ndarray,
    fitness_cache: dict,
    rank_weights: dict[str, float],
    rank_targets: dict[str, float],
) -> tuple[pd.DataFrame, list]:
    """Prepare a ranked backfill pool from seq-filter stage rows only.

    This pool is used when later structural / pocket stages collapse. We
    keep the original discrete fail-count signal, but also prefer
    near-threshold numeric misses over rows that violate the same filters
    by a much larger margin.
    """
    if not seq_stage_rows:
        return pd.DataFrame(), []

    from protein_chisel.sampling.fitness_score import (
        deduplicate_by_sequence, score_dataframe_fitness,
    )
    from protein_chisel.scoring.multi_objective import (
        DEFAULT_METRIC_SPECS, apply_cli_overrides, compute_topsis_scores_v2,
        select_specs_by_label,
    )

    pool = pd.concat(seq_stage_rows, ignore_index=True)
    if len(pool) == 0:
        return pd.DataFrame(), []
    pool = deduplicate_by_sequence(pool)

    if existing_pool is not None and len(existing_pool) > 0:
        if "seq_hash" in pool.columns and "seq_hash" in existing_pool.columns:
            existing_hashes = set(existing_pool["seq_hash"].astype(str).tolist())
            pool = pool.loc[~pool["seq_hash"].astype(str).isin(existing_hashes)].copy()
        elif "sequence" in pool.columns and "sequence" in existing_pool.columns:
            existing_sequences = set(existing_pool["sequence"].astype(str).tolist())
            pool = pool.loc[
                ~pool["sequence"].astype(str).isin(existing_sequences)
            ].copy()
    if len(pool) == 0:
        return pd.DataFrame(), []

    pool = score_dataframe_fitness(
        pool,
        log_probs_esmc,
        log_probs_saprot,
        weights_per_position,
        fitness_cache=fitness_cache,
    )
    if "source_id" not in pool.columns and "id" in pool.columns:
        pool["source_id"] = pool["id"].astype(str)

    fail_reason_count = (
        pool.get("fail_reasons", pd.Series([""] * len(pool), index=pool.index))
        .fillna("")
        .astype(str)
        .map(lambda s: 0 if not s.strip() else len([x for x in s.split(";") if x.strip()]))
    )
    pool["selection__seq_backfill_reason_count"] = fail_reason_count.astype(int)
    seq_numeric_gap = pd.to_numeric(
        pool.get(
            "selection__seq_filter_numeric_gap",
            pd.Series([0.0] * len(pool), index=pool.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    pool["selection__seq_backfill_numeric_gap"] = seq_numeric_gap.astype(float)
    pool["selection__bucket"] = np.where(
        pool.get("passed_seq_filter", pd.Series([False] * len(pool), index=pool.index))
            .fillna(False)
            .astype(bool),
        "seq_stage_passed_seq_filter",
        "seq_stage_failed_seq_filter",
    )
    pool["selection__bucket_priority"] = np.where(
        pool["selection__bucket"].eq("seq_stage_passed_seq_filter"),
        3, 4,
    ).astype(int)
    pool["selection__hard_final_filter_passed"] = False
    pool["selection__fpocket_gap"] = np.inf
    pool["fpocket__status"] = "not_run"
    if "passed_struct_filter" not in pool.columns:
        pool["passed_struct_filter"] = pd.NA
    if "struct_fail" not in pool.columns:
        pool["struct_fail"] = "not_evaluated_for_struct_filter"

    active_specs = apply_cli_overrides(
        DEFAULT_METRIC_SPECS, rank_weights, rank_targets,
    )
    active_specs = select_specs_by_label(active_specs, _RANKING_LABEL_FILTER)
    scores, used_specs, _debug = compute_topsis_scores_v2(pool, active_specs)
    pool["mo_topsis"] = scores
    pool["legacy_rank_score"] = pool["fitness__logp_fused_mean"].rank(
        ascending=False,
    )
    pool = pool.sort_values(
        [
            "selection__bucket_priority",
            "selection__seq_backfill_reason_count",
            "selection__seq_backfill_numeric_gap",
            "mo_topsis",
            "fitness__logp_fused_mean",
        ],
        ascending=[True, True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    if len(pool) > 0:
        LOGGER.info(
            "seq-stage backfill ranking: passed_seq=%d failed_seq=%d "
            "numeric_gap[min/median/max]=%.3f/%.3f/%.3f",
            int(pool["selection__bucket"].eq("seq_stage_passed_seq_filter").sum()),
            int(pool["selection__bucket"].eq("seq_stage_failed_seq_filter").sum()),
            float(pool["selection__seq_backfill_numeric_gap"].min()),
            float(pool["selection__seq_backfill_numeric_gap"].median()),
            float(pool["selection__seq_backfill_numeric_gap"].max()),
        )
    return pool, used_specs


def _deferred_rescue_shortlist_size(*, deficit: int, target_k: int, cap: int = 200) -> int:
    """Size the shortlist that gets deferred downstream rescue scoring.

    We only pay this extra cost when seq-stage backfill is actually needed.
    The shortlist is intentionally larger than the exact deficit so the rescue
    pass has enough candidates to improve ranking quality, but it is capped to
    keep worst-case runtime bounded on large production sweeps.
    """
    need = max(1, int(deficit))
    return min(int(cap), max(int(target_k), need * 4, need + 10))


def _overlay_rows_by_id(base_df: pd.DataFrame, updates_df: pd.DataFrame) -> pd.DataFrame:
    """Overlay per-design updates onto ``base_df`` while preserving row order."""
    if len(base_df) == 0 or len(updates_df) == 0:
        return base_df.copy()
    if "id" not in base_df.columns or "id" not in updates_df.columns:
        return base_df.copy()

    out = base_df.copy()
    upd = updates_df.drop_duplicates(subset=["id"], keep="last").copy()
    for col in upd.columns:
        if col not in out.columns:
            out[col] = pd.NA
    out_idx = out.set_index("id", drop=False)
    upd_idx = upd.set_index("id", drop=False)
    common = out_idx.index.intersection(upd_idx.index)
    for col in upd.columns:
        out_idx.loc[common, col] = upd_idx.loc[common, col]
    return out_idx.reset_index(drop=True)


def _within_solubility_band(
    df: pd.DataFrame,
    *,
    gravy_min: float,
    gravy_max: float,
    net_charge_min: float,
    net_charge_max: float,
) -> pd.Series:
    """Boolean mask: each row is inside the GRAVY + net-charge solubility band.

    Thin wrapper over the single source of truth ``scoring.solubility``
    (also used by WS-F's PLM-refresh representative selection) — kept here under its
    historical name so the WS-A veto call sites are unchanged and byte-identical.
    Charge EXCLUSIVE, GRAVY INCLUSIVE, missing data fails closed.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.scoring.solubility import within_solubility_band
    return within_solubility_band(
        df, gravy_min=gravy_min, gravy_max=gravy_max,
        net_charge_min=net_charge_min, net_charge_max=net_charge_max,
    )


def _apply_solubility_veto(
    top: pd.DataFrame,
    *,
    enabled: bool,
    gravy_min: Optional[float],
    gravy_max: Optional[float],
    net_charge_min: Optional[float],
    net_charge_max: Optional[float],
) -> pd.DataFrame:
    """Opt-in hard solubility veto on a final-selection frame (modular, pure).

    With ``enabled=False`` (default) returns ``top`` UNCHANGED — no copy, no new
    column → byte-identical. With ``enabled=True`` it drops every row outside the
    GRAVY + net-charge band (so the deferred-rescue / backfill path can never ship
    a seq-filter-failing design, e.g. GRAVY=1.05), adds a truthful
    ``selection__solubility_passed`` column, and logs the dropped offenders.

    The band must match the final cycle's ACTUAL ``stage_seq_filter`` bounds — the
    caller passes strategy-correct values (GRAVY is ``args.gravy_*`` under
    'constant', ``cyc.gravy_*`` under 'annealing'; charge is the final
    ``CycleConfig`` band). May return fewer than ``target_k`` rows (intended:
    ship soluble-only). Applied in the caller BEFORE the row-count bookkeeping so
    a legitimate veto drop is never mistaken for a PDB-export failure.

    Distinct from ``selection__hard_final_filter_passed`` (= passed fpocket
    druggability), which is NOT a solubility signal.
    """
    if not enabled or len(top) == 0:
        return top
    missing = [n for n, v in (("gravy_min", gravy_min), ("gravy_max", gravy_max),
                              ("net_charge_min", net_charge_min),
                              ("net_charge_max", net_charge_max)) if v is None]
    if missing:
        raise ValueError(
            "_apply_solubility_veto: enabled but missing band bound(s): "
            f"{', '.join(missing)} (caller must pass all four)")
    band_ok = _within_solubility_band(
        top, gravy_min=gravy_min, gravy_max=gravy_max,
        net_charge_min=net_charge_min, net_charge_max=net_charge_max,
    )
    out = top.copy()
    out["selection__solubility_passed"] = band_ok.values
    n_veto = int((~band_ok).sum())
    if n_veto:
        cols = [c for c in ("id", "gravy", "net_charge_full_HH",
                            "selection__bucket") if c in out.columns]
        worst = out.loc[~band_ok, cols].head(5).to_dict("records")
        LOGGER.error(
            "ship_solubility_veto: dropping %d/%d final top-K rows OUTSIDE the "
            "solubility band (GRAVY[%.2f, %.2f], charge(%.1f, %.1f)); shipping %d. "
            "Worst offenders: %s",
            n_veto, len(out), gravy_min, gravy_max,
            net_charge_min, net_charge_max, len(out) - n_veto, worst,
        )
    return out.loc[band_ok].reset_index(drop=True)


def _deferred_rescue_score_candidates(
    *,
    candidates_df: pd.DataFrame,
    pdb_map: dict[str, Path],
    out_dir: Path,
    catalytic_his_resnos: Iterable[int],
    catalytic_resnos: Iterable[int],
    fixed_resnos: Iterable[int],
    sap_max_threshold: float,
    clash_filter: bool,
    clash_severe_distance: float,
    seed_dfi_metrics: Optional[dict],
    tunnel_metrics_enabled: bool,
    ligand_min_radius: Optional[float],
    ligand_resname: Optional[str],
    log_probs_esmc: np.ndarray,
    log_probs_saprot: np.ndarray,
    weights_per_position: np.ndarray,
    fitness_cache: dict,
    wt_fitness: Optional[float],
    rank_weights: dict[str, float],
    rank_targets: dict[str, float],
    fpocket_druggability_min: float,
    chain: str = CHAIN,
) -> pd.DataFrame:
    """Run a deferred downstream rescue pass on fallback candidates.

    This path is used when seq-stage backfill rows are under serious
    consideration for the final output. We try to give those rows the same
    structural / tunnel / fpocket annotations as normal pipeline survivors so:

      1. final selection can prefer rows that truly pass more downstream gates
      2. shipped TSVs are internally consistent and rarely carry ``not_run``

    The rescue pass is crash-tolerant by construction:
      - struct filter emits explicit fail rows instead of crashing
      - tunnel scoring catches per-design exceptions and writes partial metrics
      - fpocket failures become ``fpocket__status=failed`` + NaNs
    """
    if len(candidates_df) == 0:
        return candidates_df.head(0).copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    rescue_df = candidates_df.copy().reset_index(drop=True)
    if "source_id" not in rescue_df.columns and "id" in rescue_df.columns:
        rescue_df["source_id"] = rescue_df["id"].astype(str)
    rescue_df["selection__origin_bucket"] = rescue_df.get(
        "selection__bucket", pd.Series([pd.NA] * len(rescue_df), index=rescue_df.index),
    )
    rescue_df["selection__deferred_rescue_requested"] = True
    # Drop any previously-computed downstream metrics before replaying the
    # rescue stages. The seq-stage backfill pool already carries fitness
    # columns; keeping them would duplicate headers after the rescue
    # stage_fitness_score() pass and can break downstream pandas writes.
    downstream_prefixes = (
        "fitness__", "fpocket__", "tunnel__", "pkvf__", "clash__",
        "ligand_int__", "preorg__", "dfi__",
    )
    downstream_exact = {
        "n_hbonds_to_cat_his",
        "sap_max", "sap_mean", "sap_p95",
        "passed_struct_filter", "struct_fail",
        "legacy_rank_score", "mo_topsis",
        "selection__hard_final_filter_passed",
        "selection__fpocket_gap",
        "selection__deferred_rescue_attempted",
        "selection__deferred_rescue_struct_failed",
        "selection__deferred_rescue_tunnel_failed",
    }
    keep_cols = [
        c for c in rescue_df.columns
        if c not in downstream_exact
        and not any(c.startswith(prefix) for prefix in downstream_prefixes)
    ]
    rescue_df = rescue_df[keep_cols].copy()

    source_ids = _candidate_source_id_series(rescue_df)
    materializable_mask = source_ids.map(
        lambda sid: bool((src := pdb_map.get(sid)) and src.is_file()),
    )
    if int((~materializable_mask).sum()) > 0:
        LOGGER.warning(
            "deferred_rescue: dropping %d/%d candidates with missing source PDBs",
            int((~materializable_mask).sum()), len(rescue_df),
        )
    rescue_df = rescue_df.loc[materializable_mask].reset_index(drop=True)
    if len(rescue_df) == 0:
        return rescue_df

    rescue_pdb_map = {
        str(row["id"]): pdb_map.get(str(row.get("source_id", row["id"])))
        for _, row in rescue_df.iterrows()
    }
    input_tsv = out_dir / "rescue_candidates.tsv"
    rescue_df.to_csv(input_tsv, sep="\t", index=False)

    struct_dir = out_dir / "03_struct_filter"
    survivors_struct_tsv = stage_struct_filter(
        survivors_seq_tsv=input_tsv,
        pdb_map=rescue_pdb_map,
        out_dir=struct_dir,
        sap_max_threshold=sap_max_threshold,
        catalytic_his_resnos=catalytic_his_resnos,
        fixed_resnos=fixed_resnos,
        clash_filter=clash_filter,
        clash_severe_distance=clash_severe_distance,
        seed_dfi_metrics=seed_dfi_metrics,
    )

    struct_frames: list[pd.DataFrame] = []
    for fname in ("survivors_struct.tsv", "rejects_struct.tsv"):
        p = struct_dir / fname
        if p.is_file():
            try:
                struct_frames.append(pd.read_csv(p, sep="\t"))
            except Exception as exc:
                LOGGER.warning("deferred_rescue: could not read %s: %s", p, exc)
    if struct_frames:
        struct_all = pd.concat(struct_frames, ignore_index=True)
        struct_all = struct_all.drop_duplicates(subset=["id"], keep="first")
    else:
        LOGGER.warning(
            "deferred_rescue: struct filter produced no readable outputs; "
            "falling back to unannotated candidate rows",
        )
        struct_all = rescue_df.copy()
    struct_all_tsv = struct_dir / "all_struct_scored.tsv"
    struct_all.to_csv(struct_all_tsv, sep="\t", index=False)

    current_tsv = struct_all_tsv
    if tunnel_metrics_enabled:
        tunnel_dir = out_dir / "035_tunnel"
        current_tsv = stage_tunnel_metrics(
            survivors_struct_tsv=current_tsv,
            pdb_map=rescue_pdb_map,
            out_dir=tunnel_dir,
            catalytic_resnos=catalytic_resnos,
            ligand_min_radius=ligand_min_radius,
            ligand_resname=ligand_resname,
            chain=chain,
            hard_gate=False,
        )

    fitness_dir = out_dir / "04_fitness"
    scored_tsv = stage_fitness_score(
        survivors_struct_tsv=current_tsv,
        out_dir=fitness_dir,
        log_probs_esmc=log_probs_esmc,
        log_probs_saprot=log_probs_saprot,
        weights_per_position=weights_per_position,
        fitness_cache=fitness_cache,
        wt_fitness=wt_fitness,
    )

    fpocket_dir = out_dir / "05_fpocket"
    ranked_tsv = stage_fpocket_rank(
        scored_tsv=scored_tsv,
        pdb_map=rescue_pdb_map,
        out_dir=fpocket_dir,
        catalytic_resnos=catalytic_resnos,
        chain=chain,
    )
    rescued = pd.read_csv(ranked_tsv, sep="\t")
    if "source_id" not in rescued.columns and "id" in rescued.columns:
        rescued["source_id"] = rescued["id"].astype(str)

    from protein_chisel.scoring.multi_objective import (
        DEFAULT_METRIC_SPECS, apply_cli_overrides, compute_topsis_scores_v2,
        select_specs_by_label,
    )
    active_specs = apply_cli_overrides(
        DEFAULT_METRIC_SPECS, rank_weights, rank_targets,
    )
    active_specs = select_specs_by_label(active_specs, _RANKING_LABEL_FILTER)
    rescue_scores, _used_specs, _debug = compute_topsis_scores_v2(
        rescued, active_specs,
    )
    rescued["mo_topsis"] = rescue_scores
    rescued["selection__deferred_rescue_attempted"] = True

    struct_failed = ~rescued.get(
        "passed_struct_filter",
        pd.Series([False] * len(rescued), index=rescued.index),
    ).fillna(False).astype(bool)
    # tunnel/fpocket gating also applies to the rescue bucketing, so a deselected
    # filter can't push designs down via bucket priority. At the defaults
    # (_filter_active True) these are exactly the original branches.
    if tunnel_metrics_enabled and _filter_active("tunnel"):
        tunnel_failed = rescued.get(
            "tunnel__verdict",
            pd.Series(["not_run"] * len(rescued), index=rescued.index),
        ).fillna("not_run").astype(str).isin(["buried", "ligand_too_big", "error", "missing_pdb"])
    else:
        tunnel_failed = pd.Series([False] * len(rescued), index=rescued.index)

    fpocket_status = rescued.get(
        "fpocket__status",
        pd.Series(["failed"] * len(rescued), index=rescued.index),
    ).fillna("failed").astype(str)
    fpocket_druggability = pd.to_numeric(
        rescued.get(
            "fpocket__druggability",
            pd.Series([float("nan")] * len(rescued), index=rescued.index),
        ),
        errors="coerce",
    )
    if fpocket_druggability_min > 0 and _filter_active("fpocket"):
        fpocket_pass = fpocket_status.eq("ok") & (fpocket_druggability >= fpocket_druggability_min)
        fpocket_near = fpocket_status.eq("ok") & fpocket_druggability.notna() & ~fpocket_pass
        rescued["selection__fpocket_gap"] = np.where(
            fpocket_druggability.notna(),
            np.maximum(0.0, float(fpocket_druggability_min) - fpocket_druggability),
            np.inf,
        )
    else:
        # fpocket filter deselected (or no cutoff): treat any computed pocket as
        # passing and apply no druggability gap in bucketing.
        fpocket_pass = fpocket_status.eq("ok")
        fpocket_near = pd.Series([False] * len(rescued), index=rescued.index)
        rescued["selection__fpocket_gap"] = 0.0
    fpocket_failed = ~fpocket_pass & ~fpocket_near

    rescued["selection__hard_final_filter_passed"] = fpocket_pass
    rescued["selection__deferred_rescue_struct_failed"] = struct_failed
    rescued["selection__deferred_rescue_tunnel_failed"] = tunnel_failed
    rescued["selection__bucket"] = np.select(
        [
            fpocket_pass & ~tunnel_failed & ~struct_failed,
            fpocket_near & ~tunnel_failed & ~struct_failed,
            fpocket_failed & ~tunnel_failed & ~struct_failed,
            tunnel_failed & ~struct_failed,
            struct_failed,
        ],
        [
            "rescued_final_filters",
            "rescued_fpocket_near_miss",
            "rescued_fpocket_failed",
            "rescued_tunnel_hard_gate",
            "rescued_struct_failed",
        ],
        default="rescued_unclassified",
    )
    rescued["selection__bucket_priority"] = np.select(
        [
            rescued["selection__bucket"].eq("rescued_final_filters"),
            rescued["selection__bucket"].eq("rescued_fpocket_near_miss"),
            rescued["selection__bucket"].eq("rescued_fpocket_failed"),
            rescued["selection__bucket"].eq("rescued_tunnel_hard_gate"),
            rescued["selection__bucket"].eq("rescued_struct_failed"),
        ],
        [2, 3, 4, 5, 6],
        default=7,
    ).astype(int)
    rescued = rescued.sort_values(
        [
            "selection__bucket_priority",
            "selection__seq_backfill_reason_count",
            "selection__seq_backfill_numeric_gap",
            "selection__fpocket_gap",
            "mo_topsis",
            "fitness__logp_fused_mean",
        ],
        ascending=[True, True, True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    LOGGER.info(
        "deferred_rescue: scored %d fallback candidates -> buckets=%s",
        len(rescued),
        rescued["selection__bucket"].value_counts().to_dict(),
    )
    return rescued


def _write_empty_final_artifacts(
    *,
    final_dir: Path,
    run_dir: Path,
    template_df: pd.DataFrame,
    status: str,
    reason: str,
    extra_meta: Optional[dict] = None,
) -> None:
    """Write a schema-stable empty final_topk tree for downstream stages."""
    final_dir.mkdir(parents=True, exist_ok=True)
    empty = template_df.head(0).copy()
    empty.to_csv(final_dir / "all_survivors.tsv", sep="\t", index=False)
    empty.to_csv(final_dir / "topk.tsv", sep="\t", index=False)
    (final_dir / "topk.fasta").write_text("")
    (final_dir / "topk_pdbs").mkdir(exist_ok=True)

    manifest_stub = run_dir / "manifest.json"
    payload = {
        "status": status,
        "reason": reason,
        "outputs": {
            "final_topk_fasta": str(final_dir / "topk.fasta"),
            "final_topk_pdbs": str(final_dir / "topk_pdbs"),
            "final_topk_tsv": str(final_dir / "topk.tsv"),
            "all_survivors": str(final_dir / "all_survivors.tsv"),
        },
    }
    if extra_meta:
        payload.update(extra_meta)
    with open(manifest_stub, "w") as fh:
        json.dump(payload, fh, indent=2)


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
    # production sbatch (run_chisel_design.sh) handles this
    # by running protonate_final as a separate post-stage outside any
    # container. Detect the in-container case and skip with a clear
    # log line so the user knows where the missing protonation is.
    if in_apptainer():
        LOGGER.info(
            "stage_protonate_final_topk: detected we are running INSIDE "
            "a container; skipping in-driver invocation. "
            "run_chisel_design.sh invokes the protonation as "
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
    aa_reference: str = "swissprot_ec3_hydrolases_2026_01",
    balance_z_threshold: float = 2.0,
    design_ph: float = 7.5,
    instability_max: float = 60.0,
    gravy_min: float = -0.8,
    gravy_max: float = 0.3,
    aliphatic_min: float = 40.0,
    boman_max: float = 4.5,
    sap_corrected: bool = False,
    # ---- WS-C composition control (all opt-in; defaults → byte-identical) ----
    composition_suppress_all_overrep: bool = False,
    aa_fraction_cap: Optional[float] = None,
    composition_soft_bias: bool = False,
    composition_soft_bias_nats: float = 0.5,
    expression_soft_bias: Optional[dict[int, str]] = None,
    # Feature #29: opt-in composition POOL FALLBACK. When the flag is set AND
    # this cycle's survivor pool is empty, derive the cap/class-balance/soft-bias
    # from the previous cycle's full SAMPLED (pre-band-filter) pool carried in
    # ``composition_fallback_pool``. Default OFF / None pool => byte-identical.
    composition_pool_fallback: bool = False,
    composition_fallback_pool: Optional[pd.DataFrame] = None,
    bias_total_clamp: Optional[float] = None,
    bias_total_clamp_odds: Optional[float] = None,
    n_term_pad: str = "",
    c_term_pad: str = "",
    omit_M_at_pos1: bool = True,
    tunnel_metrics_enabled: bool = False,
    tunnel_hard_gate: bool = True,
    ligand_min_radius: Optional[float] = None,
    ligand_resname: Optional[str] = None,
    throat_bias_prev: Optional[np.ndarray] = None,
    throat_bias_decay: float = 0.5,
    adaptive_bias_global: Optional[dict] = None,
    adaptive_bias_delta: Optional[np.ndarray] = None,
    controller_coordinator: bool = False,
    controller_ceiling: float = 8.0,
    clash_bias: Optional[np.ndarray] = None,
) -> tuple[Optional[pd.DataFrame], dict[str, Path], dict]:
    """Run ONE iteration cycle. Returns (ranked DataFrame, pdb_map, cycle_telemetry).

    ``cycle_telemetry`` carries forward-looking signals to the next cycle:
        - ``throat_bias_delta``: (L, 20) delta to apply (decayed) at next cycle
        - ``blocker_stats``: aggregated per-position blocker stats from this cycle
    """
    if catalytic_his_resnos is None:
        catalytic_his_resnos = CATALYTIC_HIS_RESNOS
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.sampling.iterative_fusion import (
        IterationBiasConfig, build_iteration_bias,
    )

    cycle_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0. Build cycle-k bias --------------------------------------
    LOGGER.info("")
    LOGGER.info("================================================================")
    LOGGER.info("===  CYCLE %d  ==================================================", cycle_cfg.cycle_idx)
    LOGGER.info("================================================================")
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
    # ---- 0a-bis. Throat-blocker bias (cycle-to-cycle reinforcement) ---
    # Apply (decayed) bias from the previous cycle's throat-blocker
    # observations so MPNN actively avoids placing bulky AAs at known-
    # constricting positions. cumulative across cycles with exponential
    # decay so positions that successfully open up release pressure.
    throat_bias_applied: Optional[np.ndarray] = None
    if throat_bias_prev is not None and throat_bias_prev.shape == bias_k.shape:
        throat_bias_applied = (throat_bias_prev * throat_bias_decay).astype(bias_k.dtype)
        bias_k = bias_k + throat_bias_applied
        n_pos_touched = int((np.abs(throat_bias_applied) > 1e-6).any(axis=1).sum())
        n_aa_touched = int((np.abs(throat_bias_applied) > 1e-6).sum())
        max_penalty = float(throat_bias_applied.min()) if throat_bias_applied.size else 0.0
        LOGGER.info(
            "cycle %d: applied THROAT bias (decayed by %.2f) at %d positions, "
            "%d (pos,AA) cells, max penalty %.2f nats",
            cycle_cfg.cycle_idx, throat_bias_decay,
            n_pos_touched, n_aa_touched, max_penalty,
        )
        telem["throat_bias_n_positions"] = n_pos_touched
        telem["throat_bias_max_penalty"] = max_penalty
    else:
        telem["throat_bias_n_positions"] = 0
        telem["throat_bias_max_penalty"] = 0.0

    # ---- 0a-ter. Adaptive-bias per-position surface delta (opt-in) ----
    # Carried (already gain-controlled / held) from the previous cycle's adaptive
    # controller. Added in the SAME additive slot as the throat delta. None unless
    # --adaptive_bias is set, so the default path is byte-identical.
    if (adaptive_bias_delta is not None
            and adaptive_bias_delta.shape == bias_k.shape):
        bias_k = bias_k + adaptive_bias_delta.astype(bias_k.dtype)
        n_ab_pos = int((np.abs(adaptive_bias_delta) > 1e-6).any(axis=1).sum())
        telem["adaptive_bias_n_positions"] = n_ab_pos
        telem["adaptive_bias_max_penalty"] = float(adaptive_bias_delta.min())
        LOGGER.info(
            "cycle %d: applied ADAPTIVE surface bias at %d positions (max %.2f nats)",
            cycle_cfg.cycle_idx, n_ab_pos, float(adaptive_bias_delta.min()),
        )

    # ---- Feature #29: resolve the composition POOL FALLBACK source -----------
    # When --composition_pool_fallback is set AND this cycle's survivor pool is
    # empty, the WS-C cap / class-balance / soft-bias derive their composition
    # signal from the previous cycle's full SAMPLED (pre-band-filter) pool instead
    # of nothing. Returns None (no fallback) when the flag is off, survivors exist,
    # or there is no previous pool (cycle 0) => the gated blocks below run unchanged
    # on survivors => byte-identical. The fallback only carries the sampled pool
    # forward; nothing about it touches the default path.
    comp_fallback_pool = _resolve_composition_pool(
        survivors_prev=survivors_prev,
        fallback_pool=composition_fallback_pool,
        composition_pool_fallback=composition_pool_fallback,
    )
    if comp_fallback_pool is not None:
        LOGGER.info(
            "cycle %d: composition_pool_fallback ACTIVE — survivor pool empty, "
            "deriving cap/class-balance/soft-bias from the previous cycle's full "
            "SAMPLED pool (n=%d, pre-band-filter)",
            cycle_cfg.cycle_idx, len(comp_fallback_pool),
        )

    # ---- 0a-quater. Composition soft-bias per-residue delta (opt-in, WS-C) ----
    # Activate the expression engine's per-residue SOFT_BIAS tier (long-hydrophobic-
    # stretch, KR-near-catalytic-on-helix, polyproline, repetitive-segment, …) as an
    # additive bias_k delta in the SAME slot as the throat/adaptive deltas. The map
    # is POOL-DERIVED per cycle — the liabilities the *designs* introduce as the pool
    # drifts (a seed-only map is blind to them, since polyproline/repeat/hydrophobic-
    # stretch are sequence-determined). The seed map bootstraps cycle 0 (no survivors
    # yet); with --composition_pool_fallback the previous cycle's SAMPLED pool stands
    # in for an empty survivor pool. Whole-protein composition hits are excluded (span
    # filter) — those are handled globally by the suppress-all / fraction-cap levers.
    # Off unless --composition_soft_bias, so the default path is byte-identical.
    # Defensively wrapped (mirrors the adaptive controller, commit 42f2f6b): the
    # pool path adds N per-survivor engine evaluations, and a soft-bias failure must
    # NEVER abort the design run — degrade to the unbiased path and continue.
    if composition_soft_bias:
        try:
            from protein_chisel.expression.engine import (
                aggregate_pool_soft_bias, soft_bias_to_bias_array,
            )
            from protein_chisel.sampling.plm_fusion import AA_ORDER
            if survivors_prev is not None and len(survivors_prev) > 0:
                _soft_pool_seqs = survivors_prev["sequence"].astype(str).tolist()
                soft_src = "pool"
            elif comp_fallback_pool is not None:
                _soft_pool_seqs = comp_fallback_pool["sequence"].astype(str).tolist()
                soft_src = "sampled_fallback"
            else:
                _soft_pool_seqs = None
            if _soft_pool_seqs is not None:
                soft_map = aggregate_pool_soft_bias(
                    expression_engine,
                    _soft_pool_seqs,
                    ss_reduced=seed_ss_reduced, sasa=seed_sasa,
                    position_class=seed_position_class,
                    catalytic_resnos=fixed_resnos, fixed_resnos=fixed_resnos,
                    protein_resnos=seed_protein_resnos,
                    min_support=_SOFT_BIAS_MIN_SUPPORT,
                    max_span_frac=_SOFT_BIAS_MAX_SPAN_FRAC,
                )
            else:
                soft_map = expression_soft_bias or {}    # seed bootstrap (cycle 0)
                soft_src = "seed"
            if soft_map:
                soft_delta = soft_bias_to_bias_array(
                    soft_map, bias_k.shape[0],
                    magnitude=composition_soft_bias_nats, aa_order=AA_ORDER,
                )
                bias_k = bias_k + soft_delta.astype(bias_k.dtype)
                n_sb_pos = int((np.abs(soft_delta) > 1e-6).any(axis=1).sum())
                n_sb_cells = int((np.abs(soft_delta) > 1e-6).sum())
                telem["composition_soft_bias_n_positions"] = n_sb_pos
                telem["composition_soft_bias_n_cells"] = n_sb_cells
                telem["composition_soft_bias_source"] = soft_src
                LOGGER.info(
                    "cycle %d: applied COMPOSITION soft-bias (%s-derived) at %d "
                    "positions, %d (pos,AA) cells (%.2f nats each)",
                    cycle_cfg.cycle_idx, soft_src, n_sb_pos, n_sb_cells,
                    composition_soft_bias_nats,
                )
        except Exception:
            LOGGER.exception(
                "cycle %d: composition soft-bias failed; continuing WITHOUT it "
                "(degrade to unbiased — a soft-bias error never aborts the run)",
                cycle_cfg.cycle_idx,
            )

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
    #
    # Feature #29: when the survivor pool is empty and --composition_pool_fallback
    # is set, derive both the per-AA fraction cap AND the class-balance bias from
    # the previous cycle's full SAMPLED pool (``comp_fallback_pool``) instead — the
    # cap is exactly what bounds the 26%-Ala backfill mode that an empty survivor
    # pool would otherwise let through. Default (flag off / survivors exist / no
    # previous pool) keeps the survivor-only path => byte-identical.
    bias_AA_str = ""
    fraction_cap_omit: dict[str, str] = {}
    _have_survivors = survivors_prev is not None and len(survivors_prev) > 0
    _comp_pool_df = survivors_prev if _have_survivors else comp_fallback_pool
    if _comp_pool_df is not None and len(_comp_pool_df) > 0:
        from protein_chisel.expression.aa_class_balance import (
            compute_class_balanced_bias_AA,
        )
        # Pool the chosen source into one mega-sequence: this gives a count-
        # weighted average composition (each member contributes equally
        # since they're the same length L).
        pool_seq = "".join(_comp_pool_df["sequence"].astype(str).tolist())
        # exclude_aas matches cycle_cfg.omit_AA (default "X" or "CX") so
        # we don't try to up-weight an AA the sampler can't pick anyway.
        excl = _canonical_omit_aas(cycle_cfg.omit_AA)
        # ---- WS-C per-AA fraction cap (opt-in) -------------------------
        # Any AA whose fraction in the survivor pool is at/over the cap is
        # hard-omitted at every NON-FIXED designable position next cycle,
        # bounding runaway single-AA over-representation (the 26%-Ala mode).
        # Recomputed each cycle from that cycle's survivors, so an AA is
        # re-allowed once it falls back under the cap. None → no-op (empty dict).
        fraction_cap_omit = _build_fraction_cap_omit(
            pool_seq, aa_fraction_cap,
            protein_resnos=protein_resnos, fixed_resnos=fixed_resnos,
            chain=CHAIN, exclude_aas=excl,
        )
        if fraction_cap_omit:
            LOGGER.info(
                "cycle %d: aa_fraction_cap=%.3f -> omit %s at %d non-fixed "
                "designable positions (pool fractions over cap)",
                cycle_cfg.cycle_idx, aa_fraction_cap,
                next(iter(fraction_cap_omit.values())), len(fraction_cap_omit),
            )
        balance_telem = compute_class_balanced_bias_AA(
            pool_seq,
            reference=aa_reference,
            exclude_aas=excl,
            suppress_all_overrep=composition_suppress_all_overrep,
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
        # Record which pool the cap/class-balance was derived from so the
        # fallback path is visible in telemetry (survivors vs the Feature #29
        # sampled fallback). Default path => "survivors".
        telem["composition_cap_pool_source"] = (
            "survivors" if _have_survivors else "sampled_fallback"
        )
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

    # ---- 0c. Adaptive-bias global per-AA term (opt-in) -------------
    # Carried (raw, unmerged) from the previous cycle's adaptive controller; merge
    # it with THIS cycle's fresh class-balance bias (class-balance wins conflicts).
    # None unless --adaptive_bias, so the default path is byte-identical.
    # CF-3: capture the CLASS-BALANCE-ONLY bias_AA *before* the controller merge so the
    # nested-ceiling split below can recover the controller's SURVIVING global
    # contribution (merged − class_balance), respecting "class-balance wins".
    _class_balance_bias_AA = bias_AA_str
    if adaptive_bias_global:
        from protein_chisel.sampling.adaptive_bias import merge_bias_AA_strings
        bias_AA_str, _ab_conflicts = merge_bias_AA_strings(bias_AA_str, adaptive_bias_global)
        LOGGER.info("cycle %d: merged ADAPTIVE global bias_AA -> %s",
                    cycle_cfg.cycle_idx, bias_AA_str or "(empty)")

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
        "cycle %d: diversity-omit at primary+secondary shell = %s",
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
    # ---- WS-C fraction cap: layered LAST, then guarded -----------------
    # Union the cap omit on top of the structural omits (expression/diversity/
    # pos-1-M), then run the post-merge sampleable guard so no position is left
    # with too few AAs after the union with the global omit_AA. Empty cap → no-op
    # → the merged omit (and the whole run) is byte-identical. (See codex review:
    # the cap-set alone passing its guard is not enough — the union can still
    # zero out a position.)
    if fraction_cap_omit:
        _pre_cap_omit = merged_omit
        merged_omit = merge_omit_dicts(_pre_cap_omit, fraction_cap_omit)
        merged_omit = _enforce_min_sampleable_after_cap(
            merged_omit, _pre_cap_omit, cycle_cfg.omit_AA,
        )
    # WS-E: bound the EFFECTIVE (bias_k + global bias_AA) sampling bias to ±N nats.
    # None => bias_k unchanged (same object) => byte-identical. The bias.npy saved
    # above is the un-clamped per-position fusion bias (a diagnostic); the clamp
    # adjusts only what the sampler sees. Parse the FINAL serialized bias_AA so the
    # 2-decimal rounding LigandMPNN actually applies is reflected exactly.
    #
    # CF-3a: --bias_total_clamp_odds X expresses the ceiling in temperature-invariant
    # ODDS space — the per-cell clamp magnitude becomes nats_for_odds(X, T) = T*ln(X)
    # at THIS cycle's sampling temperature (threaded via cycle_cfg.sampling_temperature),
    # then runs through the SAME _clamp_bias_total path. The two clamp flags are
    # mutually exclusive (validated at parse time), so at most one of these is set;
    # neither set => bias_for_sampling stays bias_k (same object) => byte-identical.
    bias_for_sampling = bias_k
    _clamp_nats = bias_total_clamp
    if bias_total_clamp_odds is not None:
        from protein_chisel.sampling.bias_scale import nats_for_odds
        _clamp_nats = nats_for_odds(bias_total_clamp_odds, cycle_cfg.sampling_temperature)
    if _clamp_nats is not None:
        from protein_chisel.sampling.adaptive_bias import AA_TO_IDX, parse_bias_AA
        _bias_AA_vec = np.zeros(20, dtype=bias_k.dtype)
        for _aa, _v in parse_bias_AA(bias_AA_str).items():
            if _aa in AA_TO_IDX:
                _bias_AA_vec[AA_TO_IDX[_aa]] = _v
        bias_for_sampling = _clamp_bias_total(bias_k, _clamp_nats, _bias_AA_vec)
        n_clamped = int((np.abs(bias_for_sampling - bias_k) > 1e-9).sum())
        if n_clamped:
            if bias_total_clamp_odds is not None:
                LOGGER.info("cycle %d: bias_total_clamp_odds=%.2f (=%.3f nats @ T=%.3f) "
                            "adjusted %d (pos,AA) cells (effective bias_k+bias_AA "
                            "bounded to %.2fx odds)",
                            cycle_cfg.cycle_idx, bias_total_clamp_odds, _clamp_nats,
                            cycle_cfg.sampling_temperature, n_clamped,
                            bias_total_clamp_odds)
            else:
                LOGGER.info("cycle %d: bias_total_clamp=%.2f adjusted %d (pos,AA) cells "
                            "(effective bias_k+bias_AA bounded)",
                            cycle_cfg.cycle_idx, _clamp_nats, n_clamped)
    # ---- CF-3 §(c) nested ceilings (BUG-B): reserve the controller's odds headroom
    # INSIDE the whole-stack ceiling so BOTH bind. Opt-in via --controller_coordinator
    # (default OFF => this block is skipped entirely => byte-identical). The controller
    # bucket = the adaptive surface delta + the controller's SURVIVING global term
    # (merged − class-balance, so "class-balance wins" is respected); the rest = the
    # PLM+consensus+throat stack. cell_total = clip(rest, ±(total−reserve)) +
    # clip(controller, ±reserve), all in nats at THIS cycle's T (the application T).
    if controller_coordinator:
        from protein_chisel.sampling.bias_scale import nats_for_odds as _nfo
        from protein_chisel.sampling.adaptive_bias import (
            AA_TO_IDX as _AAI, parse_bias_AA as _pba)
        from protein_chisel.sampling.coordinator import (
            nested_total_clip as _nested, TOTAL_CEILING as _TOTAL_CEIL)
        _T = cycle_cfg.sampling_temperature
        if _T is not None and _T > 0:
            _total_nats = _nfo(_TOTAL_CEIL, _T)
            _reserve_nats = _nfo(controller_ceiling, _T)
            # controller's surviving global = merged bias_AA − class-balance-only.
            _merged_vec = np.zeros(20, dtype=bias_k.dtype)
            for _aa, _v in _pba(bias_AA_str).items():
                if _aa in _AAI:
                    _merged_vec[_AAI[_aa]] = _v
            _cb_vec = np.zeros(20, dtype=bias_k.dtype)
            for _aa, _v in _pba(_class_balance_bias_AA).items():
                if _aa in _AAI:
                    _cb_vec[_AAI[_aa]] = _v
            _ctrl_global_vec = (_merged_vec - _cb_vec)[None, :]
            _ctrl_delta = (adaptive_bias_delta.astype(bias_k.dtype)
                           if (adaptive_bias_delta is not None
                               and adaptive_bias_delta.shape == bias_k.shape)
                           else np.zeros_like(bias_k))
            # The controller bucket is the coordinator's BUDGET-tier output, already
            # bounded to ±reserve per cell by coordinate() (the global D/E/K/R and the
            # surface hydrophobic actuators are DISJOINT in AA space, so they never sum
            # past reserve at any one (pos,AA)). Clip to ±reserve as a belt-and-suspenders
            # carve-out — NOT a heuristic "excess == veto" split: a budget term must never
            # be reclassified as veto and allowed to bypass the total ceiling (codex). A
            # true controller VETO-tier axis (none exist today — veto = omit masks / the
            # clash floor, applied OUTSIDE the controller) would need its own separate
            # threading here; the pure coordinate() still preserves veto bypass for direct
            # callers (test_veto_tier_bypasses_budget).
            _controller_bucket = np.clip(_ctrl_delta + _ctrl_global_vec,
                                         -_reserve_nats, _reserve_nats)
            # VETO bypass for the graded-clash term: it is a ban (clash/omit/fraction-cap
            # tier) and "bypasses the budget entirely (never diluted)". It rides in bias_k
            # (folded into base_bias), so subtract it from the stack BEFORE the nested clip
            # and add it back un-clipped afterward, so the whole-stack ceiling never
            # weakens a clash discouragement. None (not threaded) => zero => unchanged.
            _veto_bias = (clash_bias.astype(bias_k.dtype)
                          if (clash_bias is not None
                              and clash_bias.shape == bias_k.shape)
                          else np.zeros_like(bias_k))
            _eff_total = bias_for_sampling + _merged_vec[None, :]
            # rest = the BUDGETED whole-stack remainder = effective total − the reserved
            # controller − the clash veto bypass (so the nested clip bounds only what
            # should be bounded; the controller gets ±reserve; the clash veto untouched).
            _rest = _eff_total - _controller_bucket - _veto_bias
            _nested_total = (_nested(_rest, _controller_bucket,
                                     total_nats=_total_nats, reserve_nats=_reserve_nats)
                             + _veto_bias)
            _new_bias = (_nested_total - _merged_vec[None, :]).astype(bias_k.dtype)
            _n_nested = int((np.abs(_new_bias - bias_for_sampling) > 1e-9).sum())
            bias_for_sampling = _new_bias
            if _n_nested:
                LOGGER.info("cycle %d: controller_coordinator nested ceilings "
                            "(total %.0fx=%.3f nats, controller reserve %.0fx=%.3f nats "
                            "@ T=%.3f) adjusted %d (pos,AA) cells",
                            cycle_cfg.cycle_idx, _TOTAL_CEIL, _total_nats,
                            controller_ceiling, _reserve_nats, _T, _n_nested)
    cand_tsv = stage_sample(
        cycle_cfg=cycle_cfg, seed_pdb=seed_pdb, bias=bias_for_sampling,
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
        sap_corrected=sap_corrected,
    )

    n_struct = len(pd.read_csv(survivors_struct, sep="\t"))
    if n_struct == 0:
        LOGGER.warning("cycle %d: zero struct survivors -- nothing to score/rank",
                        cycle_cfg.cycle_idx)
        return None, pdb_map, {}

    # ---- 4b. Tunnel patency / pocket accessibility -----------------
    if tunnel_metrics_enabled:
        tunnel_dir = cycle_dir / "035_tunnel"
        survivors_tunnel = stage_tunnel_metrics(
            survivors_struct_tsv=survivors_struct, pdb_map=pdb_map,
            out_dir=tunnel_dir,
            catalytic_resnos=fixed_resnos,
            ligand_min_radius=ligand_min_radius,
            ligand_resname=ligand_resname,
            chain=CHAIN,
            hard_gate=tunnel_hard_gate,
        )
        n_after_tunnel = len(pd.read_csv(survivors_tunnel, sep="\t"))
        if n_after_tunnel == 0:
            LOGGER.warning(
                "cycle %d: zero survivors after tunnel hard-gate "
                "-- skipping fitness/rank for this cycle",
                cycle_cfg.cycle_idx,
            )
            return None, pdb_map, {}
        survivors_struct = survivors_tunnel  # downstream uses this path

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

    # ---- 7. Compute throat-bias delta for next cycle ----------------
    cycle_telem: dict = {}
    if tunnel_metrics_enabled and len(ranked_df) > 0:
        try:
            from protein_chisel.tools.tunnel_metrics import (
                aggregate_blocker_stats, build_throat_bias_delta,
            )
            # Use this cycle's TOP-RANKED survivors (these will inform
            # which throats are recurring problems). Cap at 100 for cost.
            top_ids = ranked_df["id"].astype(str).tolist()[:100]
            top_pdbs = [pdb_map[i] for i in top_ids if i in pdb_map and pdb_map[i].is_file()]
            if top_pdbs:
                blocker_stats = aggregate_blocker_stats(
                    pdb_paths=top_pdbs,
                    catalytic_resnos=fixed_resnos,
                    chain=CHAIN,
                    ligand_resname=ligand_resname,
                )
                # Build the new (L, 20) delta. Decay applied at the
                # NEXT cycle when consumed.
                L_bias = bias_k.shape[0]
                bias_delta_new, throat_telem = build_throat_bias_delta(
                    blocker_stats=blocker_stats,
                    L=L_bias,
                    protein_resnos=protein_resnos,
                    fixed_resnos=fixed_resnos,
                )
                # ACCUMULATE: combine the previously-applied (decayed)
                # bias with this cycle's new observations. This makes
                # throat pressure cumulative while letting one-off
                # observations decay if they don't recur.
                if throat_bias_applied is not None:
                    bias_delta_carry = throat_bias_applied + bias_delta_new
                else:
                    bias_delta_carry = bias_delta_new

                cycle_telem["throat_bias_delta"] = bias_delta_carry
                cycle_telem["blocker_stats"] = {
                    int(k): v for k, v in blocker_stats.items()
                }
                cycle_telem["throat_telemetry"] = throat_telem
                LOGGER.info(
                    "cycle %d: collected throat blockers from %d top survivors; "
                    "%d positions targeted for next-cycle bias (max penalty %.2f nats)",
                    cycle_cfg.cycle_idx, len(top_pdbs),
                    throat_telem["n_positions_targeted"],
                    max((p["max_penalty_nats"] for p in throat_telem["positions"]), default=0.0),
                )
                # Save telemetry for offline inspection — into BOTH the
                # cycle dir (gets stripped by --shipping_layout) AND the
                # run_dir top level (survives stripping).
                cycle_telem_path = cycle_dir / "throat_blocker_telemetry.json"
                run_dir_path = cycle_dir.parent
                aggregate_path = run_dir_path / "throat_blocker_telemetry.json"
                payload = {
                    "blocker_stats": {str(k): v for k, v in blocker_stats.items()},
                    "throat_telemetry": throat_telem,
                }
                with open(cycle_telem_path, "w") as fh:
                    json.dump(payload, fh, indent=2, default=str)
                # Append-style: read existing if present, append this cycle
                try:
                    if aggregate_path.is_file():
                        existing = json.load(open(aggregate_path))
                        if "per_cycle" not in existing:
                            existing = {"per_cycle": {}}
                    else:
                        existing = {"per_cycle": {}}
                    existing["per_cycle"][str(cycle_cfg.cycle_idx)] = payload
                    with open(aggregate_path, "w") as fh:
                        json.dump(existing, fh, indent=2, default=str)
                except Exception:
                    pass
        except Exception as exc:
            LOGGER.warning("throat-bias delta computation failed: %s", exc)

    return ranked_df, pdb_map, cycle_telem


# ----------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed_pdb", type=Path, required=True,
                   help="Path to the input PDB (with REMARK 666 catalytic "
                        "motif lines; catres auto-derived).")
    p.add_argument("--ligand_params", type=Path, required=True,
                   help="Path to the Rosetta .params file for the ligand.")
    p.add_argument("--plm_artifacts_dir", type=Path, required=True)
    p.add_argument("--experts", default="esmc,saprot",
                   help="Comma list of fusion experts (registry names) whose "
                        "<name>_log_probs.npy artifacts to load + fuse. Default "
                        "'esmc,saprot' = the legacy two-PLM fusion (byte-identical). "
                        "Must match the --experts used in precompute.")
    p.add_argument("--plm_dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="Precision the PLM artifacts were precomputed at. Selects the "
                        "per-expert artifact filenames to load (fp32=<name>_log_probs.npy "
                        "legacy; fp16/bf16=<name>_log_probs.<dtype>.npy). MUST match the "
                        "--plm_dtype used in precompute. Default fp32 = byte-identical.")
    p.add_argument("--metrics", default="all",
                   help="Comma list of metric names (protein_chisel.metrics catalog) "
                        "to compute/report, or 'all' (default = today's full set, "
                        "byte-identical). Recorded in provenance + logged; with the "
                        "default 'all' nothing is gated. Unknown names with an "
                        "explicit (non-'all') selection are an error.")
    p.add_argument("--filters", default="all",
                   help="Comma list of FILTER metric names (subset of --metrics) "
                        "allowed to drop designs, or 'all' (default). Recorded in "
                        "provenance + logged; gating wiring is additive and a no-op "
                        "at 'all'.")
    # ---- Decode-time PoE backend (opt-in; default 'bias' = in-process sampler) ----
    p.add_argument("--mpnn_backend", choices=["bias", "poe"], default="bias",
                   help="Sampling backend. 'bias' (default) = in-process LigandMPNN "
                        "with our calibrated static fusion bias (byte-identical to "
                        "before). 'poe' = decode-time Product-of-Experts (Sebastian/"
                        "Joe Mi fused_mpnn_poe) run as a separate HOST stage, one-shot, "
                        "feeding the driver's score/rank. Orchestrated by "
                        "run_chisel_design.sh (MPNN_BACKEND=poe).")
    p.add_argument("--additional_experts", default="",
                   help="PoE only: comma list of context-aware experts mixed in at "
                        "decode time (hermes,dms,msa,esm,wt_esm,vesm,wt_vesm,e1,wt_e1).")
    p.add_argument("--additional_expert_lambdas", default="",
                   help="PoE only: comma list of mixing weights (one per expert; each "
                        "in (0,1); sum < 1 so lambda_mpnn = 1 - sum).")
    p.add_argument("--poe_emit_inputs", type=Path, default=None,
                   help="PoE internal: write this run's cycle-0 bias/fixed/omit JSONs "
                        "to this dir (for the host PoE stage) and exit. Set by the "
                        "shell wrapper; not for direct use.")
    p.add_argument("--poe_output_dir", type=Path, default=None,
                   help="PoE internal: a finished host PoE output dir (seqs/+packed/) "
                        "to use as the candidate pool (score-only). Set by the shell.")
    p.add_argument("--poe_temperature", type=float, default=0.1,
                   help="PoE internal: the temperature the host PoE stage sampled at "
                        "(recorded in the candidate sampler_params_hash). Set by shell.")
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
    p.add_argument("--debug-short-test", "--debug_short_test",
                   dest="debug_short_test", action="store_true",
                   help="Fast smoke-test preset. UNCONDITIONALLY clobbers "
                        "--target_k -> 5 and --cycles -> 3 (with a 20/10/10 "
                        "sample schedule built by debug_short_test_cycles). "
                        "If you also pass --target_k / --cycles on the same "
                        "command line, your values are ignored and a WARNING "
                        "is logged. Intended only to validate end-to-end "
                        "execution quickly; never use for production runs.")
    p.add_argument("--omit_AA", type=str, default="X",
                   help="AAs MPNN may never sample. Default 'X' (UNK only) -- "
                        "no canonical AAs are silently forbidden. Pass 'CX' "
                        "to also exclude cysteine (recommended for scaffolds "
                        "with no catalytic Cys); add others as needed.")
    p.add_argument("--catalytic_resnos", type=_parse_catalytic_resnos_arg,
                   default=None, metavar="R1,R2,...",
                   help="Comma-separated 1-indexed catalytic resnos (chain A) to "
                        "fix/protect, e.g. '41,64,187'. Default None. Resolution "
                        "priority: this flag > the seed PDB's REMARK 666 motif "
                        "block > the hard-coded PTE_i1 builtin "
                        "(60,64,128,131,132,157). Set this for ANY non-PTE "
                        "scaffold whose seed lacks REMARK 666 — otherwise the run "
                        "falls back to the PTE positions (with a loud warning) "
                        "and pins the wrong residues.")
    p.add_argument("--chain", type=str, default="A", metavar="CHAIN_ID",
                   help="Single-character chain id of the catalytic/design chain "
                        "in the seed PDB. Default 'A' (byte-identical). Set this "
                        "for any scaffold whose design chain is not 'A' (e.g. "
                        "'--chain B'). All structural reads (H-bond/clash/preorg/"
                        "interaction detection, sequence extraction, secondary "
                        "structure, tunnel lining) use this chain.")
    p.add_argument("--no_require_cat_his_hbond", action="store_true",
                   help="Disable ONLY the structural filter criterion that "
                        "requires >=1 side-chain H-bond to a catalytic HIS. "
                        "Default OFF => the requirement stays ON (byte-identical). "
                        "Set this for an enzyme whose mechanism has NO catalytic "
                        "His — otherwise that criterion rejects every design. "
                        "Other struct-filter criteria (SAP, clash) are unaffected.")
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
    p.add_argument("--use_side_chain_context_schedule", type=_scc_schedule_arg,
                   default=None, metavar="CSV01",
                   help="Opt-in per-cycle side-chain-context schedule: a comma-"
                        "separated list of 0/1, e.g. '1,1,0' = ON for cycles 0-1 "
                        "(clash avoidance — materially helps avoid clashes early), "
                        "OFF for cycle 2 (first-shell diversity late). Overrides "
                        "the uniform --use_side_chain_context per cycle. Broadcast/"
                        "truncate to the run's cycle count: a short schedule repeats "
                        "its LAST entry for later cycles; a long one is truncated. "
                        "Absent (default) => the uniform value is used unchanged "
                        "(byte-identical).")
    p.add_argument("--enhance", type=str, default=None,
                   choices=[None, *AVAILABLE_ENHANCE_CHECKPOINTS],
                   help="Optional pLDDT-enhanced fused_mpnn checkpoint name. "
                        "Default None (use base ligand_mpnn). Available choices "
                        f"({len(AVAILABLE_ENHANCE_CHECKPOINTS)}): "
                        + ", ".join(AVAILABLE_ENHANCE_CHECKPOINTS))
    p.add_argument("--pi_min", type=float, default=5.0,
                   help="Minimum theoretical pI. Default 5.0 selects the "
                        "least-acidic ~1%% of cycle-0 designs under the "
                        "net-charge band (see --net_charge_min/--net_charge_max; "
                        "constant across cycles). Low cycle-0 pass rate is fine: "
                        "consensus-bias iteration in cycle 1+ pulls subsequent "
                        "cycles toward less-acidic sequences. Relax to 4.7 for "
                        "higher cycle-0 pass at the cost of weaker selection "
                        "pressure.")
    p.add_argument("--pi_max", type=float, default=7.5)
    p.add_argument("--fpocket_druggability_min", type=float, default=0.30,
                   help="Drop designs with fpocket-druggability below this "
                        "(no detectable active-site cavity = bad design). "
                        "Set to 0 to disable.")
    p.add_argument("--final_filter_backfill", type=_parse_bool_arg, default=True,
                   metavar="{true,false}",
                   help="Default true. If the final hard fpocket-druggability "
                        "cutoff leaves fewer than target_k designs, backfill "
                        "from the closest near-miss candidates so the final "
                        "top-K stays populated when possible. Pass designs "
                        "are always prioritized ahead of backfilled ones; "
                        "tool-failed fpocket rows are only used as a last "
                        "resort after real near-misses. Set false for strict "
                        "pass-only final outputs.")
    p.add_argument("--ship_solubility_veto", action="store_true",
                   help="Opt-in hard solubility veto (default OFF → byte-identical). "
                        "When set, the final top-K writer drops any design OUTSIDE "
                        "the final-cycle GRAVY + net-charge band before shipping, so "
                        "the deferred-rescue / backfill path can never ship a "
                        "seq-filter-failing design (e.g. GRAVY=1.05) as rank-0. May "
                        "ship fewer than target_k (intended). Adds a truthful "
                        "selection__solubility_passed column (distinct from "
                        "selection__hard_final_filter_passed = fpocket druggability).")
    p.add_argument("--sap_corrected", action="store_true",
                   help="Opt-in (default OFF → byte-identical). Additionally emit "
                        "sap_corr_{max,mean,p95} columns on per-cycle designs: a "
                        "centered, polar-cancellation-free SAP (shared scoring.sap "
                        "module) that — unlike the legacy signed-KD sap_* — registers "
                        "alanine-rich hydrophobic surfaces and isn't cancelled by "
                        "exposed polar residues. Legacy sap_* are unchanged. (Rescued "
                        "backfill rows carry NaN sap_corr_* — they are not re-scored "
                        "for it.)")
    # ---- AA-composition baseline reference (opt-in; default == legacy) -----------
    p.add_argument("--aa_reference", type=_aa_reference_arg,
                   default="swissprot_ec3_hydrolases_2026_01",
                   metavar="NAME",
                   help="AA-composition baseline distribution that the over-"
                        "representation checks score against — the per-cycle "
                        "class-balanced bias_AA and the adaptive-bias hydrophobic "
                        "over-rep mask. Default 'swissprot_ec3_hydrolases_2026_01' "
                        "(EC-3 hydrolases) is unchanged, so an un-passed flag is "
                        "BYTE-IDENTICAL. Select the design's own EC class for a "
                        "non-hydrolase enzyme (e.g. 'swissprot_ec2_transferases_"
                        "2026_01', 'swissprot_enzyme_2026_01') so compositions are "
                        "compared to the right distribution. Validated at parse "
                        "time against the bundled REFERENCE_DISTRIBUTIONS keys "
                        "(unknown -> error listing the valid keys).")
    # ---- WS-C composition control (opt-in, default OFF/None → byte-identical) ----
    p.add_argument("--composition_suppress_all_overrep", action="store_true",
                   help="Opt-in (default OFF → byte-identical). In the per-cycle "
                        "class-balanced bias_AA, down-weight EVERY over-represented "
                        "member of an AA class (z > --balance_z_threshold), not just "
                        "the single class maximum. Without it, when Alanine is the "
                        "hydrophobic-class max an also-over-represented Leucine "
                        "escapes correction. The property-conserving within-class "
                        "swap up-weight is preserved.")
    p.add_argument("--aa_fraction_cap", type=_fraction_arg, default=None,
                   metavar="FRAC",
                   help="Opt-in (default None → byte-identical). Finite fraction in "
                        "(0, 1]. Hard-omit any amino acid whose fraction in a cycle's "
                        "survivor pool is >= FRAC (e.g. 0.15) at every non-fixed "
                        "designable position the next cycle, bounding runaway "
                        "single-AA over-representation (the 26%%-Ala failure mode). "
                        "Recomputed per cycle, so an AA is re-allowed once it falls "
                        "back under the cap. A cap so low it would omit nearly every "
                        "AA for a low-diversity pool is skipped that cycle with an "
                        "ERROR (never over-constrains the sampler).")
    p.add_argument("--composition_soft_bias", action="store_true",
                   help="Opt-in (default OFF → byte-identical). Activate the "
                        "expression engine's per-residue SOFT_BIAS tier "
                        "(long-hydrophobic-stretch, KR-near-catalytic-on-helix, "
                        "polyproline, repetitive-segment, …): add a negative "
                        "per-(position,AA) bias at those sites to every cycle's "
                        "sampler bias. POOL-DERIVED per cycle (the engine is "
                        "re-evaluated on the survivors and a liability is applied "
                        "only if it recurs in >= half of them), so it tracks the "
                        "liabilities the designs introduce; the seed map bootstraps "
                        "cycle 0. Whole-protein composition hits are excluded (those "
                        "are the suppress-all / fraction-cap levers' job).")
    p.add_argument("--composition_soft_bias_nats",
                   type=lambda s: _nonneg_finite_arg(s, max_value=_SOFT_BIAS_NATS_MAX),
                   default=0.5,
                   help="Down-weight magnitude (nats, finite in [0, "
                        "%g]) applied at each SOFT_BIAS (position, AA) cell when "
                        "--composition_soft_bias is set. Default 0.5. NOTE the bias "
                        "is added in LOGIT space (before the softmax temperature "
                        "divide), so the effective odds penalty is "
                        "exp(nats / sampling_temperature) — at T≈0.15-0.20 a 0.5-nat "
                        "bias is ~12-28x (a firm nudge), whereas 1.5 would be "
                        "~1800-22000x (a near-hard ban). Keep this near the adaptive "
                        "controller's ~0.6 clamp. No effect unless "
                        "--composition_soft_bias." % _SOFT_BIAS_NATS_MAX)
    p.add_argument("--composition_pool_fallback", action="store_true",
                   help="Opt-in (default OFF → byte-identical). The WS-C "
                        "composition cap / class-balance / soft-bias normally read "
                        "the previous cycle's SURVIVOR pool, so on a hydrophobic seed "
                        "where ~100%% of samples fail the GRAVY band (empty survivor "
                        "pool) they never fire and the run ships a runaway single-AA "
                        "composition (the 26%%-Ala backfill mode). With this flag, "
                        "when a cycle's survivor pool is empty the cap/class-balance/"
                        "soft-bias are built from the PREVIOUS cycle's full SAMPLED "
                        "(pre-band-filter) pool instead — the same source the adaptive "
                        "controller reads — so the levers still engage. No-op at cycle "
                        "0 (no previous pool) and whenever survivors exist.")
    p.add_argument("--copy-input-structure-into-out-dir",
                   "--copy_input_structure_into_out_dir",
                   dest="copy_input_structure_into_out_dir",
                   type=_parse_bool_arg, default=True,
                   metavar="{true,false}",
                   help="Default true. After final selection, compute a "
                        "metrics row for the original input structure and "
                        "carry it into the shipped output directory alongside "
                        "the final designs. The original input PDB filename is "
                        "preserved exactly.")
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
    p.add_argument("--tunnel_metrics", action="store_true", default=True,
                   help="Score pocket accessibility / tunnel patency "
                        "after stage_struct_filter. Adds ~0.3 s/design "
                        "(homegrown ray-cast + optional pyKVFinder). "
                        "Hard-gates designs whose verdict is 'buried' or "
                        "whose pyKVFinder cavity has no opening to bulk "
                        "solvent (the failure mode where the pocket "
                        "interior looks great but the entrance is plugged "
                        "by a residue cluster). Adds 8 tunnel__ columns + "
                        "5 pkvf__ columns to topk.tsv.")
    p.add_argument("--no_tunnel_metrics", dest="tunnel_metrics",
                   action="store_false")
    p.add_argument("--tunnel_hard_gate", action="store_true", default=True,
                   help="When --tunnel_metrics is on, drop designs with "
                        "verdict in {buried, ligand_too_big} BEFORE TOPSIS "
                        "ranking, since no sequence redesign can fix "
                        "those. Off-by-default would let them be ranked "
                        "down but kept; on (default) is more aggressive.")
    p.add_argument("--no_tunnel_hard_gate", dest="tunnel_hard_gate",
                   action="store_false")
    p.add_argument("--throat_feedback", action="store_true", default=True,
                   help="Aggregate throat-blocker observations from cycle k's "
                        "TOP survivors and add a (decayed) per-position "
                        "negative bias to MPNN's bias_AA at those positions in "
                        "cycle k+1 — actively pressures MPNN to swap bulky "
                        "residues at the entrance. Cumulative across cycles "
                        "with exponential decay (default 0.5) so positions "
                        "that successfully open up release pressure. Default ON.")
    p.add_argument("--no_throat_feedback", dest="throat_feedback",
                   action="store_false")
    p.add_argument("--throat_feedback_decay", type=float, default=0.5,
                   help="Exponential decay applied to the carried-forward "
                        "throat bias at the start of each cycle. 0.5 = halve "
                        "the previous bias before adding new observations. "
                        "Lower values release pressure faster.")
    # ---- Adaptive solubility bias controller (opt-in, default OFF) ----
    p.add_argument("--adaptive_bias", action="store_true", default=False,
                   help="OPT-IN closed-loop controller: after each cycle, measure "
                        "the candidate pool's net charge and surface hydrophobicity "
                        "and steer the next cycle's MPNN biases toward target "
                        "solubility (global D/E up-weight; per-position hydrophobic "
                        "down-weight at solvent-exposed surface positions). Fires "
                        "only when the pool is statistically out of target, holds the "
                        "bias once in-band, and reverses if it overshoots. Default "
                        "OFF => byte-identical to the legacy pipeline.")
    p.add_argument("--adaptive_bias_gain", type=float, default=0.6,
                   help="Initial integral gain (unitless band-normalized error).")
    p.add_argument("--adaptive_bias_max_nats", type=float, default=0.6,
                   help="Clamp on the per-AA / per-cell bias magnitude (nats).")
    p.add_argument("--adaptive_bias_max_odds", type=float, default=None, metavar="X",
                   help="CF-1 opt-in: clamp the controller |bias| in ODDS space at X-fold "
                        "(temperature-invariant: max_nats := T*ln(X) each cycle) instead "
                        "of the raw --adaptive_bias_max_nats. None (default) keeps the raw "
                        "nats clamp => byte-identical. Typical 2 (nudge) to 8 (strong); "
                        "MPNN odds shift is exp(bias/T) so a fixed nats clamp silently "
                        "amplifies as T anneals — this makes the controller's authority "
                        "invariant to the temperature schedule.")
    p.add_argument("--adaptive_bias_carry", type=float, default=0.9,
                   help="Integral leak during active correction (1=pure integral). "
                        "When the pool is in-band the bias is held exactly.")
    p.add_argument("--adaptive_bias_deadband", type=float, default=0.25,
                   help="Deadband as a fraction of the band half-width: the "
                        "controller aims for target but tolerates deviations within "
                        "this fraction (avoids chasing noise).")
    p.add_argument("--adaptive_bias_tmin", type=float, default=2.5,
                   help="|t|-stat threshold (pool mean vs target) for the gate.")
    p.add_argument("--adaptive_bias_fmin", type=float, default=0.15,
                   help="Fail-fraction threshold for the gate.")
    p.add_argument("--adaptive_bias_min_n", type=int, default=30,
                   help="Minimum pool size to act on an axis.")
    p.add_argument("--adaptive_bias_mode", choices=["proportional", "bangbang"],
                   default="proportional",
                   help="Control law: proportional (integral) or bang-bang "
                        "(provably non-divergent; only the sign of the plant "
                        "response matters).")
    p.add_argument("--adaptive_bias_seed_from_input", action="store_true",
                   default=False,
                   help="Warm-start cycle 0 from the INPUT scaffold's hydrophobicity/"
                        "charge instead of waiting for cycle 0's output.")
    # ---- WS-D controller expansion (opt-in; defaults reproduce today's controller) ----
    p.add_argument("--adaptive_surface_sasa_gate", type=_fraction_arg, default=None,
                   metavar="FRAC",
                   help="Opt-in (default None => legacy distal_surface scope, "
                        "byte-identical). When set (e.g. 0.20), the controller's "
                        "surface hydrophobic down-weight acts on the "
                        "non_tunnel_surface set: every exposed (sidechain-SASA "
                        "fraction >= FRAC) NON-active-site, non-tunnel-lining, "
                        "non-fixed position — a superset of distal_surface (adds "
                        "exposed nearby_surface, drops the ligand-distance gate) "
                        "minus the tunnel mouth. Steers 'what you can see by eye'.")
    p.add_argument("--adaptive_charge_band", type=str, default=None, metavar="LO,HI",
                   help="Opt-in (default None => the cycle's net-charge filter band). "
                        "Override the controller's net-charge target band, e.g. "
                        "'-15,-5'. The target becomes the band midpoint and the "
                        "fail-fraction is evaluated on raw net charge (the cycle-band "
                        "gap columns no longer apply). Steers net charge to your band "
                        "without changing the seq filter.")
    p.add_argument("--adaptive_bias_axes", type=str, default=None, metavar="LIST",
                   help="Opt-in comma list of controller axes to run (default "
                        "'charge,surface_hydrophobicity'). Restrict (e.g. 'charge') "
                        "or, as future registry entries land, extend. An unknown "
                        "axis name is rejected.")
    p.add_argument("--controller_damping", action="store_true", default=True,
                   dest="controller_damping",
                   help="Control-law DAMPING (ON by default as of 1.2.0; validated to "
                        "stabilize charge regulation — held charge at target across mid "
                        "+ hard seeds vs the legacy law's −6.8→−8.8→−1.3 limit-cycle. "
                        "Pass --no_controller_damping to revert). The base law under-"
                        "corrects against a drifting/lagging plant, relaxes its "
                        "integral the moment the pool is momentarily in-band, then "
                        "ramps hard when the pool drifts back. This bundle adds four "
                        "stabilizers in one switch: act on an EWMA of the pool mean "
                        "(measurement_ewma_alpha=0.5, filters per-cycle noise), a "
                        "derivative-on-measurement term (derivative_gain=0.5*gain, "
                        "anticipates drift before the hard ramp), a per-cycle slew "
                        "limit (slew_limit_frac=0.15 of max_nats, no full-range "
                        "lurch), and a soft 'ramp' deadband (continuous drive through "
                        "target, kills the stick-slip of the hard band). No effect "
                        "without --adaptive_bias.")
    p.add_argument("--no_controller_damping", action="store_false",
                   dest="controller_damping",
                   help="Disable the default control-law damping (revert to the legacy "
                        "under-damped integral law). No effect without --adaptive_bias.")
    p.add_argument("--controller_verbose", action="store_true", default=False,
                   help="CF-5 opt-in observability (default OFF => byte-identical). "
                        "With --adaptive_bias, after each cycle append one row per "
                        "axis to <run_dir>/controller_trace.tsv (long format: cycle, "
                        "axis, scope, measured vs target/band, signed_error, gate + "
                        "why, drive_u, effective_odds=exp(u/T), n) and emit a per-cycle "
                        "CONTROLLER REPORT to the log, for step-by-step chronological "
                        "validation. Advisory only — a trace write error never stops "
                        "the run. No effect without --adaptive_bias.")
    p.add_argument("--controller_coordinator", action="store_true", default=False,
                   help="CF-2/CF-3 opt-in multi-objective controller COORDINATOR "
                        "(default OFF => byte-identical). With --adaptive_bias, route "
                        "the controller's GLOBAL per-AA bias AND the (L,20) surface "
                        "delta through a weight-partitioned, work-conserving, signed-"
                        "sum-bounded JOINT odds budget instead of the legacy sum-then-"
                        "rescale. Axes that share an ACTUATOR (e.g. charge and pI both "
                        "drive D/E/K/R) collapse by sign-selected MAX (same-sign, anti-"
                        "double-count) or signed SUM (opposite-sign) — never a second "
                        "lock. The controller share is bounded to "
                        "--controller_ceiling odds and nested INSIDE a ~1e3x whole-"
                        "stack ceiling at the sampler (both bind, reserved headroom), "
                        "all at the application-cycle T. Enables the pI axis to share "
                        "the charge actuator. No effect without --adaptive_bias.")
    p.add_argument("--controller_ceiling", type=float, default=8.0, metavar="X",
                   help="Joint CONTROLLER odds ceiling for --controller_coordinator "
                        "(default 8 = ODDS_STRONG; moves a pool meaningfully without "
                        "locking it). Must be > 1.0. No effect without "
                        "--controller_coordinator.")
    # ---- WS-E sampling-core safety (opt-in; defaults => byte-identical) ----
    p.add_argument("--bias_total_clamp", type=_nonneg_finite_arg, default=None,
                   metavar="NATS",
                   help="Bound the EFFECTIVE per-(position,AA) sampling bias "
                        "(bias_per_residue + global bias_AA) to ±NATS. As of v1.4.0 this "
                        "is ON BY DEFAULT at 3.0 nats (a safety cap on the otherwise-"
                        "uncapped consensus(+2.0)+PLM-peak stack, which at T≈0.15 locks a "
                        "cell ~10^13x; 3 nats preserves a legit ~3-nat single-source PLM "
                        "peak while capping the 6-20-nat double-count lock). Pass an "
                        "explicit value to override the 3-nat default, or "
                        "--no_bias_total_clamp to disable entirely (the pre-1.4.0 path). "
                        "An overflow/stacking guard, not a gentle regularizer.")
    p.add_argument("--no_bias_total_clamp", action="store_true", default=False,
                   help="Opt OUT of the default 3-nat bias-sum safety cap (restores the "
                        "exact pre-1.4.0 unclamped sampling bias). Mirrors the "
                        "--no_controller_damping idiom. Mutually exclusive with an "
                        "explicit --bias_total_clamp / --bias_total_clamp_odds.")
    p.add_argument("--bias_total_clamp_odds", type=float, default=None, metavar="X",
                   help="CF-3a opt-in (default None => byte-identical). Like "
                        "--bias_total_clamp but the ceiling is expressed in ODDS space "
                        "at X-fold: the per-(pos,AA) clamp magnitude becomes "
                        "nats_for_odds(X, T) = T*ln(X) EACH cycle, so 'no cell's total "
                        "bias exceeds X-fold odds' holds invariant to the temperature "
                        "schedule (MPNN samples softmax((logits+bias)/T) so a fixed-nats "
                        "clamp silently tracks T as it anneals). Mutually exclusive with "
                        "--bias_total_clamp. Must be > 1.0 (an odds ceiling <=1 is a "
                        "non-positive clamp). Suggested ~8-100 (a stacking guard).")
    p.add_argument("--sampling_temperature_floor",
                   type=lambda s: _nonneg_finite_arg(s, max_value=2.0), default=None,
                   metavar="T",
                   help="Opt-in (default None => byte-identical). Raise any cycle's "
                        "sampling temperature to at least T (overriding the annealing "
                        "schedule). At T≈0.15 a 0.5-nat bias is ~28x (near-"
                        "deterministic); ~0.3 restores genuine multinomial diversity. "
                        "No effect under the PoE backend.")
    p.add_argument("--plm_class_strength", type=str, default="", metavar="K=V,...",
                   help="Opt-in (default '' => byte-identical). ABSOLUTE per-class "
                        "overrides of the PLM-fusion class weights, e.g. "
                        "'distal_surface=0.3,primary_sphere=0.0'. The global "
                        "--plm_strength still multiplies on top. Unknown class names "
                        "and non-finite/negative values are rejected.")
    # ---- WS-G omit tunnel-lining (opt-in/experimental; default OFF => byte-identical) ----
    p.add_argument("--omit_tunnel_lining", action="store_true", default=False,
                   help="Opt-in/experimental (default OFF => byte-identical). Hard-omit "
                        "bulky AAs (--omit_tunnel_lining_aas) at the seed's tunnel-lining "
                        "positions (is_tunnel_lining) to keep the substrate channel open "
                        "from cycle 0. Complementary to the (soft, reactive) throat-"
                        "feedback bias; a permanent hard ban is blunter, so this is off "
                        "by default. Catalytic/fixed positions are never omitted.")
    p.add_argument("--omit_tunnel_lining_aas", type=str,
                   default=_OMIT_TUNNEL_LINING_DEFAULT, metavar="AAS",
                   help="AAs to hard-omit at tunnel-lining positions when "
                        "--omit_tunnel_lining is set. Default %(default)s = the throat's "
                        "bulky-blocker set (tunnel_metrics._BLOCKER_WEIGHT >= 0.70): "
                        "aromatics W/F/Y/H plus the long charged R/K (Lys Cb->NZ ~5.5 A, "
                        "Arg ~6 A — genuine channel constrictors, same as the throat-"
                        "feedback bias). The medium hydrophobics I/L/M/V are left to that "
                        "controller's capped/decaying pressure; Ala can't constrict.")
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
    # ---- Conserved-sidechain-H-bond fixing (Feature 1) -------------------
    p.add_argument("--conserve_hbonds", type=_parse_bool_arg, nargs="?",
                   const=True, default=False,
                   help="Detect designable-residue SIDECHAIN H-bonds to the "
                        "ligand / fixed residues and probabilistically pin "
                        "those residues into the per-cycle fixed list (rolled "
                        "independently each cycle). Default: off. Shares "
                        "protein_chisel.tools.conserved_hbonds with "
                        "scripts/chisel_ligandMPNN.py.")
    p.add_argument("--conserve_hbond_prob", type=float, default=0.8,
                   help="Per-residue fix probability per cycle (0-1, or a "
                        "percentage <=100). Default 0.8.")
    p.add_argument("--conserve_anchors", type=str,
                   default="ligand,catalytic,user_fixed",
                   help="Comma list of anchor groups a designable sidechain "
                        "must H-bond to (subset of ligand,catalytic,"
                        "user_fixed). NOTE: this driver has no separate "
                        "user-fixed list, so 'user_fixed' is inert here. "
                        "Default: ligand,catalytic,user_fixed.")
    p.add_argument("--conserve_hbond_max_dist", type=float, default=3.9,
                   help="Heavy-atom donor...acceptor distance cutoff (A). "
                        "Default 3.9.")
    p.add_argument("--conserve_hbond_max_angle", type=float, default=90.0,
                   help="Antecedent-D-A angle gate (deg); larger = more "
                        "permissive (stock detector uses 70). Default 90.")
    p.add_argument("--conserve_keep_clashing", action="store_true",
                   help="Keep conserved candidates whose sidechain clashes "
                        "with fixed backbone/ligand (default: exclude with a "
                        "log line).")
    p.add_argument("--conserve_seed", type=int, default=None,
                   help="Base seed for per-cycle conservation rolls (seed = "
                        "'<base>:<cycle>'). Default: auto-generated and logged "
                        "so the run is replayable with --conserve_seed.")
    # ---- Active-site interaction-network growth (add-on #6) ---------------
    p.add_argument("--conserve_hbond_depth", type=int, default=1,
                   help="Grow the conserved network outward in shells from the "
                        "active site: shell 1 bonds the ligand/catalytic, shell 2 "
                        "bonds shell-1 residues, etc. Default 1 = legacy "
                        "single-shell fixing (byte-identical).")
    p.add_argument("--conserve_interaction_types", default="hbond",
                   help="Comma list of interaction types defining the network "
                        "edges. Currently only 'hbond' is wired (salt_bridge/"
                        "pi_pi/pi_cation/hydrophobic are a planned extension). "
                        "Default 'hbond'.")
    p.add_argument("--conserve_shell_decay", type=float, default=1.0,
                   help="Per-shell multiplier on the fix probability "
                        "(prob * decay**(depth-1)); <1 rolls outer shells less "
                        "aggressively. Default 1.0 (no decay).")
    p.add_argument("--conserve_grow_network", type=_parse_bool_arg, nargs="?",
                   const=True, default=False,
                   help="Accumulate pinned residues across cycles as extra "
                        "anchors, so the network deepens as design proceeds. "
                        "Default off.")
    # ---- Canonical REMARK transfer + DESIGN_PATH provenance (Feature 2) ---
    p.add_argument("--transfer_remarks", type=_parse_bool_arg, nargs="?",
                   const=True, default=True,
                   help="Transfer canonical REMARKs (665/666/667/668/QCB/"
                        "DESIGN_PATH) from the seed onto restored + final PDBs "
                        "and stamp DESIGN_PATH iterative_design provenance, via "
                        "the shared protein_chisel.tools.remarks module. "
                        "Default: on.")
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
    p.add_argument("--net_charge_min", type=float, default=-18.0,
                   help="Acceptance band: drop designs with net_charge_full_HH "
                        "<= this (too acidic). Default -18.0. This is the FINAL "
                        "(strictest) band; it stays CONSTANT across cycles in "
                        "both strategies (charge does not anneal).")
    p.add_argument("--net_charge_max", type=float, default=-4.0,
                   help="Acceptance band: drop designs with net_charge_full_HH "
                        ">= this (not acidic enough). Default -4.0. FINAL band; "
                        "constant across cycles.")
    p.add_argument("--sap_max_threshold", type=float, default=100.0,
                   help="Acceptance band: drop designs with SAP (freesasa-proxy "
                        "scale) above this. Default 100.0 (effectively OFF for "
                        "PTE_i1). FINAL band; constant across cycles.")
    p.add_argument("--instability_max", type=float, default=60.0,
                   help="Light filter on Guruprasad 1990 instability index. "
                        "Lit threshold for native E. coli expression is 40, "
                        "but de novo designs run higher; default 60 catches "
                        "truly broken sequences only. Set 9999 to disable. "
                        "This sets the FINAL (strictest) band; under "
                        "--strategy annealing the earlier cycles relax from it "
                        "by the fixed legacy offsets (c1=+10, c0=+20).")
    p.add_argument("--gravy_min", type=float, default=-0.8,
                   help="Light filter on Kyte-Doolittle GRAVY. Typical "
                        "soluble proteins fall in [-0.4, 0]; default [-0.8, "
                        "0.3] is generous. FINAL band; annealing relaxes "
                        "earlier cycles by the legacy offsets (c1=-0.10, "
                        "c0=-0.20).")
    p.add_argument("--gravy_max", type=float, default=0.3,
                   help="Upper Kyte-Doolittle GRAVY acceptance bound (default "
                        "0.3). FINAL band; annealing relaxes earlier cycles "
                        "(c1=+0.05, c0=+0.10).")
    p.add_argument("--aliphatic_min", type=float, default=40.0,
                   help="Light filter on Ikai 1980 aliphatic index. "
                        "Thermostable native: ~85-100. Default lower bound "
                        "40 catches only extremely low-aliphatic outliers. "
                        "FINAL band; annealing relaxes earlier cycles "
                        "(c1=-5, c0=-10).")
    p.add_argument("--boman_max", type=float, default=4.5,
                   help="Light filter on Boman index (PPI/sticky propensity). "
                        "Boman 2003 threshold ~2.5; default 4.5 catches only "
                        "extreme cases. FINAL band; annealing relaxes earlier "
                        "cycles (c1=+0.5, c0=+1.0).")
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
    # ---- Seed triage: opt-in PLM auto-skip on a pathological input (default OFF) ----
    p.add_argument("--plm_autoskip_bad_input", action="store_true", default=False,
                   help="Opt-in (default OFF => byte-identical). If the INPUT scaffold is "
                        "pathologically hydrophobic / over-represented (per the "
                        "--plm_autoskip_* thresholds), force --plm_strength to 0 for the "
                        "run, so LigandMPNN regenerates from structure + fixed residues "
                        "instead of the PLM bias amplifying the bad seed. Empirically on a "
                        "GRAVY=1.34 seed this took GRAVY->-0.5 and Ala 27%%->0.5%%.")
    p.add_argument("--plm_autoskip_gravy", type=float, default=0.4, metavar="G",
                   help="Seed-triage GRAVY ceiling (default 0.4; trips above it).")
    p.add_argument("--plm_autoskip_max_aa_frac", type=float, default=0.16, metavar="F",
                   help="Seed-triage single-AA fraction ceiling (default 0.16).")
    p.add_argument("--plm_autoskip_hydrophobic_frac", type=float, default=0.50, metavar="F",
                   help="Seed-triage hydrophobic-fraction ceiling (default 0.50).")
    # ---- F1: opt-in distribution-aware z-score over-representation gate -----------
    p.add_argument("--plm_autoskip_aa_zmax", type=float, default=None, metavar="Z",
                   help="Opt-in (default None => z-gate OFF => byte-identical). Add a "
                        "distribution-aware single-AA over-representation signal to the "
                        "seed triage: an AA trips iff its one-sided z (vs --aa_reference's "
                        "per-sequence mean+SD) >= Z AND its log2 enrichment >= "
                        "--plm_autoskip_aa_log2_floor. REDUNDANT with (ORed to) the flat "
                        "--plm_autoskip_max_aa_frac so naturally-abundant (Leu/Ala) and "
                        "rare (Trp/Cys) AAs are judged fairly. The z is a population "
                        "DISTANCE not a significance test; pass the design's own EC class "
                        "via --aa_reference (the EC-3 default is wrong for non-hydrolases). "
                        "Only active with --plm_autoskip_bad_input.")
    p.add_argument("--plm_autoskip_aa_log2_floor", type=float, default=0.25, metavar="L",
                   help="Fold-change floor for the z-gate (default 0.25): an AA must ALSO "
                        "have log2(design%%/ref-global%%) >= L to trip, so a naturally-rare "
                        "AA at high z but a trivial %% does not falsely trip. Matches the "
                        "existing aa_quality_check |log2|>0.25 precedent.")
    # ---- F2: opt-in soft/graded plm_strength reduction (CLIFF stays default) ------
    p.add_argument("--plm_autoskip_soft", action="store_true", default=False,
                   help="Opt-in (default OFF => the CLIFF: a pathological seed forces "
                        "--plm_strength to 0). When set, REDUCE plm_strength gradually "
                        "instead: full strength at the trip threshold (severity 1) decaying "
                        "linearly to 0 at --plm_autoskip_soft_zero. NOTE: soft does NOT "
                        "rescue a pathological seed (at T~0.15 even strength 0.4 is ~602x "
                        "odds >> the 8x controller), so the cliff is the validated default; "
                        "soft is for cluster A/B comparison.")
    p.add_argument("--plm_autoskip_soft_zero", type=float, default=2.0, metavar="S",
                   help="Severity at which the soft curve reaches plm_strength 0 (default "
                        "2.0 = twice over the trip threshold). Only used with "
                        "--plm_autoskip_soft; S<=1 degrades to the cliff.")
    args = p.parse_args()
    if not math.isfinite(args.plm_strength):
        p.error("--plm_strength must be finite")
    if args.plm_strength < 0:
        p.error("--plm_strength must be >= 0 "
                "(negative would invert the PLM signal)")
    if args.plm_strength > 5.0:
        LOGGER.warning(
            "--plm_strength=%.2f is very large; PLM bias may dominate "
            "MPNN's structure-conditioned logits (collapse to PLM "
            "consensus). Typical range 0.5-2.0.", args.plm_strength,
        )
    # Seed-triage thresholds: GRAVY any finite value; fractions in (0, 1] (codex).
    if not math.isfinite(args.plm_autoskip_gravy):
        p.error("--plm_autoskip_gravy must be finite")
    for _tname, _tval in (("--plm_autoskip_max_aa_frac", args.plm_autoskip_max_aa_frac),
                          ("--plm_autoskip_hydrophobic_frac", args.plm_autoskip_hydrophobic_frac)):
        if not (math.isfinite(_tval) and 0.0 < _tval <= 1.0):
            p.error(f"{_tname} must be a fraction in (0, 1], got {_tval}")
    # F1 z-gate / F2 soft validation (only meaningful with --plm_autoskip_bad_input, but
    # validate unconditionally so a typo fails fast). zmax must be a finite POSITIVE
    # threshold (one-sided over-rep; <=0 is meaningless and would let the z=0 no-signal
    # sentinel trip — codex); log2_floor finite; soft_zero finite (S<=1 is allowed —
    # graded_plm_strength degrades it to the cliff).
    if args.plm_autoskip_aa_zmax is not None and not (
            math.isfinite(args.plm_autoskip_aa_zmax) and args.plm_autoskip_aa_zmax > 0.0):
        p.error("--plm_autoskip_aa_zmax must be a finite positive z-threshold "
                f"(one-sided over-representation), got {args.plm_autoskip_aa_zmax}")
    if not math.isfinite(args.plm_autoskip_aa_log2_floor):
        p.error("--plm_autoskip_aa_log2_floor must be finite")
    if not math.isfinite(args.plm_autoskip_soft_zero):
        p.error("--plm_autoskip_soft_zero must be finite")
    if args.adaptive_bias_max_odds is not None and not (
            math.isfinite(args.adaptive_bias_max_odds) and args.adaptive_bias_max_odds > 1.0):
        p.error("--adaptive_bias_max_odds must be a finite odds multiplier > 1.0 "
                f"(T*ln(X) must be positive), got {args.adaptive_bias_max_odds}")
    # CF-3a: odds-space joint bias-total clamp validation. The odds ceiling must be
    # > 1.0 (T*ln(X) must be a positive clamp), and it is mutually exclusive with the
    # raw-nats --bias_total_clamp (clearer than a silent precedence between them).
    if args.bias_total_clamp_odds is not None and not (
            math.isfinite(args.bias_total_clamp_odds) and args.bias_total_clamp_odds > 1.0):
        p.error("--bias_total_clamp_odds must be a finite odds multiplier > 1.0 "
                f"(T*ln(X) must be positive), got {args.bias_total_clamp_odds}")
    if args.bias_total_clamp is not None and args.bias_total_clamp_odds is not None:
        p.error("--bias_total_clamp and --bias_total_clamp_odds are mutually exclusive "
                "(one bounds the total bias in raw nats, the other in odds space); "
                "set only one.")
    # F3 (v1.4.0): --no_bias_total_clamp is the opt-OUT; pairing it with an explicit
    # clamp value is contradictory (disable vs set). Check on the USER's explicit values
    # BEFORE injecting the default below.
    if args.no_bias_total_clamp and (
            args.bias_total_clamp is not None or args.bias_total_clamp_odds is not None):
        p.error("--no_bias_total_clamp (opt out of the default 3-nat cap) is mutually "
                "exclusive with an explicit --bias_total_clamp / --bias_total_clamp_odds; "
                "pass the value alone to set a custom cap, or --no_bias_total_clamp alone "
                "to disable it.")
    # F3: apply the default 3-nat safety cap (deliberate default-path change). Runs AFTER
    # the mutual-exclusion checks so the injected default never participates in them.
    args.bias_total_clamp = _resolve_bias_total_clamp_default(
        bias_total_clamp=args.bias_total_clamp,
        bias_total_clamp_odds=args.bias_total_clamp_odds,
        no_clamp=args.no_bias_total_clamp,
    )
    # CF-3 coordinator: the joint controller odds ceiling must be a positive clamp
    # (T*ln(X) > 0 <=> X > 1.0). Validated even when the coordinator is off so a typo
    # fails fast rather than silently degrading the controller.
    if not (math.isfinite(args.controller_ceiling) and args.controller_ceiling > 1.0):
        p.error("--controller_ceiling must be a finite odds multiplier > 1.0 "
                f"(T*ln(X) must be positive), got {args.controller_ceiling}")
    # Shared-actuator guard (codex): axes that share an actuator (e.g. 'pi' shares the
    # charge D/E/K/R actuator with 'charge') DOUBLE-COUNT in the legacy additive sum and
    # re-create the multiplicative lock. The coordinator's max-not-sum is what makes a
    # shared actuator safe, so reject the footgun at startup unless it is on. (--help
    # exits in parse_args before this, so the import stays off the no-PYTHONPATH path.)
    if args.adaptive_bias_axes and not args.controller_coordinator:
        _gsel = [a.strip() for a in args.adaptive_bias_axes.split(",") if a.strip()]
        try:
            from protein_chisel.sampling.adaptive_bias import default_axes as _da_guard
            _gacts = [ax.actuator for ax in _da_guard(axes=_gsel)
                      if getattr(ax, "actuator", None)]
            _gdup = sorted({a for a in _gacts if _gacts.count(a) > 1})
        except ValueError:
            _gdup = []   # an unknown/duplicate name surfaces with the full message later
        if _gdup:
            p.error(
                "--adaptive_bias_axes selects axes that SHARE an actuator (%s) — they "
                "double-count in the legacy additive sum and re-create the "
                "multiplicative lock. Add --controller_coordinator (its max-not-sum "
                "makes shared actuators safe) or drop the redundant axis (e.g. 'pi' "
                "shares the charge D/E/K/R actuator with 'charge')." % ", ".join(_gdup))
    debug_short_test_override_msg = None
    if args.debug_short_test:
        if args.target_k != 5 or args.cycles != 3:
            debug_short_test_override_msg = (
                f"debug-short-test preset overriding target_k={args.target_k} -> 5 "
                f"and cycles={args.cycles} -> 3"
            )
        args.target_k = 5
        args.cycles = 3

    if args.verbose and args.quiet:
        raise SystemExit("--verbose and --quiet are mutually exclusive")
    log_level = logging.DEBUG if args.verbose else (
        logging.WARNING if args.quiet else logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    if debug_short_test_override_msg:
        # WARNING (not INFO): the user passed --target_k / --cycles
        # explicitly AND --debug-short-test. The preset wins silently;
        # surface the conflict loudly so production runs don't get
        # accidentally truncated.
        LOGGER.warning(debug_short_test_override_msg)

    # Resolve catalytic resnos for ANY scaffold, by priority:
    #   --catalytic_resnos override > seed REMARK 666 derivation > PTE builtin.
    # This makes the same driver/sbatch work on any scaffold in the design
    # campaign — even though the catalytic His/Lys/Glu sequence positions vary
    # between scaffolds (e.g. SEED1 LYS 157 vs SEED2 LYS 19), the REMARK 666
    # block (or the explicit override) records them and we adopt those
    # positions for the filter / fixed-residue / catres-aware code paths. The
    # builtin-fallback path (no override AND no REMARK 666) is loudly warned
    # inside _resolve_catalytic_resnos — those PTE positions are wrong off-PTE.
    # Design/catalytic chain (any-enzyme generalization). Validate a single,
    # non-space chain id, then set the module CHAIN global from args.chain so all
    # structural reads (H-bond/clash/preorg/interaction/SS/sequence/tunnel) target
    # it. Default 'A' leaves the global unchanged => byte-identical. Mirrors the
    # DEFAULT_CATRES/CATALYTIC_HIS_RESNOS global-set idiom below.
    global CHAIN
    _chain_arg = str(args.chain)
    if len(_chain_arg) != 1 or _chain_arg.isspace():
        p.error("--chain must be a single non-space chain id (e.g. 'A', 'B')")
    if _chain_arg != CHAIN:
        LOGGER.info("design/catalytic chain set to %r (was default %r)",
                    _chain_arg, CHAIN)
    CHAIN = _chain_arg

    global DEFAULT_CATRES, CATALYTIC_HIS_RESNOS
    _orig_default_catres, _orig_default_his = DEFAULT_CATRES, CATALYTIC_HIS_RESNOS
    derived_catres, derived_his, catres_source = _resolve_catalytic_resnos(
        args.catalytic_resnos, args.seed_pdb,
    )
    if catres_source != "builtin" and derived_catres != _orig_default_catres:
        LOGGER.info(
            "catalytic resnos resolved from %s: all_catres=%s (was PTE default "
            "%s); his_only=%s (was PTE default %s)",
            catres_source, derived_catres, _orig_default_catres,
            derived_his, _orig_default_his,
        )
    DEFAULT_CATRES = derived_catres
    CATALYTIC_HIS_RESNOS = derived_his

    # Catalytic-HIS H-bond requirement (any-enzyme generalization). Default ON
    # (byte-identical). --no_require_cat_his_hbond disables just this criterion.
    # Bonus auto-relax: if the resolved catalytic set has NO His at all, the
    # requirement is unsatisfiable and would reject every design, so turn it off
    # with a loud warning (an explicit --no_require_cat_his_hbond stays off too).
    global REQUIRE_CAT_HIS_HBOND
    REQUIRE_CAT_HIS_HBOND = not bool(args.no_require_cat_his_hbond)
    if REQUIRE_CAT_HIS_HBOND and catres_source != "builtin" and not derived_his:
        LOGGER.warning(
            "Resolved catalytic set (source=%s) contains NO His: the cat-HIS "
            "H-bond struct filter is unsatisfiable and would reject EVERY design. "
            "Auto-relaxing it for this run (equivalent to --no_require_cat_his_hbond). "
            "Pass --no_require_cat_his_hbond explicitly to silence this, or "
            "--catalytic_resnos with a His if your active site has one.",
            catres_source,
        )
        REQUIRE_CAT_HIS_HBOND = False

    # ---- Conserved-hbond + REMARK-transfer config (Features 1 & 2) -------
    global CONSERVE_HBONDS, CONSERVE_HBOND_PROB, CONSERVE_HBOND_MAX_DIST
    global CONSERVE_HBOND_MAX_ANGLE, CONSERVE_ANCHORS, CONSERVE_KEEP_CLASHING
    global CONSERVE_SEED_BASE, TRANSFER_REMARKS
    global CONSERVE_HBOND_DEPTH, CONSERVE_INTERACTION_TYPES, CONSERVE_SHELL_DECAY
    global CONSERVE_GROW_NETWORK
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.conserved_hbonds import (
        SUPPORTED_INTERACTION_TYPES, normalize_probability,
    )
    CONSERVE_HBONDS = bool(args.conserve_hbonds)
    try:
        CONSERVE_HBOND_PROB = normalize_probability(args.conserve_hbond_prob)
    except ValueError as exc:
        p.error(f"--conserve_hbond_prob {exc}")
    CONSERVE_HBOND_MAX_DIST = float(args.conserve_hbond_max_dist)
    CONSERVE_HBOND_MAX_ANGLE = float(args.conserve_hbond_max_angle)
    CONSERVE_ANCHORS = tuple(
        a.strip() for a in (args.conserve_anchors or "").split(",") if a.strip()
    )
    CONSERVE_KEEP_CLASHING = bool(args.conserve_keep_clashing)
    CONSERVE_HBOND_DEPTH = max(1, int(args.conserve_hbond_depth))
    CONSERVE_INTERACTION_TYPES = tuple(
        t.strip() for t in (args.conserve_interaction_types or "").split(",") if t.strip()
    ) or ("hbond",)
    bad_types = [t for t in CONSERVE_INTERACTION_TYPES
                 if t not in SUPPORTED_INTERACTION_TYPES]
    if bad_types:
        p.error(f"--conserve_interaction_types {bad_types} not supported; "
                f"only {list(SUPPORTED_INTERACTION_TYPES)} are wired")
    CONSERVE_SHELL_DECAY = float(args.conserve_shell_decay)
    CONSERVE_GROW_NETWORK = bool(args.conserve_grow_network)
    _CONSERVE_GROWN_ANCHORS.clear()  # reset cross-cycle accumulator per run
    TRANSFER_REMARKS = bool(args.transfer_remarks)
    if CONSERVE_HBONDS:
        CONSERVE_SEED_BASE = (
            args.conserve_seed if args.conserve_seed is not None
            else random.SystemRandom().randint(1, 2**31 - 1)
        )
        LOGGER.info(
            "H-bond conservation ON: p=%.2f anchors=%s depth=%d types=%s "
            "shell_decay=%.2f grow=%s keep_clashing=%s seed_base=%s "
            "(rerun with --conserve_seed %s for identical rolls)",
            CONSERVE_HBOND_PROB, CONSERVE_ANCHORS, CONSERVE_HBOND_DEPTH,
            CONSERVE_INTERACTION_TYPES, CONSERVE_SHELL_DECAY, CONSERVE_GROW_NETWORK,
            CONSERVE_KEEP_CLASHING, CONSERVE_SEED_BASE, CONSERVE_SEED_BASE,
        )

    # Include microseconds + PID to prevent concurrent-job collisions on
    # second-precision timestamps (real bug observed during a 4-job
    # parallel sweep — two jobs that started in the same second wrote
    # to the same run_dir and overwrote each other's outputs). The
    # visible prefix is intentionally short to reduce nested path length
    # for downstream tool workspaces.
    import os as _os_pid
    ts_micro = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]  # ms precision
    timestamp = f"{ts_micro}-pid{_os_pid.getpid()}"
    run_dir = args.out_root / f"run_{timestamp}"
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
    expert_names = [n.strip() for n in args.experts.split(",") if n.strip()]
    # Defensive: --plm_dtype must agree with the precompute manifest, else we'd load a
    # dtype-mismatched (or missing) artifact set. Fail fast with a clear message.
    try:
        _mf_dtype = json.loads((art / "manifest.json").read_text()).get("plm_dtype", "fp32")
    except Exception:
        _mf_dtype = "fp32"            # legacy artifacts w/o a manifest plm_dtype = fp32
    if _mf_dtype != args.plm_dtype:
        raise SystemExit(
            f"--plm_dtype {args.plm_dtype!r} != precompute manifest plm_dtype "
            f"{_mf_dtype!r} ({art}/manifest.json). Re-run precompute at the same dtype "
            f"or pass --plm_dtype {_mf_dtype}.")
    # Load the per-expert log-probs at the precompute dtype (fp32 = legacy name,
    # byte-identical; fp16/bf16 = the dtype-suffixed artifacts).
    _plm_suffix = "" if args.plm_dtype == "fp32" else f".{args.plm_dtype}"
    expert_logprobs = [np.load(art / f"{n}_log_probs{_plm_suffix}.npy")
                       for n in expert_names]
    # Back-compat bindings: downstream fitness/refresh code references the two
    # PLMs by name. For the default ["esmc","saprot"] these ARE the two experts;
    # with a custom set they bind to the first two so existing paths still run.
    log_probs_esmc = expert_logprobs[0]
    log_probs_saprot = (expert_logprobs[1] if len(expert_logprobs) > 1
                        else expert_logprobs[0])
    cached_base_bias = np.load(art / f"fusion_bias{_plm_suffix}.npy")
    _weights_npy = art / f"fusion_weights{_plm_suffix}.npy"
    cached_weights = np.load(_weights_npy) if _weights_npy.exists() else None
    LOGGER.info("loaded raw PLM log-probs for experts %s: L=%d",
                 expert_names, log_probs_esmc.shape[0])

    # ---- Metric/filter selection (add-on #7 — registry) -------------
    # Resolve the requested --metrics / --filters into the active catalog set.
    # This increment only RECORDS + LOGS the selection (default "all" =
    # today's full set); computation still flows through the stage_* chain, so
    # the default run is byte-identical. Capability-aware gating is additive.
    from protein_chisel.metrics import (
        available_metrics as _avail_metrics, resolve_metrics as _resolve_metrics,
    )
    from protein_chisel.metrics.base import ROLE_FILTER as _ROLE_FILTER
    _metrics_sel = _resolve_metrics(args.metrics)
    _filters_sel = _resolve_metrics(args.filters, role=_ROLE_FILTER)
    _metrics_is_all = args.metrics.strip().lower() in ("", "all", "*")
    _filters_is_all = args.filters.strip().lower() in ("", "all", "*")
    # Fail fast on a malformed EXPLICIT selection so a typo / wrong-role / empty
    # selection can't silently no-op. The default "all" can never hit any of these.
    _errs: list[str] = []
    _unknown = sorted(set(_metrics_sel.skipped_unknown + _filters_sel.skipped_unknown))
    if _unknown:
        _errs.append(f"unknown metric name(s) {_unknown}")
    if _filters_sel.skipped_wrong_role:
        _errs.append("--filters name(s) that are not filter metrics: "
                     f"{sorted(set(_filters_sel.skipped_wrong_role))}")
    if not _metrics_is_all and not _metrics_sel.selected:
        _errs.append(f"--metrics {args.metrics!r} selects no known metrics")
    if not _filters_is_all and not _filters_sel.selected:
        _errs.append(f"--filters {args.filters!r} selects no filter metrics")
    if not _metrics_is_all and _metrics_sel.selected and not _unknown:
        # A selected filter must also be computed (can't gate on a deselected
        # metric). Skipped when --metrics is itself empty/typo'd (its own error
        # already fires) to avoid a noisy orphan cascade.
        _m_names = set(_metrics_sel.names())
        _orphan = [n for n in _filters_sel.names() if n not in _m_names]
        if _orphan:
            _errs.append(f"--filters {sorted(_orphan)} not in the --metrics "
                         "selection (can't gate on a metric that isn't computed)")
    if _errs:
        raise SystemExit("metric selection error(s): " + "; ".join(_errs)
                         + f". Available metrics: {_avail_metrics()}")
    active_metric_names = _metrics_sel.names()
    # Gating handles (None at the default 'all' so gating is a literal no-op =>
    # byte-identical). selected_obj_labels filters the ranking basket; active_filters
    # gates which filters may drop designs.
    _selected_obj_labels = (None if _metrics_is_all
                            else frozenset(_metrics_sel.objective_labels()))
    _active_filters = (None if _filters_is_all
                       else frozenset(_filters_sel.names()))
    # Publish the gating selections as module globals for the stage/worker/helper
    # functions (None at the defaults => byte-identical).
    global _RANKING_LABEL_FILTER, _ACTIVE_FILTERS
    _RANKING_LABEL_FILTER = _selected_obj_labels
    _ACTIVE_FILTERS = _active_filters

    # ---- PoE backend wiring (opt-in; default 'bias' leaves these inert) -------
    global _POE_EMIT_INPUTS_DIR, _POE_SAMPLE_DIR, _POE_EXPERTS, _POE_LAMBDAS, _POE_TEMPERATURE
    _POE_EMIT_INPUTS_DIR = args.poe_emit_inputs
    _POE_SAMPLE_DIR = args.poe_output_dir
    _POE_TEMPERATURE = args.poe_temperature
    # Backend / mode must agree (avoid sampling with the wrong backend or mislabeling
    # provenance): --poe_output_dir (score-only) <=> --mpnn_backend poe; and a poe run
    # must be either emit-inputs or score-only.
    if (args.poe_output_dir is not None) != (args.mpnn_backend == "poe"):
        raise SystemExit("PoE: --poe_output_dir (score-only) and --mpnn_backend poe "
                         "must be used together")
    if args.mpnn_backend == "poe" and args.poe_output_dir is None \
            and args.poe_emit_inputs is None:
        raise SystemExit("--mpnn_backend poe needs --poe_output_dir (score-only) or "
                         "--poe_emit_inputs (emit); the shell wrapper sets these")
    if args.additional_experts:
        from protein_chisel.sampling.mpnn_backends import validate_expert_lambdas
        _exp, _lam = validate_expert_lambdas(
            args.additional_experts, args.additional_expert_lambdas)
        _POE_EXPERTS, _POE_LAMBDAS = tuple(_exp), tuple(_lam)
    elif args.mpnn_backend == "poe" and args.poe_output_dir is not None:
        # score-only PoE invocation must know which experts produced the pool
        raise SystemExit("--mpnn_backend poe (score-only) requires "
                         "--additional_experts/--additional_expert_lambdas")
    if args.poe_emit_inputs is not None:
        LOGGER.info("PoE emit-inputs mode: will write bias/fixed/omit JSONs for the "
                    "host PoE stage and exit at the first stage_sample.")
    if args.poe_output_dir is not None:
        LOGGER.info("PoE score-only mode: candidate pool from %s "
                    "(experts=%s lambdas=%s temp=%.3f); forcing a single cycle.",
                    args.poe_output_dir, list(_POE_EXPERTS), list(_POE_LAMBDAS),
                    _POE_TEMPERATURE)
    # Optional-stage gating: skip the whole tunnel stage only if NEITHER 'tunnel'
    # nor 'pkvf' is selected (they share stage_tunnel_metrics; we can't partially
    # skip within the stage). At --metrics all both are selected, so this is exactly
    # args.tunnel_metrics (byte-identical). Deselecting only 'pkvf' still computes it
    # but drops it from ranking; to skip the stage, deselect both.
    _tunnel_selected = ("tunnel" in active_metric_names) or ("pkvf" in active_metric_names)
    _tunnel_metrics_enabled = bool(args.tunnel_metrics) and _tunnel_selected
    if args.tunnel_metrics and not _tunnel_metrics_enabled:
        LOGGER.info("metric selection: tunnel stage SKIPPED (neither 'tunnel' nor "
                    "'pkvf' selected)")
    LOGGER.info(
        "metric selection: metrics=%r -> %d active %s | filters=%r -> %d gating %s",
        args.metrics, len(active_metric_names), active_metric_names,
        args.filters, len(_filters_sel.selected), _filters_sel.names())
    LOGGER.info("metric selection: stages=%s | ranking objectives=%s",
                _metrics_sel.stages(), _metrics_sel.objective_labels())

    # ---- Load PositionTable -----------------------------------------
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.io.schemas import PositionTable
    from protein_chisel.io.pdb import extract_sequence
    from protein_chisel.sampling.plm_fusion import (
        FusionConfig, decoupled_fitness_weights, fuse_experts,
    )

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
    # WS-E: opt-in ABSOLUTE per-class PLM weight overrides (default {} -> unchanged
    # -> byte-identical fusion). Fail-fast on a bad class name / value at startup.
    try:
        _plm_class_overrides = _parse_plm_class_strength(args.plm_class_strength)
    except ValueError as _exc:
        raise SystemExit(str(_exc))
    # ---- Seed triage (opt-in): drop the PLM bias on a pathological input scaffold ----
    # The PLM fusion is conditioned on the seed; on a hydrophobic / over-represented
    # scaffold it AMPLIFIES the bad composition (a near-lock at low T). When enabled and
    # the seed trips the triage, force plm_strength -> 0 BEFORE building the fusion so
    # LigandMPNN regenerates from structure + fixed residues. Default OFF => the fusion
    # below is byte-identical (imports + work happen only inside the opt-in branch).
    # Whole-protein over-rep AAs the z-gate flags (always defined; only populated when the
    # opt-in z-gate fires) — logged below; the composition cap (#29) handles them per-cycle.
    _triage_over_rep_aas: list[str] = []
    if args.plm_autoskip_bad_input:
        # ALL imports + work inside the try so a missing dep (e.g. Biopython for
        # protparam) degrades to "no skip" rather than crashing the run (codex).
        _triage_new_strength = args.plm_strength
        try:
            from protein_chisel.sampling.seed_triage import (
                assess_seed, graded_plm_strength, severity as _seed_severity,
            )
            from protein_chisel.io.pdb import extract_sequence as _triage_extract_seq
            from protein_chisel.filters.protparam import protparam_metrics as _triage_ppm
            _triage_seq = _triage_extract_seq(args.seed_pdb, chain=CHAIN)
            _triage_gravy = float(_triage_ppm(
                _triage_seq, ph=args.design_ph,
                n_term_pad=args.n_term_pad, c_term_pad=args.c_term_pad).gravy)
            # exclude_aas: drop EVERY hard-omitted canonical AA from the z-gate — flagging
            # over-representation of an AA the sampler cannot pick is meaningless and would
            # falsely amputate the PLM. Derived from the global omit the same way the WS-C
            # composition path does (codex: not just cysteine — --omit_AA WX must drop W too).
            _triage_exclude = _canonical_omit_aas(args.omit_AA)
            _seed_assessment = assess_seed(
                _triage_seq, _triage_gravy,
                gravy_max=args.plm_autoskip_gravy,
                max_aa_frac=args.plm_autoskip_max_aa_frac,
                hydrophobic_frac_max=args.plm_autoskip_hydrophobic_frac,
                aa_zmax=args.plm_autoskip_aa_zmax,
                aa_reference=args.aa_reference,
                aa_z_log2_floor=args.plm_autoskip_aa_log2_floor,
                exclude_aas=_triage_exclude)
            # Log the LOGIC of every z-gate trip: each tripped AA's z, log2, and reference
            # (the reasons list already carries all three — emit them prominently).
            if _seed_assessment.over_rep_aas:
                _triage_over_rep_aas = list(_seed_assessment.over_rep_aas)
                for _r in _seed_assessment.reasons:
                    if "z=" in _r and "log2=" in _r:
                        LOGGER.warning("SEED TRIAGE z-gate: %s", _r)
                LOGGER.warning(
                    "SEED TRIAGE z-gate flagged over-represented AAs %s (vs %s); they "
                    "contribute to the triage decision and are capped per-cycle by the "
                    "composition cap (--composition_pool_fallback / --aa_fraction_cap).",
                    "".join(_triage_over_rep_aas), args.aa_reference)
            # F2 policy: cliff (default) or soft/graded reduction of plm_strength.
            _triage_new_strength = graded_plm_strength(
                _seed_assessment, enabled=True,
                current_plm_strength=args.plm_strength,
                soft=args.plm_autoskip_soft,
                soft_zero=args.plm_autoskip_soft_zero)
            if _triage_new_strength != args.plm_strength:
                _sev = _seed_severity(_seed_assessment)
                _mode = ("soft (soft_zero=%.2f)" % args.plm_autoskip_soft_zero
                         ) if args.plm_autoskip_soft else "cliff"
                LOGGER.warning(
                    "SEED TRIAGE [%s]: input scaffold is pathological (%s) -> "
                    "severity %.3f -> --plm_strength %.2f -> %.3f (PLM bias would "
                    "amplify the seed; LigandMPNN leans on structure + fixed residues).",
                    _mode, "; ".join(_seed_assessment.reasons), _sev,
                    args.plm_strength, _triage_new_strength)
        except Exception as _triage_exc:               # advisory; never crash the run
            LOGGER.warning("seed triage skipped (%s)", _triage_exc)
        args.plm_strength = _triage_new_strength
    fusion_cfg = FusionConfig(global_strength=args.plm_strength)
    if _plm_class_overrides:
        fusion_cfg.class_weights.update(_plm_class_overrides)
        LOGGER.info("PLM per-class strength overrides (absolute): %s",
                    _plm_class_overrides)
    # Default ["esmc","saprot"] (no per-expert knobs) -> fuse_experts delegates to
    # the legacy fuse_plm_logits via its N=2 fast-path -> byte-identical bias
    # (see tests/sampling/test_fuse_experts.py). >=3 experts use the N-way path.
    fusion_res = fuse_experts(
        expert_logprobs, position_classes,
        config=fusion_cfg, expert_names=expert_names,
    )
    base_bias = fusion_res.bias
    # WS-E PLM decouple: for --plm_strength > 0, reuse the EXACT strength-scaled
    # fusion weights (byte-identical — the fused-mean RANK is scale-invariant in
    # plm_strength, but the scalar is not bit-exact, so recomputing at 1.0 would
    # perturb the fitness TSV column by ~1 ULP). Only at strength == 0 — where the
    # sampling bias is off AND the legacy weights are all-zero (every design ties at
    # 0, silently zeroing the weight-2.0 fitness objective) — substitute the
    # structural strength-1.0 weights to rescue a meaningful rank. The (L, 2) fitness
    # path is the 2-expert default; >2-expert runs keep the legacy None unchanged
    # (the decoupled helper would return (L, N), which the fitness gather rejects).
    weights_per_position = fusion_res.weights_per_position
    if args.plm_strength == 0 and weights_per_position is not None:
        weights_per_position = decoupled_fitness_weights(
            expert_logprobs, position_classes, config=fusion_cfg,
            expert_names=expert_names,
        )
    fusion_dir = run_dir / "fusion_runtime"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    np.save(fusion_dir / "base_bias.npy", base_bias)
    # Legacy snapshots (populated for the default 2-expert case); guarded so a
    # custom >=3-expert set doesn't crash on the None legacy fields.
    if fusion_res.weights_per_position is not None:
        # Save the BIAS-strength weights (byte-identical write-only diagnostic); the
        # fitness path uses the decoupled strength=1.0 weights in `weights_per_position`.
        np.save(fusion_dir / "weights_per_position.npy",
                fusion_res.weights_per_position)
    if fusion_res.log_odds_esmc is not None:
        np.save(fusion_dir / "log_odds_esmc.npy", fusion_res.log_odds_esmc)
    if fusion_res.log_odds_saprot is not None:
        np.save(fusion_dir / "log_odds_saprot.npy", fusion_res.log_odds_saprot)
    if fusion_res.weights_per_expert is not None and len(expert_names) != 2:
        np.save(fusion_dir / "weights_per_expert.npy", fusion_res.weights_per_expert)
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
    # ---- Run provenance (version control) ----------------------------
    # Records experts (+versions), fusion math, backend, and conserve-network
    # settings so this run is reproducible/auditable. expert_versions come from
    # the precompute manifest when available. Additive artifact (no PDB change).
    from protein_chisel.provenance import RunProvenance
    from protein_chisel.paths import POE_MPNN_COMMIT
    _expert_versions = {}
    try:
        _pre_manifest = json.loads((art / "manifest.json").read_text())
        _expert_versions = _pre_manifest.get("expert_versions", {})
    except Exception:
        pass
    run_provenance = RunProvenance(
        experts=list(expert_names),
        expert_versions=_expert_versions,
        fusion_version=getattr(fusion_cfg, "version", "fusion-v1"),
        conserve_hbonds=CONSERVE_HBONDS,
        conserve_depth=CONSERVE_HBOND_DEPTH,
        conserve_interaction_types=list(CONSERVE_INTERACTION_TYPES),
        conserve_grow_network=CONSERVE_GROW_NETWORK,
        conserve_seed_base=(CONSERVE_SEED_BASE if CONSERVE_HBONDS else None),
        metrics_selection=args.metrics,
        filters_selection=args.filters,
        active_metrics=active_metric_names,
        mpnn_backend=args.mpnn_backend,
        extra=({"poe_experts": list(_POE_EXPERTS),
                "poe_expert_lambdas": list(_POE_LAMBDAS),
                "poe_commit": POE_MPNN_COMMIT}
               if args.mpnn_backend == "poe" else {}),
    )
    run_provenance.write_json(run_dir / "provenance.json")
    LOGGER.info("provenance -> %s (experts=%s, fusion=%s)",
                 run_dir / "provenance.json", run_provenance.experts,
                 run_provenance.fusion_version)
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

    # ---- Input-scaffold hydrophobicity warning (logging only) --------
    # A scaffold whose whole-sequence GRAVY is already strongly positive folds into
    # a hydrophobic blob with little polar surface; designs will track the backbone
    # and tend to fail the solubility filters. --adaptive_bias can steer the surface
    # but cannot make a hydrophobic FOLD soluble — surface this up front. Reuses the
    # protparam GRAVY; never changes selection.
    try:
        from protein_chisel.filters.protparam import protparam_metrics as _pp_metrics
        _seed_pp = _pp_metrics(wt_seq, ph=args.design_ph,
                               n_term_pad=args.n_term_pad, c_term_pad=args.c_term_pad)
        _seed_gravy = float(_seed_pp.gravy)
        _seed_charge = float(_seed_pp.charge_at_pH_full_HH)
        if _seed_gravy > 0.4:
            LOGGER.warning(
                "INPUT SCAFFOLD is hydrophobic: seed GRAVY=%+.2f (>+0.4), "
                "net_charge=%+.1f. Designs will track this backbone and likely fail "
                "the solubility/SAP filters; %s can steer the exposed surface but "
                "cannot make a hydrophobic fold soluble. Consider pre-filtering "
                "inputs by GRAVY upstream.", _seed_gravy, _seed_charge,
                "--adaptive_bias" if args.adaptive_bias else "the adaptive controller")
    except Exception:                              # pragma: no cover - advisory only
        _seed_gravy = None
        _seed_charge = None

    # Compute ligand geometry summary ONCE — scaffold-invariant. The
    # min_projected_radius is the relevant tunnel-fit threshold.
    try:
        from protein_chisel.tools.ligand_geometry import ligand_geometry_from_pdb
        ligand_geometry_summary = ligand_geometry_from_pdb(args.seed_pdb)
        LOGGER.info(
            "ligand geometry: resname=%s n_heavy=%d n_metals=%d "
            "Rg=%.2f min_proj_radius=%.2f bbox_diag=%.2f",
            ligand_geometry_summary["ligand_resname"],
            ligand_geometry_summary["n_heavy_atoms"],
            ligand_geometry_summary["n_metal_atoms"],
            ligand_geometry_summary["radius_of_gyration"],
            ligand_geometry_summary["min_projected_radius"],
            ligand_geometry_summary["bounding_box_diagonal"],
        )
    except Exception as exc:
        LOGGER.warning("ligand_geometry_from_pdb failed (%s); "
                        "tunnel ligand-fit gate will be disabled", exc)
        ligand_geometry_summary = {
            "ligand_resname": None, "min_projected_radius": None,
        }
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
    # WS-C: seed-derived SOFT_BIAS map (0-indexed body position -> AAs to
    # down-weight), LOCAL hits only (whole-protein composition hits excluded —
    # those are the suppress-all / fraction-cap levers' job). Used by run_cycle
    # only as the cycle-0 BOOTSTRAP; cycles 1+ rebuild the map from the survivor
    # pool. Inert + free to compute unless --composition_soft_bias is set.
    expression_soft_bias = wt_eng.soft_bias_per_residue(
        max_span_frac=_SOFT_BIAS_MAX_SPAN_FRAC,
    )
    # F1 consequence (b): the z-gate's over-rep AAs are HANDLED PER-CYCLE by the existing
    # composition cap (#29 --composition_pool_fallback + --aa_fraction_cap + suppress-overrep),
    # which caps whatever is over-represented in each cycle's sampled pool. The earlier
    # cycle-0 SEED bootstrap (forcing those AAs down at all positions from cycle 0) was
    # REMOVED: cluster validation showed it over-committed and degraded hard seeds (Chigh
    # GRAVY -0.01 -> +0.46, Clow -0.57 -> +0.03), while #29 alone steers them correctly. The
    # z-gate's value is the triage CONTRIBUTION (above) + the prominent LOG (above); the cap
    # is #29's job, not a one-shot seed forcing.
    if args.composition_soft_bias:
        LOGGER.info(
            "expression-engine SOFT_BIAS seed-bootstrap map: %d local positions "
            "(cycle 0 only; cycles 1+ rebuild from the survivor pool; %.2f nats "
            "each)",
            len(expression_soft_bias), args.composition_soft_bias_nats,
        )

    # Graded clash-aware bias replacing the previous hard-omit. Per the
    # rotamer-feasibility audit (commit logs + scripts/audit_clash_omits.py),
    # no (clash-prone-pos, bulky-AA) pair has >50% clashing rotamers in
    # a 9-rotamer grid stub, so the previous hard-omit was unjustified.
    # Now: add a per-position per-AA bias proportional to the clash %
    # to the base PLM-fusion bias (max -20 nats at 100% clash, 0 at 0%).
    # MPNN can still pick a "clash-prone" AA when other context strongly
    # favors it; the filter-time severe-clash check (1.5 A) catches the
    # remaining hard failures.
    clash_bias, clash_telem = compute_graded_clash_bias(
        seed_pdb=args.seed_pdb,
        position_table_df=pt.df,
        fixed_resnos=DEFAULT_CATRES,
        chain=CHAIN,
        cb_clearance_threshold=5.0,
        bulky_aas=_CLASH_BULKY_AAS,
    )
    LOGGER.info(
        "graded clash bias: %d positions biased, mean magnitude=%.3f nats",
        clash_telem["n_positions_biased"], float(np.abs(clash_bias).mean()),
    )
    base_bias = base_bias + clash_bias   # fusion baseline, carried into every cycle
    omit_AA_per_residue = expression_omit
    # ---- WS-G: opt-in tunnel-lining hard-omit (merged ONLY when on) -----------
    # Merge only inside the `if` so a no-flag run leaves expression_omit byte-for-byte
    # untouched (merge_omit_dicts re-sorts AA strings, so even a `{}` merge is not a
    # guaranteed no-op — codex). The lining set is the seed is_tunnel_lining
    # annotation (the SAME source WS-D's surface scope uses).
    if args.omit_tunnel_lining:
        _lining = _read_seed_tunnel_lining(seed_tunnel_path)
        try:
            _tunnel_omit = _build_tunnel_lining_omit(
                sorted(_lining), CHAIN, args.omit_tunnel_lining_aas,
                fixed_resnos=DEFAULT_CATRES,
            )
        except ValueError as _exc:
            raise SystemExit(str(_exc))
        if _tunnel_omit:
            omit_AA_per_residue = merge_omit_dicts(expression_omit, _tunnel_omit)
            LOGGER.info("WS-G omit_tunnel_lining: hard-omit %r at %d non-catalytic "
                        "lining positions; merged omit now %d positions",
                        args.omit_tunnel_lining_aas, len(_tunnel_omit),
                        len(omit_AA_per_residue))
            if args.throat_feedback:
                LOGGER.warning("WS-G omit_tunnel_lining + throat_feedback both ON: at "
                               "lining∩throat positions the hard omit shadows the soft "
                               "throat bias for those AAs (harmless; the omit wins).")
        else:
            LOGGER.warning("WS-G omit_tunnel_lining set but no tunnel-lining positions "
                           "found (empty/failed seed annotation) — no-op this run.")
    LOGGER.info("structural omit_AA (from rule engine only): %s", omit_AA_per_residue)

    # ---- Cycle schedule ---------------------------------------------
    cycle_builder = debug_short_test_cycles if args.debug_short_test else default_cycles
    cycles = cycle_builder(
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
        net_charge_min=args.net_charge_min,
        net_charge_max=args.net_charge_max,
        sap_max_threshold=args.sap_max_threshold,
        instability_max=args.instability_max,
        gravy_min=args.gravy_min,
        gravy_max=args.gravy_max,
        aliphatic_min=args.aliphatic_min,
        boman_max=args.boman_max,
    )
    if args.debug_short_test:
        LOGGER.info(
            "debug-short-test preset active: cycle_samples=%s, target_k=%d",
            [c.n_samples for c in cycles], args.target_k,
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
    if args.poe_output_dir is not None:
        # PoE one-shot: the candidate pool is pre-sampled by the host PoE stage, so
        # there is no per-cycle resampling/bias-refinement to iterate — one cycle.
        cycles = cycles[:1]
        LOGGER.info("PoE score-only: forcing a single cycle (one-shot pool).")
        # WS-C: suppress-all-overrep and the fraction cap act on the PREVIOUS
        # cycle's survivor pool, which never exists in a one-shot PoE run — so they
        # are silent no-ops here. (The composition soft-bias still applies via its
        # cycle-0 seed bootstrap.) Surface the dead flags rather than fail silently.
        _poe_dead = [
            name for name, on in (
                ("--composition_suppress_all_overrep", args.composition_suppress_all_overrep),
                ("--aa_fraction_cap", args.aa_fraction_cap is not None),
            ) if on
        ]
        if _poe_dead:
            LOGGER.warning(
                "PoE one-shot backend: %s have NO effect (they steer the next "
                "cycle from the survivor pool, and PoE runs a single cycle-0 pool "
                "with no survivors). Use --mpnn_backend bias for these levers.",
                " and ".join(_poe_dead),
            )
    # ---- Side-chain-context schedule (opt-in) --------------------------------
    # Override each cycle's use_side_chain_context from a per-cycle 0/1 schedule
    # (ON early for clash avoidance, OFF late for first-shell diversity). Applied
    # ONCE here AFTER construction / truncation / PoE forcing so it matches the
    # FINAL cycle count. None (flag absent) => the uniform --use_side_chain_context
    # value built into every cycle is left untouched => byte-identical.
    if args.use_side_chain_context_schedule is not None:
        _apply_scc_schedule(cycles, args.use_side_chain_context_schedule)
        LOGGER.info(
            "use_side_chain_context schedule applied (broadcast/truncate to %d "
            "cycles): per-cycle sc = %s",
            len(cycles), [c.use_side_chain_context for c in cycles],
        )
    # ---- WS-E: sampling temperature floor (opt-in) ---------------------------
    # Raise any cycle's sampling temperature to at least the floor, applied ONCE
    # here (after construction / truncation / PoE forcing) so the sampler AND the
    # logged/telemetered temperature agree. At T≈0.15 even a 0.5-nat bias is ~28x
    # (near-deterministic); a floor ~0.3 restores genuine multinomial diversity.
    # None => byte-identical (the schedule is untouched). The floor OVERRIDES the
    # annealing schedule (a floor above the max cycle temp disables annealing).
    if args.sampling_temperature_floor is not None:
        _floored = []
        for cyc in cycles:
            if cyc.sampling_temperature < args.sampling_temperature_floor:
                _floored.append((cyc.cycle_idx, cyc.sampling_temperature))
                cyc.sampling_temperature = args.sampling_temperature_floor
        if _floored:
            LOGGER.info("sampling_temperature_floor=%.3f raised cycles %s",
                        args.sampling_temperature_floor,
                        [f"{i}:{t:.3f}->{args.sampling_temperature_floor:.3f}"
                         for i, t in _floored])
        if args.poe_output_dir is not None:
            LOGGER.warning("--sampling_temperature_floor has NO effect under the PoE "
                           "backend (it already sampled at POE_TEMPERATURE).")
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
    all_seq_stage_rows: list[pd.DataFrame] = []
    all_pdb_maps: dict[str, Path] = {}
    fitness_cache: dict = {}
    survivors_prev: Optional[pd.DataFrame] = None
    fixed_resnos = list(DEFAULT_CATRES)
    # Throat-blocker bias delta carried forward across cycles. None for
    # cycle 0 (no prior data); populated from cycle k for cycle k+1.
    throat_bias_prev: Optional[np.ndarray] = None
    # Feature #29: the previous cycle's full SAMPLED (pre-band-filter) pool,
    # carried forward so run_cycle can fall back to it for the WS-C cap /
    # class-balance / soft-bias when --composition_pool_fallback is set and the
    # survivor pool is empty. None for cycle 0 (no prior pool) and whenever the
    # flag is off (it is simply never read in run_cycle). This is the SAME pool
    # the adaptive controller measures (``seq_stage_df`` below).
    seq_stage_pool_prev: Optional[pd.DataFrame] = None

    # ---- Adaptive solubility-bias controller (opt-in) ----------------
    # The controller LOGIC lives here in the loop (where the full per-cycle
    # candidate pool is available); run_cycle only APPLIES the carried biases.
    # All three carried objects are None until the controller produces them, so
    # when --adaptive_bias is off run_cycle receives None and the path is
    # byte-identical.
    adaptive_state: Optional[dict] = None          # {axis: AxisState.to_dict()}
    adaptive_global: Optional[dict] = None         # raw global per-AA bias to apply next
    adaptive_delta: Optional[np.ndarray] = None    # (L,20) surface delta to apply next
    _ab_sasa = None
    _ab_fixed_idx: set = set()
    _ab_cfg = None
    _ab_surface_mask = None        # WS-D non_tunnel_surface mask (None => legacy)
    _ab_charge_band = None         # WS-D --adaptive_charge_band override (None => cycle)
    _ab_axes_sel = None            # WS-D --adaptive_bias_axes selector (None => default)
    # pI controller band/target (default_axes' own defaults until set from --pi_min/max
    # below) — defined at function scope so every default_axes() call site is safe.
    _pi_band = (float(args.pi_min), float(args.pi_max))
    _pi_target = min(args.pi_min + 0.5, (args.pi_min + args.pi_max) / 2.0)
    if args.adaptive_bias:
        from protein_chisel.sampling.adaptive_bias import (
            AdaptiveBiasConfig, compute_adaptive_bias, default_axes,
        )
        # OPT-IN control-law damping bundle (default OFF => the four damping fields
        # keep their no-op defaults => byte-identical to the legacy controller). When
        # --controller_damping is set: act on an EWMA of the pool mean (alpha=0.5),
        # add a derivative-on-measurement term (gain 0.5*the integral gain), slew-limit
        # |Δu| to 0.15*max_nats/cycle, and use the soft 'ramp' deadband.
        _ab_damp = dict(measurement_ewma_alpha=0.5,
                        derivative_gain=0.5 * args.adaptive_bias_gain,
                        slew_limit_frac=0.15,
                        deadband_mode="ramp") if args.controller_damping else {}
        _ab_cfg = AdaptiveBiasConfig(
            gain=args.adaptive_bias_gain, max_nats=args.adaptive_bias_max_nats,
            carry=args.adaptive_bias_carry, t_min=args.adaptive_bias_tmin,
            f_min=args.adaptive_bias_fmin, min_n=args.adaptive_bias_min_n,
            mode=args.adaptive_bias_mode, max_odds=args.adaptive_bias_max_odds,
            # CF-3 coordinator (opt-in; both default to the byte-identical legacy path
            # when --controller_coordinator is absent).
            coordinator=args.controller_coordinator,
            controller_ceiling=args.controller_ceiling,
            **_ab_damp,
        )
        try:
            _ab_r2s = dict(zip(pt.df["resno"].astype(int),
                               pt.df["sasa_sc_fraction"].astype(float)))
            _ab_sasa = np.array([_ab_r2s.get(int(r), 0.0) for r in protein_resnos],
                                dtype=float)
        except Exception:                          # pragma: no cover - defensive
            _ab_sasa = None
        _ab_r2i = {int(r): i for i, r in enumerate(protein_resnos)}
        _ab_fixed_idx = {_ab_r2i[int(r)] for r in fixed_resnos if int(r) in _ab_r2i}
        # ---- WS-D: opt-in charge band + axes selector (None => today's behavior) ----
        if args.adaptive_charge_band:
            try:
                _ab_charge_band = _parse_charge_band_arg(args.adaptive_charge_band)
            except ValueError as _exc:
                raise SystemExit(str(_exc))
        _ab_axes_sel = ([a.strip() for a in args.adaptive_bias_axes.split(",") if a.strip()]
                        if args.adaptive_bias_axes else None)
        # Fail-fast on a bad band/axis selector at STARTUP — the per-cycle controller
        # is defensively wrapped, so without this an invalid band (lo>=hi) or unknown
        # axis name would silently degrade the run to unbiased every cycle.
        try:
            default_axes(charge_band=_ab_charge_band, axes=_ab_axes_sel,
                         pi_band=_pi_band, pi_target=_pi_target)
        except ValueError as _exc:
            raise SystemExit(f"adaptive-bias config error: {_exc}")
        # ---- WS-D: non_tunnel_surface scope mask (structure-invariant; built once).
        # None unless --adaptive_surface_sasa_gate is set, so build_surface_delta keeps
        # the legacy distal_surface gate => byte-identical. Tunnel-lining comes from the
        # seed annotation; the throat-band source is not yet wired (frozenset()).
        if args.adaptive_surface_sasa_gate is not None:
            from protein_chisel.sampling.adaptive_bias import surface_scope
            # Shared single source of truth (also WS-G's omit source): resno set
            # from the seed is_tunnel_lining annotation, mapped to 0-based indices.
            _ab_tunnel_lining = {
                _ab_r2i[r] for r in _read_seed_tunnel_lining(seed_tunnel_path)
                if r in _ab_r2i
            }
            _ab_surface_mask = surface_scope(
                L=base_bias.shape[0], position_classes=position_classes,
                sasa_fraction=_ab_sasa, fixed_idx=_ab_fixed_idx,
                sasa_gate=args.adaptive_surface_sasa_gate,
                tunnel_lining_idx=frozenset(_ab_tunnel_lining),
                throat_band_idx=frozenset(),
            )
            LOGGER.info("adaptive surface scope: non_tunnel_surface (sasa_gate=%.2f) => "
                        "%d steerable positions (%d tunnel-lining, %d fixed excluded)",
                        args.adaptive_surface_sasa_gate, int(_ab_surface_mask.sum()),
                        len(_ab_tunnel_lining), len(_ab_fixed_idx))
        LOGGER.info("adaptive-bias controller ENABLED (gain=%.2f max=%.2f carry=%.2f "
                    "tmin=%.1f fmin=%.2f min_n=%d mode=%s)",
                    _ab_cfg.gain, _ab_cfg.max_nats, _ab_cfg.carry, _ab_cfg.t_min,
                    _ab_cfg.f_min, _ab_cfg.min_n, _ab_cfg.mode)
        if args.controller_damping:
            LOGGER.info("controller DAMPING ENABLED (ewma_alpha=%.2f derivative_gain=%.3f "
                        "slew_limit_frac=%.2f deadband_mode=%s)",
                        _ab_cfg.measurement_ewma_alpha, _ab_cfg.derivative_gain,
                        _ab_cfg.slew_limit_frac, _ab_cfg.deadband_mode)
        if args.controller_coordinator:
            LOGGER.info("controller COORDINATOR ENABLED (CF-2/CF-3): joint odds budget "
                        "%.0fx (controller share), nested inside ~%.0fx whole-stack "
                        "ceiling; shared actuators collapse by sign-selected max/sum.",
                        args.controller_ceiling, 1.0e3)
        # Optional cycle-0 warm-start from the input scaffold's own properties.
        if args.adaptive_bias_seed_from_input and _seed_gravy is not None:
            from protein_chisel.sampling.adaptive_bias import seed_warmstart
            _ab_seed_axes = default_axes(
                gravy_band=(args.gravy_min, args.gravy_max),
                net_charge_band=(cycles[0].net_charge_min, cycles[0].net_charge_max),
                deadband_frac=args.adaptive_bias_deadband,
                charge_band=_ab_charge_band, axes=_ab_axes_sel,
                pi_band=_pi_band, pi_target=_pi_target,
            )
            adaptive_global, adaptive_delta, adaptive_state, _ab_seed_tele = seed_warmstart(
                seed_metrics={"gravy": _seed_gravy, "net_charge_full_HH": _seed_charge},
                axes=_ab_seed_axes, cfg=_ab_cfg, L=base_bias.shape[0],
                position_classes=position_classes, sasa_fraction=_ab_sasa,
                fixed_idx=_ab_fixed_idx, surface_mask=_ab_surface_mask,
                # The warm-start bias is APPLIED at cycle 0, so the coordinator (when on)
                # sizes its budget at cycle 0's T. Only consumed under the coordinator;
                # legacy path ignores it => byte-identical.
                temperature=cycles[0].sampling_temperature,
            )
            adaptive_global = adaptive_global or None
            adaptive_delta = adaptive_delta if np.any(adaptive_delta) else None
            LOGGER.info("adaptive-bias seed warm-start from input: global=%s surface=%s",
                        adaptive_global or "{}", _ab_seed_tele["seed_warmstart"])

    # Per-cycle metrics snapshot — written to run_dir/cycle_metrics.tsv at the
    # end so the user can grep / plot how filter populations and quality
    # change across cycles. Always written; --verbose adds more granular
    # debug output to the log itself but the TSV is always there.
    cycle_metric_rows: list[dict] = []

    for _cyc_pos, cyc in enumerate(cycles):
        cycle_dir = run_dir / f"cycle_{cyc.cycle_idx:02d}"
        # The adaptive controller measures THIS cycle's pool but its bias is APPLIED at
        # the NEXT cycle (a possibly lower, annealed T). The CF-3 coordinator sizes its
        # odds budget at the APPLICATION-cycle T (math review §2.4); on the last cycle
        # there is no next application, so fall back to this cycle's T (harmless).
        _next_apply_T = (cycles[_cyc_pos + 1].sampling_temperature
                         if _cyc_pos + 1 < len(cycles) else cyc.sampling_temperature)
        ranked_df, pdb_map, cyc_telem = run_cycle(
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
            aa_reference=args.aa_reference,
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
            sap_corrected=args.sap_corrected,
            composition_suppress_all_overrep=args.composition_suppress_all_overrep,
            aa_fraction_cap=args.aa_fraction_cap,
            composition_soft_bias=args.composition_soft_bias,
            composition_soft_bias_nats=args.composition_soft_bias_nats,
            expression_soft_bias=expression_soft_bias,
            composition_pool_fallback=args.composition_pool_fallback,
            composition_fallback_pool=seq_stage_pool_prev,
            bias_total_clamp=args.bias_total_clamp,
            bias_total_clamp_odds=args.bias_total_clamp_odds,
            n_term_pad=args.n_term_pad,
            c_term_pad=args.c_term_pad,
            omit_M_at_pos1=not args.no_omit_M_at_pos1,
            tunnel_metrics_enabled=_tunnel_metrics_enabled,
            tunnel_hard_gate=args.tunnel_hard_gate,
            ligand_min_radius=ligand_geometry_summary.get("min_projected_radius"),
            ligand_resname=ligand_geometry_summary.get("ligand_resname"),
            throat_bias_prev=throat_bias_prev,
            throat_bias_decay=args.throat_feedback_decay,
            adaptive_bias_global=adaptive_global,
            adaptive_bias_delta=adaptive_delta,
            # The coordinator's nested-ceiling clamp at the sampler is meaningful only
            # when the controller is actually running (--adaptive_bias); without it the
            # controller buckets are empty and the nesting would just clamp the PLM
            # stack. Gate on BOTH so --controller_coordinator alone is a true no-op.
            controller_coordinator=(args.controller_coordinator and args.adaptive_bias),
            controller_ceiling=args.controller_ceiling,
            # The graded-clash bias (VETO tier) so the nested whole-stack clamp can
            # bypass it (a ban must never be diluted by the total ceiling).
            clash_bias=clash_bias,
        )
        # Carry throat-bias forward to next cycle (None if disabled or
        # this cycle didn't produce one).
        throat_bias_prev = (
            cyc_telem.get("throat_bias_delta")
            if (args.throat_feedback and cyc_telem)
            else None
        )
        seq_stage_df = _load_cycle_seq_stage_pool(cycle_dir, cyc.cycle_idx)
        if len(seq_stage_df) > 0:
            all_seq_stage_rows.append(seq_stage_df)
        # Feature #29: carry THIS cycle's full sampled pool forward as the next
        # cycle's composition fallback source (only when the opt-in flag is set,
        # so the default path holds no extra state). This is the SAME sampled pool
        # the adaptive controller measures just below.
        if args.composition_pool_fallback:
            seq_stage_pool_prev = seq_stage_df if len(seq_stage_df) > 0 else None

        # ---- Adaptive controller: measure THIS cycle's full candidate pool and
        # produce the bias to apply NEXT cycle (mirrors the throat carry pattern).
        # Defensively wrapped: a controller failure must NEVER abort the design run
        # (degrade to the legacy UNBIASED path and continue). Byte-identical on the
        # success path; the except only triggers on an unanticipated edge.
        if args.adaptive_bias and _ab_cfg is not None:
            try:
                ab_axes = default_axes(
                    gravy_band=(
                        cyc.gravy_min if args.strategy == "annealing" else args.gravy_min,
                        cyc.gravy_max if args.strategy == "annealing" else args.gravy_max),
                    net_charge_band=(cyc.net_charge_min, cyc.net_charge_max),
                    deadband_frac=args.adaptive_bias_deadband,
                    charge_band=_ab_charge_band, axes=_ab_axes_sel,
                    pi_band=_pi_band, pi_target=_pi_target,
                )
                from protein_chisel.sampling.adaptive_bias import hydrophobic_over_rep_mask
                _ab_overrep = (hydrophobic_over_rep_mask(
                    seq_stage_df["sequence"].astype(str).tolist(),
                    reference=args.aa_reference)
                    if "sequence" in seq_stage_df.columns else None)
                # CF-3 owns bias-application timing: under --controller_coordinator the
                # joint odds budget is sized at the APPLICATION-cycle T (the bias is
                # applied NEXT cycle). The legacy CF-1 odds clamp keeps using THIS
                # cycle's T so existing --adaptive_bias_max_odds runs stay byte-identical
                # (the budget is the only thing that should track the application T).
                _ctrl_T = (_next_apply_T if args.controller_coordinator
                           else cyc.sampling_temperature)
                ab_res = compute_adaptive_bias(
                    pool_df=seq_stage_df, axes=ab_axes, cfg=_ab_cfg, state=adaptive_state,
                    L=base_bias.shape[0], position_classes=position_classes,
                    sasa_fraction=_ab_sasa, fixed_idx=_ab_fixed_idx,
                    over_rep_mask=_ab_overrep, surface_mask=_ab_surface_mask,
                    temperature=_ctrl_T,
                )
                adaptive_state = ab_res.new_state
                adaptive_global = ab_res.controller_global or None
                adaptive_delta = (ab_res.per_position_delta
                                  if np.any(ab_res.per_position_delta) else None)
                try:
                    ab_bias_dir = cycle_dir / "00_bias"
                    ab_bias_dir.mkdir(parents=True, exist_ok=True)
                    with open(ab_bias_dir / "adaptive_bias_telemetry.json", "w") as fh:
                        json.dump(ab_res.telemetry, fh, indent=2, default=str)
                except Exception:                  # pragma: no cover - telemetry only
                    pass
                # CF-5: opt-in verbose controller trace. Default OFF => byte-identical
                # (nothing below runs, controller_trace is not imported). Advisory only:
                # a write/format error is logged and swallowed, never stopping the run.
                if args.controller_verbose:
                    try:
                        from protein_chisel.sampling.controller_trace import (
                            TRACE_COLUMNS, controller_trace_rows,
                            format_controller_report_lines,
                        )
                        _axes_by_name = {ax.name: ax for ax in ab_axes}
                        _trace_rows = controller_trace_rows(
                            ab_res.outcomes, _axes_by_name,
                            cycle_idx=cyc.cycle_idx,
                            temperature=cyc.sampling_temperature,
                        )
                        # append-only long-format TSV at the RUN ROOT (one row per axis
                        # per cycle across the whole run); header written once.
                        _trace_tsv = run_dir / "controller_trace.tsv"
                        _need_header = not _trace_tsv.exists()
                        with open(_trace_tsv, "a") as _tfh:
                            if _need_header:
                                _tfh.write("\t".join(TRACE_COLUMNS) + "\n")
                            for _row in _trace_rows:
                                _tfh.write("\t".join(
                                    str(_row[_c]) for _c in TRACE_COLUMNS) + "\n")
                        LOGGER.info("cycle %d CONTROLLER REPORT (T=%.3f):",
                                    cyc.cycle_idx, cyc.sampling_temperature)
                        for _line in format_controller_report_lines(_trace_rows):
                            LOGGER.info("%s", _line)
                    except Exception:              # pragma: no cover - advisory only
                        LOGGER.exception(
                            "cycle %d controller_verbose trace failed (advisory; "
                            "run continues)", cyc.cycle_idx)
                _open = [o.name for o in ab_res.outcomes if o.gate_open]
                LOGGER.info("cycle %d adaptive controller: axes_active=%s global=%s "
                            "surface_positions=%d", cyc.cycle_idx, _open or "none",
                            adaptive_global or "{}",
                            ab_res.telemetry.get("n_surface_positions_touched", 0))
            except Exception:
                LOGGER.exception(
                    "cycle %d adaptive controller FAILED; continuing UNBIASED "
                    "(legacy path) for the rest of this run", cyc.cycle_idx)
                adaptive_global, adaptive_delta = None, None
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
                    select_specs_by_label,
                )
                cycle_specs = apply_cli_overrides(
                    DEFAULT_METRIC_SPECS,
                    {**parse_kv_string(args.rank_weights),
                     **cyc.topsis_weight_overrides},
                    parse_kv_string(args.rank_targets),
                )
                cycle_specs = select_specs_by_label(cycle_specs, _selected_obj_labels)
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
        elif args.composition_pool_fallback:
            # Feature #29: this cycle collapsed (zero ranked survivors). Clear the
            # carried survivor pool so the NEXT cycle correctly sees "no survivors
            # from the previous cycle" and routes WS-C through the sampled fallback
            # (seq_stage_pool_prev, set above) instead of a STALE older survivor
            # pool. Gated on the opt-in flag, so the default path keeps today's
            # behavior (survivors_prev unchanged on the empty-ranked path) =>
            # byte-identical. (codex review.)
            survivors_prev = None
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
                ("fpocket__bottleneck_radius", "bottleneck"),
                ("fpocket__volume", "volume"),
                ("net_charge_no_HIS", "charge"),
                ("pi", "pi"),
            ):
                if col in ranked_df.columns:
                    vals = pd.to_numeric(ranked_df[col], errors="coerce").dropna()
                    if len(vals) > 0:
                        cycle_row[f"{prefix}_mean"] = float(vals.mean())
                        cycle_row[f"{prefix}_min"] = float(vals.min())
                        cycle_row[f"{prefix}_max"] = float(vals.max())

            # Sequence-space diversity + AA composition (cheap; useful for
            # diagnosing whether bias / consensus reinforcement is collapsing
            # or expanding the population)
            if "sequence" in ranked_df.columns:
                seqs = ranked_df["sequence"].dropna().astype(str).tolist()
                if len(seqs) >= 2:
                    cycle_row["n_unique_seqs"] = len(set(seqs))
                    # full-sequence pairwise hamming (sample to bound cost)
                    sample = seqs if len(seqs) <= 80 else random.sample(seqs, 80)
                    arr = np.array([[ord(c) for c in s] for s in sample],
                                    dtype=np.uint8)
                    if arr.size > 0:
                        n = arr.shape[0]
                        # Vectorized pairwise hamming
                        diff_counts = []
                        for i in range(n):
                            d = (arr[i+1:] != arr[i]).sum(axis=1)
                            diff_counts.extend(d.tolist())
                        if diff_counts:
                            d_arr = np.array(diff_counts, dtype=float)
                            cycle_row["pairwise_hamming_mean"] = float(d_arr.mean())
                            cycle_row["pairwise_hamming_std"]  = float(d_arr.std(ddof=0))
                    # AA composition (mean count + Shannon entropy)
                    aa_counts: dict[str, list[int]] = {}
                    for aa in "ACDEFGHIKLMNPQRSTVWY":
                        aa_counts[aa] = [s.count(aa) for s in seqs]
                    cycle_row["n_unique_aas_observed"] = sum(
                        1 for aa, cs in aa_counts.items() if max(cs) > 0
                    )
                    # Shannon entropy of mean-AA-frequency distribution (bits)
                    means = np.array([np.mean(cs) for cs in aa_counts.values()])
                    if means.sum() > 0:
                        p = means / means.sum()
                        p = p[p > 0]
                        cycle_row["aa_shannon_entropy_bits"] = float(
                            -(p * np.log2(p)).sum()
                        )
                    # Active-site (first-shell) diversity using fixed_resnos.
                    # Catalytic positions are FIXED so they shouldn't vary
                    # across designs; we measure variation at all OTHER
                    # designable positions for "effective design diversity".
                    L_seq = len(sample[0]) if sample else 0
                    fixed_idx = {r - 1 for r in fixed_resnos
                                  if 1 <= r <= L_seq}
                    designable_idx = [i for i in range(L_seq) if i not in fixed_idx]
                    if designable_idx and arr.size > 0:
                        # Mean number of unique AAs per designable position
                        unique_per_pos = [
                            len(set(arr[:, i].tolist())) for i in designable_idx
                        ]
                        cycle_row["unique_aas_per_pos_mean"] = float(np.mean(unique_per_pos))
                        cycle_row["unique_aas_per_pos_max"]  = float(np.max(unique_per_pos))
                        cycle_row["pct_pos_with_2plus_aas"]  = float(
                            100.0 * sum(1 for u in unique_per_pos if u >= 2) / len(unique_per_pos)
                        )
        cycle_metric_rows.append(cycle_row)
        if args.verbose:
            LOGGER.info("cycle %d metrics snapshot: %s",
                        cyc.cycle_idx,
                        ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in cycle_row.items()))

    # ---- Final pool: concat + dedup --------------------------------
    final_dir = run_dir / "final_topk"
    final_dir.mkdir(parents=True, exist_ok=True)
    from protein_chisel.scoring.multi_objective import parse_kv_string
    rank_weights = parse_kv_string(args.rank_weights)
    rank_targets = parse_kv_string(args.rank_targets)
    final_cycle_cfg = cycles[-1]

    if all_ranked:
        from protein_chisel.sampling.fitness_score import deduplicate_by_sequence

        pool_prefilter = pd.concat(all_ranked, ignore_index=True)
        pool_prefilter = deduplicate_by_sequence(pool_prefilter)
        druggability_min = cycles[0].fpocket_druggability_min
        n_before = len(pool_prefilter)
        fpocket_status = (
            pool_prefilter["fpocket__status"].astype(str)
            if "fpocket__status" in pool_prefilter.columns
            else pd.Series(["unknown"] * len(pool_prefilter), index=pool_prefilter.index)
        )
        fpocket_druggability = (
            pd.to_numeric(pool_prefilter["fpocket__druggability"], errors="coerce")
            if "fpocket__druggability" in pool_prefilter.columns
            else pd.Series([float("nan")] * len(pool_prefilter), index=pool_prefilter.index)
        )
        pass_mask = pd.Series([True] * len(pool_prefilter), index=pool_prefilter.index)
        if ("fpocket__druggability" in pool_prefilter.columns and druggability_min > 0
                and _filter_active("fpocket")):
            pass_mask = fpocket_druggability >= druggability_min
            LOGGER.info(
                "fpocket-druggability filter: %d -> %d (cutoff=%.2f)",
                n_before, int(pass_mask.sum()), druggability_min,
            )
        # ---- Multi-objective ranking over the full pool ---------------
        # TOPSIS over a configurable basket of metrics with CLI-tunable
        # weights / targets. Replaces the legacy 2-key (fitness,
        # alpha_radius) sort. The legacy sort is still computed and
        # written as `legacy_rank_score` for back-compat / debugging.
        from protein_chisel.scoring.multi_objective import (
            DEFAULT_METRIC_SPECS, apply_cli_overrides, compute_topsis_scores_v2,
            parse_kv_string, select_diverse_topk_two_axis, select_specs_by_label,
        )
        active_specs = apply_cli_overrides(
            DEFAULT_METRIC_SPECS, rank_weights, rank_targets,
        )
        active_specs = select_specs_by_label(active_specs, _selected_obj_labels)
        scores, used_specs, _debug = compute_topsis_scores_v2(
            pool_prefilter, active_specs,
        )
        pool_prefilter["mo_topsis"] = scores
        # Legacy diagnostic — keep next to mo_topsis so we can compare.
        pool_prefilter["legacy_rank_score"] = (
            pool_prefilter["fitness__logp_fused_mean"].rank(ascending=False)
            + pool_prefilter["fpocket__mean_alpha_sphere_radius"].rank(ascending=True)
        )
        pool_prefilter["selection__hard_final_filter_passed"] = pass_mask.astype(bool)
        if druggability_min > 0 and "fpocket__druggability" in pool_prefilter.columns:
            if _filter_active("fpocket"):
                pool_prefilter["selection__fpocket_gap"] = np.where(
                    fpocket_druggability.notna(),
                    np.maximum(0.0, float(druggability_min) - fpocket_druggability),
                    np.inf,
                )
            else:
                # fpocket filter deselected (--filters): no druggability penalty
                # in the backfill ordering either.
                pool_prefilter["selection__fpocket_gap"] = 0.0
        else:
            pool_prefilter["selection__fpocket_gap"] = 0.0

        near_miss_mask = (~pass_mask) & fpocket_status.eq("ok") & fpocket_druggability.notna()
        failed_mask = ~pass_mask & ~near_miss_mask
        pool_prefilter["selection__bucket"] = np.select(
            [pass_mask, near_miss_mask],
            ["passed_final_filters", "fpocket_near_miss"],
            default="fpocket_failed_or_missing",
        )
        pool_prefilter["selection__bucket_priority"] = np.select(
            [pass_mask, near_miss_mask],
            [0, 1],
            default=2,
        ).astype(int)

        strict_pool = pool_prefilter[pass_mask].copy()
        strict_pool = strict_pool.sort_values(
            ["mo_topsis", "fitness__logp_fused_mean"],
            ascending=[False, False], na_position="last",
        ).reset_index(drop=True)

        if len(strict_pool) == 0 and not args.final_filter_backfill:
            LOGGER.error(
                "All %d designs filtered out by fpocket-druggability >= %.2f. "
                "Writing empty final artifacts and exiting cleanly. "
                "Diagnose: (a) inspect cycle_NN/05_fpocket/ranked.tsv "
                "fpocket__status column for tool-failure counts, "
                "(b) grep slurm-stderr for 'fpocket failed for ... rc=' "
                "to see specific subprocess return codes.",
                n_before, druggability_min,
            )
            _write_empty_final_artifacts(
                final_dir=final_dir,
                run_dir=run_dir,
                template_df=pool_prefilter,
                status="EMPTY_POOL_AFTER_FPOCKET_FILTER",
                reason=(
                    f"All {n_before} designs failed the final "
                    f"fpocket-druggability >= {druggability_min:.2f} cutoff"
                ),
                extra_meta={
                    "n_before_filter": n_before,
                    "druggability_cutoff": float(druggability_min),
                },
            )
            return

        if args.final_filter_backfill and druggability_min > 0:
            primary_pool = pool_prefilter[~failed_mask].copy()
            primary_pool = primary_pool.sort_values(
                [
                    "selection__bucket_priority",
                    "selection__fpocket_gap",
                    "mo_topsis",
                    "fitness__logp_fused_mean",
                ],
                ascending=[True, True, False, False],
                na_position="last",
            ).reset_index(drop=True)
            failed_pool = pool_prefilter[failed_mask].copy()
            failed_pool = failed_pool.sort_values(
                ["mo_topsis", "fitness__logp_fused_mean"],
                ascending=[False, False],
                na_position="last",
            ).reset_index(drop=True)
            seq_backfill_pool, seq_backfill_specs = _build_seq_stage_backfill_pool(
                seq_stage_rows=all_seq_stage_rows,
                existing_pool=pool_prefilter,
                log_probs_esmc=log_probs_esmc,
                log_probs_saprot=log_probs_saprot,
                weights_per_position=weights_per_position,
                fitness_cache=fitness_cache,
                rank_weights=rank_weights,
                rank_targets=rank_targets,
            )
            rescue_deficit = max(0, args.target_k - len(primary_pool))
            rescued_seq_backfill_pool = seq_backfill_pool.head(0).copy()
            raw_seq_backfill_pool = seq_backfill_pool.copy()
            if rescue_deficit > 0 and len(seq_backfill_pool) > 0:
                shortlist_n = min(
                    len(seq_backfill_pool),
                    _deferred_rescue_shortlist_size(
                        deficit=rescue_deficit,
                        target_k=args.target_k,
                    ),
                )
                rescue_shortlist = seq_backfill_pool.head(shortlist_n).copy()
                rescued_seq_backfill_pool = _deferred_rescue_score_candidates(
                    candidates_df=rescue_shortlist,
                    pdb_map=all_pdb_maps,
                    out_dir=final_dir / "_deferred_rescue_seq_backfill",
                    catalytic_his_resnos=CATALYTIC_HIS_RESNOS,
                    # XXX assumes catalytic == fixed: works today because
                    # `fixed_resnos = list(DEFAULT_CATRES)` upstream, so
                    # the two lists are identical. If a future scaffold
                    # ever decouples "fixed by design" from "catalytic",
                    # this call will silently use the wrong set for
                    # tunnel / fpocket active-site anchoring. Pass an
                    # explicit catalytic_resnos= value at that point.
                    catalytic_resnos=fixed_resnos,
                    fixed_resnos=fixed_resnos,
                    sap_max_threshold=final_cycle_cfg.sap_max_threshold,
                    clash_filter=final_cycle_cfg.clash_filter,
                    clash_severe_distance=final_cycle_cfg.clash_severe_distance,
                    seed_dfi_metrics=seed_dfi_metrics,
                    tunnel_metrics_enabled=_tunnel_metrics_enabled,
                    ligand_min_radius=ligand_geometry_summary.get("min_projected_radius"),
                    ligand_resname=ligand_geometry_summary.get("ligand_resname"),
                    log_probs_esmc=log_probs_esmc,
                    log_probs_saprot=log_probs_saprot,
                    weights_per_position=weights_per_position,
                    fitness_cache=fitness_cache,
                    wt_fitness=wt_fitness,
                    rank_weights=rank_weights,
                    rank_targets=rank_targets,
                    fpocket_druggability_min=druggability_min,
                    chain=CHAIN,
                )
                rescued_source_ids = set(
                    _candidate_source_id_series(rescued_seq_backfill_pool).tolist(),
                )
                raw_source_ids = _candidate_source_id_series(seq_backfill_pool)
                raw_seq_backfill_pool = seq_backfill_pool.loc[
                    ~raw_source_ids.isin(rescued_source_ids)
                ].copy().reset_index(drop=True)
                LOGGER.info(
                    "deferred_rescue: shortlisted %d seq-stage backfill "
                    "candidates (deficit=%d) -> rescued=%d raw_remaining=%d",
                    shortlist_n, rescue_deficit,
                    len(rescued_seq_backfill_pool), len(raw_seq_backfill_pool),
                )
            pool_parts = [primary_pool]
            if len(rescued_seq_backfill_pool) > 0:
                pool_parts.append(rescued_seq_backfill_pool)
            if len(failed_pool) > 0:
                pool_parts.append(failed_pool)
            if len(raw_seq_backfill_pool) > 0:
                pool_parts.append(raw_seq_backfill_pool)
            pool = pd.concat(pool_parts, ignore_index=True)
            sort_pairs = [
                ("selection__bucket_priority", True),
                ("selection__seq_backfill_reason_count", True),
                ("selection__seq_backfill_numeric_gap", True),
                ("selection__fpocket_gap", True),
                ("mo_topsis", False),
                ("fitness__logp_fused_mean", False),
            ]
            pool_sort_cols = [c for c, _asc in sort_pairs if c in pool.columns]
            pool_sort_asc = [asc for c, asc in sort_pairs if c in pool.columns]
            if pool_sort_cols:
                pool = pool.sort_values(
                    pool_sort_cols,
                    ascending=pool_sort_asc,
                    na_position="last",
                ).reset_index(drop=True)
            LOGGER.info(
                "final filter backfill: strict_pass=%d near_miss=%d "
                "fpocket_failed_or_missing=%d rescued_seq_stage=%d "
                "raw_seq_stage=%d "
                "target_k=%d",
                int(pass_mask.sum()),
                int(near_miss_mask.sum()),
                int(failed_mask.sum()),
                len(rescued_seq_backfill_pool),
                len(raw_seq_backfill_pool),
                args.target_k,
            )
        else:
            pool = strict_pool

        pool.to_csv(final_dir / "all_survivors.tsv", sep="\t", index=False)
        LOGGER.info("final pool: %d unique survivors across %d cycles",
                     len(pool_prefilter), len(all_ranked))
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

        top, topk_telemetry = _select_diverse_topk_progressive(
            pool,
            target_k=args.target_k,
            min_hamming_full=args.min_hamming,
            primary_sphere_positions=primary_positions,
            min_hamming_active=args.min_hamming_active,
        )
        materializable_candidates = int(
            _candidate_source_id_series(pool).map(
                lambda sid: bool((src := all_pdb_maps.get(sid)) and src.is_file()),
            ).sum(),
        )
        if (
            args.final_filter_backfill
            and materializable_candidates < args.target_k
        ):
            LOGGER.warning(
                "final_filter_backfill requested %d outputs, but only %d "
                "unique candidates had a materializable source PDB after "
                "cycle union/dedup. Final output will be capped accordingly.",
                args.target_k, materializable_candidates,
            )

        top, materialize_telemetry = _select_materializable_topk(
            selected_df=top,
            candidate_pool=pool,
            pdb_map=all_pdb_maps,
            target_k=args.target_k,
            min_hamming_full=args.min_hamming,
            primary_sphere_positions=primary_positions,
            min_hamming_active=args.min_hamming_active,
            allow_backfill=args.final_filter_backfill,
        )
        n_ids_renamed = int(materialize_telemetry.get("n_ids_renamed", 0))
        if n_ids_renamed:
            LOGGER.warning(
                "stage_diverse_topk: renamed %d duplicate final ids across "
                "cycles to keep exported PDBs collision-free",
                n_ids_renamed,
            )
        if materialize_telemetry.get("dropped_missing_source_pdbs", 0):
            LOGGER.warning(
                "stage_diverse_topk: dropped %d selected rows with missing "
                "source PDBs before writing final artifacts",
                materialize_telemetry["dropped_missing_source_pdbs"],
            )
        if materialize_telemetry.get("refill_rounds"):
            LOGGER.info(
                "stage_diverse_topk: materialization refill rounds=%s",
                materialize_telemetry["refill_rounds"],
            )

        deferred_rescue_mask = (
            ~top.get(
                "selection__deferred_rescue_attempted",
                pd.Series([False] * len(top), index=top.index),
            ).fillna(False).astype(bool)
        ) & (
            top.get(
                "selection__bucket",
                pd.Series([""] * len(top), index=top.index),
            ).fillna("").astype(str).str.startswith("seq_stage_")
            | top.get(
                "fpocket__status",
                pd.Series(["failed"] * len(top), index=top.index),
            ).fillna("failed").astype(str).isin(["not_run", "failed", "missing"])
        )
        if deferred_rescue_mask.any():
            deferred_top = top.loc[deferred_rescue_mask].copy()
            LOGGER.info(
                "deferred_rescue: final top-K still contains %d raw seq-stage "
                "rows without full downstream scoring; rescuing before export",
                len(deferred_top),
            )
            rescued_top = _deferred_rescue_score_candidates(
                candidates_df=deferred_top,
                pdb_map=all_pdb_maps,
                out_dir=final_dir / "_deferred_rescue_selected_topk",
                catalytic_his_resnos=CATALYTIC_HIS_RESNOS,
                # XXX assumes catalytic == fixed: works today because
                # `fixed_resnos = list(DEFAULT_CATRES)` upstream, so the
                # two lists are identical. If a future scaffold ever
                # decouples "fixed by design" from "catalytic", this
                # call will silently use the wrong set for tunnel/
                # fpocket active-site anchoring. Pass an explicit
                # catalytic_resnos= value at that point.
                catalytic_resnos=fixed_resnos,
                fixed_resnos=fixed_resnos,
                sap_max_threshold=final_cycle_cfg.sap_max_threshold,
                clash_filter=final_cycle_cfg.clash_filter,
                clash_severe_distance=final_cycle_cfg.clash_severe_distance,
                seed_dfi_metrics=seed_dfi_metrics,
                tunnel_metrics_enabled=_tunnel_metrics_enabled,
                ligand_min_radius=ligand_geometry_summary.get("min_projected_radius"),
                ligand_resname=ligand_geometry_summary.get("ligand_resname"),
                log_probs_esmc=log_probs_esmc,
                log_probs_saprot=log_probs_saprot,
                weights_per_position=weights_per_position,
                fitness_cache=fitness_cache,
                wt_fitness=wt_fitness,
                rank_weights=rank_weights,
                rank_targets=rank_targets,
                fpocket_druggability_min=druggability_min,
                chain=CHAIN,
            )
            top = _overlay_rows_by_id(top, rescued_top)

        top = _apply_solubility_veto(
            top,
            enabled=args.ship_solubility_veto,
            gravy_min=(final_cycle_cfg.gravy_min if args.strategy == "annealing"
                       else args.gravy_min),
            gravy_max=(final_cycle_cfg.gravy_max if args.strategy == "annealing"
                       else args.gravy_max),
            net_charge_min=final_cycle_cfg.net_charge_min,
            net_charge_max=final_cycle_cfg.net_charge_max,
        )
        requested_topk_rows = len(top)
        topk_tsv, top = _write_final_topk_artifacts(
            top=top,
            final_dir=final_dir,
            pdb_map=all_pdb_maps,
            seed_pdb=args.seed_pdb,
        )
        copied_pdbs = len(top)
        if copied_pdbs != requested_topk_rows:
            LOGGER.error(
                "stage_diverse_topk: wrote %d rows to topk.tsv but only %d "
                "PDBs were copied to final_topk/topk_pdbs",
                requested_topk_rows, copied_pdbs,
            )
        if args.final_filter_backfill and copied_pdbs < args.target_k:
            LOGGER.warning(
                "stage_diverse_topk: final materialized top-K underfilled "
                "(%d/%d requested)%s. materializable_candidates=%d",
                copied_pdbs, args.target_k,
                " (solubility veto active — underfill is expected)"
                if args.ship_solubility_veto else "",
                materializable_candidates,
            )
        if (
            args.final_filter_backfill
            and not args.ship_solubility_veto
            and copied_pdbs < args.target_k
            and materializable_candidates >= args.target_k
        ):
            LOGGER.error(
                "stage_diverse_topk: backfill was enabled and at least %d "
                "materializable candidates existed, but only %d final PDBs "
                "were written. Inspect the preceding export errors.",
                args.target_k, copied_pdbs,
            )

        if args.copy_input_structure_into_out_dir:
            # User-visible signal that we're about to spend ~20-40 sec
            # re-running struct + tunnel + fpocket on the seed PDB so it
            # appears as an `input_reference` row in the shipped TSV
            # alongside the designs. Disable with
            # --copy_input_structure_into_out_dir false if walltime is
            # tight (e.g. large sweeps).
            LOGGER.info(
                "input-structure scoring: running struct + tunnel + fpocket "
                "on seed PDB so it appears as an input_reference row in "
                "chiseled_design_metrics.tsv (~20-40 sec extra walltime; "
                "disable with --copy_input_structure_into_out_dir false)"
            )
            final_cycle_cfg = cycles[-1]
            input_reference_df = _build_input_reference_row(
                template_df=top,
                seed_pdb=args.seed_pdb,
                wt_seq=wt_seq,
                wt_fitness=wt_fitness,
                expression_result=wt_eng,
                seed_ss_reduced=ss.ss_reduced,
                seed_sasa=seed_sasa,
                position_classes=position_classes,
                protein_resnos=protein_resnos,
                catalytic_resnos=DEFAULT_CATRES,
                fixed_resnos=fixed_resnos,
                design_ph=7.5,
                n_term_pad="",
                c_term_pad="",
                net_charge_max=final_cycle_cfg.net_charge_max,
                net_charge_min=final_cycle_cfg.net_charge_min,
                instability_max=final_cycle_cfg.instability_max,
                gravy_min=final_cycle_cfg.gravy_min,
                gravy_max=final_cycle_cfg.gravy_max,
                aliphatic_min=final_cycle_cfg.aliphatic_min,
                boman_max=final_cycle_cfg.boman_max,
                pi_min=final_cycle_cfg.pi_min,
                pi_max=final_cycle_cfg.pi_max,
                clash_filter=final_cycle_cfg.clash_filter,
                clash_severe_distance=final_cycle_cfg.clash_severe_distance,
                sap_max_threshold=final_cycle_cfg.sap_max_threshold,
                sap_corrected=args.sap_corrected,
                seed_dfi_metrics=seed_dfi_metrics,
                tunnel_metrics_enabled=_tunnel_metrics_enabled,
                ligand_min_radius=ligand_geometry_summary.get("min_projected_radius"),
                ligand_resname=ligand_geometry_summary.get("ligand_resname"),
                log_probs_esmc=log_probs_esmc,
                log_probs_saprot=log_probs_saprot,
                weights_per_position=weights_per_position,
                final_dir=final_dir,
                fpocket_druggability_min=druggability_min,
            )
            input_reference_path = final_dir / "input_reference.tsv"
            input_reference_df.to_csv(input_reference_path, sep="\t", index=False)
            LOGGER.info(
                "input reference metrics: wrote %s (id=%s)",
                input_reference_path, args.seed_pdb.stem,
            )
        if "selection__bucket" in top.columns:
            bucket_counts = top["selection__bucket"].value_counts().to_dict()
            LOGGER.info(
                "stage_diverse_topk: selected %d / %d "
                "(target=%d, min_hamming=%d, min_hamming_active=%d, "
                "bucket_counts=%s, unconstrained_fill=%d)",
                len(top), len(pool), args.target_k,
                args.min_hamming, args.min_hamming_active,
                bucket_counts, topk_telemetry.get("filled_without_hamming", 0),
            )
        else:
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
        seq_backfill_pool, used_specs = _build_seq_stage_backfill_pool(
            seq_stage_rows=all_seq_stage_rows,
            existing_pool=None,
            log_probs_esmc=log_probs_esmc,
            log_probs_saprot=log_probs_saprot,
            weights_per_position=weights_per_position,
            fitness_cache=fitness_cache,
            rank_weights=rank_weights,
            rank_targets=rank_targets,
        )
        if args.final_filter_backfill and len(seq_backfill_pool) > 0:
            LOGGER.warning(
                "final: zero ranked survivors across all cycles; falling back "
                "to seq-stage backfill pool (n=%d)",
                len(seq_backfill_pool),
            )
            pool_prefilter = seq_backfill_pool.copy()
            shortlist_n = min(
                len(seq_backfill_pool),
                _deferred_rescue_shortlist_size(
                    deficit=args.target_k,
                    target_k=args.target_k,
                ),
            )
            rescue_shortlist = seq_backfill_pool.head(shortlist_n).copy()
            rescued_seq_backfill_pool = _deferred_rescue_score_candidates(
                candidates_df=rescue_shortlist,
                pdb_map=all_pdb_maps,
                out_dir=final_dir / "_deferred_rescue_seq_backfill",
                catalytic_his_resnos=CATALYTIC_HIS_RESNOS,
                # XXX assumes catalytic == fixed: works today because
                # `fixed_resnos = list(DEFAULT_CATRES)` upstream, so the
                # two lists are identical. If a future scaffold ever
                # decouples "fixed by design" from "catalytic", this
                # call will silently use the wrong set for tunnel/
                # fpocket active-site anchoring. Pass an explicit
                # catalytic_resnos= value at that point.
                catalytic_resnos=fixed_resnos,
                fixed_resnos=fixed_resnos,
                sap_max_threshold=final_cycle_cfg.sap_max_threshold,
                clash_filter=final_cycle_cfg.clash_filter,
                clash_severe_distance=final_cycle_cfg.clash_severe_distance,
                seed_dfi_metrics=seed_dfi_metrics,
                tunnel_metrics_enabled=_tunnel_metrics_enabled,
                ligand_min_radius=ligand_geometry_summary.get("min_projected_radius"),
                ligand_resname=ligand_geometry_summary.get("ligand_resname"),
                log_probs_esmc=log_probs_esmc,
                log_probs_saprot=log_probs_saprot,
                weights_per_position=weights_per_position,
                fitness_cache=fitness_cache,
                wt_fitness=wt_fitness,
                rank_weights=rank_weights,
                rank_targets=rank_targets,
                fpocket_druggability_min=cycles[0].fpocket_druggability_min,
                chain=CHAIN,
            )
            rescued_source_ids = set(
                _candidate_source_id_series(rescued_seq_backfill_pool).tolist(),
            )
            raw_source_ids = _candidate_source_id_series(seq_backfill_pool)
            raw_seq_backfill_pool = seq_backfill_pool.loc[
                ~raw_source_ids.isin(rescued_source_ids)
            ].copy().reset_index(drop=True)
            LOGGER.info(
                "deferred_rescue: shortlisted %d seq-stage backfill "
                "candidates (deficit=%d) -> rescued=%d raw_remaining=%d",
                shortlist_n, args.target_k,
                len(rescued_seq_backfill_pool), len(raw_seq_backfill_pool),
            )
            pool_parts = []
            if len(rescued_seq_backfill_pool) > 0:
                pool_parts.append(rescued_seq_backfill_pool)
            if len(raw_seq_backfill_pool) > 0:
                pool_parts.append(raw_seq_backfill_pool)
            pool = pd.concat(pool_parts, ignore_index=True)
            sort_pairs = [
                ("selection__bucket_priority", True),
                ("selection__seq_backfill_reason_count", True),
                ("selection__seq_backfill_numeric_gap", True),
                ("selection__fpocket_gap", True),
                ("mo_topsis", False),
                ("fitness__logp_fused_mean", False),
            ]
            pool_sort_cols = [c for c, _asc in sort_pairs if c in pool.columns]
            pool_sort_asc = [asc for c, asc in sort_pairs if c in pool.columns]
            if pool_sort_cols:
                pool = pool.sort_values(
                    pool_sort_cols,
                    ascending=pool_sort_asc,
                    na_position="last",
                ).reset_index(drop=True)
            pool.to_csv(final_dir / "all_survivors.tsv", sep="\t", index=False)
            LOGGER.info(
                "final pool: %d seq-stage backfill candidates across %d cycles",
                len(pool_prefilter), len(cycles),
            )
            LOGGER.info("multi-objective ranking applied with %d active specs:",
                         len(used_specs))
            for s in used_specs:
                LOGGER.info("  %-25s direction=%-7s weight=%.2f target=%s",
                             s.label, s.direction, s.weight, s.target)
            LOGGER.info("top-5 mo_topsis scores: %s",
                         pool["mo_topsis"].head(5).round(3).tolist())

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

            top, topk_telemetry = _select_diverse_topk_progressive(
                pool,
                target_k=args.target_k,
                min_hamming_full=args.min_hamming,
                primary_sphere_positions=primary_positions,
                min_hamming_active=args.min_hamming_active,
            )
            materializable_candidates = int(
                _candidate_source_id_series(pool).map(
                    lambda sid: bool((src := all_pdb_maps.get(sid)) and src.is_file()),
                ).sum(),
            )
            top, materialize_telemetry = _select_materializable_topk(
                selected_df=top,
                candidate_pool=pool,
                pdb_map=all_pdb_maps,
                target_k=args.target_k,
                min_hamming_full=args.min_hamming,
                primary_sphere_positions=primary_positions,
                min_hamming_active=args.min_hamming_active,
                allow_backfill=args.final_filter_backfill,
            )
            deferred_rescue_mask = (
                ~top.get(
                    "selection__deferred_rescue_attempted",
                    pd.Series([False] * len(top), index=top.index),
                ).fillna(False).astype(bool)
            ) & (
                top.get(
                    "selection__bucket",
                    pd.Series([""] * len(top), index=top.index),
                ).fillna("").astype(str).str.startswith("seq_stage_")
                | top.get(
                    "fpocket__status",
                    pd.Series(["failed"] * len(top), index=top.index),
                ).fillna("failed").astype(str).isin(["not_run", "failed", "missing"])
            )
            if deferred_rescue_mask.any():
                deferred_top = top.loc[deferred_rescue_mask].copy()
                LOGGER.info(
                    "deferred_rescue: final top-K still contains %d raw seq-stage "
                    "rows without full downstream scoring; rescuing before export",
                    len(deferred_top),
                )
                rescued_top = _deferred_rescue_score_candidates(
                    candidates_df=deferred_top,
                    pdb_map=all_pdb_maps,
                    out_dir=final_dir / "_deferred_rescue_selected_topk",
                    catalytic_his_resnos=CATALYTIC_HIS_RESNOS,
                    # XXX assumes catalytic == fixed: works today because
                    # `fixed_resnos = list(DEFAULT_CATRES)` upstream, so
                    # the two lists are identical. If a future scaffold
                    # ever decouples "fixed by design" from "catalytic",
                    # this call will silently use the wrong set for
                    # tunnel / fpocket active-site anchoring. Pass an
                    # explicit catalytic_resnos= value at that point.
                    catalytic_resnos=fixed_resnos,
                    fixed_resnos=fixed_resnos,
                    sap_max_threshold=final_cycle_cfg.sap_max_threshold,
                    clash_filter=final_cycle_cfg.clash_filter,
                    clash_severe_distance=final_cycle_cfg.clash_severe_distance,
                    seed_dfi_metrics=seed_dfi_metrics,
                    tunnel_metrics_enabled=_tunnel_metrics_enabled,
                    ligand_min_radius=ligand_geometry_summary.get("min_projected_radius"),
                    ligand_resname=ligand_geometry_summary.get("ligand_resname"),
                    log_probs_esmc=log_probs_esmc,
                    log_probs_saprot=log_probs_saprot,
                    weights_per_position=weights_per_position,
                    fitness_cache=fitness_cache,
                    wt_fitness=wt_fitness,
                    rank_weights=rank_weights,
                    rank_targets=rank_targets,
                    fpocket_druggability_min=cycles[0].fpocket_druggability_min,
                    chain=CHAIN,
                )
                top = _overlay_rows_by_id(top, rescued_top)
            top = _apply_solubility_veto(
                top,
                enabled=args.ship_solubility_veto,
                gravy_min=(final_cycle_cfg.gravy_min if args.strategy == "annealing"
                           else args.gravy_min),
                gravy_max=(final_cycle_cfg.gravy_max if args.strategy == "annealing"
                           else args.gravy_max),
                net_charge_min=final_cycle_cfg.net_charge_min,
                net_charge_max=final_cycle_cfg.net_charge_max,
            )
            requested_topk_rows = len(top)
            topk_tsv, top = _write_final_topk_artifacts(
                top=top,
                final_dir=final_dir,
                pdb_map=all_pdb_maps,
                seed_pdb=args.seed_pdb,
            )
            copied_pdbs = len(top)
            if copied_pdbs != requested_topk_rows:
                LOGGER.error(
                    "stage_diverse_topk: wrote %d rows to topk.tsv but only %d "
                    "PDBs were copied to final_topk/topk_pdbs",
                    requested_topk_rows, copied_pdbs,
                )
            if args.final_filter_backfill and copied_pdbs < args.target_k:
                LOGGER.warning(
                    "stage_diverse_topk: final materialized top-K underfilled "
                    "(%d/%d requested)%s. materializable_candidates=%d",
                    copied_pdbs, args.target_k,
                    " (solubility veto active — underfill is expected)"
                    if args.ship_solubility_veto else "",
                    materializable_candidates,
                )

            if args.copy_input_structure_into_out_dir:
                # See note at the matching call site above — surfaces
                # the ~20-40 sec input-reference scoring step in the log.
                LOGGER.info(
                    "input-structure scoring: running struct + tunnel + "
                    "fpocket on seed PDB (rescue-path final selection)"
                )
                final_cycle_cfg = cycles[-1]
                input_reference_df = _build_input_reference_row(
                    template_df=top,
                    seed_pdb=args.seed_pdb,
                    wt_seq=wt_seq,
                    wt_fitness=wt_fitness,
                    expression_result=wt_eng,
                    seed_ss_reduced=ss.ss_reduced,
                    seed_sasa=seed_sasa,
                    position_classes=position_classes,
                    protein_resnos=protein_resnos,
                    catalytic_resnos=DEFAULT_CATRES,
                    fixed_resnos=fixed_resnos,
                    design_ph=7.5,
                    n_term_pad="",
                    c_term_pad="",
                    net_charge_max=final_cycle_cfg.net_charge_max,
                    net_charge_min=final_cycle_cfg.net_charge_min,
                    instability_max=final_cycle_cfg.instability_max,
                    gravy_min=final_cycle_cfg.gravy_min,
                    gravy_max=final_cycle_cfg.gravy_max,
                    aliphatic_min=final_cycle_cfg.aliphatic_min,
                    boman_max=final_cycle_cfg.boman_max,
                    pi_min=final_cycle_cfg.pi_min,
                    pi_max=final_cycle_cfg.pi_max,
                    clash_filter=final_cycle_cfg.clash_filter,
                    clash_severe_distance=final_cycle_cfg.clash_severe_distance,
                    sap_max_threshold=final_cycle_cfg.sap_max_threshold,
                    sap_corrected=args.sap_corrected,
                    seed_dfi_metrics=seed_dfi_metrics,
                    tunnel_metrics_enabled=_tunnel_metrics_enabled,
                    ligand_min_radius=ligand_geometry_summary.get("min_projected_radius"),
                    ligand_resname=ligand_geometry_summary.get("ligand_resname"),
                    log_probs_esmc=log_probs_esmc,
                    log_probs_saprot=log_probs_saprot,
                    weights_per_position=weights_per_position,
                    final_dir=final_dir,
                    fpocket_druggability_min=cycles[0].fpocket_druggability_min,
                )
                input_reference_path = final_dir / "input_reference.tsv"
                input_reference_df.to_csv(input_reference_path, sep="\t", index=False)
                LOGGER.info(
                    "input reference metrics: wrote %s (id=%s)",
                    input_reference_path, args.seed_pdb.stem,
                )
        else:
            LOGGER.warning("final: zero survivors across all cycles!")
            _write_empty_final_artifacts(
                final_dir=final_dir,
                run_dir=run_dir,
                template_df=pd.DataFrame(columns=["id", "sequence"]),
                status="EMPTY_POOL_ALL_CYCLES",
                reason="No ranked survivors across any cycle and no seq-stage backfill candidates were available",
                extra_meta={"n_cycles_run": len(cycles)},
            )

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
        "pipeline": "iterative_design",
        "seed_pdb": str(args.seed_pdb),
        "ligand_params": str(args.ligand_params),
        "plm_artifacts_dir": str(args.plm_artifacts_dir),
        "position_table": str(args.position_table),
        "fixed_resnos": list(DEFAULT_CATRES),  # auto-derived from REMARK 666 in main()
        "catalytic_his_resnos": list(CATALYTIC_HIS_RESNOS),  # auto-derived from REMARK 666 in main()
        "wt_length": L,
        "target_k": args.target_k,
        "diversity_min_hamming": args.min_hamming,
        "final_filter_backfill": getattr(args, "final_filter_backfill", True),
        "copy_input_structure_into_out_dir": getattr(
            args, "copy_input_structure_into_out_dir", True,
        ),
        "n_cycles_run": len(cycles),
        "cycle_configs": [asdict(c) for c in cycles],
        "ptm_spec": getattr(args, "ptm", ""),
        "ligand_geometry": ligand_geometry_summary,
        "tunnel_metrics_enabled": getattr(args, "tunnel_metrics", False),
        "tunnel_hard_gate": getattr(args, "tunnel_hard_gate", True),
        "final_topk_count": final_topk_count,
        "final_unique_sequences": final_unique_seqs,
        "final_pdb_count": final_pdb_count,
        "outputs": {
            "run_dir": str(run_dir),
            "final_topk_fasta": str(final_dir / "topk.fasta"),
            "final_topk_pdbs": str(final_dir / "topk_pdbs"),
            "final_topk_pdbs_protonated": str(final_dir / "topk_pdbs_protonated"),
            "all_survivors": str(final_dir / "all_survivors.tsv"),
            "input_reference_tsv": str(final_dir / "input_reference.tsv"),
            "cycle_metrics_tsv": str(run_dir / "cycle_metrics.tsv"),
        },
        "started_at": timestamp,
    }
    with open(run_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("=== DONE -- top-K at %s ===", final_dir)


if __name__ == "__main__":
    main()
