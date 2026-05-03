"""Structure-aware bias derivation for LigandMPNN.

Pipeline #2 from the architectural review: take a packer's per-residue
chi log-likelihoods (from PIPPack or FlowPacker) and use them to refine
the PLM-derived (L, 20) AA bias before feeding to LigandMPNN.

Two formulations are supported:

V1 (trust modulation) -- :func:`apply_chi_trust_to_bias`
-------------------------------------------------------
- Compute a per-position trust score in [0, 1] from the packer's chi NLL.
- Modulate the existing PLM fusion bias toward uniform at low-trust
  positions; preserve at high-trust positions.
- Doesn't change which AAs are favoured; only how strongly.
- Cheap, defensible as a "PLM is unreliable when structure disagrees" prior.

V2 (rigorous AA marginalisation) -- :func:`aa_logprior_from_chi`
----------------------------------------------------------------
- For each (position, AA) pair compute log p(a | predicted_chi_i, backbone)
  by integrating the packer's chi distribution against the Dunbrack
  rotamer library for AA a at this position's (phi, psi).
- Returns a (L, 20) structure-conditioned AA log-prior that fuses
  multiplicatively with ESM-C/SaProt.
- More rigorous; depends on accessible Dunbrack tables. SKELETON
  ONLY in this commit -- the integration math + Dunbrack table
  loading land in a follow-up once V1 is empirically validated.

Both feed through :func:`fuse_plm_struct_logits` which extends
:func:`sampling.plm_fusion.fuse_plm_logits` with an optional 3rd input.

Catalytic-residue safety
========================
Whatever modulation we apply, catalytic residues (REMARK 666) MUST be
left alone -- the design intentionally has unusual rotamers there
(carbamylated lysines, attack-poised histidines etc). All public
functions accept a ``catalytic_resnos`` parameter; if a position is
catalytic, its bias row is passed through unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Sequence

import numpy as np

from protein_chisel.sampling.plm_fusion import (
    AA_ORDER,
    FusionConfig,
    FusionResult,
    calibrate_log_odds,
    cosine_similarity_per_position,
    entropy_match_temperature,
    fuse_plm_logits,
    per_position_entropy,
)


LOGGER = logging.getLogger("protein_chisel.struct_aware_bias")


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# AAs with no chi degrees of freedom -- the packer's chi-based trust
# score is undefined at these positions. We return a neutral trust
# score of `NO_CHI_TRUST` so the PLM bias is preserved (no modulation).
NO_CHI_AAS: frozenset[str] = frozenset({"G", "A"})
NO_CHI_TRUST: float = 1.0   # i.e. don't modulate; PLM bias stays as-is


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


_DEFAULT_CHI_BIN_COUNT = 36   # PIPPack default (10° bins); FlowPacker is continuous.


@dataclass
class StructAwareBiasConfig:
    """Knobs for V1 structure-aware bias shrinkage.

    Trust formula::

        trust_i = clip(sigmoid((chi_logp_i - reference_logp) / T),
                       trust_floor, trust_ceiling)

    The cross-review caught that a raw ``sigmoid(chi_logp / T)`` maps
    chi_logp=0 (max-confidence packer) to trust=0.5, not 1.0 -- the
    sigmoid input must be CENTERED against a "uniform packer" baseline
    so trust=0.5 corresponds to "no information" rather than to "max
    confidence". We default ``reference_logp`` to ``-log(20)`` (a packer
    that's perfectly uniform over 20 chi bins), so:
      - chi_logp = 0   (peaked / confident): trust ~ 0.82
      - chi_logp = -3  (uniform packer):     trust = 0.50
      - chi_logp = -7  (confidently wrong):  trust ~ 0.12
      - chi_logp = -20 (very wrong):         trust ~ 0.00

    Attributes:
        chi_bin_count: number of chi bins the upstream packer
            discretizes over. PIPPack uses 36 (10° bins); FlowPacker
            is continuous so this is a coarse approximation.
            ``reference_logp`` is auto-derived as ``-log(chi_bin_count)``
            so trust=0.5 at "uniform packer" regardless of the bin count.
        reference_logp: optional explicit override of the trust=0.5
            anchor. When None (default), auto-computed from
            ``chi_bin_count``.
        trust_temperature: sigmoid sharpness. Smaller T = sharper
            transition. Default 2.0 = transition over ~4 nats.
        trust_floor: minimum trust value, in [0, 1]. Default 0.5
            so even badly-explained positions retain HALF of the
            PLM bias -- the packer's signal in V1 conflates
            "uncertain" with "WT-disagreeing", and we don't want
            to fully erase a useful LM prior on the basis of a
            possibly-spurious structural signal.
        trust_ceiling: maximum trust value, in [0, 1]. Caps how much
            we trust the PLM at any position.
        catalytic_passthrough: if True (default), catalytic resnos
            keep their original PLM bias unmodulated -- structural
            outliers there are intentional design features.

    Cross-review notes (V1 limitations):

    - Agent: chi_logp_per_position currently measures WT-NLL ("how
      well does the packer's distribution explain the OBSERVED chi"),
      which conflates "packer is uncertain" with "packer disagrees
      with WT". The cleaner signal is the packer's own MODE confidence
      (max over chi bins of chi_log_probs), but FlowPacker's wrapper
      exposes per-chi WT-NLL, not the bin distribution. Future work:
      extend pippack/flowpacker wrappers to expose mode-logp.
    - Codex: V1 is a "shrinkage heuristic, not a true prior". V2
      (rigorous Dunbrack-marginalised AA log-prior) is the
      scientifically correct answer.
    """
    chi_bin_count: int = _DEFAULT_CHI_BIN_COUNT
    reference_logp: Optional[float] = None  # auto = -log(chi_bin_count)
    trust_temperature: float = 2.0
    trust_floor: float = 0.5       # preserve at least half the LM bias everywhere
    trust_ceiling: float = 1.0
    catalytic_passthrough: bool = True

    def resolved_reference_logp(self) -> float:
        """Effective reference_logp -- explicit override or
        ``-log(chi_bin_count)``. Used inside trust_from_chi_logp."""
        if self.reference_logp is not None:
            return float(self.reference_logp)
        if self.chi_bin_count <= 0:
            raise ValueError(
                f"chi_bin_count must be positive, got {self.chi_bin_count}"
            )
        return float(-np.log(self.chi_bin_count))


@dataclass
class StructAwareBiasResult:
    """What :func:`fuse_plm_struct_logits` returns.

    Attributes:
        bias: (L, 20) the final additive bias for LigandMPNN.
        plm_fusion: the underlying FusionResult from fuse_plm_logits
            (so callers can inspect what the PLM-only bias would have
            been).
        trust_per_position: (L,) trust score actually used at each
            position. NaN at positions where the packer didn't emit
            chi info; defaults to NO_CHI_TRUST in those cases.
        catalytic_indices: 0-indexed positions that were passed
            through unchanged due to catalytic_passthrough.
        modulation_path: which formulation produced the result
            ("v1_trust" or "v2_marginalisation"). For runtime sanity
            checks + manifest provenance.
    """
    bias: np.ndarray
    plm_fusion: FusionResult
    trust_per_position: np.ndarray
    catalytic_indices: tuple[int, ...]
    modulation_path: Literal["v1_trust", "v2_marginalisation"]


# ----------------------------------------------------------------------
# V1: trust modulation
# ----------------------------------------------------------------------


def per_position_chi_nll(
    chi_logp_per_residue: np.ndarray,
    *,
    aggregate: str = "mean",
) -> np.ndarray:
    """Reduce a per-residue per-chi log-likelihood matrix to one scalar
    per residue, suitable for trust-modulation downstream.

    Args:
        chi_logp_per_residue: (L, K) where K is the number of chi
            angles emitted by the packer (typically 4). Entries that
            are NaN indicate "this AA doesn't have this chi" -- they
            are ignored in the aggregate.
        aggregate: ``"mean"`` (default) or ``"sum"``. Use mean if you
            want fairness across AAs of different chi count; sum if
            you want long side-chains to contribute more total
            evidence (typically NOT what you want for trust modulation).

    Returns:
        (L,) per-position chi log-likelihood. Higher = more plausible
        rotamer under the packer. We flip sign in the trust step
        (NLL = -logp).

    Notes:
        Positions where ALL chi entries are NaN (Gly/Ala) get NaN
        in the output. The downstream trust function falls back to
        ``NO_CHI_TRUST`` for those rows.
    """
    if chi_logp_per_residue.ndim != 2:
        raise ValueError(
            f"expected (L, K) array, got {chi_logp_per_residue.shape}"
        )
    if aggregate not in ("mean", "sum"):
        raise ValueError(f"unsupported aggregate {aggregate!r}; use mean or sum")
    arr = chi_logp_per_residue
    finite_mask = ~np.isnan(arr)
    counts = finite_mask.sum(axis=1)
    # Manual sum + count to avoid `np.nanmean` warning on all-NaN rows.
    sums = np.where(finite_mask, arr, 0.0).sum(axis=1)
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    if aggregate == "mean":
        nonzero = counts > 0
        out[nonzero] = sums[nonzero] / counts[nonzero]
    else:  # sum
        nonzero = counts > 0
        out[nonzero] = sums[nonzero]
    return out


def trust_from_chi_logp(
    chi_logp_per_position: np.ndarray,
    *,
    config: Optional[StructAwareBiasConfig] = None,
) -> np.ndarray:
    """Convert per-position chi log-likelihood to a trust score in [0, 1].

    trust_i = clip(sigmoid((chi_logp_i - reference_logp) / T),
                   trust_floor, trust_ceiling)

    The sigmoid is CENTERED at ``reference_logp`` (default ~-log(20))
    so a packer with no opinion (uniform over chi bins) gets trust
    = 0.5, not 0.5 of-the-max. This way max-confidence packers
    correctly approach trust=1.0 and confidently-wrong packers
    approach trust=0.0.

    NaN inputs (no-chi AAs like Gly/Ala) get :data:`NO_CHI_TRUST`
    (= 1.0 by default), meaning the PLM bias is preserved at those
    positions.

    Args:
        chi_logp_per_position: (L,) per-position aggregated chi
            log-likelihood (<= 0 by construction; it's a log-prob).
        config: knobs for temperature / floor / ceiling / reference.

    Returns:
        (L,) trust scores in [trust_floor, trust_ceiling].
    """
    cfg = config or StructAwareBiasConfig()
    # Defensive validation: NaN passes ``<= 0`` silently (NaN comparisons
    # return False), so an explicit isfinite check is required.
    if not (np.isfinite(cfg.trust_temperature) and cfg.trust_temperature > 0):
        raise ValueError(
            f"trust_temperature must be a positive finite number, got {cfg.trust_temperature}"
        )
    if not (0.0 <= cfg.trust_floor <= cfg.trust_ceiling <= 1.0):
        raise ValueError(
            f"need 0 <= trust_floor ({cfg.trust_floor}) <= "
            f"trust_ceiling ({cfg.trust_ceiling}) <= 1"
        )
    reference_logp = cfg.resolved_reference_logp()
    if not np.isfinite(reference_logp):
        raise ValueError(
            f"reference_logp must be finite, got {reference_logp}"
        )

    out = np.empty_like(chi_logp_per_position, dtype=np.float64)
    finite = ~np.isnan(chi_logp_per_position)
    # Centered sigmoid: input (chi_logp - reference) so "uniform packer"
    # corresponds to z=0 -> sigmoid=0.5.
    z = (chi_logp_per_position[finite] - reference_logp) / cfg.trust_temperature
    # Numerically stable sigmoid: branch explicitly so np.exp doesn't
    # overflow for extreme negative or positive z. (np.where would
    # evaluate both branches and trigger overflow warnings.)
    sig = np.empty_like(z)
    pos = z >= 0
    sig[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg_exp = np.exp(z[~pos])
    sig[~pos] = neg_exp / (1.0 + neg_exp)
    out[finite] = np.clip(sig, cfg.trust_floor, cfg.trust_ceiling)
    out[~finite] = NO_CHI_TRUST
    return out


def apply_chi_trust_to_bias(
    plm_bias: np.ndarray,
    trust_per_position: np.ndarray,
    *,
    catalytic_resnos: Iterable[int] = (),
    resno_to_index: Optional[dict[int, int]] = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Apply per-position trust modulation to a PLM bias matrix.

    Modulation::

        modulated_bias[i, a] = trust_i * plm_bias[i, a]

    Multiplying by trust in [0, 1] flattens low-trust positions toward
    the zero (= uniform) bias. We DON'T mix toward UniProt background
    because that would inject information not coming from the structure.

    Args:
        plm_bias: (L, 20) additive bias from fuse_plm_logits.
        trust_per_position: (L,) trust scores.
        catalytic_resnos: PDB resseq numbers to keep unmodulated.
        resno_to_index: optional mapping from PDB resseq -> 0-indexed
            position. If None and catalytic_resnos is non-empty, we
            assume identity mapping (resno - 1 = index), which is the
            common case for proteins designed with continuous numbering
            from 1.

    Returns:
        (modulated_bias, catalytic_0idx_tuple)
    """
    if plm_bias.ndim != 2 or plm_bias.shape[1] != 20:
        raise ValueError(f"expected (L, 20) plm_bias, got {plm_bias.shape}")
    if trust_per_position.shape != (plm_bias.shape[0],):
        raise ValueError(
            f"trust shape {trust_per_position.shape} != plm_bias[:0] {plm_bias.shape[0]}"
        )
    if np.any(np.isnan(trust_per_position)):
        raise ValueError("trust values must not be NaN")
    if not np.all((trust_per_position >= 0) & (trust_per_position <= 1)):
        raise ValueError("trust values must be in [0, 1]")

    cat_set: set[int] = set()
    if catalytic_resnos:
        if resno_to_index is None:
            LOGGER.warning(
                "apply_chi_trust_to_bias: resno_to_index=None; falling back "
                "to identity mapping (resno-1=index). This is INCORRECT for "
                "multi-chain proteins, gapped numbering, and PDBs with "
                "insertion codes -- callers should pass an explicit "
                "resno_to_index map."
            )
            cat_set = {int(r) - 1 for r in catalytic_resnos}
        else:
            cat_set = {resno_to_index[int(r)] for r in catalytic_resnos if int(r) in resno_to_index}
        # Drop indices outside [0, L).
        cat_set = {i for i in cat_set if 0 <= i < plm_bias.shape[0]}

    # Multiply bias by trust, but PRESERVE -inf rows -- a -inf entry
    # is a hard-mask ("never propose this AA here"), and trust=0 must
    # not turn that into NaN. We start from a copy of plm_bias (so
    # non-finite cells stay untouched) and apply the trust modulation
    # only on the finite cells via explicit indexing -- doing this
    # with np.where would still evaluate the multiplication in the
    # discarded branch and emit a warning.
    out = plm_bias.copy()
    finite_mask = np.isfinite(plm_bias)
    if finite_mask.any():
        # broadcast trust to the bias shape, then mask
        trust_b = np.broadcast_to(trust_per_position[:, None], plm_bias.shape)
        out[finite_mask] = plm_bias[finite_mask] * trust_b[finite_mask]
    if cat_set:
        for i in cat_set:
            out[i] = plm_bias[i]   # passthrough: original bias

    return out, tuple(sorted(cat_set))


# ----------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------


def fuse_plm_struct_logits(
    log_probs_esmc: np.ndarray,
    log_probs_saprot: np.ndarray,
    chi_logp_per_position: np.ndarray,
    position_classes: Sequence[str],
    *,
    catalytic_resnos: Iterable[int] = (),
    resno_to_index: Optional[dict[int, int]] = None,
    fusion_config: Optional[FusionConfig] = None,
    struct_config: Optional[StructAwareBiasConfig] = None,
) -> StructAwareBiasResult:
    """Fuse ESM-C + SaProt + structure-aware chi info into a bias matrix.

    V1 path: PLM fusion -> trust-modulate by per-position chi NLL.

    Args:
        log_probs_esmc: (L, 20) ESM-C masked-LM log-probabilities.
        log_probs_saprot: (L, 20) SaProt log-probabilities.
        chi_logp_per_position: (L,) per-position chi log-likelihood
            from PIPPack/FlowPacker (use :func:`per_position_chi_nll`
            to reduce a (L, K) matrix to (L,)). NaN entries (e.g. for
            Gly/Ala) are treated as "no info, preserve PLM bias".
        position_classes: length-L list driving FusionConfig class_weights.
        catalytic_resnos: PDB resseq numbers to keep unmodulated.
        resno_to_index: maps PDB resseq -> 0-indexed position.
        fusion_config: forwarded to :func:`fuse_plm_logits`.
        struct_config: trust-modulation knobs.

    Returns:
        StructAwareBiasResult.
    """
    if log_probs_esmc.shape != log_probs_saprot.shape:
        raise ValueError(
            f"PLM shape mismatch: esmc={log_probs_esmc.shape}, "
            f"saprot={log_probs_saprot.shape}"
        )
    L = log_probs_esmc.shape[0]
    if L == 0:
        raise ValueError("empty protein (L=0) is not supported")
    if chi_logp_per_position.shape != (L,):
        raise ValueError(
            f"chi_logp shape {chi_logp_per_position.shape} != L {L}"
        )

    cfg_struct = struct_config or StructAwareBiasConfig()

    plm_fusion = fuse_plm_logits(
        log_probs_esmc, log_probs_saprot, position_classes, config=fusion_config,
    )

    trust = trust_from_chi_logp(chi_logp_per_position, config=cfg_struct)
    # Honor catalytic_passthrough=False by zeroing the catalytic_resnos
    # iterable before passing to the modulator -- otherwise the
    # resno-conditional passthrough in apply_chi_trust_to_bias would
    # ignore the config flag.
    cat_for_modulation = catalytic_resnos if cfg_struct.catalytic_passthrough else ()
    bias, cat_idx = apply_chi_trust_to_bias(
        plm_fusion.bias, trust,
        catalytic_resnos=cat_for_modulation,
        resno_to_index=resno_to_index,
    )

    return StructAwareBiasResult(
        bias=bias,
        plm_fusion=plm_fusion,
        trust_per_position=trust,
        catalytic_indices=cat_idx,
        modulation_path="v1_trust",
    )


# ----------------------------------------------------------------------
# V2 skeleton (rigorous AA marginalisation; not yet implemented)
# ----------------------------------------------------------------------


def aa_logprior_from_chi(
    chi_log_probs_per_residue_per_chi: np.ndarray,
    phi_psi_per_residue: np.ndarray,
    *,
    dunbrack_table: Optional["DunbrackTable"] = None,
) -> np.ndarray:
    """V2: derive per-(position, AA) log-prior from the packer's chi
    distribution + Dunbrack rotamer library.

    log p(a | predicted_chi_i, backbone_i) = log integral over a's rotamers r
        [ p_pred(chi=r | i) * p_dun(r | a, phi_i, psi_i) ]

    NOT IMPLEMENTED -- this commit ships the V1 trust-modulation path
    only. The V2 path requires:
      - a Dunbrack 2011 backbone-dependent rotamer table (loaded from
        the rlabduke/reference_data submodule we already have, or from
        PyRosetta);
      - per-AA rotamer count + bin-center coordinates;
      - per-bin chi-log-prob lookup from the packer (PIPPack exposes
        chi_log_probs over discretized bins; FlowPacker is continuous
        and would need numerical integration).

    The skeleton is here so callers can declare ``modulation_path="v2"``
    in their TierFilterConfig once the implementation lands; for now
    raises NotImplementedError.

    Args:
        chi_log_probs_per_residue_per_chi: (L, K, n_bins) where K is
            the number of chi angles (4) and n_bins is the packer's
            chi discretization (typically 36 for 10-deg bins).
        phi_psi_per_residue: (L, 2) backbone dihedrals in degrees.
        dunbrack_table: pre-loaded Dunbrack table; defaults to the
            shipped rlabduke/reference_data Top8000 set.

    Returns:
        (L, 20) per-position AA log-prior.
    """
    raise NotImplementedError(
        "V2 (rigorous AA marginalisation) is a skeleton in this commit; "
        "use V1 trust-modulation via fuse_plm_struct_logits() with "
        "modulation_path='v1_trust' for now"
    )


class DunbrackTable:
    """V2 placeholder: Dunbrack 2011 backbone-dependent rotamer table.

    Future implementation will load from
    ``/net/databases/lab/molprobity/chem_data/reference_data/`` (Top8000
    rlabduke tables we cloned earlier) or via PyRosetta's
    ``RotamerLibrary``. Leaving as a forward declaration so type hints
    in :func:`aa_logprior_from_chi` are coherent.
    """
    def __init__(self) -> None:
        raise NotImplementedError("Dunbrack table loader not yet implemented")


__all__ = [
    "DunbrackTable",
    "NO_CHI_AAS",
    "NO_CHI_TRUST",
    "StructAwareBiasConfig",
    "StructAwareBiasResult",
    "aa_logprior_from_chi",
    "apply_chi_trust_to_bias",
    "fuse_plm_struct_logits",
    "per_position_chi_nll",
    "trust_from_chi_logp",
]
