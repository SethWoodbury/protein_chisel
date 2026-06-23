"""Detect a pathologically hydrophobic / over-represented INPUT scaffold (seed triage).

The PLM fusion bias is conditioned on the seed, so on a pathological seed it AMPLIFIES
the bad composition — at the operating temperature (~0.15) the PLM mean (~1.36 nats) is
an ~8700x odds lock (see :mod:`protein_chisel.sampling.bias_scale`). Empirically, dropping
the PLM (``plm_strength=0``) on such a seed took GRAVY 1.34 -> -0.50 and Ala 27% -> 0.5%.

``assess_seed`` is the PURE detector. The opt-in policies that consume it
(:func:`should_skip_plm` for the cliff, :func:`graded_plm_strength` for the graded
reduction) are also pure, so the driver glue is a one-liner and everything is trivially
unit-testable. The base detector is reference-free; the opt-in z-gate (Feature 1) reuses
:mod:`protein_chisel.expression.aa_composition` and is lazy-imported so this module stays
reference-free + ``--help``-import-light when the gate is off.

Design notes (the WHY):

* **z-gate (F1)** is a *redundant* signal, ORed with the flat ``max_aa_frac`` cap so a
  naturally-abundant AA (Leu/Ala) and a naturally-rare one (Trp/Cys) are judged fairly by
  the per-AA mean±SD instead of one flat 16% line. The z is a **population distance, not a
  significance test** — it divides by the *between-sequence* SD of the reference, so a high
  ``z`` means "far from the typical member of this family", NOT "p<0.001". We require BOTH
  one-sided ``z >= aa_zmax`` AND ``log2_enrichment >= aa_z_log2_floor`` (the codex
  fold-change floor): the floor stops a naturally-rare AA at high z but a trivial % from
  tripping, the one-sidedness keeps an *under*-represented AA (negative z — not a bad-seed
  signal) from tripping, and ``exclude_aas`` drops already-hard-omitted AAs (e.g. Cys). The
  reference is wrong for a non-hydrolase seed, so the driver must pass the design's own EC
  class via ``--aa_reference`` (the skeptic's caveat — see docs/cli_reference.md).

* **severity (F2)** = the max over-threshold ratio across the *active* trips (each ratio is
  1.0 exactly at its trip threshold, >1 above). It is the single scalar the graded policy
  needs, and 0.0 for a clean seed.

* **graded vs cliff (F2)** — the cliff (force ``plm_strength=0``) is the DEFAULT because a
  soft reduction does NOT rescue a pathological seed: at T≈0.15 even ``plm_strength=0.4``
  is ~602x odds, far past the 8x controller authority, so soft only meaningfully differs
  from "off" near ``plm_strength≈0.13``. Soft is therefore opt-in + cluster-validated.
"""
from __future__ import annotations

import dataclasses

# Hydrophobic residues for the hydrophobic-fraction trip (the classic aggregation-prone
# aliphatic + aromatic set; Cys included — hydrophobic and usually omitted anyway).
DEFAULT_HYDROPHOBIC_AAS = "AVLIMFWC"

# Default AA-composition reference for the opt-in z-gate. The EC-3 hydrolase baseline
# matches PTE (the campaign's seed family); it is WRONG for a non-hydrolase seed, so the
# driver threads the design's own EC class via --aa_reference (carried into docs).
DEFAULT_AA_REFERENCE = "swissprot_ec3_hydrolases_2026_01"

# z-gate defaults: the |z|>3 AND |log2|>0.25 precedent already used by
# aa_composition.aa_quality_check (the existing over-rep rule we reuse).
DEFAULT_AA_ZMAX = 3.0
DEFAULT_AA_Z_LOG2_FLOOR = 0.25

# graded_plm_strength: severity at which the soft curve reaches 0 (full PLM strength at
# the trip threshold severity=1, linear to 0 here). 2.0 = "twice over the threshold".
DEFAULT_SOFT_ZERO = 2.0


@dataclasses.dataclass(frozen=True)
class SeedAssessment:
    """Result of :func:`assess_seed`.

    ``pathological`` is the actionable bit; ``reasons`` is a human-readable list (empty
    iff not pathological) for logging; ``over_rep_aas`` are the 1-letter AAs the opt-in
    z-gate flagged (empty when the gate is off); ``severity`` is the F2 over-threshold
    ratio (0.0 when not pathological); the rest are the measured values.
    """
    pathological: bool
    reasons: list[str]
    gravy: float
    max_aa: str
    max_aa_frac: float
    hydrophobic_frac: float
    over_rep_aas: list[str] = dataclasses.field(default_factory=list)
    severity: float = 0.0


def assess_seed(
    sequence: str | None,
    gravy: float,
    *,
    gravy_max: float = 0.4,
    max_aa_frac: float = 0.16,
    hydrophobic_frac_max: float = 0.50,
    hydrophobic_aas: str = DEFAULT_HYDROPHOBIC_AAS,
    aa_zmax: float | None = None,
    aa_reference: str = DEFAULT_AA_REFERENCE,
    aa_z_log2_floor: float = DEFAULT_AA_Z_LOG2_FLOOR,
    exclude_aas: str = "",
) -> SeedAssessment:
    """Flag a seed as pathological if its GRAVY, single-AA over-representation,
    hydrophobic fraction, or (opt-in) per-AA z-score exceed the given thresholds.

    Reference-free by default (raw observed values, no per-class baseline). ``gravy`` is
    supplied by the caller (already computed via protparam); composition is derived from
    ``sequence``. Any tripped threshold sets ``pathological=True`` and appends a reason.
    Defaults: GRAVY>0.4 (matches the existing input-hydrophobicity warning + the cycle
    band), any single AA >=16%, or >50% hydrophobic.

    The opt-in **z-gate** (``aa_zmax`` not None) ADDS a redundant per-AA over-representation
    signal, ORed with the flat ``max_aa_frac`` (the user wants both fireable). For each AA
    it flags over-representation iff one-sided ``z >= aa_zmax`` AND
    ``log2_enrichment >= aa_z_log2_floor`` — see the module docstring for the
    population-distance / floor / one-sided / exclusion rationale. Flagged AAs land in
    ``over_rep_aas`` and each appends a reason carrying the AA, its z, its log2 enrichment,
    AND the reference (so the *logic* of the decision is logged, not just the verdict).

    ``severity`` (the max over-threshold ratio across active trips) is always populated for
    the graded PLM policy (:func:`graded_plm_strength`).
    """
    from protein_chisel.expression.aa_composition import aa_composition_pct
    seq = sequence or ""                                   # None/empty -> no trips, no crash
    pct = aa_composition_pct(seq)                          # {AA: percent 0-100}
    # aa_composition_pct returns all-zero on an empty/non-canonical sequence; treat that as
    # "no dominant AA" rather than an arbitrary max.
    nonzero = {a: p for a, p in pct.items() if p > 0.0}
    # No composition data at all (empty / non-canonical sequence) => there is no seed to
    # assess. Return non-pathological regardless of the supplied GRAVY (which, with no
    # residues, is meaningless) — the edge-case contract is "empty sequence yields no
    # trips" (codex). A real run always has a sequence; the protparam GRAVY would itself
    # have failed first if it were truly empty.
    if not nonzero:
        return SeedAssessment(
            pathological=False, reasons=[], gravy=float(gravy),
            max_aa="", max_aa_frac=0.0, hydrophobic_frac=0.0,
            over_rep_aas=[], severity=0.0,
        )
    max_aa, max_pct = max(nonzero.items(), key=lambda kv: kv[1])
    max_aa_frac_val = max_pct / 100.0
    hydro = {c for c in hydrophobic_aas.upper()}
    hydrophobic_frac = sum(p for a, p in pct.items() if a in hydro) / 100.0

    reasons: list[str] = []
    if gravy > gravy_max:
        reasons.append(f"GRAVY {gravy:+.2f} > {gravy_max:+.2f}")
    if max_aa_frac_val >= max_aa_frac:
        reasons.append(
            f"{max_aa!r} {max_aa_frac_val * 100:.1f}% >= {max_aa_frac * 100:.0f}% "
            f"(single-AA over-representation)")
    if hydrophobic_frac > hydrophobic_frac_max:
        reasons.append(
            f"hydrophobic fraction {hydrophobic_frac * 100:.1f}% > "
            f"{hydrophobic_frac_max * 100:.0f}%")

    # ---- Opt-in z-gate (Feature 1): redundant per-AA over-representation -----------
    over_rep_aas: list[str] = []
    max_z = 0.0
    if aa_zmax is not None:
        # Lazy import keeps this module reference-free (and --help import-light) until the
        # gate is actually used. Reuse the existing z + log2 machinery verbatim.
        from protein_chisel.expression.aa_composition import (
            aa_log2_enrichment,
            aa_z_scores,
        )
        z = aa_z_scores(seq, reference=aa_reference, exclude_aas=exclude_aas)
        log2 = aa_log2_enrichment(seq, reference=aa_reference, exclude_aas=exclude_aas)
        excl = set(exclude_aas.upper())
        for aa in sorted(z):                               # deterministic order
            # Never flag an explicitly-excluded AA (defensive: aa_z_scores already returns
            # the 0.0 sentinel for it, but skip it outright so a degenerate aa_zmax<=0 can
            # never let the sentinel trip — codex).
            if aa in excl:
                continue
            z_val = z[aa]
            l2 = log2.get(aa, 0.0)
            # One-sided over-representation: require a GENUINE positive z (z > 0) AND
            # z >= aa_zmax AND the fold-change floor (log2 >= floor). The z > 0 guard
            # rejects the 0.0 sentinel that aa_z_scores returns for a zero-SD AA (and any
            # AA exactly at its reference mean), so a degenerate aa_zmax <= 0 still can't
            # spuriously flag a no-signal AA (codex).
            if z_val > 0.0 and z_val >= aa_zmax and l2 >= aa_z_log2_floor:
                over_rep_aas.append(aa)
                max_z = max(max_z, z_val)
                reasons.append(
                    f"{aa!r} z={z_val:+.2f} >= {aa_zmax:.2f} "
                    f"AND log2={l2:+.2f} >= {aa_z_log2_floor:.2f} "
                    f"(over-rep vs {aa_reference})")

    sev = _severity_from_values(
        bool(reasons), gravy=gravy, gravy_max=gravy_max,
        max_aa_frac_val=max_aa_frac_val, max_aa_frac=max_aa_frac,
        hydrophobic_frac=hydrophobic_frac, hydrophobic_frac_max=hydrophobic_frac_max,
        max_z=max_z, aa_zmax=aa_zmax,
    )

    return SeedAssessment(
        pathological=bool(reasons),
        reasons=reasons,
        gravy=float(gravy),
        max_aa=max_aa,
        max_aa_frac=max_aa_frac_val,
        hydrophobic_frac=hydrophobic_frac,
        over_rep_aas=over_rep_aas,
        severity=sev,
    )


def _severity_from_values(
    pathological: bool,
    *,
    gravy: float,
    gravy_max: float,
    max_aa_frac_val: float,
    max_aa_frac: float,
    hydrophobic_frac: float,
    hydrophobic_frac_max: float,
    max_z: float,
    aa_zmax: float | None,
) -> float:
    """Max over-threshold ratio across the ACTIVE trips (the kernel behind
    :func:`severity`). Each ratio is 1.0 exactly at its trip threshold and >1 above;
    only trips that actually fired contribute. 0.0 when not pathological.

    Kept separate so :func:`assess_seed` (which already has the measured values) and the
    public :func:`severity` (which reads them off a :class:`SeedAssessment`) share ONE
    definition — no duplicated ratio math.
    """
    if not pathological:
        return 0.0
    ratios: list[float] = []
    if gravy > gravy_max and gravy_max > 0:
        ratios.append(gravy / gravy_max)
    if max_aa_frac_val >= max_aa_frac and max_aa_frac > 0:
        ratios.append(max_aa_frac_val / max_aa_frac)
    if hydrophobic_frac > hydrophobic_frac_max and hydrophobic_frac_max > 0:
        ratios.append(hydrophobic_frac / hydrophobic_frac_max)
    if aa_zmax is not None and max_z >= aa_zmax and aa_zmax > 0:
        ratios.append(max_z / aa_zmax)
    # A trip fired (pathological) but no positive-denominator ratio could be formed
    # (degenerate thresholds) -> treat as "exactly at threshold" so the graded policy
    # still reduces rather than dividing by zero.
    return max(ratios) if ratios else 1.0


def severity(assessment: SeedAssessment | None) -> float:
    """Public severity accessor: the max over-threshold ratio for a :class:`SeedAssessment`.

    :func:`assess_seed` already computes severity from the EXACT thresholds the seed was
    judged against (via the shared :func:`_severity_from_values` kernel) and caches it on
    ``assessment.severity``, so this is a thin, side-effect-free accessor with the
    None/non-pathological guard (0.0 in both cases). Provided as a named function so the
    graded policy and tests read severity through one documented entry point rather than
    poking the dataclass field directly.
    """
    if assessment is None or not assessment.pathological:
        return 0.0
    return float(assessment.severity)


def should_skip_plm(
    assessment: SeedAssessment | None,
    *,
    enabled: bool,
    current_plm_strength: float,
) -> bool:
    """Policy: skip the PLM bias (force ``plm_strength=0``) iff the triage is enabled,
    the seed is pathological, and the PLM is currently on (>0).

    Pure so the driver glue is a one-liner: ``if should_skip_plm(...): plm_strength = 0``.
    Returns ``False`` for a missing assessment (triage failed → leave the run unchanged).
    This is the CLIFF path (kept for the default behaviour + its tests);
    :func:`graded_plm_strength` is the superset that also serves the opt-in soft curve.
    """
    return bool(
        enabled
        and assessment is not None
        and assessment.pathological
        and current_plm_strength > 0
    )


def graded_plm_strength(
    assessment: SeedAssessment | None,
    *,
    enabled: bool,
    current_plm_strength: float,
    soft: bool = False,
    soft_zero: float = DEFAULT_SOFT_ZERO,
) -> float:
    """Return the PLM ``plm_strength`` to use after triage — the unified policy for both
    the cliff (default) and the opt-in soft/graded reduction.

    Gating mirrors :func:`should_skip_plm`: when the triage is disabled, the seed is clean,
    the assessment is missing, or the PLM is already off, the strength is returned
    UNCHANGED (so the run is byte-identical when the triage is off / the seed is fine).

    When the seed IS pathological and the PLM is on:

    * ``soft=False`` (DEFAULT) → the cliff: ``0.0`` (identical to ``should_skip_plm`` → 0).
      This is the default because a soft reduction does not rescue a pathological seed (see
      the module docstring's 0.4=602x≫8x math), so soft stays opt-in + validated.
    * ``soft=True`` → ``current * clamp(1 - (severity-1)/(soft_zero-1), 0, 1)``: full
      strength at the trip threshold (``severity==1``) decaying linearly to 0 at
      ``severity>=soft_zero``. Guards ``soft_zero<=1`` (a non-positive denominator) by
      degrading to the cliff (0.0) rather than producing a NaN. A pathological seed whose
      ``severity`` is degenerate (``< 1``, e.g. a hand-built assessment that never set the
      field) ALSO degrades to the cliff — we already KNOW the seed is pathological, so the
      safe action is to drop the PLM, never to silently leave it at full strength (codex).
    """
    if not (enabled and assessment is not None and assessment.pathological
            and current_plm_strength > 0):
        return current_plm_strength
    if not soft:
        return 0.0
    if soft_zero <= 1.0:                                    # degenerate denominator -> cliff
        return 0.0
    sev = float(assessment.severity)
    if not (sev >= 1.0):                                    # unset/degenerate severity (or NaN)
        return 0.0                                         # pathological but no graded signal -> cliff
    frac = 1.0 - (sev - 1.0) / (soft_zero - 1.0)
    frac = min(1.0, max(0.0, frac))                        # clamp to [0, 1]
    return float(current_plm_strength) * frac
