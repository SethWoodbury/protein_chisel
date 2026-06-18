"""Closed-loop adaptive bias controller for iterative MPNN sampling.

This module steers the LigandMPNN sampling biases across design cycles toward
target solubility (net charge, surface hydrophobicity) using **feedback control**
on the survivor pool's measured properties. It is opt-in; when the driver does
not invoke it, sampling is byte-identical to the legacy pipeline.

Why an integral-hold controller (and not proportional release)
--------------------------------------------------------------
Each cycle samples a pool, filters it on biophysical properties, and keeps the
survivors. We treat the *survivor-pool mean* of a property as the controlled
variable and the *additive MPNN bias* (nats, log-space) as the actuator. The pool
mean responds to the bias applied **during the current cycle** and largely forgets
earlier cycles' biases (a near-static plant; the only pool-to-pool memory is the
consensus reinforcement). So to *hold* the pool at target you must keep applying the
bias that achieves it — a controller that releases the bias when the pool looks
healthy would limit-cycle (bias on -> pool good -> release -> pool bad -> ...). The
correct controller for regulating a static plant at a setpoint is therefore an
**integral controller that holds its accumulated bias** (with a gentle leak for
robustness against the consensus-memory component and gain misestimation), NOT a
proportional controller that resets each cycle.

Control law (per axis, cycle k -> k+1)
--------------------------------------
    GATE open & out of deadband:  u_{k+1} = clip( carry*u_k + g_eff*e_norm , +-max )
    GATE closed or in deadband:   u_{k+1} = u_k                    (purely HOLD)
    wrong-sign plant response:     u_{k+1} = 0                      (freeze)

  * ``e_norm`` is the *signed, deadbanded, band-normalized* error of the pool mean
    vs the property target. Its sign flips when the pool overshoots past target, so
    the controller automatically **reverses** to pull an over-corrected pool back —
    bidirectional anti-overshoot. (Leaky-integral fixed point sits inside the
    deadband, which is the tolerated region by construction.)
  * ``carry`` (default 0.9) is the integral leak applied ONLY during active
    integration (anti-windup hedge against the partial consensus-memory component).
    When the pool is healthy the bias is held *exactly* (no leak), so it neither
    reverts (the limit-cycle failure of "release to 0") nor drifts (the slow-bleed
    failure of leaking during hold). The bias relaxes only by the controller
    actively reversing when the pool drifts back out of band.
  * ``g_eff = g * gain_correction`` where ``gain_correction`` is a two-point secant
    estimate of the plant gain (how much the pool moved per nat applied), so the
    controller self-calibrates over the ~2 correction cycles available. If the pool
    moved the *wrong* way after a bias (wrong-sign plant), the axis is **frozen**
    (u -> 0) rather than doubling down — the anti-divergence safety net.

Gate vs drive
-------------
The controller fires an axis only when the pool is **statistically in trouble**
(binary GATE, in statistical units) and then drives by the **physical** error
(continuous DRIVE, in property units). The two are deliberately separated:

  * GATE open  <=>  N >= min_n  AND  |t| > t_min  AND  fail_fraction passes a
    Wald lower-bound test ( > 0 ) and exceeds f_min.
  * DRIVE      =  signed, deadbanded error of the pool mean vs target.

The surface down-weight is additionally FOCUSED on amino acids that are "way
over-represented" (z > 3 vs the hydrolase reference) when any are, via
``hydrophobic_over_rep_mask`` (reusing ``aa_composition.aa_z_scores``).

Actuators (partitioned by safety)
---------------------------------
  * **charge**  -> GLOBAL per-AA bias (up-weight D/E, mild down-weight K/R; the
    reverse when the pool overshoots too negative). Composed with the existing
    class-balance ``bias_AA`` (class-balance wins conflicts).
  * **surface hydrophobicity** -> PER-POSITION (L, 20) delta that down-weights
    hydrophobic residues ONLY at solvent-exposed ``distal_surface`` positions
    (CA > 10A from the ligand AND sidechain-SASA fraction >= 0.20), never the buried
    core, never catalytic/fixed positions, and never the binding region — so
    conserved-H-bond anchors (which sit at the ligand/catalytic site) are excluded by
    construction. When a specific hydrophobic AA is "way over-represented" (z > 3 vs
    the hydrolase reference, via ``hydrophobic_over_rep_mask``) the down-weight is
    focused on it; otherwise it spreads across all hydrophobic AAs.

The two actuators act on disjoint amino-acid sets (charge: D/E/K/R; hydrophobic:
A/V/L/I/M/F/W/C), so they never double-count the same residue. The held bias is
emitted whenever ``|u| > 0`` (whether actively integrating or holding), so a healthy
pool does not cause the bias to be dropped (which would revert the pool).

All functions here are pure (numpy / pandas / stdlib only) and do no I/O, so the
control logic is unit-testable without models, PyRosetta, or the cluster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Kyte-Doolittle hydrophobicity lives in exactly ONE place — scoring.sap, the
# module that also backs the driver's SAP proxy. Importing it here (rather than
# re-declaring the dict) removes the duplication the 2026-06 audit flagged and
# gives the controller + SAP one hydrophobicity model to evolve. Re-exported via
# __all__ for back-compat with importers of adaptive_bias.KD_HYDROPHOBICITY.
from protein_chisel.scoring.sap import KD_HYDROPHOBICITY

# ---------------------------------------------------------------------------
# Constants. AA_ORDER matches sampling.plm_fusion.AA_ORDER.
# ---------------------------------------------------------------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}

HYDROPHOBIC_AAS = frozenset("AVLIMFWC")   # matches expression.builtin_rules
ACIDIC_AAS = frozenset("DE")
BASIC_AAS = frozenset("KR")

# Solvent-exposed, NOT active-site-adjacent. nearby_surface is within 10A of the
# ligand (binding region) so it is deliberately excluded — we steer solubility
# surface only, never the catalytic environment. Strictly the current taxonomy's
# distal_surface (CA >10A AND sidechain SASA fraction >= 0.20); a legacy table that
# lacks this class simply yields no surface term (safe degradation).
SURFACE_CLASSES = frozenset({"distal_surface"})

# Position classes the non_tunnel_surface scope is allowed to steer. POSITIVE
# gating (admit only these) is deliberately safer than negatively excluding the
# active site: a negative gate would sweep in any *other* class — legacy core
# classes ("buried"/"first_shell") or an unknown class on a new scaffold — if its
# SASA happened to clear the gate, down-weighting a load-bearing core hydrophobic.
# Both are exposed by construction (distal_surface = exposed distal; nearby_surface
# = the binding-region surface the user wants back). Configurable per call.
STEERABLE_SURFACE_CLASSES = SURFACE_CLASSES | frozenset({"nearby_surface"})

CLASS_BALANCE_MAX_NATS = 2.5   # the clamp compute_class_balanced_bias_AA uses


def surface_scope(*, L: int, position_classes: list,
                  sasa_fraction: Optional[np.ndarray], fixed_idx: set,
                  sasa_gate: Optional[float] = None,
                  tunnel_lining_idx: frozenset = frozenset(),
                  throat_band_idx: frozenset = frozenset(),
                  surface_classes: frozenset = SURFACE_CLASSES,
                  steerable_classes: frozenset = STEERABLE_SURFACE_CLASSES) -> np.ndarray:
    """``(L,)`` boolean membership mask for the surface actuator's positions.

    The surface down-weight is applied only where this mask is True. Two modes:

    * ``sasa_gate is None`` → **LEGACY**: position ``i`` is in scope iff its class
      is in ``surface_classes`` (default ``distal_surface``), it is not fixed, and
      it is solvent-exposed (``sasa_fraction`` missing/None ⇒ treated exposed, else
      finite and ``> 0``). This reproduces *exactly* the gate historically inlined
      in :func:`build_surface_delta`, so a no-gate run is byte-identical.

    * ``sasa_gate`` set → **non_tunnel_surface** (the user's "what I can steer by
      eye" scope): ``i`` is in scope iff ``sasa_fraction[i] >= sasa_gate`` AND its
      class is in ``steerable_classes`` (POSITIVE gating — only vetted exposed
      classes, so a core/unknown class can never leak in via a high SASA) AND ``i``
      is not in ``tunnel_lining_idx`` / ``throat_band_idx`` AND ``i`` is not fixed.
      This is a *superset* of ``distal_surface`` (drops the ligand-distance gate,
      adds back exposed ``nearby_surface``) *minus* the tunnel mouth / throat /
      active site. NOTE the gate uses the (total-residue-SASA) ``sasa_fraction``
      proxy, which can overstate sidechain exposure for an exposed-backbone /
      buried-sidechain ``nearby_surface`` residue — restrict ``steerable_classes``
      to ``{"distal_surface"}`` for a more conservative run.

    Structure-invariant (depends only on classes/SASA/fixed/tunnel sets), so the
    driver computes it once and threads it through every cycle.
    """
    mask = np.zeros(int(L), dtype=bool)
    legacy = sasa_gate is None
    for i in range(int(L)):
        if i in fixed_idx:
            continue
        cls = position_classes[i] if i < len(position_classes) else None
        s = (float(sasa_fraction[i])
             if (sasa_fraction is not None and i < len(sasa_fraction)) else None)
        if legacy:
            if cls not in surface_classes:
                continue
            if s is not None and (not math.isfinite(s) or s <= 0.0):
                continue
            mask[i] = True
        else:
            if cls not in steerable_classes:        # POSITIVE gate (vetted exposed)
                continue
            if i in tunnel_lining_idx or i in throat_band_idx:
                continue
            if s is None or not math.isfinite(s) or s < sasa_gate:
                continue
            mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# Per-AA projections
# ---------------------------------------------------------------------------
def kd_centered_weights() -> np.ndarray:
    """Length-20 weight vector (AA_ORDER) for the hydrophobicity axis.

    ``w[a] = (KD[a] - mean KD) / max|KD - mean KD|`` so hydrophobic AAs are
    positive (~+1 for Ile), hydrophilic AAs negative (~-0.9 for Arg), and the
    largest magnitude is exactly 1. A "too hydrophobic" error (positive) times
    ``-w`` therefore down-weights hydrophobic AAs and up-weights hydrophilic ones.
    """
    kd = np.array([KD_HYDROPHOBICITY[a] for a in AA_ORDER], dtype=float)
    centered = kd - kd.mean()
    return centered / np.abs(centered).max()


def hydrophobic_surface_weights() -> np.ndarray:
    """Length-20 vector, positive on hydrophobic AAs (KD-scaled, max 1), 0 else.

    Used to distribute the per-position surface down-weight across hydrophobic
    residues in proportion to their hydrophobicity (Ile/Val/Leu move most).
    """
    w = np.zeros(20, dtype=float)
    kd_h = {a: KD_HYDROPHOBICITY[a] for a in HYDROPHOBIC_AAS}
    lo = min(kd_h.values())
    hi = max(kd_h.values())
    for a in HYDROPHOBIC_AAS:
        # map KD within the hydrophobic set to (0, 1]; the most hydrophobic -> 1
        w[AA_TO_IDX[a]] = (KD_HYDROPHOBICITY[a] - lo) / (hi - lo) * 0.8 + 0.2
    return w


def charge_weights(basic_downweight_scale: float = 0.35) -> np.ndarray:
    """Length-20 vector: +1 on D/E, ``-scale`` on K/R, 0 else (AA_ORDER).

    A "too positive / not negative enough" error (positive) times this vector
    up-weights acidic D/E and mildly down-weights basic K/R; the signs flip when
    the pool overshoots too negative.
    """
    w = np.zeros(20, dtype=float)
    for a in ACIDIC_AAS:
        w[AA_TO_IDX[a]] = 1.0
    for a in BASIC_AAS:
        w[AA_TO_IDX[a]] = -float(basic_downweight_scale)
    return w


# ---------------------------------------------------------------------------
# Configuration + axis declaration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ControlAxis:
    """Declarative description of one controlled property axis.

    Mirrors the shape of ``scoring.multi_objective.MetricSpec`` so a new axis
    (e.g. a future CamSol/DeepSP solubility predictor) is one registry entry with
    no change to the control logic. ``aa_weights`` is the length-20 projection
    (AA_ORDER) that maps a scalar drive ``u`` to per-AA nats.
    """
    name: str
    metric_column: str            # per-design column in the survivor pool
    target: float
    band_lo: float
    band_hi: float
    deadband: float               # physical units around target; |err|<=db -> 0
    error_scale: float            # physical units; normalizes err so gain is unitless
    scope: str                    # "global" or "surface"
    aa_weights: np.ndarray        # (20,) projection in AA_ORDER
    # axis-specific per-design "fails this filter" predicate, evaluated on the
    # pool dataframe; returns a boolean Series. Defaults to out-of-band on the
    # metric column.
    fail_high_column: Optional[str] = None   # gap column (>0 => fails high)
    fail_low_column: Optional[str] = None    # gap column (>0 => fails low)
    # Online plant-gain estimation is reliable only when the applied drive equals
    # what the controller requested. The GLOBAL charge axis is merged with the
    # class-balance bias_AA (which can suppress it), so its requested u may not be
    # what was applied -> disable the secant there and use the fixed gain.
    estimate_gain: bool = True
    # Which candidate pool this axis MEASURES. "seq" (default) = the per-cycle
    # seq-stage pool the controller reads today; a future axis on a different stage
    # (e.g. a struct-stage SAP axis, whose sap_corr_* only exists post-fold) sets
    # its own key and the driver passes that frame in the ``pools`` mapping.
    pool_key: str = "seq"


@dataclass
class AdaptiveBiasConfig:
    gain: float = 0.6              # initial integral gain (unitless error)
    max_nats: float = 0.6         # clamp on |u| (below PLM fusion mean_abs ~0.85)
    carry: float = 0.9            # integral HOLD factor (~1 holds bias; leak=1-carry)
    t_min: float = 2.5            # |t| threshold for the gate
    f_min: float = 0.15           # fail-fraction threshold for the gate
    min_n: int = 30               # minimum primary survivors to act on an axis
    mode: str = "proportional"    # "proportional" | "bangbang"
    bangbang_step: float = 0.3    # nats per step in bang-bang mode
    gain_correction_clip: float = 6.0   # cap range [1/c, c] for the online gain est;
    #                                     wide enough to attenuate a ~6x-too-strong
    #                                     plant to near-deadbeat (g_eff*|K| ~ 1).


@dataclass
class AxisState:
    """Cross-cycle memory for one axis (small; carried like throat_bias_prev)."""
    last_u: float = 0.0                       # u applied to produce the CURRENT pool
    history: list = field(default_factory=list)  # [(u_applied, resulting_mean), ...]

    def to_dict(self) -> dict:
        return {"last_u": self.last_u, "history": [list(p) for p in self.history]}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AxisState":
        if not d:
            return cls()
        return cls(last_u=float(d.get("last_u", 0.0)),
                   history=[tuple(p) for p in d.get("history", [])])


@dataclass
class AxisOutcome:
    name: str
    gate_open: bool
    reason: str
    n: int
    mean: float
    t_stat: float
    fail_fraction: float
    signed_error: float
    e_norm: float
    gain_correction: float
    frozen_wrong_sign: bool
    u: float                      # new drive scalar to apply this cycle
    scope: str


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wald_lower_bound(f: float, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wald CI for a proportion (clamped to >= 0)."""
    if n <= 0:
        return 0.0
    half = z * math.sqrt(max(0.0, f * (1.0 - f)) / n)
    return f - half


def _controller_pool(pool_df: pd.DataFrame) -> pd.DataFrame:
    """The candidate distribution the controller measures.

    Use the FULL per-cycle candidate pool (all sampled+scored designs), NOT just
    the post-filter survivors: when a cycle is failing badly almost nothing passes,
    so a survivors-only view would be empty exactly when the controller must act.
    The full pool is the true distribution the bias is shaping, and its fail-fraction
    is the honest "are we in trouble" signal. We only drop explicit rescue/backfill
    rows (deliberately-worst near-misses) when such a column is present — the
    per-cycle seq-stage pool has none, so this is usually a no-op.
    """
    if pool_df is None or len(pool_df) == 0:
        return pool_df
    if "selection__bucket" in pool_df.columns:
        mask = ~pool_df["selection__bucket"].astype(str).str.contains(
            "rescue|backfill", case=False, na=False)
        prim = pool_df[mask]
        if len(prim) > 0:
            return prim
    return pool_df


def axis_pool_stats(pool_df: pd.DataFrame, axis: ControlAxis) -> dict:
    """Mean / se / n / fail-fraction for one axis over the primary pool."""
    col = pool_df[axis.metric_column].astype(float)
    col = col[np.isfinite(col)]
    n = int(len(col))
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan"), "fail_fraction": 0.0}
    mean = float(col.mean())
    std = float(col.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else float("inf")
    # fail fraction: prefer the precomputed gap columns (already per-design,
    # band-relative). A design fails the axis if it is above the high bound or
    # below the low bound.
    fail = np.zeros(n, dtype=bool)
    used_gap = False
    if axis.fail_high_column and axis.fail_high_column in pool_df.columns:
        fh = pd.to_numeric(pool_df.loc[col.index, axis.fail_high_column], errors="coerce").fillna(0.0)
        fail = fail | (fh.to_numpy() > 0.0)
        used_gap = True
    if axis.fail_low_column and axis.fail_low_column in pool_df.columns:
        fl = pd.to_numeric(pool_df.loc[col.index, axis.fail_low_column], errors="coerce").fillna(0.0)
        fail = fail | (fl.to_numpy() > 0.0)
        used_gap = True
    if not used_gap:
        vals = col.to_numpy()
        fail = (vals > axis.band_hi) | (vals < axis.band_lo)
    return {"n": n, "mean": mean, "se": se, "fail_fraction": float(fail.mean())}


# ---------------------------------------------------------------------------
# Online plant-gain estimate + wrong-sign freeze
# ---------------------------------------------------------------------------
def estimate_gain_correction(history: list, cfg: AdaptiveBiasConfig,
                             error_scale: float) -> tuple[float, bool]:
    """Secant estimate of plant gain -> (gain_correction, wrong_sign).

    history is [(u_applied, resulting_mean), ...]. We pair the latest point with the
    most recent EARLIER point whose ``u`` differs meaningfully (so a held/converged
    axis with du~0 is not used — that would divide by ~0 and explode on sampling
    noise, a spurious wrong-sign freeze). K = d(mean)/d(u) in physical units.

    The drive is ``u = gain * corr * (e / error_scale)`` and we want a near-deadbeat
    step ``K * u ~ -e``, so ``corr = error_scale / (gain * |K|)`` (the error_scale
    factor was the missing unit). K<0 is the expected sign (a hydrophilic/acidic push
    lowers the mean); a positive K means the pool moved the wrong way -> freeze.
    """
    if len(history) < 2:
        return 1.0, False
    u_last, m_last = history[-1]
    du_min = max(1e-2, 0.05 * cfg.max_nats)
    u_prev = m_prev = None
    for u0, m0 in reversed(history[:-1]):
        if abs(u_last - u0) >= du_min:
            u_prev, m_prev = u0, m0
            break
    if u_prev is None:
        # No distinct-u history yet (held/converged): trust the current gain.
        return 1.0, False
    du = u_last - u_prev
    K = (m_last - m_prev) / du
    if abs(K) < 1e-9:
        return 1.0, False
    if K > 0:
        return 0.0, True                       # wrong-sign plant -> freeze
    corr = error_scale / max(1e-6, cfg.gain * abs(K))
    corr = float(np.clip(corr, 1.0 / cfg.gain_correction_clip, cfg.gain_correction_clip))
    return corr, False


# ---------------------------------------------------------------------------
# Per-axis drive
# ---------------------------------------------------------------------------
def signed_deadband_error(mean: float, axis: ControlAxis) -> float:
    """Signed physical error of the pool mean vs target, with a deadband.

    Positive = pool is on the "too hydrophobic / too positive" side of target.
    Zero inside the deadband (so a pool already near target is left alone).
    """
    e = mean - axis.target
    if abs(e) <= axis.deadband:
        return 0.0
    return math.copysign(abs(e) - axis.deadband, e)


def step_axis(pool_df: pd.DataFrame, axis: ControlAxis, state: AxisState,
              cfg: AdaptiveBiasConfig) -> tuple[AxisOutcome, AxisState]:
    """One control step for one axis: gate, estimate gain, drive, advance memory.

    Returns ``(outcome, new_state)``. ``new_state`` carries the drive applied this
    cycle (``last_u``) and the (u, resulting-mean) history used for the online gain
    estimate. History is advanced in exactly one place (here) to avoid drift.
    """
    stats = axis_pool_stats(pool_df, axis)
    n, mean, se, f = stats["n"], stats["mean"], stats["se"], stats["fail_fraction"]

    # Close the loop on the PREVIOUS action: pair the u we applied last cycle with
    # the pool mean it produced (observed now). This is what the secant estimate
    # of plant gain consumes.
    history = list(state.history)
    if n > 0 and math.isfinite(mean):
        history.append((float(state.last_u), float(mean)))
    history = history[-4:]

    if axis.estimate_gain:
        gain_corr, wrong_sign = estimate_gain_correction(history, cfg, axis.error_scale)
    else:
        gain_corr, wrong_sign = 1.0, False

    # ---- GATE (binary, statistical) ----
    # A zero-variance pool (all designs identical) has se=0; if it is also out of
    # band that is the WORST case (collapsed/deadbeat to a wrong value), so treat
    # |t| as infinite rather than 0 (which would fool the gate into closing).
    if se and math.isfinite(se) and se > 0:
        t_stat = (mean - axis.target) / se
    elif math.isfinite(mean) and abs(mean - axis.target) > 1e-9:
        t_stat = math.copysign(float("inf"), mean - axis.target)
    else:
        t_stat = 0.0
    reasons = []
    if n < cfg.min_n:
        reasons.append(f"N={n}<{cfg.min_n}")
    if abs(t_stat) <= cfg.t_min:
        reasons.append(f"|t|={abs(t_stat):.2f}<={cfg.t_min}")
    if not (f > cfg.f_min and wald_lower_bound(f, n) > 0.0):
        reasons.append(f"fail_frac={f:.2f} not sig (>{cfg.f_min}, CI>0)")
    gate_open = len(reasons) == 0

    # ---- DRIVE (continuous, physical) ----
    se_err = signed_deadband_error(mean, axis) if math.isfinite(mean) else 0.0
    e_norm = se_err / axis.error_scale if axis.error_scale else 0.0

    if wrong_sign:
        # Plant responded the wrong way -> stop biasing this axis entirely.
        u_new = 0.0
        reason = "frozen: wrong-sign plant response"
    elif not gate_open:
        # Pool healthy on this axis -> PURELY HOLD the achieved bias (do NOT release
        # or leak, or a near-static plant slowly reverts). The bias is what keeps the
        # pool in-spec; maintain it exactly. The gate re-opens (and integration
        # resumes/reverses) only if the pool drifts back out of band.
        u_new = float(np.clip(state.last_u, -cfg.max_nats, cfg.max_nats))
        reason = "gate closed (hold): " + "; ".join(reasons)
    elif e_norm == 0.0:
        # In trouble overall but within the deadband of target -> hold, don't chase.
        u_new = float(np.clip(state.last_u, -cfg.max_nats, cfg.max_nats))
        reason = "in deadband (hold)"
    elif cfg.mode == "bangbang":
        step = math.copysign(cfg.bangbang_step, e_norm)
        u_new = float(np.clip(cfg.carry * state.last_u + step, -cfg.max_nats, cfg.max_nats))
        reason = "active (bangbang integrate)"
    else:
        g_eff = cfg.gain * gain_corr
        u_new = float(np.clip(cfg.carry * state.last_u + g_eff * e_norm,
                              -cfg.max_nats, cfg.max_nats))
        reason = "active (integral)"

    outcome = AxisOutcome(
        name=axis.name, gate_open=gate_open, reason=reason, n=n, mean=mean,
        t_stat=t_stat, fail_fraction=f, signed_error=se_err, e_norm=e_norm,
        gain_correction=gain_corr, frozen_wrong_sign=wrong_sign, u=u_new,
        scope=axis.scope,
    )
    # On a wrong-sign freeze, clear history so recovery is deterministic (the next
    # cycle re-estimates from fresh points) rather than relying on stale pairings.
    new_history = [] if wrong_sign else history
    return outcome, AxisState(last_u=u_new, history=new_history)


# ---------------------------------------------------------------------------
# Composition: global per-AA bias + class-balance reconciliation
# ---------------------------------------------------------------------------
def parse_bias_AA(s: str) -> dict:
    """Parse a LigandMPNN ``bias_AA`` string 'E:-1.40,D:1.45' -> {AA: nats}."""
    out: dict = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        aa, val = part.split(":", 1)
        aa = aa.strip().upper()
        try:
            out[aa] = float(val)
        except ValueError:
            continue
    return out


def format_bias_AA(d: dict, drop_below: float = 0.05) -> str:
    """Serialize {AA: nats} -> 'AA:val,...' (sorted, near-zeros dropped)."""
    parts = [f"{aa}:{val:.2f}" for aa, val in sorted(d.items()) if abs(val) > drop_below]
    return ",".join(parts)


def merge_bias_AA_strings(existing: str, controller: dict,
                          max_nats: float = CLASS_BALANCE_MAX_NATS) -> tuple[str, list]:
    """Compose the controller's global per-AA bias with the class-balance string.

    Policy (class-balance is the validated, larger-magnitude actuator and wins):
      * class-balance already moved AA the SAME direction -> controller SKIPS it
        (no stacking; class-balance already addressed that AA).
      * class-balance moved AA the OPPOSITE direction      -> class-balance WINS;
        controller backs off (telemetered conflict).
      * class-balance did not touch AA                     -> controller applies.
    Final values are clamped to +-max_nats. Returns (string, conflicts).
    """
    base = parse_bias_AA(existing)
    result = dict(base)
    conflicts = []
    for aa, val in controller.items():
        if abs(val) <= 1e-9:
            continue
        if aa in base and abs(base[aa]) > 1e-9:
            same = (base[aa] > 0) == (val > 0)
            conflicts.append({"aa": aa, "class_balance": round(base[aa], 3),
                              "controller": round(val, 3),
                              "resolution": "skip_same_dir" if same else "class_balance_wins"})
            continue  # class-balance wins either way
        result[aa] = val
    for aa in list(result.keys()):
        result[aa] = float(np.clip(result[aa], -max_nats, max_nats))
    return format_bias_AA(result), conflicts


def global_per_aa_bias(outcomes: list, axes_by_name: dict,
                       cfg: AdaptiveBiasConfig) -> dict:
    """Sum the open GLOBAL axes' per-AA contributions with a shared budget cap.

    Each axis contributes ``u * aa_weights``. To stop two axes from stacking the
    same AA past the clamp, the summed vector is rescaled so its max magnitude is
    at most ``max_nats`` before the per-entry clamp.
    """
    vec = np.zeros(20, dtype=float)
    for oc in outcomes:
        # Emit any NONZERO drive, whether it is actively integrating or being HELD
        # (gate closed). Gating emission on gate_open would drop the held bias the
        # cycle the pool looks healthy -> the pool reverts -> limit cycle (the exact
        # failure the integral-hold design prevents).
        if oc.scope != "global" or abs(oc.u) < 1e-9:
            continue
        vec = vec + oc.u * axes_by_name[oc.name].aa_weights
    peak = np.abs(vec).max()
    if peak > cfg.max_nats:
        vec = vec * (cfg.max_nats / peak)
    vec = np.clip(vec, -cfg.max_nats, cfg.max_nats)
    return {AA_ORDER[i]: float(vec[i]) for i in range(20) if abs(vec[i]) > 1e-9}


# ---------------------------------------------------------------------------
# Per-position surface delta (L, 20), vectorized
# ---------------------------------------------------------------------------
def build_surface_delta(outcome: AxisOutcome, *, L: int, position_classes: list,
                        sasa_fraction: Optional[np.ndarray], fixed_idx: set,
                        cfg: AdaptiveBiasConfig,
                        over_rep_mask: Optional[np.ndarray] = None,
                        surface_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """(L, 20) bias delta: down-weight hydrophobic AAs at exposed surface positions.

    delta[i, a] = -u * expo_i * w_hydro[a]   for surface, non-fixed positions i and
    hydrophobic AAs a. Sign follows ``u`` (so an over-corrected, too-hydrophilic pool
    reverses to up-weight). Buried / catalytic positions stay 0.

    ``over_rep_mask`` (length 20, bool) optionally restricts the down-weight to AAs
    that are statistically over-represented (the "way over-represented" gate). It is
    recomputed each cycle from the current pool, so while a bias is held the focus
    tracks whichever hydrophobic AA is currently most over-represented (intentional
    adaptive re-focusing; the magnitude ``u`` is what is held). Applied only on the
    down-weight (u>0); see below.
    """
    delta = np.zeros((L, 20), dtype=np.float32)
    # Emit any nonzero drive (held or active) — see global_per_aa_bias for why we do
    # NOT gate emission on gate_open.
    if abs(outcome.u) < 1e-9:
        return delta
    w_hydro = hydrophobic_surface_weights()
    # Focus the DOWN-weight (u>0) on the over-represented hydrophobic AAs. On a
    # reversal (u<0, pool overshot too hydrophilic) we are UP-weighting hydrophobics
    # to add some back — do NOT restrict that to the already-over-represented AA;
    # spread it across all hydrophobic AAs.
    if over_rep_mask is not None and outcome.u > 0:
        w_hydro = w_hydro * over_rep_mask.astype(float)
    for i in range(L):
        if surface_mask is not None:
            # Opt-in non_tunnel_surface (or any precomputed) membership mask.
            if not surface_mask[i]:
                continue
        else:
            # LEGACY membership (unchanged → byte-identical): distal_surface,
            # non-fixed (the active-site/binding region is excluded by class).
            if i in fixed_idx:
                continue
            cls = position_classes[i] if i < len(position_classes) else None
            if cls not in SURFACE_CLASSES:
                continue
        # Per-position magnitude scales with SASA fraction (shared by both modes).
        expo = 1.0
        if sasa_fraction is not None and i < len(sasa_fraction):
            expo = float(sasa_fraction[i])
            if not math.isfinite(expo) or expo <= 0.0:
                continue
        contrib = -outcome.u * expo * w_hydro
        contrib = np.clip(contrib, -cfg.max_nats, cfg.max_nats)
        delta[i, :] = contrib
    return delta


# ---------------------------------------------------------------------------
# Axis registry factory
# ---------------------------------------------------------------------------
def default_axes(*, gravy_target: float = -0.2, gravy_band: tuple = (-0.8, 0.3),
                 net_charge_target: float = -10.0,
                 net_charge_band: tuple = (-18.0, -4.0),
                 deadband_frac: float = 0.25,
                 basic_downweight_scale: float = 0.35,
                 charge_band: Optional[tuple] = None,
                 axes: Optional[list] = None) -> list:
    """Build the controller axis registry (global charge + surface hydrophobicity).

    Targets default to ``scoring.multi_objective.DEFAULT_METRIC_SPECS``; bands come
    from the cycle's filter configuration. The deadband (``deadband_frac`` of the
    band half-width) keeps the controller from chasing tiny deviations while still
    aiming close to target; ``error_scale`` normalizes the drive so one ``gain``
    fits both axes.

    ``charge_band`` (opt-in ``--adaptive_charge_band``, e.g. ``(-15, -5)``):
    overrides the charge band, sets the charge target to the band MIDPOINT, and —
    critically — NULLS the charge axis's precomputed gap columns. Those gaps were
    computed against the *cycle filter* band, so a custom adaptive band must fall
    back to raw-value band evaluation or the fail-fraction would be wrong (codex).
    Validated finite with ``lo < hi``.

    ``axes`` (opt-in ``--adaptive_bias_axes``): the ordered subset of axis names to
    return; default ``["charge", "surface_hydrophobicity"]`` (today's behavior). An
    unknown name is a configuration error (fail-fast). This is the extension point
    for future registry entries (a struct-stage SAP axis, etc.).
    """
    charge_fail_high: Optional[str] = "selection__seq_filter_gap_charge_high"
    charge_fail_low: Optional[str] = "selection__seq_filter_gap_charge_low"
    if charge_band is not None:
        lo, hi = float(charge_band[0]), float(charge_band[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(
                f"charge_band must be finite (lo, hi) with lo < hi, got {charge_band!r}")
        net_charge_band = (lo, hi)
        net_charge_target = (lo + hi) / 2.0
        charge_fail_high = None     # cycle-band gaps are wrong for a custom band
        charge_fail_low = None      # -> force raw-value band evaluation

    g_lo, g_hi = gravy_band
    g_half = max(g_hi - gravy_target, gravy_target - g_lo)
    c_lo, c_hi = net_charge_band
    c_half = max(c_hi - net_charge_target, net_charge_target - c_lo)
    registry = {
        "charge": ControlAxis(
            name="charge", metric_column="net_charge_full_HH",
            target=net_charge_target, band_lo=c_lo, band_hi=c_hi,
            deadband=deadband_frac * c_half, error_scale=c_half, scope="global",
            aa_weights=charge_weights(basic_downweight_scale),
            fail_high_column=charge_fail_high,
            fail_low_column=charge_fail_low,
            estimate_gain=False,   # merged with class-balance -> secant unreliable
        ),
        "surface_hydrophobicity": ControlAxis(
            name="surface_hydrophobicity", metric_column="gravy",
            target=gravy_target, band_lo=g_lo, band_hi=g_hi,
            deadband=deadband_frac * g_half, error_scale=g_half, scope="surface",
            aa_weights=kd_centered_weights(),
            # gap_gravy is a two-sided interval gap (>0 below gravy_min OR above
            # gravy_max), so it captures both the too-hydrophobic and the
            # over-corrected too-hydrophilic directions via this single column.
            fail_high_column="selection__seq_filter_gap_gravy",
            fail_low_column=None,
        ),
    }
    selected = axes if axes is not None else ["charge", "surface_hydrophobicity"]
    if not selected:
        raise ValueError("adaptive-bias axes selection is empty")
    if len(selected) != len(set(selected)):
        raise ValueError(
            f"duplicate adaptive-bias axes would double-stack an actuator: {selected}")
    out = []
    for name in selected:
        if name not in registry:
            raise ValueError(
                f"unknown adaptive-bias axis {name!r}; choose from {sorted(registry)}")
        out.append(registry[name])
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class AdaptiveBiasResult:
    bias_AA_string: str                 # merged global bias_AA (controller + class-balance)
    controller_global: dict             # RAW per-AA global bias (unmerged), to carry forward
    per_position_delta: np.ndarray      # (L, 20) surface delta to add to bias_k
    outcomes: list                      # list[AxisOutcome]
    new_state: dict                     # {axis_name: AxisState.to_dict()}
    telemetry: dict
    conflicts: list


def compute_adaptive_bias(*, pool_df: Optional[pd.DataFrame] = None,
                          pools: Optional[dict] = None, axes: list,
                          cfg: AdaptiveBiasConfig, state: Optional[dict],
                          L: int, position_classes: list,
                          sasa_fraction: Optional[np.ndarray], fixed_idx: set,
                          class_balance_bias_AA: str = "",
                          over_rep_mask: Optional[np.ndarray] = None,
                          surface_mask: Optional[np.ndarray] = None
                          ) -> AdaptiveBiasResult:
    """Run all axes for one cycle and produce the global + per-position biases.

    Each axis measures the pool named by its ``pool_key`` in the ``pools`` mapping.
    Back-compat: a lone ``pool_df`` is treated as ``pools={"seq": pool_df}`` (every
    default axis is ``pool_key="seq"``), so existing single-pool callers are
    unchanged. Returns an :class:`AdaptiveBiasResult`. When an axis's pool is
    empty/None or no gate opens, the per-position delta is all-zero and the merged
    bias_AA is just the class-balance string unchanged (no-op when nothing to do).
    """
    state = state or {}
    delta = np.zeros((L, 20), dtype=np.float32)
    outcomes: list = []
    new_state: dict = {}

    if pools is None:
        pools = {"seq": pool_df}
    # Reduce each named pool to its controller distribution once (a key may back
    # several axes); a None/empty frame becomes None (those axes hold).
    controller_pools = {
        k: (_controller_pool(v) if v is not None else None) for k, v in pools.items()
    }

    for axis in axes:
        st = AxisState.from_dict(state.get(axis.name))
        primary = controller_pools.get(axis.pool_key)
        have_pool = primary is not None and len(primary) > 0
        if not have_pool or axis.metric_column not in primary.columns:
            # No measurement this cycle: HOLD the previously-applied bias (keep
            # last_u and emit it) rather than dropping it. History is preserved.
            held = AxisOutcome(
                name=axis.name, gate_open=False, reason="no pool (hold)", n=0,
                mean=float("nan"), t_stat=0.0, fail_fraction=0.0, signed_error=0.0,
                e_norm=0.0, gain_correction=1.0, frozen_wrong_sign=False,
                u=float(st.last_u), scope=axis.scope)
            outcomes.append(held)
            new_state[axis.name] = AxisState(last_u=st.last_u, history=st.history).to_dict()
            if axis.scope == "surface":
                delta = delta + build_surface_delta(
                    held, L=L, position_classes=position_classes,
                    sasa_fraction=sasa_fraction, fixed_idx=fixed_idx, cfg=cfg,
                    over_rep_mask=over_rep_mask, surface_mask=surface_mask)
            continue
        oc, st_new = step_axis(primary, axis, st, cfg)
        outcomes.append(oc)
        new_state[axis.name] = st_new.to_dict()
        if axis.scope == "surface":
            delta = delta + build_surface_delta(
                oc, L=L, position_classes=position_classes,
                sasa_fraction=sasa_fraction, fixed_idx=fixed_idx, cfg=cfg,
                over_rep_mask=over_rep_mask, surface_mask=surface_mask)

    axes_by_name = {a.name: a for a in axes}
    controller_global = global_per_aa_bias(outcomes, axes_by_name, cfg)
    merged, conflicts = merge_bias_AA_strings(class_balance_bias_AA, controller_global)

    telemetry = {
        "axes": [vars(oc) | {"u": round(oc.u, 4)} for oc in outcomes],
        "controller_global_bias_AA": controller_global,
        "merged_bias_AA": merged,
        "class_balance_conflicts": conflicts,
        "n_surface_positions_touched": int((np.abs(delta) > 1e-6).any(axis=1).sum()),
        "max_surface_penalty_nats": float(delta.min()) if delta.size else 0.0,
        "config": vars(cfg),
    }
    return AdaptiveBiasResult(
        bias_AA_string=merged, controller_global=controller_global,
        per_position_delta=delta, outcomes=outcomes, new_state=new_state,
        telemetry=telemetry, conflicts=conflicts)


def hydrophobic_over_rep_mask(pool_sequences, *,
                              reference: str = "swissprot_ec3_hydrolases_2026_01",
                              z_thresh: float = 3.0) -> Optional[np.ndarray]:
    """(20,) bool-ish mask of hydrophobic AAs that are 'way over-represented'.

    Pools the survivor sequences and z-scores their composition against the hydrolase
    reference (reusing ``aa_composition.aa_z_scores``, already used by the class-
    balance path). Returns a length-20 mask (1 for hydrophobic AAs with z > z_thresh)
    or ``None`` if no hydrophobic AA is over-represented (the surface term then spreads
    across all hydrophobic AAs — the aggregate-GRAVY signal still applies). This
    implements the "OR an amino acid is way over-represented" focusing gate.
    """
    seqs = [s for s in (pool_sequences or []) if isinstance(s, str) and s]
    if not seqs:
        return None
    try:
        from protein_chisel.expression.aa_composition import aa_z_scores
        z = aa_z_scores("".join(seqs), reference=reference)
    except Exception:                              # pragma: no cover - defensive
        return None
    mask = np.zeros(20, dtype=float)
    for a in HYDROPHOBIC_AAS:
        if z.get(a, 0.0) > z_thresh:
            mask[AA_TO_IDX[a]] = 1.0
    return mask if mask.sum() > 0 else None


def seed_warmstart(*, seed_metrics: dict, axes: list, cfg: AdaptiveBiasConfig,
                   L: int, position_classes: list,
                   sasa_fraction: Optional[np.ndarray], fixed_idx: set,
                   strength: float = 0.5,
                   surface_mask: Optional[np.ndarray] = None
                   ) -> tuple[dict, np.ndarray, dict]:
    """Warm-start the controller from the INPUT scaffold's scalar properties.

    Cycle 0 has no candidate pool yet. When ``--adaptive_bias_seed_from_input`` is
    set we instead use the seed's single GRAVY / net-charge values to pre-bias cycle
    0 in the correcting direction. There is no statistical gate (a single structure
    is a point estimate, not a distribution), so the push is deliberately gentle
    (``strength`` x the normal gain) and only fires when the seed sits OUTSIDE the
    deadband. The closed loop then refines from cycle 1 onward.

    Returns ``(global_bias_AA_dict, per_position_delta, state, telemetry)`` where
    ``state`` seeds each axis's ``last_u`` so the end-of-cycle-0 gain estimate
    correctly pairs the seed bias with the pool it produced.
    """
    delta = np.zeros((L, 20), dtype=np.float32)
    outcomes: list = []
    tele: dict = {}
    state: dict = {}
    for axis in axes:
        val = seed_metrics.get(axis.metric_column)
        if val is None or not math.isfinite(val):
            continue
        e = signed_deadband_error(float(val), axis)
        if e == 0.0:
            tele[axis.name] = {"seed_value": val, "u": 0.0, "in_deadband": True}
            continue
        u = float(np.clip(strength * cfg.gain * (e / axis.error_scale),
                          -cfg.max_nats, cfg.max_nats))
        oc = AxisOutcome(name=axis.name, gate_open=True, reason="seed warm-start",
                         n=0, mean=float(val), t_stat=0.0, fail_fraction=0.0,
                         signed_error=e, e_norm=e / axis.error_scale,
                         gain_correction=1.0, frozen_wrong_sign=False, u=u,
                         scope=axis.scope)
        outcomes.append(oc)
        state[axis.name] = AxisState(last_u=u, history=[]).to_dict()
        tele[axis.name] = {"seed_value": val, "u": round(u, 4)}
        if axis.scope == "surface":
            delta = delta + build_surface_delta(
                oc, L=L, position_classes=position_classes,
                sasa_fraction=sasa_fraction, fixed_idx=fixed_idx, cfg=cfg,
                surface_mask=surface_mask)
    axes_by_name = {a.name: a for a in axes}
    global_bias = global_per_aa_bias(outcomes, axes_by_name, cfg)
    return global_bias, delta, state, {"seed_warmstart": tele}


__all__ = [
    "AA_ORDER", "KD_HYDROPHOBICITY", "HYDROPHOBIC_AAS", "ACIDIC_AAS", "BASIC_AAS",
    "SURFACE_CLASSES", "STEERABLE_SURFACE_CLASSES", "surface_scope",
    "ControlAxis", "AdaptiveBiasConfig", "AxisState",
    "AxisOutcome", "AdaptiveBiasResult", "kd_centered_weights",
    "hydrophobic_surface_weights", "charge_weights", "wald_lower_bound",
    "axis_pool_stats", "estimate_gain_correction", "signed_deadband_error",
    "step_axis", "parse_bias_AA", "format_bias_AA",
    "merge_bias_AA_strings", "global_per_aa_bias", "build_surface_delta",
    "default_axes", "compute_adaptive_bias", "seed_warmstart",
    "hydrophobic_over_rep_mask",
]
