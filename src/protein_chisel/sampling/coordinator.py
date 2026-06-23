"""CF-3 multi-objective controller **coordinator** (pure, opt-in).

Generalises :func:`adaptive_bias.global_per_aa_bias` ("sum a (20,) vector then
rescale") into a per-cell, weight-partitioned, work-conserving, **signed-sum-bounded**
odds budget, reused for BOTH the (20,) global per-AA vector AND each row of the
(L, 20) surface delta. Many control axes can declare the SAME ``actuator`` (CF-2);
the coordinator resolves a shared actuator by the **sign** of its grouped drives:

  * same-sign  → ``sign · max(|drive|)``  (MAX-not-sum: two agreeing objectives are
                 satisfied by the single larger push — the double-count is
                 structurally impossible);
  * mixed-sign → signed ``Σ drive``       (a genuine trade-off; net intent wins).

The budget itself is bounded on the **signed sum** of grants (``|Σ grant| ≤
budget_nats``) with a final unconditional clip (review BUG-A), and the controller
budget nests INSIDE the whole-stack total ceiling with reserved headroom so both
ceilings bind simultaneously (review BUG-B, enforced by :func:`nested_total_clip`).

Everything here is expressed in **nats** and bounded via
``bias_scale.nats_for_odds(odds, T_apply)`` at the *application-cycle* temperature, so
the realized odds ``exp(bias / T)`` are temperature-invariant (math review §2.4): no
bias source can be silently over-emphasised relative to the MPNN logits, because the
coordinator keeps the controller share ≤ ``CONTROLLER_CEILING`` and the whole stack
≤ ``TOTAL_CEILING`` at the actual sampling T.

All functions are pure (numpy / stdlib only) and do no I/O, so the combination
arithmetic is unit-testable without models, PyRosetta, or the cluster.
"""
from __future__ import annotations

import math

import numpy as np

from protein_chisel.sampling.adaptive_bias import (
    AA_ORDER, build_surface_delta,
)
from protein_chisel.sampling.bias_scale import (
    ODDS_STRONG, nats_for_odds,
)

# ---------------------------------------------------------------------------
# Ceilings (odds space; temperature-invariant — converted to nats at T_apply).
# ---------------------------------------------------------------------------
# Joint CONTROLLER budget: 8x (ODDS_STRONG). Moves a pool meaningfully (p_top≈0.89 on
# a contested cell) without locking it; the controller share never exceeds this.
CONTROLLER_CEILING = ODDS_STRONG          # 8.0
# Whole-stack ceiling (PLM + consensus + throat + controllers): ~1e3x. At 1e4x a cell
# is already deterministic; 1e3x still lets the PLM express a real strong preference
# while leaving ~0.1% on alternatives (math review §5.3). The controller's reserved
# headroom is carved out INSIDE this.
TOTAL_CEILING = 1.0e3


def _actuator_key(axis) -> str:
    """Grouping key: the declared actuator, or the axis name (legacy ⇒ own group)."""
    return axis.actuator or axis.name


def _resolve_group_drive(drives: list) -> float:
    """Collapse the per-axis desired drives of ONE actuator group to a single drive.

    ``drives`` are signed nats (``u · weight_proj[aa]``) for the axes sharing the
    actuator AT THIS CELL. The selector is the SIGN (review §4.2):

      * every drive shares a sign → ``sign · max(|drive|)``  (MAX-not-sum;
        anti-double-count for correlated, agreeing objectives);
      * signs disagree            → signed ``Σ drive``       (genuine trade-off).

    A drive of exactly 0 carries no sign and never forces the mixed-sign branch
    (an idle axis on a shared actuator must not flip max → sum).
    """
    nz = [d for d in drives if d != 0.0]
    if not nz:
        return 0.0
    pos = any(d > 0 for d in nz)
    neg = any(d < 0 for d in nz)
    if pos and neg:
        return float(sum(nz))                       # mixed sign → signed sum
    sign = 1.0 if pos else -1.0
    return sign * max(abs(d) for d in nz)            # same sign → max-by-magnitude


def _allocate_cell(group_drives: dict, group_weights: dict, group_priority: dict,
                   budget_nats: float) -> float:
    """Weight-partition ``budget_nats`` across one cell's per-actuator drives.

    ``group_drives`` maps actuator → already-sign-resolved signed drive (nats).
    Returns the budgeted cell controller bias (a single signed nats value), with the
    **signed-sum** invariant ``|Σ grant| ≤ budget_nats`` enforced and a final
    unconditional clip (review BUG-A). Work-conserving: leftover budget is re-lent to
    groups that still want more, in deterministic ``(priority, name)`` order, checking
    the SIGNED running total — never ``Σ|grant|``.
    """
    active = [g for g, d in group_drives.items() if d != 0.0]
    if not active:
        return 0.0
    w_total = sum(group_weights[g] for g in active)
    if w_total <= 0.0:
        # Degenerate (all weights 0) — fall back to an equal split so the budget is
        # still honoured rather than dividing by zero.
        w_total = float(len(active))
        share = {g: budget_nats / w_total for g in active}
    else:
        share = {g: budget_nats * group_weights[g] / w_total for g in active}

    grant = {g: math.copysign(min(abs(group_drives[g]), share[g]), group_drives[g])
             for g in active}
    order = sorted(active, key=lambda g: (group_priority[g], g))
    for g in order:
        # Re-check the SIGNED running total each step (NOT Σ|grant|, BUG-A). ``continue``
        # (not ``break``) so a later OPPOSITE-sign group — which would REDUCE |Σ grant|
        # and thus free headroom — still gets its turn; a same-sign group with no
        # headroom simply takes nothing this pass.
        signed_sum = sum(grant.values())
        headroom = budget_nats - abs(signed_sum)
        want = abs(group_drives[g]) - abs(grant[g])
        if headroom <= 0.0 or want <= 0.0:
            continue
        grant[g] += math.copysign(min(want, headroom), group_drives[g])
    cell = sum(grant.values())
    # Final UNCONDITIONAL clip on the signed sum (belt-and-suspenders, BUG-A).
    return float(np.clip(cell, -budget_nats, budget_nats))


def _grouped(axes_by_name: dict, drive_of: dict):
    """Return (groups present, weight, priority) maps over the NON-veto budget axes.

    Group weight = the MAX axis weight in the group (adding a second axis on the same
    actuator must not inflate the shared budget share — consistent with max-not-sum);
    group priority = the MIN axis priority (earliest re-lend wins for the group).
    """
    g_axes: dict = {}
    g_weight: dict = {}
    g_priority: dict = {}
    for name, u in drive_of.items():
        ax = axes_by_name[name]
        if getattr(ax, "tier", "budget") == "veto":
            continue
        key = _actuator_key(ax)
        g_axes.setdefault(key, []).append(name)
        w = float(getattr(ax, "weight", 1.0))
        p = int(getattr(ax, "priority", 100))
        g_weight[key] = max(g_weight.get(key, w), w)
        g_priority[key] = min(g_priority.get(key, p), p)
    return g_axes, g_weight, g_priority


def coordinate_global(outcomes: list, axes_by_name: dict, *,
                      budget_nats: float) -> dict:
    """Budget-bounded GLOBAL per-AA controller bias ({AA: nats}).

    Drop-in generalisation of :func:`adaptive_bias.global_per_aa_bias` for the (20,)
    global vector: per-AA, group by actuator (sign-selected max/sum), weight-partition
    the joint ``budget_nats`` with the signed-sum invariant + final clip. ``veto``-tier
    axes bypass the budget entirely (added on top, never diluted).
    """
    drive_of = {oc.name: float(oc.u) for oc in outcomes
                if oc.scope == "global" and abs(oc.u) > 1e-12}
    veto = np.zeros(20, dtype=float)
    for oc in outcomes:
        ax = axes_by_name[oc.name]
        if oc.scope == "global" and getattr(ax, "tier", "budget") == "veto" \
                and abs(oc.u) > 1e-12:
            veto = veto + oc.u * np.asarray(ax.aa_weights, dtype=float)
    g_axes, g_weight, g_priority = _grouped(axes_by_name, drive_of)
    out: dict = {}
    for i in range(20):
        aa = AA_ORDER[i]
        group_drives: dict = {}
        for key, names in g_axes.items():
            drives = [drive_of[n] * float(axes_by_name[n].aa_weights[i])
                      for n in names]
            group_drives[key] = _resolve_group_drive(drives)
        cell = _allocate_cell(group_drives, g_weight, g_priority, budget_nats)
        total = cell + float(veto[i])
        if abs(total) > 1e-9:
            out[aa] = total
    return out


def coordinate_surface(outcomes: list, axes_by_name: dict, *,
                       budget_nats: float, L: int, position_classes: list,
                       sasa_fraction, fixed_idx: set, cfg,
                       over_rep_mask=None, surface_mask=None) -> np.ndarray:
    """Budget-bounded (L, 20) SURFACE delta across all surface-scope axes.

    Reuses :func:`adaptive_bias.build_surface_delta` to project each surface axis to
    its own (L, 20) contribution (so the position gate / SASA scaling / over-rep focus
    are IDENTICAL to the legacy path), then applies the SAME per-cell actuator-grouped
    budget (:func:`_allocate_cell`) row-by-row. With a single surface axis (today's
    default) the per-cell collapse is the identity on that axis's drive, so the only
    effect vs legacy is the budget clamp — which the caller sizes generously when not
    coordinating.
    """
    surf_ocs = [oc for oc in outcomes if oc.scope == "surface"]
    if not surf_ocs:
        return np.zeros((L, 20), dtype=np.float32)
    # Per-axis (L,20) contributions via the shared, byte-identical projector.
    contrib = {}
    for oc in surf_ocs:
        if abs(oc.u) <= 1e-12:
            continue
        contrib[oc.name] = build_surface_delta(
            oc, L=L, position_classes=position_classes,
            sasa_fraction=sasa_fraction, fixed_idx=fixed_idx, cfg=cfg,
            over_rep_mask=over_rep_mask, surface_mask=surface_mask)
    if not contrib:
        return np.zeros((L, 20), dtype=np.float32)
    drive_of = {n: 1.0 for n in contrib}   # the (L,20) already carries u; group by name
    g_axes, g_weight, g_priority = _grouped(axes_by_name, drive_of)
    out = np.zeros((L, 20), dtype=np.float32)
    veto_names = {n for n in contrib
                  if getattr(axes_by_name[n], "tier", "budget") == "veto"}
    for i in range(L):
        for a in range(20):
            group_drives: dict = {}
            for key, names in g_axes.items():
                drives = [float(contrib[n][i, a]) for n in names]
                group_drives[key] = _resolve_group_drive(drives)
            cell = _allocate_cell(group_drives, g_weight, g_priority, budget_nats)
            veto_val = sum(float(contrib[n][i, a]) for n in veto_names)
            out[i, a] = cell + veto_val
    return out


def nested_total_clip(rest: np.ndarray, controller: np.ndarray, *,
                      total_nats: float, reserve_nats: float) -> np.ndarray:
    """Nest the controller budget INSIDE the total ceiling (review BUG-B).

    ``cell_total = clip(rest, ±(total_nats − reserve_nats)) + clip(controller,
    ±reserve_nats)`` so ``|cell_total| ≤ total_nats`` AND the controller always keeps
    its reserved ``reserve_nats`` of headroom — neither ceiling shadows the other. The
    reserve is carved out of the total for the controller even where the controller is
    silent (a saturated ``rest`` cell tops out at ``total_nats − reserve_nats``).

    The controller reserve is capped at the total ceiling (``reserve = min(reserve,
    total)``): a configured controller ceiling LARGER than the whole-stack ceiling
    cannot break ``|cell_total| ≤ total_nats`` — it just collapses the ``rest`` budget
    to zero (the controller owns the whole ceiling), never overruns it.
    """
    reserve = min(max(0.0, reserve_nats), max(0.0, total_nats))
    outer = max(0.0, total_nats - reserve)
    r = np.clip(rest, -outer, outer)
    c = np.clip(controller, -reserve, reserve)
    return r + c


def coordinate(outcomes: list, axes_by_name: dict, cfg, T_apply: float, *,
               L: int, position_classes: list, sasa_fraction, fixed_idx: set,
               over_rep_mask=None, surface_mask=None,
               controller_ceiling: float = CONTROLLER_CEILING):
    """Run the full CF-3 coordinator for one cycle.

    Returns ``(global_dict, delta)`` — the budget-bounded GLOBAL per-AA bias ({AA:
    nats}) and the (L, 20) SURFACE delta — both clamped so the controller share is
    ≤ ``controller_ceiling`` odds at ``T_apply`` (the application-cycle temperature;
    math review §2.4). The whole-stack nesting (``TOTAL_CEILING``) is enforced
    separately at the sampler via :func:`nested_total_clip`.

    Drop-in for the ``global_per_aa_bias`` + surface-delta accumulation in
    :func:`adaptive_bias.compute_adaptive_bias` when ``--controller_coordinator`` is on.
    """
    budget_nats = nats_for_odds(controller_ceiling, T_apply)
    global_dict = coordinate_global(outcomes, axes_by_name, budget_nats=budget_nats)
    delta = coordinate_surface(
        outcomes, axes_by_name, budget_nats=budget_nats, L=L,
        position_classes=position_classes, sasa_fraction=sasa_fraction,
        fixed_idx=fixed_idx, cfg=cfg, over_rep_mask=over_rep_mask,
        surface_mask=surface_mask)
    return global_dict, delta


__all__ = [
    "CONTROLLER_CEILING", "TOTAL_CEILING",
    "coordinate", "coordinate_global", "coordinate_surface", "nested_total_clip",
]
