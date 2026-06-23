"""Pure formatter for the verbose controller trace (CF-5).

The adaptive-bias controller emits, each cycle, one :class:`AxisOutcome` per axis
(what it measured, whether the gate opened, and the drive ``u`` it will apply). For
step-by-step *chronological* validation the user wants a flat, long-format trace —
one row per axis per cycle — appended to ``run_dir/controller_trace.tsv`` plus a
human-readable per-cycle "CONTROLLER REPORT". This module owns the *formatting*; the
driver owns the (opt-in) I/O and logging.

Everything here is PURE (stdlib only, no numpy/pandas, no I/O), so the trace contents
are unit-testable without models, PyRosetta, or the cluster — and so importing it has
no heavy cost. It is only ever called when ``--controller_verbose`` is set, so a
default run is byte-identical: nothing in this module runs and nothing new is imported
at module/parse time (the driver keeps the import inside the opt-in branch).

The single legibility number is ``effective_odds = odds_for_nats(u, T) = exp(u/T)`` —
the actual odds multiplier the drive ``u`` nats produce at the cycle's sampling
temperature ``T``. A reader judges "nudge vs strong vs effective lock" from that, not
from the raw nats (which silently amplify as ``T`` anneals).
"""
from __future__ import annotations

from typing import Optional

from protein_chisel.sampling.bias_scale import odds_for_nats

# The exact, stable column set for controller_trace.tsv (long format). Pinned by a
# test so the offline-validation schema can't drift silently.
TRACE_COLUMNS = (
    "cycle",
    "axis",
    "scope",
    "measured_mean",
    "target",
    "band_lo",
    "band_hi",
    "signed_error",
    "gate_open",
    "reason",
    "drive_u",
    "effective_odds",
    "n",
)


def controller_trace_rows(outcomes, axes_by_name, *, cycle_idx: int,
                          temperature: float) -> list[dict]:
    """One trace row per :class:`AxisOutcome`, in outcome order.

    Parameters
    ----------
    outcomes:
        Iterable of ``AxisOutcome`` (from ``AdaptiveBiasResult.outcomes``).
    axes_by_name:
        Mapping ``axis.name -> ControlAxis`` for the cycle, used to lift the
        declarative ``target`` / ``band_lo`` / ``band_hi`` setpoints into the row.
        An outcome whose name is absent gets ``None`` for those fields (advisory:
        the trace must never crash the run), so the row count always equals the
        number of outcomes.
    cycle_idx:
        The current cycle index (``cyc.cycle_idx``).
    temperature:
        The cycle's sampling temperature ``T`` (``cyc.sampling_temperature``); the
        drive's effective odds is ``odds_for_nats(u, T) = exp(u / T)``. Must be > 0
        (``odds_for_nats`` raises otherwise — same contract as the sampler).

    Returns
    -------
    list[dict]
        Each dict has exactly the keys in :data:`TRACE_COLUMNS`.
    """
    rows: list[dict] = []
    for o in outcomes:
        axis = axes_by_name.get(o.name)
        target: Optional[float] = float(axis.target) if axis is not None else None
        band_lo: Optional[float] = float(axis.band_lo) if axis is not None else None
        band_hi: Optional[float] = float(axis.band_hi) if axis is not None else None
        rows.append({
            "cycle": int(cycle_idx),
            "axis": o.name,
            "scope": o.scope,
            "measured_mean": float(o.mean),
            "target": target,
            "band_lo": band_lo,
            "band_hi": band_hi,
            "signed_error": float(o.signed_error),
            "gate_open": bool(o.gate_open),
            "reason": o.reason,
            "drive_u": float(o.u),
            # exp(u / T): the temperature-aware odds multiplier this drive produces.
            "effective_odds": odds_for_nats(float(o.u), temperature),
            "n": int(o.n),
        })
    return rows


def format_controller_report_lines(rows: list[dict]) -> list[str]:
    """Human-readable per-cycle CONTROLLER REPORT lines (one per axis row).

    Pure string formatting over :func:`controller_trace_rows` output, factored out so
    the exact log shape is testable. The driver emits these via ``LOGGER.info`` only
    when ``--controller_verbose`` is set.
    """
    lines: list[str] = []
    for r in rows:
        tgt = "n/a" if r["target"] is None else f"{r['target']:.3f}"
        lo = "n/a" if r["band_lo"] is None else f"{r['band_lo']:.3f}"
        hi = "n/a" if r["band_hi"] is None else f"{r['band_hi']:.3f}"
        gate = "OPEN" if r["gate_open"] else "hold"
        lines.append(
            f"  axis={r['axis']} ({r['scope']}) "
            f"measured={r['measured_mean']:.3f} vs target={tgt} "
            f"band=[{lo},{hi}] err={r['signed_error']:+.3f} "
            f"gate={gate}({r['reason']}) drive_u={r['drive_u']:+.4f} "
            f"effective_odds={r['effective_odds']:.3g}x n={r['n']}"
        )
    return lines
