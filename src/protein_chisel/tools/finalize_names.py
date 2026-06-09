"""Final post-publish step: rank-order rename + DESIGN_PATH collapse.

Runs LAST, on the already-published, already-selected, already-ranked top-K design
set (the cross-cycle winners), in the caller's final output directory. It:

  1. Renames each shipped design to ``<stem>_chisel_<NNN>.pdb`` where NNN is a
     zero-padded, **rank-ordered** index (rank 0 = best, from the metrics-TSV row
     order), keeping the input stem.
  2. Collapses the per-cycle intermediate DESIGN_PATH stamps (iterative_design +
     protonate_topk, which point at deleted node-local scratch) into a single
     ``REMARK DESIGN_PATH chisel_iterative_design output <abs final path>`` line
     (unless ``keep_intermediate``); upstream provenance is preserved.
  3. Keeps the metrics TSV (id + pdb_path) consistent with the renamed files.

Pure-Python (no PyRosetta/containers). Collision-safe: builds the renamed set in a
fresh temp dir, validates, then atomically swaps — a newly-assigned ``chisel_28``
can never clobber a pre-existing source ``chisel_28``. Idempotent + safe on
partial failure (originals untouched until the validated swap).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
from pathlib import Path

LOGGER = logging.getLogger("protein_chisel.tools.finalize_names")

_TSV_NAME = "chiseled_design_metrics.tsv"
_CHISEL_RE = re.compile(r"^(?P<stem>.+)_chisel_\d+.*$")  # strip trailing _chisel_<idx>[suffix]


def _read_metrics_tsv(tsv: Path):
    """Return (run_meta_first_line_or_None, DataFrame). Preserves the RUN_META
    comment header verbatim; reads everything else as strings (no NaN coercion)."""
    import pandas as pd
    from io import StringIO

    raw = tsv.read_text().splitlines(keepends=True)
    meta = None
    body = raw
    if raw and raw[0].startswith("# RUN_META:"):
        meta, body = raw[0], raw[1:]
    df = pd.read_csv(StringIO("".join(body)), sep="\t", dtype=str,
                     keep_default_na=False)
    return meta, df


def _is_input_row(row, seed_basenames: set[str]) -> bool:
    """Triple-guarded: a row is the input reference (NOT a design) if it's flagged
    is_input, OR its id has no _chisel_ marker, OR its pdb basename is the seed."""
    val = str(row.get("is_input", "")).strip().lower()
    if val in {"true", "1", "yes"}:
        return True
    rid = str(row.get("id", ""))
    if "_chisel_" not in rid:
        return True
    pp = str(row.get("pdb_path", ""))
    if pp and Path(pp).name in seed_basenames:
        return True
    return False


def finalize_design_names(
    final_root: str | Path,
    *,
    keep_intermediate: bool = False,
    tsv_name: str = _TSV_NAME,
) -> dict:
    """Rename + DESIGN_PATH-collapse the published designs under ``final_root``.

    Returns a summary dict. No-op (exit-friendly) when the TSV is absent or there
    are no design rows.
    """
    final_root = Path(final_root)
    tsv = final_root / tsv_name
    if not tsv.is_file():
        LOGGER.warning("finalize: no %s under %s; nothing to do", tsv_name, final_root)
        return {"status": "no_tsv", "renamed": 0}

    meta, df = _read_metrics_tsv(tsv)
    if "id" not in df.columns:
        LOGGER.warning("finalize: TSV has no 'id' column; nothing to do")
        return {"status": "no_id_col", "renamed": 0}

    seed_basenames = {
        Path(str(r["pdb_path"])).name
        for _, r in df.iterrows()
        if str(r.get("is_input", "")).strip().lower() in {"true", "1", "yes"}
        and str(r.get("pdb_path", ""))
    }

    # Design rows in rank (TSV row) order; input-reference row(s) excluded.
    design_idx = [i for i, r in df.iterrows()
                  if not _is_input_row(r, seed_basenames)]
    n = len(design_idx)
    if n == 0:
        LOGGER.info("finalize: 0 design rows; nothing to finalize")
        return {"status": "no_designs", "renamed": 0}

    # Locate the designs dir from the first design row's pdb_path (robust to
    # flat/minimal vs designs/ layouts); fall back to final_root[/designs].
    def _resolve_old(row) -> Path:
        pp = str(row.get("pdb_path", ""))
        if pp and Path(pp).is_file():
            return Path(pp)
        for cand in (final_root / f"{row['id']}.pdb",
                     final_root / "designs" / f"{row['id']}.pdb"):
            if cand.is_file():
                return cand
        return Path(pp) if pp else final_root / f"{row['id']}.pdb"

    first_old = _resolve_old(df.loc[design_idx[0]])
    designs_dir = first_old.parent
    width = max(2, len(str(n - 1)))

    # Build the rename map (old path, new name, new id) in rank order.
    plan: list[tuple[int, Path, str, str]] = []  # (df_index, old_path, new_name, new_id)
    for rank, di in enumerate(design_idx):
        row = df.loc[di]
        old_path = _resolve_old(row)
        if not old_path.is_file():
            raise FileNotFoundError(
                f"finalize: design PDB for id={row['id']!r} not found ({old_path})")
        m = _CHISEL_RE.match(old_path.stem)
        stem = m.group("stem") if m else old_path.stem
        new_id = f"{stem}_chisel_{rank:0{width}d}"
        plan.append((di, old_path, f"{new_id}.pdb", new_id))

    new_names = [nm for (_, _, nm, _) in plan]
    if len(set(new_names)) != n:
        raise RuntimeError(f"finalize: non-unique target names {new_names}")

    # Phase 1: build renamed + rewritten files in a FRESH temp dir (originals
    # untouched -> any failure here is safe).
    from protein_chisel.tools.remarks import finalize_design_path
    tmp = designs_dir / f".finalize_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        for di, old_path, new_name, new_id in plan:
            staged = tmp / new_name
            shutil.copy2(old_path, staged)
            final_abs = str((designs_dir / new_name).resolve())
            finalize_design_path(staged, final_path=final_abs,
                                 keep_intermediate=keep_intermediate)
        staged_files = sorted(tmp.glob("*.pdb"))
        if len(staged_files) != n:
            raise RuntimeError(
                f"finalize: staged {len(staged_files)} != {n} expected")

        # Phase 2: atomic-ish swap — remove old design PDBs, move staged in.
        old_paths = {old_path for (_, old_path, _, _) in plan}
        for op in old_paths:
            op.unlink()
        for di, old_path, new_name, new_id in plan:
            shutil.move(str(tmp / new_name), str(designs_dir / new_name))
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    # Phase 3: update TSV id + pdb_path for design rows (others untouched),
    # re-emit preserving the RUN_META header + column order; atomic replace.
    for (di, _old, new_name, new_id) in plan:
        df.at[di, "id"] = new_id
        if "pdb_path" in df.columns:
            df.at[di, "pdb_path"] = str((designs_dir / new_name).resolve())
    out = (meta or "") + df.to_csv(sep="\t", index=False)
    tmp_tsv = tsv.with_suffix(".tsv.tmp")
    tmp_tsv.write_text(out)
    os.replace(tmp_tsv, tsv)

    LOGGER.info("finalize: renamed %d designs -> <stem>_chisel_%0*d..%0*d in %s "
                "(DESIGN_PATH collapsed=%s)", n, width, 0, width, n - 1,
                designs_dir, not keep_intermediate)
    return {"status": "ok", "renamed": n, "width": width,
            "designs_dir": str(designs_dir)}
