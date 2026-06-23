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

Pure-Python (no PyRosetta/containers). Collision- and crash-safe: stages the
renamed set in a fresh temp dir, validates, then swaps each file in with an atomic
``os.replace`` (so a newly-assigned ``chisel_28`` can never clobber a pre-existing
source ``chisel_28``, and a crash mid-swap never loses a design — every target
always holds valid content). Idempotent. Only ever touches files inside the
published designs directory.
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
_TRUE = {"true", "1", "yes"}


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


def finalize_design_names(
    final_root: str | Path,
    *,
    keep_intermediate: bool = False,
    tsv_name: str = _TSV_NAME,
    design_token: str = "chisel",
) -> dict:
    """Rename + DESIGN_PATH-collapse the published designs under ``final_root``.

    Returns a summary dict. No-op (exit-friendly) when the TSV is absent/empty or
    there are no design rows.

    ``design_token`` is the name component used in the shipped filename
    ``<stem>_<design_token>_<NNN>.pdb`` (default ``"chisel"`` → byte-identical to
    the legacy naming). It must be a non-empty alphanumeric string (no ``_``/``.``)
    so the trailing-index strip stays unambiguous. The strip also recognises the
    legacy ``chisel`` token, so a run minting ``_chisel_<idx>`` ids is renamed to a
    custom token, and a re-run with the same custom token is idempotent.
    """
    if not re.fullmatch(r"[A-Za-z0-9]+", design_token or ""):
        raise ValueError(
            f"design_token must be non-empty alphanumeric (no '_' or '.'); "
            f"got {design_token!r}")
    # Strip a trailing ``_<design_token>_<idx>`` OR the legacy ``_chisel_<idx>``.
    # Greedy ``.+`` keeps any earlier same-token occurrence that is part of the
    # input stem (e.g. an input named ``..._chisel_62_..._chisel_5`` strips only
    # the final ``_chisel_5``). When design_token == "chisel" this is identical to
    # the legacy chisel-only strip (``^(.+)_chisel_\d+.*$``).
    strip_re = re.compile(
        rf"^(?P<stem>.+)_(?:{re.escape(design_token)}|chisel)_\d+.*$")
    final_root = Path(final_root)
    tsv = final_root / tsv_name
    if not tsv.is_file() or tsv.stat().st_size == 0:
        LOGGER.warning("finalize: no usable %s under %s; nothing to do",
                       tsv_name, final_root)
        return {"status": "no_tsv", "renamed": 0}

    meta, df = _read_metrics_tsv(tsv)
    if "id" not in df.columns:
        LOGGER.warning("finalize: TSV has no 'id' column; nothing to do")
        return {"status": "no_id_col", "renamed": 0}

    has_is_input = "is_input" in df.columns
    seed_basenames = {
        Path(str(r["pdb_path"])).name
        for _, r in df.iterrows()
        if has_is_input and str(r.get("is_input", "")).strip().lower() in _TRUE
        and str(r.get("pdb_path", ""))
    }

    def _is_input(row) -> bool:
        # Authoritative when the column exists; heuristic fallback only if absent
        # (so a design with an unusual id is NEVER misclassified as the input).
        if has_is_input:
            return str(row.get("is_input", "")).strip().lower() in _TRUE
        rid = str(row.get("id", ""))
        if "_chisel_" not in rid:
            return True
        pp = str(row.get("pdb_path", ""))
        return bool(pp) and Path(pp).name in seed_basenames

    def _resolve_old(row) -> Path | None:
        pp = str(row.get("pdb_path", "")).strip()
        if pp and Path(pp).is_file():
            return Path(pp)
        rid = str(row.get("id", "")).strip()
        for cand in (final_root / f"{rid}.pdb", final_root / "designs" / f"{rid}.pdb"):
            if cand.is_file():
                return cand
        return None

    design_idx = [i for i, r in df.iterrows() if not _is_input(r)]
    n = len(design_idx)
    if n == 0:
        LOGGER.info("finalize: 0 design rows; nothing to finalize")
        return {"status": "no_designs", "renamed": 0}

    first = _resolve_old(df.loc[design_idx[0]])
    if first is None:
        raise FileNotFoundError(
            f"finalize: design PDB for id={df.loc[design_idx[0]].get('id')!r} not found")
    designs_dir = first.parent                       # logical (keeps /net vs /mnt)
    designs_real = os.path.realpath(str(designs_dir))
    # Pad to the digit width of the LARGEST 0-based index (n-1), not the count.
    # Indices run 0..n-1, so the width grows only when an index itself crosses a
    # power of ten: 10 designs -> chisel_0..9 (1 digit, never chisel_10),
    # 11 -> chisel_00..10 (2), 100 -> chisel_00..99 (2), 101 -> chisel_000..100 (3).
    width = len(str(n - 1))                           # n >= 1 here (early-return at 0)

    # ---- Preflight: build + validate the full plan BEFORE touching any file ----
    plan: list[tuple[int, Path, str, str]] = []      # (df_index, old, new_name, new_id)
    for rank, di in enumerate(design_idx):
        row = df.loc[di]
        rid = str(row.get("id", "")).strip()
        if not rid:
            raise ValueError(f"finalize: design row {di} has empty id")
        old_path = _resolve_old(row)
        if old_path is None:
            raise FileNotFoundError(f"finalize: design PDB for id={rid!r} not found")
        # Safety: only ever operate on files inside the designs dir.
        if os.path.realpath(str(old_path.parent)) != designs_real:
            raise ValueError(
                f"finalize: id={rid!r} pdb {old_path} is outside the designs dir "
                f"{designs_dir}; refusing to rename")
        m = strip_re.match(old_path.stem)
        stem = m.group("stem") if m else old_path.stem
        new_id = f"{stem}_{design_token}_{rank:0{width}d}"
        plan.append((di, old_path, f"{new_id}.pdb", new_id))

    new_names = [nm for (_, _, nm, _) in plan]
    if len(set(new_names)) != n:
        raise RuntimeError(f"finalize: non-unique target names {new_names}")
    target_paths = {designs_dir / nm for nm in new_names}

    # ---- Phase 1: stage renamed + DESIGN_PATH-rewritten copies in a fresh tmp ----
    from protein_chisel.tools.remarks import finalize_design_path
    tmp = designs_dir / f".finalize_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=False)
    staged_ok = False
    swapped = False
    try:
        for di, old_path, new_name, new_id in plan:
            staged = tmp / new_name
            shutil.copy2(old_path, staged)
            final_abs = os.path.abspath(str(designs_dir / new_name))  # logical abs
            finalize_design_path(staged, final_path=final_abs,
                                 keep_intermediate=keep_intermediate)
        staged_files = sorted(tmp.glob("*.pdb"))
        if len(staged_files) != n:
            raise RuntimeError(f"finalize: staged {len(staged_files)} != {n} expected")
        staged_ok = True

        # ---- Phase 2: per-file ATOMIC swap. os.replace is atomic on the same FS
        # (tmp is a subdir of designs_dir) and overwrites any collided original at
        # the target name. Originals are NOT pre-deleted, and the staged copies are
        # retained (see finally) until the swap fully completes, so no design can be
        # lost even on a mid-swap crash in the collision case. ----
        for di, old_path, new_name, new_id in plan:
            os.replace(str(tmp / new_name), str(designs_dir / new_name))
        swapped = True
        # Remove leftover originals whose name isn't itself a target (source had a
        # different chisel index). Never deletes a target or any non-plan file.
        for op in {op for (_, op, _, _) in plan}:
            if op not in target_paths and op.exists():
                op.unlink()
    finally:
        # Remove tmp on full success, or on a PRE-swap failure (originals intact).
        # On a mid-swap failure keep tmp so the staged copies survive for recovery
        # (a partial swap may have overwritten a collided original whose only other
        # copy is in tmp). Such a leftover .finalize_tmp_* dir flags manual recovery.
        if tmp.exists() and (swapped or not staged_ok):
            shutil.rmtree(tmp, ignore_errors=True)
        elif tmp.exists():
            LOGGER.error("finalize: swap interrupted; staged copies retained at %s "
                         "for recovery (designs dir may be partially renamed)", tmp)

    # ---- Phase 3: update TSV id + pdb_path for design rows (others untouched);
    # preserve the RUN_META header + column order; atomic replace. ----
    for (di, _old, new_name, _new_id) in plan:
        df.at[di, "id"] = _new_id
        if "pdb_path" in df.columns:
            df.at[di, "pdb_path"] = os.path.abspath(str(designs_dir / new_name))
    out = (meta or "") + df.to_csv(sep="\t", index=False)
    tmp_tsv = tsv.with_suffix(".tsv.tmp")
    tmp_tsv.write_text(out)
    os.replace(tmp_tsv, tsv)

    LOGGER.info("finalize: renamed %d designs -> <stem>_chisel_%0*d..%0*d in %s "
                "(DESIGN_PATH collapsed=%s)", n, width, 0, width, n - 1,
                designs_dir, not keep_intermediate)
    return {"status": "ok", "renamed": n, "width": width,
            "designs_dir": str(designs_dir)}
