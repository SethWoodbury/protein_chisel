"""Load + concatenate ``chiseled_design_metrics.tsv`` files across many runs.

Designed for jupyterhub use:

    >>> from protein_chisel.tools.load_chiseled_runs import load_runs, load_one
    >>> df = load_runs("/net/scratch/woodbuse/iterative_design_*/chiseled_design_metrics.tsv")
    >>> df.shape
    (N_runs * 50, ~150)

Or via CLI:

    $ python scripts/load_chiseled_runs.py \\
        '/net/scratch/woodbuse/iterative_design_*/chiseled_design_metrics.tsv' \\
        --out /tmp/all_runs.tsv

Each row in the returned DataFrame includes:
    - the original 145+ design metrics
    - ``pdb_path``    absolute path to the design PDB
    - ``seed_pdb``    the seed used for that run
    - ``seed_basename`` just the seed PDB stem (good for groupby)
    - ``run_dir``     the run directory containing the TSV (provenance)
    - ``_meta_*`` columns extracted from the embedded RUN_META JSON
      (manifest fields like ``ptm_spec``, ``tunnel_metrics_enabled``,
      and a few useful summary stats from manifest/cycle_metrics)

The first line of every minimal-layout TSV is a comment starting with
``# RUN_META: <single-line-json>``; pandas skips it with ``comment='#'``,
and the JSON is recovered separately so per-run config is queryable.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


_RUN_META_RE = re.compile(r"^#\s*RUN_META:\s*(\{.*\})\s*$")


def extract_run_meta(tsv_path: str | Path) -> dict | None:
    """Return the RUN_META JSON dict embedded in the first line of a
    minimal-layout TSV, or None if absent / malformed.
    """
    p = Path(tsv_path)
    try:
        with open(p) as fh:
            first = fh.readline()
    except OSError:
        return None
    m = _RUN_META_RE.match(first)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _flatten_meta(meta: dict, prefix: str = "_meta") -> dict[str, object]:
    """Pull a few generally-useful fields from RUN_META into top-level
    columns. Keep the shape stable across runs so concat works.
    """
    out: dict[str, object] = {}
    if not meta:
        return out
    m = meta.get("manifest") or {}
    for k in (
        "seed_pdb", "ligand_params", "ptm_spec",
        "tunnel_metrics_enabled", "tunnel_hard_gate",
        "throat_feedback", "throat_feedback_decay",
        "n_cycles_run", "wt_length", "started_at",
        "final_topk_count", "final_unique_sequences",
    ):
        if k in m:
            out[f"{prefix}_{k}"] = m[k]
    lg = m.get("ligand_geometry") or {}
    for k in ("ligand_resname", "n_heavy_atoms", "n_metal_atoms",
               "min_projected_radius"):
        if k in lg:
            out[f"{prefix}_lig_{k}"] = lg[k]
    return out


def load_one(tsv_path: str | Path, *, with_meta: bool = True) -> pd.DataFrame:
    """Load one chiseled_design_metrics.tsv into a DataFrame.

    The first comment line (RUN_META) is skipped by pandas; if
    ``with_meta`` is True (default), useful manifest fields are added
    as ``_meta_*`` columns to every row.
    """
    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    if with_meta:
        meta = extract_run_meta(tsv_path)
        flat = _flatten_meta(meta) if meta else {}
        for k, v in flat.items():
            df[k] = v
        if meta is not None:
            # Stash full meta as JSON string in case caller wants to
            # inspect arbitrary fields. Fairly small (~5 KB / run).
            df["_meta_json"] = json.dumps(meta, default=str,
                                            separators=(",", ":"))
    return df


def load_runs(
    pattern_or_paths: str | Iterable[str | Path],
    *,
    with_meta: bool = True,
    add_run_id: bool = True,
) -> pd.DataFrame:
    """Concatenate many chiseled_design_metrics.tsv files into one DataFrame.

    Args:
        pattern_or_paths: a glob string (e.g.
            ``"/net/scratch/woodbuse/iterative_design_*/chiseled_design_metrics.tsv"``)
            or an iterable of explicit paths.
        with_meta: pull manifest fields into ``_meta_*`` columns.
        add_run_id: add a ``run_id`` short-name column derived from the
            run_dir (useful as a groupby key in plots).

    Returns:
        A single concatenated DataFrame with the union of columns
        (missing values are NaN). Preserves row order: each run's
        rows stay together, in the order returned by the glob.
    """
    if isinstance(pattern_or_paths, (str, Path)):
        paths = sorted(_glob.glob(str(pattern_or_paths)))
    else:
        paths = [str(p) for p in pattern_or_paths]
    if not paths:
        raise FileNotFoundError(f"no TSV files matched: {pattern_or_paths!r}")

    dfs: list[pd.DataFrame] = []
    for tsv in paths:
        try:
            df = load_one(tsv, with_meta=with_meta)
        except Exception as exc:
            print(f"WARN: failed to load {tsv}: {exc}", file=sys.stderr)
            continue
        if add_run_id:
            df["run_id"] = Path(tsv).parent.name
        dfs.append(df)
    if not dfs:
        raise RuntimeError("no TSVs loaded successfully")
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    return combined


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pattern", help="glob pattern for chiseled_design_metrics.tsv files")
    p.add_argument("--out", type=Path, default=None,
                   help="optional TSV path to write the concatenated DataFrame")
    p.add_argument("--no_meta", action="store_true",
                   help="don't pull RUN_META fields into _meta_* columns")
    args = p.parse_args()

    df = load_runs(args.pattern, with_meta=not args.no_meta)
    print(f"Loaded {len(df)} rows × {len(df.columns)} cols from "
          f"{df['run_id'].nunique() if 'run_id' in df.columns else '?'} runs",
          file=sys.stderr)
    if args.out:
        df.to_csv(args.out, sep="\t", index=False)
        print(f"wrote -> {args.out}", file=sys.stderr)
    else:
        # Print a small summary to stdout
        cols = [c for c in df.columns if not c.startswith("_meta_")][:6]
        print(df[cols].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
