"""Audit-only diagnostic tool for the iterative_design_v2 fpocket panel.

Runs fpocket on every PDB in a top-K folder (or any PDB folder) using
the same active-site-aware logic as ``stage_fpocket_rank`` in
``iterative_design_v2.py``, and prints / writes a wide-format table of
the new pocket diagnostics:

    fpocket__druggability
    fpocket__volume
    fpocket__mean_alpha_sphere_radius
    fpocket__bottleneck_radius          ← NEW (rim-quartile narrow point)
    fpocket__min_alpha_sphere_radius    ← NEW (absolute narrowest sphere)
    fpocket__alpha_sphere_radius_p10    ← NEW (robust narrow stat)
    fpocket__hydrophobicity_score
    fpocket__charge_score
    fpocket__polarity_score
    fpocket__polar_atoms_pct
    fpocket__apolar_atoms_pct
    fpocket__apolar_alpha_sphere_proportion
    fpocket__polar_alpha_sphere_proportion
    fpocket__total_sasa / polar_sasa / apolar_sasa
    fpocket__alpha_sphere_density
    fpocket__cent_of_mass_alpha_sphere_max_dist
    fpocket__mean_alpha_sphere_solvent_acc
    fpocket__mean_alpha_sphere_dist_to_catalytic
    fpocket__n_alpha_spheres_near_catalytic
    fpocket__n_rim_spheres

Use this to identify designs whose active-site pocket is blocked by a
bulky residue: low ``bottleneck_radius`` despite high druggability is
the signature of "the substrate can't reach the catalytic triad even
though fpocket sees a deep pocket".

Usage:
    apptainer exec ... python scripts/audit_pocket_metrics.py \\
        --pdb_dir /net/scratch/.../final_topk/topk_pdbs \\
        --catalytic_resnos 60,64,128,131,132,157 \\
        --out /tmp/pocket_audit.tsv

If ``--ranked_tsv`` is given (e.g. the run's final ``topk.tsv``), the
audit table is left-merged with the existing fitness/druggability
columns so you can sort by both at once.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Reuse the pocket-extraction primitives from the production driver so
# this audit tool stays bit-exact with what the pipeline computes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from iterative_design_v2 import _run_fpocket  # noqa: E402


_DIAG_COLS = [
    "fpocket__druggability",
    "fpocket__volume",
    "fpocket__bottleneck_radius",
    "fpocket__min_alpha_sphere_radius",
    "fpocket__alpha_sphere_radius_p10",
    "fpocket__mean_alpha_sphere_radius",
    "fpocket__alpha_sphere_density",
    "fpocket__hydrophobicity_score",
    "fpocket__charge_score",
    "fpocket__polarity_score",
    "fpocket__polar_atoms_pct",
    "fpocket__apolar_atoms_pct",
    "fpocket__apolar_alpha_sphere_proportion",
    "fpocket__polar_alpha_sphere_proportion",
    "fpocket__total_sasa",
    "fpocket__polar_sasa",
    "fpocket__apolar_sasa",
    "fpocket__mean_alpha_sphere_solvent_acc",
    "fpocket__cent_of_mass_alpha_sphere_max_dist",
    "fpocket__mean_alpha_sphere_dist_to_catalytic",
    "fpocket__n_alpha_spheres_near_catalytic",
    "fpocket__n_rim_spheres",
    "fpocket__n_alpha_spheres",
    "fpocket__mean_local_hydrophobic_density",
]


def _info_to_row(info: dict | None) -> dict:
    """Map ``_run_fpocket`` output dict to wide-format TSV columns,
    matching the production schema in ``stage_fpocket_rank``."""
    if info is None:
        return {c: float("nan") for c in _DIAG_COLS}
    polar_pct = info.get("proportion_of_polar_atoms", float("nan"))
    apolar_pct = (
        100.0 - polar_pct
        if isinstance(polar_pct, (int, float)) and polar_pct == polar_pct
        else float("nan")
    )
    return {
        "fpocket__druggability": info.get("druggability_score", 0.0),
        "fpocket__volume": info.get("volume", 0.0),
        "fpocket__bottleneck_radius": info.get("bottleneck_radius", float("nan")),
        "fpocket__min_alpha_sphere_radius":
            info.get("min_alpha_sphere_radius", float("nan")),
        "fpocket__alpha_sphere_radius_p10":
            info.get("alpha_sphere_radius_p10", float("nan")),
        "fpocket__mean_alpha_sphere_radius":
            info.get("mean_alpha_sphere_radius", 0.0),
        "fpocket__alpha_sphere_density": info.get("alpha_sphere_density", 0.0),
        "fpocket__hydrophobicity_score":
            info.get("hydrophobicity_score", float("nan")),
        "fpocket__charge_score": info.get("charge_score", float("nan")),
        "fpocket__polarity_score": info.get("polarity_score", float("nan")),
        "fpocket__polar_atoms_pct": polar_pct,
        "fpocket__apolar_atoms_pct": apolar_pct,
        "fpocket__apolar_alpha_sphere_proportion":
            info.get("apolar_alpha_sphere_proportion", float("nan")),
        "fpocket__polar_alpha_sphere_proportion":
            info.get("polar_alpha_sphere_proportion", float("nan")),
        "fpocket__total_sasa": info.get("total_sasa", float("nan")),
        "fpocket__polar_sasa": info.get("polar_sasa", float("nan")),
        "fpocket__apolar_sasa": info.get("apolar_sasa", float("nan")),
        "fpocket__mean_alpha_sphere_solvent_acc":
            info.get("mean_alp_sph_solvent_access", float("nan")),
        "fpocket__cent_of_mass_alpha_sphere_max_dist":
            info.get("cent_of_mass_alpha_sphere_max_dist", float("nan")),
        "fpocket__mean_alpha_sphere_dist_to_catalytic":
            info.get("mean_alpha_sphere_dist_to_catalytic", float("nan")),
        "fpocket__n_alpha_spheres_near_catalytic":
            info.get("n_alpha_spheres_near_catalytic", 0),
        "fpocket__n_rim_spheres": info.get("n_rim_spheres", 0),
        "fpocket__n_alpha_spheres": info.get("number_of_alpha_spheres", 0),
        "fpocket__mean_local_hydrophobic_density":
            info.get("mean_local_hydrophobic_density", float("nan")),
    }


def main() -> None:
    _setup_logging()
    log = logging.getLogger("audit_pocket_metrics")

    p = argparse.ArgumentParser()
    p.add_argument("--pdb_dir", type=Path, required=True,
                   help="Folder containing design .pdb files")
    p.add_argument("--catalytic_resnos", type=str,
                   default="60,64,128,131,132,157",
                   help="Comma-separated catalytic residue numbers "
                        "(default: PTE: 60,64,128,131,132,157)")
    p.add_argument("--chain", default="A")
    p.add_argument("--workspace", type=Path, default=None,
                   help="Where to keep fpocket outputs (default: tmpdir, "
                        "wiped at end). Pass a real dir to retain.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output TSV path")
    p.add_argument("--ranked_tsv", type=Path, default=None,
                   help="Optional: existing ranked.tsv / topk.tsv to "
                        "left-merge audit columns into")
    p.add_argument("--limit", type=int, default=0,
                   help="Audit only the first N PDBs (0 = all)")
    p.add_argument("--print_top", type=int, default=10,
                   help="Print this many rows of the audit table")
    args = p.parse_args()

    catres = tuple(int(x) for x in args.catalytic_resnos.split(",") if x.strip())
    pdbs = sorted(args.pdb_dir.glob("*.pdb"))
    if args.limit > 0:
        pdbs = pdbs[: args.limit]
    log.info("auditing %d PDB(s) from %s (catres=%s)",
             len(pdbs), args.pdb_dir, catres)

    if args.workspace is not None:
        args.workspace.mkdir(parents=True, exist_ok=True)

    rows = []
    t_total = 0.0
    for pdb in pdbs:
        cid = pdb.stem
        if args.workspace:
            wd = args.workspace / cid
        else:
            import tempfile
            wd = Path(tempfile.mkdtemp(prefix=f"audit_{cid[:20]}_"))
        t0 = time.time()
        try:
            info = _run_fpocket(
                pdb, wd, catalytic_resnos=catres, chain=args.chain,
            )
        except Exception as exc:
            log.warning("fpocket failed on %s: %s", pdb.name, exc)
            info = None
        dt = time.time() - t0
        t_total += dt
        row = {"id": cid, "wall_s": round(dt, 3)}
        row.update(_info_to_row(info))
        rows.append(row)
        def _f(k):
            # Fall back to nan only if missing/None — don't coerce 0.0 to nan
            v = row.get(k)
            return float("nan") if v is None else v
        log.info(
            "%-65s %5.2fs drug=%5.3f vol=%7.1f bottleneck=%4.2f hydro=%6.1f charge=%4.1f polar%%=%4.1f",
            cid[:65], dt,
            _f("fpocket__druggability"),
            _f("fpocket__volume"),
            _f("fpocket__bottleneck_radius"),
            _f("fpocket__hydrophobicity_score"),
            _f("fpocket__charge_score"),
            _f("fpocket__polar_atoms_pct"),
        )
        if args.workspace is None:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    df = pd.DataFrame(rows)
    log.info("done: %d designs, total wall %.1fs, mean %.2fs/design",
             len(df), t_total, t_total / max(1, len(df)))

    if args.ranked_tsv is not None and args.ranked_tsv.is_file():
        base = pd.read_csv(args.ranked_tsv, sep="\t")
        # Drop existing fpocket__* columns that would collide
        keep = [c for c in base.columns if not c.startswith("fpocket__")]
        merged = base[keep].merge(df, on="id", how="left")
        merged.to_csv(args.out, sep="\t", index=False)
        log.info("wrote merged audit table -> %s (n=%d, %d cols)",
                 args.out, len(merged), merged.shape[1])
    else:
        df.to_csv(args.out, sep="\t", index=False)
        log.info("wrote audit table -> %s", args.out)

    # Pretty-print the most-actionable subset
    show_cols = [
        "id",
        "fpocket__druggability",
        "fpocket__volume",
        "fpocket__bottleneck_radius",
        "fpocket__min_alpha_sphere_radius",
        "fpocket__alpha_sphere_radius_p10",
        "fpocket__hydrophobicity_score",
        "fpocket__charge_score",
        "fpocket__polar_atoms_pct",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    sorted_df = df.sort_values("fpocket__druggability", ascending=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print()
    print("=== Top by druggability ===")
    print(sorted_df[show_cols].head(args.print_top).to_string(index=False))
    print()
    print("=== Bottom by druggability ===")
    print(sorted_df[show_cols].tail(args.print_top).to_string(index=False))
    print()
    print("=== Sorted by bottleneck_radius (ascending — narrowest first) ===")
    if "fpocket__bottleneck_radius" in df.columns:
        nb = df.sort_values("fpocket__bottleneck_radius", ascending=True)
        print(nb[show_cols].head(args.print_top).to_string(index=False))


if __name__ == "__main__":
    main()
