"""Test whether alternative side-chain packers can resolve the catalytic
clashes that fused_mpnn's built-in packer leaves on PTE_i1 designs.

Background
----------
fused_mpnn (LigandMPNN lab build at /net/software/lab/fused_mpnn/seth_temp)
ships a learned diffusion-style side-chain packer (sc_utils.Packer) that
samples chi torsions from a Von Mises mixture. With
``--repack_everything 0`` it preserves catalytic rotamers but packs the
designed residues around them. On PTE_i1 we observe ~90% of designs
clashing at residue 35 (Y or H) vs catalytic E131. This script tests
whether re-packing those problem designs with FASPR / PIPPack /
AttnPacker fixes the clashes, with the catalytic residues kept fixed.

Usage (host, FASPR-only, no GPU)::

    PYTHONPATH=src python scripts/test_packer_swap.py \\
        --packers faspr \\
        --out-dir /tmp/test_packer_swap

Usage (cluster, GPU, all packers)::

    sbatch scripts/run_test_packer_swap.sbatch

Output: a TSV with one row per (design, packer), columns:
    design_id, packer, n_clashes_before, n_clashes_after,
    severe_before, severe_after, runtime_s, error.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Defaults tuned for the PTE_i1 cycle-0 run that motivated this script.
DEFAULT_REJECTS_TSV = Path(
    "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260503-210056/"
    "cycle_00/03_struct_filter/rejects_struct.tsv"
)
DEFAULT_PDB_DIR = Path(
    "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260503-210056/"
    "cycle_00/01_sample/pdbs_restored"
)
CATALYTIC_RESNOS = [60, 64, 128, 131, 132, 157]


@dataclass
class TrialRow:
    design_id: str
    aa_at_35: str
    packer: str
    n_clashes_before: int
    severe_before: int
    detail_before: str
    n_clashes_after: int
    severe_after: int
    detail_after: str
    runtime_s: float
    error: str = ""


def pick_designs(rejects_tsv: Path, n_y: int = 3, n_h: int = 2) -> list[tuple[str, str]]:
    """Pick a few rejected designs with varied AA at PDB residue 35."""
    rows_y, rows_h = [], []
    with rejects_tsv.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("passed_struct_filter", "True") != "False":
                continue
            detail = row.get("clash__detail", "") or ""
            if "35-131" not in detail:
                continue
            seq = row.get("sequence", "")
            if len(seq) < 35:
                continue
            aa = seq[34]  # PDB resno 35 → 0-indexed 34
            if aa == "Y" and len(rows_y) < n_y:
                rows_y.append((row["id"], aa))
            elif aa == "H" and len(rows_h) < n_h:
                rows_h.append((row["id"], aa))
            if len(rows_y) >= n_y and len(rows_h) >= n_h:
                break
    return rows_y + rows_h


def score_clashes(pdb: Path) -> tuple[int, int, str]:
    from protein_chisel.structure.clash_check import detect_clashes

    res = detect_clashes(pdb, catalytic_resnos=CATALYTIC_RESNOS, chain="A")
    detail = "; ".join(f"{a}-{b}({d:.2f})" for a, b, d in res.clash_positions[:5])
    return res.n_clashes, int(res.has_severe_clash), detail


def run_faspr(src: Path, dst: Path) -> float:
    """Repack with FASPR, keeping catalytic residues fixed."""
    from protein_chisel.tools.sidechain_packing_and_scoring.faspr_pack import (
        _extract_sequence_from_pdb,
        faspr_pack,
    )

    seq = _extract_sequence_from_pdb(src)
    fixed_idx = [r - 1 for r in CATALYTIC_RESNOS if r - 1 < len(seq)]
    t0 = time.perf_counter()
    faspr_pack(src, out_pdb_path=dst, fixed_residues=fixed_idx)
    return time.perf_counter() - t0


def run_attnpacker(src: Path, dst: Path) -> float:
    """Repack with AttnPacker (esmc.sif, GPU).

    AttnPacker has no public 'fixed residues' API in its Inference class,
    so this re-packs *everything*. After AttnPacker we splice the
    catalytic residues' sidechains back from the source PDB to preserve
    their geometry.
    """
    from protein_chisel.tools.sidechain_packing_and_scoring.attnpacker_pack import (
        attnpacker_pack,
    )

    t0 = time.perf_counter()
    attnpacker_pack(src, out_pdb_path=dst)
    runtime = time.perf_counter() - t0
    _splice_catalytic_sidechains(src=src, dst=dst, catalytic_resnos=CATALYTIC_RESNOS)
    return runtime


def run_pippack(src: Path, dst: Path) -> float:
    """Repack with PIPPack (esmc.sif, GPU). Same fixed-splice trick as AttnPacker."""
    from protein_chisel.tools.sidechain_packing_and_scoring.pippack_score import (
        pippack_pack,
    )

    out_dir = dst.parent / f"{dst.stem}_pippack_workdir"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = pippack_pack(src, out_dir=out_dir)
    runtime = time.perf_counter() - t0
    # pippack_pack writes <stem>.pippack.pdb; copy it to dst
    import shutil

    shutil.copy2(res.out_pdb_path, dst)
    _splice_catalytic_sidechains(src=src, dst=dst, catalytic_resnos=CATALYTIC_RESNOS)
    return runtime


def _splice_catalytic_sidechains(
    src: Path, dst: Path, catalytic_resnos: list[int], chain: str = "A"
) -> None:
    """Replace catalytic-residue ATOM records in ``dst`` with those from ``src``.

    Used after a packer that doesn't honor a fixed-residue list -- we
    want to score the design against the *original* catalytic geometry.
    """
    src_lines = {}  # (chain, resno, atom) -> full ATOM line
    for line in src.read_text().splitlines(keepends=True):
        if not line.startswith("ATOM"):
            continue
        ch = line[21]
        try:
            rn = int(line[22:26])
        except ValueError:
            continue
        atom = line[12:16].strip()
        if ch == chain and rn in catalytic_resnos:
            src_lines[(ch, rn, atom)] = line

    new_lines: list[str] = []
    seen_replacements: set[tuple[str, int, str]] = set()
    for line in dst.read_text().splitlines(keepends=True):
        if not line.startswith("ATOM"):
            new_lines.append(line)
            continue
        ch = line[21]
        try:
            rn = int(line[22:26])
        except ValueError:
            new_lines.append(line)
            continue
        atom = line[12:16].strip()
        if ch == chain and rn in catalytic_resnos:
            key = (ch, rn, atom)
            if key in src_lines:
                new_lines.append(src_lines[key])
                seen_replacements.add(key)
            # else: drop (atom present in dst but not src — unlikely)
        else:
            new_lines.append(line)
    dst.write_text("".join(new_lines))


PACKERS = {
    "faspr": run_faspr,
    "attnpacker": run_attnpacker,
    "pippack": run_pippack,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rejects-tsv", type=Path, default=DEFAULT_REJECTS_TSV)
    parser.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    parser.add_argument(
        "--packers",
        type=str,
        default="faspr",
        help="Comma-separated subset of: faspr,attnpacker,pippack",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-y", type=int, default=3)
    parser.add_argument("--n-h", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    packer_names = [p.strip() for p in args.packers.split(",") if p.strip()]
    for name in packer_names:
        if name not in PACKERS:
            raise SystemExit(f"unknown packer {name!r} (have {sorted(PACKERS)})")

    designs = pick_designs(args.rejects_tsv, n_y=args.n_y, n_h=args.n_h)
    print(f"selected {len(designs)} designs to test:")
    for did, aa in designs:
        print(f"  {did}  ({aa}@35)")
    print()

    rows: list[TrialRow] = []
    for design_id, aa in designs:
        src_pdb = args.pdb_dir / f"{design_id}.pdb"
        if not src_pdb.is_file():
            print(f"  skip: {src_pdb} not found")
            continue
        n0, sev0, det0 = score_clashes(src_pdb)
        for packer in packer_names:
            dst_pdb = args.out_dir / f"{design_id}__{packer}.pdb"
            print(f"  {design_id} → {packer} ...", flush=True)
            err = ""
            t = 0.0
            n1 = sev1 = -1
            det1 = ""
            try:
                t = PACKERS[packer](src_pdb, dst_pdb)
                n1, sev1, det1 = score_clashes(dst_pdb)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            row = TrialRow(
                design_id=design_id,
                aa_at_35=aa,
                packer=packer,
                n_clashes_before=n0,
                severe_before=sev0,
                detail_before=det0,
                n_clashes_after=n1,
                severe_after=sev1,
                detail_after=det1,
                runtime_s=round(t, 3),
                error=err,
            )
            rows.append(row)
            print(
                f"    before n={n0}/sev={sev0}, after n={n1}/sev={sev1}, "
                f"t={t:.2f}s {('ERR='+err) if err else ''}"
            )

    out_tsv = args.out_dir / "results.tsv"
    with out_tsv.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "design_id",
                "aa_at_35",
                "packer",
                "n_clashes_before",
                "severe_before",
                "detail_before",
                "n_clashes_after",
                "severe_after",
                "detail_after",
                "runtime_s",
                "error",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.design_id,
                    r.aa_at_35,
                    r.packer,
                    r.n_clashes_before,
                    r.severe_before,
                    r.detail_before,
                    r.n_clashes_after,
                    r.severe_after,
                    r.detail_after,
                    r.runtime_s,
                    r.error,
                ]
            )
    print(f"\nwrote {out_tsv}")

    # Print a small markdown summary.
    print()
    print("| design | aa | packer | clash_before | severe_before | clash_after | severe_after | runtime_s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r.design_id[-6:]} | {r.aa_at_35} | {r.packer} | "
            f"{r.n_clashes_before} | {r.severe_before} | "
            f"{r.n_clashes_after} | {r.severe_after} | {r.runtime_s} |"
        )


if __name__ == "__main__":
    main()
