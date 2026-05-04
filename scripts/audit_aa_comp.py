"""Audit AA-composition z-scores on the latest top-50 designs.

For each canonical AA, compute distribution of per-design z-scores
against `swissprot_ec3_hydrolases_2026_01` and recommend a sane
SOFT_BIAS / HARD_FILTER threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protein_chisel.expression.aa_composition import aa_z_scores, aa_composition_pct

TOPK = Path(
    "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260503-212622/"
    "final_topk/topk.tsv"
)


def main():
    df = pd.read_csv(TOPK, sep="\t")
    seqs = df["sequence"].tolist()
    print(f"Loaded {len(seqs)} sequences from {TOPK}")
    aa_z_all = []
    aa_pct_all = []
    for s in seqs:
        z = aa_z_scores(s, reference="swissprot_ec3_hydrolases_2026_01")
        p = aa_composition_pct(s)
        aa_z_all.append(z)
        aa_pct_all.append(p)
    z_df = pd.DataFrame(aa_z_all)
    p_df = pd.DataFrame(aa_pct_all)
    print(f"\n=== Per-AA z-score distribution (n={len(seqs)}) ===")
    print(f"{'AA':>4} {'mean_z':>8} {'med_z':>8} {'min_z':>8} {'max_z':>8} "
          f"{'mean_pct':>9} {'p>2':>5} {'p>3':>5} {'p>4':>5} {'p>5':>5}")
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        zs = z_df[aa].values
        ps = p_df[aa].values
        print(f"{aa:>4} {zs.mean():8.2f} {np.median(zs):8.2f} "
              f"{zs.min():8.2f} {zs.max():8.2f} "
              f"{ps.mean()*100:9.2f} "
              f"{(np.abs(zs) > 2).mean():5.2f} "
              f"{(np.abs(zs) > 3).mean():5.2f} "
              f"{(np.abs(zs) > 4).mean():5.2f} "
              f"{(np.abs(zs) > 5).mean():5.2f}")

    # Per-design max abs z (top-1 outlier per design)
    max_abs = z_df.abs().max(axis=1)
    print(f"\nPer-design max |z| (across all 20 AAs):")
    print(f"  mean={max_abs.mean():.2f}  median={max_abs.median():.2f}  "
          f"min={max_abs.min():.2f}  max={max_abs.max():.2f}")
    # Universally OOD AAs
    print("\n=== Universally out-of-dist AAs (|z|>2 in 100% of designs) ===")
    universal = []
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        if (np.abs(z_df[aa]) > 2).all():
            universal.append((aa, z_df[aa].mean()))
            print(f"  {aa}: mean z = {z_df[aa].mean():+.2f} "
                  f"(min {z_df[aa].min():+.2f}, max {z_df[aa].max():+.2f})")

    # Threshold recommendation
    # Find the smallest z-threshold such that no design fails purely on
    # "expected outliers" (E above its expected -10 charge value).
    print("\n=== Threshold recommendation ===")
    for t in (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0):
        n_fail = (z_df.abs() > t).any(axis=1).sum()
        n_hits = (z_df.abs() > t).sum().sum()
        print(f"  |z|>{t}: {n_fail}/{len(seqs)} designs trip rule "
              f"(total {n_hits} (AA,design) hits)")


if __name__ == "__main__":
    main()
