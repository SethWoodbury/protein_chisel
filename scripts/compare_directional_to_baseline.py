"""Compare a directional-classifier run to the user's pre-rewrite baseline.

Reads the top-K table from a run directory and reports the same headline
metrics the user has been tracking (fitness, charge, pI, SAP, druggability,
ligand interactions, h-bonds, AA composition, Hamming diversity) plus the
NEW directional metrics surfaced by this rewrite (n_primary_sphere,
n_secondary_sphere, etc.; secondary_score distribution; preorganization
columns if present).

Usage:
    python scripts/compare_directional_to_baseline.py <run_dir>

where <run_dir> is the iterative_design_v2_PTE_i1_<ts>/ folder containing
final_topk/all_survivors.tsv.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# User's pre-rewrite top-36 baseline (from the user's prior message).
BASELINE = {
    "fitness__logp_fused_mean":       (-1.756, "−1.756 (max −1.699)"),
    "ligand_int__strength_total":     (30.4,   "30.4"),
    "ligand_int__n_hbond":            (10.7,   "10.7"),
    "fpocket__druggability":          (0.98,   "0.98 mean (range 0.73-0.99)"),
    "n_hbonds_to_cat_his":            (1.06,   "1.06"),
    "net_charge_no_HIS":              (-13.5,  "−13.5 (range −15.9..−9.9)"),
    "pI":                             (5.06,   "5.06"),
    "sap__max":                       (1.51,   "1.51 (well below 15)"),
    "n_K":                            (16.5,   "16.5 (WT 17)"),
    "n_R":                            (14.2,   "14.2 (WT 10)"),
    "n_M":                            ( 2.8,   " 2.8 (WT 8)"),
    "hamming_min":                    (22,     "22"),
    "hamming_mean":                   (39.1,   "39.1"),
    "hamming_max":                    (56,     "56"),
    "n_severe_clashes":               ( 0,     "0/36"),
}


def _load(run_dir: Path) -> pd.DataFrame:
    candidates = [
        run_dir / "final_topk" / "all_survivors.tsv",
        run_dir / "final_topk" / "topk.tsv",
    ]
    for c in candidates:
        if c.exists():
            df = pd.read_csv(c, sep="\t")
            print(f"loaded: {c}  rows={len(df)}")
            return df
    raise FileNotFoundError(f"no all_survivors.tsv or topk.tsv under {run_dir}/final_topk/")


def _hamming(seqs: list[str]) -> tuple[int, float, int]:
    """Pairwise Hamming distances (min, mean, max) over a small list."""
    if len(seqs) < 2:
        return 0, 0.0, 0
    L = len(seqs[0])
    seqs_arr = np.array([list(s) for s in seqs])
    n = len(seqs_arr)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(int(np.sum(seqs_arr[i] != seqs_arr[j])))
    if not dists:
        return 0, 0.0, 0
    return min(dists), float(np.mean(dists)), max(dists)


def _aa_count_mean(seqs: list[str], aa: str) -> float:
    return float(np.mean([s.count(aa) for s in seqs]))


def _select_topk_by_fitness(df: pd.DataFrame, k: int = 36) -> pd.DataFrame:
    """Top-K by fitness mirroring the user's baseline reporting."""
    if "fitness__logp_fused_mean" not in df.columns:
        raise KeyError("fitness__logp_fused_mean column missing")
    return df.sort_values("fitness__logp_fused_mean", ascending=False).head(k).copy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--top_k", type=int, default=36,
                    help="match the user's baseline top-36 reporting")
    args = p.parse_args()

    run_dir = args.run_dir
    df_all = _load(run_dir)
    df = _select_topk_by_fitness(df_all, k=args.top_k)
    print(f"\n=== TOP-{len(df)} by fitness__logp_fused_mean ===\n")

    print("Baseline columns (legacy):")
    print(f"  fitness__logp_fused_mean  mean={df['fitness__logp_fused_mean'].mean():.3f}  "
          f"max={df['fitness__logp_fused_mean'].max():.3f}")
    if "net_charge_no_HIS" in df.columns:
        print(f"  net_charge_no_HIS         mean={df['net_charge_no_HIS'].mean():.2f}  "
              f"range=[{df['net_charge_no_HIS'].min():.1f}, {df['net_charge_no_HIS'].max():.1f}]")
    if "pI" in df.columns:
        print(f"  pI                        mean={df['pI'].mean():.2f}")
    sap_col = next((c for c in df.columns if c.startswith("sap__max") or c == "sap_max"), None)
    if sap_col:
        print(f"  {sap_col:<25} mean={df[sap_col].mean():.2f}")
    if "fpocket__druggability" in df.columns:
        d = df["fpocket__druggability"]
        print(f"  fpocket__druggability     mean={d.mean():.3f}  "
              f"range=[{d.min():.2f}, {d.max():.2f}]")
    for c in ("ligand_int__strength_total", "ligand_int__n_hbond",
                "ligand_int__n_salt_bridge", "ligand_int__n_pi"):
        if c in df.columns:
            print(f"  {c:<30} mean={df[c].mean():.2f}")
    if "n_hbonds_to_cat_his" in df.columns:
        print(f"  n_hbonds_to_cat_his       mean={df['n_hbonds_to_cat_his'].mean():.2f}")
    if "clash__n_total" in df.columns:
        n_severe = int((df.get("clash__has_severe", pd.Series([0]*len(df))) > 0).sum())
        print(f"  clash__n_total mean       {df['clash__n_total'].mean():.2f}  "
              f"severe_count={n_severe}/{len(df)}")

    seqs = df["sequence"].astype(str).tolist()
    for aa in ("K", "R", "M", "E", "D", "L", "F", "Y", "W", "H"):
        c = _aa_count_mean(seqs, aa)
        print(f"  count[{aa}]                  mean={c:.1f}")

    h_min, h_mean, h_max = _hamming(seqs)
    print(f"  hamming                   min={h_min}  mean={h_mean:.1f}  max={h_max}")

    print()
    print("New directional / preorganization metrics (if present):")
    for c in (
        "preorganization__n_first_shell",
        "preorganization__n_second_shell",
        "preorganization__hbonds_to_cat",
        "preorganization__salt_bridges_to_cat",
        "preorganization__strength_total",
        "preorganization__interactome_density",
        "fpocket__bottleneck_radius",
        "fpocket__polar_alpha_sphere_proportion",
        "fpocket__hydrophobicity_score",
        "fpocket__charge_score",
    ):
        if c in df.columns:
            print(f"  {c:<45} mean={df[c].mean():.2f}")

    print()
    print("=== BASELINE COMPARISON (user's prior top-36) ===")
    print(f"{'Metric':<32}  {'Baseline':<32}  {'Now (top-' + str(len(df)) + ')':<20}  Δ")
    for k, (val, label) in BASELINE.items():
        if k == "n_severe_clashes":
            now = int((df.get("clash__has_severe", pd.Series([0]*len(df))) > 0).sum())
        elif k == "hamming_min":
            now = h_min
        elif k == "hamming_mean":
            now = h_mean
        elif k == "hamming_max":
            now = h_max
        elif k == "n_K":
            now = _aa_count_mean(seqs, "K")
        elif k == "n_R":
            now = _aa_count_mean(seqs, "R")
        elif k == "n_M":
            now = _aa_count_mean(seqs, "M")
        elif k == "sap__max":
            sap_col = next((c for c in df.columns if c == "sap__max" or c == "sap_max"), None)
            now = df[sap_col].mean() if sap_col else float("nan")
        elif k in df.columns:
            now = df[k].mean()
        else:
            now = float("nan")
        delta = now - val if isinstance(now, (int, float)) and isinstance(val, (int, float)) else None
        delta_str = f"{delta:+.2f}" if delta is not None else "—"
        if isinstance(now, (int, float)):
            now_str = f"{now:.2f}" if isinstance(now, float) else f"{now}"
        else:
            now_str = "—"
        print(f"  {k:<30}  {label:<32}  {now_str:<20}  {delta_str}")

    # Look for re-classify telemetry
    telem_path = run_dir / "fusion_runtime" / "fusion_config.json"
    if telem_path.exists():
        cfg = json.loads(telem_path.read_text())
        print()
        print("Runtime PLM fusion config:")
        print(f"  global_strength      {cfg.get('global_strength')}")
        print(f"  class_weights        {cfg.get('class_weights')}")
        print(f"  cached_mean_abs_bias {cfg.get('cached_mean_abs_bias'):.4f}")
        print(f"  runtime_mean_abs_bias {cfg.get('runtime_mean_abs_bias'):.4f}")

    # PositionTable — show the new class distribution.
    pt_v2 = run_dir / "position_table_v2.parquet"
    if pt_v2.exists():
        pt_df = pd.read_parquet(pt_v2)
        print()
        print("Re-classified PositionTable (directional 6-class):")
        print(f"  classes: {pt_df[pt_df['is_protein']]['class'].value_counts().to_dict()}")
        print(f"  schema_version: {pt_df['schema_version'].iloc[0]}")


if __name__ == "__main__":
    main()
