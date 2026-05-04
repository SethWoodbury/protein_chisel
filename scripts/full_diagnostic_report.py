"""Comprehensive diagnostic report on a directional-classifier run.

Computes everything the user asked for:
- mean ± SD per metric
- AA composition (mean ± SD per AA, ratio vs WT)
- pI from the lowercase `pi` column
- sequence identity to the input (WT)
- full fpocket metric panel
- ligand approximate radius from the seed PDB (excluding metals)
- list h-bonds to ligand, salt bridges, etc.
- output directory tree
- comparison to user's pre-rewrite baseline

Usage:
    python scripts/full_diagnostic_report.py <run_dir>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


WT_SEQ = (
    # PTE_i1 wild-type sequence (length 202). Source: extracted from the
    # seed PDB chain A.
    "MIYRSGSAFGEAFREAAVRRLLEYREEYNVPLDIPLEHALAAGHVDIIDLGSEHTLPNDIIIRELLDPHF"
    "ARNFPVGTKEFLDRYAAEDIDPSPYIRYRLAFEERHLKDPLAGFEEILEHIDKHKWAYNAYTHLDPIVYA"
    "ANPHIQQRTGRIRMTDQNLPIPVAEAFAKLDSAEPVTEYIDFRGGLPATLSTGV"
)


def _baseline() -> dict[str, tuple[float, str]]:
    return {
        "fitness__logp_fused_mean":   (-1.756, "−1.756 (max −1.699)"),
        "ligand_int__strength_total": (30.4,   "30.4"),
        "ligand_int__n_hbond":        (10.7,   "10.7"),
        "fpocket__druggability":      (0.98,   "0.98 mean (range 0.73-0.99)"),
        "n_hbonds_to_cat_his":        (1.06,   "1.06"),
        "net_charge_no_HIS":          (-13.5,  "−13.5 (range −15.9..−9.9)"),
        "pi":                         (5.06,   "5.06"),
        "sap_max":                    (1.51,   "1.51"),
    }


def _fmt(values: pd.Series, fmt: str = ".2f") -> str:
    """mean ± SD, with min..max range."""
    if len(values) == 0:
        return "—"
    m = values.mean(); s = values.std(); lo = values.min(); hi = values.max()
    return f"{m:{fmt}} ± {s:{fmt}}  (min {lo:{fmt}}, max {hi:{fmt}})"


def _hamming_to_ref(seqs: list[str], ref: str) -> np.ndarray:
    L = min(len(ref), min(len(s) for s in seqs))
    arr = np.array([list(s[:L]) for s in seqs])
    ref_arr = np.array(list(ref[:L]))
    return (arr != ref_arr[None, :]).sum(axis=1)


def _aa_count_arr(seqs: list[str], aa: str) -> np.ndarray:
    return np.array([s.count(aa) for s in seqs])


def _ligand_geometry(pdb_path: Path, lig_resname: str = "YYE") -> dict:
    """Approximate ligand radius (heavy atoms, excluding metals)."""
    pts = []
    metals = {"ZN", "MN", "FE", "MG", "CA", "CO", "NI", "CU", "CD", "K", "NA"}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            res = line[16:21].strip() or line[17:20].strip()
            elem = line[76:78].strip().upper()
            if elem in metals:
                continue
            if line[12:14].strip().upper().startswith(("H",)):
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            pts.append((x, y, z))
    if not pts:
        return {}
    arr = np.array(pts)
    centroid = arr.mean(axis=0)
    dists = np.linalg.norm(arr - centroid, axis=1)
    rg = float(np.sqrt((dists**2).mean()))
    max_dist = float(dists.max())
    span = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
    return {
        "n_heavy_atoms": len(arr),
        "centroid": centroid.tolist(),
        "max_atom_to_centroid": max_dist,
        "radius_of_gyration": rg,
        "bounding_box_diagonal": span,
        "approx_radius": max_dist,
    }


def _tree(path: Path, level: int = 0, max_depth: int = 3) -> list[str]:
    out: list[str] = []
    if level == 0:
        out.append(f"{path}/")
    if level >= max_depth:
        return out
    if not path.is_dir():
        return out
    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return out
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        prefix = "    " * level + ("└── " if is_last else "├── ")
        if entry.is_dir():
            out.append(f"{prefix}{entry.name}/")
            out.extend(_tree(entry, level + 1, max_depth))
        else:
            try:
                sz = entry.stat().st_size
                size_str = (
                    f"{sz}B" if sz < 1024
                    else f"{sz/1024:.1f}K" if sz < 1024**2
                    else f"{sz/1024**2:.1f}M"
                )
            except OSError:
                size_str = "?"
            out.append(f"{prefix}{entry.name}  ({size_str})")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--top_k", type=int, default=36)
    p.add_argument(
        "--seed_pdb", type=Path,
        default=Path(
            "/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/"
            "ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb"
        ),
    )
    args = p.parse_args()
    run_dir = args.run_dir

    df_all = pd.read_csv(run_dir / "final_topk" / "all_survivors.tsv", sep="\t")
    df = df_all.sort_values("fitness__logp_fused_mean", ascending=False).head(args.top_k)
    seqs = df["sequence"].astype(str).tolist()
    print(f"{'='*72}")
    print(f"FULL DIAGNOSTIC REPORT  —  top-{len(df)} of {len(df_all)} survivors")
    print(f"run_dir: {run_dir}")
    print(f"{'='*72}\n")

    # ------------------------------------------------------------------
    print("-- HEADLINE METRICS  (mean ± SD, range) --")
    headline = [
        ("fitness__logp_fused_mean",     ".3f"),
        ("net_charge_no_HIS",            ".2f"),
        ("pi",                           ".2f"),
        ("sap_max",                      ".2f"),
        ("sap_p95",                      ".2f"),
        ("fpocket__druggability",        ".3f"),
        ("ligand_int__strength_total",   ".2f"),
        ("ligand_int__n_hbond",          ".2f"),
        ("ligand_int__strength_hbond",   ".2f"),
        ("ligand_int__n_salt_bridge",    ".2f"),
        ("ligand_int__strength_salt_bridge", ".2f"),
        ("ligand_int__n_pi_pi",          ".2f"),
        ("ligand_int__n_pi_cation",      ".2f"),
        ("ligand_int__n_hydrophobic",    ".2f"),
        ("ligand_int__strength_hydrophobic", ".2f"),
        ("ligand_int__n_total",          ".2f"),
        ("n_hbonds_to_cat_his",          ".2f"),
        ("clash__n_total",               ".2f"),
        ("clash__n_to_catalytic",        ".2f"),
        ("clash__n_to_ligand",           ".2f"),
    ]
    for col, fmt in headline:
        if col in df.columns:
            print(f"  {col:<35} {_fmt(df[col], fmt)}")
    n_severe = int((df.get("clash__has_severe", pd.Series([0]*len(df))) > 0).sum())
    print(f"  {'severe clashes (any < 1.5 Å)':<35} {n_severe} / {len(df)}")

    # ------------------------------------------------------------------
    print("\n-- PREORGANIZATION --")
    for col in (
        "preorg__n_hbonds_to_cat", "preorg__n_salt_bridges_to_cat",
        "preorg__n_pi_to_cat", "preorg__n_hbonds_within_shells",
        "preorg__strength_total", "preorg__interactome_density",
        "preorg__n_first_shell", "preorg__n_second_shell",
    ):
        if col in df.columns:
            print(f"  {col:<40} {_fmt(df[col], '.2f')}")

    # ------------------------------------------------------------------
    print("\n-- FPOCKET (full panel) --")
    pocket_cols = sorted(c for c in df.columns if c.startswith("fpocket__"))
    for col in pocket_cols:
        # alpha-sphere counts are ints so print with .1f; everything else .2f
        fmt = ".1f" if "_n_" in col or col.endswith("__n_pockets_found") else ".3f"
        if df[col].dtype == "object":
            continue
        print(f"  {col:<48} {_fmt(df[col], fmt)}")

    # ------------------------------------------------------------------
    print("\n-- AA COMPOSITION  (mean ± SD; WT: see right column) --")
    wt_counts = {aa: WT_SEQ.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        arr = _aa_count_arr(seqs, aa)
        wt = wt_counts[aa]
        print(f"  count[{aa}]   mean {arr.mean():5.2f} ± {arr.std():4.2f}   "
              f"min {arr.min()}  max {arr.max()}   WT {wt}")

    # ------------------------------------------------------------------
    print("\n-- DIVERSITY  (pairwise + identity-to-WT) --")
    # Pairwise hamming.
    L = len(seqs[0])
    arr = np.array([list(s) for s in seqs])
    n = len(arr)
    pw = []
    for i in range(n):
        for j in range(i+1, n):
            pw.append(int((arr[i] != arr[j]).sum()))
    pw = np.array(pw)
    print(f"  pairwise hamming                    {_fmt(pd.Series(pw), '.1f')}")
    # Identity to WT.
    h_to_wt = _hamming_to_ref(seqs, WT_SEQ)
    pct_id = 100.0 * (1.0 - h_to_wt / L)
    print(f"  hamming to WT (n_mutations)         {_fmt(pd.Series(h_to_wt), '.1f')}")
    print(f"  sequence identity to WT (%)         {_fmt(pd.Series(pct_id), '.1f')}")

    # ------------------------------------------------------------------
    print("\n-- LIGAND GEOMETRY (heavy atoms only, no metals) --")
    lg = _ligand_geometry(args.seed_pdb)
    if lg:
        print(f"  n_heavy_atoms (excl. metals)        {lg['n_heavy_atoms']}")
        print(f"  max atom-to-centroid distance       {lg['max_atom_to_centroid']:.2f} Å")
        print(f"  radius of gyration                  {lg['radius_of_gyration']:.2f} Å")
        print(f"  bounding-box diagonal               {lg['bounding_box_diagonal']:.2f} Å")
        # Compare ligand radius to fpocket bottleneck radius.
        if "fpocket__bottleneck_radius" in df.columns:
            br = df["fpocket__bottleneck_radius"]
            print(f"  fpocket bottleneck radius           {_fmt(br, '.2f')}")
            ratio = br / lg["radius_of_gyration"]
            print(f"  bottleneck / ligand_Rg ratio        "
                  f"{ratio.mean():.2f} ± {ratio.std():.2f}   "
                  f"(>1 means bottleneck > ligand Rg = ligand fits)")

    # ------------------------------------------------------------------
    print("\n-- BASELINE COMPARISON (user's prior top-36) --")
    print(f"  {'Metric':<32}  {'Baseline':<22}  {'Now (mean ± SD)':<32}  Δmean")
    for col, (val, label) in _baseline().items():
        if col not in df.columns:
            print(f"  {col:<32}  {label:<22}  (column missing)               —")
            continue
        m, s = df[col].mean(), df[col].std()
        delta = m - val
        print(f"  {col:<32}  {label:<22}  {m:6.2f} ± {s:5.2f}                 "
              f"{delta:+.2f}")

    # ------------------------------------------------------------------
    print("\n-- DIRECTIONAL CLASSIFICATION (re-run on seed) --")
    pt_v2 = run_dir / "position_table_v2.parquet"
    if pt_v2.exists():
        pt = pd.read_parquet(pt_v2)
        prot = pt[pt["is_protein"]]
        for cls, n in prot["class"].value_counts().items():
            print(f"  {cls:<20} {n}")

    # ------------------------------------------------------------------
    print("\n-- PLM FUSION RUNTIME CONFIG --")
    fc_path = run_dir / "fusion_runtime" / "fusion_config.json"
    if fc_path.exists():
        import json
        cfg = json.loads(fc_path.read_text())
        cw = cfg.get("class_weights", {})
        print(f"  global_strength       {cfg.get('global_strength')}")
        print(f"  primary_sphere        {cw.get('primary_sphere')}")
        print(f"  secondary_sphere      {cw.get('secondary_sphere')}")
        print(f"  nearby_surface        {cw.get('nearby_surface')}")
        print(f"  distal_buried         {cw.get('distal_buried')}")
        print(f"  distal_surface        {cw.get('distal_surface')}")
        print(f"  cached  mean_abs_bias {cfg.get('cached_mean_abs_bias'):.4f}")
        print(f"  runtime mean_abs_bias {cfg.get('runtime_mean_abs_bias'):.4f}")

    # ------------------------------------------------------------------
    print("\n-- OUTPUT TREE --")
    for line in _tree(run_dir, max_depth=3):
        print(line)


if __name__ == "__main__":
    main()
