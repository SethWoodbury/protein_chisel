"""Aggregate optimization metrics across the 6 PLM × device benchmark runs.

For each run, read final_topk/scored.tsv (or final_topk.tsv) and report
mean ± SD for every metric the iterative_design_v2 driver optimizes:
  - fitness, sap_max, charge, pI
  - fpocket: druggability, volume, bottleneck_radius, hydrophobicity, n_alpha_spheres, score
  - ligand_int: strength_total, n_hbond, n_hydrophobic, n_total
  - preorganization metrics
  - n_hbonds_to_cat_his
  - clash counts
  - diversity (pairwise hamming)
  - AA composition vs WT

Output: a side-by-side markdown table.
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np

RUNS = [
    ("GPU_300m+35m",  "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260504-151837-837-pid1482610"),
    ("GPU_600m+650m", "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260504-153120-648-pid1419866"),
    ("GPU_600m+1.3b", "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260504-152450-564-pid1489494"),
    ("CPU_300m+35m",  "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260504-154730-219-pid2043383"),
    ("CPU_600m+650m", "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260504-155419-631-pid3116317"),
    ("CPU_600m+1.3b", "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260504-155819-436-pid311520"),
]

# Optimization metrics to summarize (mean ± SD)
OPT_METRICS = [
    "fitness__logp_fused_mean",
    "delta_fitness_vs_wt",
    "net_charge_no_HIS",
    "pi",
    "sap_max",
    "sap_p95",
    "boman",
    "aliphatic",
    "gravy",
    "fpocket__druggability",
    "fpocket__volume",
    "fpocket__score",
    "fpocket__bottleneck_radius",
    "fpocket__hydrophobicity_score",
    "fpocket__n_alpha_spheres",
    "fpocket__apolar_atoms_pct",
    "ligand_int__strength_total",
    "ligand_int__n_hbond",
    "ligand_int__strength_hbond",
    "ligand_int__n_salt_bridge",
    "ligand_int__n_hydrophobic",
    "ligand_int__strength_hydrophobic",
    "ligand_int__n_total",
    "preorg__n_hbonds_to_cat",
    "preorg__strength_total",
    "preorg__n_hbonds_within_shells",
    "preorg__interactome_density",
    "n_hbonds_to_cat_his",
    "clash__n_total",
    "clash__n_to_catalytic",
    "clash__n_to_ligand",
]

AAS = "ACDEFGHIKLMNPQRSTVWY"


def load_topk_df(run_dir: Path) -> pd.DataFrame:
    cands = [
        run_dir / "final_topk" / "topk.tsv",
        run_dir / "final_topk" / "scored.tsv",
        run_dir / "final_topk" / "final_topk.tsv",
    ]
    for p in cands:
        if p.exists():
            return pd.read_csv(p, sep="\t")
    # Look for any .tsv in final_topk
    final = run_dir / "final_topk"
    if final.is_dir():
        for f in final.iterdir():
            if f.suffix == ".tsv":
                return pd.read_csv(f, sep="\t")
    raise FileNotFoundError(f"no top-k tsv under {run_dir}/final_topk/")


def aa_counts(seqs: list[str]) -> dict:
    return {a: np.array([s.count(a) for s in seqs]) for a in AAS}


def pairwise_hamming(seqs: list[str]) -> tuple[float, float]:
    n = len(seqs)
    if n < 2:
        return 0.0, 0.0
    L = len(seqs[0])
    arr = np.array([[ord(c) for c in s] for s in seqs], dtype=np.uint8)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(int((arr[i] != arr[j]).sum()))
    a = np.array(dists, dtype=float)
    return float(a.mean()), float(a.std())


def hamming_to_wt(seqs: list[str], wt: str) -> tuple[float, float]:
    arr = []
    wt_a = np.array([ord(c) for c in wt], dtype=np.uint8)
    for s in seqs:
        s_a = np.array([ord(c) for c in s], dtype=np.uint8)
        L = min(len(wt_a), len(s_a))
        arr.append(int((s_a[:L] != wt_a[:L]).sum()))
    a = np.array(arr, dtype=float)
    return float(a.mean()), float(a.std())


def fmt(mean: float, sd: float, prec: int = 2) -> str:
    if np.isnan(mean) or np.isnan(sd):
        return "—"
    return f"{mean:.{prec}f} ± {sd:.{prec}f}"


def main() -> None:
    rows: dict[str, dict[str, str]] = {}
    aa_per_run: dict[str, dict[str, tuple[float, float]]] = {}
    diversity: dict[str, dict[str, str]] = {}

    wt_seq = None

    for label, rdir in RUNS:
        rdir_p = Path(rdir)
        try:
            df = load_topk_df(rdir_p)
        except FileNotFoundError as e:
            print(f"!! {label}: {e}", file=sys.stderr)
            continue

        run_metrics = {}
        for m in OPT_METRICS:
            if m in df.columns:
                vals = pd.to_numeric(df[m], errors="coerce").dropna()
                if len(vals) > 0:
                    prec = 0 if "n_" in m or "count" in m else 2
                    run_metrics[m] = fmt(float(vals.mean()), float(vals.std(ddof=0)), prec)
                else:
                    run_metrics[m] = "—"
            else:
                run_metrics[m] = "(missing)"
        rows[label] = run_metrics

        # AA counts
        seq_col = "sequence" if "sequence" in df.columns else "seq" if "seq" in df.columns else None
        if seq_col:
            seqs = df[seq_col].astype(str).tolist()
            counts = aa_counts(seqs)
            aa_per_run[label] = {a: (float(c.mean()), float(c.std(ddof=0))) for a, c in counts.items()}
            ph_mean, ph_sd = pairwise_hamming(seqs)

            # Try to grab WT from a manifest or recompute from seed
            if wt_seq is None:
                manifest = rdir_p.parent / "manifest.json"
                # Just use the seed PDB sequence (from first run, all share the same seed)
                pass

            # WT-vs identity needs WT sequence
            wt_path = "/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb"
            if wt_seq is None:
                try:
                    sys.path.insert(0, "/home/woodbuse/codebase_projects/protein_chisel/src")
                    from protein_chisel.io.pdb import extract_sequence
                    wt_seq = extract_sequence(Path(wt_path), chain="A")
                except Exception as e:
                    print(f"!! could not read WT from {wt_path}: {e}", file=sys.stderr)

            if wt_seq:
                hwt_mean, hwt_sd = hamming_to_wt(seqs, wt_seq)
                ident_mean = 100.0 * (1 - hwt_mean / len(wt_seq))
                ident_sd = 100.0 * hwt_sd / len(wt_seq)
                diversity[label] = {
                    "pairwise_hamming": fmt(ph_mean, ph_sd, 1),
                    "hamming_to_wt": fmt(hwt_mean, hwt_sd, 1),
                    "identity_to_wt_pct": fmt(ident_mean, ident_sd, 1),
                }

    # ---------- emit table ----------
    headers = list(rows.keys())
    print("\n## Optimized metrics (mean ± SD across top-K)\n")
    print("| metric | " + " | ".join(headers) + " |")
    print("|" + "---|" * (len(headers)+1))
    for m in OPT_METRICS:
        line = "| " + m + " | " + " | ".join(rows.get(h, {}).get(m, "—") for h in headers) + " |"
        print(line)

    # AA composition (count per AA, mean ± SD; then ratio to WT)
    print("\n## Amino-acid composition (count per AA, mean ± SD, n=top-K)\n")
    if wt_seq:
        print(f"WT sequence length L={len(wt_seq)}; WT counts shown in last column.\n")
    print("| AA | " + " | ".join(headers) + (" | WT |" if wt_seq else " |"))
    print("|" + "---|" * (len(headers) + (2 if wt_seq else 1)))
    for a in AAS:
        wt_count = wt_seq.count(a) if wt_seq else None
        cells = []
        for h in headers:
            d = aa_per_run.get(h, {}).get(a)
            cells.append(fmt(d[0], d[1], 1) if d else "—")
        wt_cell = f" {wt_count} |" if wt_seq else ""
        print(f"| {a} | " + " | ".join(cells) + (f" |{wt_cell}" if wt_seq else " |"))

    print("\n## Diversity\n")
    print("| | " + " | ".join(headers) + " |")
    print("|" + "---|" * (len(headers)+1))
    for k in ["pairwise_hamming", "hamming_to_wt", "identity_to_wt_pct"]:
        print("| " + k + " | " + " | ".join(diversity.get(h, {}).get(k, "—") for h in headers) + " |")


if __name__ == "__main__":
    main()
