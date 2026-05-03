"""Iterative PLM-fusion-driven design loop for the PTE_i1 scaffold.

Runs INSIDE universal.sif (which has fused_mpnn + Bio + pandas + the
host-mounted fpocket binary). Reads precomputed PLM artifacts:

    plm_artifacts/
        esmc_log_probs.npy            (L, 20)
        saprot_log_probs.npy          (L, 20)
        fusion_bias.npy               (L, 20) -- cycle-0 bias
        fusion_log_odds_esmc.npy      (L, 20) -- calibrated log-odds
        fusion_log_odds_saprot.npy    (L, 20)
        fusion_weights.npy            (L, 2)  -- per-pos β, γ
        manifest.json

And a PositionTable (from pyrosetta.sif's classify_positions).

Per-cycle flow:
    sample (LigandMPNN with cycle-k bias)
    -> restore PDBs (REMARK 666 + HETNAM + LINK)
    -> dedup by sequence
    -> cheap seq filter (charge, OmpT, length)
    -> struct filter (h-bond + SAP-proxy)
    -> per-sequence fitness from cached PLM marginals
    -> fpocket scoring on survivors
    -> rank survivors

Across cycles:
    cycle 0:  bias = base PLM-fusion bias
    cycle k+1: bias = base + consensus augmentation from cycle k survivors
                 (only at non-fixed positions of class buried/surface/
                  first_shell/pocket; capped at 30% of L)

Final stage:
    union all cycles' survivors, dedup, pick top_k via greedy
    Hamming-distance diversity, write top_k.fasta + top_k_pdbs/.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("iterative_design_v2")


# ----------------------------------------------------------------------
# Hard-coded constants for the PTE_i1 scaffold
# ----------------------------------------------------------------------

DEFAULT_INPUT_PDB = Path(
    "/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/"
    "ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb"
)
DEFAULT_LIG_PARAMS = Path(
    "/home/woodbuse/testing_space/scaffold_optimization/"
    "ZZZ_MERGED_PRELIM_FILTER_DIR_ZZZ/params/YYE.params"
)

# REMARK 666 catalytic resnos (1-indexed PDB resseq) on chain A.
DEFAULT_CATRES = (60, 64, 128, 131, 132, 157)
CATALYTIC_HIS_RESNOS = (60, 64, 128, 132)
CHAIN = "A"

# Apptainer / cluster paths
UNIVERSAL_SIF = Path("/net/software/containers/universal.sif")
FPOCKET_BIN = Path("/net/software/lab/fpocket/bin/fpocket")

DEFAULT_OUT_ROOT = Path("/net/scratch/woodbuse")

# fused_mpnn checkpoints (same as v1 driver; see notes there)
LMPNN_CKPT = "/net/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt"
SC_CKPT = "/net/databases/mpnn/packer_weights/s_300756.pt"


# ----------------------------------------------------------------------
# Cycle config
# ----------------------------------------------------------------------


@dataclass
class CycleConfig:
    cycle_idx: int
    n_samples: int = 500
    sampling_temperature: float = 0.20
    net_charge_max: float = -10.0     # net_charge_no_HIS strictly < this
    sap_max_threshold: float = 15.0
    consensus_threshold: float = 0.85
    consensus_strength: float = 2.0


@dataclass
class IterativeRunConfig:
    cycles: list[CycleConfig] = field(default_factory=list)
    target_n_topk: int = 50
    diversity_min_hamming: int = 3
    chain: str = CHAIN
    fix_remark_666: bool = True


def default_cycles() -> list[CycleConfig]:
    """Three-cycle exploration → exploitation schedule."""
    return [
        CycleConfig(
            cycle_idx=0, n_samples=500, sampling_temperature=0.20,
            net_charge_max=-10.0, sap_max_threshold=15.0,
        ),
        CycleConfig(
            cycle_idx=1, n_samples=400, sampling_temperature=0.18,
            net_charge_max=-11.0, sap_max_threshold=13.0,
        ),
        CycleConfig(
            cycle_idx=2, n_samples=300, sampling_temperature=0.15,
            net_charge_max=-12.0, sap_max_threshold=12.0,
        ),
    ]


# ----------------------------------------------------------------------
# Stage helpers (lifted / adapted from iterative_design_PTE_i1.py)
# ----------------------------------------------------------------------


def _parse_remark_block(ref_pdb: Path) -> list[str]:
    """Pull REMARK 666 / HETNAM / LINK / REMARK PDBinfo-LABEL from ref."""
    head: list[str] = []
    with open(ref_pdb) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                break
            if (line.startswith("REMARK 666")
                or line.startswith("REMARK PDBinfo-LABEL")
                or line.startswith("HETNAM")
                or line.startswith("LINK")):
                head.append(line)
    return head


def stage_sample(
    *,
    cycle_cfg: CycleConfig,
    seed_pdb: Path,
    bias: np.ndarray,
    protein_resnos: list[int],
    fixed_resnos: Iterable[int],
    out_dir: Path,
    chain: str = CHAIN,
) -> Path:
    """Sample N candidates via LigandMPNN with the given (L,20) bias."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make sure protein_chisel src is on path (we run inside universal.sif
    # but PYTHONPATH was set by the sbatch wrapper).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.ligand_mpnn import (
        LigandMPNNConfig, sample_with_ligand_mpnn,
    )

    cfg = LigandMPNNConfig(
        temperature=cycle_cfg.sampling_temperature,
        batch_size=10,
        repack_everything=0,        # CRITICAL: keep catalytic rotamers intact
        pack_side_chains=1,         # write packed PDBs
        ligand_mpnn_use_side_chain_context=1,
        omit_AA="CX",
        # No global bias_AA -- per-residue bias is the whole point.
        extra_flags=(
            "--checkpoint_ligand_mpnn", LMPNN_CKPT,
            "--checkpoint_path_sc", SC_CKPT,
        ),
    )

    LOGGER.info(
        "stage_sample[cycle=%d]: n=%d, T=%.3f, fixed=%s, mean_abs_bias=%.4f",
        cycle_cfg.cycle_idx, cycle_cfg.n_samples,
        cycle_cfg.sampling_temperature, sorted(set(fixed_resnos)),
        float(np.abs(bias).mean()),
    )

    res = sample_with_ligand_mpnn(
        pdb_path=seed_pdb,
        chain=chain,
        fixed_resnos=sorted(set(fixed_resnos)),
        bias_per_residue=bias,
        protein_resnos=protein_resnos,
        n_samples=cycle_cfg.n_samples,
        config=cfg,
        out_dir=out_dir,
        parent_design_id=f"PTE_i1_c{cycle_cfg.cycle_idx}",
        via_apptainer=False,         # already inside universal.sif
    )
    cand_fasta = out_dir / "candidates.fasta"
    cand_tsv = out_dir / "candidates.tsv"
    res.candidate_set.to_disk(cand_fasta, cand_tsv)
    LOGGER.info("stage_sample[cycle=%d]: produced %d rows (incl. WT input header)",
                 cycle_cfg.cycle_idx, len(res.candidate_set.df))
    return cand_tsv


def stage_restore_pdbs(
    *,
    sample_dir: Path,
    ref_pdb: Path,
    out_pdb_dir: Path,
    pdb_basename: str,
    candidate_ids: list[str],
) -> dict[str, Path]:
    """Re-prepend REMARK 666/HETNAM/LINK to packed MPNN PDBs."""
    out_pdb_dir.mkdir(parents=True, exist_ok=True)
    head_lines = _parse_remark_block(ref_pdb)
    LOGGER.info("stage_restore_pdbs: prepending %d header lines", len(head_lines))

    packed_dir = sample_dir / "packed"
    if not packed_dir.is_dir():
        raise FileNotFoundError(
            f"No packed/ subdir under {sample_dir} -- did fused_mpnn "
            "run with pack_side_chains=1?"
        )

    out_map: dict[str, Path] = {}
    for cid in candidate_ids:
        m = re.match(rf"^{re.escape(pdb_basename)}_lmpnn_(\d+)$", cid)
        if not m:
            continue
        idx = int(m.group(1))
        if idx == 0:
            continue   # WT input row, no PDB
        src = packed_dir / f"{pdb_basename}_packed_{idx}_1.pdb"
        if not src.is_file():
            LOGGER.warning("stage_restore_pdbs: missing PDB %s", src.name)
            continue
        dst = out_pdb_dir / f"{cid}.pdb"
        with open(src) as fin, open(dst, "w") as fout:
            fout.writelines(head_lines)
            for line in fin:
                if line.startswith("REMARK"):
                    continue   # MPNN's redundant REMARKs
                fout.write(line)
        out_map[cid] = dst
    LOGGER.info("stage_restore_pdbs: restored %d PDBs", len(out_map))
    return out_map


OMPT_ONLY_PATTERNS = [
    ("ompT_KK", r"KK"), ("ompT_KR", r"KR"),
    ("ompT_RK", r"RK"), ("ompT_RR", r"RR"),
]


def _count_ompt_motifs(sequence: str) -> int:
    """Count OmpT-class dibasic motifs (KK/KR/RK/RR) anywhere in seq."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.filters.protease_sites import find_protease_sites
    return len(find_protease_sites(
        sequence, extra_patterns=OMPT_ONLY_PATTERNS, skip_default=True,
    ).hits)


def stage_seq_filter(
    *,
    candidates_tsv: Path,
    out_dir: Path,
    net_charge_max: float,
    wt_length: int,
    wt_ompt_count: int,
) -> Path:
    """Cheap sequence-only filter: charge, length, OmpT motifs.

    OmpT threshold is "no worse than WT" — we don't degrade dibasic-motif
    count below the natural enzyme. This is the right policy: WT
    expresses fine in E. coli, so matching WT's dibasic count is OK.
    Critically, the catalytic K157 forces a KK at 157-158 in this
    scaffold, so a strict 0-motif rule rejects WT and every faithful
    design.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.filters.protparam import protparam_metrics
    from protein_chisel.sampling.fitness_score import deduplicate_by_sequence

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(candidates_tsv, sep="\t")
    if "is_input" in df.columns:
        df = df[~df["is_input"].astype(bool)].copy()
    df = deduplicate_by_sequence(df)
    LOGGER.info("stage_seq_filter: input n=%d (post-dedup, post-WT-drop)", len(df))

    rows: list[dict] = []
    for _, row in df.iterrows():
        seq = row["sequence"]
        reasons: list[str] = []
        if len(seq) != wt_length:
            reasons.append(f"length {len(seq)} != WT {wt_length}")
        pp = protparam_metrics(seq)
        if pp.charge_at_pH7_no_HIS >= net_charge_max:
            reasons.append(
                f"net_charge_no_HIS={pp.charge_at_pH7_no_HIS:.2f} >= {net_charge_max}"
            )
        n_hits = _count_ompt_motifs(seq)
        if n_hits > wt_ompt_count:
            reasons.append(f"OmpT motifs={n_hits} > WT={wt_ompt_count}")
        rows.append({
            **row.to_dict(),
            "length": len(seq),
            "net_charge_no_HIS": pp.charge_at_pH7_no_HIS,
            "instability_index": pp.instability_index,
            "gravy": pp.gravy,
            "pi": pp.pi,
            "n_ecoli_sites": n_hits,
            "passed_seq_filter": not reasons,
            "fail_reasons": "; ".join(reasons),
        })

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0:
        survivors = out_df[out_df["passed_seq_filter"]].copy()
        rejects = out_df[~out_df["passed_seq_filter"]].copy()
    else:
        survivors = out_df.copy()
        rejects = out_df.copy()
    survivors.to_csv(out_dir / "survivors_seq.tsv", sep="\t", index=False)
    rejects.to_csv(out_dir / "rejects_seq.tsv", sep="\t", index=False)
    LOGGER.info("stage_seq_filter: %d / %d passed (charge<%.0f, OmpT<=%d)",
                 len(survivors), len(out_df), net_charge_max, wt_ompt_count)
    return out_dir / "survivors_seq.tsv"


def _read_atoms(pdb_path: Path) -> list[dict]:
    """Minimal stdlib PDB ATOM parser."""
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                atoms.append({
                    "atom_name": line[12:16].strip(),
                    "res_name": line[17:20].strip(),
                    "chain_id": line[21].strip(),
                    "res_seq": int(line[22:26].strip() or 0),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "element": line[76:78].strip(),
                })
            except (ValueError, IndexError):
                continue
    return atoms


def _detect_hbond_to_his_sidechain(
    pdb_path: Path,
    catalytic_his_resnos: Iterable[int],
    chain: str = CHAIN,
    distance_cutoff: float = 3.5,
) -> list[dict]:
    """Heavy-atom geometric H-bond detection: catalytic HIS NE2/ND1 ↔ any
    non-self protein donor/acceptor atom (N, O, S) within 3.5 Å."""
    atoms = _read_atoms(pdb_path)
    cat_set = set(catalytic_his_resnos)
    his_targets = [
        a for a in atoms
        if (a["chain_id"] == chain and a["res_seq"] in cat_set
            and a["res_name"] in ("HIS", "HIS_D", "HIP")
            and a["atom_name"] in ("ND1", "NE2"))
    ]
    DA_NAMES = {
        "N", "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
        "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "ND1", "SG",
    }
    cands = [
        a for a in atoms
        if (a["chain_id"] == chain and a["atom_name"] in DA_NAMES
            and a["element"] in ("N", "O", "S")
            and not (a["res_seq"] in cat_set
                     and a["atom_name"] in ("ND1", "NE2")))
    ]
    hits = []
    for h in his_targets:
        for c in cands:
            if abs(c["res_seq"] - h["res_seq"]) <= 1:
                continue
            d = ((h["x"]-c["x"])**2 + (h["y"]-c["y"])**2 + (h["z"]-c["z"])**2)**0.5
            if d <= distance_cutoff:
                hits.append({
                    "his_resno": h["res_seq"], "his_atom": h["atom_name"],
                    "partner_resno": c["res_seq"], "partner_atom": c["atom_name"],
                    "partner_resname": c["res_name"], "distance": round(d, 3),
                })
    return hits


# Kyte-Doolittle hydrophobicity + Tien max-SASA — same as v1 driver.
KD_HYDROPHOBICITY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
SASA_MAX_RESIDUE = {
    "A": 121, "C": 148, "D": 187, "E": 214, "F": 228, "G": 97,  "H": 216,
    "I": 195, "K": 230, "L": 191, "M": 203, "N": 187, "P": 154, "Q": 214,
    "R": 265, "S": 143, "T": 163, "V": 165, "W": 264, "Y": 255,
}
THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","HID":"H","HIE":"H","HIP":"H","HIS_D":"H","ILE":"I",
    "LEU":"L","LYS":"K","KCX":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}


def _compute_sap_proxy(pdb_path: Path) -> Optional[dict]:
    """Lauer-style SAP via freesasa SASA + Kyte-Doolittle hydrophobicity.

    Per-residue SAP_i = sum over atoms within 10 Å of CA(i):
        (SASA(j) / SASA_max(j)) * KD(restype(j))
    """
    try:
        import freesasa
    except ImportError:
        return None
    try:
        freesasa.setVerbosity(freesasa.silent)
        struct = freesasa.Structure(str(pdb_path))
        result = freesasa.calc(struct)
    except Exception as e:
        LOGGER.warning("freesasa load failed for %s: %s", pdb_path.name, e)
        return None

    n_atoms = struct.nAtoms()
    res_data: dict[tuple[str, int], dict] = {}
    for i in range(n_atoms):
        chain_id = struct.chainLabel(i)
        if chain_id != CHAIN:
            continue
        rn = struct.residueNumber(i).strip()
        try:
            res_seq = int(rn)
        except ValueError:
            continue
        rname = struct.residueName(i).strip()
        atom = struct.atomName(i).strip()
        x, y, z = struct.coord(i)
        sasa_i = result.atomArea(i)
        key = (chain_id, res_seq)
        if key not in res_data:
            res_data[key] = {"resname": rname, "ca": None, "sasa": 0.0, "atoms_xyz": []}
        res_data[key]["sasa"] += sasa_i
        res_data[key]["atoms_xyz"].append((x, y, z))
        if atom == "CA":
            res_data[key]["ca"] = (x, y, z)

    if not res_data:
        return None
    keys_sorted = sorted(res_data.keys(), key=lambda k: k[1])
    cas = []
    for k in keys_sorted:
        ca = res_data[k]["ca"]
        if ca is None:
            ca = res_data[k]["atoms_xyz"][0] if res_data[k]["atoms_xyz"] else (0, 0, 0)
        cas.append(ca)
    cas_a = np.array(cas, dtype=float)

    sap_per_res = []
    for i, k_i in enumerate(keys_sorted):
        d = np.linalg.norm(cas_a - cas_a[i], axis=1)
        nbrs = np.where(d <= 10.0)[0]
        s = 0.0
        for j in nbrs:
            rj = res_data[keys_sorted[j]]["resname"]
            aa = THREE_TO_ONE.get(rj)
            if aa is None:
                continue
            sasa_j = res_data[keys_sorted[j]]["sasa"]
            sasa_max = SASA_MAX_RESIDUE.get(aa, 200.0)
            s += (sasa_j / sasa_max) * KD_HYDROPHOBICITY.get(aa, 0.0)
        sap_per_res.append(s)

    arr = np.array(sap_per_res)
    return {
        "sap_max": float(np.max(arr)),
        "sap_mean": float(np.mean(arr)),
        "sap_p95": float(np.percentile(arr, 95)),
    }


def stage_struct_filter(
    *,
    survivors_seq_tsv: Path,
    pdb_map: dict[str, Path],
    out_dir: Path,
    sap_max_threshold: float,
    catalytic_his_resnos: Iterable[int] = CATALYTIC_HIS_RESNOS,
) -> Path:
    """Apply h-bond + SAP-proxy structural filter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(survivors_seq_tsv, sep="\t")
    LOGGER.info("stage_struct_filter: input n=%d", len(df))

    rows: list[dict] = []
    hbond_rows: list[dict] = []
    for _, row in df.iterrows():
        cid = row["id"]
        pdb = pdb_map.get(cid)
        if pdb is None or not pdb.is_file():
            rows.append({**row.to_dict(),
                         "n_hbonds_to_cat_his": 0,
                         "sap_max": float("nan"),
                         "sap_mean": float("nan"),
                         "sap_p95": float("nan"),
                         "passed_struct_filter": False,
                         "struct_fail": f"pdb_missing: {pdb}"})
            continue
        hbonds = _detect_hbond_to_his_sidechain(pdb, catalytic_his_resnos)
        for h in hbonds:
            hbond_rows.append({"id": cid, **h})
        sap = _compute_sap_proxy(pdb) or {}
        sap_max = sap.get("sap_max", float("nan"))
        reasons: list[str] = []
        if len(hbonds) < 1:
            reasons.append("no h-bonds to catalytic HIS")
        if sap_max == sap_max and sap_max > sap_max_threshold:
            reasons.append(f"sap_max={sap_max:.2f} > {sap_max_threshold}")
        rows.append({**row.to_dict(),
                     "n_hbonds_to_cat_his": len(hbonds),
                     "sap_max": sap_max,
                     "sap_mean": sap.get("sap_mean", float("nan")),
                     "sap_p95": sap.get("sap_p95", float("nan")),
                     "passed_struct_filter": not reasons,
                     "struct_fail": "; ".join(reasons)})

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0 and "passed_struct_filter" in out_df.columns:
        survivors = out_df[out_df["passed_struct_filter"]].copy()
        rejects = out_df[~out_df["passed_struct_filter"]].copy()
    else:
        # Empty input -> empty output with the canonical columns. Don't
        # crash later filters that read this file.
        empty_cols = list(df.columns) + [
            "n_hbonds_to_cat_his", "sap_max", "sap_mean", "sap_p95",
            "passed_struct_filter", "struct_fail",
        ]
        survivors = pd.DataFrame(columns=empty_cols)
        rejects = pd.DataFrame(columns=empty_cols)
    survivors.to_csv(out_dir / "survivors_struct.tsv", sep="\t", index=False)
    rejects.to_csv(out_dir / "rejects_struct.tsv", sep="\t", index=False)
    pd.DataFrame(hbond_rows).to_csv(out_dir / "hbond_details.tsv", sep="\t", index=False)
    LOGGER.info(
        "stage_struct_filter: %d / %d passed (h-bond to cat-HIS + sap_max<=%.0f)",
        len(survivors), len(out_df), sap_max_threshold,
    )
    return out_dir / "survivors_struct.tsv"


def stage_fitness_score(
    *,
    survivors_struct_tsv: Path,
    out_dir: Path,
    log_probs_esmc: np.ndarray,
    log_probs_saprot: np.ndarray,
    weights_per_position: np.ndarray,
    fitness_cache: dict,
) -> Path:
    """Score each survivor's fitness from cached PLM marginals."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.sampling.fitness_score import score_dataframe_fitness

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(survivors_struct_tsv, sep="\t")
    LOGGER.info("stage_fitness_score: input n=%d (cache size=%d)",
                 len(df), len(fitness_cache))
    scored = score_dataframe_fitness(
        df, log_probs_esmc, log_probs_saprot, weights_per_position,
        fitness_cache=fitness_cache,
    )
    out_path = out_dir / "scored.tsv"
    scored.to_csv(out_path, sep="\t", index=False)
    LOGGER.info(
        "stage_fitness_score: cache size now %d; logp_fused mean=%.3f",
        len(fitness_cache), float(scored["fitness__logp_fused_mean"].mean()),
    )
    return out_path


def _run_fpocket(pdb_path: Path, work_dir: Path) -> Optional[dict]:
    """Run fpocket. fpocket has a buffer overflow on long filenames so we
    copy to ``design.pdb`` inside work_dir before invoking."""
    work_dir.mkdir(parents=True, exist_ok=True)
    local = work_dir / "design.pdb"
    local.write_bytes(pdb_path.read_bytes())
    try:
        proc = subprocess.run(
            [str(FPOCKET_BIN), "-f", str(local)],
            cwd=str(work_dir),
            capture_output=True, text=True, timeout=300, check=False,
        )
        info_txt = work_dir / "design_out" / "design_info.txt"
        if proc.returncode != 0 or not info_txt.is_file():
            LOGGER.warning("fpocket failed for %s: rc=%d", pdb_path.name, proc.returncode)
            return None
        return _parse_fpocket_largest(info_txt)
    except subprocess.TimeoutExpired:
        LOGGER.warning("fpocket timeout (>300s) for %s", pdb_path.name)
        return None


def _parse_fpocket_largest(info_txt: Path) -> Optional[dict]:
    """Parse ``info.txt``; return the most druggable pocket dict."""
    pockets: list[dict] = []
    cur: dict = {}
    with open(info_txt) as fh:
        for line in fh:
            m = re.match(r"^Pocket\s+(\d+)\s*:", line)
            if m:
                if cur:
                    pockets.append(cur)
                cur = {"pocket_idx": int(m.group(1))}
                continue
            m = re.match(r"^\s*([A-Za-z0-9_/. ]+):\s*([+-]?[\d.eE+-]+)", line)
            if m and cur:
                key = m.group(1).strip().lower().replace(" ", "_").replace(".", "")
                try:
                    cur[key] = float(m.group(2))
                except ValueError:
                    pass
    if cur:
        pockets.append(cur)
    if not pockets:
        return None
    pockets.sort(key=lambda p: p.get("druggability_score", 0), reverse=True)
    return pockets[0]


def stage_fpocket_rank(
    *,
    scored_tsv: Path,
    pdb_map: dict[str, Path],
    out_dir: Path,
) -> Path:
    """Run fpocket on every fitness-scored survivor and emit ranked.tsv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(scored_tsv, sep="\t")
    LOGGER.info("stage_fpocket_rank: input n=%d", len(df))
    fpocket_dir = out_dir / "per_design_fpocket"
    fpocket_dir.mkdir(exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        cid = row["id"]
        pdb = pdb_map.get(cid)
        info = _run_fpocket(pdb, fpocket_dir / cid) if pdb else None
        rows.append({
            **row.to_dict(),
            "fpocket__druggability": info.get("druggability_score", 0.0) if info else 0.0,
            "fpocket__volume": info.get("volume", 0.0) if info else 0.0,
            "fpocket__mean_alpha_sphere_radius":
                info.get("mean_alpha_sphere_radius", 0.0) if info else 0.0,
            "fpocket__alpha_sphere_density":
                info.get("alpha_sphere_density", 0.0) if info else 0.0,
            "fpocket__n_pockets_found": 1 if info else 0,
        })
    ranked = pd.DataFrame(rows).sort_values(
        # primary: fitness desc; secondary: tighter pocket
        ["fitness__logp_fused_mean", "fpocket__mean_alpha_sphere_radius"],
        ascending=[False, True], na_position="last",
    )
    out_path = out_dir / "ranked.tsv"
    ranked.to_csv(out_path, sep="\t", index=False)
    LOGGER.info("stage_fpocket_rank: top fitness=%.3f, top radius=%.3f",
                 float(ranked["fitness__logp_fused_mean"].iloc[0]) if len(ranked) else float('nan'),
                 float(ranked["fpocket__mean_alpha_sphere_radius"].iloc[0]) if len(ranked) else float('nan'))
    return out_path


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def stage_diverse_topk(
    *,
    pool_df: pd.DataFrame,
    pdb_map: dict[str, Path],
    out_dir: Path,
    target_k: int,
    min_hamming: int,
) -> Path:
    """Greedy top-K with Hamming-distance diversity over all sequences in
    ``pool_df`` (already deduped, sorted descending by fitness)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(pool_df) == 0:
        LOGGER.warning("stage_diverse_topk: empty pool")
        return out_dir / "topk.tsv"

    selected_idx: list[int] = []
    selected_seqs: list[str] = []
    for i, row in pool_df.iterrows():
        seq = row["sequence"]
        if all(_hamming(seq, s) >= min_hamming for s in selected_seqs):
            selected_idx.append(i)
            selected_seqs.append(seq)
        if len(selected_idx) >= target_k:
            break
    top = pool_df.loc[selected_idx].copy()
    top.to_csv(out_dir / "topk.tsv", sep="\t", index=False)
    fasta = out_dir / "topk.fasta"
    with open(fasta, "w") as fh:
        for _, row in top.iterrows():
            fh.write(f">{row['id']}\n{row['sequence']}\n")
    pdb_out = out_dir / "topk_pdbs"
    pdb_out.mkdir(exist_ok=True)
    for _, row in top.iterrows():
        src = pdb_map.get(row["id"])
        if src and src.is_file():
            shutil.copy2(src, pdb_out / src.name)
    LOGGER.info("stage_diverse_topk: selected %d / %d (target=%d, min_hamming=%d)",
                 len(top), len(pool_df), target_k, min_hamming)
    return out_dir / "topk.tsv"


# ----------------------------------------------------------------------
# One iteration cycle
# ----------------------------------------------------------------------


def run_cycle(
    *,
    cycle_cfg: CycleConfig,
    seed_pdb: Path,
    base_bias: np.ndarray,
    log_probs_esmc: np.ndarray,
    log_probs_saprot: np.ndarray,
    weights_per_position: np.ndarray,
    position_classes: list[str],
    protein_resnos: list[int],
    fixed_resnos: list[int],
    survivors_prev: Optional[pd.DataFrame],
    cycle_dir: Path,
    fitness_cache: dict,
    wt_length: int,
    wt_ompt_count: int,
    catalytic_his_resnos: Iterable[int] = CATALYTIC_HIS_RESNOS,
) -> tuple[Optional[pd.DataFrame], dict[str, Path]]:
    """Run ONE iteration cycle. Returns (ranked DataFrame, pdb_map)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.sampling.iterative_fusion import (
        IterationBiasConfig, build_iteration_bias,
    )

    cycle_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0. Build cycle-k bias --------------------------------------
    bias_dir = cycle_dir / "00_bias"
    bias_dir.mkdir(exist_ok=True)
    if survivors_prev is None or len(survivors_prev) == 0:
        bias_k = base_bias.copy()
        telem = {"used_consensus": False, "n_survivors": 0}
    else:
        # Convert catalytic resnos to zero-indexed protein-array indices
        resno_to_idx = {r: i for i, r in enumerate(protein_resnos)}
        fixed_idx = [resno_to_idx[r] for r in fixed_resnos if r in resno_to_idx]
        cfg = IterationBiasConfig(
            consensus_threshold=cycle_cfg.consensus_threshold,
            consensus_strength=cycle_cfg.consensus_strength,
        )
        bias_k, t = build_iteration_bias(
            base_bias=base_bias,
            survivor_sequences=survivors_prev["sequence"].tolist(),
            position_classes=position_classes,
            fixed_resnos_zero_indexed=fixed_idx,
            config=cfg,
        )
        telem = {
            "used_consensus": True,
            "n_survivors": t.n_survivors,
            "n_positions_eligible": t.n_positions_eligible,
            "n_positions_augmented": t.n_positions_augmented,
            "augmented_resnos_1idx": t.augmented_resnos,
            "capped": t.capped,
        }
    np.save(bias_dir / "bias.npy", bias_k)
    with open(bias_dir / "telemetry.json", "w") as fh:
        json.dump(telem, fh, indent=2)

    # ---- 1. Sample --------------------------------------------------
    sample_dir = cycle_dir / "01_sample"
    cand_tsv = stage_sample(
        cycle_cfg=cycle_cfg, seed_pdb=seed_pdb, bias=bias_k,
        protein_resnos=protein_resnos, fixed_resnos=fixed_resnos,
        out_dir=sample_dir,
    )

    # ---- 2. Restore PDBs --------------------------------------------
    cand_df = pd.read_csv(cand_tsv, sep="\t")
    pdb_basename = seed_pdb.stem
    pdb_map = stage_restore_pdbs(
        sample_dir=sample_dir, ref_pdb=seed_pdb,
        out_pdb_dir=sample_dir / "pdbs_restored",
        pdb_basename=pdb_basename,
        candidate_ids=cand_df["id"].tolist(),
    )

    # ---- 3. Cheap seq filter (does its own dedup) -------------------
    seq_filter_dir = cycle_dir / "02_seq_filter"
    survivors_seq = stage_seq_filter(
        candidates_tsv=cand_tsv, out_dir=seq_filter_dir,
        net_charge_max=cycle_cfg.net_charge_max,
        wt_length=wt_length,
        wt_ompt_count=wt_ompt_count,
    )

    # ---- 4. Struct filter -------------------------------------------
    struct_filter_dir = cycle_dir / "03_struct_filter"
    survivors_struct = stage_struct_filter(
        survivors_seq_tsv=survivors_seq, pdb_map=pdb_map,
        out_dir=struct_filter_dir,
        sap_max_threshold=cycle_cfg.sap_max_threshold,
        catalytic_his_resnos=catalytic_his_resnos,
    )

    n_struct = len(pd.read_csv(survivors_struct, sep="\t"))
    if n_struct == 0:
        LOGGER.warning("cycle %d: zero struct survivors -- nothing to score/rank",
                        cycle_cfg.cycle_idx)
        return None, pdb_map

    # ---- 5. Fitness scoring -----------------------------------------
    fitness_dir = cycle_dir / "04_fitness"
    scored = stage_fitness_score(
        survivors_struct_tsv=survivors_struct, out_dir=fitness_dir,
        log_probs_esmc=log_probs_esmc,
        log_probs_saprot=log_probs_saprot,
        weights_per_position=weights_per_position,
        fitness_cache=fitness_cache,
    )

    # ---- 6. Fpocket rank --------------------------------------------
    fpocket_dir = cycle_dir / "05_fpocket"
    ranked = stage_fpocket_rank(
        scored_tsv=scored, pdb_map=pdb_map, out_dir=fpocket_dir,
    )
    ranked_df = pd.read_csv(ranked, sep="\t")
    return ranked_df, pdb_map


# ----------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed_pdb", type=Path, default=DEFAULT_INPUT_PDB)
    p.add_argument("--ligand_params", type=Path, default=DEFAULT_LIG_PARAMS)
    p.add_argument("--plm_artifacts_dir", type=Path, required=True)
    p.add_argument("--position_table", type=Path, required=True)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--target_k", type=int, default=50)
    p.add_argument("--min_hamming", type=int, default=3)
    p.add_argument("--cycles", type=int, default=3,
                   help="Use the default 3-cycle schedule when 3 (default), "
                        "or override with a single-cycle short test (1).")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / f"iterative_design_v2_PTE_i1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("=== run dir: %s ===", run_dir)

    # ---- Load PLM artifacts -----------------------------------------
    art = args.plm_artifacts_dir
    log_probs_esmc = np.load(art / "esmc_log_probs.npy")
    log_probs_saprot = np.load(art / "saprot_log_probs.npy")
    base_bias = np.load(art / "fusion_bias.npy")
    weights_per_position = np.load(art / "fusion_weights.npy")
    LOGGER.info("loaded PLM artifacts: L=%d, mean_abs_bias=%.4f",
                 base_bias.shape[0], float(np.abs(base_bias).mean()))

    # ---- Load PositionTable -----------------------------------------
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.io.schemas import PositionTable
    from protein_chisel.io.pdb import extract_sequence

    pt = PositionTable.from_parquet(args.position_table)
    protein_rows = pt.df[pt.df["is_protein"]].sort_values("resno").reset_index(drop=True)
    position_classes = protein_rows["class"].tolist()
    protein_resnos = protein_rows["resno"].astype(int).tolist()
    L = len(protein_resnos)
    LOGGER.info("position table: L=%d, class counts=%s",
                 L, protein_rows["class"].value_counts().to_dict())

    wt_seq = extract_sequence(args.seed_pdb, chain=CHAIN)
    if len(wt_seq) != L:
        raise RuntimeError(f"WT seq length {len(wt_seq)} != PositionTable {L}")
    # Threshold OmpT motifs at WT count: don't reject sequences that
    # are no worse than WT for E. coli expression. WT has dibasic motifs
    # that are forced by the catalytic K157 (KK at 157-158); a strict
    # 0-motif rule rejects WT and every faithful design.
    wt_ompt_count = _count_ompt_motifs(wt_seq)
    LOGGER.info("WT OmpT motif count: %d (filter threshold)", wt_ompt_count)

    # ---- Cycle schedule ---------------------------------------------
    cycles = default_cycles()
    if args.cycles == 1:
        cycles = cycles[:1]   # short-test mode
    elif args.cycles != 3:
        # Honor any positive int by truncating / extending the default schedule.
        cycles = cycles[: max(1, args.cycles)]
    LOGGER.info("cycle schedule: %d cycles", len(cycles))

    # ---- Loop cycles ------------------------------------------------
    all_ranked: list[pd.DataFrame] = []
    all_pdb_maps: dict[str, Path] = {}
    fitness_cache: dict = {}
    survivors_prev: Optional[pd.DataFrame] = None
    fixed_resnos = list(DEFAULT_CATRES)

    for cyc in cycles:
        cycle_dir = run_dir / f"cycle_{cyc.cycle_idx:02d}"
        ranked_df, pdb_map = run_cycle(
            cycle_cfg=cyc, seed_pdb=args.seed_pdb,
            base_bias=base_bias,
            log_probs_esmc=log_probs_esmc,
            log_probs_saprot=log_probs_saprot,
            weights_per_position=weights_per_position,
            position_classes=position_classes,
            protein_resnos=protein_resnos,
            fixed_resnos=fixed_resnos,
            survivors_prev=survivors_prev,
            cycle_dir=cycle_dir,
            fitness_cache=fitness_cache,
            wt_length=L,
            wt_ompt_count=wt_ompt_count,
        )
        if ranked_df is not None and len(ranked_df) > 0:
            ranked_df = ranked_df.copy()
            ranked_df["cycle"] = cyc.cycle_idx
            all_ranked.append(ranked_df)
            survivors_prev = ranked_df
        all_pdb_maps.update(pdb_map)

    # ---- Final pool: concat + dedup --------------------------------
    final_dir = run_dir / "final_topk"
    final_dir.mkdir(parents=True, exist_ok=True)
    if all_ranked:
        from protein_chisel.sampling.fitness_score import deduplicate_by_sequence

        pool = pd.concat(all_ranked, ignore_index=True)
        pool = deduplicate_by_sequence(pool)
        pool = pool.sort_values(
            ["fitness__logp_fused_mean", "fpocket__mean_alpha_sphere_radius"],
            ascending=[False, True], na_position="last",
        ).reset_index(drop=True)
        pool.to_csv(final_dir / "all_survivors.tsv", sep="\t", index=False)
        LOGGER.info("final pool: %d unique survivors across %d cycles",
                     len(pool), len(all_ranked))

        stage_diverse_topk(
            pool_df=pool, pdb_map=all_pdb_maps,
            out_dir=final_dir,
            target_k=args.target_k, min_hamming=args.min_hamming,
        )
    else:
        LOGGER.warning("final: zero survivors across all cycles!")

    # ---- Manifest ---------------------------------------------------
    manifest = {
        "pipeline": "iterative_design_v2",
        "seed_pdb": str(args.seed_pdb),
        "ligand_params": str(args.ligand_params),
        "plm_artifacts_dir": str(args.plm_artifacts_dir),
        "position_table": str(args.position_table),
        "fixed_resnos": list(DEFAULT_CATRES),
        "catalytic_his_resnos": list(CATALYTIC_HIS_RESNOS),
        "wt_length": L,
        "target_k": args.target_k,
        "diversity_min_hamming": args.min_hamming,
        "n_cycles_run": len(cycles),
        "cycle_configs": [asdict(c) for c in cycles],
        "outputs": {
            "run_dir": str(run_dir),
            "final_topk_fasta": str(final_dir / "topk.fasta"),
            "final_topk_pdbs": str(final_dir / "topk_pdbs"),
            "all_survivors": str(final_dir / "all_survivors.tsv"),
        },
        "started_at": timestamp,
    }
    with open(run_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("=== DONE -- top-K at %s ===", final_dir)


if __name__ == "__main__":
    main()
