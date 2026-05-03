"""Iterative design loop for the PTE_i1 scaffold.

Single-shot driver that:
  1. Oversamples N=500 sequences via LigandMPNN (fused_mpnn) with the
     REMARK 666 catalytic residues fixed (HIS 60, 64, 128, 132 +
     LYS 157 + GLU 131 -- including the carbamylated catalytic Lys).
  2. Restores REMARK 666 + HIS tautomers + hydrogens on every MPNN
     output PDB so downstream tools see the same metadata as the input.
  3. Applies cheap sequence-level filters:
        - net charge (no-HIS) < -10
        - E. coli expression -- no OmpT cleavage motifs
        - length matches WT
  4. Applies medium structure-level filters on survivors:
        - >=1 hydrogen bond between any catalytic HIS sidechain N and
          any non-backbone donor/acceptor (excludes hits to the ligand
          carbamate, which is the canonical catalytic geometry; we
          want sequences where the design provides EXTRA H-bond
          support to the catalytic histidines).
  5. Ranks survivors by an fpocket-derived "bottleneck" proxy:
        - mean alpha-sphere radius (smaller = tighter pocket).
  6. Picks top 50 with greedy hamming-distance diversity over
     mutable positions.

Outputs land in
``/net/scratch/woodbuse/iterative_design_PTE_i1_<timestamp>/``::

    01_sample/
        candidates.fasta           # 500 LigandMPNN outputs
        candidates.tsv             # FASTA + per-sample LigandMPNN metadata
        pdbs/                      # 500 PDB files, REMARK 666 restored
    02_seq_filter/
        survivors_seq.tsv          # post-cheap-filter (~50-200 expected)
        rejects_seq.tsv            # rejected + reason column
    03_struct_filter/
        survivors_struct.tsv       # post-h-bond filter
        rejects_struct.tsv
        hbond_details.tsv
    04_fpocket_rank/
        ranked.tsv                 # all survivors, ranked
        per_design_fpocket/<id>/   # raw fpocket output per survivor
    05_top50/
        top50.fasta
        top50.tsv
        top50_pdbs/                # the 50 PDBs ready for AF3
    manifest.json                  # per-stage counts + paths + git commit

The driver is host-runnable (host python 3.12 has Bio + numpy via
user-site). It spawns apptainer subprocesses for LigandMPNN
(universal.sif) and fpocket (host binary).

NOT a fully-polished pipeline module -- a focused driver to validate
the iterative-design workflow on a real scaffold and surface what
parts of the framework still need work.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


LOGGER = logging.getLogger("iterative_design_PTE_i1")

# ----------------------------------------------------------------------
# Hard-coded constants for this scaffold (PTE_i1)
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
# HIS 132, HIS 128, LYS 157 (carbamylated KCX), HIS 64, HIS 60, GLU 131.
DEFAULT_CATRES = (60, 64, 128, 131, 132, 157)
CATALYTIC_HIS_RESNOS = (60, 64, 128, 132)
CHAIN = "A"

# Pipeline parameters
N_SAMPLES = 500
TARGET_N_TOPK = 50
NET_CHARGE_MAX = -10.0          # net_charge_no_HIS strictly < -10
LENGTH_TOLERANCE = 0            # WT length must match exactly
HBOND_DISTANCE_CUTOFF = 3.5     # angstrom, donor-acceptor
HBOND_ANGLE_CUTOFF = 110.0      # degrees, donor-H...acceptor (lax)

# Apptainer / cluster
UNIVERSAL_SIF = Path("/net/software/containers/universal.sif")
ESMC_SIF = Path("/net/software/containers/users/woodbuse/esmc.sif")
FPOCKET_BIN = Path("/net/software/lab/fpocket/bin/fpocket")

# Output root (on /net/scratch so the cluster nodes can write)
DEFAULT_OUT_ROOT = Path("/net/scratch/woodbuse")


# ----------------------------------------------------------------------
# Stage 1: LigandMPNN sampling via the existing tool wrapper
# ----------------------------------------------------------------------


def stage_sample(
    input_pdb: Path,
    fixed_resnos: Iterable[int],
    out_dir: Path,
    n_samples: int,
    chain: str = CHAIN,
    sequence_id_prefix: str = "PTE_i1",
) -> Path:
    """Sample N candidates via LigandMPNN (fused_mpnn build).

    Uses the existing protein_chisel wrapper which spawns apptainer
    against universal.sif. Outputs FASTA + per-sample PDB.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.tools.ligand_mpnn import (
        LigandMPNNConfig, sample_with_ligand_mpnn,
    )

    # fused_mpnn defaults its checkpoint paths to /databases/... but the
    # weights live at /net/databases/... -- pass them explicitly so we
    # don't have to rebind paths.
    LMPNN_CKPT = (
        "/net/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt"
    )
    SC_CKPT = "/net/databases/mpnn/packer_weights/s_300756.pt"
    cfg = LigandMPNNConfig(
        temperature=0.2,             # mild diversity; we'll filter heavily
        batch_size=10,
        repack_everything=0,         # CRITICAL: keep catalytic rotamers intact
        pack_side_chains=1,          # write packed PDBs
        ligand_mpnn_use_side_chain_context=1,
        omit_AA="CX",
        # Push toward acidic sampling: discourages basic AAs (which fuel
        # OmpT cleavage motifs) and encourages D/E (drives net charge
        # below -10). Aligned with the lab default bias for soluble
        # acidic enzyme designs.
        bias_AA="K:-0.75,R:-0.75,E:1.0,D:1.0",
        extra_flags=(
            "--checkpoint_ligand_mpnn", LMPNN_CKPT,
            "--checkpoint_path_sc", SC_CKPT,
        ),
    )
    LOGGER.info(
        "stage_sample: %s -> %s, n=%d, fixed=%s",
        input_pdb.name, out_dir, n_samples, sorted(set(fixed_resnos)),
    )
    # We already run the driver inside universal.sif (which bundles
    # fused_mpnn + Bio + pandas + fpocket via /net/software bind), so
    # don't nest another apptainer call.
    res = sample_with_ligand_mpnn(
        pdb_path=input_pdb,
        chain=chain,
        fixed_resnos=sorted(set(fixed_resnos)),
        n_samples=n_samples,
        config=cfg,
        out_dir=out_dir,
        parent_design_id=sequence_id_prefix,
        via_apptainer=False,
    )
    cand_fasta = out_dir / "candidates.fasta"
    cand_tsv = out_dir / "candidates.tsv"
    res.candidate_set.to_disk(cand_fasta, cand_tsv)
    LOGGER.info(
        "stage_sample: produced %d candidates", len(res.candidate_set.df),
    )
    return cand_tsv


# ----------------------------------------------------------------------
# Stage 2: PDB restoration (REMARK 666 + hydrogens + HIS tautomers)
# ----------------------------------------------------------------------


def stage_restore_pdbs(
    sample_dir: Path,
    ref_pdb: Path,
    out_pdb_dir: Path,
    pdb_basename: str,
    candidate_ids: list[str],
) -> dict[str, Path]:
    """Prepend REMARK 666 + HETNAM + REMARK PDBinfo-LABEL lines from the
    reference scaffold to every fused_mpnn output PDB and write a
    candidate_id -> restored PDB mapping.

    LigandMPNN writes heavy-atom-only PDBs without REMARK 666 / HETNAM
    / REMARK PDBinfo-LABEL lines. We copy those header lines from the
    reference -- the MPNN output already has the correct catalytic
    rotamers because we ran with ``repack_everything=0``.

    fused_mpnn writes packed PDBs at::

        <sample_dir>/packed/<pdb_basename>_packed_<ix>_<c_pack>.pdb

    where ``ix`` is the 1-indexed sample number matching the FASTA
    line and ``c_pack`` is the side-chain pack index (we use 1).
    Candidate ids come from the wrapper as
    ``<pdb_basename>_lmpnn_<i:03d>`` where i==0 is the input pose
    (no PDB written) and i>=1 maps to ``ix=i``.
    """
    out_pdb_dir.mkdir(parents=True, exist_ok=True)

    # Pull REMARK 666 + HETNAM + PDBinfo-LABEL block from the reference.
    head_lines: list[str] = []
    with open(ref_pdb) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                break
            if (line.startswith("REMARK 666")
                or line.startswith("REMARK PDBinfo-LABEL")
                or line.startswith("HETNAM")
                or line.startswith("LINK")):
                head_lines.append(line)

    LOGGER.info(
        "stage_restore_pdbs: prepending %d REMARK/HETNAM/LINK lines",
        len(head_lines),
    )

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
            continue  # input pose, no PDB
        src = packed_dir / f"{pdb_basename}_packed_{idx}_1.pdb"
        if not src.is_file():
            LOGGER.warning("stage_restore_pdbs: missing PDB for %s (looked for %s)",
                            cid, src.name)
            continue
        dst = out_pdb_dir / f"{cid}.pdb"
        with open(src) as fin, open(dst, "w") as fout:
            fout.writelines(head_lines)
            for line in fin:
                if line.startswith("REMARK"):
                    continue  # skip MPNN's redundant REMARK lines
                fout.write(line)
        out_map[cid] = dst

    LOGGER.info("stage_restore_pdbs: restored %d / %d PDBs",
                 len(out_map), max(len(candidate_ids) - 1, 1))
    return out_map


# ----------------------------------------------------------------------
# Stage 3: cheap sequence-level filter
# ----------------------------------------------------------------------


def stage_seq_filter(
    candidates_tsv: Path,
    out_dir: Path,
    *,
    net_charge_max: float = NET_CHARGE_MAX,
    wt_length: int,
) -> Path:
    """Apply cheap, sequence-only filters.

    Survivors must:
      - have net_charge_no_HIS < net_charge_max (i.e. very negative)
      - pass E. coli protease-site filter (no OmpT KK/KR/RK/RR motifs in
        regions that would matter for expression -- conservative: any
        hit fails the design)
      - have length == wt_length
    """
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.filters.protparam import protparam_metrics
    from protein_chisel.filters.protease_sites import find_protease_sites

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(candidates_tsv, sep="\t")
    # Drop the input/WT row (fused_mpnn writes it as the first FASTA entry)
    if "is_input" in df.columns:
        df = df[~df["is_input"].astype(bool)].copy()
    LOGGER.info("stage_seq_filter: input n=%d (after dropping input row)", len(df))

    # OmpT-only patterns: dibasic motifs are the actual cleavage threat in
    # E. coli expression. Tryptic [KR][^P] would match nearly every protein
    # and is irrelevant for in-vivo expression -- skip it.
    OMPT_ONLY = [
        ("ompT_KK", r"KK"),
        ("ompT_KR", r"KR"),
        ("ompT_RK", r"RK"),
        ("ompT_RR", r"RR"),
    ]

    # Apply filters
    rows: list[dict] = []
    for _, row in df.iterrows():
        seq = row["sequence"]
        cid = row["id"]
        reasons: list[str] = []

        # Length
        if len(seq) != wt_length:
            reasons.append(f"length {len(seq)} != WT {wt_length}")

        # ProtParam
        pp = protparam_metrics(seq)
        if pp.charge_at_pH7_no_HIS >= net_charge_max:
            reasons.append(
                f"net_charge_no_HIS={pp.charge_at_pH7_no_HIS:.2f} >= {net_charge_max}"
            )

        # E. coli OmpT cleavage sites
        site_hits = find_protease_sites(
            seq, extra_patterns=OMPT_ONLY, skip_default=True,
        )
        n_hits = len(site_hits.hits)
        if n_hits > 0:
            reasons.append(f"e_coli OmpT sites: {n_hits}")

        keep = not reasons
        rows.append({
            "id": cid,
            "sequence": seq,
            "length": len(seq),
            "net_charge_no_HIS": pp.charge_at_pH7_no_HIS,
            "instability_index": pp.instability_index,
            "gravy": pp.gravy,
            "pi": pp.pi,
            "n_ecoli_sites": n_hits,
            "passed_seq_filter": keep,
            "fail_reasons": "; ".join(reasons) if reasons else "",
        })

    out_df = pd.DataFrame(rows)
    survivors = out_df[out_df["passed_seq_filter"]].copy()
    rejects = out_df[~out_df["passed_seq_filter"]].copy()

    survivors.to_csv(out_dir / "survivors_seq.tsv", sep="\t", index=False)
    rejects.to_csv(out_dir / "rejects_seq.tsv", sep="\t", index=False)
    LOGGER.info(
        "stage_seq_filter: %d / %d passed (rejected for: %s)",
        len(survivors), len(out_df),
        ", ".join(sorted({r.split(":")[0] for r in rejects["fail_reasons"] if r}))
        or "n/a",
    )
    return out_dir / "survivors_seq.tsv"


# ----------------------------------------------------------------------
# Stage 4: H-bond filter (>=1 H-bond to a catalytic HIS sidechain N)
# ----------------------------------------------------------------------


def _read_atoms(pdb_path: Path) -> list[dict]:
    """Minimal PDB atom parser -- pure stdlib, returns one dict per ATOM/HETATM line."""
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                atoms.append({
                    "record": line[:6].strip(),
                    "atom_id": int(line[6:11].strip() or 0),
                    "atom_name": line[12:16].strip(),
                    "alt_loc": line[16].strip(),
                    "res_name": line[17:20].strip(),
                    "chain_id": line[21].strip(),
                    "res_seq": int(line[22:26].strip() or 0),
                    "icode": line[26].strip(),
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
    *,
    distance_cutoff: float = HBOND_DISTANCE_CUTOFF,
) -> list[dict]:
    """Detect H-bonds where any catalytic HIS sidechain N is donor or
    acceptor.

    Simple geometric criterion (no hydrogens needed):
      - donor_atom (N or O of any residue) within distance_cutoff of
        a catalytic HIS NE2/ND1 atom
      - protein-protein only (not counting hits to the YYE ligand --
        that's the catalytic geometry baseline; we want EXTRA support)

    Returns one dict per detected H-bond.
    """
    atoms = _read_atoms(pdb_path)

    # Catalytic HIS sidechain N atoms (from the input PDB, on chain A).
    his_targets = [
        a for a in atoms
        if (a["chain_id"] == chain
            and a["res_seq"] in set(catalytic_his_resnos)
            and a["res_name"] in ("HIS", "HIS_D", "HIP")
            and a["atom_name"] in ("ND1", "NE2"))
    ]

    # Candidate donor/acceptor atoms: any backbone N or O of any other
    # protein residue (not the catalytic HIS itself), and any sidechain
    # heteroatom (N, O, S).
    DONOR_ACCEPTOR_NAMES = {
        "N", "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
        "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "ND1", "SG",
    }
    cand = [
        a for a in atoms
        if (a["chain_id"] == chain
            and a["atom_name"] in DONOR_ACCEPTOR_NAMES
            and not (a["res_seq"] in set(catalytic_his_resnos)
                     and a["atom_name"] in ("ND1", "NE2"))
            and a["element"] in ("N", "O", "S"))
    ]

    hbonds = []
    for h in his_targets:
        for c in cand:
            d = ((h["x"] - c["x"]) ** 2 + (h["y"] - c["y"]) ** 2 + (h["z"] - c["z"]) ** 2) ** 0.5
            # exclude self-residue and 1-2 / 1-3 (covalent neighbors)
            if abs(c["res_seq"] - h["res_seq"]) <= 1 and c["chain_id"] == h["chain_id"]:
                continue
            if d <= distance_cutoff:
                hbonds.append({
                    "his_resno": h["res_seq"],
                    "his_atom": h["atom_name"],
                    "partner_resno": c["res_seq"],
                    "partner_atom": c["atom_name"],
                    "partner_resname": c["res_name"],
                    "distance": round(d, 3),
                })
    return hbonds


# Kyte-Doolittle hydrophobicity (1982). Used as a proxy for the
# Black-Mould scale Lauer 2012 SAP uses; scaled differently but ranks
# residues nearly identically. Positive = hydrophobic.
KD_HYDROPHOBICITY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Per-residue SASA references (max sidechain SASA when fully exposed in
# Gly-X-Gly tripeptide), Tien et al. 2013.
SASA_MAX_RESIDUE = {
    "A": 121, "C": 148, "D": 187, "E": 214, "F": 228, "G": 97,  "H": 216,
    "I": 195, "K": 230, "L": 191, "M": 203, "N": 187, "P": 154, "Q": 214,
    "R": 265, "S": 143, "T": 163, "V": 165, "W": 264, "Y": 255,
}


def _compute_sap_proxy(pdb_path: Path) -> Optional[dict]:
    """Compute a SAP-style spatial aggregation proxy.

    Lauer 2012 SAP_i = sum over atoms within 10 A of CA(i):
        (SASA(j) / SASA_max(j)) * Black-Mould_hydrophobicity(restype(j))

    We use freesasa for SASA + Kyte-Doolittle (which ranks residues
    nearly identically to Black-Mould, just scaled differently). Returns:
        sap_max:     max per-residue SAP (Lauer's "hot-spot" metric)
        sap_mean:    mean over all residues
        hydrophobic_surface_area_fraction: fraction of total SASA from
                                            hydrophobic residues
    Returns None on failure (so the fpocket-rank stage can still proceed).
    """
    try:
        import freesasa
        import numpy as np
    except ImportError:
        return None
    try:
        # Suppress freesasa warnings about non-standard residues (HIS_D, KCX)
        freesasa.setVerbosity(freesasa.silent)
        struct = freesasa.Structure(str(pdb_path))
        result = freesasa.calc(struct)
    except Exception as e:
        LOGGER.warning("freesasa load failed for %s: %s", pdb_path.name, e)
        return None

    # Read CA positions + per-residue total SASA + restype, on chain A.
    n_atoms = struct.nAtoms()
    res_data: dict[tuple[str, int], dict] = {}
    for i in range(n_atoms):
        chain = struct.chainLabel(i)
        if chain != CHAIN:
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
        key = (chain, res_seq)
        if key not in res_data:
            res_data[key] = {
                "resname": rname, "resno": res_seq, "ca": None,
                "sasa": 0.0, "atoms_xyz": [],
            }
        res_data[key]["sasa"] += sasa_i
        res_data[key]["atoms_xyz"].append((x, y, z))
        if atom == "CA":
            res_data[key]["ca"] = (x, y, z)

    if not res_data:
        return None

    # Per-residue SAP = sum_{j in 10A nbrs of CA_i}
    #     (SASA_j / SASA_max_j) * KD(restype_j)
    keys_sorted = sorted(res_data.keys(), key=lambda k: k[1])
    cas = []
    for k in keys_sorted:
        ca = res_data[k]["ca"]
        if ca is None:
            ca = res_data[k]["atoms_xyz"][0] if res_data[k]["atoms_xyz"] else (0, 0, 0)
        cas.append(ca)
    cas_a = np.array(cas, dtype=float)

    sap_per_residue = []
    for i, k_i in enumerate(keys_sorted):
        # neighbors with CA within 10 A
        d = np.linalg.norm(cas_a - cas_a[i], axis=1)
        nbrs = np.where(d <= 10.0)[0]
        s = 0.0
        for j in nbrs:
            k_j = keys_sorted[j]
            rj = res_data[k_j]["resname"]
            # map non-canonical residue names to canonical 1-letter
            three_to_one = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
                "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","HID":"H","HIE":"H",
                "HIP":"H","HIS_D":"H","ILE":"I","LEU":"L","LYS":"K","KCX":"K",
                "MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
                "TYR":"Y","VAL":"V"}
            aa = three_to_one.get(rj, None)
            if aa is None:
                continue
            sasa_j = res_data[k_j]["sasa"]
            sasa_max = SASA_MAX_RESIDUE.get(aa, 200.0)
            kd = KD_HYDROPHOBICITY.get(aa, 0.0)
            s += (sasa_j / sasa_max) * kd
        sap_per_residue.append(s)

    sap_arr = np.array(sap_per_residue)
    return {
        "sap_max": float(np.max(sap_arr)),
        "sap_mean": float(np.mean(sap_arr)),
        "sap_p95": float(np.percentile(sap_arr, 95)),
        "n_residues": int(len(sap_arr)),
    }


def stage_struct_filter(
    survivors_seq_tsv: Path,
    pdb_map: dict[str, Path],
    out_dir: Path,
    catalytic_his_resnos: Iterable[int] = CATALYTIC_HIS_RESNOS,
    sap_max_threshold: float = 15.0,
) -> Path:
    """Apply structure-level h-bond filter.

    Survivors must have >=1 detected H-bond between any catalytic HIS
    sidechain N and any non-self protein heteroatom on the design.
    """
    import pandas as pd
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
                          "struct_fail": f"pdb_missing: {pdb}",
            })
            continue
        hbonds = _detect_hbond_to_his_sidechain(pdb, catalytic_his_resnos)
        for h in hbonds:
            hbond_rows.append({"id": cid, **h})
        sap = _compute_sap_proxy(pdb) or {}
        sap_max = sap.get("sap_max", float("nan"))
        reasons = []
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
                     "struct_fail": "; ".join(reasons),
        })

    out_df = pd.DataFrame(rows)
    survivors = out_df[out_df["passed_struct_filter"]].copy()
    rejects = out_df[~out_df["passed_struct_filter"]].copy()

    survivors.to_csv(out_dir / "survivors_struct.tsv", sep="\t", index=False)
    rejects.to_csv(out_dir / "rejects_struct.tsv", sep="\t", index=False)
    pd.DataFrame(hbond_rows).to_csv(
        out_dir / "hbond_details.tsv", sep="\t", index=False,
    )
    LOGGER.info(
        "stage_struct_filter: %d / %d passed (h-bond to cat-HIS + sap_max<=%.0f)",
        len(survivors), len(out_df), sap_max_threshold,
    )
    return out_dir / "survivors_struct.tsv"


# ----------------------------------------------------------------------
# Stage 5: fpocket-based ranking
# ----------------------------------------------------------------------


def _run_fpocket(pdb_path: Path, work_dir: Path) -> Optional[dict]:
    """Run fpocket on a single PDB; return the largest-pocket dict or None.

    fpocket 4.0 has a known buffer overflow in
    ``do_eraseall_in_dir.c`` when the input filename exceeds ~64 chars.
    Our designs have ~67-char stems, so we copy to a short local
    filename (``design.pdb``) inside work_dir before invoking fpocket.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    # Use a SHORT filename to avoid the upstream buffer overflow.
    local = work_dir / "design.pdb"
    local.write_bytes(pdb_path.read_bytes())
    try:
        proc = subprocess.run(
            [str(FPOCKET_BIN), "-f", str(local)],
            cwd=str(work_dir),
            capture_output=True, text=True, timeout=300,
            check=False,
        )
        info_txt = work_dir / "design_out" / "design_info.txt"
        if proc.returncode != 0 or not info_txt.is_file():
            LOGGER.warning(
                "fpocket failed for %s: rc=%d info_exists=%s\n  stderr: %s",
                pdb_path.name, proc.returncode, info_txt.is_file(),
                proc.stderr[-300:],
            )
            return None
        return _parse_fpocket_largest(info_txt)
    except subprocess.TimeoutExpired:
        LOGGER.warning("fpocket timeout (>300s) for %s", pdb_path.name)
        return None


def _parse_fpocket_largest(info_txt: Path) -> Optional[dict]:
    """Parse <stem>_info.txt; return the pocket with largest volume."""
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
    # Return the pocket with the largest druggability_score.
    pockets.sort(key=lambda p: p.get("druggability_score", 0), reverse=True)
    return pockets[0]


def stage_fpocket_rank(
    survivors_struct_tsv: Path,
    pdb_map: dict[str, Path],
    out_dir: Path,
) -> Path:
    """Run fpocket on every survivor; rank by mean alpha-sphere radius.

    Smaller mean alpha-sphere radius = tighter pocket = better
    "bottleneck" proxy for substrate channel.
    """
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(survivors_struct_tsv, sep="\t")
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
        # Lower radius = tighter pocket.
        "fpocket__mean_alpha_sphere_radius", ascending=True,
        na_position="last",
    )
    ranked.to_csv(out_dir / "ranked.tsv", sep="\t", index=False)
    LOGGER.info(
        "stage_fpocket_rank: top design has mean_alpha_sphere_radius=%.3f",
        ranked["fpocket__mean_alpha_sphere_radius"].iloc[0]
        if len(ranked) else float("nan"),
    )
    return out_dir / "ranked.tsv"


# ----------------------------------------------------------------------
# Stage 6: top-K diversity selection
# ----------------------------------------------------------------------


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def stage_top50(
    ranked_tsv: Path,
    pdb_map: dict[str, Path],
    out_dir: Path,
    target_k: int = TARGET_N_TOPK,
    min_hamming: int = 2,
) -> Path:
    """Greedy top-K with hamming-distance diversity over mutable positions."""
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ranked_tsv, sep="\t")
    if len(df) == 0:
        LOGGER.warning("stage_top50: zero survivors; nothing to write")
        return out_dir / "top50.tsv"

    selected_idx: list[int] = []
    selected_seqs: list[str] = []
    for i, row in df.iterrows():
        seq = row["sequence"]
        if all(_hamming(seq, s) >= min_hamming for s in selected_seqs):
            selected_idx.append(i)
            selected_seqs.append(seq)
        if len(selected_idx) >= target_k:
            break

    top = df.loc[selected_idx].copy()
    top.to_csv(out_dir / "top50.tsv", sep="\t", index=False)
    # FASTA
    fasta = out_dir / "top50.fasta"
    with open(fasta, "w") as fh:
        for _, row in top.iterrows():
            fh.write(f">{row['id']}\n{row['sequence']}\n")
    # PDBs
    pdb_out = out_dir / "top50_pdbs"
    pdb_out.mkdir(exist_ok=True)
    for _, row in top.iterrows():
        src = pdb_map.get(row["id"])
        if src and src.is_file():
            shutil.copy2(src, pdb_out / src.name)

    LOGGER.info(
        "stage_top50: selected %d / %d (target=%d)",
        len(top), len(df), target_k,
    )
    return out_dir / "top50.tsv"


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_pdb", type=Path, default=DEFAULT_INPUT_PDB)
    p.add_argument("--lig_params", type=Path, default=DEFAULT_LIG_PARAMS)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--n_samples", type=int, default=N_SAMPLES)
    p.add_argument("--target_k", type=int, default=TARGET_N_TOPK)
    p.add_argument("--net_charge_max", type=float, default=NET_CHARGE_MAX)
    p.add_argument("--skip_sample", action="store_true",
                   help="Skip the sampling stage (assumes 01_sample exists)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / f"iterative_design_PTE_i1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("=== run dir: %s ===", run_dir)

    sample_dir = run_dir / "01_sample"
    seq_filter_dir = run_dir / "02_seq_filter"
    struct_filter_dir = run_dir / "03_struct_filter"
    fpocket_dir = run_dir / "04_fpocket_rank"
    top50_dir = run_dir / "05_top50"
    restored_pdb_dir = run_dir / "01_sample" / "pdbs_restored"

    # Stage 1
    if not args.skip_sample:
        stage_sample(
            input_pdb=args.input_pdb,
            fixed_resnos=DEFAULT_CATRES,
            out_dir=sample_dir,
            n_samples=args.n_samples,
            chain=CHAIN,
            sequence_id_prefix="PTE_i1",
        )

    # Determine WT length from the input PDB
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from protein_chisel.io.pdb import extract_sequence
    wt_seq = extract_sequence(args.input_pdb, chain=CHAIN)
    LOGGER.info("WT sequence length: %d", len(wt_seq))

    # Stage 2: restore REMARK 666 + HETNAM + LINK lines on every MPNN PDB
    import pandas as pd
    cand_df = pd.read_csv(sample_dir / "candidates.tsv", sep="\t")
    pdb_basename = args.input_pdb.stem
    pdb_map = stage_restore_pdbs(
        sample_dir=sample_dir,
        ref_pdb=args.input_pdb,
        out_pdb_dir=restored_pdb_dir,
        pdb_basename=pdb_basename,
        candidate_ids=cand_df["id"].tolist(),
    )

    # Stage 3: cheap sequence filter
    survivors_seq = stage_seq_filter(
        candidates_tsv=sample_dir / "candidates.tsv",
        out_dir=seq_filter_dir,
        net_charge_max=args.net_charge_max,
        wt_length=len(wt_seq),
    )

    # Stage 4: structure h-bond filter
    survivors_struct = stage_struct_filter(
        survivors_seq_tsv=survivors_seq,
        pdb_map=pdb_map,
        out_dir=struct_filter_dir,
    )

    # Stage 5: fpocket ranking
    ranked = stage_fpocket_rank(
        survivors_struct_tsv=survivors_struct,
        pdb_map=pdb_map,
        out_dir=fpocket_dir,
    )

    # Stage 6: top 50
    stage_top50(
        ranked_tsv=ranked,
        pdb_map=pdb_map,
        out_dir=top50_dir,
        target_k=args.target_k,
    )

    # Manifest
    manifest = {
        "pipeline": "iterative_design_PTE_i1",
        "input_pdb": str(args.input_pdb),
        "lig_params": str(args.lig_params),
        "fixed_resnos": list(DEFAULT_CATRES),
        "catalytic_his_resnos": list(CATALYTIC_HIS_RESNOS),
        "n_samples_requested": args.n_samples,
        "target_k": args.target_k,
        "net_charge_max": args.net_charge_max,
        "wt_length": len(wt_seq),
        "outputs": {
            "sample_dir": str(sample_dir),
            "seq_filter_dir": str(seq_filter_dir),
            "struct_filter_dir": str(struct_filter_dir),
            "fpocket_dir": str(fpocket_dir),
            "top50_dir": str(top50_dir),
            "top50_fasta": str(top50_dir / "top50.fasta"),
            "top50_pdbs": str(top50_dir / "top50_pdbs"),
        },
        "started_at": timestamp,
    }
    with open(run_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("=== DONE -- top50 at %s ===", top50_dir)


if __name__ == "__main__":
    main()
