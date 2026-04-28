"""LigandMPNN wrapper — ligand-aware sequence sampler with PLM bias.

Wraps the lab's ``/net/software/lab/mpnn/proteinmpnn/ligandMPNN/protein_mpnn_run.py``
behind a clean Python API:

    sample_with_ligand_mpnn(
        pdb_path, ligand_params, fixed_resnos=[...],
        bias_per_residue=fusion.bias,  # (L, 20) from sampling/plm_fusion
        n_samples=200, sampling_temp=0.1,
    ) -> CandidateSet

Internally:
- Writes a temp ``bias_by_res.jsonl`` ({pdb_name: {chain: (L, 21) array}})
  with the AA columns matching MPNN's ``ACDEFGHIKLMNPQRSTVWYX`` alphabet.
  Our plm_fusion gives 20 cols (no X); we pad column 21 with 0.
- Writes a temp ``fixed_positions.jsonl`` for active-site residues (so
  MPNN keeps their wild-type identity).
- Calls ``protein_mpnn_run.py`` inside ``mlfold.sif`` via apptainer.
- Parses the output FASTA into a ``CandidateSet`` with sampler metadata.

Run inside mlfold.sif (or any sif that has the LigandMPNN deps); the
wrapper invokes it via apptainer if `via_apptainer=True` (default).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from protein_chisel.io.schemas import CandidateSet
from protein_chisel.paths import (
    LIGAND_MPNN_DIR, LIGAND_MPNN_RUN, LIGAND_MPNN_WEIGHTS, MLFOLD_SIF,
)
from protein_chisel.sampling.plm_fusion import AA_ORDER as PLM_AA_ORDER


LOGGER = logging.getLogger("protein_chisel.ligand_mpnn")


# LigandMPNN's vocabulary order; X (unknown) is the 21st column.
MPNN_AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
MPNN_AA_TO_IDX = {aa: i for i, aa in enumerate(MPNN_AA_ORDER)}


def _bias_to_mpnn_format(bias_20: np.ndarray) -> np.ndarray:
    """Convert a (L, 20) bias matrix in PLM_AA_ORDER to (L, 21) in MPNN_AA_ORDER.

    plm_fusion.AA_ORDER = "ACDEFGHIKLMNPQRSTVWY" (same as MPNN minus X), so
    we just pad column 20 (X) with zeros.
    """
    if PLM_AA_ORDER != "ACDEFGHIKLMNPQRSTVWY":
        raise RuntimeError(
            f"PLM_AA_ORDER changed unexpectedly: {PLM_AA_ORDER!r}; please update "
            "_bias_to_mpnn_format to align with MPNN_AA_ORDER"
        )
    if bias_20.shape[-1] != 20:
        raise ValueError(f"expected (L, 20) bias matrix, got {bias_20.shape}")
    L = bias_20.shape[0]
    out = np.zeros((L, 21), dtype=np.float64)
    out[:, :20] = bias_20
    return out


@dataclass
class LigandMPNNConfig:
    model_name: str = "v_32_020"        # backbone-noise variant (most-used)
    sampling_temp: float = 0.1
    batch_size: int = 1
    backbone_noise: float = 0.0
    pack_side_chains: int = 0
    use_ligand: int = 1
    use_DNA_RNA: int = 0
    seed: int = 0  # 0 = random
    weights_dir: Optional[Path] = None  # defaults to LIGAND_MPNN_WEIGHTS
    extra_flags: tuple[str, ...] = ()


@dataclass
class LigandMPNNResult:
    candidate_set: CandidateSet
    out_dir: Path
    raw_fasta: Path
    config: LigandMPNNConfig


def _build_bias_by_res_jsonl(
    pdb_name: str, chain: str, bias_20: np.ndarray
) -> str:
    """Serialize a (L, 20) bias matrix into LigandMPNN's bias_by_res JSONL."""
    bias_21 = _bias_to_mpnn_format(bias_20)
    payload = {pdb_name: {chain: bias_21.tolist()}}
    return json.dumps(payload)


def _build_fixed_positions_jsonl(
    pdb_name: str, chain: str, fixed_resnos: Iterable[int]
) -> str:
    """LigandMPNN's fixed_positions JSONL: dict[pdb][chain] = [resnos]."""
    payload = {pdb_name: {chain: sorted(set(int(r) for r in fixed_resnos))}}
    return json.dumps(payload)


def _parse_output_fasta(fasta_path: Path) -> list[tuple[str, str, dict]]:
    """Parse LigandMPNN's output FASTA.

    Each design has a header like:
        ``>T=0.1, sample=1, score=1.234, seq_recovery=0.85``
    followed by the sequence (one chain per line, separated by '/').

    Returns a list of (header, sequence, metadata) tuples.
    """
    out: list[tuple[str, str, dict]] = []
    if not fasta_path.exists():
        return out
    text = fasta_path.read_text()
    blocks = [b for b in text.split(">") if b.strip()]
    for b in blocks:
        lines = b.strip().splitlines()
        header = lines[0]
        seq = "".join(lines[1:]).strip()
        meta = _parse_header(header)
        out.append((header, seq, meta))
    return out


_RE_KV = re.compile(r"([A-Za-z_]+)\s*=\s*([+-]?[\d.eE+-]+)")


def _parse_header(header: str) -> dict:
    """Extract numeric fields from a LigandMPNN FASTA header."""
    out: dict[str, float] = {}
    for m in _RE_KV.finditer(header):
        key = m.group(1).strip().lower()
        try:
            out[key] = float(m.group(2))
        except ValueError:
            pass
    return out


def sample_with_ligand_mpnn(
    pdb_path: str | Path,
    ligand_params: str | Path,
    chain: str = "A",
    fixed_resnos: Iterable[int] = (),
    bias_per_residue: Optional[np.ndarray] = None,
    n_samples: int = 100,
    config: Optional[LigandMPNNConfig] = None,
    out_dir: Optional[str | Path] = None,
    parent_design_id: Optional[str] = None,
    via_apptainer: bool = True,
) -> LigandMPNNResult:
    """Run LigandMPNN and return parsed designed sequences.

    Args:
        pdb_path: input PDB.
        ligand_params: ligand .params file.
        chain: protein chain to design.
        fixed_resnos: 1-indexed residue numbers to keep fixed (typically
            the active-site residues from REMARK 666).
        bias_per_residue: optional (L, 20) bias matrix from sampling/plm_fusion.
            Each row is added to MPNN's per-position logits. Use this to
            inject ESM-C / SaProt naturalness signal.
        n_samples: total number of sequences to sample.
        config: LigandMPNNConfig.
        out_dir: where to write outputs. Defaults to a tempdir.
        parent_design_id: id stored on every CandidateSet row.
        via_apptainer: run inside mlfold.sif. Set False if you've
            entered the right env yourself.
    """
    cfg = config or LigandMPNNConfig()
    pdb = Path(pdb_path).resolve()
    pdb_name = pdb.stem
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="chisel_lmpnn_"))
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(cfg.weights_dir) if cfg.weights_dir else LIGAND_MPNN_WEIGHTS

    # Bias JSONL
    bias_path = None
    if bias_per_residue is not None:
        bias_path = out_dir / "bias_by_res.jsonl"
        bias_path.write_text(
            _build_bias_by_res_jsonl(pdb_name, chain, bias_per_residue) + "\n"
        )

    # Fixed positions JSONL
    fixed_path = None
    if fixed_resnos:
        fixed_path = out_dir / "fixed_positions.jsonl"
        fixed_path.write_text(
            _build_fixed_positions_jsonl(pdb_name, chain, fixed_resnos) + "\n"
        )

    # Build the LigandMPNN command
    cmd = [
        "python", str(LIGAND_MPNN_RUN),
        "--pdb_path", str(pdb),
        "--ligand_params_path", str(Path(ligand_params).resolve()),
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(int(n_samples)),
        "--sampling_temp", str(float(cfg.sampling_temp)),
        "--batch_size", str(int(cfg.batch_size)),
        "--model_name", cfg.model_name,
        "--path_to_model_weights", str(weights),
        "--use_ligand", str(int(cfg.use_ligand)),
        "--use_DNA_RNA", str(int(cfg.use_DNA_RNA)),
        "--pack_side_chains", str(int(cfg.pack_side_chains)),
        "--backbone_noise", str(float(cfg.backbone_noise)),
        "--seed", str(int(cfg.seed)),
        "--pdb_path_chains", chain,
    ]
    if bias_path:
        cmd += ["--bias_by_res_jsonl", str(bias_path)]
    if fixed_path:
        cmd += ["--fixed_positions_jsonl", str(fixed_path)]
    cmd.extend(cfg.extra_flags)

    LOGGER.info("running LigandMPNN: %s", " ".join(cmd))

    if via_apptainer:
        from protein_chisel.utils.apptainer import mlfold_call

        result = mlfold_call(nv=True).run(cmd, check=True, timeout=3600)
    else:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        from protein_chisel.utils.apptainer import ApptainerResult
        result = ApptainerResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            command=cmd,
        )

    fasta = out_dir / "seqs" / f"{pdb_name}.fa"
    parsed = _parse_output_fasta(fasta)
    if not parsed:
        raise RuntimeError(
            f"LigandMPNN produced no sequences. Output dir: {out_dir}\n"
            f"stderr tail: {result.stderr[-1000:]}"
        )

    # First entry of LigandMPNN output is usually the input ("score=...,
    # input_pdb_path=...") — keep it but flag separately.
    rows: list[dict] = []
    for i, (header, seq, meta) in enumerate(parsed):
        rows.append({
            "id": f"{pdb_name}_lmpnn_{i:03d}",
            "sequence": seq.replace("/", ""),
            "parent_design_id": parent_design_id or pdb_name,
            "sampler": "ligand_mpnn",
            "sampler_params_hash": _config_hash(cfg, n_samples, fixed_resnos, bool(bias_path)),
            "is_input": (i == 0),
            "header": header,
            **{f"mpnn_{k}": v for k, v in meta.items()},
        })
    cs = CandidateSet(df=pd.DataFrame(rows))

    return LigandMPNNResult(
        candidate_set=cs,
        out_dir=out_dir,
        raw_fasta=fasta,
        config=cfg,
    )


def _config_hash(cfg: LigandMPNNConfig, n: int, fixed: Iterable[int], biased: bool) -> str:
    import hashlib
    payload = {
        "model_name": cfg.model_name,
        "sampling_temp": cfg.sampling_temp,
        "n_samples": n,
        "fixed": sorted(set(int(r) for r in fixed)),
        "biased": biased,
        "backbone_noise": cfg.backbone_noise,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


__all__ = [
    "LigandMPNNConfig",
    "LigandMPNNResult",
    "MPNN_AA_ORDER",
    "sample_with_ligand_mpnn",
]
