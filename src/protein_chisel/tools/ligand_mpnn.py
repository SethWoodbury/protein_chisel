"""LigandMPNN wrapper — uses the fused_mpnn lab build (seth_temp/run.py).

This wraps ``/net/software/lab/fused_mpnn/seth_temp/run.py`` which is the
modern lab default. Compared to the older ``protein_mpnn_run.py`` in
``proteinmpnn/ligandMPNN``, this version:

- Honors ``--repack_everything 0`` correctly (does NOT repack residues
  passed in ``--fixed_residues``).
- Accepts ``--fixed_residues_multi`` JSON of ``{pdb_path: [chain+resno
  strings]}`` (e.g. ``["A92", "A136"]``).
- Accepts a different ``--bias_AA_per_residue`` format: a JSON mapping of
  ``{"<chain><resno>": {AA: bias_in_nats}}`` keyed by residue, NOT the
  array-shaped per-position dict used by the older runner.
- Supports ``--enhance plddt_residpo_alpha_*`` for plddt-enhanced checkpoints.
- Supports ``--ligand_mpnn_use_side_chain_context 1`` so the model sees
  the catalytic side-chain rotamers (important — without this the model
  often samples residues that clash with the catalytic ones).

Runs inside ``universal.sif``. The wrapper writes input JSONs, calls
``run.py`` via apptainer exec, and parses the output FASTA into a
``CandidateSet`` with sampler metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from protein_chisel.io.schemas import CandidateSet
from protein_chisel.paths import FUSED_MPNN_RUN, UNIVERSAL_SIF
from protein_chisel.sampling.plm_fusion import AA_ORDER as PLM_AA_ORDER


LOGGER = logging.getLogger("protein_chisel.ligand_mpnn")


# fused_mpnn's internal AA alphabet maps. The 20 canonical AAs are the
# columns we care about for our (L, 20) bias matrices; X (UNK) is index
# 20 and we always set its bias to 0.
MPNN_AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
MPNN_AA_TO_IDX = {aa: i for i, aa in enumerate(MPNN_AA_ORDER)}
assert PLM_AA_ORDER == "ACDEFGHIKLMNPQRSTVWY", "alignment with plm_fusion changed"


@dataclass
class LigandMPNNConfig:
    model_type: str = "ligand_mpnn"        # ligand_mpnn / protein_mpnn / soluble_mpnn
    temperature: float = 0.1
    batch_size: int = 1
    number_of_batches: int = 10            # total = batch_size * number_of_batches
    pack_side_chains: int = 1
    sc_num_denoising_steps: int = 3
    repack_everything: int = 0             # critical: 0 keeps fixed_residues frozen
    ligand_mpnn_use_side_chain_context: int = 1
    ligand_mpnn_use_atom_context: int = 1
    omit_AA: str = "CX"                    # don't sample C or X by default
    bias_AA: str = ""                      # global e.g. "K:-0.5,R:-0.75,E:0.75,D:0.75"
    enhance: Optional[str] = None          # e.g. "plddt_residpo_alpha_20250116-aec4d0c4"
    seed: int = 0                          # 0 = random
    file_ending: str = ""
    extra_flags: tuple[str, ...] = ()


@dataclass
class LigandMPNNResult:
    candidate_set: CandidateSet
    out_dir: Path
    raw_fasta: Path
    config: LigandMPNNConfig


# ---------------------------------------------------------------------------
# Helpers — build the JSON inputs the runner expects
# ---------------------------------------------------------------------------


def _residue_label(chain: str, resno: int) -> str:
    """fused_mpnn residue label format: ``A92`` (chain + resno, no separator)."""
    return f"{chain}{int(resno)}"


def _build_fixed_residues_multi(
    pdb_path: str | Path, fixed_resnos: Iterable[int], chain: str
) -> dict[str, list[str]]:
    return {str(Path(pdb_path).resolve()): [_residue_label(chain, r) for r in sorted(set(fixed_resnos))]}


def _build_bias_per_residue_multi(
    pdb_path: str | Path,
    bias_per_residue: np.ndarray,
    chain: str,
    protein_resnos: Sequence[int],
) -> dict[str, dict[str, dict[str, float]]]:
    """Convert a (L, 20) bias matrix to the fused_mpnn JSON layout.

    ``bias_per_residue[i, j]`` = bias for AA ``PLM_AA_ORDER[j]`` at the i-th
    protein residue. ``protein_resnos[i]`` is the pose-resno of that
    residue.

    Returns ``{pdb_path: {<chain><resno>: {AA: bias}}}``.
    """
    if bias_per_residue.shape[0] != len(protein_resnos):
        raise ValueError(
            f"bias rows ({bias_per_residue.shape[0]}) != protein_resnos "
            f"({len(protein_resnos)})"
        )
    if bias_per_residue.shape[1] != 20:
        raise ValueError(f"bias must have 20 AA columns, got {bias_per_residue.shape[1]}")
    out: dict[str, dict[str, float]] = {}
    for i, resno in enumerate(protein_resnos):
        label = _residue_label(chain, resno)
        per_aa = {
            aa: float(bias_per_residue[i, j]) for j, aa in enumerate(PLM_AA_ORDER)
        }
        # Drop trivial-zero rows to keep the JSON small.
        if all(abs(v) < 1e-12 for v in per_aa.values()):
            continue
        out[label] = per_aa
    return {str(Path(pdb_path).resolve()): out}


# ---------------------------------------------------------------------------
# Output FASTA parsing
# ---------------------------------------------------------------------------


_RE_KV = re.compile(r"([A-Za-z_]+)\s*=\s*([+-]?[\d.eE+-]+)")


def _parse_header(header: str) -> dict:
    """Numeric fields from a fused_mpnn header (T=, seed=, seq_rec= ...)."""
    out: dict[str, float] = {}
    for m in _RE_KV.finditer(header):
        key = m.group(1).strip().lower()
        try:
            out[key] = float(m.group(2))
        except ValueError:
            pass
    return out


def _parse_output_fasta(fasta_path: Path) -> list[tuple[str, str, dict]]:
    """Parse fused_mpnn's output FASTA (one entry per design + an input header)."""
    if not fasta_path.exists():
        return []
    text = fasta_path.read_text()
    blocks = [b for b in text.split(">") if b.strip()]
    out: list[tuple[str, str, dict]] = []
    for b in blocks:
        lines = b.strip().splitlines()
        header = lines[0]
        seq = "".join(lines[1:]).strip()
        out.append((header, seq, _parse_header(header)))
    return out


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def sample_with_ligand_mpnn(
    pdb_path: str | Path,
    chain: str = "A",
    fixed_resnos: Iterable[int] = (),
    bias_per_residue: Optional[np.ndarray] = None,
    protein_resnos: Optional[Sequence[int]] = None,
    n_samples: int = 100,
    config: Optional[LigandMPNNConfig] = None,
    out_dir: Optional[str | Path] = None,
    parent_design_id: Optional[str] = None,
    via_apptainer: bool = True,
    ligand_params: Optional[str | Path] = None,  # accepted for API parity; not used
) -> LigandMPNNResult:
    """Sample sequences with LigandMPNN (fused_mpnn build).

    Args:
        pdb_path: input PDB. Ligand atoms must be HETATM in the same file.
        chain: protein chain to design.
        fixed_resnos: 1-indexed pose residue numbers to keep fixed (catalytic
            residues from REMARK 666). When ``repack_everything=0`` (the
            default), fused_mpnn correctly leaves these residues' identities
            and rotamers untouched.
        bias_per_residue: (L, 20) bias matrix in PLM_AA_ORDER. If provided,
            ``protein_resnos`` must be passed too so we know which pose
            resno each row corresponds to.
        protein_resnos: pose residue numbers (1-indexed) for the protein
            residues *in row order* of bias_per_residue.
        n_samples: total sequences = ``batch_size * number_of_batches``;
            ``n_samples`` adjusts ``number_of_batches`` for convenience.
        config: LigandMPNNConfig.
        ligand_params: ignored (the modern runner reads ligand atoms from
            HETATMs); kept in the signature so the older callsite still works.
    """
    cfg = config or LigandMPNNConfig()
    pdb = Path(pdb_path).resolve()
    pdb_name = pdb.stem
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="chisel_lmpnn_"))
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # n_samples → number_of_batches given the configured batch_size
    n_batches = max(1, int(np.ceil(n_samples / max(1, cfg.batch_size))))

    # Build JSON inputs
    fixed_path = None
    if fixed_resnos:
        fixed_path = out_dir / "fixed_residues.json"
        fixed_path.write_text(json.dumps(
            _build_fixed_residues_multi(pdb, fixed_resnos, chain), indent=2,
        ))

    bias_path = None
    if bias_per_residue is not None:
        if protein_resnos is None:
            raise ValueError(
                "bias_per_residue requires protein_resnos so we can label rows "
                "with the correct pose residue numbers"
            )
        bias_path = out_dir / "bias_per_residue.json"
        bias_path.write_text(json.dumps(
            _build_bias_per_residue_multi(pdb, bias_per_residue, chain, protein_resnos),
            indent=2,
        ))

    # Build CLI
    cmd = [
        "python", str(FUSED_MPNN_RUN),
        "--model_type", cfg.model_type,
        "--pdb_path", str(pdb),
        "--out_folder", str(out_dir),
        "--temperature", str(float(cfg.temperature)),
        "--batch_size", str(int(cfg.batch_size)),
        "--number_of_batches", str(n_batches),
        "--pack_side_chains", str(int(cfg.pack_side_chains)),
        "--sc_num_denoising_steps", str(int(cfg.sc_num_denoising_steps)),
        "--repack_everything", str(int(cfg.repack_everything)),
        "--ligand_mpnn_use_side_chain_context", str(int(cfg.ligand_mpnn_use_side_chain_context)),
        "--ligand_mpnn_use_atom_context", str(int(cfg.ligand_mpnn_use_atom_context)),
        "--omit_AA", cfg.omit_AA,
        "--seed", str(int(cfg.seed)),
        "--packed_suffix", "_packed",
        "--file_ending", cfg.file_ending,
    ]
    if cfg.bias_AA:
        cmd += ["--bias_AA", cfg.bias_AA]
    if cfg.enhance:
        cmd += ["--enhance", cfg.enhance]
    if fixed_path:
        cmd += ["--fixed_residues_multi", str(fixed_path)]
    if bias_path:
        cmd += ["--bias_AA_per_residue_multi", str(bias_path)]
    cmd.extend(cfg.extra_flags)

    LOGGER.info("running fused_mpnn: %s", " ".join(cmd))

    # Set torch / OpenMP / MKL thread counts in the subprocess env so
    # fused_mpnn's PyTorch matches the slurm cpus_per_task allocation
    # instead of defaulting to all node cores. Big win on CPU runs
    # (codex round-2: ~150-300s saved on a 21-min CPU pipeline).
    # Inherits from os.environ so caller can override.
    import os as _os
    sub_env = dict(_os.environ)
    try:
        from protein_chisel.utils.resources import detect_n_cpus
        cpus, _ = detect_n_cpus()
    except Exception:
        cpus = int(_os.environ.get("SLURM_CPUS_PER_TASK", _os.cpu_count() or 1))
    sub_env.setdefault("OMP_NUM_THREADS", str(cpus))
    sub_env.setdefault("MKL_NUM_THREADS", str(cpus))
    sub_env.setdefault("OPENBLAS_NUM_THREADS", str(cpus))

    if via_apptainer:
        from protein_chisel.utils.apptainer import universal_call

        result = universal_call(nv=True).run(cmd, check=True, timeout=7200)
    else:
        proc = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=sub_env,
        )
        from protein_chisel.utils.apptainer import ApptainerResult
        result = ApptainerResult(
            returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, command=cmd,
        )

    # fused_mpnn writes seqs/<name>.fa<file_ending>
    fasta = out_dir / "seqs" / f"{pdb_name}.fa{cfg.file_ending}"
    parsed = _parse_output_fasta(fasta)
    if not parsed:
        raise RuntimeError(
            f"fused_mpnn produced no sequences. Expected {fasta}\n"
            f"stdout tail: {result.stdout[-1000:]}\n"
            f"stderr tail: {result.stderr[-1000:]}"
        )

    # First entry is the input ("input_pdb_path=...") header; others are samples.
    rows: list[dict] = []
    for i, (header, seq, meta) in enumerate(parsed):
        rows.append({
            "id": f"{pdb_name}_lmpnn_{i:03d}",
            "sequence": seq.replace("/", ""),
            "parent_design_id": parent_design_id or pdb_name,
            "sampler": "fused_mpnn",
            "sampler_params_hash": _config_hash(cfg, n_samples, fixed_resnos, bool(bias_path)),
            "is_input": (i == 0),
            "header": header,
            **{f"mpnn_{k}": v for k, v in meta.items()},
        })

    return LigandMPNNResult(
        candidate_set=CandidateSet(df=pd.DataFrame(rows)),
        out_dir=out_dir,
        raw_fasta=fasta,
        config=cfg,
    )


def _config_hash(cfg: LigandMPNNConfig, n: int, fixed: Iterable[int], biased: bool) -> str:
    payload = {
        "model_type": cfg.model_type,
        "temperature": cfg.temperature,
        "n_samples": n,
        "fixed": sorted(set(int(r) for r in fixed)),
        "biased": biased,
        "pack_side_chains": cfg.pack_side_chains,
        "repack_everything": cfg.repack_everything,
        "side_chain_context": cfg.ligand_mpnn_use_side_chain_context,
        "enhance": cfg.enhance,
        "omit_AA": cfg.omit_AA,
        "bias_AA": cfg.bias_AA,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


__all__ = [
    "LigandMPNNConfig",
    "LigandMPNNResult",
    "MPNN_AA_ORDER",
    "sample_with_ligand_mpnn",
]
