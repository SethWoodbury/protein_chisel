"""MPNN sampling backends — the default in-driver LigandMPNN bias backend vs the
optional decode-time Product-of-Experts (PoE) backend.

The pipeline's default sampler (``--mpnn_backend bias``) runs LigandMPNN *in-process*
inside the stage-3 container with our calibrated static fusion bias — unchanged and
byte-identical. This module adds the optional ``poe`` backend, which mixes
context-aware experts (HERMES / ESM / E1 / VESM / MSA / DMS) into the per-position
log-probs at decode time on top of our same calibrated bias.

Because nested ``apptainer exec`` is blocked inside the stage-3 container, the PoE
sampler cannot run in-process; it runs as a **separate host stage**
(``apptainer exec poe_mpnn.sif python run.py``), and its sampled candidates feed the
driver's score/rank (one-shot, not per-cycle). This module provides the pure,
host-testable pieces: expert-lambda validation, the ``run.py`` command construction,
and loading the PoE output FASTA into candidate records. Orchestration (writing the
bias/omit/fixed JSONs, launching the host stage, the driver's score-only mode) lives
in ``run_chisel_design.sh`` + ``iterative_design.py``.

PoE design + the expert/lambda interface are borrowed from Sebastian (sebols) and
Joe Mi's ``fused_mpnn_poe`` (``log P_final = λ_mpnn·log P_mpnn + Σ λ_i·log P_i``,
``λ_mpnn = 1 - Σ λ_i``). HERMES experts are by Visani et al. (Nourmohammad lab, UW).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from protein_chisel import paths

# Experts supported by fused_mpnn_poe run.py (README + experts/ package).
SUPPORTED_POE_EXPERTS = (
    "hermes", "dms", "msa", "esm", "wt_esm", "vesm", "wt_vesm", "e1", "wt_e1",
)
# Default LigandMPNN checkpoint baked into run.py (kept here so the command is
# explicit + auditable; override to match the main pipeline's checkpoint).
DEFAULT_POE_LIGAND_CHECKPOINT = "/net/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt"

BACKEND_BIAS = "bias"
BACKEND_POE = "poe"
MPNN_BACKENDS = (BACKEND_BIAS, BACKEND_POE)


def _parse_csv(value: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(value, str):
        value = value.split(",")
    return [v.strip() for v in value if str(v).strip()]


def validate_expert_lambdas(
    experts: Union[str, Sequence[str]],
    lambdas: Union[str, Sequence[float]],
) -> Tuple[List[str], List[float]]:
    """Parse + validate the PoE expert list and their mixing weights.

    Enforces the ``fused_mpnn_poe`` contract: one λ per expert, each λ in (0, 1),
    and ``Σ λ_i < 1`` (so ``λ_mpnn = 1 - Σ λ_i > 0``). Returns ``(experts, lambdas)``
    as parallel lists. Raises ``ValueError`` with a clear message on any violation.
    """
    exp = _parse_csv(experts)
    if isinstance(lambdas, str):
        lam_raw = _parse_csv(lambdas)
    else:
        lam_raw = [str(x) for x in lambdas]
    if not exp:
        raise ValueError("PoE backend: --additional_experts is empty")
    unknown = [e for e in exp if e not in SUPPORTED_POE_EXPERTS]
    if unknown:
        raise ValueError(
            f"PoE backend: unknown expert(s) {unknown}; supported: "
            f"{list(SUPPORTED_POE_EXPERTS)}")
    if len(lam_raw) != len(exp):
        raise ValueError(
            f"PoE backend: {len(exp)} experts but {len(lam_raw)} lambdas "
            f"(experts={exp}, lambdas={lam_raw}) — counts must match")
    try:
        lam = [float(x) for x in lam_raw]
    except ValueError as e:
        raise ValueError(f"PoE backend: non-numeric expert lambda in {lam_raw}: {e}")
    if any(x <= 0.0 or x >= 1.0 for x in lam):
        raise ValueError(f"PoE backend: each expert lambda must be in (0, 1); got {lam}")
    total = sum(lam)
    if total >= 1.0:
        raise ValueError(
            f"PoE backend: expert lambdas sum to {total:.4f} (must be < 1.0 so "
            f"lambda_mpnn = 1 - sum > 0); got {dict(zip(exp, lam))}")
    return exp, lam


def build_poe_command(
    *,
    pdb_path: Union[str, Path],
    out_folder: Union[str, Path],
    experts: Sequence[str],
    lambdas: Sequence[float],
    bias_json: Optional[Union[str, Path]] = None,
    omit_json: Optional[Union[str, Path]] = None,
    fixed_json: Optional[Union[str, Path]] = None,
    checkpoint: str = DEFAULT_POE_LIGAND_CHECKPOINT,
    batch_size: int = 1,
    number_of_batches: int = 10,
    temperature: float = 0.1,
    seed: int = 0,
    use_atom_context: int = 1,
    use_side_chain_context: int = 0,
    hermes_probs: Optional[Union[str, Path]] = None,
    file_ending: str = "",
    sif: Union[str, Path] = paths.POE_MPNN_SIF,
    run_script: Union[str, Path] = paths.POE_MPNN_RUN,
    python_bin: str = "python",
) -> List[str]:
    """Build the host ``apptainer exec poe_mpnn.sif python run.py …`` argv.

    Pure: constructs + returns the command list (no execution), so it is unit-
    testable. Targets ``--model_type ligand_mpnn`` so the PoE sampler is ligand-
    aware like our pipeline, and passes our calibrated bias straight through via
    ``--bias_AA_per_residue_multi`` (same JSON our LigandMPNN wrapper emits).
    ``experts``/``lambdas`` should already be validated by
    :func:`validate_expert_lambdas`.
    """
    if len(experts) != len(lambdas):
        raise ValueError("build_poe_command: experts/lambdas length mismatch")
    cmd: List[str] = [
        "apptainer", "exec", "--nv", str(sif),
        python_bin, str(run_script),
        "--model_type", "ligand_mpnn",
        "--checkpoint_ligand_mpnn", str(checkpoint),
        "--pdb_path", str(pdb_path),
        "--out_folder", str(out_folder),
        "--batch_size", str(int(batch_size)),
        "--number_of_batches", str(int(number_of_batches)),
        "--temperature", str(float(temperature)),
        "--seed", str(int(seed)),
        "--ligand_mpnn_use_atom_context", str(int(use_atom_context)),
        "--ligand_mpnn_use_side_chain_context", str(int(use_side_chain_context)),
    ]
    if experts:
        cmd += ["--additional_experts", ",".join(experts),
                "--additional_expert_lambdas", ",".join(str(x) for x in lambdas)]
    if bias_json is not None:
        cmd += ["--bias_AA_per_residue_multi", str(bias_json)]
    if omit_json is not None:
        cmd += ["--omit_AA_per_residue_multi", str(omit_json)]
    if fixed_json is not None:
        cmd += ["--fixed_residues_multi", str(fixed_json)]
    if hermes_probs is not None:
        cmd += ["--hermes_probs", str(hermes_probs)]
    if file_ending:
        cmd += ["--file_ending", str(file_ending)]
    return cmd


def poe_output_fasta(out_folder: Union[str, Path], pdb_stem: str,
                     file_ending: str = "") -> Path:
    """Path run.py writes the sampled sequences to: ``<out>/seqs/<stem>.fa<ending>``."""
    return Path(out_folder) / "seqs" / f"{pdb_stem}.fa{file_ending}"


def load_poe_candidates(fasta_path: Union[str, Path]) -> List[Tuple[str, str, dict]]:
    """Parse a PoE output FASTA into ``[(header, sequence, parsed_fields), …]``.

    Reuses the LigandMPNN FASTA parser — fused_mpnn_poe writes the same
    ``seqs/<name>.fa`` format (an input-header record first, then one record per
    sampled design). The driver turns these into its candidate pool for the
    score/rank stages.
    """
    from protein_chisel.tools.ligand_mpnn import _parse_output_fasta
    return _parse_output_fasta(Path(fasta_path))


__all__ = [
    "SUPPORTED_POE_EXPERTS", "DEFAULT_POE_LIGAND_CHECKPOINT",
    "BACKEND_BIAS", "BACKEND_POE", "MPNN_BACKENDS",
    "validate_expert_lambdas", "build_poe_command", "poe_output_fasta",
    "load_poe_candidates",
]
