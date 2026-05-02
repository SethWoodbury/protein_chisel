"""FlowPacker wrapper -- side-chain packing and per-residue chi log-likelihood.

FlowPacker (Lee 2025, MIT) packs side chains via torsional flow matching
on an Equiformer-V2 backbone. The repo ships a ``likelihood.py`` that
runs a Hutchinson-trace integration of the reverse-time ODE to give
**per-residue per-chi log-likelihoods** -- exactly the modern
context-aware rotamer-plausibility signal we want as a complement to
Rosetta `fa_dun` and MolProbity rotalyze.

Source vendored as a git submodule under ``external/flowpacker/``;
runtime lives inside ``esmc.sif`` at ``/opt/flowpacker``. Trained
checkpoints (~600 MB total: bc40, cluster, confidence) live at
``/net/databases/lab/flowpacker/checkpoints/``.

Two entry points:

- :func:`flowpacker_pack` runs ``sampler_pdb.py`` to repack a PDB.
- :func:`flowpacker_score` runs ``likelihood.py`` and returns per-residue
  per-chi log-likelihoods (4 chis per residue, NaN where undefined).
  This is the headline scoring use case.

Both inference and likelihood read their checkpoint path from
``config/inference/base.yaml``. We bind ``/net/databases/lab/flowpacker
/checkpoints`` over ``/opt/flowpacker/checkpoints`` at runtime.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger("protein_chisel.flowpacker_score")


@dataclass
class FlowPackerPackResult:
    """Result of a FlowPacker repack."""
    out_pdb_path: Path
    runtime_seconds: float = 0.0
    checkpoint: str = "cluster"     # which checkpoint we used


@dataclass
class FlowPackerScoreResult:
    """Result of a FlowPacker likelihood evaluation.

    Per-residue df columns: resno, name3,
    logp_chi1, logp_chi2, logp_chi3, logp_chi4 (NaN when undefined).
    Aggregates: mean / sum log p over all valid chis; mean per chi index.
    Higher logp = more plausible rotamer under the FlowPacker prior.
    """
    per_residue_df: object = None  # pandas.DataFrame
    n_residues_scored: int = 0
    n_chis_scored: int = 0           # total chi log-probs across residues
    logp_sum: float = 0.0            # sum of all valid per-residue per-chi logp
    logp_mean: float = 0.0           # mean over all valid (residue, chi)
    logp_mean_chi1: float = 0.0
    logp_mean_chi2: float = 0.0
    logp_mean_chi3: float = 0.0
    logp_mean_chi4: float = 0.0
    runtime_seconds: float = 0.0
    checkpoint: str = "cluster"

    def to_dict(self, prefix: str = "flowpacker__") -> dict[str, float | int]:
        return {
            f"{prefix}n_residues_scored": int(self.n_residues_scored),
            f"{prefix}n_chis_scored": int(self.n_chis_scored),
            f"{prefix}logp_sum": float(self.logp_sum),
            f"{prefix}logp_mean": float(self.logp_mean),
            f"{prefix}logp_mean_chi1": float(self.logp_mean_chi1),
            f"{prefix}logp_mean_chi2": float(self.logp_mean_chi2),
            f"{prefix}logp_mean_chi3": float(self.logp_mean_chi3),
            f"{prefix}logp_mean_chi4": float(self.logp_mean_chi4),
        }


def _stage_pdb_for_flowpacker(
    pdb_path: Path, workdir: Path,
) -> Path:
    """FlowPacker reads PDBs from a folder. Return a folder containing
    just the input.
    """
    folder = workdir / "input"
    folder.mkdir()
    shutil.copy2(pdb_path, folder / pdb_path.name)
    return folder


def _write_flowpacker_config(
    out_path: Path,
    *,
    test_path: str,
    ckpt: str,
    conf_ckpt: Optional[str],
) -> None:
    """Write a base.yaml that FlowPacker's load_config will accept.

    sampler_pdb.py / likelihood.py have argparse CLIs that don't accept
    dotted overrides -- all configuration goes through the YAML. We
    write a minimal config matching the upstream schema with our
    overrides for test_path / ckpt / conf_ckpt.
    """
    conf_line = (
        f"conf_ckpt: '{conf_ckpt}'\n" if conf_ckpt else "conf_ckpt:\n"
    )
    out_path.write_text(
        "mode: vf\n"
        "\n"
        "data:\n"
        "  data: bc40\n"
        "  train_path:\n"
        "  cluster_path:\n"
        f"  test_path: '{test_path}'\n"
        "  min_length: 40\n"
        "  max_length: 512\n"
        "  edge_type: knn\n"
        "  max_radius: 16.0\n"
        "  max_neighbors: 30\n"
        "\n"
        f"ckpt: '{ckpt}'\n"
        f"{conf_line}"
        "\n"
        "sample:\n"
        "  batch_size: 1\n"
        "  n_samples: 1\n"
        "  use_ema: True\n"
        "  eps: 2.0e-3\n"
        "  save_trajectory: False\n"
        "  coeff: 5.0\n"
        "  num_steps: 10\n"
    )


def flowpacker_pack(
    pdb_path: str | Path,
    out_dir: str | Path,
    checkpoint: str = "cluster",      # 'cluster' (recommended) or 'bc40'
    use_confidence: bool = True,      # whether to rank samples with the conf model
    use_gt_masks: bool = False,       # if False, only repack residues with missing chis
    seed: int = 42,
    timeout: float = 600.0,
) -> FlowPackerPackResult:
    """Repack side chains with FlowPacker.

    Args:
        pdb_path: input PDB.
        out_dir: directory for the repacked PDB.
        checkpoint: which trained checkpoint to use. ``cluster`` is
            FlowPacker's default for general use (PDB-S40 training
            set); ``bc40`` is the BC40 benchmark checkpoint.
        use_confidence: whether to also load ``confidence.pth`` and
            pick the best sample under the confidence reranker.
        use_gt_masks: if True, repack ALL chi angles regardless of
            whether they're present in the input PDB.
        seed: rng seed.
    """
    import time

    from protein_chisel.paths import (
        FLOWPACKER_GUEST_SOURCE_DIR,
        FLOWPACKER_WEIGHTS_DIR,
    )
    from protein_chisel.utils.apptainer import esmc_call

    pdb_path = Path(pdb_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    workdir = Path(tempfile.mkdtemp(prefix="chisel_flowpacker_pack_"))
    try:
        input_folder = _stage_pdb_for_flowpacker(pdb_path, workdir)
        run_name = f"chisel_run_{int(time.time())}"

        # sampler_pdb.py uses argparse with positional `config name` and
        # flags `--save_traj/--seed/--use_gt_masks/--inpaint`. It does
        # NOT accept dotted overrides. The data.test_path / ckpt /
        # conf_ckpt all come from the YAML config file; we bind a
        # patched copy of base.yaml over /opt/flowpacker/config/inference/
        # base.yaml at runtime.
        patched_yaml = workdir / "base.yaml"
        ckpt_in_sif = (
            f"/opt/flowpacker/checkpoints/{checkpoint}.pth"
        )
        conf_ckpt_in_sif = (
            "/opt/flowpacker/checkpoints/confidence.pth" if use_confidence else None
        )
        _write_flowpacker_config(
            patched_yaml,
            test_path=str(input_folder),
            ckpt=ckpt_in_sif,
            conf_ckpt=conf_ckpt_in_sif,
        )

        sif_workdir = "/tmp/flowpacker_run"   # writable scratch inside the sif
        call = (
            esmc_call(nv=True)
            .with_bind(str(workdir))
            .with_bind(str(FLOWPACKER_WEIGHTS_DIR), str(FLOWPACKER_GUEST_SOURCE_DIR / "checkpoints"))
            # Override base.yaml with our patched version (file-level bind).
            .with_bind(
                str(patched_yaml),
                str(FLOWPACKER_GUEST_SOURCE_DIR / "config" / "inference" / "base.yaml"),
            )
        )

        # Drive sampler_pdb.py from a scratch dir (it writes ./samples/
        # and ./logs/ relative to cwd). Symlink /opt/flowpacker children
        # into the scratch so the relative `./config/...` paths resolve.
        # Replace the entry script symlinks with patched copies that pass
        # weights_only=False to torch.load (PyTorch 2.6 changed the
        # default to True; FlowPacker checkpoints have non-tensor metadata
        # and would otherwise fail to load).
        gt_flag = "--use_gt_masks True" if use_gt_masks else ""
        bash = (
            f"set -e; "
            f"mkdir -p {sif_workdir} && cd {sif_workdir} && "
            f"ln -sf {FLOWPACKER_GUEST_SOURCE_DIR}/* . && "
            f"rm -f sampler_pdb.py likelihood.py && "
            f"cp {FLOWPACKER_GUEST_SOURCE_DIR}/sampler_pdb.py . && "
            f"cp {FLOWPACKER_GUEST_SOURCE_DIR}/likelihood.py . && "
            f"sed -i 's|torch.load(self.config.ckpt)|torch.load(self.config.ckpt, weights_only=False)|g; "
            f"s|torch.load(self.config.conf_ckpt)|torch.load(self.config.conf_ckpt, weights_only=False)|g' "
            f"sampler_pdb.py likelihood.py && "
            f"python sampler_pdb.py base {run_name} --seed {seed} {gt_flag}; "
            f"cp -r {sif_workdir}/samples/{run_name} {workdir}/output; "
        )
        argv = ["bash", "-c", bash]

        LOGGER.info("running FlowPacker sampler_pdb: %s", run_name)
        t0 = time.perf_counter()
        result = call.run(argv, timeout=timeout, check=True)
        runtime = time.perf_counter() - t0

        # Find the repacked PDB. FlowPacker writes
        # samples/<run_name>/run_<i>/<input_stem>.pdb (one per ODE sample).
        # If confidence is on, samples/<run_name>/best_run/ symlinks to
        # the highest-scoring run.
        out_root = workdir / "output"
        if not out_root.exists():
            raise RuntimeError(
                f"FlowPacker produced no output dir at {out_root}; "
                f"stdout tail:\n{result.stdout[-2000:]}"
            )
        # Pick best_run if present, else the first run_*
        best = out_root / "best_run"
        if not best.is_dir():
            runs = sorted(out_root.glob("run_*"))
            if not runs:
                raise RuntimeError(f"No runs under {out_root}")
            best = runs[0]
        produced_pdbs = list(best.glob("*.pdb"))
        if not produced_pdbs:
            raise RuntimeError(f"No PDB under {best}")

        final = out_dir / pdb_path.with_suffix(".flowpacker.pdb").name
        shutil.copy2(produced_pdbs[0], final)
        return FlowPackerPackResult(
            out_pdb_path=final,
            runtime_seconds=runtime,
            checkpoint=checkpoint,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def flowpacker_score(
    pdb_path: str | Path,
    checkpoint: str = "cluster",
    seed: int = 42,
    timeout: float = 600.0,
) -> FlowPackerScoreResult:
    """Score per-residue chi log-likelihood under FlowPacker's flow.

    Runs FlowPacker's ``likelihood.py`` which integrates the reverse-time
    ODE with a Hutchinson trace estimator. Output is a per-PDB pickle
    with ``logp_raw`` (per-residue, per-chi tensor [N,4], with masking)
    plus aggregate ``logp_sum`` / ``logp_mean``.

    Higher log-likelihood = more plausible rotamer under FlowPacker.
    """
    import pickle
    import time

    import numpy as np
    import pandas as pd

    from protein_chisel.paths import (
        FLOWPACKER_GUEST_SOURCE_DIR,
        FLOWPACKER_WEIGHTS_DIR,
    )
    from protein_chisel.utils.apptainer import esmc_call

    pdb_path = Path(pdb_path).resolve()
    workdir = Path(tempfile.mkdtemp(prefix="chisel_flowpacker_score_"))
    try:
        input_folder = _stage_pdb_for_flowpacker(pdb_path, workdir)
        run_name = f"chisel_score_{int(time.time())}"

        # Same as flowpacker_pack: bind a patched base.yaml over the
        # in-sif config so test_path + ckpt point at our paths.
        patched_yaml = workdir / "base.yaml"
        _write_flowpacker_config(
            patched_yaml,
            test_path=str(input_folder),
            ckpt=f"/opt/flowpacker/checkpoints/{checkpoint}.pth",
            conf_ckpt=None,  # likelihood doesn't need confidence model
        )

        sif_workdir = "/tmp/flowpacker_run"

        call = (
            esmc_call(nv=True)
            .with_bind(str(workdir))
            .with_bind(str(FLOWPACKER_WEIGHTS_DIR), str(FLOWPACKER_GUEST_SOURCE_DIR / "checkpoints"))
            .with_bind(
                str(patched_yaml),
                str(FLOWPACKER_GUEST_SOURCE_DIR / "config" / "inference" / "base.yaml"),
            )
        )
        # After running likelihood.py we convert the torch-saved .pth to
        # a stdlib-pickle-able dict of numpy arrays + scalars, so the
        # host wrapper (which has no torch) can read it via pickle.load.
        # The conversion script writes <workdir>/score.pkl which the
        # wrapper then reads.
        convert_py = workdir / "convert_score.py"
        convert_py.write_text(
            f"""\
import pickle
import sys
import numpy as np
import torch

src = '{sif_workdir}/likelihood/{run_name}.pth'
dst = '{workdir}/score.pkl'

def to_pure(v):
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    if isinstance(v, dict):
        return {{k: to_pure(vv) for k, vv in v.items()}}
    if isinstance(v, (list, tuple)):
        return type(v)(to_pure(vv) for vv in v)
    return v

d = torch.load(src, map_location='cpu', weights_only=False)
with open(dst, 'wb') as f:
    pickle.dump(to_pure(d), f)
print('converted', src, '->', dst)
""".strip() + "\n"
        )

        bash = (
            f"set -e; "
            f"mkdir -p {sif_workdir} && cd {sif_workdir} && "
            f"ln -sf {FLOWPACKER_GUEST_SOURCE_DIR}/* . && "
            f"rm -f sampler_pdb.py likelihood.py && "
            f"cp {FLOWPACKER_GUEST_SOURCE_DIR}/likelihood.py . && "
            f"sed -i 's|torch.load(self.config.ckpt)|torch.load(self.config.ckpt, weights_only=False)|g; "
            f"s|torch.load(self.config.conf_ckpt)|torch.load(self.config.conf_ckpt, weights_only=False)|g' "
            f"likelihood.py && "
            f"python likelihood.py base {run_name} --seed {seed} && "
            f"python {convert_py} "
        )
        argv = ["bash", "-c", bash]

        LOGGER.info("running FlowPacker likelihood: %s", run_name)
        t0 = time.perf_counter()
        call.run(argv, timeout=timeout, check=True)
        runtime = time.perf_counter() - t0

        # The conversion script inside the sif wrote a stdlib-pickle file
        # with all tensors converted to numpy arrays (host has no torch).
        score_path = workdir / "score.pkl"
        if not score_path.is_file():
            raise RuntimeError(
                f"FlowPacker likelihood produced no output at {score_path}"
            )
        with open(score_path, "rb") as fh:
            payload = pickle.load(fh)

        # likelihood.py output shape: dict[pdb_name -> dict] with at least
        # 'logp_raw' (Tensor [N, 4]), 'logp_sum' (scalar), 'logp_mean'.
        if isinstance(payload, dict):
            entries = list(payload.values())
            d = entries[0] if entries else {}
        else:
            d = payload

        logp_raw = np.asarray(d.get("logp_raw", []), dtype=float)
        logp_sum = float(d.get("logp_sum", 0.0) or 0.0)
        logp_mean = float(d.get("logp_mean", 0.0) or 0.0)

        rows = []
        if logp_raw.ndim == 2 and logp_raw.shape[1] == 4:
            for i in range(logp_raw.shape[0]):
                row = {"resseq": i + 1}
                for k in range(4):
                    v = float(logp_raw[i, k])
                    row[f"logp_chi{k+1}"] = (
                        v if not np.isnan(v) else None
                    )
                rows.append(row)
        df = pd.DataFrame(rows)

        # Aggregate per-chi-index means (NaN-safe).
        def _safe_mean(arr):
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0:
                return 0.0
            return float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0

        per_chi_means = (
            [_safe_mean(logp_raw[:, k]) for k in range(4)]
            if (logp_raw.ndim == 2 and logp_raw.shape[1] >= 4)
            else [0.0, 0.0, 0.0, 0.0]
        )
        n_chis_scored = int(np.sum(~np.isnan(logp_raw))) if logp_raw.size else 0

        return FlowPackerScoreResult(
            per_residue_df=df,
            n_residues_scored=int(logp_raw.shape[0] if logp_raw.ndim == 2 else 0),
            n_chis_scored=n_chis_scored,
            logp_sum=logp_sum,
            logp_mean=logp_mean,
            logp_mean_chi1=per_chi_means[0],
            logp_mean_chi2=per_chi_means[1],
            logp_mean_chi3=per_chi_means[2],
            logp_mean_chi4=per_chi_means[3],
            runtime_seconds=runtime,
            checkpoint=checkpoint,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


__all__ = [
    "FlowPackerPackResult",
    "FlowPackerScoreResult",
    "flowpacker_pack",
    "flowpacker_score",
]
